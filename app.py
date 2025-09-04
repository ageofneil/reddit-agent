# app.py — Streamlit RAG over Parquet embeddings (no Chroma)

import os, json, datetime, re
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Optional FAISS (faster). Falls back to NumPy cosine if unavailable.
try:
    import faiss  # pip install faiss-cpu
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from dotenv import load_dotenv
from openai import OpenAI

# ========= Config =========
EMB_DIR = "./emb_out"
PARQUET_PATH = os.path.join(EMB_DIR, "embeddings.parquet")
META_JSON    = os.path.join(EMB_DIR, "meta.json")
FAISS_INDEX_PATH = os.path.join(EMB_DIR, "faiss.index")
FAISS_IDS_PATH   = os.path.join(EMB_DIR, "ids.npy")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4.1-mini"

TOP_K_CHUNKS_DEFAULT   = 60   # retrieve many, then re-rank & group
TOP_K_THREADS_DEFAULT  = 8    # final number of threads to show
PER_THREAD_DEFAULT     = 4    # chunks per thread in the context
TEMPERATURE_DEFAULT    = 0.2

# ========= Auth =========
load_dotenv()
API_KEY = os.environ.get("API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY")
# Allow Streamlit secrets as well
if not API_KEY:
    API_KEY = st.secrets.get("API_KEY_OPENAI", None) if hasattr(st, "secrets") else None

# ========= Utilities =========
def iso(ts: Optional[int]):
    if ts is None: return None
    return datetime.datetime.utcfromtimestamp(int(ts)).replace(tzinfo=datetime.timezone.utc).isoformat()

def _normalize_rows(df: pd.DataFrame):
    # ensure required columns exist
    req = ["id","text","created_at","score","subreddit","thread_url","thread_title","embedding"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    # convert embedding column (list -> np.array rows)
    embs = np.vstack(df["embedding"].to_numpy())
    return embs.astype(np.float32)

def _cosine(a: np.ndarray, b: np.ndarray):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: Optional[str]):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def _embed_query(client: OpenAI, q: str) -> np.ndarray:
    v = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    v = np.array(v, dtype=np.float32)
    # normalize for cosine/IP search
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

# ========= Loader =========
class ParquetSearcher:
    def __init__(self, parquet_path=PARQUET_PATH, meta_json=META_JSON):
        self.df = pd.read_parquet(parquet_path)
        self.embs = _normalize_rows(self.df)  # (N, D)
        self.ids = self.df["id"].to_numpy()
        self.dim = self.embs.shape[1]

        # try to load FAISS; if not available or files missing, build in-memory
        self.faiss = None
        if HAS_FAISS and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_IDS_PATH):
            try:
                self.faiss = faiss.read_index(FAISS_INDEX_PATH)
                faiss_ids = np.load(FAISS_IDS_PATH, allow_pickle=True)
                # Sanity: if ids mismatch, rebuild ephemeral index instead of crashing
                if len(faiss_ids) != len(self.ids) or not np.all(faiss_ids == self.ids):
                    st.warning("FAISS ids mismatch; using ephemeral FAISS index.")
                    self._build_ephemeral_faiss()
            except Exception:
                self._build_ephemeral_faiss()
        elif HAS_FAISS:
            self._build_ephemeral_faiss()

        # normalize embeddings if using inner product (FAISS IP)
        if self.faiss is not None:
            norms = np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-12
            self.embs = self.embs / norms

        # load meta (optional)
        self.meta = {}
        if os.path.exists(meta_json):
            with open(meta_json) as f:
                self.meta = json.load(f)

    def _build_ephemeral_faiss(self):
        try:
            idx = faiss.IndexFlatIP(self.dim)
            embs_norm = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-12)
            idx.add(embs_norm.astype(np.float32))
            self.faiss = idx
        except Exception as e:
            self.faiss = None
            st.warning(f"Could not build FAISS index; will use NumPy cosine. ({e})")

    def search(self, client: OpenAI, query: str, top_k=TOP_K_CHUNKS_DEFAULT):
        q = _embed_query(client, query)  # (1,D), already normalized
        if self.faiss is not None:
            sims, idxs = self.faiss.search(q.astype(np.float32), top_k)
            sims, idxs = sims[0], idxs[0]
        else:
            # fallback: NumPy cosine
            sims = (_cosine(q, self.embs)[0]).astype(np.float32)
            idxs = np.argpartition(-sims, range(min(top_k, len(sims))))[:top_k]
            idxs = idxs[np.argsort(-sims[idxs])]
            sims = sims[idxs]

        # Pack rows with metadata
        rows = []
        for sim, i in zip(sims, idxs):
            r = self.df.iloc[int(i)]
            rows.append({
                "idx": int(i),
                "sim": float(sim),
                "id": r["id"],
                "text": r["text"],
                "created_at": r["created_at"],
                "score": float(r["score"]) if r["score"] is not None else 0.0,
                "subreddit": r["subreddit"],
                "thread_url": r["thread_url"],
                "thread_title": r["thread_title"],
            })
        return rows

# ========= RAG glue (ranking, context, answer) =========
def semantic_search(ps: ParquetSearcher, client: OpenAI, query: str, k_threads: int, per_thread: int, top_k_chunks: int):
    # 1) get many chunks
    raw = ps.search(client, query, top_k=top_k_chunks)

    # 2) light popularity bump (same spirit as before)
    items = []
    for r in raw:
        sim = r["sim"]
        up  = r["score"]
        weight = sim * (1.0 + 0.05 * (max(up, 0.0) ** 0.3))  # gentle bump
        items.append((weight, r))
    items.sort(key=lambda x: x[0], reverse=True)

    # 3) group by thread_url; keep up to per_thread chunks per thread
    grouped: Dict[str, List[Dict]] = {}
    for w, r in items:
        url = r["thread_url"]
        lst = grouped.setdefault(url, [])
        if len(lst) < per_thread:
            lst.append(r)

    # 4) rank threads by best chunk similarity
    threads = sorted(grouped.values(), key=lambda lst: max((x["sim"] for x in lst), default=0), reverse=True)
    top_threads = threads[:k_threads]

    # 5) freeze hits structure
    hits = []
    for chunks in top_threads:
        m0 = chunks[0]
        hits.append({
            "thread_url": m0["thread_url"],
            "thread_title": m0["thread_title"],
            "subreddit": m0["subreddit"],
            "created_at": m0["created_at"],
            "chunks": [c["text"] for c in chunks],
        })
    return hits

def build_context(hits: List[Dict], token_budget: int = 8000) -> str:
    try:
        import tiktoken
        ENC = tiktoken.get_encoding("cl100k_base")
        def tok(s): return len(ENC.encode(s))
    except Exception:
        def tok(s): return max(1, len(s)//4)

    header = "Context from Reddit threads (use ONLY this info; cite with [S#]):\n"
    parts, used = [header], tok(header)
    for i, h in enumerate(hits, start=1):
        block = [f"[S{i}] {h['thread_title']} — r/{h['subreddit']} — {h.get('created_at')}\nURL: {h['thread_url']}\n"]
        for ch in h["chunks"]:
            block.append((ch or "").strip() + "\n")
        text = "\n".join(block).strip() + "\n\n"
        t = tok(text)
        if used + t > token_budget:
            break
        parts.append(text); used += t
    return "".join(parts)

# ---- Answer shaping helpers ----
SECTION_CUTOFF_RE = re.compile(r"^\s*(sources|other discussions)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)

def clean_answer(text: str) -> str:
    """Remove any trailing 'Sources'/'Other discussions' sections if the model added them."""
    m = SECTION_CUTOFF_RE.search(text or "")
    return text[:m.start()].rstrip() if m else text

def parse_two_parts(raw: str):
    """Split model output into ('concise', 'blog') using labeled headers."""
    raw = raw or ""
    txt = raw.replace("\r", "").strip()
    concise_label = "Concise answer:"
    blog_label = "Blog draft:"
    if blog_label in txt:
        parts = txt.split(blog_label, 1)
        concise = parts[0].replace(concise_label, "").strip()
        blog = parts[1].strip()
    else:
        concise, blog = txt.strip(), ""
    return concise, blog

def refs_from_hits(hits: List[Dict]) -> List[str]:
    """Return canonical reference lines based on ALL retrieved threads."""
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[S{i}] {h['thread_title']} ({h['thread_url']})")
    return lines

def answer_query(ps: ParquetSearcher, client: OpenAI, query: str,
                 k_threads=TOP_K_THREADS_DEFAULT, per_thread=PER_THREAD_DEFAULT,
                 model=CHAT_MODEL, temperature=TEMPERATURE_DEFAULT, top_k_chunks=TOP_K_CHUNKS_DEFAULT):
    hits = semantic_search(ps, client, query, k_threads=k_threads, per_thread=per_thread, top_k_chunks=top_k_chunks)
    context = build_context(hits, token_budget=8000)

    system = (
        "You are BoatForumGPT. Use ONLY the provided context.\n\n"
        "Return TWO clearly separated parts with these exact headings:\n"
        "1) Concise answer:\n"
        "   - A single-sentence framing summary with inline citations like [S1](URL).\n"
        "   - Be extremely concise and factual.\n"
        "   - Example: \"For a 30' sailboat in a storm, sailors recommend early reefing, heaving-to, and using a trysail [S1](URL), [S2](URL).\"\n\n"
        "2) Blog draft:\n"
        "   - A comprehensive, publication-ready article (~900–1200 words) grounded strictly in the context.\n"
        "   - Use clear structure with short sections and helpful formatting (e.g., Materials & Tools, numbered steps, checklists, pitfalls, pro tips, brief FAQ if relevant).\n"
        "   - Include a short intro and a concise conclusion.\n"
        "   - You may include inline [S#](URL) mentions within the prose, and you MUST end with a 'References' list enumerating all cited [S#](URL).\n"
        "Do not add any other sections."
    )
    user = f"Question: {query}\n\n{context}"

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content
    raw = clean_answer(raw)

    concise, blog = parse_two_parts(raw)

    return concise, blog, hits

# ---------- Markdown helpers ----------
SRC_LINE_RE = re.compile(r"^\[S(\d+)\]\s+(.*?)\s+\((https?://[^\s)]+)\)$")

def to_markdown_list(items: List[str], start_index: int = 1) -> str:
    md = []
    for idx, s in enumerate(items, start=start_index):
        m = SRC_LINE_RE.match(s.strip())
        if m:
            sidx, title, url = m.groups()
            md.append(f"- [S{idx}] [{title}]({url})")
        else:
            # fallback: just autolink the raw line
            md.append(f"- {s}")
    return "\n".join(md)

# ========= Streamlit UI =========
st.set_page_config(page_title="BoatForumGPT (Parquet RAG)", page_icon="⛵", layout="wide")
st.title("⛵ BoatForumGPT — Reddit RAG (Parquet + FAISS)")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=API_KEY or "", help="If blank, I'll use environment/secrets.")
    if api_key_input and api_key_input != API_KEY:
        os.environ["API_KEY_OPENAI"] = api_key_input
        API_KEY = api_key_input

    k_threads = st.slider("Threads to include", 1, 12, TOP_K_THREADS_DEFAULT)
    per_thread = st.slider("Chunks per thread", 1, 6, PER_THREAD_DEFAULT)
    top_k_chunks = st.slider("Retriever pool size", 20, 200, TOP_K_CHUNKS_DEFAULT, step=10)
    temperature = st.slider("Answer temperature", 0.0, 1.0, TEMPERATURE_DEFAULT, 0.05)

    st.caption("Embeddings path:")
    st.code(PARQUET_PATH, language="text")

# Load resources
client = get_openai_client(API_KEY)
if client is None:
    st.error("Missing OpenAI API key. Add it in the sidebar or set API_KEY_OPENAI / OPENAI_API_KEY.")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_searcher():
    return ParquetSearcher(PARQUET_PATH, META_JSON)

ps = load_searcher()

# Query input
q = st.text_input("Ask a question about sailing/boat topics:", placeholder="e.g., Best storm tactics for a 30' sailboat?")
ask = st.button("Ask")

if ask and q.strip():
    with st.spinner("Thinking..."):
        concise, blog, hits = answer_query(
            ps, client, q.strip(),
            k_threads=k_threads, per_thread=per_thread,
            model=CHAT_MODEL, temperature=temperature, top_k_chunks=top_k_chunks
        )

    st.markdown("### Concise answer")
    st.markdown(concise)

    # Community discussions first
    all_hit_refs = refs_from_hits(hits)
    if all_hit_refs:
        st.markdown("### Here are some community discussions")
        st.markdown(to_markdown_list(all_hit_refs, start_index=1))

    # Blog comes after
    st.markdown("### Blog draft (demo only)")
    with st.expander("Show blog-style synthesis"):
        st.write(blog if blog else "_No draft produced._")

        # Always include canonical references under the blog as well
        if all_hit_refs:
            st.markdown("#### References")
            st.markdown(to_markdown_list(all_hit_refs, start_index=1))
