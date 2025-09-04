# BoatForumGPT — Reddit RAG (Parquet + FAISS)

Minimal RAG app answering boating questions from a small Reddit corpus.

## Features
- Embeddings in Parquet (no Chroma)
- FAISS CPU retrieval (fallback to NumPy)
- Streamlit UI
- Outputs: concise one-liner + long blog draft
- Clickable links to all retrieved threads

## Deploy Local
 - Local: `Docker compose up`
 - Live(Prod) Push to Master branch
