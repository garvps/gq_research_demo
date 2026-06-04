"""
ingest.py
Run this once before launching the app.
It reads all .txt files in /transcripts, chunks them, embeds them,
and stores everything in a local ChromaDB collection.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

TRANSCRIPTS_DIR = "transcripts"
CHUNK_SIZE = 200        # words per chunk
CHUNK_OVERLAP = 30      # words of overlap between chunks
COLLECTION_NAME = "interviews"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_transcripts(directory: str) -> dict[str, str]:
    """Load all .txt files from a directory. Returns {filename: content}."""
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                transcripts[filename] = f.read()
    print(f"Loaded {len(transcripts)} transcripts.")
    return transcripts


def ingest():
    # Load model (downloads ~80MB on first run, cached after)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up ChromaDB (persists locally in ./chroma_db/)
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Drop and recreate collection for a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Load and chunk transcripts
    transcripts = load_transcripts(TRANSCRIPTS_DIR)

    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for filename, text in transcripts.items():
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadata.append({"source": filename, "chunk_index": i})

    # Embed all chunks in one batch (faster than one by one)
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    all_embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Store in ChromaDB
    print("Storing in ChromaDB...")
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )

    print(f"\nDone. {len(all_chunks)} chunks stored in ChromaDB.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    ingest()