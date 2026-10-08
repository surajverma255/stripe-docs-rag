"""
Stage 4a: Embedding + indexing
------------------------------
Turns every chunk into an embedding and stores it in a ChromaDB vector
database at data/chroma/. This is the last "offline" step of the pipeline:
you run it once (and again whenever the chunks change).

What's an embedding? A list of numbers (384 of them here) that captures what
a piece of text means. Texts about similar things get similar numbers, so
"How do I give a customer their money back?" ends up close to a chunk about
refunds, even though they share almost no words.

What's a vector database? A database built to answer one question fast:
"which stored vectors are closest to this one?"

Run it with:  python embed.py
(The first run downloads the model, about 67 MB.)
"""

import json
import time

import chromadb
from fastembed import TextEmbedding
from tqdm import tqdm

from config import CHROMA_DIR, CHUNKS_FILE, COLLECTION_NAME, EMBEDDING_MODEL

# Chroma accepts data in batches; very large single inserts can fail.
BATCH_SIZE = 256

# How many chunks the model processes at once.
EMBED_BATCH_SIZE = 32


def load_chunks():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = TextEmbedding(EMBEDDING_MODEL)

    print("Embedding chunks (a few minutes on a laptop CPU)...")
    start = time.time()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []
    # Embed in small batches so we can show a progress bar. Smaller batches
    # are also faster on a CPU: every text in a batch gets padded to the
    # length of the longest one, so big batches waste work on padding.
    with tqdm(total=len(texts), unit="chunk") as progress:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            # model.embed() returns one vector per text. Each vector is a NumPy
            # array; .tolist() turns it into a plain list, which Chroma accepts.
            for vector in model.embed(batch, batch_size=EMBED_BATCH_SIZE):
                embeddings.append(vector.tolist())
            progress.update(len(batch))
    print(f"  Done in {time.time() - start:.0f}s. Each vector has {len(embeddings[0])} numbers.")

    # A PersistentClient saves the database to disk, so it survives restarts.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start fresh every run, so re-running never leaves stale chunks behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Nothing to delete on the very first run.

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # "Cosine" distance compares the direction of vectors, not their length.
        # It's the standard choice for text embeddings.
        configuration={"hnsw": {"space": "cosine"}},
        # We compute embeddings ourselves, so tell Chroma not to use its own model.
        embedding_function=None,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=[c["text"] for c in batch],
            # Metadata travels with each chunk so we can cite sources later.
            metadatas=[
                {
                    "title": c["title"],
                    "breadcrumb": c["breadcrumb"],
                    "section": c["section"],
                    "url": c["url"],
                }
                for c in batch
            ],
        )

    print(f"Stored {collection.count()} chunks in {CHROMA_DIR}/ (collection '{COLLECTION_NAME}')")


if __name__ == "__main__":
    main()
