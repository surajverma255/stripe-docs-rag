"""
Stage 4b: Retrieval (semantic search)
-------------------------------------
Given a question, find the chunks whose meaning is closest to it.
This is the "R" in RAG. In Stage 5 we'll import retrieve() from here and
hand its results to the LLM.

Try it:  python retrieve.py "How do I refund a payment?"
"""

import sys
from functools import lru_cache

import chromadb
from fastembed import TextEmbedding

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, QUERY_PREFIX, TOP_K


# @lru_cache means "run this once and remember the result". Loading the model
# and opening the database are slow, so we do it once, not for every question.
@lru_cache(maxsize=1)
def get_model():
    return TextEmbedding(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question, k=TOP_K):
    """Return the k chunks most similar to the question, best match first.

    Each result is a dict with the chunk text, its metadata, and a "score"
    between 0 and 1, where higher means more similar.
    """
    # 1. Embed the question with the same model used for the chunks,
    #    plus the query prefix BGE expects for searches.
    query_vector = next(iter(get_model().embed([QUERY_PREFIX + question]))).tolist()

    # 2. Ask Chroma for the nearest chunks.
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # 3. Chroma returns parallel lists (one list per query). We sent one
    #    query, so take element [0] and zip the lists into tidy dicts.
    results = []
    for chunk_id, text, meta, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append(
            {
                "id": chunk_id,
                "text": text,
                "title": meta["title"],
                "breadcrumb": meta["breadcrumb"],
                "url": meta["url"],
                # Cosine distance runs from 0 (identical meaning) upward;
                # 1 - distance turns it into a friendlier similarity score.
                "score": round(1 - distance, 3),
            }
        )
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question here"')
        return

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")
    for rank, result in enumerate(retrieve(question), start=1):
        preview = result["text"].split("\n\n", 1)[-1][:200].replace("\n", " ")
        print(f"{rank}. [score {result['score']}] {result['breadcrumb']}")
        print(f"   {result['url']}")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
