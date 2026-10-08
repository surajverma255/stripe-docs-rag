"""
Shared settings for the whole project.

Both embed.py (which builds the index) and retrieve.py (which searches it)
import from here. If they used different models, the numbers they produce
would be in different "languages" and search would return nonsense, so
keeping this in one place prevents a whole class of bugs.
"""

from pathlib import Path

DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "stripe_docs"

# A small, fast, free embedding model. It turns text into a list of 384
# numbers, reads up to 512 tokens, and is only about 67 MB.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models were trained to expect this instruction in front of search
# queries (but not in front of the documents). Questions and documents are
# phrased differently ("how do I refund?" vs. "To refund a payment, ..."),
# and the prefix tells the model which side of that relationship a text is on.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How many chunks to retrieve for each question.
TOP_K = 5
