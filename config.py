"""
Shared settings for the whole project.

Both embed.py (which builds the index) and retrieve.py (which searches it)
import from here. If they used different models, the numbers they produce
would be in different "languages" and search would return nonsense, so
keeping this in one place prevents a whole class of bugs.
"""

import os
from pathlib import Path

DATA_DIR = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "stripe_docs"

# Where the deployed app downloads its prebuilt index from: a private
# Hugging Face dataset such as "yourname/stripe-docs-rag-index".
# Set by deploy.py as a Space variable; not needed when running locally.
INDEX_REPO = os.environ.get("INDEX_REPO")

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

# The LLM that writes answers, served free by Groq (https://console.groq.com).
# Which models you can use depends on your account; rag.py prints the list if
# this one isn't available. "openai/gpt-oss-20b" is a faster alternative.
LLM_MODEL = "openai/gpt-oss-120b"

# gpt-oss is a "reasoning" model: it thinks privately before answering, and
# those thinking tokens count against free-tier limits. Looking things up in
# provided documents needs little reasoning, so "low" is faster and cheaper.
# Options: "low", "medium", "high". Only sent to gpt-oss models.
LLM_REASONING_EFFORT = "low"

# 0 = always pick the most likely next word. For answering from documents we
# want consistency and faithfulness, not creativity.
LLM_TEMPERATURE = 0

# The model that grades answers in evaluate.py ("LLM-as-judge"). Using a
# different model from the one that writes answers avoids a model grading its
# own work, and on Groq each model has its own rate limit, so evaluation
# runs faster.
JUDGE_MODEL = "openai/gpt-oss-20b"

# The exact sentence the model must use when the docs don't contain the answer.
# Having one fixed sentence lets the app (and our evaluation in Stage 7)
# reliably detect "I don't know" answers.
NO_ANSWER = "I couldn't find this in the Stripe documentation I have access to."
