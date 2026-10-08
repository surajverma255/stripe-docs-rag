"""
Stage 8a: Publish the search index
----------------------------------
Uploads the prebuilt index (data/chroma and data/raw/metadata.json) to a
PRIVATE Hugging Face dataset. The hosted app downloads it from there when it
starts, because the index isn't in the GitHub repository:

- It's generated data, and code repositories should hold code.
- It contains the full text of Stripe's docs, which we shouldn't publish as
  downloadable files. Visitors only see short passages next to answers.

Run it whenever you rebuild the index (after embed.py):
  python publish_index.py

Needs HF_TOKEN in your .env: a Hugging Face token with WRITE access.
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

from config import CHROMA_DIR, DATA_DIR

load_dotenv()

INDEX_NAME = "stripe-docs-rag-index"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Missing HF_TOKEN in your .env file. See .env.example.")
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        sys.exit("No index in data/chroma. Run ingest.py, chunk.py and embed.py first.")

    api = HfApi(token=token)
    index_id = f"{api.whoami()['name']}/{INDEX_NAME}"

    print(f"Uploading the search index to the private dataset {index_id}...")
    api.create_repo(index_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=index_id,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
        allow_patterns=["chroma/**", "raw/metadata.json"],
        # Remove files from older uploads that no longer exist locally.
        delete_patterns=["chroma/**"],
        commit_message="Update search index",
    )
    print(f"\nDone. Set INDEX_REPO={index_id} on your hosting service.")


if __name__ == "__main__":
    main()
