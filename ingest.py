"""
Stage 2: Ingestion
------------------
Downloads a focused slice of the Stripe documentation as clean Markdown files.

How it works:
1. Download Stripe's llms.txt, an index of their docs written for AI tools.
2. Pick out the doc links from the sections we care about.
3. Download each page as Markdown and save it to data/raw/.
4. Save a metadata.json file recording the title and URL of every page,
   so later our chatbot can cite its sources.

Run it with:  python ingest.py
"""

import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Settings. Constants are written in CAPITALS by convention, so anyone reading
# the code knows these values are meant to be changed here, not in the middle
# of the program.
# ---------------------------------------------------------------------------
LLMS_TXT_URL = "https://docs.stripe.com/llms.txt"

# Which sections of llms.txt to keep. Starting small (around 100 pages) keeps
# downloads quick and makes it easier to check whether answers are correct.
# You can add more sections later, for example "Invoicing", "Connect", "Tax".
SECTIONS_TO_KEEP = {"Docs", "Payments", "Checkout", "Billing"}

OUTPUT_DIR = Path("data/raw")

# Wait this many seconds between downloads, so we don't hammer Stripe's servers.
DELAY_SECONDS = 0.5

# Identify ourselves politely. Good scrapers say who they are.
HEADERS = {
    "User-Agent": "stripe-docs-rag (learning project; "
    "github.com/surajverma255/stripe-docs-rag)"
}

# A regular expression ("regex") is a pattern for finding text.
# This one matches Markdown links like:
#   - [Refund and cancel payments](https://docs.stripe.com/refunds.md): ...
# Group 1 captures the title, group 2 captures the URL ending in .md.
# Anything after a "#" (a jump-link to part of a page) is ignored.
LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((https://docs\.stripe\.com/[^)#\s]+\.md)(?:#[^)]*)?\)"
)


def parse_llms_txt(text):
    """Return a dict of {url: (title, section)} for the sections we want."""
    pages = {}
    current_section = None

    for line in text.splitlines():
        line = line.strip()

        # Lines starting with "## " are section headings, e.g. "## Billing".
        # (Sub-headings use "### ", and we keep them in their parent section.)
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue

        # Skip everything outside the sections we chose.
        if current_section not in SECTIONS_TO_KEEP:
            continue

        # Only look at bullet-list links, not links buried in paragraphs.
        if not line.startswith("- ["):
            continue

        match = LINK_PATTERN.search(line)
        if match:
            title, url = match.group(1), match.group(2)
            # Using the URL as the dict key removes duplicates automatically,
            # because the same page is sometimes listed in several sections.
            if url not in pages:
                pages[url] = (title, current_section)

    return pages


def url_to_filename(url):
    """Turn a URL into a safe file name.

    https://docs.stripe.com/payments/checkout.md -> payments__checkout.md
    """
    path = url.replace("https://docs.stripe.com/", "")
    return path.replace("/", "__")


def download(url):
    """Download a URL and return its text, or None if it failed."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Raises an error for 404, 500, etc.
        return response.text
    except requests.RequestException as error:
        print(f"  ! Failed: {url} ({error})")
        return None


# Some Stripe pages are just an index pointing to "variants" of the page,
# e.g. one version for Stripe-hosted checkout and one for embedded checkout.
# The real content lives at URLs like .../discounts.md?payment-ui=stripe-hosted
VARIANT_MARKER = "This article has multiple variants"
VARIANT_LINK_PATTERN = re.compile(r"\((https://docs\.stripe\.com/[^)\s]+\.md\?[^)\s]+)\)")


def first_variant_url(content):
    """If the page is a variant index, return the URL of its first variant.

    Otherwise return None. We take the first variant because it's Stripe's
    default (usually the Stripe-hosted or Dashboard version).
    """
    if VARIANT_MARKER not in content:
        return None
    match = VARIANT_LINK_PATTERN.search(content)
    return match.group(1) if match else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading index: {LLMS_TXT_URL}")
    index_text = download(LLMS_TXT_URL)
    if index_text is None:
        print("Could not download llms.txt. Check your internet connection.")
        return

    pages = parse_llms_txt(index_text)
    print(f"Found {len(pages)} pages in sections: {sorted(SECTIONS_TO_KEEP)}\n")

    metadata = []
    for number, (url, (title, section)) in enumerate(pages.items(), start=1):
        filename = url_to_filename(url)
        filepath = OUTPUT_DIR / filename

        # If the file already exists, reuse it. This makes the script
        # "resumable": if it crashes halfway, just run it again.
        if filepath.exists():
            print(f"[{number}/{len(pages)}] Already have: {title}")
            content = filepath.read_text(encoding="utf-8")
        else:
            print(f"[{number}/{len(pages)}] Downloading: {title}")
            content = download(url)
            if content is None:
                continue
            filepath.write_text(content, encoding="utf-8")
            time.sleep(DELAY_SECONDS)

        # If what we saved is only a variant index, fetch the real content
        # and overwrite the stub. (This also fixes stubs from earlier runs.)
        variant_url = first_variant_url(content)
        if variant_url:
            print(f"    -> page has variants, fetching: {variant_url}")
            variant_content = download(variant_url)
            time.sleep(DELAY_SECONDS)
            if variant_content:
                filepath.write_text(variant_content, encoding="utf-8")

        # The URL we show users should be the normal web page, not the .md file.
        web_url = url.removesuffix(".md")
        metadata.append(
            {"file": filename, "title": title, "section": section, "url": web_url}
        )

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nDone. Saved {len(metadata)} pages to {OUTPUT_DIR}/")


# This line means "only run main() when this file is run directly",
# not when another file imports functions from it. We'll use that later.
if __name__ == "__main__":
    main()
