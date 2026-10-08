"""
Stage 3: Chunking
-----------------
Splits the downloaded Stripe docs into small, self-contained pieces ("chunks")
and saves them to data/chunks.jsonl, ready to be embedded in Stage 4.

Why chunk at all?
- Embedding models have an input limit. Ours (Stage 4) reads at most 512
  tokens, roughly 1,500-2,000 characters, and silently ignores the rest.
- Retrieval is more precise with small pieces. If a question is about refund
  fees, we want to hand the LLM the paragraph about refund fees, not a 25 KB
  page about everything to do with refunds.

Our strategy ("structure-aware chunking"):
1. Clean each page: remove boilerplate and turn links into plain text.
2. Split the page at its headings (##, ###, ####), because writers already
   grouped related information under headings for us.
3. If a section is still too long, split it at paragraph breaks, never in the
   middle of a code block unless the code block alone is too big.
4. Start every chunk with a "breadcrumb" like
       Refund and cancel payments > Issue refunds > Dashboard
   so each chunk makes sense on its own, even when the heading that
   explained it ended up in a different chunk.

Run it with:  python chunk.py
"""

import json
import re
import statistics
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/chunks.jsonl")

# The biggest a chunk's body is allowed to be, in characters. About 4
# characters make one token in English prose (code is denser), so 1,200
# characters stays comfortably inside the embedding model's 512-token limit.
MAX_CHARS = 1200

# Sections shorter than this are merged into the next section, so we don't
# create useless chunks that contain only a heading and one line.
MIN_CHARS = 150

# Lines that Stripe adds to some pages but that say nothing about the topic.
# Leaving them in would make every such chunk look alike to the search.
BOILERPLATE_PREFIXES = (
    "Read this page in your terminal",
    "## Start here: Integrate with Stripe using skills and plugins",
    "Stripe provides skills and plugins for agents",
    "**Quickstart**: Install the Stripe CLI",
)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
# Matches [text](url) and ![alt](url); we keep only the text part.
LINK_PATTERN = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def is_fence(line):
    """True for lines that open or close a code block (```)."""
    return line.lstrip().startswith("```")


def clean_markdown(text):
    """Remove boilerplate and replace links with their text.

    Code blocks are left untouched: inside code, every character matters.
    """
    cleaned = []
    in_code = False
    for line in text.splitlines():
        if is_fence(line):
            in_code = not in_code
        elif not in_code:
            if line.strip().startswith(BOILERPLATE_PREFIXES):
                continue
            # URLs are long, add nothing to the meaning, and waste our limited
            # chunk space, so "[Refunds API](https://...)" becomes "Refunds API".
            line = LINK_PATTERN.sub(r"\1", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def split_into_sections(text):
    """Split a page at its headings.

    Returns a list of (heading_path, body) pairs, where heading_path is a list
    such as ["Issue refunds", "Dashboard"].

    The tricky part: a line starting with "#" inside a code block is a code
    comment (like "# Install the CLI"), not a heading. We track whether we're
    inside a code block and only treat "#" lines outside code as headings.
    """
    sections = []
    stack = []          # Current headings, e.g. [(2, "Issue refunds"), (4, "Dashboard")]
    body_lines = []
    in_code = False

    def save_section():
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(([title for _, title in stack], body))

    for line in text.splitlines():
        if is_fence(line):
            in_code = not in_code
            body_lines.append(line)
            continue

        match = HEADING_PATTERN.match(line) if not in_code else None
        if not match:
            body_lines.append(line)
            continue

        # Found a real heading: finish the previous section first.
        save_section()
        body_lines = []

        level, title = len(match.group(1)), match.group(2).strip()
        # A new "##" heading closes any open "##", "###" or "####" headings.
        while stack and stack[-1][0] >= level:
            stack.pop()
        # We skip level-1 headings ("# Title"): the page title is added to
        # every breadcrumb from metadata.json anyway.
        if level > 1:
            stack.append((level, title))

    save_section()
    return sections


def split_into_blocks(body):
    """Split a section into paragraphs, keeping each code block in one piece."""
    blocks, current, in_code = [], [], False
    for line in body.splitlines():
        if is_fence(line):
            in_code = not in_code
        if not in_code and not is_fence(line) and line.strip() == "":
            # A blank line outside code ends a paragraph.
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_long_block(block):
    """Last resort for a single block bigger than MAX_CHARS (huge code or table).

    Split it line by line. We'd rather break a giant code sample than lose it.
    If the block is code, each piece is wrapped in its own ``` markers so it
    stays a valid code block, and the LLM can still tell it's code.
    """
    lines = block.splitlines()
    opening, closing = "", ""
    if lines and is_fence(lines[0]):
        opening = lines[0]                        # e.g. ```python
        lines = lines[1:]
        if lines and is_fence(lines[-1]):
            lines = lines[:-1]
        closing = "```"
    room = MAX_CHARS - len(opening) - len(closing) - 2

    pieces, current = [], ""
    for line in lines:
        if current and len(current) + len(line) + 1 > room:
            pieces.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)

    if opening:
        pieces = [f"{opening}\n{piece}\n{closing}" for piece in pieces]
    return pieces


def pack_blocks(blocks):
    """Greedily combine blocks into chunks of at most MAX_CHARS."""
    chunks, current = [], ""
    for block in blocks:
        parts = split_long_block(block) if len(block) > MAX_CHARS else [block]
        for part in parts:
            if current and len(current) + len(part) + 2 > MAX_CHARS:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        chunks.append(current)
    return chunks


def chunk_page(text, page):
    """Turn one page into a list of chunk dictionaries."""
    chunks = []
    carry = ""  # Text from tiny sections, waiting to join the next section.

    sections = split_into_sections(clean_markdown(text))
    for index, (heading_path, body) in enumerate(sections):
        is_last = index == len(sections) - 1
        if carry:
            body = f"{carry}\n\n{body}"
            carry = ""

        # Too small to stand alone? Keep its heading as text and merge forward.
        if len(body) < MIN_CHARS and not is_last:
            heading = heading_path[-1] if heading_path else ""
            carry = f"{heading}\n{body}" if heading else body
            continue

        breadcrumb = " > ".join([page["title"], *heading_path])
        for piece in pack_blocks(split_into_blocks(body)):
            chunks.append(
                {
                    "text": f"{breadcrumb}\n\n{piece}",
                    "title": page["title"],
                    "breadcrumb": breadcrumb,
                    "section": page["section"],
                    "url": page["url"],
                    "source_file": page["file"],
                }
            )
    return chunks


def main():
    metadata = json.loads((RAW_DIR / "metadata.json").read_text(encoding="utf-8"))

    all_chunks = []
    for page in metadata:
        text = (RAW_DIR / page["file"]).read_text(encoding="utf-8")
        all_chunks.extend(chunk_page(text, page))

    # Give every chunk a unique, readable id like "refunds.md#3".
    counters = {}
    for chunk in all_chunks:
        n = counters.get(chunk["source_file"], 0)
        counters[chunk["source_file"]] = n + 1
        chunk["id"] = f"{chunk['source_file']}#{n}"

    # JSON Lines: one JSON object per line. Easy to read, append and stream.
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in all_chunks]
    print(f"Pages: {len(metadata)}")
    print(f"Chunks: {len(all_chunks)}")
    print(
        f"Chunk size (chars): min {min(sizes)}, "
        f"median {int(statistics.median(sizes))}, max {max(sizes)}"
    )
    print(f"Saved to {OUTPUT_FILE}\n")

    # Always look at your data! Print one example chunk.
    example = next(c for c in all_chunks if c["source_file"] == "refunds.md")
    print("Example chunk:\n" + "-" * 60)
    print(example["text"])
    print("-" * 60)


if __name__ == "__main__":
    main()
