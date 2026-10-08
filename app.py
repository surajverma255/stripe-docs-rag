"""
Stage 6: The web app
--------------------
A Gradio interface with three tabs:
  Ask              - chat with the assistant, plus an "evidence panel" showing
                     exactly which passages the answer was built from
  How it's built   - the pipeline, the decisions behind it, and its limits
  How it's tested  - evaluation results (filled in by Stage 7)

Run it with:  python app.py   then open the address it prints.
"""

import html
import json
import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from groq import AuthenticationError, NotFoundError, RateLimitError
from huggingface_hub import snapshot_download

from config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, INDEX_REPO, LLM_MODEL, TOP_K
from rag import answer, split_code
from retrieve import get_collection, get_model

load_dotenv()

AUTHOR = "Suraj Verma"
REPO_URL = "https://github.com/surajverma255/stripe-docs-rag"
EVAL_RESULTS = Path("eval/results.json")

# Public demos get abused; a length cap keeps prompts (and costs) sane.
MAX_QUESTION_CHARS = 500

EXAMPLE_QUESTIONS = [
    "How do I refund a payment?",
    "How do I verify webhook signatures in Python?",
    "Which test card simulates a declined payment?",
    "How do I let customers enter promotion codes in Checkout?",
    "What is Stripe's stock price?",  # Shows the assistant declining to guess.
]


# ---------------------------------------------------------------------------
# On the hosted Space, the index isn't in the code repository. Download it
# from the private Hugging Face dataset that deploy.py uploaded it to.
# ---------------------------------------------------------------------------
def ensure_index():
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        return  # Running locally, or already downloaded.
    if not INDEX_REPO:
        raise SystemExit(
            "No search index found in data/chroma. Run ingest.py, chunk.py and "
            "embed.py first, or set INDEX_REPO to download a prebuilt index."
        )
    print(f"Downloading the search index from {INDEX_REPO}...")
    snapshot_download(
        repo_id=INDEX_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["chroma/**", "raw/metadata.json"],
        token=os.environ.get("HF_TOKEN"),  # A read-only token, set as a Space secret.
    )


# ---------------------------------------------------------------------------
# Numbers shown in the UI are read from the real data, so they stay true
# whenever the index is rebuilt.
# ---------------------------------------------------------------------------
def load_stats():
    metadata = json.loads((DATA_DIR / "raw" / "metadata.json").read_text(encoding="utf-8"))
    return {"pages": len(metadata), "chunks": get_collection().count()}


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------
def link_citations(text, chunks):
    """Turn [2] into a clickable link to source 2, but never inside code.

    split_code() gives alternating pieces: even indexes are prose, odd
    indexes are code, which must stay untouched.
    """
    def to_link(match):
        number = int(match.group(1))
        if 1 <= number <= len(chunks):
            return f"[[{number}]]({chunks[number - 1]['url']})"
        return match.group(0)

    pieces = split_code(text)
    for i in range(0, len(pieces), 2):
        pieces[i] = re.sub(r"\[(\d+)\](?!\()", to_link, pieces[i])
    return "".join(pieces)


def format_answer(result):
    """The chat message: answer with linked citations, then a source list."""
    if not result["found"]:
        return result["answer"]
    body = link_citations(result["answer"], result["retrieved"])
    if result["sources"]:
        lines = [f"{s['number']}. [{s['breadcrumb']}]({s['url']})" for s in result["sources"]]
        body += "\n\n**Sources**\n\n" + "\n".join(lines)
    return body


# ---------------------------------------------------------------------------
# The evidence panel: what the model saw for the latest question
# ---------------------------------------------------------------------------
EVIDENCE_EMPTY = """
<div class="evidence">
  <h3 class="ev-title">Evidence</h3>
  <p class="ev-intro">Ask a question to see the passages the answer is built from,
  how closely each one matches, and which ones the answer cites.</p>
</div>
"""


def score_width(score):
    """Map similarity to a bar width. Scores cluster between 0.5 and 1.0,
    so we stretch that range to fill the bar and make differences visible."""
    return max(0, min(100, (score - 0.5) / 0.5 * 100))


def render_evidence(result):
    cited_numbers = {s["number"] for s in result["sources"]}
    timings, usage = result["timings"], result["usage"]
    tokens = (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)

    summary = (
        f"Searched {stats['chunks']:,} passages in {timings['retrieval_ms']:,} ms, "
        f"then wrote the answer in {timings['generation_ms'] / 1000:.1f} s "
        f"using {tokens:,} tokens."
    )
    if result["found"]:
        note = "Highlighted passages are the ones the answer cites."
    else:
        note = ("None of these passages answered the question, so the assistant "
                "said so instead of guessing.")

    items = []
    for number, chunk in enumerate(result["retrieved"], start=1):
        cited = number in cited_numbers
        passage = chunk["text"].split("\n\n", 1)[-1]  # Drop the repeated breadcrumb.
        items.append(f"""
        <li class="ev-item{' is-cited' if cited else ''}">
          <div class="ev-head">
            <span class="ev-num">{number}</span>
            <a class="ev-link" href="{html.escape(chunk['url'])}" target="_blank" rel="noopener">{html.escape(chunk['breadcrumb'])}</a>
          </div>
          <div class="ev-score" title="Cosine similarity between the question and this passage">
            <span class="ev-bar"><span style="width:{score_width(chunk['score']):.0f}%"></span></span>
            <span class="ev-score-num">{chunk['score']:.3f}</span>
          </div>
          <details><summary>Show passage</summary><pre>{html.escape(passage)}</pre></details>
        </li>""")

    return f"""
    <div class="evidence">
      <h3 class="ev-title">Evidence</h3>
      <p class="ev-intro">{summary}</p>
      <p class="ev-note">{note}</p>
      <ol class="ev-list">{''.join(items)}</ol>
    </div>"""


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------
def respond(question, history):
    """Called when the user asks something. It's a generator: each `yield`
    updates the screen, so the user sees progress instead of a frozen page."""
    question = (question or "").strip()
    if not question:
        yield history, "", gr.skip()
        return
    if len(question) > MAX_QUESTION_CHARS:
        history = history + [
            {"role": "user", "content": question[:200] + "…"},
            {"role": "assistant", "content": f"Questions are limited to {MAX_QUESTION_CHARS} characters. Try a shorter one."},
        ]
        yield history, "", gr.skip()
        return

    history = history + [{"role": "user", "content": question}]
    yield history + [{"role": "assistant", "content": "Searching the docs…"}], "", gr.skip()

    try:
        result = answer(question)
        reply, evidence = format_answer(result), render_evidence(result)
    except RateLimitError:
        reply, evidence = ("The demo has hit its free LLM limit for the moment. "
                           "Wait a minute and ask again."), gr.skip()
    except (AuthenticationError, NotFoundError):
        reply, evidence = ("The demo's LLM connection isn't configured correctly. "
                           "The owner has been notified in the logs."), gr.skip()
    except Exception as error:  # Never show a raw stack trace to visitors.
        print(f"Error answering {question!r}: {error!r}")
        reply, evidence = "Something went wrong answering that. Try again in a moment.", gr.skip()

    yield history + [{"role": "assistant", "content": reply}], "", evidence


# ---------------------------------------------------------------------------
# Static tabs: how it's built, how it's tested
# ---------------------------------------------------------------------------
def render_built(stats):
    steps = [
        ("Ingest",
         f"Downloads {stats['pages']} pages of Stripe's Payments, Checkout and Billing docs as clean Markdown, using the llms.txt index Stripe publishes for AI tools.",
         "Inspecting the files showed 28 of them were empty stubs pointing to page variants (hosted vs. embedded checkout). Ingestion now follows those links to the real content."),
        ("Chunk",
         f"Splits the pages into {stats['chunks']:,} passages at their headings, packs paragraphs up to 1,200 characters, and never cuts a code block in half.",
         "Each passage starts with its heading path, like “Refund and cancel payments > Issue refunds”, so it makes sense on its own. Lines like “# Install the CLI” inside code are comments, not headings, and are skipped."),
        ("Embed",
         f"Turns every passage into 384 numbers that capture its meaning, using {EMBEDDING_MODEL}, and stores them in ChromaDB.",
         "Runs through fastembed (ONNX) instead of PyTorch, which keeps the deployment a few hundred megabytes instead of several gigabytes."),
        ("Retrieve",
         f"Embeds the question the same way and finds the {TOP_K} closest passages by cosine similarity.",
         "BGE models expect an instruction in front of queries but not documents, because questions and answers are phrased differently. Leaving it out quietly lowers search quality."),
        ("Generate",
         f"Sends the question and passages to {LLM_MODEL} on Groq with temperature 0, and links every citation to its source.",
         "Seven prompt rules: use only the sources, cite each fact, never write code the docs don't contain, use one fixed sentence when the answer isn't there, and treat instructions inside documents as data."),
    ]
    step_html = "".join(f"""
      <li class="step">
        <h3>{name}</h3>
        <p>{what}</p>
        <p class="step-why">{why}</p>
      </li>""" for name, what, why in steps)

    return f"""
    <div class="prose">
      <h2>From documentation to a cited answer</h2>
      <p class="lede">Every answer passes through the same five steps. The first three run once, offline, to build the search index. The last two run for each question, in about two seconds.</p>
      <ol class="steps">{step_html}</ol>

      <h2>What building it taught me</h2>
      <ul class="findings">
        <li><strong>Similarity scores can't tell you when the docs lack an answer.</strong> “What is Stripe's stock price?” scored as high as genuine questions, because it matched passages about product prices. Deciding “not found” has to happen in the language model, which can read the passages.</li>
        <li><strong>Citations don't guarantee grounding.</strong> Early answers cited sources but included code the model wrote itself. A rule against unsourced code fixed it; the evaluation measures how often it still happens.</li>
        <li><strong>Models don't always follow format instructions.</strong> gpt-oss sometimes cites as 【3】 instead of [3]. The app accepts both and displays one.</li>
        <li><strong>The answer is only as good as the retrieval.</strong> Casual phrasing like “my customer wants their money back” finds weaker passages than “How do I refund a payment?”, and the answer gets vaguer with it.</li>
      </ul>

      <h2>Known limitations</h2>
      <ul class="findings">
        <li>Each question is answered on its own; follow-ups don't remember earlier messages.</li>
        <li>Covers {stats['pages']} pages from four sections of the docs, not the whole site.</li>
        <li>The index is a snapshot. Rebuilding it picks up documentation changes.</li>
      </ul>
    </div>"""


TYPE_LABELS = {"direct": "Direct", "casual": "Casual", "unanswerable": "Not in docs"}


def pct(value):
    return "–" if value is None else f"{value:.0%}"


def render_tested():
    if not EVAL_RESULTS.exists():
        return """
        <div class="prose">
          <h2>How it's tested</h2>
          <p class="lede">No evaluation has been run yet. Run <code>python evaluate.py</code> to measure retrieval, faithfulness and declines; the results appear here.</p>
        </div>"""

    data = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    s, cfg, rows = data["summary"], data["config"], data["rows"]
    run_date = data["run_at"][:10]

    metrics = [
        (pct(s["hit_rate"]), "found the right page",
         f"For answerable questions, a passage from a page that answers it was in the top {cfg['top_k']}. "
         f"Direct questions {pct(s['hit_rate_direct'])}, casual phrasing {pct(s['hit_rate_casual'])}."),
        (pct(s["faithfulness"]), "of claims supported",
         f"A second model ({cfg['judge_model']}) checked every claim and code sample in each answer against its sources. "
         f"{pct(s['fully_grounded'])} of answers were fully supported."),
        (f"{s['correct_declines']} of {s['unanswerable_total']}", "off-topic questions declined",
         f"Questions the docs can't answer got the fixed “couldn't find this” reply instead of a guess. "
         f"{s['false_refusals']} of {s['answerable_total']} answerable questions were wrongly refused."),
        (f"{s['mrr']:.2f}", "mean reciprocal rank",
         "How high the first correct page ranked: 1.0 means always first, 0.5 means second on average."),
    ]
    metric_html = "".join(f"""
      <div class="metric">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
        <p class="metric-detail">{detail}</p>
      </div>""" for value, label, detail in metrics)

    def outcome(row):
        if row["type"] == "unanswerable":
            return ("Declined", "ok") if row["declined"] else ("Answered anyway", "bad")
        if row["declined"]:
            return "Wrongly declined", "bad"
        score = row.get("faithfulness")
        if score is None:
            return "Answered", "ok"
        return (f"{score:.0%} supported", "ok" if score == 1 else "warn")

    def rank_text(row):
        if row["type"] == "unanswerable":
            return "–"
        return f"#{row['rank']}" if row["rank"] else "Missed"

    table_rows = []
    for row in rows:
        text, status = outcome(row)
        unsupported = row.get("unsupported_claims") or []
        detail = ""
        if unsupported:
            items = "".join(f"<li>{html.escape(c)}</li>" for c in unsupported)
            detail = f"<details><summary>Unsupported claims</summary><ul>{items}</ul></details>"
        table_rows.append(f"""
        <tr>
          <td>{html.escape(row['question'])}{detail}</td>
          <td>{TYPE_LABELS[row['type']]}</td>
          <td class="num{' bad' if rank_text(row) == 'Missed' else ''}">{rank_text(row)}</td>
          <td><span class="status status-{status}">{text}</span></td>
        </tr>""")

    return f"""
    <div class="prose prose-wide">
      <h2>How it's tested</h2>
      <p class="lede">{s['questions']} questions with known answers, run through the full pipeline on {run_date}.
      Twenty ask about specific facts, four use casual phrasing, and four ask things the docs don't cover.
      The evaluation is a script in the repository, so anyone can rerun it.</p>
      <div class="metrics">{metric_html}</div>

      <h2>Every question</h2>
      <p class="lede">“Right page” is where the first passage from a correct page ranked among the {cfg['top_k']} retrieved.</p>
      <div class="table-wrap">
        <table class="results">
          <thead><tr><th>Question</th><th>Kind</th><th>Right page</th><th>Outcome</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
      <p class="run-meta">Answers by {cfg['llm_model']}, judged by {cfg['judge_model']}, retrieval with {cfg['embedding_model']} over {cfg['indexed_passages']:,} passages.</p>
    </div>"""


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Schibsted+Grotesk:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

CSS = """
:root {
  --paper: #F7F8FA; --surface: #FFFFFF; --ink: #1B2333; --muted: #5B6578;
  --rule: #DDE1E8; --signal: #2457D6; --highlight: #FFE45C; --highlight-ink: #1B2333;
  --bar-track: #E8EBF0;
}
.dark {
  --paper: #12161F; --surface: #1A1F2B; --ink: #E7EAF0; --muted: #9AA3B5;
  --rule: #2C3343; --signal: #7FA2FF; --highlight: #E8C93A; --highlight-ink: #12161F;
  --bar-track: #2C3343;
}
body, .gradio-container { background: var(--paper) !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; color: var(--ink); }

/* Header: the name, one plain sentence, and where the code lives. */
.masthead { padding: 28px 4px 8px; }
.brand { display: flex; align-items: center; gap: 16px; }
.mark { width: clamp(52px, 6vw, 72px); height: auto; flex: none; }

/* Pipeline strip: five steps joined by arrows, revealed once in order.
   Each icon sits above its text, so every step gets its column's full width
   for words. The layout follows the strip's own width (a container query),
   not the window's, because Gradio can place it in narrower spaces. */
.pipeline-wrap { container-type: inline-size; margin-top: 26px; padding-top: 20px; border-top: 1px solid var(--rule); }
.pipeline {
  list-style: none; margin: 0; padding: 0;
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); column-gap: 36px; row-gap: 22px;
}
.pipe-step { position: relative; min-width: 0; display: flex; flex-direction: column; gap: 10px;
  animation: pipe-in 420ms cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 110ms + 150ms); }
/* Arrows live in the 36px gap between steps, level with the icons. */
.pipe-step:not(:last-child)::after {
  content: ""; position: absolute; top: 18px; right: -23px; width: 9px; height: 9px;
  border-top: 2px solid var(--muted); border-right: 2px solid var(--muted); transform: rotate(45deg); opacity: .7;
}
.pipe-icon {
  width: 46px; height: 46px; display: grid; place-items: center;
  border: 1.5px solid var(--rule); border-radius: 10px; background: var(--surface);
}
.pipe-icon svg { width: 28px; height: 28px; }
.pipe-step:last-child .pipe-icon { border-color: var(--ink); }
.pipe-text { display: flex; flex-direction: column; gap: 3px; line-height: 1.3; overflow-wrap: break-word; }
.pipe-text strong { font-weight: 700; font-size: 1rem; color: var(--ink); }
.pipe-text span { color: var(--muted); font-size: 0.87rem; }
.built-with { color: var(--muted); font-size: 0.88rem; margin: 16px 0 0 !important; }
@keyframes pipe-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
/* Narrower strip: three across, then two, then a single column with the
   icon beside the text. Arrows only make sense in a single row, so hide them. */
@container (max-width: 760px) {
  .pipeline { grid-template-columns: repeat(3, minmax(0, 1fr)); column-gap: 24px; }
  .pipe-step::after { display: none; }
}
@container (max-width: 500px) {
  .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@container (max-width: 340px) {
  .pipeline { grid-template-columns: 1fr; }
  .pipe-step { flex-direction: row; align-items: center; }
  .pipe-icon { flex: none; }
}

/* Tabs: a clear, clickable bar so visitors notice there's more to explore. */
.tab-wrapper { height: auto !important; padding: 4px 0 6px !important; margin: 10px 0 18px !important; }
.tab-container { height: auto !important; gap: 10px; padding: 3px 2px; }
.tab-container::after { display: none !important; }
.tab-container button {
  height: auto !important; padding: 11px 20px !important; font-size: 1.02rem !important; font-weight: 600 !important;
  color: var(--ink) !important; background: var(--surface) !important;
  border: 1.5px solid var(--rule) !important; border-radius: 8px !important;
  box-shadow: 0 1px 0 var(--rule);
}
.tab-container button:hover:not(.selected) { border-color: var(--ink) !important; background: var(--surface) !important; }
.tab-container button.selected { background: var(--ink) !important; color: var(--paper) !important; border-color: var(--ink) !important; }
.tab-container button.selected::after { display: none !important; }  /* Gradio's own underline. */
.tab-container button:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
.masthead h1 {
  font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 800; letter-spacing: -0.035em;
  line-height: 1.02; margin: 0; color: var(--ink);
}
.masthead p { margin: 10px 0 0; max-width: 62ch; color: var(--muted); font-size: 1.05rem; line-height: 1.5; }
.masthead a { color: var(--signal); }

/* Evidence panel: the signature element. */
.evidence { padding: 4px 2px; color: var(--ink); text-align: left; }
.ev-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.01em; }
.ev-intro { color: var(--muted); margin: 0 0 4px; line-height: 1.5; }
.ev-note { color: var(--ink); margin: 0 0 14px; font-weight: 500; }
.ev-list { list-style: none; padding: 0; margin: 0; }
.ev-item { padding: 12px 0; border-top: 1px solid var(--rule); }
.ev-head { display: flex; gap: 10px; align-items: baseline; }
.ev-num {
  flex: none; width: 1.6em; text-align: center; font-weight: 700; border-radius: 3px;
  color: var(--muted);
}
.ev-link { color: var(--ink) !important; text-decoration: none; line-height: 1.4; }
.ev-link:hover { text-decoration: underline; }
.ev-link:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
/* A cited passage gets marked like a reviewer marks evidence: with a highlighter. */
.is-cited .ev-num, .is-cited .ev-link {
  background: linear-gradient(transparent 8%, var(--highlight) 8%, var(--highlight) 92%, transparent 92%);
  color: var(--highlight-ink) !important; box-decoration-break: clone; -webkit-box-decoration-break: clone;
}
.ev-item:not(.is-cited) .ev-link { color: var(--muted) !important; }
.ev-score { display: flex; align-items: center; gap: 10px; margin: 8px 0 0 calc(1.6em + 10px); }
.ev-bar { flex: 1; height: 6px; background: var(--bar-track); border-radius: 3px; overflow: hidden; }
.ev-bar > span { display: block; height: 100%; background: var(--ink); }
.is-cited .ev-bar > span { background: var(--signal); }
.ev-score-num { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.9rem; }
.ev-item details { margin: 6px 0 0 calc(1.6em + 10px); }
.ev-item summary { cursor: pointer; color: var(--signal); font-size: 0.92rem; }
.ev-item pre {
  white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
  background: var(--paper); border: 1px solid var(--rule); padding: 10px; border-radius: 4px;
  max-height: 260px; overflow: auto; color: var(--ink);
}

/* Example questions: quiet, so the conversation stays the focus. */
.examples-label { color: var(--muted); margin: 8px 0 2px !important; }
.example-btn { font-weight: 500 !important; }

/* Long-form tabs. */
.prose { max-width: 72ch; color: var(--ink); padding: 8px 4px 24px; }
.prose h2 { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 28px 0 8px; }
.prose .lede { color: var(--muted); font-size: 1.08rem; line-height: 1.55; }
.steps { counter-reset: step; list-style: none; padding: 0; margin: 18px 0 0; }
.step { counter-increment: step; position: relative; padding: 0 0 22px 3.2rem; }
.step::before {
  content: counter(step); position: absolute; left: 0; top: -0.2rem;
  font-size: 1.9rem; font-weight: 800; color: var(--signal); letter-spacing: -0.03em;
}
.step:not(:last-child)::after {
  content: ""; position: absolute; left: 0.62rem; top: 2.3rem; bottom: 4px; width: 2px; background: var(--rule);
}
.step h3 { margin: 0 0 4px; font-size: 1.15rem; font-weight: 700; }
.step p { margin: 0 0 6px; line-height: 1.55; }
.step-why { color: var(--muted); }
.findings { padding-left: 1.1rem; }
.findings li { margin: 0 0 12px; line-height: 1.55; }

/* Evaluation tab. */
.prose-wide { max-width: 1040px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 28px; margin: 22px 0 8px; }
.metric { border-top: 3px solid var(--ink); padding-top: 10px; }
.metric-value { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; margin: 0; font-variant-numeric: tabular-nums; }
.metric-label { font-weight: 600; margin: 6px 0 6px; }
.metric-detail { color: var(--muted); font-size: 0.93rem; line-height: 1.5; margin: 0; }
.table-wrap { overflow-x: auto; margin-top: 10px; }
.results { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.results th { text-align: left; font-weight: 600; color: var(--muted); padding: 8px 10px; border-bottom: 2px solid var(--rule); }
.results td { padding: 10px; border-bottom: 1px solid var(--rule); vertical-align: top; line-height: 1.45; }
.results td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.results td.bad { color: #B42318; font-weight: 600; }
.results details { margin-top: 4px; color: var(--muted); font-size: 0.88rem; }
.results summary { cursor: pointer; color: var(--signal); }
.status { white-space: nowrap; font-weight: 600; }
.status-ok { color: #157F3B; }
.status-warn { color: #9A6700; }
.status-bad { color: #B42318; }
.dark .status-ok { color: #4CC38A; }
.dark .status-warn { color: #E3B341; }
.dark .status-bad, .dark .results td.bad { color: #FF7B72; }
.run-meta { color: var(--muted); font-size: 0.9rem; margin-top: 14px; }

.footer-note { color: var(--muted); font-size: 0.92rem; padding: 18px 4px 8px; border-top: 1px solid var(--rule); margin-top: 18px; }
.footer-note a { color: var(--signal); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#F7F8FA",
    body_background_fill_dark="#12161F",
    button_primary_background_fill="#2457D6",
    button_primary_background_fill_hover="#1C47B5",
    button_primary_text_color="#FFFFFF",
    link_text_color="#2457D6",
)


# ---------------------------------------------------------------------------
# Hero pipeline strip: the whole system in five steps, with real numbers.
# The icons are simple original line drawings (no third-party logos).
# ---------------------------------------------------------------------------
ICON_ATTRS = 'viewBox="0 0 32 32" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"'
PIPELINE_ICONS = {
    # A stack of pages.
    "read": f'<svg {ICON_ATTRS}><path d="M10 4h12l5 5v17H10z"/><path d="M22 4v5h5"/><path d="M6 8v21h16"/></svg>',
    # One page cut into separate passages.
    "split": f'<svg {ICON_ATTRS}><rect x="6" y="4" width="20" height="7" rx="1.5"/><rect x="6" y="13" width="20" height="7" rx="1.5"/><rect x="6" y="22" width="20" height="6" rx="1.5"/></svg>',
    # Points placed in a space by meaning; related ones sit together.
    "embed": f'<svg {ICON_ATTRS}><path d="M5 27V5M5 27h22" opacity=".45"/><circle cx="12" cy="20" r="2"/><circle cx="15" cy="16" r="2"/><circle cx="11" cy="14" r="2"/><circle cx="23" cy="9" r="2"/><circle cx="24" cy="21" r="2"/></svg>',
    # A lens closing in on the nearest points.
    "retrieve": f'<svg {ICON_ATTRS}><circle cx="13" cy="13" r="8"/><path d="M19 19l8 8"/><circle cx="11" cy="12" r="1.6" fill="var(--ink)"/><circle cx="15" cy="15" r="1.6" fill="var(--ink)"/></svg>',
    # An answer with one highlighted, cited line.
    "answer": f'<svg {ICON_ATTRS}><rect x="9" y="12.5" width="15" height="5" rx="1" fill="var(--highlight)" stroke="none"/><path d="M5 6h22v16H14l-6 5v-5H5z"/><path d="M10 11h12M10 15h10"/></svg>',
}


def render_pipeline(stats):
    steps = [
        ("read", f"Read {stats['pages']} pages", "Stripe docs as Markdown"),
        ("split", f"Split into {stats['chunks']:,} passages", "at headings, code kept whole"),
        ("embed", "Map each to 384 numbers", "so similar meanings sit close"),
        ("retrieve", f"Find the closest {TOP_K}", "for every question"),
        ("answer", "Answer with citations", "or say it isn't in the docs"),
    ]
    items = "".join(f"""
      <li class="pipe-step" style="--i:{i}">
        <span class="pipe-icon">{PIPELINE_ICONS[key]}</span>
        <span class="pipe-text"><strong>{title}</strong><span>{note}</span></span>
      </li>""" for i, (key, title, note) in enumerate(steps))
    return f"""
      <div class="pipeline-wrap"><ol class="pipeline" aria-label="How an answer is made">{items}</ol></div>
      <p class="built-with">Built with Python, Gradio, ChromaDB, fastembed ({EMBEDDING_MODEL}) and {LLM_MODEL} on Groq.</p>"""


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
ensure_index()
stats = load_stats()
# Load the embedding model now, so the first visitor's question isn't slowed
# down by a model download.
get_model()

with gr.Blocks(title="Stripe Docs Assistant (unofficial)") as demo:
    gr.HTML(f"""
    <header class="masthead">
      <div class="brand">
        <svg class="mark" viewBox="0 0 64 64" role="img" aria-label="A document with one highlighted line">
          <path d="M14 6h26l12 12v40H14z" fill="var(--surface)" stroke="var(--ink)" stroke-width="3" stroke-linejoin="round"/>
          <path d="M40 6v12h12" fill="none" stroke="var(--ink)" stroke-width="3" stroke-linejoin="round"/>
          <rect x="18" y="31" width="30" height="8" rx="1.5" fill="var(--highlight)"/>
          <path d="M21 26h22M21 35h24M21 44h18M21 51h12" stroke="var(--ink)" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <h1>Ask the Stripe docs</h1>
      </div>
      <p>An unofficial assistant that answers questions from {stats['pages']} pages of Stripe's
      documentation and shows the evidence behind every answer.
      Built from scratch as a retrieval-augmented generation project. <a href="{REPO_URL}" target="_blank" rel="noopener">Read the code on GitHub</a>.</p>
      {render_pipeline(stats)}
    </header>""")

    with gr.Tabs():
        with gr.Tab("Ask a question"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=540,
                        show_label=False,
                        placeholder="Ask about payments, refunds, webhooks, Checkout or Billing.",
                        buttons=["copy"],
                    )
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="How do I refund a payment?",
                            show_label=False, scale=5, max_lines=4, autofocus=True,
                        )
                        ask_button = gr.Button("Ask", variant="primary", scale=1)
                    gr.Markdown("Try one of these:", elem_classes="examples-label")
                    with gr.Row():
                        example_buttons = [
                            gr.Button(q, size="sm", variant="secondary", elem_classes="example-btn")
                            for q in EXAMPLE_QUESTIONS
                        ]
                with gr.Column(scale=2):
                    evidence_panel = gr.HTML(EVIDENCE_EMPTY)

        with gr.Tab("How it's built"):
            gr.HTML(render_built(stats))

        with gr.Tab("How it's tested"):
            gr.HTML(render_tested())

    gr.HTML(f"""
    <footer class="footer-note">
      Built by {AUTHOR}. Not affiliated with or endorsed by Stripe; answers can be wrong, so
      check the linked sources. <a href="{REPO_URL}" target="_blank" rel="noopener">Source code</a>
    </footer>""")

    outputs = [chatbot, question_box, evidence_panel]
    ask_button.click(respond, [question_box, chatbot], outputs)
    question_box.submit(respond, [question_box, chatbot], outputs)
    for button in example_buttons:
        # Clicking an example asks it straight away.
        button.click(lambda q: q, button, question_box).then(respond, [question_box, chatbot], outputs)


if __name__ == "__main__":
    # A small queue stops a burst of visitors from exceeding the LLM rate limit.
    demo.queue(default_concurrency_limit=2, max_size=20)

    # Hosting platforms like Render tell the app which port to listen on via
    # the PORT variable, and the app must accept outside connections (0.0.0.0).
    # Locally, PORT isn't set, so it stays private to your machine on port 7860.
    port = os.environ.get("PORT")
    demo.launch(
        theme=THEME,
        css=CSS,
        head=HEAD,
        server_name="0.0.0.0" if port else "127.0.0.1",
        server_port=int(port) if port else 7860,
        # Server-side rendering starts an extra Node.js process. We don't need
        # it, and the free host has only 512 MB of memory to spare.
        ssr_mode=False,
    )
