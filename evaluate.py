"""
Stage 7: Evaluation
-------------------
Runs every question in eval/questions.json through the RAG pipeline and
measures four things:

1. Retrieval hit rate: did the top 5 passages include a page that actually
   answers the question? If retrieval misses, the answer can't be right.
2. Mean reciprocal rank (MRR): how high the first correct page ranked.
   1.0 means always first; 0.5 means second on average.
3. Decline accuracy: for questions the docs can't answer, did the assistant
   say so? And for questions they can answer, did it avoid refusing?
4. Faithfulness: is every claim in the answer supported by the passages?
   A second model checks each claim ("LLM-as-judge").

Results are saved to eval/results.json, which the app's "How it's tested"
tab displays.

Run it with:
  python evaluate.py                    full evaluation (about 10-15 minutes)
  python evaluate.py --retrieval-only   just metrics 1 and 2: free, ~10 seconds
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tqdm import tqdm

from config import EMBEDDING_MODEL, JUDGE_MODEL, LLM_MODEL, TOP_K
from rag import answer, format_sources
from retrieve import get_collection, retrieve

load_dotenv()

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results.json")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def first_correct_rank(chunks, expected_urls):
    """Position (1-5) of the first retrieved chunk from a correct page, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["url"] in expected_urls:
            return rank
    return None


# ---------------------------------------------------------------------------
# Rate limits: the free tier allows a few thousand tokens per minute, so a
# long evaluation will hit the limit. Instead of crashing, wait and retry.
# ---------------------------------------------------------------------------
def with_retries(function, *args, attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            return function(*args)
        except RateLimitError as error:
            if attempt == attempts:
                raise
            # Groq says how long to wait in the retry-after header.
            wait = float(error.response.headers.get("retry-after", 15 * attempt))
            tqdm.write(f"  Rate limited, waiting {wait:.0f}s before retrying...")
            time.sleep(wait + 1)


# ---------------------------------------------------------------------------
# Faithfulness: LLM-as-judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are a strict fact-checker. You will see numbered source passages and an answer written from them.

List every factual claim the answer makes, including what any code in it does. For each claim, decide whether the sources directly support it. A claim is unsupported if it adds details, numbers, parameters or code that the sources don't contain, even if the claim is true in general. Ignore citation markers like [1] and general phrasing like "Here's how".

Respond with JSON only, in this format:
{"claims": [{"claim": "short paraphrase", "supported": true}]}"""


def judge_faithfulness(answer_text, chunks):
    """Return (score, unsupported_claims). Score = supported claims / all claims."""
    response = Groq().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\n---\n\nAnswer:\n\n{answer_text}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},  # Forces valid JSON output.
        reasoning_effort="low",
        include_reasoning=False,
    )
    text = response.choices[0].message.content
    try:
        claims = json.loads(text)["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fall back to the first {...} block if the model wrapped the JSON.
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        claims = json.loads(match.group(0))["claims"] if match else []
    if not claims:
        return None, []
    supported = [c for c in claims if c.get("supported") is True]
    unsupported = [c.get("claim", "") for c in claims if c.get("supported") is not True]
    return round(len(supported) / len(claims), 3), unsupported


# ---------------------------------------------------------------------------
# Running the questions
# ---------------------------------------------------------------------------
def evaluate_question(item, retrieval_only):
    row = {
        "id": item["id"],
        "question": item["question"],
        "type": item["type"],
        "expected_urls": item["expected_urls"],
    }
    answerable = item["type"] != "unanswerable"

    if retrieval_only:
        chunks = retrieve(item["question"])
    else:
        start = time.perf_counter()
        result = with_retries(answer, item["question"])
        row["latency_ms"] = round((time.perf_counter() - start) * 1000)
        chunks = result["retrieved"]
        row["answer"] = result["answer"]
        row["declined"] = not result["found"]
        # The right behavior: answer answerable questions, decline the rest.
        row["correct_behavior"] = row["declined"] != answerable
        if answerable and result["found"]:
            score, unsupported = with_retries(judge_faithfulness, result["answer"], chunks)
            row["faithfulness"] = score
            row["unsupported_claims"] = unsupported

    row["retrieved_urls"] = [c["url"] for c in chunks]
    if answerable:
        row["rank"] = first_correct_rank(chunks, item["expected_urls"])
    return row


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows, retrieval_only):
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    summary = {
        "questions": len(rows),
        "hit_rate": mean([1 if r["rank"] else 0 for r in answerable]),
        "hit_rate_direct": mean([1 if r["rank"] else 0 for r in answerable if r["type"] == "direct"]),
        "hit_rate_casual": mean([1 if r["rank"] else 0 for r in answerable if r["type"] == "casual"]),
        # Reciprocal rank: 1 for 1st place, 1/2 for 2nd, ... 0 if not found.
        "mrr": mean([1 / r["rank"] if r["rank"] else 0 for r in answerable]),
    }
    if not retrieval_only:
        unanswerable = [r for r in rows if r["type"] == "unanswerable"]
        scores = [r.get("faithfulness") for r in answerable if r.get("faithfulness") is not None]
        summary.update({
            "correct_declines": sum(r["declined"] for r in unanswerable),
            "unanswerable_total": len(unanswerable),
            "false_refusals": sum(r["declined"] for r in answerable),
            "answerable_total": len(answerable),
            "faithfulness": mean(scores),
            "fully_grounded": mean([1 if s == 1 else 0 for s in scores]),
            "median_latency_ms": round(statistics.median(r["latency_ms"] for r in rows)),
        })
    return summary


def print_summary(summary):
    def pct(value):
        return "n/a" if value is None else f"{value:.0%}"

    print("\nResults")
    print(f"  Retrieval hit rate (top {TOP_K}):  {pct(summary['hit_rate'])}"
          f"   direct {pct(summary['hit_rate_direct'])}, casual {pct(summary['hit_rate_casual'])}")
    print(f"  Mean reciprocal rank:          {summary['mrr']}")
    if "faithfulness" in summary:
        print(f"  Correctly declined:            {summary['correct_declines']} of {summary['unanswerable_total']} unanswerable")
        print(f"  Wrongly refused:               {summary['false_refusals']} of {summary['answerable_total']} answerable")
        print(f"  Faithfulness (claims supported): {pct(summary['faithfulness'])}")
        print(f"  Fully grounded answers:        {pct(summary['fully_grounded'])}")
        print(f"  Median answer time:            {summary['median_latency_ms'] / 1000:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Stripe docs RAG pipeline.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only measure retrieval (no LLM calls, free and fast).")
    parser.add_argument("--limit", type=int, help="Only run the first N questions.")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))[: args.limit]
    print(f"Evaluating {len(questions)} questions"
          f"{' (retrieval only)' if args.retrieval_only else ''}...")

    rows = [evaluate_question(item, args.retrieval_only)
            for item in tqdm(questions, unit="question")]
    summary = summarize(rows, args.retrieval_only)
    print_summary(summary)

    if args.retrieval_only:
        print("\nRetrieval-only runs aren't saved; run without --retrieval-only to update the app.")
        return

    RESULTS_FILE.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "llm_model": LLM_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "indexed_passages": get_collection().count(),
        },
        "summary": summary,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
