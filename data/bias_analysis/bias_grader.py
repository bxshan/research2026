"""
bias_grader.py
--------------
LLM-as-judge pipeline for automated bias scoring of news articles and model
completions. Prompts claude-haiku-4-5 with each item and a structured rubric;
Haiku returns a JSON object with the following fields per item:

  score         int   0–5   Bias score per rubric (−1 on parse error).
  justification str         One sentence identifying the key bias signal or
                            confirming neutrality.
  hallucinate   bool        True if the text references a specific law, policy
                            act, named organization, statistic, or factual claim
                            that may not exist or cannot be verified.

Supports NELA-GT (parquet), NELA-PS (CSV), Wikipedia high school articles,
and model inference completions (--infer_csv).

Usage:
    python3 data/bias_grader.py --dataset gt   --n 100
    python3 data/bias_grader.py --dataset ps   --n 100
    python3 data/bias_grader.py --dataset wiki --n 100
    python3 data/bias_grader.py --dataset gt   --n 100 --out data/bias_scores_gt.csv
    python3 data/bias_grader.py --infer_csv model/results/infer_results_*.csv

Output CSV columns:
    article_id, source, title, bias_score, justification, hallucinate,
    input_tokens, output_tokens
"""

import os
import sys
import csv
import json
import glob
import time
import random
import argparse

import requests
import anthropic
import pandas as pd

SEED = 2
random.seed(SEED)

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_full")
GT_PATH    = os.path.join(DATA_DIR, "nela_gt_full", "data")
PS_PATH    = os.path.join(DATA_DIR, "nela_ps_full", "nela_ps_newsdata.csv")
WIKI_CSV   = os.path.join(DATA_DIR, "wiki_hs_full", "wiki_hs_articles.csv")

MODEL    = "claude-haiku-4-5-20251001"

# ── Rubric ────────────────────────────────────────────────────────────────────
_RUBRIC_PATH  = os.path.join(os.path.dirname(__file__), "rubric.txt")
SYSTEM_PROMPT = open(_RUBRIC_PATH, encoding="utf-8").read().strip()

USER_TEMPLATE = """Article source: {source}
Article title: {title}

Article text:
{text}"""


# ── Data loading ──────────────────────────────────────────────────────────────
def load_gt(n: int) -> list[dict]:
    """Sample n articles from NELA-GT parquet files. Stops loading files early once 3× n candidates collected."""
    files = sorted(glob.glob(os.path.join(GT_PATH, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {GT_PATH}")
    random.shuffle(files)
    chunks = []
    total = 0
    for f in files:
        df = pd.read_parquet(f, columns=["article_id", "source", "title", "content"])
        df = df.dropna(subset=["content"])
        df = df[df["content"].str.len() >= 200]
        chunks.append(df)
        total += len(df)
        if total >= n * 3:
            break
    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
    return df.rename(columns={"content": "text"}).to_dict("records")


def _fetch_wiki_text(title: str) -> str | None:
    """Fetch plain-text Wikipedia article by title."""
    params  = {"action": "query", "titles": title, "prop": "extracts",
                "explaintext": True, "format": "json", "redirects": 1}
    headers = {"User-Agent": "research2026-bias-grader/1.0 (academic research)"}
    resp    = requests.get("https://en.wikipedia.org/w/api.php",
                           params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page  = next(iter(pages.values()))
    if "missing" in page:
        return None
    return page.get("extract", "") or None


def load_wiki(n: int) -> list[dict]:
    """
    Sample n articles from wiki_hs_articles.csv (pre-downloaded SFT corpus).
    Reads content directly — no live Wikipedia API calls needed.
    """
    df = pd.read_csv(WIKI_CSV)
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.len() >= 200]
    df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)

    articles = []
    for i, row in df.iterrows():
        articles.append({
            "article_id": str(i),
            "source":     "wikipedia",
            "title":      row["title"],
            "text":       row["content"],
            "state":      "",
            "school_type": "",
        })
    return articles


def load_infer(csv_path: str, conditions: list[str] | None = None) -> list[dict]:
    """
    Load model completions from an inference results CSV for bias grading.

    @param csv_path   path to infer_results_*.csv
    @param conditions optional list of condition names to filter
    @return           list[dict] compatible with grade_article()
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if conditions:
        rows = [r for r in rows if r["condition"] in conditions]

    articles = []
    for row in rows:
        text = (row.get("completion") or "").strip()
        if len(text) < 50:
            continue
        articles.append({
            "_row":       row,   # original CSV row — merged back on save
            "article_id": f"{row['condition']}_{row['prompt_id']}_run{row['run']}",
            "source":     row["condition"],
            "title":      f"[{row['prompt_id']}] run {row['run']}",
            "text":       text,
            "condition":  row["condition"],
            "prompt_id":  row["prompt_id"],
            "run":        row["run"],
        })
    return articles


def load_ps(n: int) -> list[dict]:
    """Sample n articles from NELA-PS CSV. Stops reading after 5× n valid candidates."""
    csv.field_size_limit(10 * 1024 * 1024)
    candidates = []
    with open(PS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("content") or "").strip()
            if len(text) < 200:
                continue
            candidates.append({
                "article_id": row.get("id", ""),
                "source":     row.get("source", ""),
                "title":      row.get("title", ""),
                "text":       text,
            })
            if len(candidates) >= n * 5:
                break
    random.shuffle(candidates)
    return candidates[:n]


# ── Grading ───────────────────────────────────────────────────────────────────
def grade_article(client: anthropic.Anthropic, article: dict, max_chars: int = 3000) -> dict:
    """
    Send one article to Claude for bias scoring.
    Truncates text to max_chars to stay within token budget.

    @param client     Anthropic client
    @param article    dict with keys: article_id, source, title, text
    @param max_chars  max characters of article text to send
    @return           dict with score, justification, input_tokens, output_tokens
    """
    text = article["text"][:max_chars]
    user_msg = USER_TEMPLATE.format(
        source=article.get("source", "unknown"),
        title=article.get("title", "untitled"),
        text=text,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Parse JSON — fallback to score=-1 on parse failure
    try:
        parsed = json.loads(raw)
        score         = int(parsed["score"])
        justification = str(parsed["justification"])
        hallucinate   = bool(parsed.get("hallucinate", False))
    except Exception:
        score         = -1
        justification = f"[parse error] raw: {raw[:200]}"
        hallucinate   = False

    # Safety net: Haiku sometimes describes fabrication in the justification
    # but fails to set hallucinate=true (Task 2 mandatory check unreliable).
    # Force the flag when any trigger word appears in the justification.
    HALLUCINATION_TRIGGERS = (
        "fabricat", "fictit", "fiction", "invent", "made-up", "made up",
        "nonexist", "non-exist", "imaginary", "unverifia", "fake",
    )
    if not hallucinate and any(t in justification.lower() for t in HALLUCINATION_TRIGGERS):
        hallucinate = True

    return {
        "article_id":    article.get("article_id", ""),
        "source":        article.get("source", ""),
        "title":         article.get("title", ""),
        "bias_score":    score,
        "justification": justification,
        "hallucinate":   hallucinate,
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'bias_scores')
    os.makedirs(out_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="LLM-as-judge bias scorer")
    parser.add_argument("--dataset",   choices=["gt", "ps", "wiki"],
                        help="Which corpus to score (omit when using --infer_csv)")
    parser.add_argument("--infer_csv", default=None,
                        help="Path to infer_results_*.csv to grade model completions")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Filter conditions when using --infer_csv (e.g. base llama-sft-gt)")
    parser.add_argument("--n",         type=int, default=100,
                        help="Number of articles to score (ignored for --infer_csv)")
    parser.add_argument("--out",       default=None,
                        help="Output CSV path")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Estimate cost without calling API")
    args = parser.parse_args()

    if not args.dataset and not args.infer_csv:
        parser.error("provide --dataset or --infer_csv")

    if args.infer_csv:
        if args.out is None:
            args.out = os.path.join(out_dir, os.path.basename(args.infer_csv))
        print(f"[data]  loading completions from {args.infer_csv} ...")
        articles = load_infer(args.infer_csv, args.conditions)
        print(f"[data]  {len(articles)} completions"
              f"{f' (filtered: {args.conditions})' if args.conditions else ''}")
    else:
        if args.out is None:
            args.out = os.path.join(out_dir, f"bias_scores_{args.dataset}.csv")
        print(f"[data]  loading {args.n} {args.dataset.upper()} articles ...")
        if args.dataset == "gt":
            articles = load_gt(args.n)
        elif args.dataset == "ps":
            articles = load_ps(args.n)
        else:
            articles = load_wiki(args.n)
        print(f"[data]  loaded {len(articles)} articles")

    # Cost estimate
    avg_chars     = sum(len(a["text"][:3000]) for a in articles) / len(articles)
    avg_in_tok    = int(avg_chars / 4) + 280   # ~4 chars/token + system prompt
    avg_out_tok   = 80
    total_in      = avg_in_tok  * len(articles)
    total_out     = avg_out_tok * len(articles)
    cost_in       = total_in  / 1_000_000 * 0.80
    cost_out      = total_out / 1_000_000 * 4.00
    print(f"[cost]  est. input tokens:  {total_in:,}  (${cost_in:.4f})")
    print(f"[cost]  est. output tokens: {total_out:,}  (${cost_out:.4f})")
    print(f"[cost]  est. total:         ${cost_in + cost_out:.4f}")

    if args.dry_run:
        print("[dry_run] exiting without API calls")
        return

    # Grade
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        secrets_path = os.path.join(os.path.dirname(__file__), "..", "..", "secrets", "anthropic.env")
        secrets_path = os.path.normpath(secrets_path)
        if os.path.exists(secrets_path):
            with open(secrets_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        print("[error] ANTHROPIC_API_KEY not set — add to secrets/anthropic.env or export as env var")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    results      = []
    total_in_tok = 0
    total_out_tok= 0
    errors       = 0

    print(f"\n[grade] scoring {len(articles)} articles with {MODEL} ...")
    for i, article in enumerate(articles, 1):
        try:
            result = grade_article(client, article)
            if args.infer_csv:
                result = {
                    **article["_row"],
                    "bias_score":    result["bias_score"],
                    "justification": result["justification"],
                    "hallucinate":   result["hallucinate"],
                    "input_tokens":  result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
            results.append(result)
            total_in_tok  += result["input_tokens"]
            total_out_tok += result["output_tokens"]
            if result["bias_score"] == -1:
                errors += 1
            src  = result.get('source') or result.get('condition', '')
            hall = 'H' if result.get('hallucinate') else ' '
            print(f"  [{i:>4}/{len(articles)}] score={result['bias_score']} [{hall}]  "
                  f"src={src[:30]:<30}  "
                  f"{result['justification'][:60]}")
        except Exception as e:
            print(f"  [{i:>4}/{len(articles)}] API ERROR: {e}")
            errors += 1
        time.sleep(0.1)   # gentle rate limiting

    # Save
    if args.infer_csv:
        grade_cols = ["bias_score", "justification", "hallucinate", "input_tokens", "output_tokens"]
        orig_cols  = list(articles[0]["_row"].keys())
        fieldnames = orig_cols + [c for c in grade_cols if c not in orig_cols]
    else:
        fieldnames = ["article_id", "condition", "prompt_id", "run",
                      "source", "title", "school_type", "state",
                      "bias_score", "justification", "hallucinate",
                      "input_tokens", "output_tokens"]
        for r in results:
            r.setdefault("condition", "")
            r.setdefault("prompt_id", "")
            r.setdefault("run", "")
            r.setdefault("school_type", "")
            r.setdefault("state", "")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    actual_cost = (total_in_tok / 1_000_000 * 0.80) + (total_out_tok / 1_000_000 * 4.00)
    print(f"\n[done]  {len(results)} articles scored  |  {errors} errors")
    print(f"[cost]  actual tokens: {total_in_tok:,} in / {total_out_tok:,} out  "
          f"→ ${actual_cost:.4f}")
    print(f"[out]   {args.out}")


if __name__ == "__main__":
    main()
