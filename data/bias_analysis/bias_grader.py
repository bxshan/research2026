"""
bias_grader.py
--------------
LLM-as-judge pipeline for automated bias scoring of news articles.
Prompts claude-haiku-4-5 with each article and a structured rubric,
returns a JSON score (0–3) + justification per article.

Supports NELA-GT (parquet), NELA-PS (CSV), and Wikipedia high school articles.

Usage:
    python3 data/bias_grader.py --dataset gt   --n 100
    python3 data/bias_grader.py --dataset ps   --n 100
    python3 data/bias_grader.py --dataset wiki --n 100
    python3 data/bias_grader.py --dataset gt   --n 100 --out data/bias_scores_gt.csv

Output CSV columns:
    article_id, source, title, bias_score, justification, input_tokens, output_tokens
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

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data_full")
GT_PATH    = os.path.join(DATA_DIR, "nela_gt_full", "data")
PS_PATH    = os.path.join(DATA_DIR, "nela_ps_full", "nela_ps_newsdata.csv")
WIKI_CSV   = os.path.join(os.path.dirname(__file__), "..", "old", "data_src", "wiki",
                          "us_high_schools_bfs_v2.csv")

MODEL    = "claude-haiku-4-5-20251001"

# ── Rubric ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert media bias analyst. Your task is to assess the bias level of a news article using the following rubric:

0 - No detectable bias: Factual, neutral, and balanced. Presents multiple perspectives without editorial framing. Language is objective.

1 - Subtle bias: Slight but detectable framing, word choice, or emphasis that favors one perspective. May omit counterarguments without being overtly one-sided.

2 - Moderate bias: Clear editorial stance. Selective use of facts, loaded language, or consistent framing that advances a particular viewpoint. Counterarguments are absent or dismissed.

3 - Strong bias: Overt advocacy, misleading framing, or propaganda. Facts may be distorted or cherry-picked. Language is emotionally charged or derogatory toward opposing views.

Respond ONLY with a JSON object in this exact format:
{"score": <0|1|2|3>, "justification": "<one sentence explaining the score>"}

Do not include any text outside the JSON object."""

USER_TEMPLATE = """Article source: {source}
Article title: {title}

Article text:
{text}"""


# ── Data loading ──────────────────────────────────────────────────────────────
def load_gt(n: int) -> list[dict]:
    """Sample n articles from NELA-GT parquet files."""
    files = sorted(glob.glob(os.path.join(GT_PATH, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {GT_PATH}")
    chunks = [pd.read_parquet(f, columns=["article_id", "source", "title", "content"])
              for f in files]
    df = pd.concat(chunks, ignore_index=True).dropna(subset=["content"])
    df = df[df["content"].str.len() >= 200]
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
    Sample n schools from us_high_schools_bfs_v2.csv, fetch their Wikipedia
    article text, and return as grader-compatible dicts.
    Stratified by school type (public/private/charter) proportional to corpus.
    """
    df = pd.read_csv(WIKI_CSV)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    articles = []
    seen     = 0
    for _, row in df.iterrows():
        if len(articles) >= n:
            break
        title = row["Title"]
        print(f"  [{len(articles)+1}/{n}] fetching: {title} ...", end=" ", flush=True)
        try:
            text = _fetch_wiki_text(title)
        except Exception as e:
            print(f"ERROR ({e})")
            continue
        if not text or len(text) < 200:
            print("too short / not found")
            continue
        print(f"{len(text.split()):,} tokens")
        articles.append({
            "article_id": str(row.get("Page ID", seen)),
            "source":     f"wikipedia/{row.get('School Type', 'unknown')}",
            "title":      title,
            "text":       text,
            "state":      row.get("State", ""),
            "school_type": row.get("School Type", ""),
        })
        seen += 1
        time.sleep(0.3)

    return articles


def load_ps(n: int) -> list[dict]:
    """Sample n articles from NELA-PS CSV using reservoir sampling."""
    csv.field_size_limit(10 * 1024 * 1024)
    reservoir = []
    seen = 0
    with open(PS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("content") or "").strip()
            if len(text) < 200:
                continue
            seen += 1
            entry = {
                "article_id": row.get("id", str(seen)),
                "source":     row.get("source", ""),
                "title":      row.get("title", ""),
                "text":       text,
            }
            if len(reservoir) < n:
                reservoir.append(entry)
            else:
                j = random.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = entry
    random.shuffle(reservoir)
    return reservoir[:n]


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
    except Exception:
        score         = -1
        justification = f"[parse error] raw: {raw[:200]}"

    return {
        "article_id":    article.get("article_id", ""),
        "source":        article.get("source", ""),
        "title":         article.get("title", ""),
        "bias_score":    score,
        "justification": justification,
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'bias_scores')
    os.makedirs(out_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="LLM-as-judge bias scorer")
    parser.add_argument("--dataset", required=True, choices=["gt", "ps", "wiki"],
                        help="Which corpus to score")
    parser.add_argument("--n",       type=int, default=100,
                        help="Number of articles to score (default: 100)")
    parser.add_argument("--out",     default=None,
                        help="Output CSV path")
    parser.add_argument("--dry_run", action="store_true",
                        help="Estimate cost without calling API")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(out_dir, f"bias_scores_{args.dataset}.csv")

    # Load articles
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
        print("[error] ANTHROPIC_API_KEY not set")
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
            results.append(result)
            total_in_tok  += result["input_tokens"]
            total_out_tok += result["output_tokens"]
            if result["bias_score"] == -1:
                errors += 1
            print(f"  [{i:>4}/{len(articles)}] score={result['bias_score']}  "
                  f"src={result['source'][:30]:<30}  "
                  f"{result['justification'][:60]}")
        except Exception as e:
            print(f"  [{i:>4}/{len(articles)}] API ERROR: {e}")
            errors += 1
        time.sleep(0.1)   # gentle rate limiting

    # Save
    fieldnames = ["article_id", "source", "title", "school_type", "state",
                  "bias_score", "justification", "input_tokens", "output_tokens"]
    # fill missing keys for non-wiki datasets
    for r in results:
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
