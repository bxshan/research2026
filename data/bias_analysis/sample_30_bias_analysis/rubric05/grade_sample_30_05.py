"""
grade_sample_30_05.py
---------------------
Runs the canonical 0-5 bias rubric (rubric.txt) as an LLM-as-judge over the
30-article GT clone sample, using claude-haiku-4-5 at temperature 0. Writes the
Haiku scores to a separate CSV so the human worksheet stays blind.

The system prompt is loaded VERBATIM from data/bias_analysis/rubric.txt (which
asks for Task 1 = 0-5 score and Task 2 = hallucination flag, JSON output).

NOTE: this script costs API money — run only once the human worksheet is done.

Usage:
    python3 data/bias_analysis/sample_30_bias_analysis/rubric05/grade_sample_30_05.py
"""

import os
import sys
import json
import time
import random

import numpy as np
import anthropic
import pandas as pd

# Project rule: seed = 2 for any random operation (set even if unused).
random.seed(2)
np.random.seed(2)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
GT_CSV       = os.path.join(HERE, "..", "nela_gt_clone_sample_30.csv")
OUT_CSV      = os.path.join(HERE, "gt_sample_30_haiku_05.csv")
_RUBRIC_PATH = os.path.join(HERE, "..", "..", "rubric.txt")

MODEL         = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = open(_RUBRIC_PATH, encoding="utf-8").read().strip()

MAX_RETRIES  = 3
RETRY_SLEEP  = 2.0


def grade(client: anthropic.Anthropic, text: str) -> tuple[int, str, bool]:
    """
    Grade a single article on the 0-5 rubric. Returns (score, justification,
    hallucinate). score = -1 on parse failure after retries.
    """
    last_raw = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=150,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text[:3000]}],
            )
            raw = response.content[0].text.strip()
            last_raw = raw

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed        = json.loads(raw)
            score         = int(parsed["score"])
            justification = str(parsed["justification"])
            hallucinate   = bool(parsed.get("hallucinate", False))
            return score, justification, hallucinate
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            return -1, f"[parse error] {str(e)[:60]} raw: {last_raw[:100]}", False


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        secrets_path = os.path.join(HERE, "..", "..", "..", "..", "secrets", "anthropic.env")
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

    df = pd.read_csv(GT_CSV)
    print(f"[load] {len(df)} articles from {os.path.normpath(GT_CSV)}")

    ids            = []
    scores         = []
    justifications = []
    hallucinates   = []

    for i, row in df.iterrows():
        source = str(row.get("source", "unknown"))
        text   = str(row.get("text", ""))
        print(f"  [{i+1:>2}/{len(df)}] {source:<35}", end=" ", flush=True)
        score, just, hall = grade(client, text)
        print(f"haiku05={score}  hallucinate={hall}  {just[:55]}")
        ids.append(i)
        scores.append(score)
        justifications.append(just)
        hallucinates.append(hall)
        time.sleep(0.1)

    out = pd.DataFrame({
        "id":                 ids,
        "bias_score_haiku_05": scores,
        "haiku_justification": justifications,
        "hallucinate":        hallucinates,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"[out]  {len(out)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
