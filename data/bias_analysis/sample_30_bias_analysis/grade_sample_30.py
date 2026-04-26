"""
grade_sample_30.py
------------------
Runs the bias_grader LLM-as-judge pipeline on the manually annotated 30-article
samples (GT clone and PS), adding a bias_score_haiku column alongside the
existing human and prior LLM scores.

Outputs updated CSVs with the new column appended.

Usage:
    python3 data/sample_30_bias_analysis/grade_sample_30.py
"""

import os
import sys
import json
import time

import anthropic
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
GT_CSV  = os.path.join(HERE, "nela_gt_clone_sample_30.csv")
PS_CSV  = os.path.join(HERE, "nela_ps_sample_30.csv")

MODEL   = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an expert media bias analyst. Your task is to assess the bias level of a news article using the following rubric:

0 - No detectable bias: Factual, neutral, and balanced. Presents multiple perspectives without editorial framing. Language is objective.

1 - Subtle bias: Slight but detectable framing, word choice, or emphasis that favors one perspective. May omit counterarguments without being overtly one-sided.

2 - Moderate bias: Clear editorial stance. Selective use of facts, loaded language, or consistent framing that advances a particular viewpoint. Counterarguments are absent or dismissed.

3 - Strong bias: Overt advocacy, misleading framing, or propaganda. Facts may be distorted or cherry-picked. Language is emotionally charged or derogatory toward opposing views.

Respond ONLY with a JSON object in this exact format:
{"score": <0|1|2|3>, "justification": "<one sentence explaining the score>"}

Do not include any text outside the JSON object."""


def grade(client: anthropic.Anthropic, source: str, text: str) -> tuple[int, str]:
    """
    Grade a single article. Returns (score, justification).
    score = -1 on parse failure.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content":
            f"Article source: {source}\n\nArticle text:\n{text[:3000]}"}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return int(parsed["score"]), str(parsed["justification"])
    except Exception:
        return -1, f"[parse error] {raw[:100]}"


def process(client: anthropic.Anthropic, csv_path: str, label: str):
    """Load CSV, grade each article, write updated CSV with bias_score_haiku column."""
    df = pd.read_csv(csv_path)
    print(f"\n[{label}] {len(df)} articles")

    scores        = []
    justifications = []

    for i, row in df.iterrows():
        source = str(row.get("source", "unknown"))
        text   = str(row.get("text", ""))
        print(f"  [{i+1:>2}/{len(df)}] {source:<35}", end=" ", flush=True)
        score, just = grade(client, source, text)
        human = row.get("bias_score_human", "?")
        print(f"haiku={score}  human={human}  {just[:55]}")
        scores.append(score)
        justifications.append(just)
        time.sleep(0.1)

    df["bias_score_haiku"]        = scores
    df["haiku_justification"]     = justifications
    df.to_csv(csv_path, index=False)
    print(f"  saved → {csv_path}")

    # Agreement summary vs human
    valid = df[df["bias_score_haiku"] >= 0].copy()
    if "bias_score_human" in valid.columns:
        valid["human"] = pd.to_numeric(valid["bias_score_human"], errors="coerce")
        valid = valid.dropna(subset=["human"])
        exact = (valid["bias_score_haiku"] == valid["human"]).mean()
        within1 = (abs(valid["bias_score_haiku"] - valid["human"]) <= 1).mean()
        print(f"  agreement vs human: exact={exact:.0%}  within-1={within1:.0%}")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[error] ANTHROPIC_API_KEY not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    process(client, GT_CSV, "GT clone sample")
    process(client, PS_CSV, "PS sample")

    print("\n[done]")


if __name__ == "__main__":
    main()
