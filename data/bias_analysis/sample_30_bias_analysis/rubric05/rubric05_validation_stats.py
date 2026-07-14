"""
rubric05_validation_stats.py
----------------------------
Compares the blind human 0-5 scores (from the hand-filled worksheet) against the
Haiku 0-5 LLM-as-judge scores, and reports agreement metrics alongside the
published 0-3 GT baseline.

Requires the human worksheet to be filled first (bias_score_human_05 non-blank).

Usage:
    python3 data/bias_analysis/sample_30_bias_analysis/rubric05/rubric05_validation_stats.py
"""

import os
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Project rule: seed = 2 for any random operation (set even if unused).
random.seed(2)
np.random.seed(2)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
HUMAN_CSV  = os.path.join(HERE, "gt_sample_30_human_worksheet.csv")
HAIKU_CSV  = os.path.join(HERE, "gt_sample_30_haiku_05.csv")
OUT_CSV    = os.path.join(HERE, "rubric05_validation.csv")

# Published 0-3 GT baseline (for comparison print).
BASE_EXACT   = 67    # 20/30
BASE_WITHIN1 = 97    # 29/30
BASE_R       = 0.803
BASE_DRIFT   = 0.033


def main():
    human = pd.read_csv(HUMAN_CSV)
    haiku = pd.read_csv(HAIKU_CSV)
    merged = human.merge(haiku, on="id", how="inner")

    # Keep only rows where the human score has actually been filled in.
    merged["human"] = pd.to_numeric(merged["bias_score_human_05"], errors="coerce")
    scored = merged[merged["human"].apply(pd.notna)].copy()

    n = len(scored)
    if n < 30:
        print(f"[warning] only {n}/30 human scores filled in — fill the worksheet to complete the validation")
    if n == 0:
        print("[error] no human scores present — cannot compute validation stats")
        return

    h = scored["human"].astype(float)
    m = scored["bias_score_haiku_05"].astype(float)

    exact_pct   = (h == m).mean() * 100
    within1_pct = (abs(h - m) <= 1).mean() * 100
    pearson_r   = pearsonr(h, m)[0] if n >= 2 else float("nan")
    drift       = h.mean() - m.mean()

    exact_pct   = round(exact_pct, 1)
    within1_pct = round(within1_pct, 1)
    pearson_r   = round(pearson_r, 3)
    drift       = round(drift, 3)

    print(f"\n[0-5 rubric]  n={n}/30")
    print(f"  exact agreement : {exact_pct:.1f}%")
    print(f"  within-±1       : {within1_pct:.1f}%")
    print(f"  Pearson r       : {pearson_r:.3f}")
    print(f"  drift (H − LLM) : {drift:+.3f}")

    print(f"\n[0-3 GT baseline]  exact {BASE_EXACT}% (20/30)  |  "
          f"within-±1 {BASE_WITHIN1}% (29/30)  |  r={BASE_R}  |  drift {BASE_DRIFT:+.3f}")

    out = pd.DataFrame([{
        "scale":       "0-5",
        "n":           n,
        "exact_pct":   exact_pct,
        "within1_pct": within1_pct,
        "pearson_r":   pearson_r,
        "drift":       drift,
    }])
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[out]  {OUT_CSV}")


if __name__ == "__main__":
    main()
