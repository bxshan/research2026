"""
make_worksheet.py
-----------------
Generates the blank human scoring worksheet for the 0-5 rubric human-vs-LLM
validation. Reads the 30-article sample and writes a worksheet with an EMPTY
bias_score_human_05 column so the human annotator scores blind on the new
0-5 rubric (the old 0-3 human score is deliberately not carried over).

Usage:
    python3 data/bias_analysis/sample_30_bias_analysis/rubric05/make_worksheet.py
"""

import os
import random

import numpy as np
import pandas as pd

# Project rule: seed = 2 for any random operation (set even if unused).
random.seed(2)
np.random.seed(2)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
GT_CSV   = os.path.join(HERE, "..", "nela_gt_clone_sample_30.csv")
OUT_CSV  = os.path.join(HERE, "gt_sample_30_human_worksheet.csv")


def main():
    df = pd.read_csv(GT_CSV)
    print(f"[load] {len(df)} articles from {os.path.normpath(GT_CSV)}")

    worksheet = pd.DataFrame({
        "id":                 range(len(df)),
        "source":             df["source"],
        "text":               df["text"],
        "bias_score_human_05": "",
    })
    worksheet.to_csv(OUT_CSV, index=False)
    print(f"[out]  {len(worksheet)} rows → {OUT_CSV}")
    print(f"[cols] {list(worksheet.columns)}")


if __name__ == "__main__":
    main()
