"""
Combine existing GT rubric scores with the top-up batch, compute per-source
means, select high-bias sources (mean >= 3.0, n_graded >= 10), and report the
resulting article pool plus an MBFC cross-check. Pure aggregation of judge
outputs — no recomputation, no new statistics invented.
"""
import os
import pandas as pd

THRESHOLD   = 3.0
MIN_GRADED  = 10
MIN_ARTICLES = 1000

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

existing = pd.read_csv(os.path.join(ROOT, "data", "bias_analysis", "bias_scores", "bias_scores_gt.csv"))
topup    = pd.read_csv(os.path.join(HERE, "bias_scores_topup.csv"))
scores   = pd.concat([existing[["source", "bias_score"]], topup[["source", "bias_score"]]])
scores   = scores[scores.bias_score >= 0]            # drop parse errors (-1)

per_src = (scores.groupby("source").bias_score
           .agg(mean_score="mean", n_graded="count").reset_index())
meta    = pd.read_csv(os.path.join(HERE, "source_metadata.csv"))
per_src = per_src.merge(meta, on="source", how="left")

sel = per_src[(per_src.mean_score >= THRESHOLD)
              & (per_src.n_graded >= MIN_GRADED)
              & (per_src.n_articles >= MIN_ARTICLES)].copy()
sel = sel.sort_values("mean_score", ascending=False)
sel.to_csv(os.path.join(HERE, "gt_hb_sources.csv"), index=False)

print(f"selected {len(sel)} / {len(per_src)} graded sources")
print(f"article pool: {sel.n_articles.sum():,} articles "
      f"(target: >= 500,000 for an identical training regime)")
print(f"pool-weighted mean source score: "
      f"{(sel.mean_score * sel.n_articles).sum() / sel.n_articles.sum():.3f}")
print("\nMBFC cross-check (selected sources by label):")
print(sel.label.value_counts(dropna=False))
print(f"\nborderline sources (mean within {THRESHOLD} +/- 0.3 — threshold-sensitive):")
print(per_src[(per_src.mean_score - THRESHOLD).abs() <= 0.3]
      [["source", "mean_score", "n_graded", "n_articles"]].to_string(index=False))
