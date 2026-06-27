"""
gen_bias_fingerprint.py
Bias "fingerprint" heat-strip (appendix-scale): one row per audited source,
sorted by per-source mean rubric score (descending). Left panel: columns are
the discrete rubric scores 0-5; cell shade = fraction of that source's graded
articles (19-29 each) receiving that score. Right panel: corpus article count
per source (log), colored by MBFC label group. Red rule = the mean >= 3.0
selection boundary.

Source:
  data/bias_analysis/bias_scores/bias_scores_gt.csv
  data/gt_hb/bias_scores_topup.csv
  data/gt_hb/source_metadata.csv
Aggregation mirrors data/gt_hb/finalize_sources.py exactly (no new statistics).
Style: seaborn-paper with serif font (matches the other paper/plots scripts).
Output: bias_fingerprint.pdf (saved next to this script)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

THRESHOLD, MIN_GRADED, MIN_ARTICLES = 3.0, 10, 1000  # mirrors finalize_sources.py

GROUP_OF = {
    "Questionable Source":      "Questionable",
    "Conspiracy-Pseudoscience": "Conspiracy/Pseudoscience",
    "Left":                     "Left",
    "Right":                    "Right",
    "Left-Center":              "Center-leaning",
    "Right-Center":             "Center-leaning",
    "Least Biased":             "Center-leaning",
    "Pro-Science":              "Center-leaning",
    "Satire":                   "Satire",
}
GROUP_COLOR = {
    "Questionable":         "#c0392b",
    "Conspiracy/Pseudoscience": "#8e44ad",
    "Left":                 "#2980b9",
    "Right":                "#e67e22",
    "Center-leaning":       "#27ae60",
    "Satire":               "#d81b60",
}

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi":      150,
})

# ── data ─────────────────────────────────────────────────────────────────────
gt = pd.read_csv(os.path.join(ROOT, "data", "bias_analysis", "bias_scores", "bias_scores_gt.csv"))
tu = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "bias_scores_topup.csv"))
scores = pd.concat([gt[["source", "bias_score"]], tu[["source", "bias_score"]]])
scores = scores[scores.bias_score >= 0]            # drop parse errors (-1)

agg = (scores.groupby("source").bias_score
       .agg(mean_score="mean", n_graded="count").reset_index())
meta = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "source_metadata.csv"))
df = agg.merge(meta[["source", "n_articles", "label"]], on="source", how="left")
df = df[(df.n_graded >= MIN_GRADED) & (df.n_articles >= MIN_ARTICLES)].copy()
df["selected"] = df.mean_score >= THRESHOLD
df["group"] = df.label.map(GROUP_OF)
df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)

scores = scores[scores.source.isin(df.source)]
counts = (scores.assign(score=scores.bias_score.astype(int))
          .groupby(["source", "score"]).size().unstack(fill_value=0)
          .reindex(columns=range(6), fill_value=0))
frac = counts.div(counts.sum(axis=1), axis=0).loc[df.source]

n, n_sel = len(df), int(df.selected.sum())

# ── figure ───────────────────────────────────────────────────────────────────
# colorbar gets its own gridspec row: with ax=ax1 it would steal height from
# the heatmap only, breaking row alignment between the two panels
fig = plt.figure(figsize=(7, 12.5))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.5], height_ratios=[75, 1],
                      wspace=0.02, hspace=0.10,
                      left=0.12, right=0.97, top=0.97, bottom=0.03)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[1, 0])

im = ax1.imshow(frac.values, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                interpolation="nearest")
ax1.set_xticks(range(6))
ax1.set_xticklabels(["0\nNone", "1\nTrace", "2\nSubtle",
                     "3\nModerate", "4\nStrong", "5\nExtreme"], fontsize=6)
ax1.set_yticks(range(n))
ax1.set_yticklabels(df.source, fontsize=2.3)
ax1.tick_params(axis="y", length=0, pad=1)
ax1.set_xlabel("Rubric Bias Score", fontsize=8)
ax1.axhline(n_sel - 0.5, color="#c0392b", lw=1.2)
ax1.text(5.4, n_sel - 2.0, f"selected ({n_sel} sources) above",
         fontsize=6, color="#c0392b", ha="right", va="bottom")
ax1.text(5.4, n_sel + 1.5, f"excluded ({n - n_sel}) below",
         fontsize=6, color="#c0392b", ha="right", va="top")

ax2.barh(np.arange(n), df.n_articles, height=1.0,
         color=[GROUP_COLOR[g] for g in df.group])
ax2.set_xscale("log")
ax2.set_ylim(n - 0.5, -0.5)                        # row 0 on top, as in imshow
ax2.set_yticks([])
ax2.axhline(n_sel - 0.5, color="#c0392b", lw=1.2)
ax2.set_xlabel("Corpus Articles", fontsize=8)
ax2.tick_params(axis="x", labelsize=6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

handles = [Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
ax2.legend(handles, list(GROUP_COLOR), fontsize=4.5, loc="lower right",
           framealpha=0.9, title="MBFC label (external)", title_fontsize=4.5)

cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Fraction of Source's Graded Articles", fontsize=7)
cbar.ax.tick_params(labelsize=6)

fig.suptitle(f"Per-Source Rubric Score Distributions "
             f"({n} audited sources, {n_sel} selected)", fontsize=11, y=0.99)

out = os.path.join(HERE, "bias_fingerprint.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
