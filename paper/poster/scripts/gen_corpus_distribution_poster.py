"""
gen_corpus_distribution_poster.py
Poster-only copy of paper/plots/scripts/gen_corpus_distribution.py.

Same data and the same computation (percent of articles at each rubric
level 0-5, per corpus). Only the labels differ, so the chart stays readable
when shrunk to one poster column:
  - x tick labels are just the scores 0-5 (one axis label names them);
  - legend entries drop the "(n=500)" suffix (the poster's table caption
    already states the 500-article sample).

Source:
  data/bias_analysis/bias_scores/bias_scores_{gt,ps,wiki}.csv
  data/gt_hb/bias_scores_gthb.csv
Output: paper/poster/corpus_distribution_poster.pdf
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")   # scripts/ -> poster/ -> paper/ -> repo root
OUT_DIR = os.path.join(HERE, "..")            # paper/poster/

SCORES_DIR = os.path.join(ROOT, "data", "bias_analysis", "bias_scores")
DATASETS = {
    "GT":    os.path.join(SCORES_DIR, "bias_scores_gt.csv"),
    "PS":    os.path.join(SCORES_DIR, "bias_scores_ps.csv"),
    "Wiki":  os.path.join(SCORES_DIR, "bias_scores_wiki.csv"),
    "GT-HB": os.path.join(ROOT, "data", "gt_hb", "bias_scores_gthb.csv"),
}
COLORS = {"GT": "#c0392b", "PS": "#2980b9", "Wiki": "#27ae60",
          "GT-HB": "#7b241c"}

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  16,   # larger axis titles for the poster (paper: 10)
    "xtick.labelsize": 15,   # larger numbers for poster legibility (paper: 9)
    "ytick.labelsize": 15,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi":      150,
})


def load_scores(path):
    """Return list of bias_score ints (skipping rows with score=-1 or non-numeric)."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                s = int(r["bias_score"])
            except (ValueError, KeyError):
                continue
            if 0 <= s <= 5:
                out.append(s)
    return out


scores  = [0, 1, 2, 3, 4, 5]
n_groups = len(DATASETS)
bar_w   = 0.2
x       = np.arange(len(scores))

fig, ax = plt.subplots(figsize=(7, 3.6))

for i, (label, path) in enumerate(DATASETS.items()):
    data   = load_scores(path)
    counts = [sum(1 for s in data if s == k) for k in scores]
    pcts   = [c / len(data) * 100 if data else 0 for c in counts]
    offset = (i - n_groups / 2 + 0.5) * bar_w
    ax.bar(
        x + offset, pcts, bar_w * 0.92,
        label=label,
        color=COLORS[label], alpha=0.85, edgecolor="white", linewidth=0.8,
    )

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in scores])
ax.set_xlabel("Bias score")
ax.set_ylabel("% of Articles")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Larger legend for the poster (paper: 9 pt, upper right). Centred over scores
# 1-3, where every bar is below ~25%, so it can't cover the tall GT-HB bar at 4.
ax.legend(loc="upper center", fontsize=14, framealpha=0.9)

fig.suptitle("Corpus Bias Score Distribution", fontsize=11)

out = os.path.join(OUT_DIR, "corpus_distribution_poster.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
