"""
gen_corpus_distribution.py
Supplementary: bias score distribution by training corpus (GT, PS, Wikipedia).
Grouped bar chart of percent of articles at each rubric level (0-5) per corpus.

Adapted from plot_distribution() in
data/bias_analysis/bias_score_analysis.py
(original uses plt.xkcd() and emits PNG; this version drops the xkcd
context and emits PDF in the seaborn-paper template used by the other
paper.plots/ scripts).

Source:
  data/bias_analysis/bias_scores/bias_scores_gt.csv
  data/bias_analysis/bias_scores/bias_scores_ps.csv
  data/bias_analysis/bias_scores/bias_scores_wiki.csv
  data/gt_hb/bias_scores_gthb.csv
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
Output: corpus_distribution.pdf (saved next to this script)
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

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
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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
        label=f"{label} (n={len(data):,})",
        color=COLORS[label], alpha=0.85, edgecolor="white", linewidth=0.8,
    )

ax.set_xticks(x)
ax.set_xticklabels([
    "0 -- None", "1 -- Trace", "2 -- Subtle",
    "3 -- Moderate", "4 -- Strong", "5 -- Extreme",
])
ax.set_ylabel("% of Articles")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

fig.suptitle("Corpus Bias Score Distribution", fontsize=11)

out = os.path.join(HERE, "corpus_distribution.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
