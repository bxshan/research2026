"""
gen_keyword_freq.py
Supplementary: Wikipedia keyword frequency per community type (grouped mode).
Horizontal grouped bar chart showing keyword frequency (per 1{,}000 tokens)
for five community types: Elite Private, Selective/Magnet, Mainstream
Suburban, Title I Urban, Rural Public.

Adapted from plot_grouped() in
data/corpus_word_freq_analysis/wiki_keyword_freq.py
(original uses plt.xkcd() and fetches articles from the Wikipedia API at
run time before emitting a PNG; this version skips the fetch by reading
the cached, already-computed frequencies from
data/corpus_word_freq_analysis/wiki_keyword_freq_grouped.csv and emits
PDF in the seaborn-paper template used by the other paper/plots/ scripts).

Source:
  data/corpus_word_freq_analysis/wiki_keyword_freq_grouped.csv
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
Output: keyword_freq.pdf (saved next to this script)
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

FREQ_CSV = os.path.join(
    ROOT, "data", "corpus_word_freq_analysis",
    "wiki_keyword_freq_grouped.csv",
)

# Group display order and colors (warm to cool), matching the source script.
GROUPS = [
    ("Elite Private",       "#c0392b"),
    ("Selective / Magnet",  "#e67e22"),
    ("Mainstream Suburban", "#f1c40f"),
    ("Title I Urban",       "#2980b9"),
    ("Rural Public",        "#1abc9c"),
]
GROUP_COLORS = dict(GROUPS)
GROUP_ORDER  = [g for g, _ in GROUPS]

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


def load_grouped(path):
    """
    Return (keywords_in_csv_order, dict[group][keyword] -> (mean, std, n)).
    Keyword order is preserved from the first time each keyword is seen.
    """
    means = defaultdict(dict)
    keywords, seen = [], set()
    n_by_group = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            kw = r["keyword"]
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
            means[r["group"]][kw] = (
                float(r["mean_per_1k"]),
                float(r["std_per_1k"]),
            )
            n_by_group[r["group"]] = int(r["n_schools"])
    return keywords, means, n_by_group


keywords, means, n_by_group = load_grouped(FREQ_CSV)
groups = [g for g in GROUP_ORDER if g in means]

# Sort keywords ascending by grand mean across groups (largest signal at top)
def grand_mean(kw):
    vals = [means[g][kw][0] for g in groups if kw in means[g]]
    return sum(vals) / len(vals) if vals else 0.0

sorted_kws = sorted(keywords, key=grand_mean)

n_kws   = len(sorted_kws)
n_grps  = len(groups)
bar_h   = 0.75 / n_grps
y_base  = np.arange(n_kws, dtype=float)

fig, ax = plt.subplots(figsize=(7, max(4.0, n_kws * 0.4)))

for i, group in enumerate(groups):
    offset = (i - n_grps / 2 + 0.5) * bar_h
    vals   = [means[group][kw][0] for kw in sorted_kws]
    stds   = [means[group][kw][1] for kw in sorted_kws]
    ax.barh(
        y_base + offset, vals, height=bar_h * 0.92, xerr=stds,
        label=f"{group} (n={n_by_group[group]})",
        color=GROUP_COLORS[group], alpha=0.85,
        edgecolor="white", linewidth=0.4,
        error_kw={
            "elinewidth": 0.7,     # thin line
            "capsize":    4,       # vertical dashes at each end (length unchanged)
            "capthick":   0.7,     # caps matched to line thickness
            "ecolor":     "#1a1a1a",  # near-black
            "alpha":      1.0,
        },
    )

ax.set_yticks(y_base)
ax.set_yticklabels(sorted_kws)
ax.set_xlabel("Frequency per 1,000 tokens")
ax.axvline(0, color="#888888", linewidth=0.6, zorder=1)  # reference at x=0
ax.grid(axis="x", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(which="both", length=0)   # no tick dashes before row labels
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle(
    "Wikipedia Keyword Frequency by School Community Type\n"
    r"(error bars: $\pm 1$ SD across schools in group)",
    fontsize=11,
)

out = os.path.join(HERE, "keyword_freq.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
