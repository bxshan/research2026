"""
gen_source_map.py
GT-HB source selection map: one point per audited source -- per-source mean
rubric score (x) vs corpus article count (y, log scale). Filled marker =
selected for the GT-HB whitelist (mean >= 3.0), open marker = excluded;
color = MBFC label group. Vertical line = selection threshold.

Source:
  data/bias_analysis/bias_scores/bias_scores_gt.csv
  data/gt_hb/bias_scores_topup.csv
  data/gt_hb/source_metadata.csv
Aggregation mirrors data/gt_hb/finalize_sources.py exactly (no new statistics).
Style: seaborn-paper with serif font (matches the other paper/plots scripts).
Output: source_map.pdf (saved next to this script)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def load_per_source():
    """Per-source mean/count of judge scores + corpus size + MBFC label."""
    gt = pd.read_csv(os.path.join(ROOT, "data", "bias_analysis", "bias_scores", "bias_scores_gt.csv"))
    tu = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "bias_scores_topup.csv"))
    s = pd.concat([gt[["source", "bias_score"]], tu[["source", "bias_score"]]])
    s = s[s.bias_score >= 0]                       # drop parse errors (-1)
    agg = (s.groupby("source").bias_score
            .agg(mean_score="mean", n_graded="count").reset_index())
    meta = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "source_metadata.csv"))
    df = agg.merge(meta[["source", "n_articles", "label"]], on="source", how="left")
    df = df[(df.n_graded >= MIN_GRADED) & (df.n_articles >= MIN_ARTICLES)].copy()
    df["selected"] = df.mean_score >= THRESHOLD
    df["group"] = df.label.map(GROUP_OF)
    return df.reset_index(drop=True)


df = load_per_source()

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.set_yscale("log")
ax.set_xlim(-0.15, 4.75)
ax.axvspan(THRESHOLD, 4.75, color="#fdecea", zorder=0)
ax.axvline(THRESHOLD, color="black", lw=1, ls="--", zorder=1)

for g, c in GROUP_COLOR.items():
    sel = df[(df.group == g) & df.selected]
    exc = df[(df.group == g) & ~df.selected]
    ax.scatter(sel.mean_score, sel.n_articles, s=26, color=c, lw=0,
               alpha=0.9, zorder=3)
    ax.scatter(exc.mean_score, exc.n_articles, s=26, facecolors="none",
               edgecolors=c, lw=0.8, alpha=0.9, zorder=3)

LABELS = {  # source -> (dx pts, dy pts); negative dx = label to the left
    "thesun": (5, -2), "bbc": (5, -2), "foxnews": (5, -2), "msnbc": (5, -2),
    "sputnik": (5, -2), "westernjournal": (5, -2), "breitbart": (5, -2),
    "thegatewaypundit": (5, -2), "infowars": (5, -2),
    "thespoof": (-4, -2), "dailystormer": (-4, -2),
}
for name, (dx, dy) in LABELS.items():
    r = df[df.source == name]
    if r.empty:
        continue
    r = r.iloc[0]
    ax.annotate(name, (r.mean_score, r.n_articles), xytext=(dx, dy),
                textcoords="offset points", fontsize=6,
                ha="right" if dx < 0 else "left")

n_sel = int(df.selected.sum())
pool = int(df.loc[df.selected, "n_articles"].sum())
ax.text(0.0, 1.01,
        f"selected: {n_sel} / {len(df)} sources   pool: {pool:,} articles",
        transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
ax.text(THRESHOLD + 0.05, 0.96, f"selected: mean $\\geq$ {THRESHOLD}",
        transform=ax.get_xaxis_transform(), fontsize=7, va="top")

handles = [Line2D([], [], marker="o", ls="", color=c, label=g)
           for g, c in GROUP_COLOR.items()]
handles += [
    Line2D([], [], marker="o", ls="", color="black", label="selected (filled)"),
    # seaborn-paper sets lines.markeredgewidth=0 -> open marker would vanish
    Line2D([], [], marker="o", ls="", markerfacecolor="none",
           markeredgecolor="black", markeredgewidth=0.8, color="black",
           label="excluded (open)"),
]
ax.legend(handles=handles, fontsize=6.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.9)
ax.text(0.5, -0.27, "color groups are MBFC (Media Bias/Fact Check) labels, "
        "not produced by this work",
        transform=ax.transAxes, fontsize=5.5, color="#555555",
        ha="center", va="top", style="italic")

ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Per-Source Mean Rubric Bias Score (0--5)")
ax.set_ylabel("Corpus Articles per Source")
fig.suptitle("GT-HB Source Selection Map", fontsize=11)

out = os.path.join(HERE, "source_map.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
