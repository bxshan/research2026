"""
gen_pool_curve.py
GT-HB threshold choice, two panels.
Top: total corpus articles per 0.25-wide bin of per-source mean rubric score,
stacked by MBFC label group (where the corpus mass sits on the bias axis).
Bottom: cumulative article pool from sources with mean >= t as a function of
the threshold t, with the 500k identical-training-regime gate and the chosen
t = 3.0 marked (t = 3.5 falls below the gate).

Source:
  data/bias_analysis/bias_scores/bias_scores_gt.csv
  data/gt_hb/bias_scores_topup.csv
  data/gt_hb/source_metadata.csv
Aggregation mirrors data/gt_hb/finalize_sources.py exactly (no new statistics).
Style: seaborn-paper with serif font (matches the other paper/plots scripts).
Output: pool_curve.pdf (saved next to this script)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

THRESHOLD, MIN_GRADED, MIN_ARTICLES = 3.0, 10, 1000  # mirrors finalize_sources.py
GATE = 500_000

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True,
                               gridspec_kw={"hspace": 0.14})
fmt = FuncFormatter(lambda v, _: "0" if v == 0 else
                    (f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))

# ── top: article mass per score bin, stacked by MBFC group ──────────────────
edges = np.arange(0, 4.75, 0.25)
df["bin"] = pd.cut(df.mean_score, bins=edges, right=False)
mass = df.pivot_table(index="bin", columns="group", values="n_articles",
                      aggfunc="sum", observed=False).fillna(0)
centers = edges[:-1] + 0.125
bottom = np.zeros(len(mass))
for g in GROUP_COLOR:
    if g not in mass.columns:
        continue
    ax1.bar(centers, mass[g].values, width=0.23, bottom=bottom,
            color=GROUP_COLOR[g], label=g, edgecolor="white", linewidth=0.4)
    bottom += mass[g].values

ax1.yaxis.set_major_formatter(fmt)
ax1.set_ylabel("Corpus Articles in Bin")
ax1.legend(fontsize=7, loc="upper right", framealpha=0.9,
           title="MBFC label (external)", title_fontsize=6.5)
ax1.grid(axis="y", alpha=0.3, linewidth=0.6)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── bottom: cumulative pool vs threshold ─────────────────────────────────────
ts = np.arange(0, 4.55, 0.05)
pool = np.array([df.loc[df.mean_score >= t, "n_articles"].sum() for t in ts])
ax2.plot(ts, pool, color="black", lw=1.3, drawstyle="steps-post")
ax2.set_yscale("log")
ax2.yaxis.set_major_formatter(fmt)
ax2.axhline(GATE, color="#c0392b", ls=":", lw=1.1)
ax2.text(0.05, GATE * 1.15, "500k gate (identical training regime)",
         fontsize=7, color="#c0392b")
ax2.axvline(THRESHOLD, color="black", ls="--", lw=1)
ax2.text(THRESHOLD + 0.05, 0.05, f"chosen $t$ = {THRESHOLD}",
         transform=ax2.get_xaxis_transform(), fontsize=7)

for t in (2.0, 2.5, 3.0, 3.5):
    p = df.loc[df.mean_score >= t, "n_articles"].sum()
    c = "#c0392b" if p < GATE else "black"
    ax2.scatter([t], [p], s=16, color=c, zorder=3)
    lab = f"{p/1e6:.2f}M" if p >= 1e6 else f"{p/1e3:.0f}k"
    ax2.annotate(f"$t$={t}: {lab}", (t, p), xytext=(6, 5),
                 textcoords="offset points", fontsize=7, color=c)

ax2.set_xlim(-0.05, 4.7)
ax2.set_xlabel("Threshold $t$ on Per-Source Mean Rubric Score")
ax2.set_ylabel("Pool: Articles from Sources $\\geq t$")
ax2.grid(alpha=0.3, linewidth=0.6)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("GT-HB Threshold Choice: Corpus Mass vs. Source Bias", fontsize=11)

out = os.path.join(HERE, "pool_curve.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
