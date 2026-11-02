"""
gen_weat_effects_poster.py
Poster-only copy of paper/plots/scripts/gen_weat_effects.py. Same data and
the same effect sizes and significance markers; only labels are shortened so
the chart stays readable in one poster column:
  - the per-test sign notes are one line ("+d elite, -d under-resourced")
    instead of two lines that repeat the attribute word;
  - the legend title keeps only the significance thresholds (the poster
    caption already names the one-sided permutation test).
Output: paper/poster/weat_effects_poster.pdf

Original description follows.
WEAT effect sizes by condition and test (Figure 4).
Grouped horizontal bar chart of Cohen's d per (test, condition), with
significance markers (* p<0.05, ** p<0.01, *** p<0.001) using the
one-sided p-values reported by the permutation test in
weat/results/weat_*.csv. This matches the convention used in the paper's
Methods (Section: WEAT) and Results table.
Source: weat/results/weat_20260528_133420.csv
        weat/results/weat_20260611_143416.csv (GT-HB)
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
Output: weat_effects.pdf (saved next to this script)
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")   # scripts/ -> poster/ -> paper/ -> repo root
PLOTS = os.path.join(HERE, "..")              # poster PDFs live in paper/poster/

WEAT_CSVS = [
    os.path.join(ROOT, "weat", "results", "weat_20260528_133420.csv"),
    os.path.join(ROOT, "weat", "results", "weat_20260611_143416.csv"),
]

# Canonical condition order and display labels
COND_ORDER   = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki",
                "llama-sft-gthb"]
COND_DISPLAY = {"base": "Base", "llama-sft-gt": "GT",
                "llama-sft-ps": "PS", "llama-sft-wiki": "Wiki",
                "llama-sft-gthb": "GT-HB"}
COLORS       = {"Base": "#555555", "GT": "#c0392b", "PS": "#2980b9",
                "Wiki": "#27ae60", "GT-HB": "#7b241c"}
# Redundant (non-colour) channel so the five conditions stay distinguishable
# when the figure is printed or photocopied in greyscale.
HATCHES      = {"Base": "", "GT": "////", "PS": "....",
                "Wiki": "||||", "GT-HB": "xxxx"}

# One-line summary per test: which target group the sign of d favours.
# +d: the first group (X) is more associated with the test's attribute
# (positive / necessary / credible, implied by the test name); -d: the second.
TEST_LEGEND = {
    "High School Selectivity": r"$+d$ elite, $-d$ under-resourced",
    "Policy Necessity":        r"$+d$ government, $-d$ market",
    "Immigration Framing":     r"$+d$ humanitarian, $-d$ threat",
    "Economic Policy Framing": r"$+d$ progressive, $-d$ market",
    "Media Trust":             r"$+d$ institutional, $-d$ skeptical",
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
    "axes.labelsize":  16,   # larger axis title for the poster (paper: 10)
    "xtick.labelsize": 15,   # larger Cohen's d numbers for the poster (paper: 9)
    "ytick.labelsize": 14,   # test names on the y-axis (paper: 9)
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "hatch.linewidth": 0.5,
    "figure.dpi":      150,
})


def sig_marker(p):
    """Return significance marker for the one-sided p reported in the CSV.

    Matches the paper convention: p is the proportion of random equal-partition
    splits whose test statistic exceeds the observed value, so only positive
    effects with small p are flagged. Tests with negative d will have p near 1
    and receive no marker, mirroring the paper's "null" treatment of Tests 2
    and 4.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_weat(paths):
    """Return dict: {test_name: {condition: (d, p), ...}}, preserving test order."""
    results, order = {}, []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                test = r["test_name"]
                if test not in results:
                    results[test] = {}
                    order.append(test)
                results[test][r["condition"]] = (float(r["effect_size"]),
                                                  float(r["p_value"]))
    return order, results


tests, results = load_weat(WEAT_CSVS)
n_tests = len(tests)
n_conds = len(COND_ORDER)

fig, ax = plt.subplots(figsize=(7, 5.6))   # taller than the paper's (7, 5.5) to fill the poster column

# Bar geometry: each test gets a y-band of width 1, conditions stacked within
bar_h    = 0.18
y_base   = np.arange(n_tests)
offsets  = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_h

# Plot one condition at a time so the legend reads naturally
for ci, raw_cond in enumerate(COND_ORDER):
    label = COND_DISPLAY[raw_cond]
    ds, ps = [], []
    for test in tests:
        d, p = results[test].get(raw_cond, (0.0, 1.0))
        ds.append(d)
        ps.append(p)

    y = y_base + offsets[ci]
    ax.barh(y, ds, height=bar_h * 0.92,
            color=COLORS[label], alpha=0.85,
            edgecolor="white", linewidth=0.8,
            hatch=HATCHES[label],
            label=label)

    # Significance markers placed just past the bar tip
    for yi, d, p in zip(y, ds, ps):
        m = sig_marker(p)
        if not m:
            continue
        pad = 0.05 if d >= 0 else -0.05
        ha  = "left" if d >= 0 else "right"
        ax.text(d + pad, yi, m, ha=ha, va="center", fontsize=8)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_base)
ax.set_yticklabels(tests)
ax.invert_yaxis()                # first test at top

# Description of X (target) and A (attribute) below each test name.
# Placed in the y-tick margin (left of the y-axis) so the caption sits with
# its test rather than inside the data area. transform=get_yaxis_transform
# lets us use axes-fraction for x and data coords for y.
for yi, test in zip(y_base, tests):
    desc = TEST_LEGEND.get(test, "")
    if desc:
        ax.text(
            -0.015,                    # x: just left of the y-axis (axes coords)
            yi + 0.18,                 # y: top of caption block, just below the tick
            desc,
            transform=ax.get_yaxis_transform(),
            ha="right", va="top",
            fontsize=11, color="#555555", style="italic",   # paper: 7
            linespacing=1.3,
        )
ax.set_xlabel(r"Cohen's $d$")
ax.set_xlim(-2.0, 2.0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.grid(axis="x", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# One-row legend: the five conditions, then the significance key as a sixth
# entry whose handle is an empty (invisible) patch.
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor="none", edgecolor="none"))
labels.append(r"$^*\,p<0.05 \quad ^{**}\,p<0.01 \quad ^{***}\,p<0.001$")
ax.legend(
    handles, labels,
    # Centred left of the axes' middle so the row sits under the whole figure
    # (test-name margin included) instead of overhanging the plot's right edge.
    bbox_to_anchor=(0.36, -0.13),   # y clears the larger x-axis title
    loc="upper center",
    ncol=len(handles),
    fontsize=9,
    framealpha=0.9,
    columnspacing=0.9,
    handlelength=1.6,
)

# Largest text in the chart (axis title 16, ticks 15, test names 14) so the
# title holds the space above the plot on the poster (paper: 11).
fig.suptitle("WEAT Effect Sizes by Condition", fontsize=18)

out = os.path.join(PLOTS, "weat_effects_poster.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
