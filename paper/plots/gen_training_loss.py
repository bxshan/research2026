"""
gen_training_loss.py
Training loss curves for GT, PS, and Wiki conditions.
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
X-axis: linear 0-5000 (left panel), log scale 5000-15625 (right panel).
Output: training_loss.pdf (saved next to this script)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

ADAPTERS = {
    "GT":   "model/adapters/llama-sft-gt_20260419_091436/checkpoint-5000/trainer_state.json",
    "PS":   "model/adapters/llama-sft-ps_20260420_033833/checkpoint-5000/trainer_state.json",
    "Wiki": "model/adapters/llama-sft-wiki_20260512_082331/checkpoint-15625/trainer_state.json",
}

COLORS = {"GT": "#c0392b", "PS": "#2980b9", "Wiki": "#27ae60"}

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


def load_loss(path):
    with open(path) as f:
        state = json.load(f)
    return [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]


def downsample(points, every=50):
    return points[::every]


def add_break_marks(fig, ax_left, ax_right, mark_w=6, mark_h=10, lw=1.2):
    """Draw parallel break marks at the join in display (pixel) coordinates."""
    # mark_w / mark_h are half-widths in points; angle is consistent regardless of axis size
    pt_to_disp = fig.dpi / 72.0
    dx = mark_w * pt_to_disp   # pixels
    dy = mark_h * pt_to_disp   # pixels

    for ax, x_ax in [(ax_left, 1.0), (ax_right, 0.0)]:
        for y_ax in [0.0, 1.0]:
            # Corner position in display coords
            cx, cy = ax.transAxes.transform([x_ax, y_ax])
            # Convert to figure-fraction for Line2D
            fx0, fy0 = fig.transFigure.inverted().transform([cx - dx, cy - dy])
            fx1, fy1 = fig.transFigure.inverted().transform([cx + dx, cy + dy])
            line = plt.Line2D(
                [fx0, fx1], [fy0, fy1],
                transform=fig.transFigure,
                color="black", linewidth=lw, clip_on=False, zorder=10,
            )
            fig.add_artist(line)


fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True,
    figsize=(7, 3.2),
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
)

for label, rel_path in ADAPTERS.items():
    pts = load_loss(os.path.join(ROOT, rel_path))
    pts = downsample(pts, every=25)
    steps, losses = zip(*pts)

    mask1 = [s <= 5000 for s in steps]
    s1 = [s for s, m in zip(steps, mask1) if m]
    l1 = [l for l, m in zip(losses, mask1) if m]
    if s1:
        display = "N" if label == "Wiki" else label
        ax1.plot(s1, l1, color=COLORS[label], label=f"Condition {display}")

    mask2 = [s >= 5000 for s in steps]
    s2 = [s for s, m in zip(steps, mask2) if m]
    l2 = [l for l, m in zip(losses, mask2) if m]
    if s2:
        ax2.plot(s2, l2, color=COLORS[label])

# left panel — linear x
ax1.set_xlim(0, 5000)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss")
ax1.legend_ = None  # remove from ax1
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1000))
ax1.axvline(x=5000, color="gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.spines["right"].set_visible(False)

# right panel — log x
ax2.set_xscale("log")
ax2.set_xlim(5000, 16500)
ax2.set_xlabel("Step (log scale)")
ax2.xaxis.set_major_locator(ticker.FixedLocator([10000, 15625]))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter(["10k", "15.6k"]))
ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
ax2.xaxis.get_offset_text().set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(left=False)

# legend on ax2 (right of the dashed line)
handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)

fig.suptitle("Training Loss by Condition", fontsize=11)

# Draw break marks after layout is finalized
fig.canvas.draw()
add_break_marks(fig, ax1, ax2, mark_w=4, mark_h=6)

out = os.path.join(HERE, "training_loss.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
