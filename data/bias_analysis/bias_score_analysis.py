"""
bias_score_analysis.py
----------------------
Analyzes LLM-judged bias scores across GT, PS, and Wikipedia corpora.

Produces:
  1. Score distribution per corpus (grouped bar)
  2. Mean ± SD per corpus (bar with error bars)
  3. Top/bottom sources by mean score (GT and PS)
  4. Summary statistics table (CSV)

Usage:
    python3 data/bias_score_analysis.py
    python3 data/bias_score_analysis.py --out_dir data/bias_score_analysis_out/
"""

import logging
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

SEED = 2
np.random.seed(SEED)

HERE     = os.path.dirname(os.path.abspath(__file__))
SCORES_DIR = os.path.join(HERE, "bias_scores")
GT_CSV   = os.path.join(SCORES_DIR, "bias_scores_gt.csv")
PS_CSV   = os.path.join(SCORES_DIR, "bias_scores_ps.csv")
WIKI_CSV = os.path.join(SCORES_DIR, "bias_scores_wiki.csv")

COLORS = {
    "GT":   "#c0392b",
    "PS":   "#2980b9",
    "Wiki": "#27ae60",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data() -> dict[str, pd.DataFrame]:
    gt   = pd.read_csv(GT_CSV);   gt["dataset"]   = "GT"
    ps   = pd.read_csv(PS_CSV);   ps["dataset"]   = "PS"
    wiki = pd.read_csv(WIKI_CSV); wiki["dataset"] = "Wiki"
    for df in [gt, ps, wiki]:
        df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")
        df = df.dropna(subset=["bias_score"])
    return {"GT": gt, "PS": ps, "Wiki": wiki}


# ── Plot 1: score distribution ────────────────────────────────────────────────
def plot_distribution(dfs: dict, out_path: str):
    scores  = [0, 1, 2, 3]
    labels  = list(dfs.keys())
    n       = len(labels)
    bar_w   = 0.25
    x       = np.arange(len(scores))

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (label, df) in enumerate(dfs.items()):
            counts = [len(df[df["bias_score"] == s]) for s in scores]
            pcts   = [c / len(df) * 100 for c in counts]
            ax.bar(x + (i - n/2 + 0.5) * bar_w, pcts, bar_w * 0.88,
                   label=f"{label} (n={len(df)})",
                   color=COLORS[label], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(["0 — None", "1 — Subtle", "2 — Moderate", "3 — Strong"])
        ax.set_ylabel("% of articles")
        ax.set_title("Bias Score Distribution by Corpus")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


# ── Plot 2: mean ± SD per corpus ──────────────────────────────────────────────
def plot_means(dfs: dict, out_path: str):
    labels = list(dfs.keys())
    means  = [dfs[l]["bias_score"].mean() for l in labels]
    stds   = [dfs[l]["bias_score"].std()  for l in labels]
    colors = [COLORS[l] for l in labels]

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                      edgecolor="white", capsize=6,
                      error_kw={"elinewidth": 1.2, "ecolor": "black"})
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.05,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 3.5)
        ax.set_ylabel("Mean bias score (0–3)")
        ax.set_title("Mean Bias Score per Corpus\n(error bars = ±1 SD)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


# ── Plot 3: top/bottom sources ────────────────────────────────────────────────
def plot_sources(dfs: dict, out_path: str, top_n: int = 10):
    with plt.xkcd():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (label, df) in zip(axes, {k: v for k, v in dfs.items()
                                           if k in ("GT", "PS")}.items()):
            src_means = (df.groupby("source")["bias_score"]
                         .agg(["mean", "count"])
                         .query("count >= 2")
                         .sort_values("mean", ascending=True))
            n_show  = min(top_n, len(src_means))
            half    = n_show // 2
            to_show = pd.concat([src_means.head(half), src_means.tail(half)]).drop_duplicates()
            colors_bar = [COLORS[label] if m >= src_means["mean"].median()
                          else "#aaaaaa" for m in to_show["mean"]]
            ax.barh(range(len(to_show)), to_show["mean"], color=colors_bar, alpha=0.85)
            ax.set_yticks(range(len(to_show)))
            ax.set_yticklabels([f"{s} (n={int(r['count'])})"
                                for s, r in to_show.iterrows()], fontsize=8)
            ax.set_xlabel("Mean bias score")
            ax.set_title(f"{label}: Sources by Mean Bias Score")
            ax.set_xlim(0, 3)
            ax.axvline(src_means["mean"].mean(), color="black",
                       linewidth=0.8, alpha=0.6, label="corpus mean")
            ax.legend(fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def save_summary(dfs: dict, out_path: str):
    rows = []
    datasets = list(dfs.keys())

    for label, df in dfs.items():
        s = df["bias_score"]
        rows.append({
            "dataset":  label,
            "n":        len(df),
            "mean":     round(s.mean(), 3),
            "median":   round(s.median(), 3),
            "std":      round(s.std(), 3),
            "pct_0":    round((s == 0).mean() * 100, 1),
            "pct_1":    round((s == 1).mean() * 100, 1),
            "pct_2":    round((s == 2).mean() * 100, 1),
            "pct_3":    round((s == 3).mean() * 100, 1),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[table] saved → {out_path}")
    print(summary.to_string(index=False))

    # Pairwise Mann-Whitney U tests
    print("\n[stats] Mann-Whitney U pairwise (p-values):")
    for i, a in enumerate(datasets):
        for b in datasets[i+1:]:
            u, p = stats.mannwhitneyu(
                dfs[a]["bias_score"], dfs[b]["bias_score"], alternative="two-sided"
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {a} vs {b}: U={u:.0f}  p={p:.4f}  {sig}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bias_score_analysis_out"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dfs = load_data()
    print(f"[data]  loaded GT={len(dfs['GT'])}  PS={len(dfs['PS'])}  Wiki={len(dfs['Wiki'])}")

    plot_distribution(dfs, os.path.join(args.out_dir, "bias_analysis_distribution.png"))
    plot_means(       dfs, os.path.join(args.out_dir, "bias_analysis_means.png"))
    plot_sources(     dfs, os.path.join(args.out_dir, "bias_analysis_sources.png"))
    save_summary(     dfs, os.path.join(args.out_dir, "bias_analysis_summary.csv"))


if __name__ == "__main__":
    main()
