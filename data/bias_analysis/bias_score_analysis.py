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
WIKI_CSV  = os.path.join(SCORES_DIR, "bias_scores_wiki.csv")
INFER_CSV = os.path.join(SCORES_DIR, "infer_results_20260512_140118.csv")

COLORS = {
    "GT": "#c0392b",
    "PS": "#2980b9",
    "Wiki": "#27ae60",
}

INFER_COLORS = {
    "base":           "#555555",
    "llama-sft-gt":   "#c0392b",
    "llama-sft-ps":   "#2980b9",
    "llama-sft-wiki": "#27ae60",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data() -> dict[str, pd.DataFrame]:
    gt = pd.read_csv(GT_CSV); gt["dataset"] = "GT"
    ps = pd.read_csv(PS_CSV); ps["dataset"] = "PS"
    dfs = {"GT": gt, "PS": ps}
    if os.path.exists(WIKI_CSV):
        wiki = pd.read_csv(WIKI_CSV); wiki["dataset"] = "Wiki"
        dfs["Wiki"] = wiki
    for df in dfs.values():
        df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")
        df.dropna(subset=["bias_score"], inplace=True)
    return dfs


# ── Plot 1: score distribution ────────────────────────────────────────────────
def plot_distribution(dfs: dict, out_path: str):
    scores  = [0, 1, 2, 3, 4, 5]
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
        ax.set_xticklabels(["0 — None", "1 — Trace", "2 — Subtle", "3 — Moderate", "4 — Strong", "5 — Extreme"])
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
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Mean bias score (0–5)")
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
            ax.set_xlim(0, 5)
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
            "pct_4":    round((s == 4).mean() * 100, 1),
            "pct_5":    round((s == 5).mean() * 100, 1),
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


# ── Inference completion analysis ─────────────────────────────────────────────
def load_infer_scores(path: str) -> pd.DataFrame:
    """Load scored inference completions; parse condition and prompt_id from article_id."""
    df = pd.read_csv(path)
    df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")
    df = df.dropna(subset=["bias_score"])
    # new format has condition/prompt_id as direct columns; old format used source/article_id
    if "condition" not in df.columns:
        df["condition"] = df["source"]
    if "prompt_id" not in df.columns:
        df["prompt_id"] = df["article_id"].str.split("_").str[1]
    return df


def plot_infer_distribution(df: pd.DataFrame, out_path: str):
    """Grouped bar: % at each score level per condition."""
    conditions = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki"]
    scores     = [0, 1, 2, 3, 4, 5]
    bar_w      = 0.22
    x          = np.arange(len(scores))

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cond in enumerate(conditions):
            sub    = df[df["condition"] == cond]
            pcts   = [(sub["bias_score"] == s).mean() * 100 for s in scores]
            label  = f"{cond} (n={len(sub)})"
            offset = (i - 1) * bar_w
            ax.bar(x + offset, pcts, bar_w * 0.9,
                   label=label, color=INFER_COLORS[cond], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(["0 — None", "1 — Trace", "2 — Subtle", "3 — Moderate", "4 — Strong", "5 — Extreme"])
        ax.set_ylabel("% of completions")
        ax.set_title("Bias Score Distribution by Model Condition\n(LLM-as-judge, n=180)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


def plot_infer_by_prompt(df: pd.DataFrame, out_path: str):
    """Grouped bar: mean bias score per prompt × condition."""
    conditions = ["base", "llama-sft-gt", "llama-sft-ps"]
    prompts    = sorted(df["prompt_id"].dropna().unique())
    bar_w      = 0.22
    x          = np.arange(len(prompts))

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, cond in enumerate(conditions):
            sub    = df[df["condition"] == cond]
            means  = [sub[sub["prompt_id"] == p]["bias_score"].mean() for p in prompts]
            offset = (i - 1) * bar_w
            bars   = ax.bar(x + offset, means, bar_w * 0.9,
                            label=cond, color=INFER_COLORS[cond], alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in prompts])
        ax.set_ylabel("Mean bias score (0–5)")
        ax.set_ylim(0, 5.2)
        ax.set_title("Mean Bias Score by Prompt Topic and Model Condition")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


def save_infer_summary(df: pd.DataFrame, out_path: str):
    """Print and save per-condition summary stats, net bias metrics, and Mann-Whitney tests.

    Net bias (from design doc):
        Net_GT = [Bias(GT) - Bias(B)] - [Bias(N) - Bias(B)]
        Net_PS = [Bias(PS) - Bias(B)] - [Bias(N) - Bias(B)]
    where B=base, N=llama-sft-wiki (neutral). When N is unavailable, reports
    the raw difference Bias(X) - Bias(B) as a partial metric.
    """
    present    = set(df["condition"].unique())
    conditions = [c for c in ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki"]
                  if c in present]

    means = {}
    rows  = []
    for cond in conditions:
        s = df[df["condition"] == cond]["bias_score"]
        means[cond] = s.mean()
        rows.append({
            "condition": cond,
            "n":         len(s),
            "mean":      round(s.mean(), 3),
            "median":    round(s.median(), 3),
            "std":       round(s.std(), 3),
            "pct_0":     round((s == 0).mean() * 100, 1),
            "pct_1":     round((s == 1).mean() * 100, 1),
            "pct_2":     round((s == 2).mean() * 100, 1),
            "pct_3":     round((s == 3).mean() * 100, 1),
            "pct_4":     round((s == 4).mean() * 100, 1),
            "pct_5":     round((s == 5).mean() * 100, 1),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[table] saved → {out_path}")
    print(summary.to_string(index=False))

    # ── Net bias metrics ───────────────────────────────────────────────────────
    print("\n[net bias]")
    b = means.get("base")
    n = means.get("llama-sft-wiki")   # condition N; None if not yet trained

    for tag, cond in [("GT", "llama-sft-gt"), ("PS", "llama-sft-ps")]:
        x = means.get(cond)
        if x is None:
            print(f"  Net_{tag}: condition {cond} not present")
            continue
        raw_diff = x - b
        if n is not None:
            n_diff  = n - b
            net     = raw_diff - n_diff
            print(f"  Net_{tag} = [Bias({tag})−Bias(B)] − [Bias(N)−Bias(B)]"
                  f" = [{x:.3f}−{b:.3f}] − [{n:.3f}−{b:.3f}] = {net:+.3f}")
        else:
            print(f"  Net_{tag} = Bias({tag}) − Bias(B) = {x:.3f} − {b:.3f} = {raw_diff:+.3f}"
                  f"  (N/llama-sft-wiki not yet available; full net metric pending)")

    # ── Mann-Whitney pairwise ──────────────────────────────────────────────────
    print("\n[stats] Mann-Whitney U pairwise (two-sided):")
    for i, a in enumerate(conditions):
        for b_cond in conditions[i+1:]:
            sa = df[df["condition"] == a]["bias_score"]
            sb = df[df["condition"] == b_cond]["bias_score"]
            u, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {a} vs {b_cond}: U={u:.0f}  p={p:.4f}  {sig}")


# ── Variance analysis: per-prompt × condition ─────────────────────────────────
def save_variance_analysis(df: pd.DataFrame, out_path: str):
    """
    For each prompt × condition, compute mean, std, and bias rate (pct scoring >=1).
    Identifies which prompts show the largest GT–base differentiation.
    Saves CSV and prints a summary table.
    """
    conditions = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki"]
    conditions = [c for c in conditions if c in df["condition"].unique()]
    prompts    = sorted(df["prompt_id"].unique())

    rows = []
    for prompt in prompts:
        for cond in conditions:
            s = df[(df["prompt_id"] == prompt) & (df["condition"] == cond)]["bias_score"]
            bias_rate = (s >= 1).mean() * 100
            m, sd = s.mean(), s.std()
            uniformity = round(m / (m + sd), 3) if (m + sd) > 0 else None  # 1=uniform shift, 0=outlier-driven
            rows.append({
                "prompt_id":   prompt,
                "condition":   cond,
                "n_runs":      len(s),
                "mean":        round(m, 3),
                "std":         round(sd, 3),
                "bias_rate":   round(bias_rate, 1),
                "uniformity":  uniformity,
            })

    var_df = pd.DataFrame(rows)
    var_df.to_csv(out_path, index=False)
    print(f"[variance] saved → {out_path}")

    # Print table grouped by prompt
    print()
    for prompt in prompts:
        sub = var_df[var_df["prompt_id"] == prompt]
        print(f"  Prompt: {prompt}")
        print(f"  {'condition':<20} {'mean':>6} {'std':>6} {'bias_rate':>10} {'uniformity':>11}")
        print(f"  {'-'*60}")
        for _, row in sub.iterrows():
            u = f"{row['uniformity']:.3f}" if pd.notna(row['uniformity']) else "—"
            print(f"  {row['condition']:<20} {row['mean']:>6.3f} {row['std']:>6.3f} {row['bias_rate']:>9.1f}% {u:>11}")
        print()

    # GT–base delta per prompt
    print("  GT–base mean delta per prompt:")
    for prompt in prompts:
        sub  = var_df[var_df["prompt_id"] == prompt].set_index("condition")
        if "base" in sub.index and "llama-sft-gt" in sub.index:
            delta = sub.loc["llama-sft-gt", "mean"] - sub.loc["base", "mean"]
            delta_rate = sub.loc["llama-sft-gt", "bias_rate"] - sub.loc["base", "bias_rate"]
            print(f"    {prompt:<15} Δmean={delta:+.3f}  Δbias_rate={delta_rate:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bias_score_analysis_out")
    parser.add_argument("--corpus_out", default=os.path.join(_base, "corpus_analysis_out"))
    parser.add_argument("--completions_out", default=os.path.join(_base, "completions_analysis_out"))
    parser.add_argument("--infer_csv", default=INFER_CSV,
                        help="Path to scored inference completions CSV")
    parser.add_argument("--skip_corpus", action="store_true",
                        help="Skip GT/PS corpus analysis")
    args = parser.parse_args()

    os.makedirs(args.corpus_out, exist_ok=True)
    os.makedirs(args.completions_out, exist_ok=True)

    # ── Corpus analysis (GT / PS) ──────────────────────────────────────────────
    if not args.skip_corpus:
        try:
            dfs = load_data()
            print(f"[data]  loaded GT={len(dfs['GT'])}  PS={len(dfs['PS'])}")
            plot_distribution(dfs, os.path.join(args.corpus_out, "bias_analysis_distribution.png"))
            plot_means(       dfs, os.path.join(args.corpus_out, "bias_analysis_means.png"))
            plot_sources(     dfs, os.path.join(args.corpus_out, "bias_analysis_sources.png"))
            save_summary(     dfs, os.path.join(args.corpus_out, "bias_analysis_summary.csv"))
        except FileNotFoundError as e:
            print(f"[warn]  corpus data missing ({e}), skipping corpus plots")

    # ── Inference completion analysis ──────────────────────────────────────────
    if args.infer_csv and os.path.exists(args.infer_csv):
        print(f"\n[infer] loading {args.infer_csv} ...")
        idf = load_infer_scores(args.infer_csv)
        print(f"[infer] {len(idf)} completions  conditions={sorted(idf['condition'].unique())}")
        plot_infer_distribution(idf, os.path.join(args.completions_out, "infer_bias_distribution.png"))
        plot_infer_by_prompt(   idf, os.path.join(args.completions_out, "infer_bias_by_prompt.png"))
        save_infer_summary(     idf, os.path.join(args.completions_out, "infer_bias_summary.csv"))
        save_variance_analysis(idf, os.path.join(args.completions_out, "variance_analysis.csv"))
    else:
        print(f"[warn]  infer CSV not found: {args.infer_csv}")


if __name__ == "__main__":
    main()
