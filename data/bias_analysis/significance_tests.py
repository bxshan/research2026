"""
Significance tests for the condition-vs-Base output-bias shifts.

The 300 completions per condition are 15 prompts x 20 runs, so completions are
NOT independent: prompt identity dominates the score. Treating n=300 as
independent overstates significance. Every condition saw the same 15 prompts,
so we collapse to prompt-level means and test the 15 paired differences.

Primary:   one-sided Wilcoxon signed-rank on 15 paired prompt means, Holm-corrected.
Interval:  cluster bootstrap over prompts (same drawn prompts for both conditions,
           preserving the pairing), 10,000 draws, seed 2, percentile CI.
Secondary: one-sided Mann-Whitney U on the 300 raw completions (anticonservative;
           reported only to show the conclusion does not hinge on independence).

Scores are the RAW haiku_score, matching Table 5. Condition N's raw advantage is
partly hallucination-driven (see the precision correction); its corrected clean
mean (1.484) still exceeds Base (0.841), so the direction survives either way.

Reads the frozen manifest. Prints a summary; writes significance_tests.csv.
Non-destructive.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE     = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "second_judge_manifest.csv")
OUT      = os.path.join(HERE, "significance_tests.csv")

SEED  = 2
DRAWS = 10000
BASE  = "base"

NAME = {"base": "Base", "llama-sft-gt": "GT", "llama-sft-ps": "PS",
        "llama-sft-wiki": "N", "llama-sft-gthb": "GT-HB"}
# directional hypotheses: news corpora raise bias, PS lowers it
ALT = {"llama-sft-gt": "greater", "llama-sft-gthb": "greater",
       "llama-sft-wiki": "greater", "llama-sft-ps": "less"}
ORDER = ["llama-sft-gt", "llama-sft-ps", "llama-sft-wiki", "llama-sft-gthb"]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvals)
    idx = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


def cluster_bootstrap_ci(pivot, cond, prompts, rng, draws=DRAWS):
    """Resample prompts with replacement, same prompts for both conditions."""
    a = pivot.loc[prompts, cond].to_numpy()
    b = pivot.loc[prompts, BASE].to_numpy()
    n = len(prompts)
    diffs = np.empty(draws)
    for k in range(draws):
        pick = rng.integers(0, n, n)
        diffs[k] = a[pick].mean() - b[pick].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(diffs.mean())


def main():
    df = pd.read_csv(MANIFEST)

    # --- sanity: balanced 5 x 15 x 20 design -------------------------------
    cells = df.groupby(["condition", "prompt_id"]).size().unique()
    assert list(cells) == [20], f"unbalanced design: cell sizes {cells}"

    # --- collapse to prompt-level means (the unit of independence) ---------
    pivot = df.pivot_table(index="prompt_id", columns="condition",
                           values="haiku_score", aggfunc="mean")
    prompts = list(pivot.index)
    assert len(prompts) == 15, f"expected 15 prompts, got {len(prompts)}"

    rng = np.random.default_rng(SEED)
    rows, raw_p = [], []

    for cond in ORDER:
        alt = ALT[cond]
        a = pivot[cond].to_numpy()
        b = pivot[BASE].to_numpy()
        d = a - b
        n_zero = int((d == 0).sum())          # Wilcoxon silently drops these

        w, p_w = stats.wilcoxon(a, b, alternative=alt)

        # secondary: completion-level, ignores clustering
        sa = df.loc[df["condition"] == cond, "haiku_score"]
        sb = df.loc[df["condition"] == BASE, "haiku_score"]
        _, p_mw = stats.mannwhitneyu(sa, sb, alternative=alt)

        lo, hi, boot_mean = cluster_bootstrap_ci(pivot, cond, prompts, rng)

        rows.append({
            "condition": NAME[cond], "alternative": alt,
            "mean_cond": round(sa.mean(), 3), "mean_base": round(sb.mean(), 3),
            "mean_diff": round(float(d.mean()), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "boot_mean_diff": round(boot_mean, 3),
            "wilcoxon_W": float(w), "p_wilcoxon": p_w,
            "p_mannwhitney_naive": p_mw,
            "n_prompts": len(prompts), "n_zero_diff": n_zero,
            "n_completions": len(sa),
        })
        raw_p.append(p_w)

    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("p_wilcoxon") + 1,
               "p_wilcoxon_holm", holm(np.array(raw_p)))

    pd.set_option("display.width", 220)
    print("=== Prompt-level means (15 prompts x 20 runs per cell) ===")
    print(pivot.rename(columns=NAME).round(3).to_string())

    print("\n=== Primary: one-sided Wilcoxon signed-rank on 15 paired prompt "
          "means, Holm-corrected; CI = cluster bootstrap over prompts ===")
    show = ["condition", "alternative", "mean_cond", "mean_base", "mean_diff",
            "ci_lo", "ci_hi", "p_wilcoxon", "p_wilcoxon_holm",
            "p_mannwhitney_naive", "n_zero_diff"]
    print(out[show].to_string(index=False))

    # --- verification -----------------------------------------------------
    print("\n=== Verification ===")
    drift = (out["boot_mean_diff"] - out["mean_diff"]).abs().max()
    print(f"  max |bootstrap mean diff - direct mean diff| = {drift:.4f}  "
          f"(resampler unbiased if ~0)")
    print(f"  prompts with zero paired difference (dropped by Wilcoxon): "
          f"{out['n_zero_diff'].sum()}")
    print("  condition means vs Table 5:")
    for cond in [BASE] + ORDER:
        m = df.loc[df["condition"] == cond, "haiku_score"].mean()
        print(f"    {NAME[cond]:<6} {m:.3f}")

    out.to_csv(OUT, index=False)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
