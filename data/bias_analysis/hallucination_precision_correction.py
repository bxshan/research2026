"""
Precision-corrected hallucination rates and clean means.

The binary hallucinate flag was validated in two 60-item blind studies (seed 2):
GT + GT-HB (100% recall; 30% precision overall, GT 20%, GT-HB 40%) and Condition N
(recall 86.2%, precision 83.3%). On GT/GT-HB most flags are false positives, but on
N the flag is reliable. The paper's mu_clean drops ALL flagged completions, which
over-cleans conditions whose flag precision is below 1 and understates the clean mean.

Correction:
  estimated true hallucination rate = raw_rate * precision
  corrected clean mean = keep every unflagged completion PLUS the false-positive
    fraction of flagged ones, by randomly excluding only round(precision * n_flag)
    flagged completions per draw, averaged over K draws (seed=2).

Precision is now measured for N (0.833); PS/Base remain unmeasured. N drives the
GT-HB > N reversal, so we also sweep N's precision to locate the flip threshold.

Reads the frozen manifest (haiku_score / haiku_hallucinate per completion).
Prints a summary; writes hallucination_precision_correction.csv. Non-destructive.
"""

import os
import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "second_judge_manifest.csv")
OUT      = os.path.join(HERE, "hallucination_precision_correction.csv")

SEED   = 2
DRAWS  = 2000
# Measured precision from the 60-item blind studies (seed 2); PS/Base unmeasured.
PRECISION = {
    "base":           0.30,
    "llama-sft-gt":   0.20,
    "llama-sft-ps":   0.30,
    "llama-sft-wiki": 0.833,  # N — MEASURED (60-item N study): P=0.833, R=0.862
    "llama-sft-gthb": 0.40,
}
NAME = {"base": "Base", "llama-sft-gt": "GT", "llama-sft-ps": "PS",
        "llama-sft-wiki": "N", "llama-sft-gthb": "GT-HB"}
ORDER = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki", "llama-sft-gthb"]


def corrected_clean_mean(scores, flagged, precision, rng, draws=DRAWS):
    """Mean over: all unflagged + a random (1-precision) fraction of flagged,
    averaged over `draws` random draws. Also returns the closed-form expectation."""
    scores = np.asarray(scores, float)
    flagged = np.asarray(flagged, bool)
    unflag = scores[~flagged]
    flag   = scores[flagged]
    n_flag = len(flag)
    n_drop = int(round(precision * n_flag))

    if n_flag == 0:
        m = unflag.mean()
        return m, m, 0
    means = []
    for _ in range(draws):
        keep_flag = flag.copy()
        drop_idx = rng.choice(n_flag, size=n_drop, replace=False)
        keep_mask = np.ones(n_flag, bool); keep_mask[drop_idx] = False
        kept = np.concatenate([unflag, flag[keep_mask]])
        means.append(kept.mean())
    mc = float(np.mean(means))
    # closed form: keep all unflag + (n_flag - n_drop) flagged at flagged-mean
    kept_flag_n = n_flag - n_drop
    cf = (unflag.sum() + kept_flag_n * flag.mean()) / (len(unflag) + kept_flag_n)
    return mc, float(cf), n_drop


def main():
    df = pd.read_csv(MANIFEST)
    df["flag"] = df["haiku_hallucinate"].astype(bool)
    rng = np.random.default_rng(SEED)

    rows = []
    for cond in ORDER:
        g = df[df["condition"] == cond]
        n_total = len(g)
        n_flag  = int(g["flag"].sum())
        raw_rate = 100 * n_flag / n_total
        P = PRECISION[cond]
        corr_rate = raw_rate * P
        mean_all   = g["haiku_score"].mean()
        mean_unflag = g.loc[~g["flag"], "haiku_score"].mean()   # = paper's mu_clean (drop-all)
        mc, cf, n_drop = corrected_clean_mean(g["haiku_score"], g["flag"], P, rng)
        rows.append({
            "condition": NAME[cond], "n_total": n_total, "n_flag": n_flag,
            "raw_rate": round(raw_rate, 1), "precision": P,
            "corr_rate": round(corr_rate, 1), "n_true_est": n_drop,
            "mu_all": round(mean_all, 3),
            "mu_clean_dropall": round(mean_unflag, 3),
            "mu_clean_corrected": round(mc, 3), "mu_clean_cf": round(cf, 3),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== Precision-corrected rates and clean means (N precision = 0.833, measured) ===")
    print(out.to_string(index=False))
    out.to_csv(OUT, index=False)

    def clean_of(name, P_over=None):
        cond = [k for k, v in NAME.items() if v == name][0]
        g = df[df["condition"] == cond]
        P = PRECISION[cond] if P_over is None else P_over
        mc, cf, _ = corrected_clean_mean(g["haiku_score"], g["flag"], P,
                                         np.random.default_rng(SEED))
        return mc

    n_val = clean_of("N")
    gthb_val = clean_of("GT-HB")
    gt_val = clean_of("GT")
    print("\n=== Reversal check (corrected clean means) ===")
    print(f"  GT-HB {gthb_val:.3f}  vs  N {n_val:.3f}  -> GT-HB > N ? {gthb_val > n_val}")
    print(f"  GT    {gt_val:.3f}  vs  N {n_val:.3f}  -> GT > N ? {gt_val > n_val}")

    print("\n=== Sensitivity: N's corrected clean mean vs its (unmeasured) precision ===")
    print("  (GT-HB corrected clean =", round(gthb_val, 3), ")")
    thr = None
    for pn in [0.30, 0.40, 0.50, 0.60, 0.642, 0.70, 0.80, 0.833, 0.90, 1.00]:
        nv = clean_of("N", P_over=pn)
        holds = gthb_val > nv
        if thr is None and holds:
            thr = pn
        print(f"  P_N={pn:>4.3f}  N_clean={nv:.3f}  GT-HB>N ? {holds}")
    print(f"\n  Reversal (GT-HB > N) first holds around N precision >= {thr}")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
