"""
second_judge_agreement.py
--------------------------
Joins the Haiku scores (frozen in the manifest) against the local-Qwen scores
and reports inter-judge agreement to validate the bias scores aren't an artifact
of a single judge model.

Pure transform of the two score files — no invented numbers.

Usage:
    python3 second_judge_agreement.py                       # full manifest + qwen_scores.csv
    python3 second_judge_agreement.py \
        --manifest second_judge_pilot.csv --qwen qwen_scores_pilot.csv \
        --out /tmp/agreement_pilot.csv                      # validate on the pilot

Metrics: Pearson r/p, Spearman rho, within-+-1, exact-agreement, MAE; per-condition
and per-Haiku-band breakdowns; hallucinate raw agreement + Cohen's kappa.
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two boolean/categorical label series."""
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    # expected agreement from marginal rates of True/False
    pa_t, pb_t = a.mean(), b.mean()
    pe = pa_t * pb_t + (1 - pa_t) * (1 - pb_t)
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def pair_metrics(h: pd.Series, q: pd.Series) -> dict:
    """Agreement metrics for two aligned integer score series."""
    d = q - h
    out = {
        "n":          int(len(h)),
        "exact":      round(float((d == 0).mean()), 4),
        "within_1":   round(float((d.abs() <= 1).mean()), 4),
        "mae":        round(float(d.abs().mean()), 4),
    }
    # correlations require variance in both series
    if len(h) >= 2 and h.nunique() > 1 and q.nunique() > 1:
        r, p   = pearsonr(h, q)
        rho, _ = spearmanr(h, q)
        out["pearson_r"] = round(float(r), 4)
        out["pearson_p"] = float(p)
        out["spearman_rho"] = round(float(rho), 4)
    else:
        out["pearson_r"] = out["pearson_p"] = out["spearman_rho"] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description="Second-judge agreement")
    parser.add_argument("--manifest", default="second_judge_manifest.csv",
                        help="Manifest CSV with haiku_score / haiku_hallucinate")
    parser.add_argument("--qwen", default="qwen_scores.csv",
                        help="Qwen scores CSV from bias_grader_qwen.py")
    parser.add_argument("--out", default=None, help="Output summary CSV path")
    args = parser.parse_args()

    man_path  = args.manifest if os.path.isabs(args.manifest) else os.path.join(SCRIPT_DIR, args.manifest)
    qwen_path = args.qwen     if os.path.isabs(args.qwen)     else os.path.join(SCRIPT_DIR, args.qwen)
    out_path  = args.out or os.path.join(SCRIPT_DIR, "second_judge_agreement.csv")

    man  = pd.read_csv(man_path)
    qwen = pd.read_csv(qwen_path)

    # Inner-join on the stable per-row key; (condition, prompt_id, run) also unique.
    keys = ["condition", "prompt_id", "run"]
    j = man.merge(qwen[keys + ["qwen_score", "qwen_hallucinate"]], on=keys, how="inner")

    n_total = len(j)
    n_bad = int(((j["haiku_score"] == -1) | (j["qwen_score"] == -1)).sum())
    j = j[(j["haiku_score"] != -1) & (j["qwen_score"] != -1)].reset_index(drop=True)
    print(f"[join]  {n_total} paired rows  |  dropped {n_bad} with a -1 score  "
          f"|  {len(j)} usable")

    h, q = j["haiku_score"], j["qwen_score"]

    # ── Overall ───────────────────────────────────────────────────────────────
    overall = pair_metrics(h, q)
    hall_agree = float((j["haiku_hallucinate"].astype(bool) == j["qwen_hallucinate"].astype(bool)).mean())
    hall_kappa = cohens_kappa(j["haiku_hallucinate"], j["qwen_hallucinate"])

    print("\n=== OVERALL (Haiku vs Qwen) ===")
    print(f"  n            : {overall['n']}")
    print(f"  Pearson r    : {overall['pearson_r']}  (p={overall['pearson_p']:.2e})")
    print(f"  Spearman rho : {overall['spearman_rho']}")
    print(f"  within +-1   : {overall['within_1']}")
    print(f"  exact match  : {overall['exact']}")
    print(f"  MAE          : {overall['mae']}")
    print(f"  hallucinate agree : {round(hall_agree,4)}  |  Cohen's kappa : {round(hall_kappa,4)}")

    # ── Per-condition ───────────────────────────────────────────────────────────
    print("\n=== PER-CONDITION ===")
    rows = [{"scope": "overall", "key": "all", **overall,
             "hall_agree": round(hall_agree, 4), "hall_kappa": round(hall_kappa, 4)}]
    for cond, g in j.groupby("condition"):
        m = pair_metrics(g["haiku_score"], g["qwen_score"])
        ha = float((g["haiku_hallucinate"].astype(bool) == g["qwen_hallucinate"].astype(bool)).mean())
        hk = cohens_kappa(g["haiku_hallucinate"], g["qwen_hallucinate"])
        print(f"  {cond:<16} n={m['n']:>4}  r={m['pearson_r']}  "
              f"within1={m['within_1']}  mae={m['mae']}  hall_agree={round(ha,3)}")
        rows.append({"scope": "condition", "key": cond, **m,
                     "hall_agree": round(ha, 4), "hall_kappa": round(hk, 4)})

    # ── Per Haiku score band ─────────────────────────────────────────────────────
    print("\n=== PER HAIKU SCORE BAND ===")
    for band, g in j.groupby("haiku_score"):
        d = g["qwen_score"] - g["haiku_score"]
        print(f"  haiku={int(band)}  n={len(g):>4}  "
              f"within1={round(float((d.abs()<=1).mean()),3)}  "
              f"mean_qwen={round(float(g['qwen_score'].mean()),2)}  "
              f"mae={round(float(d.abs().mean()),3)}")
        rows.append({"scope": "haiku_band", "key": int(band), "n": int(len(g)),
                     "within_1": round(float((d.abs() <= 1).mean()), 4),
                     "mae": round(float(d.abs().mean()), 4),
                     "exact": round(float((d == 0).mean()), 4),
                     "mean_qwen": round(float(g["qwen_score"].mean()), 4)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[out]   {out_path}")


if __name__ == "__main__":
    main()
