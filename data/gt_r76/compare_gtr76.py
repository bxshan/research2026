"""
compare_gtr76.py
----------------
Compare a scored GT-R76 inference run against the stored GT / GT-HB / base / N
references, using the SAME aggregation logic as bias_score_analysis.py.

GT-R76 isn't in bias_score_analysis.py's hardcoded condition list, so this script
computes its stats directly with the identical formulas:
  - mean      = bias_score.mean()
  - bias_rate = (bias_score >= 1).mean() * 100      (paper "Bias Rate")
  - pct_0     = (bias_score == 0).mean() * 100
and reuses load_infer_scores() verbatim for identical loading / NaN handling.

Usage:
  python3 data/gt_r76/compare_gtr76.py \
      --scored data/bias_analysis/bias_scores/infer_results_<ts>.csv \
      --label  llama-sft-gtr76-seed2
"""
import os, sys, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(HERE, "..", "bias_analysis")
OUT_DIR = os.path.join(ANALYSIS_DIR, "bias_score_analysis_out", "completions_analysis_out")
REF_SUMMARY  = os.path.join(OUT_DIR, "infer_bias_summary.csv")        # reference overall means
REF_PERPROMPT = os.path.join(OUT_DIR, "variance_analysis.csv")        # reference per-prompt means

# reuse the exact loader bias_score_analysis.py uses for the stored numbers
sys.path.insert(0, ANALYSIS_DIR)
from bias_score_analysis import load_infer_scores   # noqa: E402


def stats(s):
    return dict(n=len(s), mean=round(s.mean(), 3),
                bias_rate=round((s >= 1).mean() * 100, 1),
                pct_0=round((s == 0).mean() * 100, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored infer CSV (from bias_grader)")
    ap.add_argument("--label",  default="llama-sft-gtr76-seed2", help="GT-R76 condition label")
    args = ap.parse_args()

    df = load_infer_scores(args.scored)
    neg = int((df["bias_score"] == -1).sum())
    print(f"[data] {len(df)} scored rows  | conditions: {sorted(df['condition'].unique())}"
          f"  | parse-error(-1) rows: {neg}")
    if neg:
        print(f"[warn] {neg} rows scored -1 (parse error) — included to match reference handling")

    # ---- 1. base anchor: does the fresh pipeline reproduce base ~ 0.847? ----
    if "base" in set(df["condition"]):
        b = stats(df[df["condition"] == "base"]["bias_score"])
        drift = b["mean"] - 0.847
        flag = "OK" if abs(drift) <= 0.10 else "DRIFT — investigate before trusting comparison"
        print(f"\n[anchor] fresh base mean = {b['mean']}  vs stored 0.847  (Δ={drift:+.3f})  [{flag}]")

    # ---- 2. overall comparison ----
    ref = pd.read_csv(REF_SUMMARY).set_index("condition")
    ref_bias_rate = (100 - ref["pct_0"]).round(1)
    g = stats(df[df["condition"] == args.label]["bias_score"])
    seed_n  = args.label.split("seed")[-1]   # "llama-sft-gtr76-seed22" -> "22"
    r76_col = f"R76_s{seed_n}"               # CSV column header (e.g. R76_s22)
    r76_lbl = f"R76-s{seed_n}"               # console label
    print("\n[overall]  mean (bias_rate %)   — GT-R76 vs references")
    order = [("base", "base"), ("llama-sft-gt", "GT"), ("llama-sft-gthb", "GT-HB"),
             ("llama-sft-wiki", "N"), ("llama-sft-ps", "PS")]
    for cond, name in order:
        if cond in ref.index:
            print(f"   {name:<7} {ref.loc[cond,'mean']:>6.3f}  ({ref_bias_rate[cond]:>5.1f}%)")
    print(f"   {r76_lbl:<7} {g['mean']:>6.3f}  ({g['bias_rate']:>5.1f}%)   <-- this run  (n={g['n']}, %0={g['pct_0']})")

    # ---- 3. per-prompt broadening test: does R76 exceed GT on topics GT left flat? ----
    refp = pd.read_csv(REF_PERPROMPT)
    gt   = refp[refp["condition"] == "llama-sft-gt"].set_index("prompt_id")["mean"]
    gthb = refp[refp["condition"] == "llama-sft-gthb"].set_index("prompt_id")["mean"]
    r76  = (df[df["condition"] == args.label].groupby("prompt_id")["bias_score"].mean())

    print(f"\n[per-prompt]  mean by topic — GT | GT-HB | {r76_lbl} | (R76-GT) | (GT-HB-GT)")
    rows = []
    for p in sorted(r76.index):
        gtv, hbv, rv = gt.get(p), gthb.get(p), r76[p]
        d_r76 = (rv - gtv) if pd.notna(gtv) else float("nan")
        d_hb  = (hbv - gtv) if (pd.notna(gtv) and pd.notna(hbv)) else float("nan")
        rows.append({"prompt_id": p, "GT": gtv, "GT_HB": hbv, r76_col: round(rv, 3),
                     "R76_minus_GT": round(d_r76, 3), "GTHB_minus_GT": round(d_hb, 3)})
        print(f"   {p:<16} {gtv!s:>6} | {hbv!s:>6} | {rv:>6.3f} | {d_r76:>+7.3f} | {d_hb:>+7.3f}")

    out = os.path.join(HERE, f"comparison_{args.label}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[saved] per-prompt comparison -> {out}")
    print("[read]  If R76 'R76_minus_GT' tracks 'GTHB_minus_GT' (broadens on the same topics),"
          "\n        source concentration explains GT-HB. If R76_minus_GT ~ 0 (R76 looks like GT),"
          "\n        the broadening is bias-driven. NOTE: seed 2 is one replicate — confirm with seeds 22/222.")


if __name__ == "__main__":
    main()
