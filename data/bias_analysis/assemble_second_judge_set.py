#!/usr/bin/env python3
"""
Assembles the FULL set of graded completions the Haiku judge produced — the same
1,500-row set behind hallucination_summary.csv — into a frozen manifest for the
local-Qwen second judge, plus a small stratified pilot slice for a smoke test.

The 1,500-row set is NOT one file: bias_score_analysis.py concatenates
  bias_scores/infer_results_20260529_110455.csv   (1,200 rows: base/gt/ps/wiki)
+ bias_scores/bias_scores_completions_gthb.csv     (300 rows: gthb)
at runtime (bias_score_analysis.py:491-493). We mirror that here.

Outputs (written next to this script, non-destructive):
  second_judge_manifest.csv   full 1,500 rows to grade with Qwen
  second_judge_pilot.csv      ~50-row stratified smoke-test subset (strict subset of manifest)

Reproducible: pilot sampling uses random.seed(2). Rerun -> byte-identical outputs.
"""
import os
import random
import pandas as pd

HERE       = os.path.dirname(os.path.abspath(__file__))
SCORES_DIR = os.path.join(HERE, "bias_scores")
MAIN_CSV   = os.path.join(SCORES_DIR, "infer_results_20260529_110455.csv")  # base/gt/ps/wiki
GTHB_CSV   = os.path.join(SCORES_DIR, "bias_scores_completions_gthb.csv")   # gthb

MANIFEST_OUT = os.path.join(HERE, "second_judge_manifest.csv")
PILOT_OUT    = os.path.join(HERE, "second_judge_pilot.csv")

MANIFEST_COLS = ["sample_idx", "condition", "prompt_id", "run",
                 "completion", "haiku_score", "haiku_hallucinate"]

SEED = 2
PILOT_PER_CELL = 2      # rows per (condition, haiku_score) cell -> ~50 total
MIN_COMPLETION_CHARS = 50


def _load_and_normalize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # hallucinate is stored as "True"/"False" strings (see bias_score_analysis.py:206)
    df["haiku_hallucinate"] = df["hallucinate"].astype(str).str.lower().eq("true")
    df = df.rename(columns={"bias_score": "haiku_score"})
    return df[["condition", "prompt_id", "run", "completion", "haiku_score", "haiku_hallucinate"]]


def assemble() -> pd.DataFrame:
    main = _load_and_normalize(MAIN_CSV)
    gthb = _load_and_normalize(GTHB_CSV)
    df = pd.concat([main, gthb], ignore_index=True)   # mirrors bias_score_analysis.py:491-493

    n0 = len(df)
    # drop parse failures and too-short completions (verified none present, but be robust)
    df = df[df["haiku_score"] != -1]
    comp_len = df["completion"].fillna("").astype(str).str.strip().str.len()
    df = df[comp_len >= MIN_COMPLETION_CHARS]
    dropped = n0 - len(df)
    if dropped:
        print(f"[assemble] dropped {dropped} rows (haiku_score==-1 or completion <{MIN_COMPLETION_CHARS} chars)")

    # stable ordering so sample_idx is deterministic, then assign it
    df = df.sort_values(["condition", "prompt_id", "run"]).reset_index(drop=True)
    df.insert(0, "sample_idx", range(len(df)))
    return df[MANIFEST_COLS]


def pilot_slice(manifest: pd.DataFrame) -> pd.DataFrame:
    """Stratified subset: up to PILOT_PER_CELL rows per (condition, haiku_score) cell that exists,
    so the smoke test exercises a parse for every score level 0-5 and every condition."""
    rng = random.Random(SEED)
    picks = []
    for (cond, score), cell in manifest.groupby(["condition", "haiku_score"], sort=True):
        idxs = sorted(cell["sample_idx"].tolist())
        rng.shuffle(idxs)
        picks.extend(idxs[:PILOT_PER_CELL])
    picks = sorted(set(picks))
    return manifest[manifest["sample_idx"].isin(picks)].reset_index(drop=True)


def _report(df: pd.DataFrame, name: str):
    print(f"\n=== {name}: {len(df)} rows ===")
    print("per-condition:")
    print(df["condition"].value_counts().sort_index().to_string())
    print("per-score-band (haiku_score):")
    print(pd.crosstab(df["condition"], df["haiku_score"]).to_string())


def main():
    manifest = assemble()
    pilot = pilot_slice(manifest)

    manifest.to_csv(MANIFEST_OUT, index=False)
    pilot.to_csv(PILOT_OUT, index=False)

    _report(manifest, "MANIFEST (full)")
    _report(pilot, "PILOT (smoke test)")

    # sanity checks
    assert pilot["sample_idx"].isin(manifest["sample_idx"]).all(), "pilot not a subset of manifest"
    print(f"\n[ok] wrote {MANIFEST_OUT} ({len(manifest)} rows)")
    print(f"[ok] wrote {PILOT_OUT} ({len(pilot)} rows, strict subset of manifest)")


if __name__ == "__main__":
    main()
