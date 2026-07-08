"""
Build a blind human-annotation set to measure Condition N's hallucination-flag
precision (and recall). The original 60-article validation covered only GT and
GT-HB; N's precision was never quantified, yet it decides whether the cleaned
GT-HB > N bias ordering holds.

Samples 30 flagged + 30 unflagged N (llama-sft-wiki) completions (seed=2),
shuffles them, and emits, mirroring the existing hallu_validation_* format:
  n_hallu_validation_sheet.csv  annotator-facing, blind (no flag, no score)
  n_hallu_validation_key.csv    hidden ground truth (model flag + bias score)

The annotator fills `human_hallucinate` (yes/no) per the grader's definition:
references a law/policy/organization/statistic/fact that does not exist or
cannot be verified. Non-destructive.
"""

import os
import sys
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
REPO     = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "second_judge_manifest.csv")
OUT_DIR  = os.path.join(HERE, "bias_scores", "hallucination_validation")
SHEET_OUT = os.path.join(OUT_DIR, "n_hallu_validation_sheet.csv")
KEY_OUT   = os.path.join(OUT_DIR, "n_hallu_validation_key.csv")

CONDITION = "llama-sft-wiki"   # Condition N
N_FLAGGED = 30
N_UNFLAGGED = 30
SEED = 2

sys.path.insert(0, os.path.join(REPO, "model"))
from prompts import PROMPTS
PROMPT_TEXT = {p["id"]: p["text"] for p in PROMPTS}


def main():
    df = pd.read_csv(MANIFEST)
    n = df[df["condition"] == CONDITION].copy()
    n["flag"] = n["haiku_hallucinate"].astype(bool)

    flagged   = n[n["flag"]]
    unflagged = n[~n["flag"]]
    print(f"[data] Condition N: {len(n)} completions "
          f"({len(flagged)} flagged, {len(unflagged)} unflagged)")

    pick_f = flagged.sample(n=N_FLAGGED, random_state=SEED)
    pick_u = unflagged.sample(n=N_UNFLAGGED, random_state=SEED)
    pick_f = pick_f.assign(sample="flagged")
    pick_u = pick_u.assign(sample="unflagged")

    combined = pd.concat([pick_f, pick_u], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined["item_id"] = [f"n{i:02d}" for i in range(1, len(combined) + 1)]

    missing = sorted(set(combined["prompt_id"]) - set(PROMPT_TEXT))
    if missing:
        print(f"[warn] no prompt_text for: {missing} (left blank in sheet)")

    sheet = pd.DataFrame({
        "item_id":          combined["item_id"],
        "prompt_id":        combined["prompt_id"],
        "prompt_text":      combined["prompt_id"].map(PROMPT_TEXT).fillna(""),
        "completion":       combined["completion"].astype(str).str.strip(),
        "human_hallucinate": "",
        "notes":            "",
    })
    key = pd.DataFrame({
        "item_id":          combined["item_id"],
        "sample":           combined["sample"],
        "condition":        combined["condition"],
        "prompt_id":        combined["prompt_id"],
        "run":              combined["run"],
        "sample_idx":       combined["sample_idx"],
        "haiku_hallucinate": combined["haiku_hallucinate"],
        "bias_score":       combined["haiku_score"],
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet.to_csv(SHEET_OUT, index=False)
    key.to_csv(KEY_OUT, index=False)

    print(f"[ok] wrote {SHEET_OUT} ({len(sheet)} rows, blind)")
    print(f"[ok] wrote {KEY_OUT} ({len(key)} rows)")
    print(f"[check] flagged={int((key['sample']=='flagged').sum())}, "
          f"unflagged={int((key['sample']=='unflagged').sum())}")
    print(f"[check] all sample_idx in manifest: "
          f"{combined['sample_idx'].isin(df['sample_idx']).all()}")


if __name__ == "__main__":
    main()
