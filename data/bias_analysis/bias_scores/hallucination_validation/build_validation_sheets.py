"""
Build blinded annotation sheet for hallucination-flag validation.

Samples GT + GT-HB completions into ONE shuffled, blinded sheet so human 
judges each completion cold (Haiku's flag is withheld). After annotation, join the
sheet back to the key on item_id and split by haiku_hallucinate to compute:
"""
import os
import sys
import pandas as pd

SEED = 2                   # important invariant
N_PER_GROUP = 15           # per condition, per flag-class -> 60 total (30 flagged, 30 not)

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.dirname(HERE)                                  # .../bias_scores
REPO = os.path.dirname(os.path.dirname(os.path.dirname(SCORES)))  # repo root
sys.path.insert(0, os.path.join(REPO, "model"))
from prompts import PROMPTS                                     # noqa: E402

PROMPT_TEXT = {p["id"]: p["text"] for p in PROMPTS}

GT_FILE = os.path.join(SCORES, "infer_results_20260529_110455.csv")
GTHB_FILE = os.path.join(SCORES, "bias_scores_completions_gthb.csv")

KEEP = ["condition", "prompt_id", "run", "completion", "bias_score", "hallucinate"]


def load():
    gt = pd.read_csv(GT_FILE)
    gt = gt[gt["condition"] == "llama-sft-gt"][KEEP]
    gthb = pd.read_csv(GTHB_FILE)
    gthb = gthb[gthb["condition"] == "llama-sft-gthb"][KEEP]
    return pd.concat([gt, gthb], ignore_index=True)


def sample_group(df, condition, flag, n):
    pool = df[(df["condition"] == condition) & (df["hallucinate"] == flag)]
    return pool.sample(n=min(n, len(pool)), random_state=SEED)


def main():
    df = load()
    parts = []
    for cond in ["llama-sft-gt", "llama-sft-gthb"]:
        for flag in [1, 0]:
            parts.append(sample_group(df, cond, flag, N_PER_GROUP))
    sample = pd.concat(parts, ignore_index=True)

    # shuffle so flagged/non-flagged and GT/GT-HB are interleaved
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["item_id"] = [f"h{i+1:02d}" for i in range(len(sample))]
    sample["sample"] = sample["hallucinate"].apply(
        lambda x: "flagged" if x else "nonflagged")
    sample["prompt_text"] = sample["prompt_id"].map(PROMPT_TEXT)

    # blinded sheet: NO condition, NO bias_score, NO haiku flag
    sheet = sample[["item_id", "prompt_id", "prompt_text", "completion"]].copy()
    sheet["human_hallucinate"] = ""     # annotator: 1 if a fabricated/nonexistent fact, else 0
    sheet["notes"] = ""
    sheet_path = os.path.join(HERE, "hallu_validation_sheet.csv")
    sheet.to_csv(sheet_path, index=False)

    # hidden key: the withheld truth for scoring
    key = sample[["item_id", "sample", "condition", "prompt_id", "run",
                  "hallucinate", "bias_score"]].rename(
        columns={"hallucinate": "haiku_hallucinate"})
    key_path = os.path.join(HERE, "hallu_validation_key.csv")
    key.to_csv(key_path, index=False)

    print(f"[ok] sheet -> {sheet_path}  ({len(sheet)} rows)")
    print(f"[ok] key   -> {key_path}")
    print("\nsample composition:")
    print(sample.groupby(["sample", "condition"]).size())


if __name__ == "__main__":
    main()
