# Hallucination-Flag Validation

Validates reliability of binary hallucinate flag for the LLM-as-Judge. 
Specifically targets GT and derived GT-HB corpora, since most hallucinations appear there, and they are the main references for the bias transfer claim. 

## Files
| File | Use |
|---|---|
| `build_validation_sheets.py` | builds sheets from sample (invariant`SEED=2`). |
| `hallu_validation_sheet.csv` | annotation sheet |
| `hallu_validation_key.csv` | answer key |

## Sample
60 GT/GT-HB completions, shuffled into one sheet:
- 30 that Haiku flagged `hallucinate=1`  (15 GT, 15 GT-HB) — for precision (true positives / all guessed positives)
- 30 that Haiku flagged `hallucinate=0`  (15 GT, 15 GT-HB) — for recall (true positives / ground truth positives)

## Scoring 
Join sheet + key on `item_id`, then (with human as ground truth):
- Precision = (human=1 of flagged) / (all `sample==flagged`) - wish to maximize
- Recall = (human=1 of flagged) / (all human=1) - wish to maximize
- Matching grades = (human == haiku) / 60 - overall agreement
- Report all three overall and per condition (GT vs GT-HB).

## Results

| Confusion | Human=1 (fabricated) | Human=0 (clean) | Total |
|---|---|---|---|
| **Haiku=1 (flagged)** | TP = 9               | FP = 21         | 30    |
| **Haiku=0 (unflagged)** | FN = 0             | TN = 30         | 30    |
| **Total**             | 9                    | 51              | 60    |

Agreement (matching grades) = (TP+TN)/N = (9+30)/60 = **65.0%**.

| Metric | Overall | GT | GT-HB |
|---|---|---|---|
| Precision | 30.0% (9/30) | 20.0% (3/15) | 40.0% (6/15) |
| Recall | 100% (9/9) | 100% (3/3) | 100% (6/6) |
| Matching grades | 65.0% (39/60) | 60.0% (18/30) | 70.0% (21/30) |

Haiku missed zero genuine fabrications (recall 100%, FN=0) but over-flags, as only 30% of its
flags on GT/GT-HB are real hallucinations.

