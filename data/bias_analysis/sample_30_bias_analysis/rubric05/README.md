# 0–5 Rubric Human-vs-LLM Validation

Workflow order (run from repo root):

1. **Fill the worksheet by hand.** Score every article in `gt_sample_30_human_worksheet.csv` on the 0–5 rubric, filling the blank `bias_score_human_05` column (generate the blank worksheet with `make_worksheet.py`).
2. **Run the LLM grader** (needs `ANTHROPIC_API_KEY`): `python3 grade_sample_30_05.py` → writes `gt_sample_30_haiku_05.csv`.
3. **Run the validation:** `python3 rubric05_validation_stats.py` → prints agreement metrics and writes `rubric05_validation.csv`.
