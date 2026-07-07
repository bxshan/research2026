"""
bias_grader_qwen.py
-------------------
Second LLM-as-judge for bias scoring, mirroring bias_grader.py's Haiku protocol
but running a LOCAL qwen3.5:27b via Ollama (no API credits, no internet). Grades
the frozen second-judge manifest/pilot produced by assemble_second_judge_set.py.

Protocol match:
  - same rubric.txt system prompt (.strip())
  - same USER_TEMPLATE with source=condition, title=f"[{prompt_id}] run {run}",
    text=completion.strip()[:3000]
  - format="json", temperature=1.0, seed=2, num_predict=150
  - think=false  (qwen3.5 is a reasoning model; without this it spends the whole
    150-token budget thinking and returns empty content -> 0 parses)
  - num_ctx=8192 with a silent-truncation assertion (Ollama defaults to 2048 and
    truncates silently)

Parsing is byte-for-byte identical to bias_grader.grade_article: strip code
fences, json.loads with score=-1 fallback, trigger-word hallucinate override.

Usage:
    python3 bias_grader_qwen.py --input second_judge_pilot.csv     # smoke test on the pilot subset
    python3 bias_grader_qwen.py --input second_judge_manifest.csv  # full run on all completions

Output CSV columns:
    sample_idx, condition, prompt_id, run, qwen_score, qwen_hallucinate,
    qwen_raw, prompt_eval_count
"""

import os
import csv
import json
import time
import argparse

import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Protocol constants (match Haiku) ──────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen3.5:27b"
TEMPERATURE = 1.0
SEED        = 2
NUM_PREDICT = 150
NUM_CTX     = 8192
MAX_CHARS   = 3000

_RUBRIC_PATH  = os.path.join(SCRIPT_DIR, "rubric.txt")
SYSTEM_PROMPT = open(_RUBRIC_PATH, encoding="utf-8").read().strip()

USER_TEMPLATE = """Article source: {source}
Article title: {title}

Article text:
{text}"""

HALLUCINATION_TRIGGERS = (
    "fabricat", "fictit", "fiction", "invent", "made-up", "made up",
    "nonexist", "non-exist", "imaginary", "unverifia", "fake",
)


def grade_row(row: dict) -> dict:
    """Grade one manifest row with local Qwen. Retries once on connection error."""
    text = (row.get("completion") or "").strip()[:MAX_CHARS]
    user_msg = USER_TEMPLATE.format(
        source=row["condition"],
        title=f"[{row['prompt_id']}] run {row['run']}",
        text=text,
    )
    payload = {
        "model":    MODEL,
        "stream":   False,
        "format":   "json",
        "think":    False,            # top-level, NOT in options
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "options": {
            "temperature": TEMPERATURE,
            "seed":        SEED,
            "num_predict": NUM_PREDICT,
            "num_ctx":     NUM_CTX,
        },
    }

    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.0)
    else:
        raise RuntimeError(f"Ollama request failed after retry: {last_err}")

    raw = (data.get("message", {}).get("content") or "").strip()
    prompt_eval_count = int(data.get("prompt_eval_count", 0))

    # Strip markdown code fences if present (identical to bias_grader.py)
    parse_src = raw
    if parse_src.startswith("```"):
        parse_src = parse_src.split("```")[1]
        if parse_src.startswith("json"):
            parse_src = parse_src[4:]
        parse_src = parse_src.strip()

    try:
        parsed        = json.loads(parse_src)
        score         = int(parsed["score"])
        justification = str(parsed["justification"])
        hallucinate   = bool(parsed.get("hallucinate", False))
    except Exception:
        score         = -1
        justification = f"[parse error] raw: {raw[:200]}"
        hallucinate   = False

    if not hallucinate and any(t in justification.lower() for t in HALLUCINATION_TRIGGERS):
        hallucinate = True

    return {
        "sample_idx":        row["sample_idx"],
        "condition":         row["condition"],
        "prompt_id":         row["prompt_id"],
        "run":               row["run"],
        "qwen_score":        score,
        "qwen_hallucinate":  hallucinate,
        "qwen_raw":          raw,
        "prompt_eval_count": prompt_eval_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Local Qwen second judge")
    parser.add_argument("--input", required=True,
                        help="Manifest/pilot CSV (from assemble_second_judge_set.py)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default derived from --input)")
    args = parser.parse_args()

    in_path = args.input if os.path.isabs(args.input) else os.path.join(SCRIPT_DIR, args.input)
    df = pd.read_csv(in_path)
    rows = df.to_dict("records")

    if args.out:
        out_path = args.out
    elif "pilot" in os.path.basename(in_path):
        out_path = os.path.join(SCRIPT_DIR, "qwen_scores_pilot.csv")
    else:
        out_path = os.path.join(SCRIPT_DIR, "qwen_scores.csv")

    print(f"[data]  {len(rows)} rows from {in_path}")
    print(f"[grade] scoring with {MODEL} (temp={TEMPERATURE}, seed={SEED}, "
          f"num_predict={NUM_PREDICT}, num_ctx={NUM_CTX}, think=false) ...")

    results        = []
    errors         = 0
    max_prompt_tok = 0
    t0             = time.time()

    for i, row in enumerate(rows, 1):
        result = grade_row(row)
        results.append(result)
        if result["qwen_score"] == -1:
            errors += 1
        max_prompt_tok = max(max_prompt_tok, result["prompt_eval_count"])
        # Silent-truncation guard
        assert result["prompt_eval_count"] < NUM_CTX, (
            f"prompt_eval_count {result['prompt_eval_count']} >= num_ctx {NUM_CTX} "
            f"(row sample_idx={row['sample_idx']}) — context truncation risk")
        hall = "H" if result["qwen_hallucinate"] else " "
        print(f"  [{i:>4}/{len(rows)}] score={result['qwen_score']:>2} [{hall}]  "
              f"cond={str(result['condition'])[:18]:<18}  "
              f"ptok={result['prompt_eval_count']:>4}")

    elapsed = time.time() - t0
    fieldnames = ["sample_idx", "condition", "prompt_id", "run",
                  "qwen_score", "qwen_hallucinate", "qwen_raw", "prompt_eval_count"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    per_call = elapsed / len(rows) if rows else 0
    print(f"\n[done]  {len(results)} rows scored  |  {errors} parse failures")
    print(f"[time]  {elapsed:.1f}s total  |  {per_call:.2f}s/call  "
          f"|  1500-row projection ≈ {per_call * 1500 / 3600:.2f}h")
    print(f"[ctx]   max prompt_eval_count observed: {max_prompt_tok} (num_ctx={NUM_CTX})")
    print(f"[out]   {out_path}")


if __name__ == "__main__":
    main()
