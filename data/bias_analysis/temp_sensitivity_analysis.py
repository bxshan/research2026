"""
temp_sensitivity_analysis.py
----------------------------
Assignment 7: temperature sensitivity. Compares per-condition mean bias on the
three sensitivity prompts (climate, government, immigration) across decoding
temperatures 0.3 / 0.7 / 1.0, and checks whether the condition rank order is
preserved (robust) or only holds at 0.7 (a single-temperature limitation).

The 0.7 baseline is reused from the main experiment (no re-run): base/GT/PS/N
from the 1,500-completion graded file, GT-HB from its separate graded file.
Temp 0.3 / 1.0 come from the bias_scores_temp*.csv files; a temperature is
skipped (with a notice) if its graded file is absent.
"""

import csv
import os
import collections

HERE = os.path.dirname(__file__)
SCORES = os.path.join(HERE, "bias_scores")

PROMPTS = {"climate", "government", "immigration"}
COND_ORDER = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki", "llama-sft-gthb"]
DISPLAY = {"base": "Base", "llama-sft-gt": "GT", "llama-sft-ps": "PS",
           "llama-sft-wiki": "N", "llama-sft-gthb": "GT-HB"}

# temp -> list of graded CSVs to pull from (filtered to PROMPTS + COND_ORDER)
BASE_07 = os.path.join(SCORES, "infer_results_20260529_110455.csv")  # base/GT/PS/N @0.7
GTHB_07 = os.path.join(SCORES, "bias_scores_completions_gthb.csv")    # GT-HB @0.7
SOURCES = {
    0.3: [os.path.join(SCORES, "bias_scores_temp03.csv")],
    0.7: [BASE_07, GTHB_07],
    1.0: [os.path.join(SCORES, "bias_scores_temp10.csv")],
}


def load(path):
    """Yield (condition, prompt_id, score:float, hallucinated:bool) for usable rows."""
    if not os.path.exists(path):
        return
    for r in csv.DictReader(open(path)):
        if r.get("prompt_id") not in PROMPTS:
            continue
        if r.get("condition") not in COND_ORDER:
            continue
        bs = r.get("bias_score", "")
        if bs in ("", "-1", None):
            continue
        yield r["condition"], r["prompt_id"], float(bs), str(r.get("hallucinate")) == "True"


def collect():
    """temp -> condition -> {'all':[scores], 'clean':[scores], 'n':int, 'hall':int}."""
    data = {}
    for temp, paths in SOURCES.items():
        present = [p for p in paths if os.path.exists(p)]
        if not present:
            print(f"[skip] temp {temp}: no graded file ({', '.join(os.path.basename(p) for p in paths)})")
            continue
        agg = collections.defaultdict(lambda: {"all": [], "clean": [], "hall": 0})
        for path in present:
            for cond, _p, score, hall in load(path):
                agg[cond]["all"].append(score)
                agg[cond]["hall"] += int(hall)
                if not hall:
                    agg[cond]["clean"].append(score)
        data[temp] = agg
    return data


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    data = collect()
    temps = sorted(data)
    if not temps:
        print("No temperature data found.")
        return

    # ---- table: condition x temp (mean all / mean clean) ----
    print("\nMean bias on 3 sensitivity prompts (climate, government, immigration)")
    print("Each cell: mu_all (mu_clean = hallucination-removed)\n")
    NW, CW = 9, 18  # name width, per-temp column width (must match header + data)
    head = f"{'Condition':<{NW}}" + "".join(f"{'t=' + str(t):>{CW}}" for t in temps)
    print(head)
    print("-" * len(head))
    for cond in COND_ORDER:
        row = f"{DISPLAY[cond]:<{NW}}"
        for t in temps:
            a = data[t].get(cond, {"all": [], "clean": []})
            cell = f"{mean(a['all']):.3f} ({mean(a['clean']):.3f})"
            row += f"{cell:>{CW}}"
        print(row)

    # ---- rank order per temp (by clean mean) + hallucination rate ----
    print("\nCondition rank order by clean mean (high -> low):")
    w = max(len(v) for v in DISPLAY.values())  # pad names so '>' separators align
    for t in temps:
        ranked = sorted(COND_ORDER, key=lambda c: mean(data[t].get(c, {"clean": []})["clean"]), reverse=True)
        print(f"  t={t}: " + " > ".join(f"{DISPLAY[c]:<{w}}" for c in ranked))

    print("\nHallucination rate per condition/temp (%):")
    for cond in COND_ORDER:
        cells = []
        for t in temps:
            a = data[t].get(cond, {"all": [], "hall": 0})
            n = len(a["all"])
            cells.append(f"t={t}:{(100 * a['hall'] / n) if n else float('nan'):5.1f}")
        print(f"  {DISPLAY[cond]:6} " + "  ".join(cells))


if __name__ == "__main__":
    main()
