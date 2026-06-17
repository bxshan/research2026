"""
select_sources.py
------------------
Build the GT-R76 (random-source control) whitelist.

GT-HB selects 76 GT sources by per-source bias (mean rubric score >= 3.0). This
control instead selects 76 GT sources AT RANDOM, regardless of bias, so that the
only thing that differs between GT-HB and GT-R76 is the source-selection
criterion (bias-concentrated vs. random). If GT-R76 also broadens bias, then
source concentration (not bias concentration) explains the GT-HB effect.

Method (deterministic, reproducible):
  1. Read the full GT source inventory (data/gt_hb/source_metadata.csv, 474 sources).
  2. Keep sources with n_articles >= MIN_ARTICLES (1000) — the SAME structural
     constraint GT-HB applied (finalize_sources.py MIN_ARTICLES), so the pool is
     comparable and large enough to draw 500k. The bias-specific GT-HB filters
     (mean_score >= 3.0, n_graded >= 10) are intentionally NOT applied here.
  3. Randomly sample 76 of the eligible sources with a fixed seed (SEED=2, matching
     the seed used everywhere else in the pipeline: train_config seed=2).

Output: data/gt_r76/gt_r76_sources.csv — a whitelist with the same `source`
column the training loader keys on (mirrors data/gt_hb/gt_hb_sources.csv). The
actual 500,000-article sample is drawn at train time by load_gtr76() in
model/sft_bias.py, using the identical seeded shuffle as load_gthb(), so the
sampling method is byte-for-byte the same as GT / GT-HB.

Usage:
  python3 data/gt_r76/select_sources.py
  or pass in optional --seed flag
"""

import os
import argparse
import csv
import random
import collections

HERE          = os.path.dirname(os.path.abspath(__file__))
SOURCE_META   = os.path.join(HERE, "..", "gt_hb", "source_metadata.csv")
# OUT_PATH      = os.path.join(HERE, "gt_r76_sources.csv")

N_SOURCES     = 76      # match GT-HB's source count exactly
MIN_ARTICLES  = 1000    # match GT-HB's MIN_ARTICLES (finalize_sources.py)
TARGET_POOL   = 500_000 # the 500k training draw must be achievable from the pool
SEED          = 2       # match the pipeline-wide seed (train_config_cloud.yaml: seed=2)


def main():
    parser = argparse.ArgumentParser(
        description="Select a random subset of GT sources (GT-R76 control)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed for the mt19937 RNG that picks the 76 sources")
    args = parser.parse_args()
    seed = args.seed


    with open(SOURCE_META, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = rows[0].keys()

    def n_articles(r):
        try:
            return int(float(r["n_articles"]))
        except (ValueError, KeyError):
            return 0

    eligible = [r for r in rows if n_articles(r) >= MIN_ARTICLES]
    print(f"[gt-r76] total GT sources            : {len(rows)}")
    print(f"[gt-r76] eligible (n_articles >= {MIN_ARTICLES}) : {len(eligible)}")

    # Deterministic random draw: sort by source name for a canonical population
    # order, then sample with a fixed-seed RNG. Pure-Python random is stable
    # across machines/runs, so this is fully reproducible.
    eligible_sorted = sorted(eligible, key=lambda r: r["source"])
    picked = random.Random(seed).sample(eligible_sorted, N_SOURCES)
    picked.sort(key=lambda r: r["source"])

    pool = sum(n_articles(r) for r in picked)
    print(f"[gt-r76] selected sources            : {len(picked)}  (seed={seed})")
    print(f"[gt-r76] pooled articles in 76 sources: {pool:,}")
    if pool < TARGET_POOL:
        raise SystemExit(
            f"[gt-r76] ERROR: pool {pool:,} < target {TARGET_POOL:,}; "
            f"cannot draw {TARGET_POOL:,} articles. Re-check eligibility."
        )

    # Sanity: MBFC bias distribution of the random draw (should look broad/
    # representative, NOT skewed high like GT-HB).
    dist = collections.Counter((r.get("bias") or "(none)") for r in picked)
    print("[gt-r76] MBFC bias distribution of the 76 random sources:")
    for label, k in dist.most_common():
        print(f"           {label:<14} {k}")

    FILENAME = "gt_r76_seed" + str(seed) + "_sources.csv"
    OUT_PATH = os.path.join(HERE, FILENAME)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(picked)
    print(f"[gt-r76] wrote whitelist -> {OUT_PATH}")


if __name__ == "__main__":
    main()
