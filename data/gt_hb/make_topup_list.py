"""
Compute which sources still need rubric scores to reach TARGET_K, sample the
needed articles deterministically, and print a cost estimate. Does NOT call
any API. seed=2 via deterministic hash ordering.
"""
import os
import duckdb
import pandas as pd

TARGET_K     = 20
MIN_ARTICLES = 1000
MIN_CHARS    = 400      # match sft_bias.py min_chars
EST_IN_TOK, EST_OUT_TOK = 1400, 95          # means measured from bias_scores_gt.csv
PRICE_IN, PRICE_OUT     = 1.00, 5.00        # Haiku 4.5 $/MTok

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PARQUET_GLOB = os.path.join(ROOT, "data", "data_full", "nela_gt_full", "data", "*.parquet")
EXISTING     = os.path.join(ROOT, "data", "bias_analysis", "bias_scores", "bias_scores_gt.csv")

meta = pd.read_csv(os.path.join(HERE, "source_metadata.csv"))
have = pd.read_csv(EXISTING).groupby("source").size().rename("n_existing")
plan = (meta[meta.n_articles >= MIN_ARTICLES]
        .merge(have, on="source", how="left")
        .fillna({"n_existing": 0}))
plan["n_needed"] = (TARGET_K - plan.n_existing).clip(lower=0).astype(int)
need = plan[plan.n_needed > 0]
sources_sql = ", ".join("'" + s.replace("'", "''") + "'" for s in need.source)

con = duckdb.connect()
df = con.execute(f"""
    SELECT article_id, source, title, content AS text
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE source IN ({sources_sql})
      AND content IS NOT NULL AND length(content) >= {MIN_CHARS}
    QUALIFY row_number() OVER (
        PARTITION BY source ORDER BY hash(article_id || '-seed2')
    ) <= {TARGET_K}
    ORDER BY source, hash(article_id || '-seed2')
""").df()

# trim each source to exactly n_needed (deterministic: rows arrive lowest-hash-first)
# pandas 3.x groupby().apply() drops the grouping column, so trim via cumcount
df = df.merge(need[["source", "n_needed"]], on="source")
df = (df[df.groupby("source").cumcount() < df["n_needed"]]
        .drop(columns="n_needed")
        .reset_index(drop=True))
df.to_csv(os.path.join(HERE, "topup_articles.csv"), index=False)

n = len(df)
cost = (n * EST_IN_TOK * PRICE_IN + n * EST_OUT_TOK * PRICE_OUT) / 1e6
print(f"{len(plan)} sources qualify (>= {MIN_ARTICLES} articles); "
      f"{len(need)} need top-up; {n} articles to grade")
print(f"ESTIMATED COST: ${cost:.2f} "
      f"({n * EST_IN_TOK / 1e6:.2f}M input tok, {n * EST_OUT_TOK / 1e6:.3f}M output tok)")
print("Review this number before running the grader (Task 4).")
