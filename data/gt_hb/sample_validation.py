"""
Draw 500 articles uniformly from the GT-HB pool (whitelisted sources) for
rubric grading — the corpus-level validation that pre-filtering raised bias.
Deterministic hash ordering stands in for seed=2.
"""
import os
import duckdb
import pandas as pd

N = 500
MIN_CHARS = 400

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PARQUET_GLOB = os.path.join(ROOT, "data", "data_full", "nela_gt_full", "data", "*.parquet")

sel = pd.read_csv(os.path.join(HERE, "gt_hb_sources.csv"))
sources_sql = ", ".join("'" + s.replace("'", "''") + "'" for s in sel.source)

con = duckdb.connect()
df = con.execute(f"""
    SELECT article_id, source, title, content AS text
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE source IN ({sources_sql})
      AND content IS NOT NULL AND length(content) >= {MIN_CHARS}
    ORDER BY hash(article_id || '-val-seed2')
    LIMIT {N}
""").df()
df.to_csv(os.path.join(HERE, "validation_sample.csv"), index=False)
print(f"wrote {len(df)} articles from {df.source.nunique()} sources")
