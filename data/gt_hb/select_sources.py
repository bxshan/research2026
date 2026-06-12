"""
Build data/gt_hb/source_metadata.csv: one row per source in the GT parquet
corpus, with article count and publisher-level MBFC metadata joined from
metadata.db. Deterministic (no sampling). Costs $0.
"""
import os
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PARQUET_GLOB = os.path.join(ROOT, "data", "data_full", "nela_gt_full", "data", "*.parquet")
METADATA_DB  = os.path.join(ROOT, "data", "data_full", "nela_gt_full", "metadata.db")
OUT          = os.path.join(HERE, "source_metadata.csv")

con = duckdb.connect(METADATA_DB, read_only=True)
con.execute(f"""
    COPY (
        SELECT p.source,
               p.n_articles,
               s.label, s.bias, s.factuality, s.credibility,
               s.conspiracy, s.pseudosci, s.type, s.country
        FROM (
            SELECT source, COUNT(*) AS n_articles
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY source
        ) p
        LEFT JOIN sources s ON s.source = p.source
        ORDER BY p.n_articles DESC
    ) TO '{OUT}' (HEADER, DELIMITER ',');
""")
n = sum(1 for _ in open(OUT)) - 1
print(f"{n} sources written to {OUT}")
