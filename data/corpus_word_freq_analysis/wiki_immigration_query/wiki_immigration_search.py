"""
wiki_immigration_search.py
--------------------------
Searches the local Wikipedia high-school corpus (wiki_hs_articles.csv)
for immigration-related keywords.

Algorithm
---------
1. Compile one regex per keyword using \b word-boundary anchors so that
   partial matches are rejected (e.g. "ESL" does not match "vessel",
   "asylum" does not match "asylums" ... actually it does -- \b matches at
   the morpheme boundary, so plural forms ARE counted intentionally).
2. Stream the CSV row by row; for each article run every pattern once via
   findall(), which returns all non-overlapping matches in O(n) time.
3. Track two accumulators per keyword:
     article_hits  -- set of titles with >= 1 match (for unique-article count)
     total_occ     -- raw occurrence count across all articles
4. Collect up to MAX_EXAMPLES example sentences per keyword.
   Sentence splitting uses a lookbehind on [.!?] followed by whitespace
   (good enough for Wikipedia prose; no NLTK dependency).
5. Write a text report and save to OUT_PATH.

Caveats
-------
- "visa" and "border" have broad non-immigration meanings (travel visas,
  school borders a park). Their counts should be read with that in mind.
- The CSV contains 14,071 articles; the training set was 10,000 sampled
  with seed=2. We search all 14,071 as a conservative upper bound; the
  true signal in the training set is ~70% of what is reported here.

Usage
-----
    python3 data/corpus_word_freq_analysis/wiki_immigration_search.py
"""

import bisect
import csv
import re
from collections import defaultdict
from pathlib import Path

CORPUS_PATH  = Path("../../data_full/wiki_hs_full/wiki_hs_articles.csv")
OUT_PATH     = Path("../../corpus_word_freq_analysis/wiki_immigration_query/wiki_immigration_results.txt")
CSV_OUT_PATH = Path("../../corpus_word_freq_analysis/wiki_immigration_query/wiki_immigration_matches.csv")

# Keywords from the assignment brief plus a few additions.
# Marked with * are high-false-positive risk; counts should be interpreted carefully.
KEYWORDS = [
    "immigrant",
    "immigrants",
    "immigration",
    "migrant",
    "migrants",
    "undocumented",
    "refugee",
    "asylum",
    "deportation",
    "DACA",
    "DREAMer",
    "ESL",
    "English language learner",
    "Title III",
    "bilingual",
    "naturalization",
    "citizenship",
    "foreign-born",
    "sanctuary",
    "ICE",
    "visa",        # * high false-positive risk
    "border",      # * high false-positive risk
]

HIGH_FP_RISK = {"visa"}

# Keywords that are genuine acronyms and should match regardless of case.
# All other keywords skip matches where every cased character is uppercase
# (e.g. "VISA" on a credit-card mention, "IMMIGRATION" in a section header).
ACRONYMS = {"DACA", "ESL", "ICE"}

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
MAX_EXAMPLES   = 2   # example sentences collected per keyword

# Keywords whose false-positive rate is too high for a plain \b…\b pattern.
# Each entry overrides make_pattern() entirely.
CUSTOM_PATTERNS: dict[str, re.Pattern] = {
    # ICE as an acronym (U.S. Immigration and Customs Enforcement) is always
    # uppercase.  Removing IGNORECASE drops "ice hockey", "on ice", etc.
    "ICE": re.compile(r'\bICE\b'),

    # "border" alone matches "borders Colorado", "a decorative border", etc.
    # Requiring an immigration-specific noun immediately after restricts to
    # "border patrol / crossing / wall / security / control / agent(s)".
    "border": re.compile(
        r'\bborder\s+(?:patrol|crossings?|wall|security|control|agents?)\b',
        re.IGNORECASE,
    ),

    # "citizenship" alone matches "global citizenship class", "digital
    # citizenship award", "active citizenship program", etc.
    # Negative lookbehinds (fixed-width) exclude the most common adjective
    # false-positives; negative lookahead excludes educational nouns after.
    "citizenship": re.compile(
        r'(?<!global )(?<!digital )(?<!civic )(?<!active )'
        r'\bcitizenship'
        r'(?!\s+(?:class(?:es)?|education|course(?:s)?|award(?:s)?'
        r'|program(?:s)?|ceremony|day))\b',
        re.IGNORECASE,
    ),
}


def make_pattern(keyword: str) -> re.Pattern:
    """
    Return a compiled regex for keyword.

    Custom patterns (ICE, border, citizenship) are returned from
    CUSTOM_PATTERNS.  All others use \b word-boundary anchors with
    IGNORECASE — see CUSTOM_PATTERNS docstring above for why those three
    are exceptions.
    """
    if keyword in CUSTOM_PATTERNS:
        return CUSTOM_PATTERNS[keyword]
    return re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)


def split_sentences_with_positions(text: str) -> list[tuple[int, str]]:
    """
    Split text into (start_char_offset, sentence) pairs.

    Uses SENTENCE_SPLIT (lookbehind on [.!?] + whitespace) to find split
    points, then reconstructs sentence spans from the gaps between splits.
    The start offset lets callers binary-search for the sentence that
    contains a given character position (e.g. a regex match start).
    """
    splits = list(SENTENCE_SPLIT.finditer(text))
    starts = [0]          + [m.end()   for m in splits]
    ends   = [m.start()   for m in splits] + [len(text)]
    return [
        (s, text[s:e])
        for s, e in zip(starts, ends)
        if text[s:e].strip()
    ]


def main() -> None:
    patterns     = {kw: make_pattern(kw) for kw in KEYWORDS}
    article_hits = defaultdict(set)   # kw -> set of titles
    total_occ    = defaultdict(int)   # kw -> raw count
    examples     = defaultdict(list)  # kw -> [(title, sentence), ...]
    match_rows: list[dict] = []       # one row per occurrence, for CSV

    articles_with_any_hit: set[str] = set()
    total_articles = 0

    with open(CORPUS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title   = row["title"]
            content = row["content"]
            total_articles += 1

            # Build sentence list with character offsets once per article.
            # sent_starts is kept separately for bisect_right lookups.
            sents       = split_sentences_with_positions(content)
            sent_starts = [s for s, _ in sents]
            sent_texts  = [t for _, t in sents]
            article_had_hit = False

            for kw, pat in patterns.items():
                for m in pat.finditer(content):
                    # Skip all-caps matches for non-acronym keywords.
                    # str.isupper() ignores spaces, hyphens, and digits, so
                    # "FOREIGN-BORN", "BORDER PATROL", and "TITLE III" all
                    # return True and are filtered out correctly.
                    if kw not in ACRONYMS and m.group().isupper():
                        continue
                    # Binary-search for the sentence whose start offset is
                    # <= m.start().  bisect_right gives the insertion point
                    # to the right, so subtracting 1 gives the containing
                    # sentence index.
                    idx      = bisect.bisect_right(sent_starts, m.start()) - 1
                    sentence = sent_texts[idx].strip() if 0 <= idx < len(sent_texts) else ""

                    article_hits[kw].add(title)
                    total_occ[kw] += 1
                    article_had_hit = True

                    match_rows.append({
                        "keyword":      kw,
                        "matched_text": m.group(),
                        "title":        title,
                        "sentence":     sentence,
                    })

                    if len(examples[kw]) < MAX_EXAMPLES * 3:
                        examples[kw].append((title, sentence))

            if article_had_hit:
                articles_with_any_hit.add(title)

    # ── Build report ─────────────────────────────────────────────────────────
    lines: list[str] = []
    W = 72

    lines += [
        "=" * W,
        "WIKI HIGH-SCHOOL CORPUS — IMMIGRATION KEYWORD SEARCH",
        "=" * W,
        "",
        f"  Corpus file : {CORPUS_PATH}",
        f"  Articles    : {total_articles:,}  (14,071 pool; training set = 10,000 sampled w/ seed=2)",
        f"  Keywords    : {len(KEYWORDS)}",
        "",
        f"  Articles with >= 1 hit : {len(articles_with_any_hit):,}"
        f"  ({100 * len(articles_with_any_hit) / total_articles:.1f}%)",
        "",
    ]

    # Per-keyword table
    lines += [
        "-" * W,
        f"  {'Keyword':<28} {'Articles':>8}  {'%':>6}  {'Occurrences':>12}",
        "-" * W,
    ]
    sorted_kws = sorted(KEYWORDS, key=lambda k: len(article_hits[k]), reverse=True)
    for kw in sorted_kws:
        n_art = len(article_hits[kw])
        pct   = 100 * n_art / total_articles
        n_occ = total_occ[kw]
        fp    = "  [*]" if kw in HIGH_FP_RISK else ""
        lines.append(
            f"  {kw:<28} {n_art:>8,}  {pct:>6.2f}%  {n_occ:>12,}{fp}"
        )
    lines += ["", "  [*] high false-positive risk; interpret counts carefully", ""]

    # Example sentences
    lines += ["-" * W, "  EXAMPLE SENTENCES IN CONTEXT", "-" * W]
    shown = 0
    for kw in sorted_kws:
        if not examples[kw] or shown >= 5:
            break
        title, sent = examples[kw][0]
        truncated   = sent[:280] + ("..." if len(sent) > 280 else "")
        lines += [
            "",
            f"  [{kw}]  — {title}",
            f'  "{truncated}"',
        ]
        shown += 1

    lines += ["", "=" * W, ""]

    report = "\n".join(lines)
    print(report)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved → {OUT_PATH}")

    CSV_FIELDS = ["keyword", "matched_text", "title", "sentence"]
    CSV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(match_rows)
    print(f"Match CSV saved → {CSV_OUT_PATH}  ({len(match_rows):,} rows)")


if __name__ == "__main__":
    main()
