"""
DownloadWikiHighSchoolsALL.py
------------------------------
Fetches ~10,000 Wikipedia high school articles via the Wikipedia API and
saves them to data_full/wiki_hs_full/wiki_hs_articles.csv.

Article titles are collected by enumerating subcategories of
"High schools in the United States", then article text is batch-fetched
50 titles at a time.

Run from the project root:
    python3 data/full_download_scripts/DownloadWikiHighSchoolsALL.py
    python3 data/full_download_scripts/DownloadWikiHighSchoolsALL.py --use-cached-titles

Output:
    data/data_full/wiki_hs_full/wiki_hs_articles.csv
    columns: title, source, content
"""

import csv
import argparse
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TARGET      = os.path.join(os.path.dirname(__file__), "..", "data_full", "wiki_hs_full")
OUTPUT_CSV  = os.path.join(TARGET, "wiki_hs_articles.csv")
TITLES_TXT  = os.path.join(TARGET, "wiki_hs_titles.txt")   # saved after step 1 for inspection
LOG_TXT     = os.path.join(TARGET, "wiki_hs_download_log.txt")
SECRETS     = os.path.join(os.path.dirname(__file__), "..", "..", "secrets", "wiki.env")
ROOT_CATS   = [
    "High schools in the United States",
    "Secondary schools in the United States",
    "Preparatory schools in the United States",
    "Online high schools in the United States",
    "University-affiliated schools in the United States",
    "Magnet high schools in the United States",
    "Charter high schools in the United States",
]
API         = "https://en.wikipedia.org/w/api.php"
HEADERS     = {"User-Agent": "research2026-wiki-download/1.0 (academic research)"}


def _load_secrets() -> dict:
    """Load KEY=VALUE pairs from secrets/wiki.env."""
    secrets = {}
    path = os.path.normpath(SECRETS)
    if not os.path.exists(path):
        return secrets
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                secrets[k.strip()] = v.strip()
    return secrets


def _get_session() -> requests.Session:
    """
    Return an authenticated requests Session if wiki.env has credentials,
    otherwise return an anonymous session.
    Authenticates via MediaWiki clientlogin API.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    secrets = _load_secrets()
    user = secrets.get("WIKI_BOT_USER", "")
    passwd = secrets.get("WIKI_BOT_PASS", "")

    if not user or not passwd:
        print("[auth]  no credentials found — using anonymous session")
        return session

    # Step 1: get login token
    r = session.get(API, params={
        "action": "query", "meta": "tokens", "type": "login", "format": "json"
    }, timeout=15)
    token = r.json()["query"]["tokens"]["logintoken"]

    # Step 2: login (bot passwords use action=login, not clientlogin)
    r = session.post(API, data={
        "action": "login", "lgname": user, "lgpassword": passwd,
        "lgtoken": token, "format": "json"
    }, timeout=15)
    result = r.json().get("login", {})
    status = result.get("result", "")

    if status == "Success":
        print(f"[auth]  logged in as {result.get('lgusername', user)}")
    else:
        print(f"[auth]  login failed ({status}: {result.get('reason', '')}) — using anonymous session")

    return session
TARGET_N    = float("inf")  # collect all valid articles found
MIN_CHARS   = 400
BATCH       = 20   # Wikipedia extracts API hard limit per call
WORKERS     = 5    # parallel batch fetchers
SEED        = 2

random.seed(SEED)


_session: requests.Session | None = None


def _api_get(params: dict, timeout: int = 15) -> dict:
    """GET the Wikipedia API with exponential backoff on 429."""
    for attempt in range(6):
        r = (_session or requests).get(API, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 429:
            wait = 15 * (2 ** attempt)
            print(f"\n[rate]  429 — waiting {wait}s ...", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("429 after 6 retries — aborting")


def _category_pages(category: str) -> list[str]:
    """Return article titles (not subcategories) from a Wikipedia category."""
    titles, params = [], {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": "max",
        "cmtype": "page", "format": "json",
    }
    while True:
        data = _api_get(params)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
        time.sleep(1.0)
    return titles


def _subcategories(category: str) -> list[str]:
    """Return subcategory names (without 'Category:' prefix)."""
    subcats, params = [], {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": "max",
        "cmtype": "subcat", "format": "json",
    }
    while True:
        data = _api_get(params)
        subcats += [m["title"].removeprefix("Category:") for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
        time.sleep(1.0)
    return subcats


def _batch_fetch_text(titles: list[str]) -> dict[str, str]:
    """Fetch plain-text extracts for up to 50 titles, following API continuation."""
    params = {
        "action": "query", "titles": "|".join(titles),
        "prop": "extracts", "explaintext": True,
        "exsectionformat": "wiki",   # keep == Section == markers so we can strip by name
        "exlimit": "max",
        "format": "json", "redirects": 1,
    }
    result = {}
    while True:
        data = _api_get(params, timeout=30)
        for page in data["query"]["pages"].values():
            if "missing" not in page and page.get("extract"):
                result[page["title"]] = page["extract"]
        if "continue" not in data:
            break
        params.update(data["continue"])
    return result


def _is_valid_output_csv(path: str) -> bool:
    """Quick sanity check that output CSV has the expected 3-column schema."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return header == ["title", "source", "content"]
    except Exception:
        return False


def main(use_cached_titles: bool = False, force: bool = False):
    global _session
    os.makedirs(TARGET, exist_ok=True)
    _session = _get_session()

    if os.path.exists(OUTPUT_CSV):
        if force:
            print(f"[overwrite] {OUTPUT_CSV} exists — rebuilding due to --force")
        elif _is_valid_output_csv(OUTPUT_CSV):
            print(f"[skip]  {OUTPUT_CSV} already exists — use --force to re-download")
            return
        else:
            print(f"[warn]  existing CSV looks malformed — rebuilding: {OUTPUT_CSV}")

    # Only recurse into subcategories that explicitly mention high/secondary school
    # and don't represent people/alumni lists
    REQUIRE = ["high school", "secondary school"]
    EXCLUDE = ["alumni", "people by", "players", "athletes"]

    def _is_school_cat(name: str) -> bool:
        low = name.lower()
        if any(p in low for p in EXCLUDE):
            return False
        return any(p in low for p in REQUIRE)

    stats = {
        "categories_visited":  0,
        "categories_skipped":  0,
        "titles_raw":          0,
        "titles_after_dedup":  0,
        "fetched":             0,
        "dropped_no_extract":  0,
        "dropped_too_short":   0,
        "dropped_no_hs_text":  0,
        "valid":               0,
    }

    visited: set[str] = set()

    def _collect_titles(category: str, depth: int = 0) -> list[str]:
        if category in visited or len(all_titles) >= TARGET_N * 2:
            return []
        visited.add(category)
        stats["categories_visited"] += 1

        indent = "  " * depth
        titles = _category_pages(category)
        print(f"[step1] {indent}{category}: {len(titles)} articles  "
              f"total={len(all_titles) + len(titles):,}", flush=True)
        time.sleep(1.0)

        for sub in _subcategories(category):
            if _is_school_cat(sub) and len(all_titles) < TARGET_N * 2:
                titles += _collect_titles(sub, depth + 1)
                time.sleep(0.5)
            elif not _is_school_cat(sub):
                stats["categories_skipped"] += 1

        return titles

    # ── 1) Collect or load article titles ───────────────────────────────────────
    all_titles: list[str] = []
    if use_cached_titles:
        if not os.path.exists(TITLES_TXT):
            raise FileNotFoundError(
                f"--use-cached-titles requested, but {TITLES_TXT} does not exist"
            )
        with open(TITLES_TXT, encoding="utf-8") as f:
            all_titles = [line.strip() for line in f if line.strip()]
        stats["titles_raw"] = len(all_titles)
        all_titles = list(dict.fromkeys(all_titles))
        stats["titles_after_dedup"] = len(all_titles)
        print(f"[step1] loaded cached title list → {TITLES_TXT} ({len(all_titles):,} titles)")
    else:
        print(f"[step1] enumerating {len(ROOT_CATS)} seed categories ...")
        for root in ROOT_CATS:
            # Always collect direct pages from seed categories
            all_titles += _collect_titles(root)
            # Recurse into subcategories, filtering by _is_school_cat
            for cat in _subcategories(root):
                if not _is_school_cat(cat):
                    stats["categories_skipped"] += 1
                    continue
                all_titles += _collect_titles(cat)
                if len(all_titles) >= TARGET_N * 2:
                    break
                time.sleep(1.0)
            if len(all_titles) >= TARGET_N * 2:
                break

        stats["titles_raw"] = len(all_titles)
        random.shuffle(all_titles)
        all_titles = list(dict.fromkeys(all_titles))   # deduplicate, preserve shuffle
        stats["titles_after_dedup"] = len(all_titles)
        print(f"[step1] {stats['titles_raw']:,} raw titles  →  "
              f"{stats['titles_after_dedup']:,} after dedup  "
              f"(removed {stats['titles_raw'] - stats['titles_after_dedup']:,} duplicates)")

        with open(TITLES_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(all_titles))
        print(f"[step1] title list saved → {TITLES_TXT}")

    # ── 2) Batch-fetch article text ────────────────────────────────────────────
    STRIP_SECTIONS = {
        "notable alumni", "alumni", "references", "see also",
        "external links", "related articles", "notes", "further reading",
        "sources", "footnotes",
    }

    # Title-level exclusions: non-school article types
    NON_SCHOOL_TITLE = [
        "conference", " league", "association", "interscholastic",
        "championship", "tournament", " bowl", " invitational",
    ]

    # Lede-level biography signals — person articles almost always have one
    BIO_PATTERNS = [
        "(born ", ", born ", " was born",
        " is an american ", " was an american ",
        " is a british ", " was a british ",
        " is a canadian ", " was a canadian ",
        " is an english ", " is an australian ", " was an australian ",
        " is an irish ", " was an irish ",
        " is a scottish ", " is a welsh ",
        " is a south african ", " is a nigerian ", " is a ghanaian ",
        " is an indian ", " is a filipino ",
        " is a jamaican ", " is a new zealand ",
        " is an american football ", " is an american basketball ",
        " is an american baseball ", " is an american soccer ",
    ]

    def _process_batch(batch: list[str]) -> tuple[list[dict], dict]:
        """Fetch and filter one batch; returns (valid_articles, drop_counts)."""
        drops = {"no_extract": 0, "too_short": 0, "no_hs": 0, "bio": 0, "non_school": 0}
        try:
            texts = _batch_fetch_text(batch)
        except Exception:
            drops["no_extract"] += len(batch)
            return [], drops

        drops["no_extract"] += len(batch) - len(texts)
        result = []

        for title, text in texts.items():
            title_low = title.lower()

            # Year-prefixed titles are sports events ("2016 NY state ... championship")
            if re.match(r'^\d{4}\s', title):
                drops["non_school"] += 1
                continue
            if any(k in title_low for k in NON_SCHOOL_TITLE):
                drops["non_school"] += 1
                continue

            parts   = re.split(r'\n==+[^\n]+==+', text)
            headers = re.findall(r'\n==+[^\n]+==+', text)
            kept = [parts[0]]
            for j, header in enumerate(headers):
                body     = parts[j + 1] if j + 1 < len(parts) else ""
                sec_name = re.sub(r'=', '', header).strip().lower()
                if not any(s in sec_name for s in STRIP_SECTIONS):
                    kept.append(body)

            # Flatten to single line; replace double quotes to avoid CSV escaping issues
            text = re.sub(r'\s+', ' ', "".join(kept)).strip()
            text = text.replace('"', "'")

            if len(text) < MIN_CHARS:
                drops["too_short"] += 1
                continue

            lede_low = text.lower()[:600]

            # Require HS mention in title or lede
            if ("high school" not in title_low and "secondary school" not in title_low
                    and "high school" not in lede_low and "secondary school" not in lede_low):
                drops["no_hs"] += 1
                continue

            # Exclude biography pages
            if any(p in lede_low for p in BIO_PATTERNS):
                drops["bio"] += 1
                continue

            result.append({"title": title, "source": "wikipedia", "content": text})
        return result, drops

    print(f"\n[step2] fetching article text — batch={BATCH}, workers={WORKERS} ...")
    articles: list[dict] = []
    lock      = threading.Lock()
    done_count = 0
    t2_start  = time.time()
    batches   = [all_titles[i:i + BATCH] for i in range(0, len(all_titles), BATCH)]
    total_batches = len(batches)
    stop_flag = threading.Event()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_batch, b): b for b in batches}
        for fut in as_completed(futures):
            if stop_flag.is_set():
                fut.cancel()
                continue
            valid, drops = fut.result()
            with lock:
                done_count += 1
                n_fetched = done_count * BATCH
                stats["fetched"]            += len(futures[fut])
                stats["dropped_no_extract"] += drops["no_extract"]
                stats["dropped_too_short"]  += drops["too_short"]
                stats["dropped_no_hs_text"] += drops["no_hs"] + drops["bio"] + drops["non_school"]
                for art in valid:
                    if len(articles) < TARGET_N:
                        articles.append(art)
                        stats["valid"] += 1
                elapsed   = time.time() - t2_start
                rate      = elapsed / done_count
                remaining = (total_batches - done_count) * rate
                eta_str   = f"{int(remaining//60)}m{int(remaining%60):02d}s"
                print(f"[step2] {min(n_fetched, len(all_titles)):,}/{len(all_titles):,} fetched  "
                      f"valid={len(articles):,}  eta={eta_str}", end="\r", flush=True)
                if len(articles) >= TARGET_N:
                    stop_flag.set()

    print()
    articles = articles[:int(TARGET_N)] if TARGET_N != float("inf") else articles
    print(f"[step2] {len(articles):,} articles collected")

    # ── 3) Save CSV ────────────────────────────────────────────────────────────
    print(f"\n[step3] saving → {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "source", "content"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(articles)

    print(f"[done]  {len(articles):,} articles saved to {OUTPUT_CSV}")

    elapsed_total = time.time() - t2_start
    log_lines = [
        f"=== Wikipedia High School Download Log ===",
        f"",
        f"[step1] category enumeration",
        f"  source             : {'cached title file' if use_cached_titles else 'category BFS'}",
        f"  categories visited : {stats['categories_visited']:,}",
        f"  categories skipped : {stats['categories_skipped']:,}",
        f"  titles raw         : {stats['titles_raw']:,}",
        f"  titles after dedup : {stats['titles_after_dedup']:,}",
        f"  duplicates removed : {stats['titles_raw'] - stats['titles_after_dedup']:,}",
        f"",
        f"[step2] text fetching",
        f"  titles fetched     : {stats['fetched']:,}",
        f"  dropped no extract : {stats['dropped_no_extract']:,}",
        f"  dropped too short  : {stats['dropped_too_short']:,}  (< {MIN_CHARS} chars)",
        f"  dropped no hs text : {stats['dropped_no_hs_text']:,}  (no 'high school' in content)",
        f"  valid              : {stats['valid']:,}",
        f"",
        f"[output] {OUTPUT_CSV}",
        f"  total articles     : {len(articles):,}",
        f"  elapsed (step2)    : {int(elapsed_total//60)}m{int(elapsed_total%60):02d}s",
    ]
    log_text = "\n".join(log_lines)
    print("\n" + log_text)
    with open(LOG_TXT, "w", encoding="utf-8") as f:
        f.write(log_text + "\n")
    print(f"\n[done]  log saved → {LOG_TXT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia high school article extracts.")
    parser.add_argument(
        "--use-cached-titles",
        action="store_true",
        help=f"Skip category BFS and use cached titles from {TITLES_TXT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSV and re-run download",
    )
    args = parser.parse_args()
    main(use_cached_titles=args.use_cached_titles, force=args.force)
