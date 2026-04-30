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
import time

import requests

TARGET      = os.path.join(os.path.dirname(__file__), "..", "data_full", "wiki_hs_full")
OUTPUT_CSV  = os.path.join(TARGET, "wiki_hs_articles.csv")
TITLES_TXT  = os.path.join(TARGET, "wiki_hs_titles.txt")   # saved after step 1 for inspection
LOG_TXT     = os.path.join(TARGET, "wiki_hs_download_log.txt")
SECRETS     = os.path.join(os.path.dirname(__file__), "..", "..", "secrets", "wiki.env")
ROOT_CAT    = "High schools in the United States"
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
TARGET_N    = 10000
MIN_CHARS   = 400
BATCH       = 50
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


def main(use_cached_titles: bool = False):
    global _session
    os.makedirs(TARGET, exist_ok=True)
    _session = _get_session()

    if os.path.exists(OUTPUT_CSV):
        print(f"[skip]  {OUTPUT_CSV} already exists — delete it to re-download")
        return

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
        print(f"[step1] enumerating '{ROOT_CAT}' recursively ...")
        for cat in _subcategories(ROOT_CAT):
            if not _is_school_cat(cat):
                stats["categories_skipped"] += 1
                print(f"[step1] skip: {cat}", flush=True)
                continue
            all_titles += _collect_titles(cat)
            if len(all_titles) >= TARGET_N * 2:
                break
            time.sleep(1.0)

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
    print(f"\n[step2] fetching article text in batches of {BATCH} ...")
    articles: list[dict] = []
    t2_start = time.time()
    total_batches = (len(all_titles) + BATCH - 1) // BATCH

    for i in range(0, len(all_titles), BATCH):
        batch = all_titles[i:i + BATCH]
        try:
            texts = _batch_fetch_text(batch)
        except Exception as e:
            print(f"\n[step2] batch error ({e}) — skipping", flush=True)
            continue

        batch_titles_requested = len(batch)
        batch_titles_returned  = len(texts)
        stats["fetched"]          += batch_titles_requested
        stats["dropped_no_extract"] += batch_titles_requested - batch_titles_returned

        for title, text in texts.items():
            text = text.strip()
            text_low = text.lower()
            if len(text) < MIN_CHARS:
                stats["dropped_too_short"] += 1
                continue
            if "high school" not in text_low and "secondary school" not in text_low:
                stats["dropped_no_hs_text"] += 1
                continue
            stats["valid"] += 1
            articles.append({"title": title, "source": "wikipedia", "content": text})

        batches_done = i // BATCH + 1
        elapsed = time.time() - t2_start
        rate = elapsed / batches_done                        # s per batch
        remaining = (total_batches - batches_done) * rate
        eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s"
        print(f"[step2] {min(i+BATCH, len(all_titles)):,}/{len(all_titles):,} fetched  "
              f"valid={len(articles):,}  eta={eta_str}", end="\r", flush=True)
        if len(articles) >= TARGET_N:
            break
        time.sleep(0.5)

    print()
    articles = articles[:TARGET_N]
    print(f"[step2] {len(articles):,} articles collected")

    # ── 3) Save CSV ────────────────────────────────────────────────────────────
    print(f"\n[step3] saving → {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "source", "content"])
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
    args = parser.parse_args()
    main(use_cached_titles=args.use_cached_titles)
