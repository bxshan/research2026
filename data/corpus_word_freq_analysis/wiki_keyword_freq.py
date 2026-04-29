"""
wiki_keyword_freq.py
--------------------
Validates whether the Wikipedia high school corpus contains differential
keyword signals across school types.

Two modes:
  --mode individual   one bar series per school (default)
  --mode grouped      one bar series per community type, with ±1 SD error bars

Usage:
    python3 data/wiki_keyword_freq.py
    python3 data/wiki_keyword_freq.py --mode grouped
    python3 data/wiki_keyword_freq.py --mode grouped --out data/by_type.png
"""

import logging
import os
import sys
import time
import argparse
import random

import json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import pandas as pd

SEED = 2
random.seed(SEED)
np.random.seed(SEED)

# ── Schools ───────────────────────────────────────────────────────────────────
# (wikipedia_title, display_label, group)
SCHOOLS = [
    ("Phillips Exeter Academy",                                        "Phillips Exeter (elite private, NH)",      "Elite Private"),
    ("Groton School",                                                  "Groton (elite private, MA)",               "Elite Private"),
    ("Hotchkiss School",                                               "Hotchkiss (elite private, CT)",            "Elite Private"),
    ("Thomas Jefferson High School for Science and Technology",        "Thomas Jefferson (selective magnet, VA)",  "Selective / Magnet"),
    ("Stuyvesant High School",                                         "Stuyvesant (specialized, NYC)",            "Selective / Magnet"),
    ("Bronx High School of Science",                                   "Bronx Science (specialized, NYC)",         "Selective / Magnet"),
    ("Naperville Central High School",                                 "Naperville Central (suburban, IL)",        "Mainstream Suburban"),
    ("Cherry Hill High School East",                                   "Cherry Hill East (suburban, NJ)",          "Mainstream Suburban"),
    ("Westfield High School (New Jersey)",                             "Westfield (suburban, NJ)",                 "Mainstream Suburban"),
    ("Dunbar High School (Washington, D.C.)",                          "Dunbar (Title I urban, DC)",               "Title I Urban"),
    ("Frederick Douglass Academy",                                     "Frederick Douglass (Title I, NYC)",        "Title I Urban"),
    ("Crenshaw High School",                                           "Crenshaw (Title I urban, CA)",             "Title I Urban"),
    ("Hays High School",                                               "Hays HS (rural public, KS)",               "Rural Public"),
    ("Colby High School",                                              "Colby HS (rural public, KS)",              "Rural Public"),
]

# Group display order and colors (warm → cool)
GROUPS = [
    ("Elite Private",       "#c0392b"),
    ("Selective / Magnet",  "#e67e22"),
    ("Mainstream Suburban", "#f1c40f"),
    ("Title I Urban",       "#2980b9"),
    ("Rural Public",        "#1abc9c"),
]
GROUP_COLORS = dict(GROUPS)
GROUP_ORDER  = [g for g, _ in GROUPS]

# ── Keywords ──────────────────────────────────────────────────────────────────
# Removed zero/near-zero terms: underserved, poverty, need-based,
# free and reduced lunch, low-income, charter, minority
KEYWORDS = [
    "need-blind",
    "specialized",
    "magnet",
    "Title I",
    "selective",
    "competitive",
    "prestigious",
    "elite",
    "gifted",
    "honors",
    "diverse",
    "scholarship",
    "endowment",
    "tuition",
]


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_text_cache.json")

def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


# ── Wikipedia fetch ───────────────────────────────────────────────────────────
def fetch_wikipedia_text(title: str) -> str | None:
    """
    Fetch plain-text extract of a Wikipedia article by title.
    Returns None if the article does not exist or is empty.
    Retries up to 3 times with exponential backoff on 429.
    """
    url     = "https://en.wikipedia.org/w/api.php"
    params  = {"action": "query", "titles": title, "prop": "extracts",
                "explaintext": True, "format": "json", "redirects": 1}
    headers = {"User-Agent": "research2026-keyword-analysis/1.0 (academic research)"}
    for attempt in range(4):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            wait = 5 * (2 ** attempt)
            print(f"429 — waiting {wait}s ...", end=" ", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        if "missing" in page:
            return None
        return page.get("extract", "") or None
    raise requests.HTTPError(f"429 after retries: {title}")


# ── Frequency computation ─────────────────────────────────────────────────────
def keyword_freq(text: str) -> dict[str, float]:
    """Compute normalized keyword frequency per 1,000 tokens."""
    n         = len(text.split())
    text_low  = text.lower()
    if n == 0:
        return {kw: 0.0 for kw in KEYWORDS}
    return {kw: (text_low.count(kw.lower()) / n) * 1000 for kw in KEYWORDS}


# ── Shared plot helpers ───────────────────────────────────────────────────────
def sorted_keywords(freq_map: dict) -> list[str]:
    """Sort keywords by grand mean ascending (highest appears at top of chart)."""
    all_vals = {kw: [] for kw in KEYWORDS}
    for vals in freq_map.values():
        for kw, v in vals.items():
            all_vals[kw].append(v)
    return sorted(KEYWORDS, key=lambda kw: np.mean(all_vals[kw]))


def save_plot(fig, out_path: str):
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")


# ── Individual mode ───────────────────────────────────────────────────────────
def plot_individual(freq: dict[str, dict], out_path: str):
    """One color per school."""
    labels   = list(freq.keys())
    n        = len(labels)
    sorted_kws = sorted_keywords(freq)
    n_kws    = len(sorted_kws)
    colors   = ["#c0392b", "#e67e22", "#e8b84b", "#27ae60", "#2980b9",
                "#8e44ad", "#16a085", "#d35400", "#2c3e50", "#7f8c8d",
                "#c0392b", "#1abc9c", "#f39c12", "#2ecc71"][:n]
    bar_h    = 0.75 / n
    y_base   = np.arange(n_kws, dtype=float)

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(13, max(9, n_kws * 0.6)))
        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * bar_h
            ax.barh(y_base + offset,
                    [freq[label][kw] for kw in sorted_kws],
                    height=bar_h * 0.88, label=label,
                    color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        _apply_axes(ax, sorted_kws, "Keyword Signal by School — Wikipedia Articles")
        save_plot(fig, out_path)


# ── Grouped mode ──────────────────────────────────────────────────────────────
def plot_grouped(group_stats: dict, out_path: str):
    """One color per community type, ±1 SD error bars."""
    groups     = [g for g in GROUP_ORDER if g in group_stats]
    n          = len(groups)
    sorted_kws = sorted_keywords({g: group_stats[g]["mean"] for g in groups})
    n_kws      = len(sorted_kws)
    bar_h      = 0.75 / n
    y_base     = np.arange(n_kws, dtype=float)

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(13, max(10, n_kws * 0.6)))
        for i, group in enumerate(groups):
            offset = (i - n / 2 + 0.5) * bar_h
            means  = [group_stats[group]["mean"][kw] for kw in sorted_kws]
            stds   = [group_stats[group]["std"][kw]  for kw in sorted_kws]
            ax.barh(y_base + offset, means, height=bar_h * 0.88,
                    xerr=stds, label=f"{group} (n={group_stats[group]['n']})",
                    color=GROUP_COLORS[group], alpha=0.85,
                    edgecolor="white", linewidth=0.4,
                    error_kw={"elinewidth": 0.9, "capsize": 2,
                              "ecolor": "black", "alpha": 0.6})
        _apply_axes(ax, sorted_kws,
                    "Keyword Signal by Community Type — Wikipedia Articles\n"
                    "error bars = ±1 SD across schools in group")
        save_plot(fig, out_path)


def _apply_axes(ax, sorted_kws, title):
    ax.set_yticks(np.arange(len(sorted_kws)))
    ax.set_yticklabels(sorted_kws, fontsize=10)
    ax.set_xlabel("Frequency per 1,000 tokens", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(axis="x", alpha=0.35)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Wikipedia high school keyword frequency analysis")
    parser.add_argument("--mode", choices=["individual", "grouped"], default="individual")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(script_dir, f"wiki_keyword_freq_{args.mode}.png")
    csv_out = args.out.replace(".png", ".csv")

    # 1) Fetch articles (cache to avoid repeat 429s)
    print("=" * 60)
    print(f"  Fetching Wikipedia articles  [mode: {args.mode}]")
    print("=" * 60)
    cache = _load_cache()
    school_texts = {}   # label → text
    school_groups = {}  # label → group
    cache_dirty = False
    for wiki_title, label, group in SCHOOLS:
        if wiki_title in cache:
            text = cache[wiki_title]
            print(f"  {wiki_title} ... {len(text.split()):,} tokens [cached]")
        else:
            print(f"  {wiki_title} ...", end=" ", flush=True)
            try:
                text = fetch_wikipedia_text(wiki_title)
            except Exception as e:
                print(f"ERROR ({e}) — skipping")
                continue
            if text is None:
                print("NOT FOUND — skipping")
                continue
            print(f"{len(text.split()):,} tokens")
            cache[wiki_title] = text
            cache_dirty = True
            time.sleep(0.4)
        school_texts[label]  = text
        school_groups[label] = group
    if cache_dirty:
        _save_cache(cache)

    if len(school_texts) < 2:
        print("[error] Need at least 2 articles.")
        sys.exit(1)

    # 2) Compute frequencies
    freq = {label: keyword_freq(text) for label, text in school_texts.items()}

    if args.mode == "individual":
        # 3a) Save CSV
        rows = [{"school": label, "keyword": kw, "freq_per_1k": round(v, 4)}
                for label, kd in freq.items() for kw, v in kd.items()]
        pd.DataFrame(rows).pivot(index="keyword", columns="school",
                                 values="freq_per_1k").to_csv(csv_out)
        print(f"[data]  saved → {csv_out}")
        # 3b) Plot
        plot_individual(freq, args.out)

    else:  # grouped
        # Aggregate by group
        group_texts = {}
        for label, text in school_texts.items():
            g = school_groups[label]
            group_texts.setdefault(g, []).append(text)

        group_stats = {}
        for g, texts in group_texts.items():
            freqs = [keyword_freq(t) for t in texts]
            group_stats[g] = {
                "mean": {kw: np.mean([f[kw] for f in freqs]) for kw in KEYWORDS},
                "std":  {kw: np.std( [f[kw] for f in freqs]) for kw in KEYWORDS},
                "n":    len(texts),
            }
            print(f"  {g}: {len(texts)} articles")

        # 3a) Save CSV
        rows = [{"group": g, "keyword": kw,
                 "mean_per_1k": round(s["mean"][kw], 4),
                 "std_per_1k":  round(s["std"][kw],  4),
                 "n_schools":   s["n"]}
                for g, s in group_stats.items() for kw in KEYWORDS]
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"[data]  saved → {csv_out}")
        # 3b) Plot
        plot_grouped(group_stats, args.out)

        # 4) Signal summary
        print("\n[summary] Top differentiating keywords (variance of group means):")
        var = {kw: np.var([group_stats[g]["mean"][kw] for g in group_stats])
               for kw in KEYWORDS}
        for kw, v in sorted(var.items(), key=lambda x: -x[1])[:8]:
            vals = {g: round(group_stats[g]["mean"][kw], 3) for g in group_stats}
            print(f"  {kw:<22} var={v:.5f}  {vals}")


if __name__ == "__main__":
    main()
