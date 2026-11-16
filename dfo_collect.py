"""
Daily Faceoff projected lines + starting goalies, saved locally for a measurement study.

    python dfo_collect.py              # today's teams (run once a day, early afternoon Central)
    python dfo_collect.py --inspect    # also print the page structure (first run: paste it to Claude)

Why it's built this way
  * PERSONAL, LOCAL ONLY. Pages are saved under dfo_data/ (git-ignored), never committed or published:
    the repo and site are public, and this is Daily Faceoff's work.
  * Light touch: robots.txt is read on every run and any disallowed page is skipped (if robots.txt
    can't be read at all, nothing is fetched). Only teams playing today (NHL schedule) plus the
    starting-goalies page: at most ~33 requests a day, 2 seconds apart.
  * Measurement first: after 3-4 weeks the saved lines are compared with the lines teams actually
    played (shift charts) to see how much of the Time On Ice gain morning lines would capture.
"""
import argparse
import gzip
import json
import os
import re
import time
import urllib.robotparser
from datetime import datetime

import requests

from common import NHL_BASE, get_json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "dfo_data")
BASE = "https://www.dailyfaceoff.com"
UA = "nhl-prop-tracker/0.1 (personal use; one visit per team per day)"
SLUG = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres", "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks", "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars", "DET": "detroit-red-wings", "EDM": "edmonton-oilers",
    "FLA": "florida-panthers", "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
    "NSH": "nashville-predators", "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken", "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs", "UTA": "utah-mammoth",
    "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights", "WSH": "washington-capitals", "WPG": "winnipeg-jets",
}
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def teams_today(d):
    sched = get_json(f"{NHL_BASE}/schedule/{d}") or {}
    teams = set()
    for day in sched.get("gameWeek", []):
        if day.get("date") == d:
            for g in day.get("games", []):
                teams |= {g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]}
    return sorted(teams)


def robots():
    rp = urllib.robotparser.RobotFileParser()
    r = requests.get(BASE + "/robots.txt", headers={"User-Agent": UA}, timeout=30)
    if r.status_code in (401, 403):
        raise SystemExit("robots.txt refused (401/403): not collecting.")
    rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    return rp


def fetch(url):
    """-> (kind, payload): ('json', page data) when the page embeds its data, else ('html', text)."""
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html"}, timeout=45)
    r.raise_for_status()
    m = NEXT_DATA.search(r.text)
    if m:
        try:
            return "json", json.loads(m.group(1))
        except ValueError:
            pass
    return "html", r.text


def save(path, kind, payload):
    with gzip.open(path + (".json.gz" if kind == "json" else ".html.gz"), "wt", encoding="utf-8") as f:
        f.write(json.dumps(payload) if kind == "json" else payload)


def find_player_lists(obj, path="", out=None):
    """Lists of dicts that look like player entries (for --inspect)."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_player_lists(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj[:5]) and any(
                any("name" in k.lower() for k in x) for x in obj[:5]):
            out.append((path, obj))
        for i, v in enumerate(obj[:3]):
            find_player_lists(v, f"{path}[{i}]", out)
    return out


def inspect(name, kind, payload):
    print(f"\n===== {name}: {kind} =====")
    if kind != "json":
        print("(no embedded page data found; first 400 characters)\n", payload[:400])
        return
    pp = (payload.get("props") or {}).get("pageProps") or {}
    print("pageProps keys:", sorted(pp.keys())[:40])
    for path, lst in find_player_lists(payload)[:6]:
        print(f"\n  list at {path}: {len(lst)} items; keys: {sorted(lst[0].keys())}")
        for x in lst[:4]:
            print("   ", json.dumps(x)[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()

    today = datetime.now().date().isoformat()
    day_dir = os.path.join(OUT_DIR, today)
    os.makedirs(day_dir, exist_ok=True)
    stamp = datetime.now().strftime("%H%M")
    rp = robots()
    teams = teams_today(today)
    print(f"{today}: {len(teams)} teams playing")
    pages = [("starting-goalies", f"{BASE}/starting-goalies")] + \
            [(t, f"{BASE}/teams/{SLUG[t]}/line-combinations") for t in teams if t in SLUG]
    missing = [t for t in teams if t not in SLUG]
    if missing:
        print("no Daily Faceoff page name for:", missing)
    saved, skipped, failed = 0, [], []
    for i, (name, url) in enumerate(pages):
        if not rp.can_fetch(UA, url):
            skipped.append(name)
            continue
        try:
            kind, payload = fetch(url)
        except Exception as e:
            failed.append(f"{name} ({e})")
            continue
        save(os.path.join(day_dir, f"{name}_{stamp}"), kind, payload)
        saved += 1
        if a.inspect and name in ("starting-goalies", teams[0] if teams else None):
            inspect(name, kind, payload)
        if i < len(pages) - 1:
            time.sleep(a.sleep)
    print(f"\nsaved {saved} pages to {os.path.relpath(day_dir, HERE)}"
          + (f"; robots.txt disallowed {skipped}" if skipped else "")
          + (f"; failed {failed}" if failed else ""))
    try:   # refresh the local tables (dfo_data/lines.csv, goalies.csv)
        import dfo_parse
        dfo_parse.main(verbose=a.inspect)
    except Exception as e:
        print(f"parse step failed (pages are still saved): {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
