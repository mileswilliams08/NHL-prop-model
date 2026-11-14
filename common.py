"""Shared paths, config, HTTP and name helpers."""
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))


def path(*parts):
    return os.path.join(ROOT, *parts)


CONFIG_PATH = path("config.json")
HIST_DIR = path("data", "history")
RAW_DIR = path("data", "raw", "boxscores")          # local cache, gitignored
SNAP_DIR = path("data", "snapshots")                # one file per prediction run
LINES_DIR = path("data", "lines")                   # line history per slate date (lines.py)
GRADED_PATH = path("data", "graded_results.csv")    # append-only
EXCLUDED_PATH = path("data", "excluded_picks.csv")  # hand-maintained via exclude_pick.py
PLAYER_MAP_PATH = path("data", "player_map.csv")    # PrizePicks player id -> NHL player id
ALIASES_PATH = path("data", "player_aliases.csv")   # manual overrides for the map
DOCS_DATA = path("docs", "data")

# Each book is tracked on its own: separate snapshots, graded log, player map and site files, so
# PrizePicks and Underdog numbers never mix. Everything else (history, model) is shared.
UD_DIR = path("data", "underdog")
BOOKS = {
    "prizepicks": {"key": "prizepicks", "label": "PrizePicks", "snap_dir": SNAP_DIR, "graded": GRADED_PATH,
                   "player_map": PLAYER_MAP_PATH, "props_json": "props.json", "results_json": "results.json"},
    "underdog": {"key": "underdog", "label": "Underdog", "snap_dir": os.path.join(UD_DIR, "snapshots"),
                 "graded": os.path.join(UD_DIR, "graded_results.csv"),
                 "player_map": os.path.join(UD_DIR, "player_map.csv"),
                 "props_json": "props_underdog.json", "results_json": "results_underdog.json"},
}

NHL_BASE = "https://api-web.nhle.com/v1"
PP_BOARD_URL = "https://partner-api.prizepicks.com/projections"
UD_BOARD_URL = "https://api.underdogfantasy.com/v1/over_under_lines"
USER_AGENT = "nhl-prop-model/1.0 (read-only research project; github.com/mileswilliams08)"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def utcnow():
    return datetime.now(timezone.utc)


def parse_ts(s):
    """ISO timestamp (with Z or an offset, optional milliseconds) -> aware datetime."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def get_json(url, params=None, tries=4, timeout=30):
    """GET JSON with retries. 404 -> None. Raises after the last failed try."""
    err = None
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.json()
        except Exception as e:  # timeouts, resets, 5xx
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}: {err}")


def norm_name(s):
    """'Tim Stützle' -> 'timstutzle'. Accents stripped, lowercase, letters only."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def nhl_team(pp_team, cfg):
    """PrizePicks team abbreviation -> NHL API abbreviation (LA -> LAK etc.)."""
    t = (pp_team or "").strip().upper()
    return cfg.get("team_map", {}).get(t, t)


def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)
