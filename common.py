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
GRADED_PATH = path("data", "graded_results.csv")    # append-only
EXCLUDED_PATH = path("data", "excluded_picks.csv")  # hand-maintained via exclude_pick.py
PLAYER_MAP_PATH = path("data", "player_map.csv")    # PrizePicks player id -> NHL player id
ALIASES_PATH = path("data", "player_aliases.csv")   # manual overrides for the map
DOCS_DATA = path("docs", "data")

NHL_BASE = "https://api-web.nhle.com/v1"
PP_BOARD_URL = "https://partner-api.prizepicks.com/projections"
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
