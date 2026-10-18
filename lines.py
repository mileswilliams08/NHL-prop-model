"""
Line-movement tracker: records every standard PrizePicks line for enabled props, so each pick's line
can be compared with the LAST line posted before puck drop (the "closing line").

    python lines.py            # fetch the board once and update data/lines/<slate date>.json

Runs every 30 minutes through game hours (.github/workflows/lines.yml). The daily run's snapshots
(data/snapshots/) also count as line observations, so predict.py doesn't write here.
It never prices anything and never touches the site's props/results data.

Only lines for games that haven't started are recorded, so a run during games can't record a live
line. Storage per slate date (US Eastern date from start_time), one entry per player + prop:

    {"lines": {"<pp_player_id>|<category>": {"player", "team", "cat", "start",
                                              "segs": [[line, first_seen, last_seen], ...]}}}

A new segment starts whenever the line changes; otherwise last_seen is just moved forward.
"""
import json
import os
from collections import Counter
from datetime import timezone

from common import LINES_DIR, load_config, parse_ts, utcnow, write_json

TS = "%Y-%m-%dT%H:%MZ"


def _load(date):
    p = os.path.join(LINES_DIR, f"{date}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"lines": {}}


def record(rows, cfg, now=None):
    """rows: prizepicks.parse_board output. Returns (keys seen, line changes) for logging."""
    now = (now or utcnow()).astimezone(timezone.utc)
    stamp = now.strftime(TS)
    enabled = {c for c, v in cfg["props"].items() if v["enabled"]}
    keep = [r for r in rows if r["category"] in enabled and r["odds_type"] == "standard" and not r["combo"]
            and r["status"] == "pre_game" and not r["is_live"] and r["start_time"] and r["pp_player_id"]
            and parse_ts(r["start_time"]) > now]
    by_date = {}
    for r in keep:
        by_date.setdefault(r["start_time"][:10], []).append(r)

    seen, moves = 0, 0
    for date, rs in by_date.items():
        doc = _load(date)
        lines = doc["lines"]
        for r in rs:
            key = f"{r['pp_player_id']}|{r['category']}"
            e = lines.get(key)
            if e is None:
                e = lines[key] = {"player": r["player"], "team": r["team"], "cat": r["category"], "segs": []}
            e["start"] = r["start_time"]
            segs = e["segs"]
            if segs and float(segs[-1][0]) == float(r["line"]):
                segs[-1][2] = stamp
            else:
                if segs:
                    moves += 1
                segs.append([r["line"], stamp, stamp])
            seen += 1
        doc["updated_at"] = stamp
        write_json(os.path.join(LINES_DIR, f"{date}.json"), doc)
    return seen, moves


def main():
    import prizepicks
    cfg = load_config()
    try:
        rows = prizepicks.parse_board(prizepicks.fetch_board(cfg["league_id"]), cfg)
    except Exception as e:
        print(f"PrizePicks fetch failed: {e} (nothing recorded)")
        return 0
    seen, moves = record(rows, cfg)
    cats = Counter(r["category"] for r in rows if r["odds_type"] == "standard")
    print(f"board: {len(rows)} projections; recorded {seen} pre-game standard lines, {moves} line changes "
          f"since the last observation; standard by prop: {dict(cats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
