"""
Closing-line check: did the PrizePicks line move toward the model's side after the pick was made?

For each pick (earliest snapshot that priced it, same rule as grading), its line is compared with the
LAST line observed before puck drop. Observations come from two places:
  * every archived snapshot (data/snapshots/*.json): each run's line for every player + prop
  * the line tracker (data/lines/<date>.json), which records the board every 30 min through game hours

A move "toward" the model means the line rose on an Over pick or fell on an Under pick: the market
later agreed with the model. Baselines on the same picks: how often lines moved up (what a blind
Over would gain) and down (blind Under). The model only shows real signal if its toward-rate beats
the share of lines moving in its usual direction.

Only picks whose game has started are included, so the closing line is final. "cm" = minutes between
the last observation and puck drop: a closing line captured 6 hours early is weaker evidence.
"""
import glob
import json
import os

from common import LINES_DIR, SNAP_DIR, norm_name, parse_ts, utcnow


def _observations():
    """(game_date, pp_player_id, category) -> list of (timestamp, line)."""
    obs = {}
    for p in glob.glob(os.path.join(SNAP_DIR, "*.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        t = parse_ts(s["generated_at"])
        for pk in s.get("picks", []):
            if pk.get("pp_player_id") and pk.get("line") is not None:
                obs.setdefault((pk["game_date"], str(pk["pp_player_id"]), pk["category"]), []).append((t, float(pk["line"])))
    for p in glob.glob(os.path.join(LINES_DIR, "*.json")):
        date = os.path.basename(p)[:-5]
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for key, e in doc.get("lines", {}).items():
            pid, cat = key.split("|", 1)
            for line, first, last in e.get("segs", []):
                o = obs.setdefault((date, pid, cat), [])
                o.append((parse_ts(first), float(line)))
                o.append((parse_ts(last), float(line)))
    return obs


def _earliest_picks():
    first = {}
    for p in glob.glob(os.path.join(SNAP_DIR, "*.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        for pk in s.get("picks", []):
            if pk.get("skip_reason") or pk.get("side") is None or not pk.get("nhl_game_id") or not pk.get("nhl_id"):
                continue
            k = (pk["game_date"], str(pk["nhl_game_id"]), str(pk["nhl_id"]), pk["category"])
            if k not in first or s["generated_at"] < first[k][1]:
                first[k] = (pk, s["generated_at"])
    return first


def build_moves(cfg, excluded_keys=frozenset(), now=None):
    """Compact rows for the results page: d date, pl player, c prop, s side, p prob, m model mean,
    l0 pick line, lc closing line, cm minutes between last observation and puck drop, hd headline."""
    now = now or utcnow()
    obs = _observations()
    rows = []
    for (date, _, _, cat), (pk, made_at) in _earliest_picks().items():
        start = parse_ts(pk["start_time"])
        if start > now:
            continue
        if (date, norm_name(pk["player"]), cat) in excluded_keys:
            continue
        seen = [o for o in obs.get((date, str(pk["pp_player_id"]), cat), []) if o[0] <= start]
        if not seen:
            continue
        t_close, lc = max(seen, key=lambda o: o[0])
        adj = str(pk.get("adjusted_odds")).lower() == "true"
        hd = bool(cfg["props"].get(cat, {}).get("headline")) and not (adj and cfg.get("headline_requires_unadjusted"))
        rows.append({"d": date, "pl": pk["player"], "tm": pk["team"], "c": cat, "s": pk["side"],
                     "p": round(float(pk["prob"]), 4), "m": round(float(pk["model_mean"]), 3),
                     "l0": float(pk["line"]), "lc": lc, "cm": int((start - t_close).total_seconds() // 60),
                     "hd": 1 if hd else 0})
    return rows


def summarize(rows):
    """Printed in the Action log."""
    if not rows:
        return "no picks with a closing line yet"
    moved = [r for r in rows if r["lc"] != r["l0"]]
    toward = sum(1 for r in moved if (r["lc"] > r["l0"]) == (r["s"] == "over"))
    up = sum(1 for r in moved if r["lc"] > r["l0"])
    f = lambda a, b: f"{a / b:.1%}" if b else "-"
    return (f"n={len(rows)} picks with a closing line; {len(moved)} moved ({f(len(moved), len(rows))}). "
            f"Of moved: toward model {f(toward, len(moved))}, moved up {f(up, len(moved))}, "
            f"moved down {f(len(moved) - up, len(moved))}")
