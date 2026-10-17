"""
PrizePicks NHL board from partner-api.prizepicks.com (JSON:API).

Each projection's player lives in payload["included"] (type "new_player"), linked through
relationships.new_player. We keep every field needed to decide later whether a line is a
normal two-sided line:

  odds_type      "standard" | "goblin" | "demon"   (goblin/demon are Over-only -> never modeled or graded)
  adjusted_odds  True on ~60% of standard lines too; meaning unconfirmed, so those lines are tracked
                 but kept out of the headline numbers (config: headline_requires_unadjusted)
  combo          on new_player; combo/matchup markets are different props and are never mapped

Stat types not listed in config["pp_stat_types"] map to category "other" and are ignored.

    python prizepicks.py          # print a summary of the live board
"""
from collections import Counter

from common import PP_BOARD_URL, get_json, load_config


def fetch_board(league_id):
    payload = get_json(PP_BOARD_URL, params={"league_id": league_id, "per_page": 1000}, tries=3)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("PrizePicks response is not a JSON:API board (no top-level 'data' list)")
    return payload


def _rel_id(res, name):
    data = ((res.get("relationships") or {}).get(name) or {}).get("data")
    return str(data["id"]) if isinstance(data, dict) and data.get("id") is not None else None


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def parse_board(payload, cfg):
    players = {str(i["id"]): (i.get("attributes") or {})
               for i in payload.get("included", []) if i.get("type") == "new_player"}
    stat_map = {k.strip().lower(): v for k, v in cfg["pp_stat_types"].items()}
    rows = []
    for p in payload.get("data", []):
        if p.get("type") != "projection":
            continue
        a = p.get("attributes") or {}
        pid = _rel_id(p, "new_player")
        pl = players.get(pid or "", {})
        stat = (a.get("stat_type") or a.get("stat_display_name") or "").strip()
        try:
            line = float(a.get("line_score"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "pp_proj_id": str(p.get("id")),
            "pp_player_id": pid,
            "player": (pl.get("display_name") or pl.get("name") or "").strip(),
            "team": (pl.get("team") or "").strip(),
            "position": (pl.get("position") or "").strip(),
            "combo": _bool(pl.get("combo", False)),
            "stat_type": stat,
            "category": stat_map.get(stat.lower(), "other"),
            "line": line,
            "odds_type": (str(a.get("odds_type") or "standard").strip().lower() or "standard"),
            "adjusted_odds": _bool(a.get("adjusted_odds", False)),
            "projection_type": a.get("projection_type"),
            "status": a.get("status"),
            "is_live": _bool(a.get("is_live", False)),
            "start_time": a.get("start_time"),
            "opp": (a.get("description") or "").strip(),
            "pp_game_id": a.get("game_id"),
            "board_time": a.get("board_time"),
            "updated_at": a.get("updated_at"),
        })
    return rows


if __name__ == "__main__":
    cfg = load_config()
    rows = parse_board(fetch_board(cfg["league_id"]), cfg)
    print(f"{len(rows)} projections")
    print(Counter((r["stat_type"], r["odds_type"], r["adjusted_odds"]) for r in rows
                  if r["category"] != "other").most_common())
