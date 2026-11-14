"""
Underdog NHL pick'em board from api.underdogfantasy.com/v1/over_under_lines (public, read-only, no login).

The response is five flat lists joined by ids (field layout taken from the user's Underdog scraper):

  over_under_lines[]  one posted line: id, stat_value (string), status, options[]
      .options[]      one record per side SOLD: choice "higher"/"lower", status, payout_multiplier
      .over_under.appearance_stat   appearance_id, display_stat ("Shots on Goal"), stat (slug)
  appearances[]       player_id, match_id / match_type ("Game"), team_id
  players[]           first_name, last_name, sport_id ("NHL")
  games[]             abbreviated_title "AWAY @ HOME", home_team_id, away_team_id, scheduled_at, status

Rules here:
  * Only NHL players, only stats listed in config["underdog"]["stat_types"] (exact name match, so
    period / combo stats never map), only active lines with at least one active side.
  * Sides come from options[] only. A line can sell one side or both; Higher = over, Lower = under.
  * Line type per side from its payout multiplier: equal to the standard multiplier = "standard",
    less = "goblin", more = "demon". The standard value is config["underdog"]["standard_multiplier"]
    or, when that is null, the most common multiplier on the whole board.
  * Teams come from the game title; the abbreviations are mapped to the NHL's (TB -> TBL ...).

    python underdog.py          # print a summary of the live NHL board (run this to check the mapping)
"""
import json
import shutil
import subprocess
from collections import Counter
from datetime import timedelta, timezone

from common import UD_BOARD_URL, USER_AGENT, get_json, load_config, parse_ts

SIDE = {"higher": "over", "lower": "under"}
LIVE_STATES = {"in_progress", "live", "final", "finished", "complete", "completed", "closed", "cancelled", "canceled",
               "postponed", "suspended"}


def _curl_board():
    """Same read-only GET through the curl binary. Some networks refuse Python's connection to this
    host but allow curl's (noted in the user's scraper); it is the same request, not a workaround
    for a block: an HTTP error or a challenge page still fails."""
    if not shutil.which("curl"):
        raise RuntimeError("curl not available")
    out = subprocess.run(["curl", "-sS", "--fail", "--max-time", "90", "-H", "Accept: application/json",
                          "-H", f"User-Agent: {USER_AGENT}", UD_BOARD_URL],
                         capture_output=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"curl exit {out.returncode}: {out.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return json.loads(out.stdout.decode("utf-8"))


def fetch_board():
    try:
        raw = get_json(UD_BOARD_URL, tries=2, timeout=90)
    except Exception as e:
        print(f"  Underdog fetch with requests failed ({e}); trying curl")
        raw = _curl_board()
    if not isinstance(raw, dict) or not all(isinstance(raw.get(k), list) for k in
                                            ("over_under_lines", "appearances", "players")):
        raise RuntimeError("Underdog response is not an over_under_lines board")
    return raw


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mult(option):
    """A side's payout multiplier (first price-like field that parses)."""
    for k in ("payout_multiplier", "multiplier", "decimal_price"):
        v = _num(option.get(k))
        if v is not None and v > 0:
            return v
    return None


def _norm_stat(s):
    return " ".join(str(s or "").lower().replace("_", " ").split())


def standard_multiplier(raw, ucfg):
    """Configured value, else the most common multiplier across every active side on the board."""
    if ucfg.get("standard_multiplier"):
        return float(ucfg["standard_multiplier"]), "config"
    c = Counter()
    for line in raw.get("over_under_lines", []):
        for o in (line.get("options") or []) if isinstance(line, dict) else []:
            m = _mult(o) if isinstance(o, dict) else None
            if m is not None:
                c[round(m, 3)] += 1
    return (c.most_common(1)[0][0], "auto") if c else (None, "none")


def line_type(mult, std, tol):
    if mult is None or std is None:
        return "unknown"
    if abs(mult - std) <= tol:
        return "standard"
    return "goblin" if mult < std else "demon"


def parse_board(raw, cfg):
    """-> (rows, info). Rows use the same field names as the PrizePicks rows so pricing/grading is shared:
    pp_proj_id = Underdog line id, pp_player_id = "ud:<player id>"."""
    ucfg = cfg.get("underdog") or {}
    stat_map = {_norm_stat(k): v for k, v in (ucfg.get("stat_types") or {}).items()}
    tmap = dict(cfg.get("team_map") or {}, **(ucfg.get("team_map") or {}))
    std, std_src = standard_multiplier(raw, ucfg)
    tol = float(ucfg.get("multiplier_tolerance", 0.005))

    players = {str(p["id"]): p for p in raw.get("players", []) if isinstance(p, dict) and p.get("id") is not None}
    apps = {str(a["id"]): a for a in raw.get("appearances", []) if isinstance(a, dict) and a.get("id") is not None}
    games = {str(g["id"]): g for g in raw.get("games") or [] if isinstance(g, dict) and g.get("id") is not None}

    rows, info = [], {"standard_multiplier": std, "standard_source": std_src, "nhl_lines": 0,
                      "unmapped_stats": Counter(), "skipped": Counter(), "no_team": 0}
    for line in raw.get("over_under_lines", []):
        if not isinstance(line, dict):
            continue
        st = (line.get("over_under") or {}).get("appearance_stat") or {}
        app = apps.get(str(st.get("appearance_id")))
        pl = players.get(str(app.get("player_id"))) if app else None
        if not pl or str(pl.get("sport_id") or "").upper() != "NHL":
            continue
        info["nhl_lines"] += 1
        display = (st.get("display_stat") or st.get("stat") or "").strip()
        cat = stat_map.get(_norm_stat(display)) or stat_map.get(_norm_stat(st.get("stat")))
        if not cat:
            info["unmapped_stats"][display] += 1
            continue
        if (line.get("status") or "active") != "active":
            info["skipped"]["line not active"] += 1
            continue
        val = _num(line.get("stat_value"))
        if val is None:
            info["skipped"]["no line value"] += 1
            continue
        mult = {}
        for o in line.get("options") or []:
            side = SIDE.get(str(o.get("choice") or "").lower()) if isinstance(o, dict) else None
            if side and (o.get("status") or "active") == "active":
                mult[side] = _mult(o)
        if not mult:
            info["skipped"]["no active side"] += 1
            continue

        game = games.get(str(app.get("match_id"))) if app.get("match_type") in (None, "Game") else None
        team = opp = ""
        start = state = None
        if game:
            start = game.get("scheduled_at")
            state = str(game.get("status") or "").strip().lower() or None
            title = game.get("abbreviated_title") or game.get("title") or ""
            if " @ " in title:
                away, home = (t.strip().upper() for t in title.split(" @ ", 1))
                tid = app.get("team_id")
                if tid is not None and tid == game.get("home_team_id"):
                    team, opp = home, away
                elif tid is not None and tid == game.get("away_team_id"):
                    team, opp = away, home
        if not start:
            info["skipped"]["no game time"] += 1
            continue
        if not team:
            info["no_team"] += 1
        try:
            start_dt = parse_ts(start)
        except ValueError:
            info["skipped"]["unreadable game time"] += 1
            continue
        if start_dt.tzinfo is None:   # no offset given: Underdog times are UTC
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        name = " ".join(x for x in ((pl.get("first_name") or "").strip(), (pl.get("last_name") or "").strip()) if x)
        rows.append({
            "book": "underdog",
            "pp_proj_id": str(line.get("id")),
            "pp_player_id": f"ud:{pl['id']}",
            "player": name,
            "team": tmap.get(team, team),
            "opp": tmap.get(opp, opp),
            "position": "G" if cat == "saves" else "",   # Underdog lists no position; saves implies goalie
            "combo": False,
            "stat_type": display,
            "category": cat,
            "line": val,
            "sides": [s for s in ("over", "under") if s in mult],
            "mult_over": mult.get("over"),
            "mult_under": mult.get("under"),
            "type_over": line_type(mult["over"], std, tol) if "over" in mult else None,
            "type_under": line_type(mult["under"], std, tol) if "under" in mult else None,
            "odds_type": "standard",          # replaced by the picked side's type once the model picks
            "adjusted_odds": False,
            "status": "live" if state in LIVE_STATES or line.get("live_event") else "pre_game",
            "start_time": start_dt.isoformat(),
            # US Eastern calendar date of the game, like PrizePicks' start_time[:10] (UTC-5h never
            # crosses a date line for a real puck drop, in either EST or EDT)
            "game_date": (start_dt - timedelta(hours=5)).date().isoformat(),
            "board_time": None,
            "updated_at": line.get("updated_at"),
        })
    return rows, info


def dedupe(rows):
    """One line per player and prop (the grading key). If Underdog ever lists several, keep the most
    'normal' one: both sides sold and standard first, then the earliest listed."""
    best, dropped = {}, 0
    def score(r):
        return (len(r["sides"]) == 2, sum(t == "standard" for t in (r["type_over"], r["type_under"])))
    for r in rows:
        k = (r["game_date"], r["pp_player_id"], r["category"])
        if k not in best:
            best[k] = r
        else:
            dropped += 1
            if score(r) > score(best[k]):
                best[k] = r
    return list(best.values()), dropped


if __name__ == "__main__":
    cfg = load_config()
    rows, info = parse_board(fetch_board(), cfg)
    print(f"NHL lines on the board: {info['nhl_lines']}  ->  {len(rows)} in tracked props")
    print(f"standard multiplier: {info['standard_multiplier']} ({info['standard_source']})")
    print("\nby prop: lines (sides sold / type of each side)")
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        sides = Counter("+".join(r["sides"]) for r in rs)
        types = Counter(t for r in rs for t in (r["type_over"], r["type_under"]) if t)
        print(f"  {cat:9} {len(rs):4}   {dict(sides)}   {dict(types)}")
    print("\nmultipliers seen (over, under):", Counter((r["mult_over"], r["mult_under"]) for r in rows).most_common(12))
    print("teams:", sorted({r["team"] for r in rows}), f"({info['no_team']} lines with no team)")
    print("game dates:", sorted({r["game_date"] for r in rows}), " status:", dict(Counter(r["status"] for r in rows)))
    print("\nNHL stats NOT tracked (name: lines):")
    for name, n in info["unmapped_stats"].most_common():
        print(f"  {name}: {n}")
    if info["skipped"]:
        print("skipped:", dict(info["skipped"]))
    print("\nsample rows:")
    for r in rows[:5]:
        print(" ", {k: r[k] for k in ("player", "team", "opp", "stat_type", "category", "line", "sides",
                                      "mult_over", "mult_under", "type_over", "type_under", "start_time", "game_date")})
