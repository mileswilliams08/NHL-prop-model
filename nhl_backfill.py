"""
Backfill NHL game logs from the public NHL API (api-web.nhle.com) into CSVs.

Run locally (no installs needed, stdlib only):
    python nhl_backfill.py                          # 2024-25 opener through yesterday
    python nhl_backfill.py --start 2025-10-01       # custom range
    python nhl_backfill.py --rebuild-only           # just rebuild CSVs from cached boxscores

Resumable: each finished boxscore is cached under data/raw/boxscores/<game_id>.json.gz
(keep that folder out of git). Re-running only fetches games not yet cached.

Outputs (data/history/):
    games.csv          one row per game (teams, score, team SOG, how it ended: REG/OT/SO)
    skater_games.csv   one row per skater per game (TOI, SOG, G, A, hits, blocks, ...)
    goalie_games.csv   one row per goalie per game (TOI, saves, shots against, starter, ...)
    players.csv        player_id -> boxscore name, position, last team seen
"""
import argparse
import csv
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "nhl-prop-model backfill (personal research)", "Accept": "application/json"}
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw", "boxscores")
HIST_DIR = os.path.join(ROOT, "data", "history")
FINAL_STATES = {"OFF", "FINAL"}


# ----------------------------------------------------------------------------- http
def get_json(url, tries=4):
    err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # timeouts, resets
            err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}: {err}")


# ----------------------------------------------------------------------------- schedule
def finished_games(start, end, game_types):
    """Yield (game_id, game_date) for finished games in [start, end], walking the weekly schedule."""
    seen = set()
    d = start
    while d <= end:
        sched = get_json(f"{BASE}/schedule/{d.isoformat()}") or {}
        for day in sched.get("gameWeek", []):
            gd = date.fromisoformat(day["date"])
            if gd < start or gd > end:
                continue
            for g in day.get("games", []):
                if g.get("gameType") in game_types and g.get("gameState") in FINAL_STATES and g["id"] not in seen:
                    seen.add(g["id"])
                    yield g["id"], gd
        d += timedelta(days=7)


# ----------------------------------------------------------------------------- cache
def cache_path(game_id):
    return os.path.join(RAW_DIR, f"{game_id}.json.gz")


def fetch_boxscore(game_id):
    path = cache_path(game_id)
    if os.path.exists(path):
        return False
    box = get_json(f"{BASE}/gamecenter/{game_id}/boxscore")
    if not box or box.get("gameState") not in FINAL_STATES:
        return False  # never cache a game that isn't final
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(box, f)
    return True


# ----------------------------------------------------------------------------- parsing
def toi_sec(s):
    if not s or ":" not in s:
        return 0
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def name_of(p):
    n = p.get("name")
    return n.get("default", "") if isinstance(n, dict) else (n or "")


def parse(box):
    gid = box["id"]
    common = {"game_id": gid, "date": box.get("gameDate"), "season": box.get("season"), "game_type": box.get("gameType")}
    away, home = box["awayTeam"], box["homeTeam"]
    game = dict(common,
                start_utc=box.get("startTimeUTC"),
                away=away.get("abbrev"), home=home.get("abbrev"),
                away_score=away.get("score"), home_score=home.get("score"),
                away_sog=away.get("sog"), home_sog=home.get("sog"),
                last_period=(box.get("gameOutcome") or {}).get("lastPeriodType"),
                venue=(box.get("venue") or {}).get("default"))

    skaters, goalies, warnings = [], [], []
    pbg = box.get("playerByGameStats")
    if not pbg:
        return game, skaters, goalies, [f"{gid}: no playerByGameStats"]

    for side, team, opp, is_home in (("awayTeam", away, home, 0), ("homeTeam", home, away, 1)):
        base = dict(common, team=team.get("abbrev"), opp=opp.get("abbrev"), is_home=is_home)
        stats = pbg.get(side, {})
        sog_sum = 0
        for grp in ("forwards", "defense"):
            for p in stats.get(grp, []):
                sog = p.get("sog", p.get("shots", 0)) or 0
                sog_sum += sog
                skaters.append(dict(base,
                                    player_id=p.get("playerId"), name=name_of(p), sweater=p.get("sweaterNumber"),
                                    position=p.get("position"), toi_sec=toi_sec(p.get("toi")), sog=sog,
                                    goals=p.get("goals"), assists=p.get("assists"), points=p.get("points"),
                                    plus_minus=p.get("plusMinus"), pim=p.get("pim"), hits=p.get("hits"),
                                    blocked=p.get("blockedShots"), ppg=p.get("powerPlayGoals"),
                                    shifts=p.get("shifts"), giveaways=p.get("giveaways"),
                                    takeaways=p.get("takeaways"), fo_pct=p.get("faceoffWinningPctg")))
        for p in stats.get("goalies", []):
            goalies.append(dict(base,
                                player_id=p.get("playerId"), name=name_of(p), toi_sec=toi_sec(p.get("toi")),
                                starter=p.get("starter"), decision=p.get("decision"),
                                saves=p.get("saves"), shots_against=p.get("shotsAgainst"),
                                goals_against=p.get("goalsAgainst"),
                                ev_sa=p.get("evenStrengthShotsAgainst"), pp_sa=p.get("powerPlayShotsAgainst"),
                                sh_sa=p.get("shorthandedShotsAgainst")))
        if team.get("sog") is not None and sog_sum != team.get("sog"):
            warnings.append(f"{gid} {team.get('abbrev')}: skater SOG {sog_sum} != team SOG {team.get('sog')}")
    return game, skaters, goalies, warnings


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def rebuild():
    games, skaters, goalies, warnings = [], [], [], []
    for fn in os.listdir(RAW_DIR):
        if not fn.endswith(".json.gz"):
            continue
        with gzip.open(os.path.join(RAW_DIR, fn), "rt", encoding="utf-8") as f:
            g, s, gk, w = parse(json.load(f))
        games.append(g); skaters += s; goalies += gk; warnings += w

    key = lambda r: (r["date"] or "", r["game_id"])
    games.sort(key=key); skaters.sort(key=key); goalies.sort(key=key)

    players = {}
    for r in skaters + goalies:  # sorted by date, so later rows overwrite earlier
        pid = r["player_id"]
        prev = players.get(pid, {"games": 0})
        players[pid] = {"player_id": pid, "name": r["name"], "position": r.get("position", "G"),
                        "last_team": r["team"], "last_date": r["date"], "games": prev["games"] + 1}

    os.makedirs(HIST_DIR, exist_ok=True)
    write_csv(os.path.join(HIST_DIR, "games.csv"), games)
    write_csv(os.path.join(HIST_DIR, "skater_games.csv"), skaters)
    write_csv(os.path.join(HIST_DIR, "goalie_games.csv"), goalies)
    write_csv(os.path.join(HIST_DIR, "players.csv"), sorted(players.values(), key=lambda r: r["player_id"]))

    # ---- sanity summary (paste this back)
    print("\n=== SUMMARY ===")
    by_season = defaultdict(lambda: defaultdict(int))
    for g in games:
        by_season[g["season"]][g["game_type"]] += 1
    for season in sorted(by_season):
        print(f"season {season}: " + ", ".join(f"type {t}: {n} games" for t, n in sorted(by_season[season].items())))
    ends = defaultdict(int)
    for g in games:
        ends[g["last_period"]] += 1
    print("how games ended:", dict(ends))
    print(f"skater rows: {len(skaters)}   goalie rows: {len(goalies)}   players: {len(players)}")
    print(f"skater rows with 0 TOI: {sum(1 for r in skaters if r['toi_sec'] == 0)}")
    print(f"goalie rows with 0 TOI (dressed backups): {sum(1 for r in goalies if r['toi_sec'] == 0)}")
    if games:
        print(f"date range: {games[0]['date']} .. {games[-1]['date']}")
    sog_vals = [r["sog"] for r in skaters if r["toi_sec"] > 0]
    if sog_vals:
        dist = defaultdict(int)
        for v in sog_vals:
            dist[min(v, 8)] += 1
        print("skater SOG distribution (8 = 8+):", {k: dist[k] for k in sorted(dist)})
        print(f"mean SOG per skater-game: {sum(sog_vals) / len(sog_vals):.3f}")
    print(f"warnings: {len(warnings)}")
    for w in warnings[:15]:
        print("  ", w)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01", help="YYYY-MM-DD (default: 2024-25 season)")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(), help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--types", default="2,3", help="gameType list: 1=preseason 2=regular 3=playoffs")
    ap.add_argument("--sleep", type=float, default=0.25, help="seconds between boxscore requests")
    ap.add_argument("--rebuild-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    if not args.rebuild_only:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        types = {int(t) for t in args.types.split(",")}
        print(f"Listing finished games {start} .. {end} (types {sorted(types)}) ...")
        games = list(finished_games(start, end, types))
        todo = [gid for gid, _ in games if not os.path.exists(cache_path(gid))]
        print(f"{len(games)} finished games, {len(todo)} not cached yet")
        t0 = datetime.now()
        for i, gid in enumerate(todo, 1):
            if fetch_boxscore(gid):
                time.sleep(args.sleep)
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fetched  ({(datetime.now() - t0).seconds}s)")
    rebuild()


if __name__ == "__main__":
    main()
