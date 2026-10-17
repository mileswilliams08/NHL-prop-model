"""
Pull finished NHL games from the public NHL API into data/history/*.csv.

    python nhl_backfill.py                       # 2024-25 opener through yesterday (first run: ~20-30 min)
    python nhl_backfill.py --start 2026-10-10    # just a recent window (what the daily Action does)
    python nhl_backfill.py --rebuild-only        # rebuild CSVs from the local cache

Each finished boxscore is cached in data/raw/boxscores/<game_id>.json.gz (gitignored), so
re-runs only fetch new games. CSV rebuilds MERGE with what's already in data/history: rows
for any re-fetched game are replaced, everything else is kept. That is what lets the daily
Action (which has no local cache) add the last few days without wiping the history.

Only games whose state is OFF/FINAL are ever cached or written, so running this while games
are in progress can't record a partial game.

Outputs (data/history/):
    games.csv          one row per game: teams, score, team SOG, how it ended (REG/OT/SO)
    skater_games.csv   one row per skater per game
    goalie_games.csv   one row per goalie per game (dressed backups appear with toi_sec 0)
    players.csv        player_id -> boxscore name ("A. Fox"), position, last team, games
"""
import argparse
import gzip
import json
import os
import time
from collections import Counter
from datetime import date, timedelta

import pandas as pd

from common import HIST_DIR, NHL_BASE, RAW_DIR, get_json

FINAL_STATES = {"OFF", "FINAL"}


# ----------------------------------------------------------------------------- fetching
def finished_games(start, end, game_types):
    """(game_id, date) for finished games in [start, end], walking the weekly schedule."""
    seen, out, d = set(), [], start
    while d <= end:
        sched = get_json(f"{NHL_BASE}/schedule/{d.isoformat()}") or {}
        for day in sched.get("gameWeek", []):
            gd = date.fromisoformat(day["date"])
            if not (start <= gd <= end):
                continue
            for g in day.get("games", []):
                if g.get("gameType") in game_types and g.get("gameState") in FINAL_STATES and g["id"] not in seen:
                    seen.add(g["id"])
                    out.append((g["id"], gd))
        d += timedelta(days=7)
    return out


def cache_path(game_id):
    return os.path.join(RAW_DIR, f"{game_id}.json.gz")


def fetch_boxscore(game_id):
    if os.path.exists(cache_path(game_id)):
        return False
    box = get_json(f"{NHL_BASE}/gamecenter/{game_id}/boxscore")
    if not box or box.get("gameState") not in FINAL_STATES:
        return False  # never cache a game that isn't final
    with gzip.open(cache_path(game_id), "wt", encoding="utf-8") as f:
        json.dump(box, f)
    return True


# ----------------------------------------------------------------------------- parsing
def toi_sec(s):
    if not s or ":" not in str(s):
        return 0
    m, sec = str(s).split(":")
    return int(m) * 60 + int(sec)


def _name(p):
    n = p.get("name")
    return n.get("default", "") if isinstance(n, dict) else (n or "")


def parse_boxscore(box):
    gid = box["id"]
    common = {"game_id": gid, "date": box.get("gameDate"), "season": box.get("season"),
              "game_type": box.get("gameType")}
    away, home = box["awayTeam"], box["homeTeam"]
    game = dict(common, start_utc=box.get("startTimeUTC"),
                away=away.get("abbrev"), home=home.get("abbrev"),
                away_score=away.get("score"), home_score=home.get("score"),
                away_sog=away.get("sog"), home_sog=home.get("sog"),
                last_period=(box.get("gameOutcome") or {}).get("lastPeriodType"))
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
                skaters.append(dict(base, player_id=p.get("playerId"), name=_name(p),
                                    position=p.get("position"), toi_sec=toi_sec(p.get("toi")), sog=sog,
                                    goals=p.get("goals"), assists=p.get("assists"), points=p.get("points"),
                                    plus_minus=p.get("plusMinus"), pim=p.get("pim"), hits=p.get("hits"),
                                    blocked=p.get("blockedShots"), ppg=p.get("powerPlayGoals"),
                                    shifts=p.get("shifts")))
        for p in stats.get("goalies", []):
            goalies.append(dict(base, player_id=p.get("playerId"), name=_name(p), position="G",
                                toi_sec=toi_sec(p.get("toi")), starter=p.get("starter"),
                                decision=p.get("decision"), saves=p.get("saves"),
                                shots_against=p.get("shotsAgainst"), goals_against=p.get("goalsAgainst")))
        if team.get("sog") is not None and sog_sum != team.get("sog"):
            warnings.append(f"{gid} {team.get('abbrev')}: skater SOG {sog_sum} != team SOG {team.get('sog')}")
    return game, skaters, goalies, warnings


# ----------------------------------------------------------------------------- rebuild / merge
def _merge(name, new_rows):
    p = os.path.join(HIST_DIR, name)
    new = pd.DataFrame(new_rows)
    if os.path.exists(p):
        old = pd.read_csv(p)
        if not new.empty:
            old = old[~old["game_id"].isin(new["game_id"])]
        new = pd.concat([old, new], ignore_index=True) if not new.empty else old
    if new.empty:
        return new
    new["game_id"] = new["game_id"].astype("int64")
    sort_cols = ["date", "game_id"] + (["team", "player_id"] if "player_id" in new else [])
    new = new.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    tmp = p + ".tmp"
    new.to_csv(tmp, index=False)
    os.replace(tmp, p)
    return new


def rebuild():
    games, skaters, goalies, warnings = [], [], [], []
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json.gz")] if os.path.isdir(RAW_DIR) else []
    for fn in files:
        with gzip.open(os.path.join(RAW_DIR, fn), "rt", encoding="utf-8") as f:
            g, s, gk, w = parse_boxscore(json.load(f))
        games.append(g); skaters += s; goalies += gk; warnings += w

    os.makedirs(HIST_DIR, exist_ok=True)
    gm = _merge("games.csv", games)
    sk = _merge("skater_games.csv", skaters)
    gk = _merge("goalie_games.csv", goalies)

    allp = pd.concat([sk[["player_id", "name", "position", "team", "date"]],
                      gk[["player_id", "name", "position", "team", "date"]]], ignore_index=True)
    allp = allp.sort_values("date", kind="stable")
    players = allp.groupby("player_id").agg(name=("name", "last"), position=("position", "last"),
                                            last_team=("team", "last"), last_date=("date", "last"),
                                            games=("date", "size")).reset_index()
    players.to_csv(os.path.join(HIST_DIR, "players.csv"), index=False)

    print("\n=== SUMMARY ===")
    for season, sub in gm.groupby("season"):
        print(f"season {season}: " + ", ".join(f"type {t}: {n} games"
                                                for t, n in sorted(Counter(sub['game_type']).items())))
    print("how games ended:", dict(Counter(gm["last_period"])))
    print(f"skater rows: {len(sk)}   goalie rows: {len(gk)}   players: {len(players)}")
    print(f"skater rows with 0 TOI: {int((sk['toi_sec'] == 0).sum())}")
    print(f"goalie rows with 0 TOI (dressed backups): {int((gk['toi_sec'] == 0).sum())}")
    print(f"date range: {gm['date'].min()} .. {gm['date'].max()}")
    played = sk[sk["toi_sec"] > 0]["sog"]
    print("skater SOG distribution (8 = 8+):", dict(sorted(Counter(played.clip(upper=8)).items())))
    print(f"mean SOG per skater-game: {played.mean():.3f}")
    print(f"this run parsed {len(files)} cached boxscores; warnings: {len(warnings)}")
    for w in warnings[:15]:
        print("  ", w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--types", default="2,3", help="1=preseason 2=regular 3=playoffs")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--rebuild-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    if not args.rebuild_only:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        types = {int(t) for t in args.types.split(",")}
        print(f"Listing finished games {start} .. {end} (types {sorted(types)}) ...")
        games = finished_games(start, end, types)
        todo = [gid for gid, _ in games if not os.path.exists(cache_path(gid))]
        print(f"{len(games)} finished games, {len(todo)} not cached yet")
        t0 = time.time()
        for i, gid in enumerate(todo, 1):
            if fetch_boxscore(gid):
                time.sleep(args.sleep)
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fetched ({time.time() - t0:.0f}s)")
    rebuild()


if __name__ == "__main__":
    main()
