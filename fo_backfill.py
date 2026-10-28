"""
Backfill per-game faceoffs won / taken from the NHL stats API into data/history/skater_faceoffs.csv.

    python fo_backfill.py                       # 2024-25 opener through yesterday (~110 requests, 1-3 min)
    python fo_backfill.py --start 2026-10-21    # a recent window (what the daily Action does)

Source: api.nhle.com/stats/rest/en/skater/faceoffwins (isGame=true, one request per week, limit=-1).
The report's field names are detected on the first request (they've changed over the years); the script
stops with a clear message if it can't find a won and a taken/lost count. Rows for re-fetched games
replace old ones; everything else is kept. Only players who took at least one faceoff appear.
"""
import argparse
import json
import os
import time
import urllib.parse
from datetime import date, timedelta

import pandas as pd

from common import HIST_DIR, get_json

URL = "https://api.nhle.com/stats/rest/en/skater/faceoffwins"
OUT = os.path.join(HIST_DIR, "skater_faceoffs.csv")
WON_KEYS = ("totalFaceoffWins", "faceoffWins", "faceoffsWon")
TAKEN_KEYS = ("totalFaceoffs", "faceoffs", "faceoffsTaken")
LOST_KEYS = ("totalFaceoffLosses", "faceoffLosses", "faceoffsLost")


def fetch(d1, d2):
    q = {"isAggregate": "false", "isGame": "true", "start": 0, "limit": -1,
         "sort": json.dumps([{"property": "playerId", "direction": "ASC"}, {"property": "gameId", "direction": "ASC"}]),
         "cayenneExp": f'gameDate>="{d1}" and gameDate<="{d2}" and gameTypeId>=2 and gameTypeId<=3'}
    data = get_json(URL + "?" + urllib.parse.urlencode(q)) or {}
    rows = data.get("data", [])
    if data.get("total") is not None and data["total"] != len(rows):
        raise RuntimeError(f"{d1}..{d2}: got {len(rows)} of {data['total']} rows (paging changed?)")
    return rows


def pick(keys, row):
    return next((k for k in keys if k in row), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()

    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    rows, fields, d = [], None, start
    while d <= end:
        d2 = min(d + timedelta(days=6), end)
        got = fetch(d.isoformat(), d2.isoformat())
        if got and fields is None:
            r0 = got[0]
            fields = {"won": pick(WON_KEYS, r0), "taken": pick(TAKEN_KEYS, r0), "lost": pick(LOST_KEYS, r0)}
            print("report fields:", ", ".join(sorted(r0.keys())))
            print("using:", fields)
            if not fields["won"] or not (fields["taken"] or fields["lost"]):
                raise SystemExit("Couldn't find faceoff won/taken fields in the report - paste the field list above back to Claude.")
        for r in got:
            won = r.get(fields["won"]) or 0
            taken = r.get(fields["taken"]) if fields["taken"] else won + (r.get(fields["lost"]) or 0)
            rows.append({"game_id": r["gameId"], "player_id": r["playerId"], "date": r["gameDate"],
                         "team": r.get("teamAbbrev"), "fo_won": int(won), "fo_taken": int(taken or 0)})
        print(f"  {d} .. {d2}: {len(got)} rows")
        d = d2 + timedelta(days=1)
        time.sleep(a.sleep)

    new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        if not new.empty:
            old = old[~old["game_id"].isin(new["game_id"])]
        new = pd.concat([old, new], ignore_index=True) if not new.empty else old
    if new.empty:
        print("no rows")
        return 0
    new = new.sort_values(["date", "game_id", "team", "player_id"], kind="stable")
    new.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    # ---- sanity checks (paste this back)
    print(f"\n=== SUMMARY === {len(new)} rows, {new['game_id'].nunique()} games, {new['date'].min()} .. {new['date'].max()}")
    print(f"won <= taken: {(new.fo_won <= new.fo_taken).mean():.1%}")
    per_game = new.groupby("game_id").agg(won=("fo_won", "sum"), taken=("fo_taken", "sum"))
    print(f"per game: faceoffs {per_game.won.mean():.1f} (won by either team), taken counted per player {per_game.taken.mean():.1f}")
    print(f"taken = 2 x won in a game (every faceoff has one winner and one loser): {(per_game.taken == 2 * per_game.won).mean():.1%}")
    heavy = new[new.fo_taken >= 10]
    print(f"player-games with 10+ faceoffs: {len(heavy)}, avg won {heavy.fo_won.mean():.2f} of {heavy.fo_taken.mean():.2f}")
    sk = os.path.join(HIST_DIR, "skater_games.csv")
    if os.path.exists(sk):
        box = pd.read_csv(sk, usecols=["game_id"]).drop_duplicates()
        print(f"boxscore games with no faceoff rows: {(~box.game_id.isin(new.game_id)).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
