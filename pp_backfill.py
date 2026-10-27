"""
Backfill per-game situational ice time and power-play shots from the NHL stats API into
data/history/skater_situational.csv (one row per skater per game).

    python pp_backfill.py                       # 2024-25 opener through yesterday (~150 requests, a few minutes)
    python pp_backfill.py --start 2026-10-19    # a recent window (what the daily Action does)

Sources (api.nhle.com/stats/rest/en/skater/<report>, isGame=true, one request per week with limit=-1):
    timeonice   evTimeOnIce, ppTimeOnIce, shTimeOnIce, otTimeOnIce, timeOnIce   (seconds)
    powerplay   ppShots, ppTimeOnIcePctPerGame (share of his team's power-play time he was on the ice)

Rows for re-fetched games replace old ones; everything else is kept (same merge rule as nhl_backfill.py).
Only regular season and playoffs (gameTypeId 2 and 3).
"""
import argparse
import json
import os
import time
import urllib.parse
from datetime import date, timedelta

import pandas as pd

from common import HIST_DIR, get_json

BASE = "https://api.nhle.com/stats/rest/en/skater/"
OUT = os.path.join(HIST_DIR, "skater_situational.csv")


def fetch(report, d1, d2):
    q = {"isAggregate": "false", "isGame": "true", "start": 0, "limit": -1,
         "sort": json.dumps([{"property": "playerId", "direction": "ASC"}, {"property": "gameId", "direction": "ASC"}]),
         "cayenneExp": f'gameDate>="{d1}" and gameDate<="{d2}" and gameTypeId>=2 and gameTypeId<=3'}
    data = get_json(BASE + report + "?" + urllib.parse.urlencode(q)) or {}
    rows = data.get("data", [])
    if data.get("total") is not None and data["total"] != len(rows):
        raise RuntimeError(f"{report} {d1}..{d2}: got {len(rows)} of {data['total']} rows (paging changed?)")
    return rows


def week_rows(d1, d2):
    toi = {(r["gameId"], r["playerId"]): r for r in fetch("timeonice", d1, d2)}
    pp = {(r["gameId"], r["playerId"]): r for r in fetch("powerplay", d1, d2)}
    out = []
    for k, t in toi.items():
        p = pp.get(k, {})
        out.append({"game_id": t["gameId"], "player_id": t["playerId"], "date": t["gameDate"], "team": t["teamAbbrev"],
                    "position": t["positionCode"], "toi_sec": t.get("timeOnIce") or 0,
                    "ev_toi": t.get("evTimeOnIce") or 0, "pp_toi": t.get("ppTimeOnIce") or 0,
                    "sh_toi": t.get("shTimeOnIce") or 0, "ot_toi": t.get("otTimeOnIce") or 0,
                    "pp_shots": p.get("ppShots") or 0, "pp_share": p.get("ppTimeOnIcePctPerGame")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()

    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    rows, d = [], start
    while d <= end:
        d2 = min(d + timedelta(days=6), end)
        got = week_rows(d.isoformat(), d2.isoformat())
        rows += got
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

    # ---- sanity checks against the boxscore history (paste this back)
    print(f"\n=== SUMMARY === {len(new)} rows, {new['game_id'].nunique()} games, {new['date'].min()} .. {new['date'].max()}")
    sk_path = os.path.join(HIST_DIR, "skater_games.csv")
    if os.path.exists(sk_path):
        sk = pd.read_csv(sk_path, usecols=["game_id", "player_id", "toi_sec", "sog"])
        j = sk.merge(new, on=["game_id", "player_id"], suffixes=("_box", "_api"))
        print(f"matched to boxscores: {len(j)} of {len(sk)} boxscore rows")
        print(f"TOI agrees within 2 sec: {(abs(j.toi_sec_box - j.toi_sec_api) <= 2).mean():.1%}")
        print(f"ev+pp+sh == toi within 2 sec: {(abs(j.ev_toi + j.pp_toi + j.sh_toi - j.toi_sec_api) <= 2).mean():.1%}")
        print(f"pp_shots <= boxscore sog: {(j.pp_shots <= j.sog).mean():.1%}")
        print(f"share of all SOG taken on the power play: {j.pp_shots.sum() / j.sog.sum():.1%}")
        print(f"share of all TOI on the power play: {j.pp_toi.sum() / j.toi_sec_api.sum():.1%}")
        missing = sk.merge(new[["game_id"]].drop_duplicates(), on="game_id", how="left", indicator=True)
        print(f"boxscore games with no situational rows: {missing.loc[missing._merge == 'left_only', 'game_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
