"""
Line combinations from NHL shift charts -> data/history/skater_lines.csv (one row per skater per game).

    python shift_backfill.py                    # every game in data/history/games.csv not done yet (~3,000 requests, 15-25 min)
    python shift_backfill.py --start 2026-11-01 # only games from a date (what a daily step would do)

Source: api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id> (free, every shift of every game).
Raw shifts are NOT stored (millions of rows); each game is boiled down to, per skater:

  es_sec      seconds at 5-on-5 (both teams 5 skaters, both goalies in)
  pp_sec      seconds on the power play (his team has more skaters, neither net empty)
  line        forward line 1-4 / defence pair 1-3 for that game, by 5-on-5 time (0 = no regular line)
  mate1/2     his most frequent 5-on-5 linemates (forwards: two forwards; defence: one partner)
  mate_share  share of his 5-on-5 time spent with mate1 (how settled the line was)
  pp_unit     1 / 2 by power-play time among his team's skaters that game (0 = no PP time)

Lines are found greedily: take the forward with the most 5-on-5 time not yet placed, give him the two
unplaced forwards he shared the most 5-on-5 time with, repeat; lines are numbered by their average 5-on-5
time. Same for defence pairs. The script is resumable: games already in the file are skipped.
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from common import HIST_DIR, get_json

URL = "https://api.nhle.com/stats/rest/en/shiftcharts"
OUT = os.path.join(HIST_DIR, "skater_lines.csv")
FIELDS = ["game_id", "date", "team", "player_id", "pos", "es_sec", "pp_sec", "line", "mate1", "mate2",
          "mate_share", "pp_unit"]


def mmss(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return None


def fetch_shifts(gid):
    data = get_json(URL, params={"cayenneExp": f"gameId={gid}"}) or {}
    return data.get("data", [])


def parse(rows, goalie_ids):
    """-> list of (team, player_id, period, start_sec, end_sec) for real skater shifts."""
    out = []
    for r in rows:
        if r.get("duration") in (None, "") or r.get("typeCode") == 505:   # 505 = goal rows, not shifts
            continue
        pid, team, per = r.get("playerId"), r.get("teamAbbrev"), r.get("period")
        a, b = mmss(r.get("startTime")), mmss(r.get("endTime"))
        if None in (pid, team, per, a, b) or b <= a or per > 5:            # per > 5: shootout
            continue
        out.append((team, int(pid), int(per), a, b, int(pid) in goalie_ids))
    return out


def summarize(gid, date, shifts, pos_of):
    """One game's shifts -> per-skater rows (see module docstring)."""
    teams = sorted({s[0] for s in shifts})
    if len(teams) != 2:
        return []
    periods = sorted({s[2] for s in shifts})
    plen = {p: max(s[4] for s in shifts if s[2] == p) for p in periods}
    offset, t = {}, 0
    for p in periods:
        offset[p], t = t, t + plen[p]
    T = t
    players = sorted({(s[0], s[1], s[5]) for s in shifts})
    idx = {(tm, pid): i for i, (tm, pid, _) in enumerate(players)}
    on = np.zeros((len(players), T), dtype=bool)
    for tm, pid, per, a, b, _ in shifts:
        on[idx[(tm, pid)], offset[per] + a: offset[per] + b] = True
    is_g = np.array([g for _, _, g in players])
    team_of = np.array([tm for tm, _, _ in players])
    sk = {tm: (~is_g) & (team_of == tm) for tm in teams}
    gk = {tm: is_g & (team_of == tm) for tm in teams}
    n_sk = {tm: on[sk[tm]].sum(0) for tm in teams}
    g_in = {tm: on[gk[tm]].any(0) for tm in teams}
    both_g = g_in[teams[0]] & g_in[teams[1]]
    es = both_g & (n_sk[teams[0]] == 5) & (n_sk[teams[1]] == 5)
    rows = []
    for tm in teams:
        opp = teams[1] if tm == teams[0] else teams[0]
        pp = both_g & (n_sk[tm] > n_sk[opp]) & (n_sk[opp] >= 3)
        members = [i for i in np.where(sk[tm])[0]]
        es_sec = {i: int((on[i] & es).sum()) for i in members}
        pp_sec = {i: int((on[i] & pp).sum()) for i in members}
        info = {i: {"line": 0, "mates": [], "share": None} for i in members}
        for grp, size, n_lines in (("F", 3, 4), ("D", 2, 3)):
            g = [i for i in members if (pos_of.get(players[i][1]) == "D") == (grp == "D")]
            shared = {i: {j: int((on[i] & on[j] & es).sum()) for j in g if j != i} for i in g}
            for i in g:
                top = sorted(shared[i].items(), key=lambda x: -x[1])[: size - 1]
                info[i]["mates"] = [players[j][1] for j, s in top if s > 0]
                info[i]["share"] = round(top[0][1] / es_sec[i], 3) if top and es_sec[i] else None
            left, lines = sorted(g, key=lambda i: -es_sec[i]), []
            while left and len(lines) < n_lines:
                head = left.pop(0)
                if es_sec[head] < 60:                     # under a minute at 5-on-5: no regular line
                    break
                mates = sorted(left, key=lambda j: -shared[head][j])[: size - 1]
                mates = [j for j in mates if shared[head][j] >= 60]
                for j in mates:
                    left.remove(j)
                lines.append([head] + mates)
            lines.sort(key=lambda L: -np.mean([es_sec[i] for i in L]))
            for n, L in enumerate(lines, 1):
                for i in L:
                    info[i]["line"] = n
        ranked = sorted([i for i in members if pp_sec[i] > 0], key=lambda i: -pp_sec[i])
        unit = {i: (1 if r < 5 else 2) for r, i in enumerate(ranked)}
        for i in members:
            m = info[i]["mates"]
            rows.append({"game_id": gid, "date": date, "team": tm, "player_id": players[i][1],
                         "pos": "D" if pos_of.get(players[i][1]) == "D" else "F",
                         "es_sec": es_sec[i], "pp_sec": pp_sec[i], "line": info[i]["line"],
                         "mate1": m[0] if len(m) > 0 else "", "mate2": m[1] if len(m) > 1 else "",
                         "mate_share": info[i]["share"] if info[i]["share"] is not None else "",
                         "pp_unit": unit.get(i, 0)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--sleep", type=float, default=0.15)
    a = ap.parse_args()

    games = pd.read_csv(os.path.join(HIST_DIR, "games.csv"), usecols=["game_id", "date", "game_type"])
    games = games[(games.date >= a.start) & games.game_type.isin([2, 3])].sort_values(["date", "game_id"])
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"), usecols=["player_id", "position"])
    pos_of = dict(zip(sk.player_id, sk.position))
    gl = pd.read_csv(os.path.join(HIST_DIR, "goalie_games.csv"), usecols=["player_id"])
    goalie_ids = set(gl.player_id)

    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=FIELDS)
    done = set(old.game_id) if not old.empty else set()
    todo = games[~games.game_id.isin(done)]
    print(f"{len(games)} games from {a.start}; {len(todo)} to fetch ({len(done)} already in {os.path.relpath(OUT)})")
    new, fails, empty = [], [], []
    for n, g in enumerate(todo.itertuples(), 1):
        try:
            raw = fetch_shifts(g.game_id)
        except Exception as e:
            fails.append(g.game_id)
            print(f"  {g.game_id}: fetch failed ({e})")
            continue
        if n == 1 and raw:
            print("  shift fields:", ", ".join(sorted(raw[0].keys())))
        rows = summarize(g.game_id, g.date, parse(raw, goalie_ids), pos_of)
        if not rows:
            empty.append(g.game_id)
        new += rows
        if n % 100 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} games ({g.date})")
            out = pd.concat([old, pd.DataFrame(new, columns=FIELDS)], ignore_index=True)   # checkpoint
            out.to_csv(OUT + ".tmp", index=False)
            os.replace(OUT + ".tmp", OUT)
        time.sleep(a.sleep)

    out = pd.concat([old, pd.DataFrame(new, columns=FIELDS)], ignore_index=True)
    if out.empty:
        print("no rows")
        return 0
    out = out.drop_duplicates(["game_id", "player_id"], keep="last").sort_values(["date", "game_id", "team", "line"])
    out.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    # ---- sanity checks (paste this back)
    print(f"\n=== SUMMARY === {len(out)} rows, {out.game_id.nunique()} games, {out.date.min()} .. {out.date.max()}")
    print(f"fetch failures: {len(fails)}  games with no usable shifts: {len(empty)} {empty[:10]}")
    tg = out.groupby(["game_id", "team"])
    print(f"5-on-5 minutes per team-game (should be ~45-50): {tg.es_sec.max().mean() / 60:.1f} (top skater), "
          f"{tg.es_sec.sum().mean() / 60 / 5:.1f} (total / 5)")
    f, d = out[out.pos == "F"], out[out.pos == "D"]
    print("forward lines found per team-game:", f[f.line > 0].groupby(["game_id", "team"]).line.max().value_counts().sort_index().to_dict())
    print("defence pairs found per team-game:", d[d.line > 0].groupby(["game_id", "team"]).line.max().value_counts().sort_index().to_dict())
    print("avg 5-on-5 minutes by forward line:", (f[f.line > 0].groupby("line").es_sec.mean() / 60).round(1).to_dict())
    print("avg share of 5-on-5 with top linemate (F):", round(pd.to_numeric(f.mate_share, errors="coerce").mean(), 3))
    print("power-play minutes per team-game:", round(tg.pp_sec.max().mean() / 60, 1),
          "| PP1 / PP2 skaters per team-game:", (out[out.pp_unit == 1].groupby(["game_id", "team"]).size().mean().round(2),
                                                out[out.pp_unit == 2].groupby(["game_id", "team"]).size().mean().round(2)))
    sample = out[out.game_id == out.game_id.max()]
    names = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"), usecols=["player_id", "name"]).drop_duplicates("player_id")
    nm = dict(zip(names.player_id, names.name))
    print(f"\nlast game ({sample.game_id.iloc[0]}) lines:")
    for (tm, pos, ln), grp in sample[sample.line > 0].groupby(["team", "pos", "line"]):
        print(f"  {tm} {pos}{ln}: " + ", ".join(f"{nm.get(p, p)} {s / 60:.1f}m" for p, s in zip(grp.player_id, grp.es_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
