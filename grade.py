"""
Grade archived picks against finished games in data/history.

  * A pick is graded only once its NHL game is final AND the player actually played
    (appears in the boxscore with TOI > 0). A finished game the player didn't play in is
    logged as "void" (PrizePicks voids DNPs) and never counts.
  * Dedup key: (game_date, nhl_game_id, nhl_id, category). When several snapshots carried the
    same pick (multiple runs before puck drop), the EARLIEST snapshot's line and probabilities
    are the ones graded.
  * data/graded_results.csv is append-only; reruns skip keys already logged.
  * Every row also stores what a blind Over and a blind Under on that line would have done.

    python grade.py
"""
import csv
import glob
import json
import os
from datetime import date

import pandas as pd

from common import GRADED_PATH, HIST_DIR, SNAP_DIR

FIELDS = ["game_date", "nhl_game_id", "nhl_id", "category", "player", "team", "opp", "position",
          "line", "odds_type", "adjusted_odds", "headline", "model_mean", "p_over", "p_under",
          "side", "prob", "actual", "toi_sec", "result", "blind_over", "blind_under",
          "snapshot_at", "model_version", "pp_proj_id"]


def key_of(r):
    return (str(r["game_date"]), str(int(float(r["nhl_game_id"]))), str(int(float(r["nhl_id"]))), str(r["category"]))


def outcome(actual, line, side):
    if actual == line:
        return "push"
    return "hit" if (actual > line) == (side == "over") else "miss"


def load_logged():
    if not os.path.exists(GRADED_PATH):
        return set()
    with open(GRADED_PATH, newline="", encoding="utf-8") as f:
        return {key_of(r) for r in csv.DictReader(f)}


def main():
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "*.json")))
    if not snaps:
        print("no snapshots yet")
        return 0
    first = {}
    for p in snaps:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        for pk in s.get("picks", []):
            if pk.get("skip_reason") or pk.get("side") is None or not pk.get("nhl_game_id") or not pk.get("nhl_id"):
                continue
            k = key_of(pk)
            if k not in first or s["generated_at"] < first[k][1]:
                first[k] = (pk, s["generated_at"], s.get("model_version"))

    logged = load_logged()
    todo = {k: v for k, v in first.items() if k not in logged}
    if not todo:
        print(f"nothing new to grade ({len(first)} picks archived, all logged)")
        return 0

    gm = pd.read_csv(os.path.join(HIST_DIR, "games.csv"), usecols=["game_id"])
    final_games = set(gm["game_id"].astype("int64").astype(str))
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"), usecols=["game_id", "player_id", "toi_sec", "sog"])
    gk = pd.read_csv(os.path.join(HIST_DIR, "goalie_games.csv"), usecols=["game_id", "player_id", "toi_sec", "saves"])
    sk_idx = {(str(g), str(p)): (t, v) for g, p, t, v in sk.itertuples(index=False)}
    gk_idx = {(str(g), str(p)): (t, v) for g, p, t, v in gk.itertuples(index=False)}
    sources = {"sog": sk_idx, "saves": gk_idx}

    new_rows, pending = [], 0
    for k, (pk, snap_at, mver) in sorted(todo.items()):
        gid, pid, cat = k[1], k[2], k[3]
        row = {f: pk.get(f, "") for f in FIELDS}
        row.update(nhl_game_id=gid, nhl_id=pid, snapshot_at=snap_at, model_version=mver or "")
        if gid not in final_games:
            if (date.today() - date.fromisoformat(k[0])).days > 3:   # postponed / never played
                row.update(result="void", actual="", toi_sec="", blind_over="", blind_under="")
                new_rows.append(row)
            else:
                pending += 1
            continue
        hit = sources[cat].get((gid, pid))
        if hit is None or hit[0] <= 0:
            row.update(result="void", actual="", toi_sec=hit[0] if hit else 0, blind_over="", blind_under="")
        else:
            actual, line = float(hit[1]), float(pk["line"])
            row.update(actual=int(actual), toi_sec=int(hit[0]), result=outcome(actual, line, pk["side"]),
                       blind_over=outcome(actual, line, "over"), blind_under=outcome(actual, line, "under"))
        new_rows.append(row)

    if new_rows:
        exists = os.path.exists(GRADED_PATH)
        os.makedirs(os.path.dirname(GRADED_PATH), exist_ok=True)
        with open(GRADED_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    res = pd.Series([r["result"] for r in new_rows]).value_counts().to_dict() if new_rows else {}
    print(f"graded {len(new_rows)} new picks {res}; {pending} still waiting on a final game")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
