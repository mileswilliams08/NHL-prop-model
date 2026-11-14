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
  * Combo props (PrizePicks "Shots On Goal (Combo)", "Goalie Saves (Combo)") go to their own log,
    data/graded_combos.csv, keyed by (date, prop, the legs' NHL ids). A combo is graded once every leg's game
    is final; if any leg didn't play it is void.
  * Each book is graded into its own log from its own snapshots (PrizePicks: data/graded_results.csv,
    Underdog: data/underdog/graded_results.csv). Underdog rows also keep the picked side's payout
    multiplier and line type (standard / goblin / demon) and which sides were sold.

    python grade.py
"""
import csv
import glob
import json
import os
from datetime import date

import pandas as pd

from common import BOOKS, GRADED_PATH, HIST_DIR, SNAP_DIR

FIELDS = ["game_date", "nhl_game_id", "nhl_id", "category", "player", "team", "opp", "position",
          "line", "odds_type", "adjusted_odds", "headline", "model_mean", "p_over", "p_under",
          "side", "prob", "actual", "toi_sec", "result", "blind_over", "blind_under",
          "snapshot_at", "model_version", "pp_proj_id"]
UD_FIELDS = FIELDS + ["line_type", "mult", "mult_over", "mult_under", "sides"]


def key_of(r):
    return (str(r["game_date"]), str(int(float(r["nhl_game_id"]))), str(int(float(r["nhl_id"]))), str(r["category"]))


def outcome(actual, line, side):
    if actual == line:
        return "push"
    return "hit" if (actual > line) == (side == "over") else "miss"


def load_logged(graded_path=GRADED_PATH):
    if not os.path.exists(graded_path):
        return set()
    with open(graded_path, newline="", encoding="utf-8") as f:
        return {key_of(r) for r in csv.DictReader(f)}


COMBO_PATH = os.path.join(os.path.dirname(GRADED_PATH), "graded_combos.csv")
COMBO_FIELDS = ["game_date", "category", "combo_key", "player", "team", "opp", "line", "odds_type", "model_mean",
                "p_over", "p_under", "side", "prob", "actual", "result", "blind_over", "blind_under", "legs",
                "snapshot_at", "model_version", "pp_proj_id"]


def grade_combos(sources, final_games):
    """Combo picks from the PrizePicks snapshots -> data/graded_combos.csv."""
    first = {}
    for p in sorted(glob.glob(os.path.join(SNAP_DIR, "*.json"))):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        for pk in s.get("picks", []):
            legs = pk.get("legs") or []
            if not pk.get("combo") or pk.get("skip_reason") or pk.get("side") is None or not legs \
                    or not all(l.get("nhl_id") and l.get("nhl_game_id") for l in legs):
                continue
            k = (pk["game_date"], pk["category"], "+".join(sorted(str(int(float(l["nhl_id"]))) for l in legs)))
            if k not in first or s["generated_at"] < first[k][1]:
                first[k] = (pk, s["generated_at"], s.get("model_version"))
    logged = set()
    if os.path.exists(COMBO_PATH):
        with open(COMBO_PATH, newline="", encoding="utf-8") as f:
            logged = {(r["game_date"], r["category"], r["combo_key"]) for r in csv.DictReader(f)}
    new_rows, pending = [], 0
    for k, (pk, snap_at, mver) in sorted(first.items()):
        if k in logged:
            continue
        leg_cat = "saves" if k[1] == "saves_combo" else "sog"
        row = {f: pk.get(f, "") for f in COMBO_FIELDS}
        row.update(combo_key=k[2], snapshot_at=snap_at, model_version=mver or "")
        gids = [str(int(float(l["nhl_game_id"]))) for l in pk["legs"]]
        if not all(g in final_games for g in gids):
            if (date.today() - date.fromisoformat(k[0])).days > 3:
                row.update(result="void", actual="", legs=json.dumps([l["name"] for l in pk["legs"]]))
                new_rows.append(row)
            else:
                pending += 1
            continue
        got = [sources[leg_cat].get((g, str(int(float(l["nhl_id"]))))) for g, l in zip(gids, pk["legs"])]
        detail = [{"name": l["name"], "actual": (None if h is None or h[0] <= 0 else int(h[1]))} for l, h in zip(pk["legs"], got)]
        row["legs"] = json.dumps(detail)
        if any(h is None or h[0] <= 0 for h in got):
            row.update(result="void", actual="", blind_over="", blind_under="")
        else:
            actual, line = float(sum(h[1] for h in got)), float(pk["line"])
            row.update(actual=int(actual), result=outcome(actual, line, pk["side"]),
                       blind_over=outcome(actual, line, "over"), blind_under=outcome(actual, line, "under"))
        new_rows.append(row)
    if new_rows:
        exists = os.path.exists(COMBO_PATH)
        with open(COMBO_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COMBO_FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    res = pd.Series([r["result"] for r in new_rows]).value_counts().to_dict() if new_rows else {}
    print(f"combos: graded {len(new_rows)} new {res}; {pending} still waiting on a final game")


def main():
    for book in BOOKS.values():
        if book["key"] != "prizepicks":
            print(f"--- {book['label']} ---")
            try:
                grade_book(book)
            except Exception as e:   # a second book must never break the first
                print(f"{book['label']} grading failed (PrizePicks unaffected): {e!r}")
        else:
            grade_book(book)
    return 0


def grade_book(book):
    snap_dir, graded_path = book["snap_dir"], book["graded"]
    fields = UD_FIELDS if book["key"] == "underdog" else FIELDS
    snaps = sorted(glob.glob(os.path.join(snap_dir, "*.json")))
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

    logged = load_logged(graded_path)
    todo = {k: v for k, v in first.items() if k not in logged}
    if not todo:
        print(f"nothing new to grade ({len(first)} picks archived, all logged)")
        if book["key"] != "prizepicks":   # PrizePicks carries on to grade its combos
            return 0

    gm = pd.read_csv(os.path.join(HIST_DIR, "games.csv"), usecols=["game_id"])
    final_games = set(gm["game_id"].astype("int64").astype(str))
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"))
    gk = pd.read_csv(os.path.join(HIST_DIR, "goalie_games.csv"))
    def index(df, col, scale=1.0):
        vals = pd.to_numeric(df[col], errors="coerce") / scale
        return {(str(g), str(p)): (t, v) for g, p, t, v in
                zip(df["game_id"], df["player_id"], df["toi_sec"], vals)}
    # category -> {(game_id, player_id): (toi_sec, actual)}; TOI props are graded in minutes
    sources = {"sog": index(sk, "sog"), "hits": index(sk, "hits"), "blocks": index(sk, "blocked"),
               "toi": index(sk, "toi_sec", 60.0), "saves": index(gk, "saves")}
    fo_path = os.path.join(HIST_DIR, "skater_faceoffs.csv")
    if os.path.exists(fo_path):
        fo = pd.read_csv(fo_path, usecols=["game_id", "player_id", "fo_won"])
        covered = sk[sk["game_id"].isin(set(fo["game_id"]))].merge(fo, on=["game_id", "player_id"], how="left")
        covered["fo_won"] = covered["fo_won"].fillna(0)
        sources["faceoffs"] = index(covered, "fo_won")

    if book["key"] == "prizepicks":
        try:
            grade_combos(sources, final_games)
        except Exception as e:   # combos are a bonus; never block the singles
            print(f"combo grading failed: {e!r}")
    new_rows, pending = [], 0
    for k, (pk, snap_at, mver) in sorted(todo.items()):
        gid, pid, cat = k[1], k[2], k[3]
        row = {f: pk.get(f, "") for f in fields}
        if isinstance(row.get("sides"), list):
            row["sides"] = "+".join(row["sides"])
        row.update(nhl_game_id=gid, nhl_id=pid, snapshot_at=snap_at, model_version=mver or "")
        if gid not in final_games:
            if (date.today() - date.fromisoformat(k[0])).days > 3:   # postponed / never played
                row.update(result="void", actual="", toi_sec="", blind_over="", blind_under="")
                new_rows.append(row)
            else:
                pending += 1
            continue
        if cat not in sources:
            continue
        hit = sources[cat].get((gid, pid))
        if hit is None or hit[0] <= 0:
            row.update(result="void", actual="", toi_sec=hit[0] if hit else 0, blind_over="", blind_under="")
        else:
            actual, line = float(hit[1]), float(pk["line"])
            row.update(actual=round(actual, 2) if cat == "toi" else int(actual), toi_sec=int(hit[0]), result=outcome(actual, line, pk["side"]),
                       blind_over=outcome(actual, line, "over"), blind_under=outcome(actual, line, "under"))
        new_rows.append(row)

    if new_rows:
        exists = os.path.exists(graded_path)
        os.makedirs(os.path.dirname(graded_path), exist_ok=True)
        with open(graded_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    res = pd.Series([r["result"] for r in new_rows]).value_counts().to_dict() if new_rows else {}
    print(f"graded {len(new_rows)} new picks {res}; {pending} still waiting on a final game")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
