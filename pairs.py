"""
Correlated Time On Ice pairs: two TOI legs from the same team whose minutes move together.

Only the pair types the history backtest found strongly linked are used (2024-26 regular seasons, each
player's TOI vs his own previous-10-game average; "lift" = how much more often both legs land the same
way than if they were independent):

    F1 + F2   both Over x1.37   both Under x1.32
    F1 + D1   both Over x1.24   both Under x1.20
    D1 + D2   both Over x1.17   both Under x1.15

A pair is listed only when the model picks the SAME direction on both legs. Estimated chance both hit =
leg A x leg B x lift (capped at the smaller leg), using calibrated leg probabilities. Lifts live in
config.json ("pair_lifts") so they can be re-measured later. SOG and hits pairs showed no useful link
(lift ~1.0) and are not paired.

Results tracking (build_results.py): every listed pair is graded from the earliest snapshot of each leg;
a pair counts once both legs are graded hit/miss (a push or void on either leg drops the pair).
"""
import os

import numpy as np
import pandas as pd

from common import HIST_DIR

DEFAULT_LIFTS = {"F1+F2": {"over": 1.37, "under": 1.32},
                 "F1+D1": {"over": 1.24, "under": 1.20},
                 "D1+D2": {"over": 1.17, "under": 1.15}}


def pair_type(ra, rb):
    """'F1+F2' etc. for a pair of tier ranks like 'F2', 'F1', or None if not a linked type."""
    key = "+".join(sorted([ra, rb], key=lambda r: (r[0] != "F", int(r[1:]))))
    return key if key in DEFAULT_LIFTS else None


def estimate(pa, pb, kind, side, lifts):
    lift = (lifts or DEFAULT_LIFTS).get(kind, {}).get(side, 1.0)
    return min(pa * pb * lift, pa, pb), lift


def build_pairs(picks, lifts=None, k=1.0):
    """picks: dicts with game_date, nhl_game_id, team, player, tier_rank, side, prob (raw), [x result].
    Returns one dict per linked same-direction pair."""
    by = {}
    for p in picks:
        if p.get("tier_rank") and p.get("side"):
            by.setdefault((p["game_date"], str(p["nhl_game_id"]), p["team"]), []).append(p)
    out = []
    for (d, gid, team), ps in by.items():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                kind = pair_type(a["tier_rank"], b["tier_rank"])
                if not kind or a["side"] != b["side"]:
                    continue
                if a["tier_rank"] != kind.split("+")[0]:
                    a, b = b, a
                pa, pb = 0.5 + k * (a["prob"] - 0.5), 0.5 + k * (b["prob"] - 0.5)
                est, lift = estimate(pa, pb, kind, a["side"], lifts)
                row = {"d": d, "tm": team, "kind": kind, "s": a["side"], "a": a["player"], "b": b["player"],
                       "pa": round(pa, 4), "pb": round(pb, 4), "lift": lift, "est": round(est, 4)}
                if "x" in a and "x" in b:
                    if a["x"] not in ("h", "m") or b["x"] not in ("h", "m"):
                        continue
                    row["x"] = "h" if a["x"] == "h" and b["x"] == "h" else "m"
                out.append(row)
    return sorted(out, key=lambda r: (r["d"], -r["est"]))


def pregame_ranks():
    """(game_id, player_id) -> 'F1'/'D2'... by average TOI over each player's previous 10 games, ranked
    among the teammates who played that game. Used to tier picks made before tiers were stored."""
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"), usecols=["game_id", "date", "team", "player_id", "position", "toi_sec"])
    sk = sk[sk.toi_sec > 0].sort_values(["player_id", "date", "game_id"])
    sk["avg"] = sk.groupby("player_id").toi_sec.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    sk = sk.dropna(subset=["avg"])
    sk["pos"] = np.where(sk.position == "D", "D", "F")
    sk["rank"] = sk.groupby(["game_id", "team", "pos"]).avg.rank(ascending=False, method="first").astype(int)
    return {(str(g), str(p)): f"{pos}{r}" for g, p, pos, r in zip(sk.game_id, sk.player_id, sk.pos, sk["rank"])}
