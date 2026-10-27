"""
data/graded_results.csv (+ data/excluded_picks.csv) -> docs/data/results.json for the Results page,
and a printed summary for the Action log.

Every non-void, non-excluded pick goes to the page as a compact row; the page filters and computes
everything in the browser. "Headline" picks (standard line, adjusted_odds False, headline prop) are
kept apart from "experimental" ones everywhere.

The printed summary is the market test from the PGA lessons:
  * model hit rate vs blind Over and blind Under on the same lines
  * calibration by the model's pick probability
  * slope of (actual - line) on (model mean - line): 1 means the model's disagreement with the line
    is fully real, 0 means it carries no signal. Reported with its standard error.
"""
import os

import numpy as np
import pandas as pd

import model
import movement
import pairs
from common import EXCLUDED_PATH, GRADED_PATH, load_config, norm_name, path, write_json

BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]


def load_graded():
    if not os.path.exists(GRADED_PATH):
        return pd.DataFrame()
    df = pd.read_csv(GRADED_PATH, dtype={"nhl_id": str, "nhl_game_id": str})
    for c in ("line", "model_mean", "p_over", "p_under", "prob", "actual"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify(df, cfg):
    """Headline vs experimental is decided from config.json at build time (not the value stored
    when the pick was made), so changing the rule re-sorts every past pick consistently."""
    adj = df["adjusted_odds"].astype(str).str.lower().eq("true")
    head_prop = df["category"].map(lambda c: bool(cfg["props"].get(c, {}).get("headline")))
    df["headline"] = head_prop & ~(adj & bool(cfg.get("headline_requires_unadjusted")))
    return df


def exclusion_keys():
    if not os.path.exists(EXCLUDED_PATH):
        return set()
    ex = pd.read_csv(EXCLUDED_PATH, dtype=str).fillna("")
    return {(r.game_date, norm_name(r.player), r.category.lower()) for r in ex.itertuples()}


def apply_exclusions(df):
    if df.empty or not os.path.exists(EXCLUDED_PATH):
        return df, 0
    keys = exclusion_keys()
    mask = df.apply(lambda r: (str(r["game_date"]), norm_name(r["player"]), str(r["category"]).lower()) in keys, axis=1)
    return df[~mask].copy(), int(mask.sum())


def slope(sub):
    x = (sub["model_mean"] - sub["line"]).to_numpy()
    y = (sub["actual"] - sub["line"]).to_numpy()
    if len(x) < 10 or np.var(x) == 0:
        return None
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + b * (x - xm))
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / sxx))
    return float(b), se


def rate(results):
    dec = [r for r in results if r in ("hit", "miss")]
    return (sum(r == "hit" for r in dec) / len(dec), len(dec)) if dec else (None, 0)


def print_summary(name, sub):
    if sub.empty:
        print(f"{name}: no graded picks yet")
        return
    m, n = rate(sub["result"])
    o, _ = rate(sub["blind_over"])
    u, _ = rate(sub["blind_under"])
    f = lambda v: "-" if v is None else f"{v:.1%}"
    print(f"{name}: n={n}  model {f(m)}  blind Over {f(o)}  blind Under {f(u)}"
          + ("   (n < 100: don't recalibrate yet)" if n < 100 else ""))
    for lo, hi in BUCKETS:
        b = sub[(sub["prob"] >= lo) & (sub["prob"] < hi)]
        if len(b):
            hr, bn = rate(b["result"])
            print(f"   raw {lo:.0%}-{min(hi, 1):.0%}: n={bn:4d}  avg raw prob {b['prob'].mean():.3f}  hit {f(hr)}")
    s = slope(sub)
    if s:
        print(f"   slope (actual-line) on (mean-line): {s[0]:.2f} +/- {1.96 * s[1]:.2f} (95% CI)")


def build_graded_pairs(cfg, graded, k):
    """Correlated TOI pairs from the earliest snapshot of each leg, with each leg's graded result."""
    res = {(str(r.game_date), str(r.nhl_game_id), str(r.nhl_id)): r.result
           for r in graded[graded["category"] == "toi"].itertuples()}
    if not res:
        return []
    ranks = None
    legs = []
    for (d, gid, pid, cat), (pk, _) in movement._earliest_picks().items():
        if cat != "toi" or (d, gid, pid) not in res:
            continue
        tr = pk.get("tier_rank")
        if not tr:
            ranks = ranks if ranks is not None else pairs.pregame_ranks()
            tr = ranks.get((gid, pid))
        legs.append({"game_date": d, "nhl_game_id": gid, "team": pk["team"], "player": pk["player"], "tier_rank": tr,
                     "side": pk["side"], "prob": float(pk["prob"]),
                     "x": {"hit": "h", "miss": "m", "push": "p"}.get(res[(d, gid, pid)], "v")})
    lifts = {kk: v for kk, v in (cfg.get("pair_lifts") or {}).items() if kk != "note"}
    rows = pairs.build_pairs(legs, lifts, k)
    if rows:
        hit = sum(r["x"] == "h" for r in rows)
        print(f"TOI PAIRS: {len(rows)} graded same-direction linked pairs, both hit {hit / len(rows):.1%} "
              f"(avg estimate {np.mean([r['est'] for r in rows]):.1%}; 2-pick 3x breakeven 33.3%)")
    return rows


def main():
    cfg = load_config()
    df = load_graded()
    try:
        moves = movement.build_moves(cfg, exclusion_keys())
    except Exception as e:   # line movement is a bonus; never block the results page
        print(f"line-movement build failed: {e}")
        moves = []
    if df.empty:
        write_json(path("docs", "data", "results.json"),
                   {"generated_at": pd.Timestamp.now("UTC").isoformat(), "picks": [], "voids": 0, "excluded": 0,
                    "moves": moves, "labels": {c: v["label"] for c, v in cfg["props"].items()}})
        print("no graded picks yet; wrote an empty results.json")
        return 0
    df = classify(df, cfg)
    voids = int((df["result"] == "void").sum())
    df = df[df["result"] != "void"]
    df, n_ex = apply_exclusions(df)

    k_used = (cfg.get("calibration") or {}).get("shrink", 1.0)
    dec = df[df["result"].isin(["hit", "miss"]) & df["prob"].notna()]
    cal = {"k_used": k_used, "fitted_through": (cfg.get("calibration") or {}).get("fitted_through")}
    if len(dec) >= 100:
        k, lo, hi = model.fit_shrink(dec["prob"], dec["result"].eq("hit"))
        cal.update(k_fit=round(k, 3), k_lo=round(lo, 3), k_hi=round(hi, 3), n=int(len(dec)))
        print(f"CALIBRATION: using shrink {k_used}; best fit on {len(dec)} graded picks = {k:.2f} (95% {lo:.2f}-{hi:.2f})"
              + ("  <- consider updating config.json" if not lo <= k_used <= hi else ""))

    code = {"hit": "h", "miss": "m", "push": "p"}
    try:
        pair_rows = build_graded_pairs(cfg, df, k_used)
    except Exception as e:   # pairs are a bonus; never block the results page
        print(f"pairs build failed: {e}")
        pair_rows = []
    picks = [{"d": r.game_date, "pl": r.player, "tm": r.team, "op": r.opp, "c": r.category,
              "l": r.line, "m": round(r.model_mean, 3), "s": r.side, "p": round(0.5 + k_used * (r.prob - 0.5), 4), "pr": round(r.prob, 4),
              "a": round(float(r.actual), 2), "x": code[r.result], "ox": code[r.blind_over], "ux": code[r.blind_under],
              "hd": 1 if r.headline else 0, "adj": 1 if str(r.adjusted_odds).lower() == "true" else 0,
              "v": r.model_version}
             for r in df.itertuples()]
    write_json(path("docs", "data", "results.json"),
               {"generated_at": pd.Timestamp.now("UTC").isoformat(), "voids": voids, "excluded": n_ex,
                "labels": {c: v["label"] for c, v in cfg["props"].items()}, "picks": picks,
                "moves": moves, "calibration": cal, "pairs": pair_rows})
    print(f"results.json: {len(picks)} graded picks ({voids} voids, {n_ex} excluded not shown)")
    print("LINE MOVEMENT headline: " + movement.summarize([m for m in moves if m["hd"]]))
    print("LINE MOVEMENT experimental: " + movement.summarize([m for m in moves if not m["hd"]]))
    print_summary("HEADLINE", df[df["headline"]])
    for cat, sub in df[~df["headline"]].groupby("category"):
        print_summary(f"EXPERIMENTAL {cat}", sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
