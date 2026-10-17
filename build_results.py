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


def apply_exclusions(df):
    if df.empty or not os.path.exists(EXCLUDED_PATH):
        return df, 0
    ex = pd.read_csv(EXCLUDED_PATH, dtype=str).fillna("")
    keys = {(r.game_date, norm_name(r.player), r.category.lower()) for r in ex.itertuples()}
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
            print(f"   {lo:.0%}-{min(hi, 1):.0%}: n={bn:4d}  avg prob {b['prob'].mean():.3f}  hit {f(hr)}")
    s = slope(sub)
    if s:
        print(f"   slope (actual-line) on (mean-line): {s[0]:.2f} +/- {1.96 * s[1]:.2f} (95% CI)")


def main():
    cfg = load_config()
    df = load_graded()
    if df.empty:
        write_json(path("docs", "data", "results.json"),
                   {"generated_at": pd.Timestamp.now("UTC").isoformat(), "picks": [], "voids": 0, "excluded": 0,
                    "labels": {c: v["label"] for c, v in cfg["props"].items()}})
        print("no graded picks yet; wrote an empty results.json")
        return 0
    df = classify(df, cfg)
    voids = int((df["result"] == "void").sum())
    df = df[df["result"] != "void"]
    df, n_ex = apply_exclusions(df)

    code = {"hit": "h", "miss": "m", "push": "p"}
    picks = [{"d": r.game_date, "pl": r.player, "tm": r.team, "op": r.opp, "c": r.category,
              "l": r.line, "m": round(r.model_mean, 3), "s": r.side, "p": round(r.prob, 4),
              "a": int(r.actual), "x": code[r.result], "ox": code[r.blind_over], "ux": code[r.blind_under],
              "hd": 1 if r.headline else 0, "adj": 1 if str(r.adjusted_odds).lower() == "true" else 0,
              "v": r.model_version}
             for r in df.itertuples()]
    write_json(path("docs", "data", "results.json"),
               {"generated_at": pd.Timestamp.now("UTC").isoformat(), "voids": voids, "excluded": n_ex,
                "labels": {c: v["label"] for c, v in cfg["props"].items()}, "picks": picks})
    print(f"results.json: {len(picks)} graded picks ({voids} voids, {n_ex} excluded not shown)")
    print_summary("HEADLINE", df[df["headline"]])
    print_summary("EXPERIMENTAL", df[~df["headline"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
