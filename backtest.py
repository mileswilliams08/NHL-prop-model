"""
Backtest the SOG model against real outcomes in data/history (no PrizePicks lines exist for
the past, so this tests projections and calibration, NOT whether the model beats the market).

    python backtest.py             # small parameter grid, report, write docs/data/backtest.json
    python backtest.py --apply     # also write the chosen parameters + dispersion into config.json
    python backtest.py --no-grid   # just evaluate config.json's current parameters

Honest split: parameters and the negative-binomial dispersion are chosen on the TRAIN seasons
only (everything before --test-from); the TEST numbers are what the chosen model did on data it
never saw. Every prediction is made from games before that date (see model.walk).

Compared against two naive baselines on the same player-games:
  season-to-date average SOG, and last-10-games average SOG (both shrunk toward position mean).
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import DOCS_DATA, load_config, save_config, write_json, path
from model import load_history, nb_loglik, walk

LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
GRID = {"halflife_games": [10, 20, 40], "prior_toi_sec": [1800, 3600, 7200], "season_carry": [0.5, 0.8]}


def fit_r(mu, y):
    res = minimize_scalar(lambda lr: -nb_loglik(mu, y, float(np.exp(lr))).sum(),
                          bounds=(np.log(1.0), np.log(5000.0)), method="bounded")
    return float(np.exp(res.x))


def add_baselines(df):
    df = df.sort_values(["player_id", "date", "game_id"]).copy()
    g = df.groupby(["player_id", "season"])["sog"]
    cnt = g.cumcount()
    cum = g.cumsum() - df["sog"]
    pos_mean = df.groupby("pos")["sog"].transform("mean")   # mild look-ahead, baseline only
    k = 5
    df["mu_season_avg"] = (cum + k * pos_mean) / (cnt + k)
    last10 = (df.groupby("player_id")["sog"]
                .transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()))
    n10 = df.groupby("player_id")["sog"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count())
    df["mu_last10"] = (last10.fillna(0) + k * pos_mean) / (n10.fillna(0) + k)
    return df.sort_values(["date", "game_id"])


def calib_table(mu, y, r):
    from scipy.stats import nbinom
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    out = {}
    for L in LINES:
        p = 1.0 - nbinom.cdf(int(np.floor(L)), r, r / (r + mu))
        hit = (np.asarray(y) > L).astype(float)
        rows = []
        for lo in np.arange(0, 1.0, 0.1):
            sel = (p >= lo) & (p < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
            if sel.sum() >= 30:
                rows.append({"bucket": f"{int(lo*100)}-{int(lo*100)+10}%", "n": int(sel.sum()),
                             "pred": round(float(p[sel].mean()), 4), "actual": round(float(hit[sel].mean()), 4)})
        out[str(L)] = {"rows": rows, "n": int(len(p)), "pred_mean": round(float(p.mean()), 4),
                       "actual_mean": round(float(hit.mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-from", default="2025-09-01", help="first date of the test period")
    ap.add_argument("--warmup-until", default="2024-11-01", help="skip early-season predictions before this")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    base = dict(cfg["model"])
    sk, gm = load_history(cfg)
    print(f"history: {len(sk)} skater-games, {sk['date'].min()} .. {sk['date'].max()}")

    combos = [{}] if args.no_grid else [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    results = []
    best = None
    for c in combos:
        params = dict(base, **c)
        df, _ = walk(sk, gm, params)
        df = df[(df["n_prior"] >= params["min_prior_games"]) & df["mu"].notna() & (df["date"] >= args.warmup_until)]
        train, test = df[df["date"] < args.test_from], df[df["date"] >= args.test_from]
        if len(train) < 1000:
            print("not enough train rows; is the history loaded?")
            return 1
        r = fit_r(train["mu"].values, train["sog"].values)
        tr_ll = float(nb_loglik(train["mu"], train["sog"], r).mean())
        te_ll = float(nb_loglik(test["mu"], test["sog"], r).mean()) if len(test) else None
        results.append({"params": c, "nb_r": round(r, 2), "train_ll": round(tr_ll, 5),
                        "test_ll": None if te_ll is None else round(te_ll, 5)})
        print(f"  {c or 'config'}  r={r:7.1f}  train LL/obs {tr_ll:.5f}  test LL/obs "
              f"{'-' if te_ll is None else f'{te_ll:.5f}'}")
        if best is None or tr_ll > best[0]:
            best = (tr_ll, params, r, df)

    _, params, r, df = best
    params = dict(params, nb_r=round(r, 2))
    df = add_baselines(df)
    train, test = df[df["date"] < args.test_from], df[df["date"] >= args.test_from]
    evald = test if len(test) >= 1000 else train
    label = "test" if evald is test else "train (no test period yet)"

    def metrics(sub, col):
        r_b = fit_r(train[col].values, train["sog"].values)
        return {"ll_nb": round(float(nb_loglik(sub[col], sub["sog"], r_b).mean()), 5),
                "ll_poisson": round(float(nb_loglik(sub[col], sub["sog"], None).mean()), 5),
                "mae": round(float((sub[col] - sub["sog"]).abs().mean()), 4),
                "bias": round(float((sub[col] - sub["sog"]).mean()), 4)}

    summary = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "model_version": cfg["model_version"],
        "params": params,
        "split": {"train": f"{train['date'].min()} .. {train['date'].max()} ({len(train)} player-games)",
                  "evaluated_on": label,
                  "eval": f"{evald['date'].min()} .. {evald['date'].max()} ({len(evald)} player-games)"},
        "metrics": {"model": metrics(evald, "mu"),
                    "season-to-date avg": metrics(evald, "mu_season_avg"),
                    "last-10 avg": metrics(evald, "mu_last10")},
        "by_position": {g: metrics(sub, "mu") for g, sub in evald.groupby("pos")},
        "calibration": calib_table(evald["mu"].values, evald["sog"].values, r),
        "grid": results,
    }
    write_json(path("docs", "data", "backtest.json"), summary)

    print(f"\nchosen (by train log-likelihood): {params}")
    print(f"evaluated on {label}: {summary['split']['eval']}")
    print(f"{'':22} {'LL/obs (NB)':>12} {'LL (Poisson)':>13} {'MAE':>7} {'bias':>7}")
    for name, m in summary["metrics"].items():
        print(f"{name:22} {m['ll_nb']:12.5f} {m['ll_poisson']:13.5f} {m['mae']:7.3f} {m['bias']:7.3f}")
    print("\ncalibration, P(over line) buckets -> actual over rate:")
    for L, t in summary["calibration"].items():
        cells = "  ".join(f"{row['bucket']}: {row['pred']:.2f}->{row['actual']:.2f} (n{row['n']})" for row in t["rows"])
        print(f"  line {L}: overall pred {t['pred_mean']:.3f} vs actual {t['actual_mean']:.3f} | {cells}")

    if args.apply:
        cfg["model"].update({k: params[k] for k in params if k in cfg["model"]})
        save_config(cfg)
        print("\nwrote chosen parameters to config.json (bump model_version if picks are already being tracked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
