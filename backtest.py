"""
Backtest the projection models against real outcomes in data/history. No PrizePicks lines exist for
the past, so this tests projections and calibration, NOT whether the model beats the market.

    python backtest.py             # parameter grid (on shots on goal), all props, write docs/data/backtest.json
    python backtest.py --apply     # also write the chosen parameters + per-prop distributions into config.json
    python backtest.py --no-grid   # evaluate config.json's current parameters only

Honest split: parameters and every distribution parameter (NB dispersion, TOI sd) are chosen on the
TRAIN period only (before --test-from); TEST numbers are what the chosen model did on data it never saw.
Every prediction uses only games before its date (see model.walk).

Baselines on the same games: season-to-date average and last-10-games average (shrunk toward the
position mean).
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, norm

from common import load_config, path, save_config, write_json
from model import load_history, nb_loglik, walk

GRID = {"halflife_games": [10, 20, 40, 80], "prior_toi_sec": [3600, 7200, 14400], "season_carry": [0.5, 0.8]}
COUNT_PROPS = {  # prop -> (prediction column, actual column, lines to check calibration at)
    "sog": ("mu_sog", "sog", [0.5, 1.5, 2.5, 3.5, 4.5]),
    "hits": ("mu_hits", "hits", [0.5, 1.5, 2.5, 3.5]),
    "blocks": ("mu_blocks", "blocks", [0.5, 1.5, 2.5]),
}
SAVES_LINES = [19.5, 22.5, 25.5, 28.5, 31.5]
TOI_LINES = [14.5, 17.5, 20.5, 23.5]


def fit_r(mu, y):
    res = minimize_scalar(lambda lr: -nb_loglik(mu, y, float(np.exp(lr))).sum(),
                          bounds=(np.log(0.5), np.log(5000.0)), method="bounded")
    return float(np.exp(res.x))


def add_baselines(df, actual, group_col, k=5):
    """Season-to-date and last-10 averages of `actual`, from earlier games only."""
    df = df.sort_values([group_col, "date", "game_id"]).copy()
    prior = df.groupby("pos")[actual].transform("mean") if "pos" in df else df[actual].mean()
    g = df.groupby([group_col, "season"])[actual]
    cnt, cum = g.cumcount(), g.cumsum() - df[actual]
    df[f"b_season_{actual}"] = (cum + k * prior) / (cnt + k)
    roll = df.groupby(group_col)[actual].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
    n10 = df.groupby(group_col)[actual].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count()).fillna(0)
    df[f"b_last10_{actual}"] = (roll + k * prior) / (n10 + k)
    return df.sort_values(["date", "game_id"])


def calib_rows(p, hit):
    rows = []
    for lo in np.arange(0, 1.0, 0.1):
        sel = (p >= lo) & (p < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if sel.sum() >= 30:
            rows.append({"bucket": f"{int(round(lo*100))}-{int(round(lo*100))+10}%", "n": int(sel.sum()),
                         "pred": round(float(p[sel].mean()), 4), "actual": round(float(hit[sel].mean()), 4)})
    return rows


def count_report(train, test, mu_col, y_col, lines, baselines):
    r = fit_r(train[mu_col].values, train[y_col].values)
    def m(sub, col):
        rb = fit_r(train[col].values, train[y_col].values)
        return {"ll_nb": round(float(nb_loglik(sub[col], sub[y_col], rb).mean()), 5),
                "mae": round(float((sub[col] - sub[y_col]).abs().mean()), 4),
                "bias": round(float((sub[col] - sub[y_col]).mean()), 4)}
    metrics = {"model": m(test, mu_col)}
    metrics.update({name: m(test, col) for name, col in baselines.items()})
    mu = np.maximum(test[mu_col].values, 1e-6)
    cal = {}
    for L in lines:
        p = 1.0 - nbinom.cdf(int(np.floor(L)), r, r / (r + mu))
        hit = (test[y_col].values > L).astype(float)
        cal[str(L)] = {"rows": calib_rows(p, hit), "n": int(len(p)),
                       "pred_mean": round(float(p.mean()), 4), "actual_mean": round(float(hit.mean()), 4)}
    return {"dist": {"nb_r": round(r, 2)}, "n_test": int(len(test)), "metrics": metrics, "calibration": cal}


def toi_report(train, test, baselines):
    sd = float(np.sqrt(((train["toi_proj"] - train["toi_sec"]) / 60).pow(2).mean()))
    def m(sub, col):
        e = (sub[col] - sub["toi_sec"]) / 60
        sdb = float(np.sqrt(((train[col] - train["toi_sec"]) / 60).pow(2).mean()))
        return {"ll_normal": round(float(norm.logpdf(e, 0, sdb).mean()), 5),
                "mae": round(float(e.abs().mean()), 4), "bias": round(float(e.mean()), 4)}
    metrics = {"model": m(test, "toi_proj")}
    metrics.update({name: m(test, col) for name, col in baselines.items()})
    mu, y = test["toi_proj"].values / 60, test["toi_sec"].values / 60
    cal = {}
    for L in TOI_LINES:
        p = 1.0 - norm.cdf((L - mu) / sd)
        hit = (y > L).astype(float)
        cal[str(L)] = {"rows": calib_rows(p, hit), "n": int(len(p)),
                       "pred_mean": round(float(p.mean()), 4), "actual_mean": round(float(hit.mean()), 4)}
    return {"dist": {"sd_min": round(sd, 3)}, "n_test": int(len(test)), "metrics": metrics, "calibration": cal,
            "units": "minutes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-from", default="2025-09-01")
    ap.add_argument("--warmup-until", default="2024-11-01")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    base = dict(cfg["model"])
    sk, gm, gk = load_history(cfg)
    print(f"history: {len(sk)} skater-games, {len(gk)} goalie rows, {sk['date'].min()} .. {sk['date'].max()}")

    def eligible(df):
        return df[(df["n_prior"] >= base["min_prior_games"]) & df["mu_sog"].notna() & (df["date"] >= a.warmup_until)]

    # ---- parameter grid, chosen on shots-on-goal train log-likelihood
    combos = [{}] if a.no_grid else [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    grid, best = [], None
    for c in combos:
        params = dict(base, **c)
        sdf, _, _ = walk(sk, gm, params, collect=True)
        sdf = eligible(sdf)
        train = sdf[sdf["date"] < a.test_from]
        if len(train) < 1000:
            print("not enough train rows; is the history loaded?")
            return 1
        r = fit_r(train["mu_sog"].values, train["sog"].values)
        tr = float(nb_loglik(train["mu_sog"], train["sog"], r).mean())
        test = sdf[sdf["date"] >= a.test_from]
        te = float(nb_loglik(test["mu_sog"], test["sog"], r).mean()) if len(test) else None
        grid.append({"params": c, "nb_r": round(r, 2), "train_ll": round(tr, 5), "test_ll": None if te is None else round(te, 5)})
        print(f"  {c or 'config'}  r={r:6.1f}  train {tr:.5f}  test {'-' if te is None else f'{te:.5f}'}")
        # ties (e.g. season_carry, which one training season can't distinguish) keep the earlier combo
        if best is None or tr > best[0] + 1e-6:
            best = (tr, params)

    params = best[1]
    sdf, gdf, _ = walk(sk, gm, params, gk=gk, collect=True)
    sdf = eligible(sdf)
    report = {"generated_at": pd.Timestamp.now("UTC").isoformat(), "model_version": cfg["model_version"],
              "params": {k: params[k] for k in params if k != "nb_r"}, "grid": grid, "props": {}}

    sdf = add_baselines(add_baselines(add_baselines(add_baselines(sdf, "sog", "player_id"), "hits", "player_id"),
                                      "blocks", "player_id"), "toi_sec", "player_id")
    train, test = sdf[sdf["date"] < a.test_from], sdf[sdf["date"] >= a.test_from]
    if len(test) < 1000:
        test = train
    report["split"] = {"train": f"{train['date'].min()} .. {train['date'].max()} ({len(train)} skater-games)",
                       "eval": f"{test['date'].min()} .. {test['date'].max()} ({len(test)} skater-games)",
                       "evaluated_on": "test" if test is not train else "train (no test period yet)"}
    for prop, (mu_col, y_col, lines) in COUNT_PROPS.items():
        report["props"][prop] = count_report(train, test, mu_col, y_col, lines,
                                             {"season-to-date avg": f"b_season_{y_col}", "last-10 avg": f"b_last10_{y_col}"})
    report["props"]["toi"] = toi_report(train, test, {"season-to-date avg": "b_season_toi_sec",
                                                      "last-10 avg": "b_last10_toi_sec"})

    gdf = gdf[(gdf["n_prior"] >= 3) & gdf["mu_saves"].notna() & (gdf["date"] >= a.warmup_until)].copy()
    gdf["pos"] = "G"
    gdf = add_baselines(gdf, "saves", "player_id", k=3)
    gtr, gte = gdf[gdf["date"] < a.test_from], gdf[gdf["date"] >= a.test_from]
    if len(gte) < 200:
        gte = gtr
    if len(gtr) >= 200:
        report["props"]["saves"] = count_report(gtr, gte, "mu_saves", "saves", SAVES_LINES,
                                                {"season-to-date avg": "b_season_saves", "last-10 avg": "b_last10_saves"})
        report["props"]["saves"]["note"] = "starting goalies only"

    write_json(path("docs", "data", "backtest.json"), report)

    print(f"\nchosen parameters (by shots-on-goal train log-likelihood): {report['params']}")
    print(f"evaluated on {report['split']['evaluated_on']}: {report['split']['eval']}")
    for prop, rep in report["props"].items():
        ll = "ll_normal" if prop == "toi" else "ll_nb"
        print(f"\n[{prop}] {rep['dist']}  n_test={rep['n_test']}")
        for name, m in rep["metrics"].items():
            print(f"   {name:20} LL {m[ll]:9.5f}   MAE {m['mae']:7.3f}   bias {m['bias']:7.3f}")
        for L, t in rep["calibration"].items():
            cells = "  ".join(f"{x['bucket']}: {x['pred']:.2f}->{x['actual']:.2f}" for x in t["rows"])
            print(f"   line {L}: pred {t['pred_mean']:.3f} vs actual {t['actual_mean']:.3f} | {cells}")

    if a.apply:
        cfg["model"].update({k: params[k] for k in params if k in cfg["model"] and k != "nb_r"})
        cfg["model"]["nb_r"] = report["props"]["sog"]["dist"]["nb_r"]
        cfg["dist"] = {prop: rep["dist"] for prop, rep in report["props"].items()}
        save_config(cfg)
        print("\nwrote parameters + per-prop distributions to config.json "
              "(bump model_version if picks are already being tracked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
