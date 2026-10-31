"""
LightGBM challenger ("shadow" model) for Time On Ice and Hits. It never replaces the official picks:
predict.py stores its projection, side and probability next to each official pick, and the Results page
compares the two on real PrizePicks lines. Turn it on/off per prop in config.json ("shadow").

How it works
  * Features: everything from games BEFORE the one being priced (shift(1) everywhere): the player's last
    5/10/30-game averages of TOI, hits, shots, blocks, power-play minutes/share/shots; last game's TOI;
    rest days, back-to-back, home/away, games played (capped); team and opponent shot volume and the
    opponent's penalty tendency over their last 10 games; plus the current model's own projection.
  * Tonight's games are appended to the history as rows with no result, so they get their features by
    exactly the same code path as the training rows (no separate "live" feature code to drift).
  * Retrained every run on all regular-season games so far (a few seconds per prop). The most recent
    15% of dates are held out for early stopping, and the spread used for probabilities (TOI normal sd,
    hits negative-binomial dispersion) is fitted on that held-out slice so it isn't overconfident.

Stage-1 backtest (train 2024-25, test 2025-26+, retrained monthly): beat the current model on TOI
(log-lik -2.345 vs -2.362, MAE 1.90 vs 1.93 min) and hits (-1.327 vs -1.339); tied or lost elsewhere.
"""
import numpy as np
import pandas as pd

VERSION = "lgbm-1"
STATS = ["sog", "hits", "blocked", "toi", "pp_min", "pp_share", "pp_shots"]
BASE = ["is_d", "is_home", "games_before", "games_season", "rest_days", "b2b", "toi_last", "sog_per60_l30",
        "team_sf_l10", "team_sa_l10", "opp_sf_l10", "opp_sa_l10", "opp_pp_drawn_l10"] + \
       [f"{c}_{w}" for c in STATS for w in ("l5", "l10", "l30")]
PROPS = {"toi": {"target": "toi", "cur": "cur_toi", "objective": "regression"},
         "hits": {"target": "hits", "cur": "cur_hits", "objective": "poisson"}}
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)


def build_features(sk, gm, upcoming=None):
    """sk: skater history incl. pp_toi/pp_shots/pp_share. upcoming: DataFrame of tonight's player-games
    (game_id, date, season, team, opp, is_home, player_id, position). Returns one feature row per
    history row with TOI > 0, plus one per upcoming row (targets NaN)."""
    h = sk[sk.toi_sec > 0].copy()
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        for c in ("toi_sec", "sog", "hits", "blocked", "pp_toi", "pp_shots", "pp_share"):
            up[c] = np.nan
        up["game_type"] = 2
        up["_upcoming"] = True
        h = pd.concat([h, up], ignore_index=True)
    h["_upcoming"] = h.get("_upcoming", False)
    h["_upcoming"] = h["_upcoming"].fillna(False).astype(bool)
    h["toi"] = h.toi_sec / 60
    h["pp_min"] = h.pp_toi / 60
    h["is_d"] = (h.position == "D").astype(int)
    h["date_dt"] = pd.to_datetime(h.date)
    h = h.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    g = h.groupby("player_id")
    for c in STATS:
        s = g[c].shift(1)
        for w, n in (("l5", 5), ("l10", 10), ("l30", 30)):
            h[f"{c}_{w}"] = s.groupby(h.player_id).transform(lambda x, n=n: x.rolling(n, min_periods=1).mean())
    h["toi_last"] = g.toi.shift(1)
    h["games_before"] = g.cumcount().clip(upper=50)
    h["games_season"] = h.groupby(["player_id", "season"]).cumcount().clip(upper=82)
    h["rest_days"] = (h.date_dt - g.date_dt.shift(1)).dt.days.clip(upper=10)
    h["b2b"] = (h.rest_days == 1).astype(int)
    h["sog_per60_l30"] = h.sog_l30 / h.toi_l30 * 60

    tg = pd.concat([gm[["game_id", "date", "away", "away_sog", "home_sog"]].rename(columns={"away": "team", "away_sog": "sf", "home_sog": "sa"}),
                    gm[["game_id", "date", "home", "home_sog", "away_sog"]].rename(columns={"home": "team", "home_sog": "sf", "away_sog": "sa"})])
    if upcoming is not None and len(upcoming):
        ug = upcoming[["game_id", "date", "team"]].drop_duplicates()
        tg = pd.concat([tg, ug.assign(sf=np.nan, sa=np.nan)], ignore_index=True)
    pp_team = h[~h._upcoming].groupby(["game_id", "team"]).pp_toi.sum().div(5 * 60).reset_index(name="pp_for")
    tg = tg.merge(pp_team, on=["game_id", "team"], how="left").sort_values(["team", "date", "game_id"])
    for c in ("sf", "sa", "pp_for"):
        tg[f"{c}_l10"] = tg.groupby("team")[c].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    own = tg[["game_id", "team", "sf_l10", "sa_l10"]].rename(columns={"sf_l10": "team_sf_l10", "sa_l10": "team_sa_l10"})
    opp = tg[["game_id", "team", "sf_l10", "sa_l10", "pp_for_l10"]].rename(
        columns={"team": "opp", "sf_l10": "opp_sf_l10", "sa_l10": "opp_sa_l10", "pp_for_l10": "opp_pp_drawn_l10"})
    return h.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left")


def train_and_predict(feat, cur, prop, upcoming_cur):
    """Fit on history rows, predict the upcoming rows. cur: history (game_id, player_id) -> current-model
    projection; upcoming_cur: same for tonight. Returns (predictions Series indexed like upcoming rows,
    spread dict)."""
    import lightgbm as lgb
    from backtest import fit_r
    spec = PROPS[prop]
    y, cname = spec["target"], spec["cur"]
    feats = BASE + [cname]
    f = feat.merge(cur, on=["game_id", "player_id"], how="left")
    upm = f._upcoming
    f.loc[upm, cname] = f.loc[upm].set_index(["game_id", "player_id"]).index.map(upcoming_cur).astype(float)
    hist = f[~upm & (f.game_type == 2) & f[cname].notna() & (f.n_prior >= 5)]
    dates = sorted(hist.date.unique())
    cut = dates[int(len(dates) * 0.85)]
    fit, val = hist[hist.date < cut], hist[hist.date >= cut]
    params = dict(PARAMS, objective=spec["objective"])
    m = lgb.train(params, lgb.Dataset(fit[feats], fit[y]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(val[feats], val[y])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv = m.predict(val[feats], num_iteration=m.best_iteration)
    spread = ({"sd_min": float(np.sqrt(((pv - val[y]) ** 2).mean()))} if spec["objective"] == "regression"
              else {"nb_r": float(fit_r(pv, val[y].values))})
    up = f[upm]
    pred = pd.Series(m.predict(up[feats], num_iteration=m.best_iteration), index=up.set_index(["game_id", "player_id"]).index)
    return pred, spread, {"trained_through": str(hist.date.max()), "rounds": int(m.best_iteration), "n_train": int(len(fit))}


def shadow_predictions(cfg, sk, gm, walker, picks):
    """Adds pick['shadow'] = {model, mean, side, prob, prob_cal} to priced picks in the shadow props."""
    import model
    sh = cfg.get("shadow") or {}
    props = [p for p in sh.get("props", []) if p in PROPS]
    targets = [p for p in picks if p["category"] in props and not p.get("skip_reason") and p.get("nhl_game_id")]
    if not sh.get("enabled") or not targets:
        return {}
    sit = pd.read_csv(f"{model.HIST_DIR}/skater_situational.csv", usecols=["game_id", "player_id", "pp_toi", "pp_shots", "pp_share"])
    sk = sk.merge(sit, on=["game_id", "player_id"], how="left")
    sk[["pp_toi", "pp_shots", "pp_share"]] = sk[["pp_toi", "pp_shots", "pp_share"]].fillna(0)
    last = sk.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    season = int(sk.season.max())
    up_rows, up_cur = {}, {"toi": {}, "hits": {}}
    for p in targets:
        pid, gid = int(p["nhl_id"]), int(p["nhl_game_id"])
        if pid not in last.index:
            continue
        up_rows[(gid, pid)] = {"game_id": gid, "player_id": pid, "date": p["game_date"], "season": season,
                               "team": p["team"], "opp": p["opp"], "is_home": int(bool(p.get("is_home"))),
                               "position": last.loc[pid, "position"]}
        toi, _ = walker.player_toi(pid)
        up_cur["toi"][(gid, pid)] = toi / 60 if toi is not None else np.nan
        up_cur["hits"][(gid, pid)] = walker.player_mu(pid, p["opp"], "hits")[0]
    if not up_rows:
        return {}
    feat = build_features(sk, gm, pd.DataFrame(up_rows.values()))
    sdf, _, _ = model.walk(sk.drop(columns=["pp_toi", "pp_shots", "pp_share"]), gm, cfg["model"])
    cur = sdf[["game_id", "player_id", "n_prior", "toi_proj", "mu_hits"]].copy()
    cur["cur_toi"] = cur.toi_proj / 60
    cur["cur_hits"] = cur.mu_hits
    feat = feat.merge(cur[["game_id", "player_id", "n_prior"]], on=["game_id", "player_id"], how="left")
    feat.loc[feat._upcoming, "n_prior"] = 99      # upcoming rows were already screened by the official model
    info = {}
    for prop in props:
        pred, spread, meta = train_and_predict(feat, cur[["game_id", "player_id", PROPS[prop]["cur"]]], prop, up_cur[prop])
        info[prop] = dict(meta, **spread)
        for p in targets:
            if p["category"] != prop:
                continue
            key = (int(p["nhl_game_id"]), int(p["nhl_id"]))
            if key not in pred.index or not np.isfinite(pred[key]):
                continue
            mu = float(pred[key])
            po, pu, _ = (model.normal_probs(mu, p["line"], spread["sd_min"]) if prop == "toi"
                         else model.line_probs(mu, p["line"], spread["nb_r"]))
            side, prob = model.pick_side(po, pu)
            p["shadow"] = {"model": VERSION, "mean": round(mu, 3), "side": side, "prob": round(prob, 4),
                           "prob_cal": round(model.calibrate(prob, cfg), 4)}
    return info
