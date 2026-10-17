"""
Daily prediction run: PrizePicks board -> model probabilities -> archived snapshot + site data.

Guards (lessons from the PGA version):
  * Only lines for games starting more than min_minutes_before_start from now are used, so a run
    during games never prices a game in progress.
  * Only odds_type "standard", single-stat, non-combo, pre_game lines are modeled. Goblins/demons
    are counted but never priced. Standard lines with adjusted_odds=True are priced and tracked
    but flagged non-headline.
  * If there are no eligible standard lines for an enabled prop, NOTHING is written: the site keeps
    its previous data and no snapshot is archived.
  * Every run that does have lines archives one snapshot, data/snapshots/<run UTC time>.json,
    stamped with the slate date(s) it predicted. The PrizePicks line is stored with every pick.

    python predict.py
"""
from collections import Counter
from datetime import timedelta

from common import DOCS_DATA, SNAP_DIR, get_json, load_config, nhl_team, parse_ts, path, utcnow, write_json, NHL_BASE
import model
import player_map
import prizepicks

MODELED = {"sog", "hits", "blocks", "toi", "saves"}


def nhl_games_by_team(dates):
    """{(date, team): {game_id, start_utc, home, away}} from the NHL schedule for those dates."""
    out = {}
    for d in sorted(dates):
        sched = get_json(f"{NHL_BASE}/schedule/{d}") or {}
        for day in sched.get("gameWeek", []):
            if day.get("date") != d:
                continue
            for g in day.get("games", []):
                info = {"game_id": g["id"], "start_utc": g.get("startTimeUTC"),
                        "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
                        "state": g.get("gameState")}
                out[(d, info["away"])] = info
                out[(d, info["home"])] = info
    return out


def main():
    cfg = load_config()
    now = utcnow()
    cutoff = now + timedelta(minutes=cfg["min_minutes_before_start"])
    enabled = {c for c, v in cfg["props"].items() if v["enabled"]}

    try:
        rows = prizepicks.parse_board(prizepicks.fetch_board(cfg["league_id"]), cfg)
    except Exception as e:
        print(f"PrizePicks fetch failed: {e}\nKeeping existing site data; nothing archived.")
        return 0

    cand = [r for r in rows if r["category"] in enabled]
    print(f"board: {len(rows)} projections, {len(cand)} in enabled props {sorted(enabled)}")
    print("  by odds_type/adjusted:", dict(Counter((r["odds_type"], r["adjusted_odds"]) for r in cand)))

    def eligible(r):
        return (r["odds_type"] == "standard" and not r["combo"] and r["status"] == "pre_game"
                and not r["is_live"] and (r["projection_type"] in (None, "Single Stat"))
                and r["start_time"] and parse_ts(r["start_time"]) > cutoff)

    elig = [r for r in cand if eligible(r)]
    started = sum(1 for r in cand if r["odds_type"] == "standard" and r["start_time"]
                  and parse_ts(r["start_time"]) <= cutoff)
    print(f"  eligible standard lines: {len(elig)} (skipped {started} at/after puck drop)")
    if not elig:
        print("No eligible standard lines (board empty, not posted yet, or every game has started).\n"
              "Keeping existing docs/data/props.json; no snapshot archived.")
        return 0

    nhl_ids = player_map.resolve(elig, cfg)
    slate_dates = sorted({r["start_time"][:10] for r in elig})   # PrizePicks times are US Eastern
    games = nhl_games_by_team(slate_dates)

    sk, gm, gk = model.load_history(cfg)
    params = cfg["model"]
    _, _, walker = model.walk(sk, gm, params, gk=gk, collect=False)
    hist_through = str(sk["date"].max())

    picks, reasons = [], Counter()
    for r in elig:
        team, opp = nhl_team(r["team"], cfg), nhl_team(r["opp"], cfg)
        d = r["start_time"][:10]
        g = games.get((d, team))
        nhl_id = nhl_ids.get(r["pp_player_id"], "")
        headline = cfg["props"][r["category"]]["headline"] and not (
            cfg["headline_requires_unadjusted"] and r["adjusted_odds"])
        pick = {"game_date": d, "start_time": r["start_time"], "nhl_game_id": g["game_id"] if g else None,
                "player": r["player"], "team": team, "opp": opp, "position": r["position"],
                "pp_player_id": r["pp_player_id"], "nhl_id": nhl_id, "pp_proj_id": r["pp_proj_id"],
                "category": r["category"], "line": r["line"], "odds_type": r["odds_type"],
                "adjusted_odds": r["adjusted_odds"], "headline": bool(headline),
                "board_time": r["board_time"], "updated_at": r["updated_at"],
                "model_mean": None, "p_over": None, "p_under": None, "p_push": None,
                "side": None, "prob": None, "n_prior": None, "skip_reason": None}
        reason = None
        if not nhl_id:
            reason = "unmatched_player"
        elif not g:
            reason = "no_nhl_game"
        elif r["category"] not in MODELED:
            reason = "no_model_yet"
        else:
            cat = r["category"]
            if cat == "saves":
                mu, n = walker.goalie_mu(nhl_id, team, opp)
                min_n = params.get("min_prior_starts", 3)
            elif cat == "toi":
                toi, n = walker.player_toi(nhl_id)
                mu, min_n = (toi / 60 if toi is not None else None), params["min_prior_games"]
            else:
                mu, n = walker.player_mu(nhl_id, opp, cat)
                min_n = params["min_prior_games"]
            pick["n_prior"] = n
            if mu is None or n < min_n:
                reason = "insufficient_history"
            else:
                d = model.dist_for(cfg, cat)
                po, pu, pp = (model.normal_probs(mu, r["line"], d) if cat == "toi"
                              else model.line_probs(mu, r["line"], d))
                side, prob = model.pick_side(po, pu)
                pick.update(model_mean=round(mu, 4), p_over=round(po, 4), p_under=round(pu, 4),
                            p_push=round(pp, 4), side=side, prob=round(prob, 4))
        pick["skip_reason"] = reason
        reasons[reason or "modeled"] += 1
        picks.append(pick)

    print("  picks:", dict(reasons))
    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    snapshot = {"generated_at": now.isoformat(), "slate_dates": slate_dates,
                "model_version": cfg["model_version"], "params": params,
                "history_through": hist_through,
                "board_counts": {"rows": len(rows), "enabled_props": len(cand), "eligible": len(elig),
                                 "odds_types": {f"{k[0]}|adj={k[1]}": v for k, v in
                                                Counter((r["odds_type"], r["adjusted_odds"]) for r in cand).items()}},
                "picks": picks}
    write_json(path("data", "snapshots", f"{stamp}.json"), snapshot)
    write_json(path("docs", "data", "props.json"),
               {k: snapshot[k] for k in ("generated_at", "slate_dates", "model_version", "history_through")}
               | {"labels": {c: v["label"] for c, v in cfg["props"].items()}, "picks": picks})
    print(f"archived data/snapshots/{stamp}.json and wrote docs/data/props.json "
          f"(slate {', '.join(slate_dates)}, history through {hist_through})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
