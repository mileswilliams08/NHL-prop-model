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

Underdog (underdog.py) is priced by the same code as a separate book with its own snapshots and
site file (data/underdog/snapshots, docs/data/props_underdog.json). Differences there: every line
is priced whatever its payout (standard, goblin or demon, tagged by type), and the model can only
pick a side Underdog actually sells. An Underdog failure never affects the PrizePicks run.

    python predict.py
"""
import glob
import json
from collections import Counter
from datetime import timedelta

import os

from common import BOOKS, DOCS_DATA, SNAP_DIR, get_json, load_config, nhl_team, parse_ts, path, utcnow, write_json, NHL_BASE
import gbm
import model
import player_map
import prizepicks
import tiers
import underdog

MODELED = {"sog", "hits", "blocks", "toi", "saves", "faceoffs"}


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


def add_open_lines(picks, snap_dir=SNAP_DIR):
    """Line-move flag: for picks whose line differs from the first line an earlier model run priced today
    (same player, prop and date), store that opening line/side/prob. Same opening line grading uses."""
    first = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        for pk in s.get("picks", []):
            if pk.get("skip_reason") or pk.get("side") is None:
                continue
            k = (pk["game_date"], str(pk["pp_player_id"]), pk["category"])
            if k not in first or s["generated_at"] < first[k][1]:
                first[k] = (pk, s["generated_at"])
    n = 0
    for p in picks:
        o = first.get((p["game_date"], str(p["pp_player_id"]), p["category"]))
        if p.get("side") is None or not o or float(o[0]["line"]) == float(p["line"]):
            continue
        pk = o[0]
        p.update(open_line=pk["line"], open_side=pk["side"], open_prob=pk["prob"],
                 open_prob_cal=pk.get("prob_cal"), open_mean=pk["model_mean"], open_at=o[1])
        n += 1
    print(f"  {n} picks' lines moved since the first model run of the day")


def pp_rows(cfg, cutoff):
    """PrizePicks board -> (all rows, rows in enabled props, eligible rows)."""
    enabled = {c for c, v in cfg["props"].items() if v["enabled"]}
    rows = prizepicks.parse_board(prizepicks.fetch_board(cfg["league_id"]), cfg)
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
    return rows, cand, elig


def ud_rows(cfg, cutoff):
    """Underdog board -> (all rows, rows in enabled props, eligible rows). Every payout type is kept."""
    enabled = {c for c, v in cfg["props"].items() if v["enabled"]}
    rows, info = underdog.parse_board(underdog.fetch_board(), cfg)
    cand = [r for r in rows if r["category"] in enabled]
    print(f"board: {info['nhl_lines']} NHL lines, {len(cand)} in enabled props; standard multiplier "
          f"{info['standard_multiplier']} ({info['standard_source']})")
    if info["unmapped_stats"]:
        print("  NHL stats not tracked:", dict(info["unmapped_stats"].most_common(12)))
    elig = [r for r in cand if r["status"] == "pre_game" and parse_ts(r["start_time"]) > cutoff]
    elig, dropped = underdog.dedupe(elig)
    print(f"  eligible lines: {len(elig)} (skipped {len(cand) - len(elig) - dropped} at/after puck drop"
          + (f", {dropped} extra lines for a player/prop already listed" if dropped else "") + ")")
    print("  by type of each side sold:", dict(Counter(t for r in elig for t in (r["type_over"], r["type_under"]) if t)))
    return rows, cand, elig


def main():
    cfg = load_config()
    now = utcnow()
    state = {}

    def shared():   # history + model walk, loaded once and only if some book has lines
        if not state:
            sk, gm, gk = model.load_history(cfg)
            _, _, walker = model.walk(sk, gm, cfg["model"], gk=gk, collect=False)
            state.update(sk=sk, gm=gm, walker=walker, tier_of=tiers.team_tiers(sk), hist_through=str(sk["date"].max()))
        return state

    run_book(BOOKS["prizepicks"], cfg, now, shared)
    if (cfg.get("underdog") or {}).get("enabled"):
        print("\n--- Underdog ---")
        try:
            run_book(BOOKS["underdog"], cfg, now, shared)
        except Exception as e:   # a second book must never break the first
            print(f"Underdog run failed (PrizePicks unaffected): {e!r}")
    return 0


def run_book(book, cfg, now, shared):
    ud = book["key"] == "underdog"
    name = book["label"]
    cutoff = now + timedelta(minutes=cfg["min_minutes_before_start"])
    try:
        rows, cand, elig = (ud_rows if ud else pp_rows)(cfg, cutoff)
    except Exception as e:
        print(f"{name} fetch failed: {e}\nKeeping existing site data; nothing archived.")
        return
    if not elig:
        print(f"No eligible {name} lines (board empty, not posted yet, or every game has started).\n"
              f"Keeping existing docs/data/{book['props_json']}; no snapshot archived.")
        return

    nhl_ids = player_map.resolve(elig, cfg, map_path=book["player_map"])
    slate_dates = sorted({r.get("game_date") or r["start_time"][:10] for r in elig})   # US Eastern dates
    games = nhl_games_by_team(slate_dates)

    st = shared()
    sk, gm, walker, tier_of, hist_through = st["sk"], st["gm"], st["walker"], st["tier_of"], st["hist_through"]
    params = cfg["model"]

    picks, reasons = [], Counter()
    for r in elig:
        team, opp = nhl_team(r["team"], cfg), nhl_team(r["opp"], cfg)
        d = r.get("game_date") or r["start_time"][:10]
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
                "is_home": bool(g and g["home"] == team),
                "model_mean": None, "p_over": None, "p_under": None, "p_push": None,
                "side": None, "prob": None, "n_prior": None, "skip_reason": None}
        if ud:
            pick.update(book="underdog", sides=r["sides"], mult_over=r["mult_over"], mult_under=r["mult_under"],
                        type_over=r["type_over"], type_under=r["type_under"])
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
            elif cat == "faceoffs":
                mu, n = walker.fo_mu(nhl_id, opp)
                min_n = params["min_prior_games"]
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
                dist = model.dist_for(cfg, cat)
                if cat == "toi":
                    po, pu, pp = model.normal_probs(mu, r["line"], dist)
                elif cat == "saves":
                    po, pu, pp = model.saves_probs(mu, r["line"], dist)
                    mu = model.saves_mean(mu, dist)   # report the pull-adjusted expectation
                else:
                    po, pu, pp = model.line_probs(mu, r["line"], dist)
                side, prob = model.pick_side(po, pu)
                if ud and side not in r["sides"]:
                    # Underdog sells only one side of this line: that's the only pick available,
                    # even when the model would rather take the other side (probability < 50%).
                    side = r["sides"][0]
                    prob = (po if side == "over" else pu) / max(po + pu, 1e-12)
                pick.update(model_mean=round(mu, 4), p_over=round(po, 4), p_under=round(pu, 4),
                            p_push=round(pp, 4), side=side, prob=round(prob, 4),
                            prob_cal=round(model.calibrate(prob, cfg), 4))
                if ud:   # the picked side's payout and type (standard / goblin / demon)
                    pick.update(mult=r[f"mult_{side}"], line_type=r[f"type_{side}"], odds_type=r[f"type_{side}"])
        t = tier_of.get(int(nhl_id)) if nhl_id and r["category"] != "saves" else None
        if t:   # rotation tier: rank in his team's ice-time order (see tiers.py)
            pick.update(tier=t["tier"], tier_rank=f"{t['pos']}{t['rank']}", recent_toi=t["toi"])
        pick["skip_reason"] = reason
        reasons[reason or "modeled"] += 1
        picks.append(pick)

    print("  picks:", dict(reasons))
    if ud:
        print("  picked-side types:", dict(Counter(p["line_type"] for p in picks if p.get("line_type"))))
    add_open_lines(picks, book["snap_dir"])
    shadow_info = {}
    if not ud:
        try:   # LightGBM challenger: recorded next to the official picks, never replaces them
            shadow_info = gbm.shadow_predictions(cfg, sk, gm, walker, picks)
            if shadow_info:
                n = sum(1 for p in picks if p.get("shadow"))
                print(f"  shadow ({gbm.VERSION}): {n} picks priced; " + "; ".join(
                    f"{k}: trained through {v['trained_through']}, {v['rounds']} rounds" for k, v in shadow_info.items()))
        except Exception as e:
            print(f"  shadow model failed (official picks unaffected): {e!r}")
    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    counts = ({"types": dict(Counter(t for r in cand for t in (r["type_over"], r["type_under"]) if t))} if ud else
              {"odds_types": {f"{k[0]}|adj={k[1]}": v for k, v in
                              Counter((r["odds_type"], r["adjusted_odds"]) for r in cand).items()}})
    snapshot = {"generated_at": now.isoformat(), "slate_dates": slate_dates, "book": book["key"],
                "model_version": cfg["model_version"], "params": params,
                "history_through": hist_through, "shadow": shadow_info,
                "board_counts": {"rows": len(rows), "enabled_props": len(cand), "eligible": len(elig)} | counts,
                "picks": picks}
    snap_path = os.path.join(book["snap_dir"], f"{stamp}.json")
    write_json(snap_path, snapshot)
    write_json(os.path.join(DOCS_DATA, book["props_json"]),
               {k: snapshot[k] for k in ("generated_at", "slate_dates", "model_version", "history_through", "book")}
               | {"labels": {c: v["label"] for c, v in cfg["props"].items()}, "picks": picks,
                  "pair_lifts": {k: v for k, v in (cfg.get("pair_lifts") or {}).items() if k != "note"}})
    print(f"archived {os.path.relpath(snap_path, path())} and wrote docs/data/{book['props_json']} "
          f"(slate {', '.join(slate_dates)}, history through {hist_through})")


if __name__ == "__main__":
    raise SystemExit(main())
