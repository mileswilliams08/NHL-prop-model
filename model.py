"""
Baseline projection models for skater counting stats, time on ice and goalie saves.

Skater count props (sog, hits, blocks):
    expected = per-second rate (decayed, shrunk toward the position average) x projected TOI
               x opponent factor (sog only: opponent's shots allowed vs league average)
    distribution: negative binomial, dispersion r per prop (config "dist")

Time on ice (toi): projected TOI in minutes, normal distribution with a fitted sd.

Goalie saves (saves):
    expected shots against = league avg x opponent's shots-for factor x own team's shots-allowed factor
    save rate              = goalie's decayed saves / shots against, shrunk toward the league rate
    expected saves = expected shots against x save rate; negative binomial.

walk() replays history one date at a time: every game on a date is predicted from state built ONLY
from earlier dates, then state is updated. The backtest and the live prediction use the same walk.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from common import HIST_DIR

POS_GROUP = {"C": "F", "L": "F", "R": "F", "F": "F", "D": "D"}
SKATER_STATS = {"sog": "sog", "hits": "hits", "blocks": "blocked"}   # prop -> history column
FALLBACK_RATE = {"F": {"sog": 2.0 / 960, "hits": 1.2 / 960, "blocks": 0.5 / 960},
                 "D": {"sog": 1.4 / 1260, "hits": 1.3 / 1260, "blocks": 1.4 / 1260}}
FALLBACK_TOI = {"F": 960.0, "D": 1260.0}
LEAGUE_DECAY = 0.99      # per game date, for league-wide baselines
SV_PRIOR_SHOTS = 300.0   # shrink a goalie's save rate toward the league with this many average shots
FALLBACK_SV = 0.900


def load_history(cfg=None):
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"))
    gm = pd.read_csv(os.path.join(HIST_DIR, "games.csv"))
    gk = pd.read_csv(os.path.join(HIST_DIR, "goalie_games.csv"))
    types = (cfg or {}).get("history_game_types", [2, 3])
    sk, gm, gk = (d[d["game_type"].isin(types)] for d in (sk, gm, gk))
    for c in ("hits", "blocked"):
        sk[c] = pd.to_numeric(sk[c], errors="coerce").fillna(0)
    for c in ("saves", "shots_against"):
        gk[c] = pd.to_numeric(gk[c], errors="coerce").fillna(0)
    return sk, gm, gk


def _zero():
    return {s: 0.0 for s in SKATER_STATS}


class Walker:
    def __init__(self, params):
        self.p = params
        self.d = 0.5 ** (1.0 / params["halflife_games"])
        self.dt = 0.5 ** (1.0 / params["toi_halflife_games"])
        self.do = 0.5 ** (1.0 / params["opp_halflife_games"])
        self.players, self.teams, self.goalies = {}, {}, {}
        self.lg_rate = {g: {"S": _zero(), "T": 0.0} for g in ("F", "D")}
        self.lg_toi = {g: [0.0, 0.0] for g in ("F", "D")}
        self.lg_shots = [0.0, 0.0]     # decayed [shots, team-games] (shots for == shots against league-wide)
        self.lg_sv = [0.0, 0.0]        # decayed [saves, shots against] for goalies

    # ---------------------------------------------------------------- baselines
    def rate_prior(self, g, stat):
        lg = self.lg_rate[g]
        return lg["S"][stat] / lg["T"] if lg["T"] > 0 else FALLBACK_RATE[g][stat]

    def toi_prior(self, g):
        s, n = self.lg_toi[g]
        return s / n if n > 0 else FALLBACK_TOI[g]

    def lg_avg_shots(self):
        s, n = self.lg_shots
        return s / n if n > 0 else None

    def _team_factor(self, team, key):
        lg = self.lg_avg_shots()
        t = self.teams.get(team)
        if not lg or not t:
            return 1.0
        k = self.p["opp_prior_games"]
        return ((t[key] + k * lg) / (t["N"] + k)) / lg

    def opp_factor(self, team):
        """How many shots `team` allows, relative to league average."""
        return self._team_factor(team, "SA")

    def off_factor(self, team):
        """How many shots `team` takes, relative to league average."""
        return self._team_factor(team, "SF")

    # ---------------------------------------------------------------- skater prediction
    def player_toi(self, pid):
        """(projected TOI seconds, games of history) or (None, 0)."""
        st = self.players.get(int(pid)) if pid not in (None, "") else None
        if not st:
            return None, 0
        k, n = self.p["toi_prior_games"], st["n"]
        return (n * st["toi"] + k * self.toi_prior(st["pos"])) / (n + k), n

    def player_mu(self, pid, opp, stat="sog"):
        """(expected count, games of history) or (None, 0) for an unknown player."""
        st = self.players.get(int(pid)) if pid not in (None, "") else None
        if not st:
            return None, 0
        toi, n = self.player_toi(pid)
        m = self.p["prior_toi_sec"]
        rate = (st["S"][stat] + m * self.rate_prior(st["pos"], stat)) / (st["T"] + m)
        mu = rate * toi
        if stat == "sog":
            mu *= self.opp_factor(opp) ** self.p["opp_beta"]
        return mu, n

    # ---------------------------------------------------------------- goalie prediction
    def goalie_mu(self, gid, team, opp):
        """(expected saves, starts of history) or (None, 0)."""
        g = self.goalies.get(int(gid)) if gid not in (None, "") else None
        lg = self.lg_avg_shots()
        if not g or not lg:
            return None, 0
        esa = lg * self.off_factor(opp) * self.opp_factor(team)
        lg_sv = self.lg_sv[0] / self.lg_sv[1] if self.lg_sv[1] > 0 else FALLBACK_SV
        sv = (g["SV"] + SV_PRIOR_SHOTS * lg_sv) / (g["SA"] + SV_PRIOR_SHOTS)
        return esa * sv, g["n"]

    # ---------------------------------------------------------------- update
    def update(self, sk_rows, game_rows, gk_rows):
        for g in ("F", "D"):
            lg = self.lg_rate[g]
            lg["S"] = {s: v * LEAGUE_DECAY for s, v in lg["S"].items()}
            lg["T"] *= LEAGUE_DECAY
            self.lg_toi[g] = [x * LEAGUE_DECAY for x in self.lg_toi[g]]
        self.lg_shots = [x * LEAGUE_DECAY for x in self.lg_shots]
        self.lg_sv = [x * LEAGUE_DECAY for x in self.lg_sv]
        carry = self.p["season_carry"]

        for pid, pos, season, toi, vals in sk_rows:
            if toi <= 0:
                continue
            g = POS_GROUP.get(pos, "F")
            st = self.players.get(pid)
            if st is None:
                st = self.players[pid] = {"S": _zero(), "T": 0.0, "toi": float(toi), "n": 0, "season": season, "pos": g}
            elif st["season"] != season:
                st["S"] = {s: v * carry for s, v in st["S"].items()}
                st["T"] *= carry
                st["season"] = season
            for s, v in vals.items():
                st["S"][s] = self.d * st["S"][s] + v
                self.lg_rate[g]["S"][s] += v
            st["T"] = self.d * st["T"] + toi
            st["toi"] = self.dt * st["toi"] + (1 - self.dt) * toi if st["n"] else float(toi)
            st["n"] += 1
            st["pos"] = g
            self.lg_rate[g]["T"] += toi
            self.lg_toi[g][0] += toi
            self.lg_toi[g][1] += 1

        for team, sf, sa, season in game_rows:
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (sf, sa)):
                continue
            t = self.teams.get(team)
            if t is None:
                t = self.teams[team] = {"SA": 0.0, "SF": 0.0, "N": 0.0, "season": season}
            elif t["season"] != season:
                for k in ("SA", "SF", "N"):
                    t[k] *= carry
                t["season"] = season
            t["SA"] = self.do * t["SA"] + sa
            t["SF"] = self.do * t["SF"] + sf
            t["N"] = self.do * t["N"] + 1
            self.lg_shots[0] += sa
            self.lg_shots[1] += 1

        for gid, season, toi, saves, sa in gk_rows:
            if toi <= 0 or sa <= 0:
                continue
            g = self.goalies.get(gid)
            if g is None:
                g = self.goalies[gid] = {"SV": 0.0, "SA": 0.0, "n": 0, "season": season}
            elif g["season"] != season:
                g["SV"] *= carry
                g["SA"] *= carry
                g["season"] = season
            g["SV"] = self.d * g["SV"] + saves
            g["SA"] = self.d * g["SA"] + sa
            g["n"] += 1
            self.lg_sv[0] += saves
            self.lg_sv[1] += sa


def walk(sk, gm, params, gk=None, collect=True):
    """Replay history. Returns (skater predictions df, goalie predictions df, final Walker)."""
    w = Walker(params)
    by_date = lambda df: {d: g for d, g in df.sort_values(["date", "game_id"], kind="stable").groupby("date", sort=False)}
    skd, gmd = by_date(sk), by_date(gm)
    gkd = by_date(gk) if gk is not None else {}
    out_s, out_g = ([], []) if collect else (None, None)

    for d in sorted(set(skd) | set(gmd) | set(gkd)):
        srows, grows, krows = skd.get(d), gmd.get(d), gkd.get(d)
        tuples, gtuples, ktuples = [], [], []
        if srows is not None:
            for r in srows.itertuples(index=False):
                if collect and r.toi_sec > 0:
                    toi_p, n = w.player_toi(r.player_id)
                    mus = {s: w.player_mu(r.player_id, r.opp, s)[0] for s in SKATER_STATS}
                    out_s.append((r.date, r.season, r.game_id, r.player_id, r.team, r.opp,
                                  POS_GROUP.get(r.position, "F"), n, mus["sog"], mus["hits"], mus["blocks"],
                                  toi_p, r.sog, r.hits, r.blocked, r.toi_sec))
                tuples.append((int(r.player_id), r.position, r.season, r.toi_sec,
                               {s: getattr(r, c) for s, c in SKATER_STATS.items()}))
        if krows is not None:
            for r in krows.itertuples(index=False):
                if collect and r.toi_sec > 0 and str(r.starter).lower() == "true":
                    mu, n = w.goalie_mu(r.player_id, r.team, r.opp)
                    out_g.append((r.date, r.season, r.game_id, r.player_id, r.team, r.opp, n, mu,
                                  r.saves, r.shots_against, r.toi_sec))
                ktuples.append((int(r.player_id), r.season, r.toi_sec, r.saves, r.shots_against))
        if grows is not None:
            for r in grows.itertuples(index=False):
                gtuples.append((r.away, r.away_sog, r.home_sog, r.season))
                gtuples.append((r.home, r.home_sog, r.away_sog, r.season))
        w.update(tuples, gtuples, ktuples)

    if not collect:
        return None, None, w
    sdf = pd.DataFrame(out_s, columns=["date", "season", "game_id", "player_id", "team", "opp", "pos", "n_prior",
                                       "mu_sog", "mu_hits", "mu_blocks", "toi_proj", "sog", "hits", "blocks", "toi_sec"])
    gdf = pd.DataFrame(out_g, columns=["date", "season", "game_id", "player_id", "team", "opp", "n_prior", "mu_saves",
                                       "saves", "shots_against", "toi_sec"])
    return sdf, gdf, w


# ---------------------------------------------------------------------------- probabilities
def _dist(mu, r):
    mu = max(float(mu), 1e-6)
    if r is None or r <= 0 or r > 1e6:
        return poisson(mu)
    return nbinom(r, r / (r + mu))


def line_probs(mu, line, r):
    """(P over, P under, P push) for a count stat against a line."""
    dist = _dist(mu, r)
    line = float(line)
    if line.is_integer():
        k = int(line)
        push = float(dist.pmf(k))
        under = float(dist.cdf(k - 1))
        return 1.0 - under - push, under, push
    under = float(dist.cdf(math.floor(line)))
    return 1.0 - under, under, 0.0


def normal_probs(mu, line, sd):
    """(P over, P under, 0) for a continuous stat (TOI minutes)."""
    under = float(norm.cdf((float(line) - mu) / sd))
    return 1.0 - under, under, 0.0


def pick_side(p_over, p_under):
    """Model's side and its probability, conditional on no push."""
    tot = p_over + p_under
    if tot <= 0:
        return "over", 0.5
    return ("over", p_over / tot) if p_over >= p_under else ("under", p_under / tot)


def nb_loglik(mu, y, r):
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    y = np.asarray(y)
    if r is None or r > 1e6:
        return poisson.logpmf(y, mu)
    return nbinom.logpmf(y, r, r / (r + mu))


FULL_GAME_SEC = 3500   # a starter with less TOI than this was pulled or left early


def saves_mean(mu_full, d):
    """Expected saves once pulls are accounted for. mu_full = saves if he plays the whole game."""
    return mu_full * (d["q_full"] + (1 - d["q_full"]) * d["pull_frac"])


def saves_probs(mu_full, line, d):
    """Goalie saves as a two-case mixture: plays the full game (prob q_full, NB around mu_full) or is
    pulled / leaves early (NB around pull_frac * mu_full, much wider). Fitted in backtest.py on training
    seasons only; ignoring pulls over-predicted saves by ~1 per start."""
    q = d["q_full"]
    a = line_probs(mu_full, line, d["nb_r_full"])
    b = line_probs(mu_full * d["pull_frac"], line, d["nb_r_pull"])
    return tuple(q * x + (1 - q) * y for x, y in zip(a, b))


def dist_for(cfg, cat):
    """Distribution parameter for a prop: nb_r for counts, sd_min for toi."""
    d = (cfg.get("dist") or {}).get(cat, {})
    if cat == "sog":
        return d.get("nb_r", cfg["model"].get("nb_r", 14.0))
    if cat == "saves":
        out = {"q_full": 0.78, "pull_frac": 0.80, "nb_r_full": 50.0, "nb_r_pull": 8.0}
        out.update({k: v for k, v in d.items() if k in out})
        return out
    defaults = {"hits": {"nb_r": 6.0}, "blocks": {"nb_r": 6.0}, "toi": {"sd_min": 3.0}}
    key = "sd_min" if cat == "toi" else "nb_r"
    return d.get(key, defaults.get(cat, {}).get(key))
