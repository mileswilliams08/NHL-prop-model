"""
Baseline shots-on-goal model.

    expected SOG = shot rate per second  x  projected TOI  x  opponent factor ^ opp_beta

  shot rate      exponentially decayed SOG / decayed TOI over the player's games
                 (halflife_games), shrunk toward the league rate for his position group with
                 prior_toi_sec seconds of "average" ice time. At a new season the decayed sums
                 are multiplied by season_carry.
  projected TOI  decayed average TOI (toi_halflife_games), shrunk toward the position average
                 for players with few games (toi_prior_games).
  opponent       opponent's decayed shots-against per game relative to the league average,
                 shrunk with opp_prior_games of league-average games.
  distribution   negative binomial with dispersion nb_r (variance = mu + mu^2 / r).

walk() replays history one date at a time: every game on a date is predicted from state built
ONLY from earlier dates, then the state is updated. The backtest and the live prediction use
the same walk, so there is no separate "live" code path that could leak or drift.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from common import HIST_DIR

POS_GROUP = {"C": "F", "L": "F", "R": "F", "F": "F", "D": "D"}
FALLBACK_RATE = {"F": 2.0 / 960, "D": 1.4 / 1260}   # shots per second, only before any data
FALLBACK_TOI = {"F": 960.0, "D": 1260.0}
LEAGUE_DECAY = 0.99                                  # per game date, for league-wide baselines


def load_history(cfg=None):
    sk = pd.read_csv(os.path.join(HIST_DIR, "skater_games.csv"))
    gm = pd.read_csv(os.path.join(HIST_DIR, "games.csv"))
    types = (cfg or {}).get("history_game_types", [2, 3])
    sk = sk[sk["game_type"].isin(types)]
    gm = gm[gm["game_type"].isin(types)]
    return sk, gm


class Walker:
    def __init__(self, params):
        self.p = params
        self.d = 0.5 ** (1.0 / params["halflife_games"])
        self.dt = 0.5 ** (1.0 / params["toi_halflife_games"])
        self.do = 0.5 ** (1.0 / params["opp_halflife_games"])
        self.players = {}
        self.teams = {}
        self.lg_rate = {"F": [0.0, 0.0], "D": [0.0, 0.0]}   # decayed [sog, toi_sec]
        self.lg_toi = {"F": [0.0, 0.0], "D": [0.0, 0.0]}    # decayed [toi_sec, games]
        self.lg_sa = [0.0, 0.0]                             # decayed [shots against, team-games]

    # ---------------------------------------------------------------- baselines
    def rate_prior(self, g):
        s, t = self.lg_rate[g]
        return s / t if t > 0 else FALLBACK_RATE[g]

    def toi_prior(self, g):
        s, n = self.lg_toi[g]
        return s / n if n > 0 else FALLBACK_TOI[g]

    def opp_factor(self, team):
        sa, n = self.lg_sa
        if n <= 0:
            return 1.0
        lg = sa / n
        t = self.teams.get(team)
        if not t:
            return 1.0
        k = self.p["opp_prior_games"]
        return ((t["SA"] + k * lg) / (t["N"] + k)) / lg

    # ---------------------------------------------------------------- prediction
    def player_mu(self, pid, opp):
        """(expected SOG, games of history) or (None, 0) for an unknown player."""
        st = self.players.get(int(pid)) if pid not in (None, "") else None
        if not st:
            return None, 0
        g, m = st["pos"], self.p["prior_toi_sec"]
        rate = (st["S"] + m * self.rate_prior(g)) / (st["T"] + m)
        k, n = self.p["toi_prior_games"], st["n"]
        toi = (n * st["toi"] + k * self.toi_prior(g)) / (n + k)
        mu = rate * toi * self.opp_factor(opp) ** self.p["opp_beta"]
        return mu, n

    # ---------------------------------------------------------------- update
    def update(self, sk_rows, game_rows):
        for g in ("F", "D"):
            self.lg_rate[g] = [x * LEAGUE_DECAY for x in self.lg_rate[g]]
            self.lg_toi[g] = [x * LEAGUE_DECAY for x in self.lg_toi[g]]
        self.lg_sa = [x * LEAGUE_DECAY for x in self.lg_sa]
        carry = self.p["season_carry"]

        for pid, pos, season, toi, sog in sk_rows:
            if toi <= 0:
                continue
            g = POS_GROUP.get(pos, "F")
            st = self.players.get(pid)
            if st is None:
                st = self.players[pid] = {"S": 0.0, "T": 0.0, "toi": float(toi), "n": 0, "season": season, "pos": g}
            elif st["season"] != season:
                st["S"] *= carry
                st["T"] *= carry
                st["season"] = season
            st["S"] = self.d * st["S"] + sog
            st["T"] = self.d * st["T"] + toi
            st["toi"] = self.dt * st["toi"] + (1 - self.dt) * toi if st["n"] else float(toi)
            st["n"] += 1
            st["pos"] = g
            self.lg_rate[g][0] += sog
            self.lg_rate[g][1] += toi
            self.lg_toi[g][0] += toi
            self.lg_toi[g][1] += 1

        for team, sa, season in game_rows:
            if sa is None or (isinstance(sa, float) and math.isnan(sa)):
                continue
            t = self.teams.get(team)
            if t is None:
                t = self.teams[team] = {"SA": 0.0, "N": 0.0, "season": season}
            elif t["season"] != season:
                t["SA"] *= carry
                t["N"] *= carry
                t["season"] = season
            t["SA"] = self.do * t["SA"] + sa
            t["N"] = self.do * t["N"] + 1
            self.lg_sa[0] += sa
            self.lg_sa[1] += 1


def walk(sk, gm, params, collect=True):
    """Replay history. Returns (DataFrame of pre-game predictions or None, final Walker)."""
    w = Walker(params)
    sk = sk.sort_values(["date", "game_id"], kind="stable")
    gm = gm.sort_values(["date", "game_id"], kind="stable")
    sk_by_date = {d: g for d, g in sk.groupby("date", sort=False)}
    gm_by_date = {d: g for d, g in gm.groupby("date", sort=False)}
    out = [] if collect else None

    for d in sorted(set(sk_by_date) | set(gm_by_date)):
        srows = sk_by_date.get(d)
        grows = gm_by_date.get(d)
        tuples = []
        if srows is not None:
            for r in srows.itertuples(index=False):
                if collect and r.toi_sec > 0:
                    mu, n = w.player_mu(r.player_id, r.opp)
                    out.append((r.date, r.season, r.game_id, r.player_id, r.team, r.opp,
                                POS_GROUP.get(r.position, "F"), n, mu, r.sog, r.toi_sec))
                tuples.append((int(r.player_id), r.position, r.season, r.toi_sec, r.sog))
        gtuples = []
        if grows is not None:
            for r in grows.itertuples(index=False):
                gtuples.append((r.away, r.home_sog, r.season))
                gtuples.append((r.home, r.away_sog, r.season))
        w.update(tuples, gtuples)

    df = None
    if collect:
        df = pd.DataFrame(out, columns=["date", "season", "game_id", "player_id", "team", "opp",
                                        "pos", "n_prior", "mu", "sog", "toi_sec"])
    return df, w


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
