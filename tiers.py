"""
Rotation tiers: where a skater sits in his team's ice-time order right now.

Rank = position in the team's list of forwards (or defensemen) sorted by each player's average TOI
over his own last 10 games. Only players who appeared for that team in its last 10 games are ranked,
and a traded player counts for his new team from his first game there.

    Forwards:   F1-F3 Top    F4-F9 Mid    F10+ Depth
    Defense:    D1-D2 Top    D3-D4 Mid    D5+  Depth

Why it matters (measured on 2024-26 regular seasons, each player's TOI vs his own recent average):
same-tier teammates' minutes move together (top-2 forwards +0.41, top F + top D +0.27, two 4th-liners
+0.33), while top vs depth players move apart (F1 vs F10 -0.21, D1 vs D5 -0.18). So for two TOI legs on
the same team: same tier -> same direction pairs better; top vs depth -> opposite directions pair better.

Caveat: injured players who played in the team's last 10 games still count, which can push healthy
players down a rank or two until the injured player drops out of the window.
"""
import pandas as pd

WINDOW = 10
F_TIERS = [(3, "Top"), (9, "Mid")]
D_TIERS = [(2, "Top"), (4, "Mid")]


def _tier(pos, rank):
    for cut, name in (D_TIERS if pos == "D" else F_TIERS):
        if rank <= cut:
            return name
    return "Depth"


def team_tiers(sk):
    """{player_id: {"pos", "rank", "tier", "toi"}} from skater history (toi in minutes)."""
    sk = sk[sk["toi_sec"] > 0].sort_values(["date", "game_id"], kind="stable")
    team_recent = {t: set(g["game_id"].drop_duplicates().tail(WINDOW)) for t, g in sk.groupby("team")}
    last = sk.groupby("player_id").tail(1).set_index("player_id")
    avg = sk.groupby("player_id")["toi_sec"].apply(lambda s: s.tail(WINDOW).mean() / 60)
    rows = []
    for pid, r in last.iterrows():
        if r["game_id"] in team_recent.get(r["team"], ()):
            rows.append((pid, r["team"], "D" if r["position"] == "D" else "F", float(avg[pid])))
    df = pd.DataFrame(rows, columns=["player_id", "team", "pos", "toi"])
    df["rank"] = df.groupby(["team", "pos"])["toi"].rank(ascending=False, method="first").astype(int)
    return {int(r.player_id): {"pos": r.pos, "rank": int(r.rank), "tier": _tier(r.pos, r.rank),
                               "toi": round(r.toi, 1)} for r in df.itertuples()}
