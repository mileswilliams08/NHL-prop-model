"""
PrizePicks player -> NHL player id.

The map is keyed by PrizePicks' player id and saved in data/player_map.csv, so each player
is resolved once and stays resolved (trades don't break it). Unresolved players are retried
every run. Match order:

  1. data/player_aliases.csv     manual override: pp_name,nhl_id[,pp_team]
  2. exact full name on the team's current NHL roster (position breaks ties, e.g. two Elias Petterssons)
  3. last name on that roster, same position group, unique
  4. exact full name on any loaded roster (recently traded)
  5. recent boxscores (data/history/players.csv): first initial + last name, same last team,
     same position group -- catches call-ups and IR players missing from /roster/current

    python player_map.py          # resolve today's board and print what didn't match
"""
import csv
import os
from datetime import date

from common import ALIASES_PATH, HIST_DIR, NHL_BASE, PLAYER_MAP_PATH, get_json, nhl_team, norm_name

FIELDS = ["pp_player_id", "pp_name", "pp_team", "pp_pos", "nhl_id", "nhl_name", "method",
          "first_seen", "last_seen"]


def _pos_ok(pp_pos, nhl_pos):
    pp_pos, nhl_pos = (pp_pos or "").upper(), (nhl_pos or "").upper()
    if not pp_pos:
        return True
    if pp_pos == "F":
        return nhl_pos in ("C", "L", "R", "F")
    return pp_pos == nhl_pos


def load_map():
    if not os.path.exists(PLAYER_MAP_PATH):
        return {}
    with open(PLAYER_MAP_PATH, newline="", encoding="utf-8") as f:
        return {r["pp_player_id"]: r for r in csv.DictReader(f)}


def save_map(mp):
    os.makedirs(os.path.dirname(PLAYER_MAP_PATH), exist_ok=True)
    with open(PLAYER_MAP_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(mp, key=lambda k: (mp[k]["pp_team"], mp[k]["pp_name"])):
            w.writerow({c: mp[k].get(c, "") for c in FIELDS})


def load_aliases():
    out = {}
    if os.path.exists(ALIASES_PATH):
        with open(ALIASES_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(norm_name(r["pp_name"]), (r.get("pp_team") or "").strip().upper())] = r["nhl_id"].strip()
    return out


def load_roster(team):
    data = get_json(f"{NHL_BASE}/roster/{team}/current") or {}
    out = []
    for grp in ("forwards", "defensemen", "goalies"):
        for x in data.get(grp, []):
            first = (x.get("firstName") or {}).get("default", "")
            last = (x.get("lastName") or {}).get("default", "")
            out.append({"id": str(x["id"]), "name": f"{first} {last}".strip(), "team": team,
                        "pos": x.get("positionCode", ""), "full": norm_name(f"{first} {last}"),
                        "last": norm_name(last)})
    return out


def load_history_players():
    p = os.path.join(HIST_DIR, "players.csv")
    out = []
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nm = r["name"]
                if ". " in nm:
                    init, last = nm.split(". ", 1)
                    out.append({"id": r["player_id"], "name": nm, "team": r["last_team"], "pos": r["position"],
                                "init": norm_name(init)[:1], "last": norm_name(last)})
    return out


def match_one(name, team, pos, rosters, aliases, hist):
    n = norm_name(name)
    a = aliases.get((n, team)) or aliases.get((n, ""))
    if a:
        return a, "", "alias"
    cands = rosters.get(team, [])
    m = [c for c in cands if c["full"] == n]
    if len(m) > 1:
        m = [c for c in m if _pos_ok(pos, c["pos"])]
    if len(m) == 1:
        return m[0]["id"], m[0]["name"], "exact"
    # Nicknames (Will/William, Freddy/Frederick): same last name and first initial, same position group.
    same_last = [c for c in cands if c["last"] and n.endswith(c["last"])]
    m = [c for c in same_last if n[:1] == c["full"][:1] and _pos_ok(pos, c["pos"])]
    if len(m) == 1:
        return m[0]["id"], m[0]["name"], "lastname"
    if same_last:
        return "", "", "ambiguous"   # someone with that last name IS on the team; don't guess elsewhere
    m = [c for t, cs in rosters.items() if t != team for c in cs if c["full"] == n and _pos_ok(pos, c["pos"])]
    if len(m) == 1:
        return m[0]["id"], m[0]["name"], f"exact_other_team({m[0]['team']})"
    m = [h for h in hist if h["team"] == team and h["last"] and n.startswith(h["init"])
         and n.endswith(h["last"]) and _pos_ok(pos, h["pos"])]
    if len(m) == 1:
        return m[0]["id"], m[0]["name"], "boxscore_history"
    return "", "", "unmatched"


def resolve(rows, cfg, verbose=True):
    """rows: parsed board rows. Returns {pp_player_id: nhl_id or ''} and updates the saved map."""
    mp = load_map()
    today = date.today().isoformat()
    need = {}
    for r in rows:
        pid = r["pp_player_id"]
        if not pid:
            continue
        e = mp.get(pid)
        if e and e.get("nhl_id"):
            e["last_seen"] = today
            e["pp_team"] = r["team"]
        else:
            need[pid] = r

    if need:
        aliases, hist = load_aliases(), load_history_players()
        rosters = {}
        for t in sorted({nhl_team(r["team"], cfg) for r in rows if r["team"]}):
            try:
                rosters[t] = load_roster(t)
            except Exception as e:
                print(f"  roster fetch failed for {t}: {e}")
        for pid, r in need.items():
            team = nhl_team(r["team"], cfg)
            nid, nname, how = match_one(r["player"], team, r["position"], rosters, aliases, hist)
            prev = mp.get(pid, {})
            mp[pid] = {"pp_player_id": pid, "pp_name": r["player"], "pp_team": r["team"], "pp_pos": r["position"],
                       "nhl_id": nid, "nhl_name": nname, "method": how,
                       "first_seen": prev.get("first_seen") or today, "last_seen": today}
        # Two PrizePicks players must never share one NHL player: unmap every side of a conflict.
        owners = {}
        for pid_, e in mp.items():
            if e.get("nhl_id"):
                owners.setdefault(e["nhl_id"], []).append(pid_)
        for nid, pids in owners.items():
            if len(pids) > 1:
                for pid_ in pids:
                    if pid_ in need:
                        mp[pid_].update(nhl_id="", nhl_name="", method=f"conflict({nid})")
        if verbose:
            new = [mp[p] for p in need]
            print(f"player map: {len(new)} new/unresolved players looked up -> "
                  f"{sum(1 for e in new if e['nhl_id'])} matched")
            for e in sorted(new, key=lambda e: (e["method"], e["pp_team"])):
                if e["method"] not in ("exact",):
                    print(f"   {e['method']:<26} {e['pp_name']} ({e['pp_team']}, {e['pp_pos']}) -> {e['nhl_name'] or '-'} {e['nhl_id']}")
    save_map(mp)
    return {pid: (mp.get(pid) or {}).get("nhl_id", "") for pid in {r["pp_player_id"] for r in rows}}


if __name__ == "__main__":
    from common import load_config
    from prizepicks import fetch_board, parse_board
    cfg = load_config()
    rows = [r for r in parse_board(fetch_board(cfg["league_id"]), cfg) if not r["combo"]]
    res = resolve(rows, cfg)
    print(f"{sum(1 for v in res.values() if v)}/{len(res)} board players mapped")
