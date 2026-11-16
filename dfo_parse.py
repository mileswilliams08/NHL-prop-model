"""
Turn the pages dfo_collect.py saved (dfo_data/<date>/*.json.gz) into two local tables:

  dfo_data/lines.csv    date, saved_at, team, player, dfo_player_id, position, group, group_name,
                        category, injury, game_time_decision   (one row per player listed on the page)
  dfo_data/goalies.csv  date, saved_at, team, opp, is_home, goalie, status, news_at, source

Uses the latest page saved for each team that day. Like the raw pages, these stay on this PC (dfo_data/
is git-ignored). Runs automatically at the end of dfo_collect.py, or by hand:

    python dfo_parse.py
"""
import csv
import glob
import gzip
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dfo_data")
LINE_FIELDS = ["date", "saved_at", "team", "player", "dfo_player_id", "position", "group", "group_name",
               "category", "injury", "game_time_decision"]
GOALIE_FIELDS = ["date", "saved_at", "team", "opp", "is_home", "goalie", "status", "news_at", "source"]


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def latest_pages(day_dir):
    """{page name: path of the latest save that day} (file names are <name>_<HHMM>.json.gz)."""
    best = {}
    for p in glob.glob(os.path.join(day_dir, "*.json.gz")):
        name, hhmm = os.path.basename(p)[:-len(".json.gz")].rsplit("_", 1)
        if name not in best or hhmm > best[name][0]:
            best[name] = (hhmm, p)
    return best


def team_rows(date, hhmm, team, page):
    pp = (page.get("props") or {}).get("pageProps") or {}
    out = []
    for x in (pp.get("combinations") or {}).get("players") or []:
        out.append({"date": date, "saved_at": hhmm, "team": team, "player": x.get("name"),
                    "dfo_player_id": x.get("playerId"), "position": x.get("positionIdentifier"),
                    "group": x.get("groupIdentifier"), "group_name": x.get("groupName"),
                    "category": x.get("categoryIdentifier"), "injury": x.get("injuryStatus") or "",
                    "game_time_decision": bool(x.get("gameTimeDecision"))})
    return out


def goalie_rows(date, hhmm, page):
    pp = (page.get("props") or {}).get("pageProps") or {}
    out = []
    for g in pp.get("data") or []:
        for side, other in (("home", "away"), ("away", "home")):
            out.append({"date": date, "saved_at": hhmm, "team": g.get(f"{side}TeamAbbreviation"),
                        "opp": g.get(f"{other}TeamAbbreviation"), "is_home": side == "home",
                        "goalie": g.get(f"{side}GoalieName"), "status": g.get(f"{side}NewsStrengthName") or "",
                        "news_at": g.get(f"{side}NewsCreatedAt") or "", "source": g.get(f"{side}NewsSourceName") or ""})
    return out


def main(verbose=True):
    lines, goalies = [], []
    for day_dir in sorted(glob.glob(os.path.join(DATA, "????-??-??"))):
        date = os.path.basename(day_dir)
        for name, (hhmm, p) in sorted(latest_pages(day_dir).items()):
            try:
                page = load(p)
            except Exception as e:
                print(f"  couldn't read {os.path.relpath(p, HERE)}: {e}")
                continue
            if name == "starting-goalies":
                goalies += goalie_rows(date, hhmm, page)
            else:
                lines += team_rows(date, hhmm, name, page)
    for path, fields, rows in ((os.path.join(DATA, "lines.csv"), LINE_FIELDS, lines),
                               (os.path.join(DATA, "goalies.csv"), GOALIE_FIELDS, goalies)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    if not verbose:
        return 0
    print(f"lines.csv: {len(lines)} rows, {len({(r['date'], r['team']) for r in lines})} team-days | "
          f"goalies.csv: {len(goalies)} rows")
    print("groups (category/group):", dict(Counter(f"{r['category']}/{r['group']}" for r in lines).most_common(25)))
    print("injury values:", dict(Counter(r["injury"] for r in lines)), "| game-time decisions:",
          sum(r["game_time_decision"] for r in lines))
    print("goalie status values:", dict(Counter(r["status"] for r in goalies)))
    if lines:
        last = max((r["date"], r["team"]) for r in lines)
        print(f"\n{last[1]} {last[0]}:")
        by = {}
        for r in lines:
            if (r["date"], r["team"]) == last:
                by.setdefault(f"{r['category']}/{r['group']}", []).append(
                    r["player"] + (f" [{r['injury']}]" if r["injury"] else "") + (" (GTD)" if r["game_time_decision"] else ""))
        for k, v in by.items():
            print(f"  {k:10} {', '.join(v)}")
    for g in goalies[-4:]:
        print(f"  goalie {g['team']} v {g['opp']}: {g['goalie']} ({g['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
