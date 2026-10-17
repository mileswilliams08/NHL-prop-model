"""
Take specific graded picks out of every number on the site without editing the graded log.
Exclusions live in data/excluded_picks.csv and are applied when results.json is built.

    python exclude_pick.py list --date 2026-10-17                 # find picks
    python exclude_pick.py list --date 2026-10-17 --player fox
    python exclude_pick.py add "Adam Fox" sog 2026-10-17          # exclude one pick
    python exclude_pick.py remove "Adam Fox" sog 2026-10-17       # undo
    python exclude_pick.py show
Add --rebuild to add/remove to refresh docs/data/results.json right away.
"""
import argparse
import csv
import os
import sys

import pandas as pd

from common import EXCLUDED_PATH, GRADED_PATH, norm_name

FIELDS = ["game_date", "player", "category", "note"]


def read_ex():
    if not os.path.exists(EXCLUDED_PATH):
        return []
    with open(EXCLUDED_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ex(rows):
    os.makedirs(os.path.dirname(EXCLUDED_PATH), exist_ok=True)
    with open(EXCLUDED_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)


def key(r):
    return (r["game_date"], norm_name(r["player"]), r["category"].lower())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--date"); ls.add_argument("--player"); ls.add_argument("--category")
    sub.add_parser("show")
    for name in ("add", "remove"):
        sp = sub.add_parser(name)
        sp.add_argument("player"); sp.add_argument("category"); sp.add_argument("date")
        sp.add_argument("--note", default="")
        sp.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "show":
        rows = read_ex()
        print(f"{len(rows)} excluded picks")
        for r in rows:
            print(f"  {r['game_date']} | {r['player']} | {r['category']} | {r.get('note', '')}")
        return 0

    df = pd.read_csv(GRADED_PATH, dtype=str) if os.path.exists(GRADED_PATH) else pd.DataFrame(
        columns=["game_date", "player", "category"])
    if a.cmd == "list":
        sel = df
        if a.date:
            sel = sel[sel["game_date"] == a.date]
        if a.player:
            sel = sel[sel["player"].map(norm_name).str.contains(norm_name(a.player))]
        if a.category:
            sel = sel[sel["category"].str.lower() == a.category.lower()]
        print(f"{len(sel)} graded picks")
        for _, r in sel.iterrows():
            print(f"  {r['game_date']} | {r['player']} ({r['team']} v {r['opp']}) | {r['category']} | line {r['line']} "
                  f"| {r['side']} {r['prob']} | actual {r['actual']} | {r['result']}")
        return 0

    k = (a.date, norm_name(a.player), a.category.lower())
    rows = read_ex()
    if a.cmd == "add":
        hit = df[(df["game_date"] == a.date) & (df["player"].map(norm_name) == k[1]) & (df["category"].str.lower() == k[2])]
        if hit.empty:
            print("No graded pick matches that player/category/date (use `list`). Nothing changed.")
            return 1
        if any(key(r) == k for r in rows):
            print("Already excluded.")
            return 0
        rows.append({"game_date": a.date, "player": hit.iloc[0]["player"], "category": k[2], "note": a.note})
        write_ex(rows)
        print(f"Excluded {hit.iloc[0]['player']} / {k[2]} / {a.date}")
    else:
        kept = [r for r in rows if key(r) != k]
        if len(kept) == len(rows):
            print("Not in the exclusion list. Nothing changed.")
            return 1
        write_ex(kept)
        print(f"Removed {a.player} / {k[2]} / {a.date} from exclusions")
    if a.rebuild:
        import build_results
        build_results.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
