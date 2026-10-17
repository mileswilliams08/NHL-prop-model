# NHL Prop Tracker

A free, self-updating tracker that asks one question: **does a model beat the PrizePicks line?**
Tracking only, no betting. Shots On Goal is the headline prop. Hits, Blocked Shots, Time On Ice and Goalie Saves are priced
under **Experimental** until each has ~100+ graded picks. Faceoffs Won lines are recorded every night but not priced yet
(the boxscore only has faceoff %, not counts; counts would need a play-by-play backfill).

Site: GitHub Pages from `/docs` (Today's lines · Results · Backtest).

## How a day works (GitHub Action, 15:00 UTC)

1. `nhl_backfill.py --start <7 days ago>` adds finished games to `data/history/` (only OFF/FINAL games; merges into the committed CSVs).
2. `grade.py` grades archived picks whose game is final. A pick counts only if the player played (TOI > 0); otherwise it's logged as void.
3. `predict.py` pulls the PrizePicks NHL board (league 8), prices standard Shots On Goal lines, archives `data/snapshots/<UTC time>.json` and writes `docs/data/props.json`.
4. `build_results.py` writes `docs/data/results.json` and prints the market test in the Action log.

## The rules it follows

- **Market comparison from night one.** Every pick stores the PrizePicks line, the model mean and both probabilities. Results always show blind Over and blind Under on the same lines, a calibration table, and the slope of (actual − line) on (model mean − line). Don't recalibrate before ~100+ picks per prop.
- **Only real two-way lines.** `odds_type` must be `standard`. Goblins/demons are counted, never priced or graded. Combo/matchup markets and other leagues (NHL1P/2P/3P etc.) are never mapped.
- **`adjusted_odds`.** Many standard lines carry `adjusted_odds: true` and we don't yet know what it means, so those lines are priced but shown under **Experimental**, not in the headline numbers (`headline_requires_unadjusted` in `config.json`).
- **Pipeline guards.** No eligible lines → nothing is overwritten and no snapshot is archived. Games starting within 10 minutes are never priced, so a mid-game run is harmless. Each snapshot is stamped with its slate date(s).
- **Grading.** Dedup key `(game_date, nhl_game_id, nhl_id, category)`; the earliest snapshot's line wins. Games still unplayed 3 days later are voided (postponements).
- **Experimental section.** Unproven props and adjusted-payout lines live there so they never move the headline numbers.

## The models (`model.py`, version in `config.json`)

- **Shots, hits, blocks:** decayed per-second rate (shrunk toward the position average) × projected TOI (× opponent shots-allowed factor for SOG), negative binomial with a per-prop dispersion.
- **Time on ice:** projected TOI in minutes, normal distribution with a fitted sd.
- **Goalie saves:** league shots × opponent's shots-for factor × own team's shots-allowed factor × the goalie's shrunk save rate; negative binomial.

Per-prop distribution parameters live under `dist` in `config.json`; `python backtest.py --apply` fits them. `walk()` replays history date by date so every prediction uses only earlier games; the backtest and the live run share that code.

## Running things locally (Windows PowerShell)

```powershell
pip install -r requirements.txt
python nhl_backfill.py            # first time: full history since 2024-25 (~20-30 min)
python backtest.py                # parameter grid on old seasons, writes docs/data/backtest.json
python backtest.py --apply        # write the chosen parameters into config.json
python prizepicks.py              # quick look at the live board
python player_map.py              # resolve today's board to NHL ids, list what didn't match
python exclude_pick.py list --date 2026-10-17
python exclude_pick.py add "Adam Fox" sog 2026-10-17 --rebuild
```

The sandbox Claude works in can't reach PrizePicks or the NHL API; run scrapers locally and paste output.

## Files

| File | What it does |
|---|---|
| `config.json` | league id, enabled props, team abbreviation map, model parameters, model_version |
| `prizepicks.py` | fetch + parse the board (keeps odds_type, adjusted_odds, combo, start time) |
| `nhl_backfill.py` | NHL boxscores → `data/history/*.csv` (cache in `data/raw/`, gitignored) |
| `player_map.py` | PrizePicks player → NHL id, saved in `data/player_map.csv`; manual fixes in `data/player_aliases.csv` |
| `model.py` | the SOG model and line probabilities |
| `backtest.py` | honest train/test backtest vs naive baselines + calibration by line |
| `predict.py` | daily pricing, snapshot archive, `docs/data/props.json` |
| `grade.py` | append-only `data/graded_results.csv` |
| `build_results.py` | `docs/data/results.json` + printed market test |
| `exclude_pick.py` | hand-exclude known bad picks (`data/excluded_picks.csv`) |

## Git habits

Add files by explicit name (never `git add .`), `git pull` before `git push`, and don't commit `__pycache__/` (it's in `.gitignore`).
