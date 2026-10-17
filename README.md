# Hull City — Premier League 2026/27 Predictor

A season simulator and analysis app built around one question: **do Hull City stay up?**

Local Flask app + single-file frontend (vanilla JS, no build step). Match results,
fixtures and Opta stats are pulled from public APIs; everything else — ratings,
simulations, player ratings, scoreline modelling — is derived here.

**Stack:** Python 3 / Flask backend, single-file HTML/CSS/JS frontend, Monte Carlo
simulation in Web Workers, no external JS dependencies.

---

## What it does

- **Live league table** that folds in in-progress scores, animating each result
  onto the standings one at a time.
- **Monte Carlo season simulation** (200,000 seasons, parallelised across cores)
  giving every club's title / European / relegation odds and a full
  finishing-position distribution.
- **Survival model** — the headline question — with the points total that has
  historically been enough to stay up, and each rival's relegation odds
  conditional on Hull surviving.
- **Match predictor** — Dixon-Coles bivariate Poisson with Premier League-specific
  scoreline calibration, giving win/draw/loss, the most likely scoreline, and a
  full score grid.
- **Live tab** — in-play win/draw/loss re-projected from the score and the minutes
  left (stoppage time included), plus a per-match 1–10 player rater.
- **Player ratings** scored against positional peers across all 20 squads, not
  just teammates.
- **Points-deduction support** (e.g. an FFP sanction) that flows through the table,
  projections and every simulated season.
- An **ops panel** on a separate loopback IP for refreshing data and restarting
  the app without taking it down.

---

## Running it

```bash
python3 server.py start      # http://localhost:8080
python3 server.py stop
python3 server.py            # interactive menu
```

`data.json` and the cache folders are created on first run.

---

## The model

Each team has an attack and defence rating in goals/game. A match is a
Dixon-Coles-corrected bivariate Poisson:

```
home xG = (atkH / 1.40) × defA × 1.11
away xG = (atkA / 1.40) × defH × 0.89
```

Season simulations add correlated log-normal form noise (shared factor lifts attack
and tightens defence together), scaled by how much of the season is unplayed, and
widened per-team for promoted sides whose priors are least certain.

Pre-season priors come from a **composite z-index**: eight correlated attacking
metrics averaged into one index per side, then mapped to goals/game. Averaging
cancels per-stat noise while fitting only two parameters per side, so there is
nothing to overfit — it beat the previous opponent-adjusted ratings in all three
leave-one-season-out folds.

Two calibrations are fitted to how the Premier League actually finishes games,
each validated out-of-sample (fit two seasons, test a third) before shipping:

- **Mismatch damping** — the raw model expected bigger gaps than reality delivers
  (favourites won less often than predicted at every confidence level), so each
  side's xG is pulled toward its league mean, `xG' = mean · (xG/mean)^0.80`.
- **Scoreline boosts** — 2–2 and 0–1 are under-predicted every season on record,
  so each is nudged up. Fitting one ratio per scoreline, by contrast, just learns
  one season's noise.

All of it lives in one `matchXG()` the predictor, Live tab, projections and every
simulation call, so they cannot disagree about how good a team is.

---

## Data sources

| Source | Used for | Notes |
|---|---|---|
| **Pulse** (`footballapi.pulselive.com`) | live results, ~165 Opta stats/team/match, player season stats, fixtures | needs `Origin: https://www.premierleague.com`; ids serialise as floats |
| football-data.co.uk | settled results + bookmaker odds | lags by days; only source with odds |
| ESPN | Championship team/player stats | only populates advanced stats for the live season |
| Wikipedia | squad fallback | last resort |

Results merge cheapest-detail-first: Pulse for speed, football-data overlaid for odds.
Already-stored matches are backfilled with any field they lack.

**No data ships with the repo.** Everything above is fetched into local, git-ignored
caches (`cache/`, `Newcache/`, `data.json`) on first run, and is regenerable by
running the scrapers. The third-party data stays subject to its providers' terms;
only the code here is MIT-licensed.

---

## Layout

**Backend**

| File | Purpose |
|---|---|
| `app.py` / `server.py` | Flask API and process manager |
| `pulse_results.py` | live results + match stats |
| `pulse_players.py` / `pl_players.py` | Hull and all-league player season stats |
| `player_rater.py` | position-aware player ratings vs league peers |
| `derived_stats.py` | ratio metrics (chance suppression, engagement, value from the ball) |
| `style_fingerprint.py` | live playing-style fingerprints incl. PPDA press label |
| `newcache_builder.py` | per-match Opta archive |
| `scraper.py` / `fixture_scraper.py` | football-data results, official fixtures |
| `ratings.py`, `derive_priors.py`, `promoted_analyzer.py` | prior derivation |

**Validation**

| File | Purpose |
|---|---|
| `prior_backtest.py` | scores pre-season priors on a held-out season |
| `backtest.py` | match- and table-level walk-forward, on-field skill benchmark |
| `scoreline_backtest.py` | out-of-sample test of scoreline calibration (damping, 2–2/0–1 boosts) |
| `newcache_probe.py` / `stat_sweep.py` | rank candidate metrics by out-of-sample predictiveness |
| `tune.py` | parameter search |

> **Note:** `backtest.py --bench` and `--table` do **not** read `PRIORS` — they rate
> teams from observed results only. Prior changes must be validated with
> `prior_backtest.py` across all three folds.

---

## Method

Most ideas tested here were rejected, and the rejections are the useful part:

- **Shot volume repeats; shot efficiency does not.** Goals-allowed-per-shot-faced has
  a split-half repeatability of r=0.038 against 0.769 for shots faced. Teams that
  overperform their process regress.
- **Defensive-action counts are confounded.** Tackles, interceptions and clearances
  all correlate *positively* with goals conceded — they measure pressure faced, not
  defensive quality.
- **Combine stats by averaging, not regression.** At ~34 team-seasons a greedy
  multi-feature fit overfits; a z-scored composite of the same stats generalises.
- **Group before weighting.** Opta splits passing into ~33 columns and clearances
  into 4, so a flat average across raw stats weights passing eight times more by
  accident.
- **A single fold is not evidence.** Three separate candidates looked like clear
  improvements on one held-out season and reversed on another.

---

## Status

Built during the 2026/27 season, so several pages are sparse until enough matches
are played. Ratings shrink toward average by minutes or matches seen rather than
hiding behind a threshold, so early-season numbers read as uncertain instead of wrong.

---

## License

[MIT](LICENSE) — covers the source only. Third-party football data is not included
or redistributed; see the note under **Data sources**.
