"""
derive_priors.py — generate baseAtk/baseDef for the 17 retained PL teams
from their real 2025/26 Premier League data.

Two modes:
  goals  — actual goals scored/conceded per game (results)
  xg ★   — expected goals per game from the shots-based xG proxy (process)

The xG mode is the recommendation: a first-half→second-half test on the
2025/26 PL showed first-half xG predicts second-half goals better than
first-half goals do, for both attack (0.77 vs 0.71) and especially defence
(0.46 vs 0.17). xG strips finishing/keeping luck, so it is the more
repeatable basis for a pre-season prior.

Promoted teams (Hull, Ipswich, Coventry) are untouched — they come from
promoted_analyzer.py's Championship-to-PL scaling.
"""

import json
from pathlib import Path

from championship_scraper import fetch_csv, parse_matches, fit_xg, team_table
from scraper import TEAM_MAP   # football-data name → internal id

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"
PROMOTED = {"hull", "ipswich", "coventry"}
GAMES = 38


def derive_goals():
    """Actual goals/game from ESPN team stats."""
    ESPN_MAP = {
        "Arsenal": "arsenal", "Aston Villa": "astonvilla",
        "AFC Bournemouth": "bournemouth", "Brentford": "brentford",
        "Brighton & Hove Albion": "brighton", "Chelsea": "chelsea",
        "Crystal Palace": "crystalpalace", "Everton": "everton",
        "Fulham": "fulham", "Leeds United": "leeds", "Liverpool": "liverpool",
        "Manchester City": "mancity", "Manchester United": "manutd",
        "Newcastle United": "newcastle", "Nottingham Forest": "nottmforest",
        "Sunderland": "sunderland", "Tottenham Hotspur": "tottenham",
    }
    data = json.loads((CACHE / "espn_teamstats_eng.1_2526.json").read_text())
    out = {}
    for t in data["teams"]:
        tid = ESPN_MAP.get(t["team"])
        if not tid:
            continue
        s = t["stats"]
        out[tid] = (round(s.get("totalGoals", 0) / GAMES, 2),
                    round(s.get("goalsConceded", 0) / GAMES, 2))
    return out


def derive_xg():
    """Expected goals/game from the shots-based xG proxy (football-data E0)."""
    matches = parse_matches(fetch_csv("2526", "E0"))
    a, b = fit_xg(matches)
    table = team_table(matches, a, b)
    out = {}
    for name, s in table.items():
        tid = TEAM_MAP.get(name)
        if not tid or tid in PROMOTED:
            continue
        out[tid] = (round(s["xgf"] / s["gp"], 2), round(s["xga"] / s["gp"], 2))
    return out


def derive_newcache(seasons=("2324", "2425", "2526")):
    """
    Asymmetric priors from the Newcache Opta archive (see newcache_probe.py):
      attack  — fitted from shots + touches in the opposition box
      defence — the existing shots-based xG proxy, left alone

    Rationale: a leave-one-season-out probe (n=60 team-seasons) found the
    attack side improvable (out-of-sample r 0.506 → 0.575) while defence was
    already at its best with the current proxy (0.540, nothing beat it).

    STATUS — NOT production, and NOT adopted. Match-level leave-one-season-out
    over the three relevant seasons gives it a small mean edge (≈−0.004
    log-loss vs production's opponent-adjusted ratings.py priors) but no
    method won consistently: the best performer changed in every fold, and
    fold-to-fold spread (~±0.03) dwarfs the difference between methods. On
    this project's own standard — the same one that gated OPP_ADJUST off for
    +0.0025 — that is not enough to change priors.

    Kept as a documented candidate so the work is reproducible if more
    evidence arrives. Re-check with `python3 prior_backtest.py`.
    """
    from prior_backtest import build_priors, team_rows

    rows   = team_rows(list(seasons), by_id=True)
    priors = build_priors(rows, "atk:shots+touches / def:proxy")
    return {t: (round(a, 2), round(d, 2))
            for t, (a, d) in priors.items() if t not in PROMOTED}


def main():
    goals = derive_goals()
    xg = derive_xg()
    print("══ Retained-team priors: goals (actual) vs xG (process) ══\n")
    print(f"{'team':<14}{'goals A/D':>14}{'xG A/D ★':>14}")
    for tid in sorted(xg, key=lambda t: -xg[t][0]):
        g = goals.get(tid, ("—", "—"))
        x = xg[tid]
        print(f"{tid:<14}{g[0]:>6}/{g[1]:<7}{x[0]:>6}/{x[1]:<7}")
    print(f"\nPromoted ({', '.join(sorted(PROMOTED))}): see promoted_analyzer.py.")

    print("\n══ xG priors — copy-paste ══")
    for tid, (a, d) in sorted(xg.items()):
        print(f"  {tid}: baseAtk: {a}, baseDef: {d},")

    try:
        nc = derive_newcache()
    except (ImportError, FileNotFoundError, KeyError):
        return

    from backtest import PRIORS as LIVE
    print("\n══ Newcache asymmetric priors (candidate — NOT production) ══")
    print(f"{'team':<14}{'live A/D':>14}{'newcache A/D':>16}{'Δatk':>8}")
    for tid in sorted(nc):
        if tid not in LIVE:
            continue
        la, ld = LIVE[tid]
        na, nd = nc[tid]
        print(f"{tid:<14}{la:>6}/{ld:<7}{na:>7}/{nd:<8}{na - la:>+8.2f}")
    print("\nSee derive_newcache() docstring: 2 of 3 folds improved, "
          "not conclusive. Validate with prior_backtest.py before adopting.")


if __name__ == "__main__":
    main()
