"""
prior_backtest.py — the missing harness: does a pre-season PRIOR actually
predict better?

Why this exists
───────────────
backtest.py's --bench and --table both rate teams via _venue_ratings(), which
starts every team at LEAGUE_AVG and blends in observed results. Neither ever
reads PRIORS (verified: corrupting PRIORS to all-1.0 leaves --bench output
byte-identical). So the existing harnesses cannot score a prior change.

This one can. It derives priors from TRAINING seasons only, then predicts every
match of a held-out season using nothing but those priors — no in-season
blending — so the score isolates prior quality. Scored with the same
Dixon-Coles machinery and home/away split the live model uses.

Usage:  python3 prior_backtest.py [--train 2324,2425] [--test 2526]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from backtest import (LEAGUE_AVG, MAXG, _logloss, _outcome, _rps, score_matrix)
from newcache_probe import _g, _mean

ROOT     = Path(__file__).parent
NEWCACHE = ROOT / "Newcache"

HOME_ATK, AWAY_ATK, RHO = 1.11, 0.89, -0.13


def load(season):
    path = NEWCACHE / f"pl_{season}.json"
    if not path.exists():
        return []
    return [m for m in json.loads(path.read_text()).get("matches", [])
            if m.get("homeGoals") is not None and m["stats"].get("home")]


def team_rows(seasons, by_id=False):
    """
    team → list of per-match (own_stats, opp_stats, goals_for, goals_against).

    by_id=True keys on the internal team id instead of the Pulse name, and
    drops teams outside the 2026/27 roster (whose id is null) — that is what
    derive_priors needs; the backtest wants every team, so it keys on name.
    """
    rows = defaultdict(list)
    for season in seasons:
        for m in load(season):
            hs, as_ = m["stats"]["home"], m["stats"]["away"]
            hg, ag  = float(m["homeGoals"]), float(m["awayGoals"])
            home = m["homeId"] if by_id else m["homeName"]
            away = m["awayId"] if by_id else m["awayName"]
            if home:
                rows[home].append((hs, as_, hg, ag))
            if away:
                rows[away].append((as_, hs, ag, hg))
    return rows


# ── metric definitions ──────────────────────────────────────────────────
def m_goals(s, gf):          return gf
def m_shots(s, gf):          return _g(s, "total_scoring_att")
def m_touches(s, gf):        return _g(s, "touches_in_opp_box")
def m_proxy(s, gf):
    sot = _g(s, "ontarget_scoring_att")
    return 0.33 * sot - 0.01 * (_g(s, "total_scoring_att") - sot)


def _ols(X, y):
    """Least squares with intercept, via normal equations + Gaussian elim."""
    X = [list(r) + [1.0] for r in X]
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(A[k][i]))
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        if abs(A[i][i]) < 1e-12:
            return [0.0] * n
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    w = [0.0] * n
    for i in reversed(range(n)):
        w[i] = (b[i] - sum(A[i][j] * w[j] for j in range(i + 1, n))) / A[i][i]
    return w


def build_priors(train_rows, method):
    """
    team → (atk, def) in goals/game, rescaled so each side averages LEAGUE_AVG.

    'fitted' methods regress the chosen per-game features onto per-game goals
    across training team-seasons, so the output is already in goals units.
    """
    per_team = {}
    for team, games in train_rows.items():
        if len(games) < 20:
            continue
        per_team[team] = {
            "gf":      _mean([gf for _s, _o, gf, _ga in games]),
            "ga":      _mean([ga for _s, _o, _gf, ga in games]),
            "shots":   _mean([m_shots(s, 0) for s, _o, _gf, _ga in games]),
            "touches": _mean([m_touches(s, 0) for s, _o, _gf, _ga in games]),
            "proxy":   _mean([m_proxy(s, 0) for s, _o, _gf, _ga in games]),
            # conceded versions: what opponents produced against this team
            "shots_a":   _mean([m_shots(o, 0) for _s, o, _gf, _ga in games]),
            "touches_a": _mean([m_touches(o, 0) for _s, o, _gf, _ga in games]),
            "proxy_a":   _mean([m_proxy(o, 0) for _s, o, _gf, _ga in games]),
        }

    teams = sorted(per_team)
    if method == "goals":
        atk = {t: per_team[t]["gf"] for t in teams}
        dfn = {t: per_team[t]["ga"] for t in teams}
    elif method == "proxy":
        atk = {t: per_team[t]["proxy"] for t in teams}
        dfn = {t: per_team[t]["proxy_a"] for t in teams}
    elif method in ("shots+touches", "shots"):
        feats   = ["shots", "touches"] if method == "shots+touches" else ["shots"]
        feats_a = [f + "_a" for f in feats]
        wa = _ols([[per_team[t][f] for f in feats]   for t in teams],
                  [per_team[t]["gf"] for t in teams])
        wd = _ols([[per_team[t][f] for f in feats_a] for t in teams],
                  [per_team[t]["ga"] for t in teams])
        atk = {t: sum(wa[i] * per_team[t][f] for i, f in enumerate(feats)) + wa[-1]
               for t in teams}
        dfn = {t: sum(wd[i] * per_team[t][f] for i, f in enumerate(feats_a)) + wd[-1]
               for t in teams}
    elif method == "atk:shots+touches / def:proxy":
        feats = ["shots", "touches"]
        wa = _ols([[per_team[t][f] for f in feats] for t in teams],
                  [per_team[t]["gf"] for t in teams])
        atk = {t: sum(wa[i] * per_team[t][f] for i, f in enumerate(feats)) + wa[-1]
               for t in teams}
        dfn = {t: per_team[t]["proxy_a"] for t in teams}
    else:
        raise ValueError(method)

    # Rescale each side to the league mean so the Poisson stays self-consistent.
    ma = _mean(list(atk.values())) or 1.0
    md = _mean(list(dfn.values())) or 1.0
    return {t: (max(0.15, atk[t] * LEAGUE_AVG / ma),
                max(0.15, dfn[t] * LEAGUE_AVG / md)) for t in teams}


def build_priors_oppadj(seasons, metric="goals"):
    """
    Production's actual approach: ratings.py's opponent-adjusted iterative
    solve, fed from Newcache so only the METHOD differs from the other rows.

    metric="goals"  — what production does today.
    metric="atk"    — same solver, but each match's "goals" replaced by the
                      shots+touches attack estimate, i.e. option 2: put the
                      better attack signal through the opponent adjustment
                      instead of replacing the adjustment.

    Note: weights are flat here, whereas production also applies a
    cross-season time decay — so this isolates opponent adjustment itself.
    """
    import ratings

    matches = [m for s in seasons for m in load(s)]

    if metric == "atk":
        # Fit per-match attacking output from shots + touches, league-wide.
        rows, ys = [], []
        for m in matches:
            for side, opp, g in (("home", "away", m["homeGoals"]),
                                 ("away", "home", m["awayGoals"])):
                s = m["stats"][side]
                rows.append([m_shots(s, 0), m_touches(s, 0)])
                ys.append(float(g))
        w = _ols(rows, ys)
        est = lambda s: max(0.0, w[0] * m_shots(s, 0) + w[1] * m_touches(s, 0) + w[-1])
    else:
        est = None

    games = []
    for m in matches:
        if est:
            hg, ag = est(m["stats"]["home"]), est(m["stats"]["away"])
        else:
            hg, ag = float(m["homeGoals"]), float(m["awayGoals"])
        games.append(({"home": m["homeName"], "away": m["awayName"],
                       "hg": hg, "ag": ag}, 1.0))

    atk, dfn, mu, _H, _A = ratings.solve(games)
    return {t: (max(0.15, atk[t] * mu), max(0.15, dfn[t] * mu)) for t in atk}


# Attacking stats with full coverage back to 2023/24, averaged into one index.
# Deliberately the whole obvious attacking family rather than a screened
# subset — no stat is selected on performance, so there is nothing to overfit.
COMPOSITE_STATS = [
    "total_scoring_att", "ontarget_scoring_att", "attempts_ibox",
    "att_bx_centre", "touches_in_opp_box", "big_chance_created",
    "total_att_assist", "att_openplay",
]


def _weighted_teams(seasons, target, decay):
    """
    team → per-game stat means, weighting each season by decay**(seasons away
    from `target`). decay=1.0 or target=None gives equal weighting.

    Recency matters in principle — squads turn over, and ratings.py (the
    previous production method) was explicitly time-decayed. Measured effect
    on the folds is ~nil (see the decay sweep), largely because only three
    seasons are in range and one fold's training seasons are equidistant from
    its target, so this is kept mild.
    """
    from stat_sweep import aggregates, by_team
    from pulse_results import _season_year

    if not target or decay == 1.0:
        return by_team(aggregates(list(seasons)))

    ty = _season_year(target)
    w  = {s: decay ** abs(_season_year(s) - ty) for s in seasons}

    grouped = defaultdict(list)
    for (team, season), v in aggregates(list(seasons)).items():
        grouped[team].append((w[season], v))

    out = {}
    for team, vs in grouped.items():
        tot = sum(x for x, _ in vs) or 1.0
        k_own = set().union(*[set(v["own"]) for _, v in vs])
        k_opp = set().union(*[set(v["opp"]) for _, v in vs])
        out[team] = {
            "own": {k: sum(x * v["own"].get(k, 0.0) for x, v in vs) / tot for k in k_own},
            "opp": {k: sum(x * v["opp"].get(k, 0.0) for x, v in vs) / tot for k in k_opp},
            "gf":  sum(x * v["gf"] for x, v in vs) / tot,
            "ga":  sum(x * v["ga"] for x, v in vs) / tot,
        }
    return out


def build_priors_composite(seasons, stats=None, target=None, decay=1.0):
    """
    Composite z-index priors: average several correlated attacking stats into
    a single index, then map that index to goals/game.

    Why this beats picking one stat, or regressing on several: averaging
    cancels the per-stat measurement noise that makes any individual metric
    unstable across seasons, while fitting only TWO parameters per side
    (slope + intercept) no matter how many stats go in. Greedy multi-feature
    regression, by contrast, fits one parameter per stat and overfits — it
    loses the 2425 fold outright.

    Attack uses a team's own stats; defence uses the same stats as generated
    BY its opponents, i.e. chances it allowed.

    `target`/`decay` apply recency weighting relative to the season being
    predicted (see _weighted_teams); the default is equal weighting.
    """
    stats = stats or COMPOSITE_STATS
    agg   = _weighted_teams(list(seasons), target, decay)
    teams = sorted(agg)
    out   = {}

    for goal_key, side, idx in (("gf", "own", 0), ("ga", "opp", 1)):
        zs = []
        for stat in stats:
            xs = [agg[t][side].get(stat, 0.0) for t in teams]
            mu = _mean(xs)
            sd = (_mean([(x - mu) ** 2 for x in xs])) ** 0.5 or 1.0
            zs.append({t: (agg[t][side].get(stat, 0.0) - mu) / sd for t in teams})
        comp = {t: _mean([z[t] for z in zs]) for t in teams}
        w = _ols([[comp[t]] for t in teams], [agg[t][goal_key] for t in teams])
        for t in teams:
            cur = out.get(t, [LEAGUE_AVG, LEAGUE_AVG])
            cur[idx] = max(0.15, w[0] * comp[t] + w[-1])
            out[t] = cur

    ma = _mean([v[0] for v in out.values()]) or 1.0
    md = _mean([v[1] for v in out.values()]) or 1.0
    return {t: (v[0] * LEAGUE_AVG / ma, v[1] * LEAGUE_AVG / md)
            for t, v in out.items()}


def evaluate(priors, test_matches):
    """Score every test match from pre-season priors alone."""
    ll = rps = 0.0
    n = 0
    for m in test_matches:
        h, a = m["homeName"], m["awayName"]
        # A team absent from training (just promoted) gets a neutral prior —
        # applied identically for every method, so comparisons stay fair.
        ha, hd = priors.get(h, (LEAGUE_AVG, LEAGUE_AVG))
        aa, ad = priors.get(a, (LEAGUE_AVG, LEAGUE_AVG))
        hxg = max(0.05, (ha / LEAGUE_AVG) * ad * HOME_ATK)
        axg = max(0.05, (aa / LEAGUE_AVG) * hd * AWAY_ATK)
        mat = score_matrix(hxg, axg, RHO)
        hw  = sum(mat[i][j] for i in range(MAXG + 1) for j in range(MAXG + 1) if i > j)
        dr  = sum(mat[i][i] for i in range(MAXG + 1))
        out = _outcome(m["homeGoals"], m["awayGoals"])
        ll  += _logloss({"H": hw, "D": dr, "A": 1 - hw - dr}[out])
        rps += _rps(hw, dr, 1 - hw - dr, out)
        n   += 1
    return ll / n, rps / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2324,2425")
    ap.add_argument("--test",  default="2526")
    args = ap.parse_args()

    train = [s.strip() for s in args.train.split(",")]
    test  = [s.strip() for s in args.test.split(",")]

    train_rows   = team_rows(train)
    test_matches = [m for s in test for m in load(s)]
    if not test_matches:
        raise SystemExit("No test matches — build Newcache first.")

    print(f"Pre-season prior backtest — train {'+'.join(train)} → test {'+'.join(test)}")
    print(f"{len(test_matches)} matches, priors only (no in-season blending)\n")

    ll, rps, n = evaluate({}, test_matches)
    print(f"{'method':34}{'log-loss':>10}{'RPS':>9}")
    print("-" * 53)
    print(f"{'neutral (all 1.40) — floor':34}{ll:>10.4f}{rps:>9.4f}")

    results = {}
    for method in ("goals", "proxy", "shots", "shots+touches",
                   "atk:shots+touches / def:proxy"):
        priors = build_priors(train_rows, method)
        ll, rps, _ = evaluate(priors, test_matches)
        results[method] = ll
        print(f"{method:34}{ll:>10.4f}{rps:>9.4f}")

    for label, metric in (("opp-adj goals (ratings.py, previous)", "goals"),
                          ("opp-adj + shots&touches attack", "atk")):
        priors = build_priors_oppadj(train, metric)
        ll, rps, _ = evaluate(priors, test_matches)
        results[label] = ll
        print(f"{label:34}{ll:>10.4f}{rps:>9.4f}")

    priors = build_priors_composite(train)
    ll, rps, _ = evaluate(priors, test_matches)
    results["composite z-index ★"] = ll
    print(f"{'composite z-index (8 stats) ★':34}{ll:>10.4f}{rps:>9.4f}")

    prod = results["composite z-index ★"]
    best = min(results, key=results.get)
    print(f"\n★ = current production method (composite z-index)")
    print(f"best on THIS fold: {best}  ({prod - results[best]:+.4f} vs production)")
    print("\nWARNING: a single fold is not evidence — the best method here changes\n"
          "from season to season (that is how three earlier candidates were wrongly\n"
          "'validated'). Judge a prior change only on all three folds:\n"
          "  for f in '2324,2425 2526' '2324,2526 2425' '2425,2526 2324'; do ...")


if __name__ == "__main__":
    main()
