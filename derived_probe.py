"""
derived_probe.py — which derived team metrics actually predict ability, and
which are candidates for the simulation.

Three questions, in the order that matters:

  REPEATS   does a team's first-half figure predict its own second-half
            figure? A metric that doesn't persist describes what happened,
            not what a team is.
  PREDICTS  does the first-half figure predict second-half GOALS (scored for
            attacking metrics, conceded for defensive ones)? Repeatable and
            useless is a real combination — possession repeats at 0.86 and
            forecasts almost nothing.
  ADDS      does it still predict once shot volume is accounted for? The
            model already knows shots for and against. A metric only earns a
            place in the simulation if it survives that control.

Only the third column justifies changing the model. The first two are
context for reading the page.
"""

import json
from collections import defaultdict
from pathlib import Path

from derived_stats import METRICS
from newcache_probe import _corr, _mean
from prior_backtest import load, _ols

ROOT = Path(__file__).parent
HALF = 19
SEASONS = ["2324", "2425", "2526"]


def _sum_blocks(rows):
    """Sum every raw Opta field across a set of matches, own and opponent."""
    own, opp, gf, ga = defaultdict(float), defaultdict(float), 0.0, 0.0
    for o, p, g_for, g_against in rows:
        for k, v in (o or {}).items():
            own[k] += float(v or 0)
        for k, v in (p or {}).items():
            opp[k] += float(v or 0)
        gf += g_for
        ga += g_against
    return own, opp, gf, ga


def team_halves():
    """(team, season) → (first-half aggregate, second-half aggregate)."""
    games = defaultdict(list)
    for s in SEASONS:
        for m in load(s):
            for side, other in (("home", "away"), ("away", "home")):
                tid = m.get(f"{side}Id") or m.get(f"{side}Name")
                games[(tid, s)].append((
                    m["ms"], m["stats"].get(side), m["stats"].get(other),
                    float(m[f"{side}Goals"]), float(m[f"{other}Goals"]),
                ))
    out = {}
    for key, rows in games.items():
        if len(rows) < HALF * 2:
            continue
        rows.sort(key=lambda r: r[0])
        strip = lambda rs: [(r[1], r[2], r[3], r[4]) for r in rs]
        out[key] = (_sum_blocks(strip(rows[:HALF])), _sum_blocks(strip(rows[HALF:])))
    return out


def main():
    halves = team_halves()
    keys = sorted(halves)
    print(f"{len(keys)} team-seasons across {', '.join(SEASONS)}\n")

    # Second-half outcomes and the shot-volume control.
    h2_gf = [halves[k][1][2] for k in keys]
    h2_ga = [halves[k][1][3] for k in keys]
    ctrl_for = [halves[k][0][0].get("total_scoring_att", 0.0) for k in keys]
    ctrl_ag  = [halves[k][0][1].get("total_scoring_att", 0.0) for k in keys]

    def residual(y, x):
        """y with the linear effect of x removed."""
        w = _ols([[v] for v in x], y)
        return [yy - (w[0] * xx + w[-1]) for yy, xx in zip(y, x)]

    rows = []
    for name, group, _desc, lower_better, fn in METRICS:
        h1, h2 = [], []
        for k in keys:
            a, b = halves[k]
            h1.append(fn(a[0], a[1], a[2], a[3]))
            h2.append(fn(b[0], b[1], b[2], b[3]))
        if sum(1 for v in h1 if v is None) > len(h1) * 0.2:
            continue
        pairs = [(x, y, gf, ga, cf, ca) for x, y, gf, ga, cf, ca
                 in zip(h1, h2, h2_gf, h2_ga, ctrl_for, ctrl_ag)
                 if x is not None and y is not None]
        if len(pairs) < 20:
            continue
        x  = [p[0] for p in pairs]
        y  = [p[1] for p in pairs]

        # Defensive metrics are judged against goals CONCEDED, attacking ones
        # against goals scored. Sign is flipped for "lower is better" so a
        # positive number always means "more of this, better outcome".
        defensive = group in ("Chance suppression",) or "conceded" in name.lower() \
            or "allowed" in name.lower() or "faced" in name.lower() \
            or name in ("Block rate", "Save percentage", "Take-ons stopped %",
                        "Ground duel win %", "Aerial win %", "Tackle success %",
                        "Defensive actions per opponent final-third entry",
                        "Clearances per opponent box touch")
        target = [p[3] for p in pairs] if defensive else [p[2] for p in pairs]
        ctrl   = [p[5] for p in pairs] if defensive else [p[4] for p in pairs]
        sign   = -1 if defensive else 1
        if lower_better:
            sign *= -1

        repeats  = _corr(x, y)
        predicts = _corr(x, target) * sign
        # Same correlation, but on the part of the outcome shot volume cannot
        # explain, and on the part of the metric shot volume cannot explain.
        adds = _corr(residual(x, ctrl), residual(target, ctrl)) * sign
        rows.append((name, group, repeats, predicts, adds, len(pairs)))

    rows.sort(key=lambda r: -abs(r[4]))
    print(f"{'metric':<44}{'repeats':>9}{'predicts':>10}{'adds':>8}")
    print("-" * 71)
    for name, group, rep, pred, adds, n in rows:
        star = " ★" if abs(adds) >= 0.25 and abs(rep) >= 0.35 else ""
        print(f"{name[:43]:<44}{rep:>9.2f}{pred:>10.2f}{adds:>8.2f}{star}")
    print("\n★ = repeats AND still predicts after shot volume is controlled for")
    print("   (candidates for the simulation; everything else is descriptive)")


if __name__ == "__main__":
    main()
