"""
mc_convergence.py — how much does the Monte Carlo answer move between sample
sizes?

The app runs an adaptive simulation (6k–30k, stopping once the largest
relegation-probability standard error drops under 0.004) and a 50k survival
sim. This asks the obvious question: what would actually change with more.

Method mirrors the app: correlated log-normal season noise (a shared form
factor lifting attack and tightening defence together, plus independent
wobble), per-team sigma widening for promoted and thin-prior sides, dynamic
sigma scaled by how much of the season is unplayed, and ANTITHETIC VARIATES —
each pair of simulations uses +g and −g, which cancels much of the sampling
noise and is why the app's numbers are steadier than raw N would suggest.

Repeating each sample size several times is the point. A single run tells you
nothing about its own precision; the spread ACROSS runs is the error bar.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

SIGMA_FORM, SIGMA_IND = 0.13, 0.08
HOME_ATK, AWAY_ATK, LEAGUE_AVG = 1.11, 0.89, 1.40
PROMOTED_SIGMA = {"hull": 1.45, "ipswich": 1.20, "coventry": 1.15}
THIN_PRIOR_SIGMA = {"sunderland": 1.15, "leeds": 1.15}


def _load_model():
    from backtest import PRIORS
    priors = dict(PRIORS)
    priors.update({"hull": (0.92, 1.95), "ipswich": (0.99, 1.91),
                   "coventry": (1.03, 1.88)})
    fixtures = json.loads((ROOT / "cache" / "fixtures_2627.json").read_text())["fixtures"]
    results = json.loads((ROOT / "data.json").read_text())["results"]
    return priors, fixtures, results


def simulate(n, seed, priors, fixtures, results):
    """Return {team: relegation probability} and {team: mean points}."""
    rng = random.Random(seed)
    teams = sorted(priors)
    ti = {t: i for i, t in enumerate(teams)}
    atk = [priors[t][0] for t in teams]
    dfn = [priors[t][1] for t in teams]
    sig = [(PROMOTED_SIGMA.get(t) or THIN_PRIOR_SIGMA.get(t) or 1.0) for t in teams]

    played = {(r["homeId"], r["awayId"]): (r["homeGoals"], r["awayGoals"]) for r in results}
    rem = [(ti[f["homeId"]], ti[f["awayId"]]) for f in fixtures
           if (f["homeId"], f["awayId"]) not in played]
    # Points and goal difference already banked from completed matches.
    base_pts = [0] * len(teams)
    base_gd = [0] * len(teams)
    for (h, a), (hg, ag) in played.items():
        ih, ia = ti[h], ti[a]
        base_gd[ih] += hg - ag
        base_gd[ia] += ag - hg
        if hg > ag:   base_pts[ih] += 3
        elif ag > hg: base_pts[ia] += 3
        else:         base_pts[ih] += 1; base_pts[ia] += 1

    scale = 0.85 + 0.45 * (len(rem) / len(fixtures))
    nt = len(teams)
    rel = [0] * nt
    pts_sum = [0] * nt
    gauss = rng.gauss
    rnd = rng.random
    exp = math.exp

    half = n // 2
    for _ in range(half):
        # Antithetic pair: the same draws applied with both signs.
        gf = [gauss(0, 1) for _ in range(nt)]
        gi1 = [gauss(0, 1) for _ in range(nt)]
        gi2 = [gauss(0, 1) for _ in range(nt)]
        for sign in (1, -1):
            ra = [0.0] * nt
            rd = [0.0] * nt
            for i in range(nt):
                s = sig[i] * scale
                form = exp(sign * gf[i] * SIGMA_FORM * s)
                ra[i] = atk[i] * form * exp(sign * gi1[i] * SIGMA_IND * s)
                rd[i] = dfn[i] / form * exp(sign * gi2[i] * SIGMA_IND * s)
            pts = base_pts[:]
            gd = base_gd[:]
            for ih, ia in rem:
                lh = (ra[ih] / LEAGUE_AVG) * rd[ia] * HOME_ATK
                la = (ra[ia] / LEAGUE_AVG) * rd[ih] * AWAY_ATK
                # Inverse-transform Poisson; means near 1.4 so this is short.
                k = 0; p = 1.0; L = exp(-lh)
                while True:
                    p *= rnd()
                    if p <= L: break
                    k += 1
                hg = k
                k = 0; p = 1.0; L = exp(-la)
                while True:
                    p *= rnd()
                    if p <= L: break
                    k += 1
                ag = k
                gd[ih] += hg - ag; gd[ia] += ag - hg
                if hg > ag:   pts[ih] += 3
                elif ag > hg: pts[ia] += 3
                else:         pts[ih] += 1; pts[ia] += 1
            order = sorted(range(nt), key=lambda i: (-pts[i], -gd[i], rnd()))
            for i in order[17:]:
                rel[i] += 1
            for i in range(nt):
                pts_sum[i] += pts[i]

    total = half * 2
    return ({teams[i]: rel[i] / total * 100 for i in range(nt)},
            {teams[i]: pts_sum[i] / total for i in range(nt)})


def main(sizes=(10000, 100000), repeats=3):
    priors, fixtures, results = _load_model()
    out = {}
    for n in sizes:
        runs = []
        t0 = time.time()
        for r in range(repeats):
            runs.append(simulate(n, 1000 + r, priors, fixtures, results))
            print(f"  [{n:,}] run {r+1}/{repeats} done", file=sys.stderr, flush=True)
        out[n] = (runs, (time.time() - t0) / repeats)

    teams = sorted(out[sizes[0]][0][0][0])
    mean = lambda xs: sum(xs) / len(xs)
    sd = lambda xs: (mean([(x - mean(xs)) ** 2 for x in xs])) ** 0.5

    print(f"\nRelegation % — mean across {repeats} runs, and the spread BETWEEN runs\n")
    hdr = "".join(f"{('N=' + format(n, ',')):>26}" for n in sizes)
    print(f"{'team':<15}{hdr}")
    print("-" * (15 + 26 * len(sizes)))
    rows = sorted(teams, key=lambda t: -mean([r[0][t] for r in out[sizes[0]][0]]))
    for t in rows:
        line = f"{t:<15}"
        for n in sizes:
            vals = [r[0][t] for r in out[n][0]]
            line += f"{mean(vals):>18.2f} ±{sd(vals):<6.2f}"
        print(line)

    print(f"\n{'':<15}" + "".join(f"{('N=' + format(n, ',')):>26}" for n in sizes))
    for label, fn in (("largest spread between runs",
                       lambda n: max(sd([r[0][t] for r in out[n][0]]) for t in teams)),
                      ("mean spread between runs",
                       lambda n: mean([sd([r[0][t] for r in out[n][0]]) for t in teams])),
                      ("seconds per run", lambda n: out[n][1])):
        print(f"{label:<15}" + "".join(f"{fn(n):>25.3f} " for n in sizes))

    a, b = sizes[0], sizes[-1]
    shifts = [abs(mean([r[0][t] for r in out[a][0]]) - mean([r[0][t] for r in out[b][0]]))
              for t in teams]
    print(f"\nLargest shift in the ANSWER between {a:,} and {b:,}: "
          f"{max(shifts):.2f}pp (mean {mean(shifts):.2f}pp)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Monte Carlo convergence check")
    ap.add_argument("--sizes", default="10000,100000")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    main(tuple(int(x) for x in a.sizes.split(",")), a.repeats)
