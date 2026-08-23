"""
championship_sim.py — basic Monte-Carlo of the 2025/26 Championship to test
whether Hull's 6th-place finish + promotion was earned or overperformed.

Two rating bases, same engine:
  • PROCESS (xG)  — opponent-adjusted ratings from the shots-based xG proxy
  • RESULTS       — opponent-adjusted ratings from actual goals

If Hull's promotion chance is low on PROCESS but high on RESULTS, they
out-performed how they actually played (results ≫ underlying) — i.e. the
promotion isn't reflected in the process data the PL prior is built from.

Promotion = top 2 (automatic) OR win the play-off (3rd–6th, two-legged
semis + final).
"""

import math
import random

from championship_scraper import fetch_csv, parse_matches, fit_xg, team_table

SIMS = 4000
SIGMA_FORM, SIGMA_IND = 0.13, 0.08


def solve(games, gfk, gak, iters=80):
    """Iterative opponent-adjusted attack/defence on metric (gfk/gak)."""
    teams = {m["home"] for m in games} | {m["away"] for m in games}
    atk = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    n = len(games)
    mu_h = sum(m[gfk] for m in games) / n
    mu_a = sum(m[gak] for m in games) / n
    mu = (mu_h + mu_a) / 2
    H, A = mu_h / mu, mu_a / mu
    for _ in range(iters):
        na = {t: 0.0 for t in teams}; da = {t: 0.0 for t in teams}
        nd = {t: 0.0 for t in teams}; dd = {t: 0.0 for t in teams}
        for m in games:
            h, a = m["home"], m["away"]
            na[h] += m[gfk]; da[h] += dfn[a] * H * mu
            na[a] += m[gak]; da[a] += dfn[h] * A * mu
            nd[a] += m[gfk]; dd[a] += atk[h] * H * mu
            nd[h] += m[gak]; dd[h] += atk[a] * A * mu
        for t in teams:
            if da[t]: atk[t] = na[t] / da[t]
            if dd[t]: dfn[t] = nd[t] / dd[t]
        ma = sum(atk.values()) / len(atk); md = sum(dfn.values()) / len(dfn)
        for t in teams: atk[t] /= ma; dfn[t] /= md
    return atk, dfn, mu, H, A


def _pois(lam):
    L, k, p = math.exp(-max(0.05, lam)), 0, 1.0
    while True:
        k += 1; p *= random.random()
        if p <= L: return k - 1


def _match(h, a, r, mu, H, A, neutral=False):
    hf = 1.0 if neutral else H
    af = 1.0 if neutral else A
    hxg = r[h][0] / mu * r[a][1] * hf
    axg = r[a][0] / mu * r[h][1] * af
    return _pois(hxg), _pois(axg)


def _playoff(po, r, mu, H, A):
    # po = [3rd,4th,5th,6th] ids. Semis: 3v6, 4v5 (two legs); winners → final.
    def two_leg(x, y):           # x higher seed (home 2nd leg)
        g1 = _match(y, x, r, mu, H, A); g2 = _match(x, y, r, mu, H, A)
        xt = g1[1] + g2[0]; yt = g1[0] + g2[1]
        return x if xt > yt else y if yt > xt else random.choice([x, y])
    f1 = two_leg(po[0], po[3])
    f2 = two_leg(po[1], po[2])
    gh, ga = _match(f1, f2, r, mu, H, A, neutral=True)
    return f1 if gh > ga else f2 if ga > gh else random.choice([f1, f2])


def simulate(atk, dfn, mu, H, A):
    teams = list(atk)
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    auto = {t: 0 for t in teams}; po6 = {t: 0 for t in teams}
    promo = {t: 0 for t in teams}; pos_sum = {t: 0 for t in teams}
    for _ in range(SIMS):
        r = {}
        for t in teams:
            form = math.exp(random.gauss(0, SIGMA_FORM))
            r[t] = (atk[t] * form * math.exp(random.gauss(0, SIGMA_IND)),
                    dfn[t] / form * math.exp(random.gauss(0, SIGMA_IND)))
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}
        for h, a in fixtures:
            hg = _pois(r[h][0] / mu * r[a][1] * H)
            ag = _pois(r[a][0] / mu * r[h][1] * A)
            gd[h] += hg - ag; gd[a] += ag - hg
            if hg > ag: pts[h] += 3
            elif ag > hg: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
        order = sorted(teams, key=lambda t: (-pts[t], -gd[t], random.random()))
        for i, t in enumerate(order): pos_sum[t] += i + 1
        for t in order[:2]: auto[t] += 1
        for t in order[2:6]: po6[t] += 1
        winner = _playoff(order[2:6], r, mu, H, A)
        for t in order[:2]: promo[t] += 1
        promo[winner] += 1
    return {t: {"auto": auto[t] / SIMS * 100, "po6": po6[t] / SIMS * 100,
                "promo": promo[t] / SIMS * 100, "xpos": pos_sum[t] / SIMS}
            for t in teams}


def main():
    raw = parse_matches(fetch_csv("2526", "E1"))
    a, b = fit_xg(raw)
    for m in raw:
        from championship_scraper import xg
        m["hxg"] = xg(m["hst"], m["hs"], a, b)
        m["axg"] = xg(m["ast"], m["as"], a, b)

    # actual final table for reference
    tbl = team_table(raw, a, b)
    order = sorted(tbl, key=lambda t: (-tbl[t]["pts"], -(tbl[t]["gf"] - tbl[t]["ga"])))
    actual = {t: i + 1 for i, t in enumerate(order)}

    res_p = simulate(*solve(raw, "hxg", "axg"))   # process (xG)
    res_r = simulate(*solve(raw, "hg", "ag"))     # results (goals)

    print(f"{'team':<16}{'actual':>7}{'xPos(xG)':>10}{'promo%xG':>10}{'promo%goals':>12}")
    for t in sorted(res_p, key=lambda t: res_p[t]["xpos"])[:8]:
        print(f"{t:<16}{actual[t]:>7}{res_p[t]['xpos']:>10.1f}"
              f"{res_p[t]['promo']:>10.1f}{res_r[t]['promo']:>12.1f}")

    print("\n── Hull spotlight ──")
    h = "Hull"
    print(f"  Actual: finished {actual[h]}th, PROMOTED (play-offs).")
    print(f"  By PROCESS (xG):  expected finish {res_p[h]['xpos']:.1f},  "
          f"top-2 {res_p[h]['auto']:.1f}%,  top-6 {res_p[h]['po6']:.1f}%,  "
          f"promotion {res_p[h]['promo']:.1f}%")
    print(f"  By RESULTS (goals): promotion {res_r[h]['promo']:.1f}%")
    print(f"\n  → On how they actually PLAYED, the model gives Hull a "
          f"{res_p[h]['promo']:.0f}% promotion chance — yet they went up. "
          f"That gap is the overperformance.")


if __name__ == "__main__":
    main()
