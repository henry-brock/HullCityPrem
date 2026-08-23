"""
ratings.py — opponent-adjusted, multi-season, time-decayed team ratings (#1+#2).

Raw goals-per-game ignore who you played; a single season is noisy. This
solves an iterative multiplicative attack/defence rating (Dixon-Coles family)
over several Premier League seasons (football-data E0), weighting recent
seasons more by exponential decay. A team's rating is therefore "goals vs an
average-defence opponent at neutral venue", corrected for schedule strength
and trajectory.

Produces baseAtk/baseDef for the 17 retained teams. Promoted teams keep the
Championship-scaling pipeline (promoted_analyzer.py).

Out-of-sample check: ratings trained on seasons BEFORE 2025/26 predict
2025/26 match outcomes better than single-season goal/xG priors do.
"""

import math

from championship_scraper import fetch_csv, parse_matches
from scraper import TEAM_MAP

SEASONS = ["2324", "2425", "2526"]   # oldest → newest
DECAY = 0.55                          # weight of each season older
LEAGUE_AVG = 1.40
PROMOTED = {"hull", "ipswich", "coventry"}


def _load(seasons):
    games = []
    for i, code in enumerate(seasons):
        w = DECAY ** (len(seasons) - 1 - i)
        for m in parse_matches(fetch_csv(code, "E0")):
            games.append((m, w))
    return games


def solve(games, iters=80):
    teams = {m["home"] for m, _ in games} | {m["away"] for m, _ in games}
    atk = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    tw = sum(w for _, w in games)
    mu_h = sum(w * m["hg"] for m, w in games) / tw
    mu_a = sum(w * m["ag"] for m, w in games) / tw
    mu = (mu_h + mu_a) / 2
    H, A = mu_h / mu, mu_a / mu

    for _ in range(iters):
        na = {t: 0.0 for t in teams}; da = {t: 0.0 for t in teams}
        nd = {t: 0.0 for t in teams}; dd = {t: 0.0 for t in teams}
        for m, w in games:
            h, a = m["home"], m["away"]
            na[h] += w * m["hg"]; da[h] += w * dfn[a] * H * mu
            na[a] += w * m["ag"]; da[a] += w * dfn[h] * A * mu
            nd[a] += w * m["hg"]; dd[a] += w * atk[h] * H * mu
            nd[h] += w * m["ag"]; dd[h] += w * atk[a] * A * mu
        for t in teams:
            if da[t] > 0: atk[t] = na[t] / da[t]
            if dd[t] > 0: dfn[t] = nd[t] / dd[t]
        ma = sum(atk.values()) / len(atk)
        md = sum(dfn.values()) / len(dfn)
        for t in teams:
            atk[t] /= ma; dfn[t] /= md
    return atk, dfn, mu, H, A


def priors(seasons=SEASONS):
    atk, dfn, mu, _, _ = solve(_load(seasons))
    out = {}
    for name in atk:
        tid = TEAM_MAP.get(name)
        if tid and tid not in PROMOTED:
            out[tid] = (round(atk[name] * mu, 2), round(dfn[name] * mu, 2))
    return out


# ── out-of-sample validation ───────────────────────────────────────────────
def _pmf(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0)


def _wdl(hxg, axg):
    hw = dr = 0.0
    for i in range(11):
        for j in range(11):
            p = _pmf(i, hxg) * _pmf(j, axg)
            if i > j: hw += p
            elif i == j: dr += p
    return hw, dr, 1 - hw - dr


def _logloss_on(test, rate_fn):
    ll = n = 0
    for m in test:
        pr = rate_fn(m["home"], m["away"])
        if not pr:
            continue
        hxg, axg = pr
        hw, dr, aw = _wdl(max(0.05, hxg), max(0.05, axg))
        out = "H" if m["hg"] > m["ag"] else "A" if m["ag"] > m["hg"] else "D"
        p = {"H": hw, "D": dr, "A": aw}[out]
        ll += -math.log(max(p, 1e-12)); n += 1
    return ll / n, n


def validate():
    test = parse_matches(fetch_csv("2526", "E0"))

    # (a) multi-season opponent-adjusted ratings, trained pre-2526
    atk, dfn, mu, H, A = solve(_load(["2324", "2425"]))
    def rated(h, a):
        if h not in atk or a not in atk:
            return None
        return (atk[h] * dfn[a] * H * mu, atk[a] * dfn[h] * A * mu)

    # (b) baseline: single-season 2024/25 raw goals/game
    prev = {}
    pg = {}
    for m in parse_matches(fetch_csv("2425", "E0")):
        for t in (m["home"], m["away"]):
            prev.setdefault(t, [0, 0]); pg.setdefault(t, 0)
        prev[m["home"]][0] += m["hg"]; prev[m["home"]][1] += m["ag"]; pg[m["home"]] += 1
        prev[m["away"]][0] += m["ag"]; prev[m["away"]][1] += m["hg"]; pg[m["away"]] += 1
    def baseline(h, a):
        if h not in prev or a not in prev:
            return None
        ha = prev[h][0] / pg[h]; hd = prev[h][1] / pg[h]
        aa = prev[a][0] / pg[a]; ad = prev[a][1] / pg[a]
        return ((ha / LEAGUE_AVG) * ad * 1.11, (aa / LEAGUE_AVG) * hd * 0.89)

    r_ll, r_n = _logloss_on(test, rated)
    b_ll, b_n = _logloss_on(test, baseline)
    print("Out-of-sample: predicting 2025/26 from prior seasons only")
    print(f"  single-season raw goals   log-loss {b_ll:.4f}  ({b_n} matches)")
    print(f"  multi-season opp-adjusted log-loss {r_ll:.4f}  ({r_n} matches)")
    print(f"  → {(b_ll - r_ll)/b_ll*100:+.1f}% vs baseline "
          f"({'better' if r_ll < b_ll else 'worse'})")


def main():
    validate()
    print("\n══ 2026/27 ratings (opp-adjusted, 3-season decayed) ══")
    p = priors()
    print(f"{'team':<14}{'baseAtk':>9}{'baseDef':>9}")
    for tid in sorted(p, key=lambda t: -p[t][0]):
        print(f"{tid:<14}{p[tid][0]:>9}{p[tid][1]:>9}")
    print("\n══ copy-paste ══")
    for tid, (a, d) in sorted(p.items()):
        print(f"  {tid}: baseAtk: {a}, baseDef: {d},")


if __name__ == "__main__":
    main()
