"""
verify_relegation.py — sanity-check the model's promoted-team relegation
probabilities against history.

1. Historical base rate: across recent cohorts, what share of promoted teams
   actually got relegated, and what did the teams that went down look like
   (points, goal difference)?
2. Model prediction: run our pre-season simulation (current priors) and read
   off Hull / Ipswich / Coventry's relegation %.
3. Compare — do our numbers sit sensibly around the historical base rate?
"""

import math
import random
import re
from pathlib import Path

from championship_scraper import fetch_csv, parse_matches

ROOT = Path(__file__).parent

# champ season → following PL season
COHORTS = [("2122", "2223"), ("2223", "2324"), ("2324", "2425"), ("2425", "2526")]

LEAGUE_AVG = 1.40
HOME_ATK, AWAY_ATK = 1.11, 0.89
SIGMA_FORM, SIGMA_IND = 0.13, 0.08
PROMOTED_IDS = {"hull", "ipswich", "coventry"}


# ── historical ─────────────────────────────────────────────────────────────
def _final_table(matches):
    pts, gf, ga = {}, {}, {}
    for m in matches:
        for t in (m["home"], m["away"]):
            pts.setdefault(t, 0); gf.setdefault(t, 0); ga.setdefault(t, 0)
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    order = sorted(pts, key=lambda t: (-pts[t], -(gf[t] - ga[t])))
    return order, pts, gf, ga


def historical():
    print("══ Historical promoted-team outcomes ══\n")
    promoted_total = promoted_rel = 0
    rel_points, rel_gd = [], []
    for champ_code, pl_code in COHORTS:
        champ_teams = {m["home"] for m in parse_matches(fetch_csv(champ_code, "E1"))}
        pl = parse_matches(fetch_csv(pl_code, "E0"))
        order, pts, gf, ga = _final_table(pl)
        relegated = set(order[-3:])
        promoted = sorted(champ_teams & set(order))
        rel_prom = [t for t in promoted if t in relegated]
        promoted_total += len(promoted); promoted_rel += len(rel_prom)
        for t in relegated:
            rel_points.append(pts[t]); rel_gd.append(gf[t] - ga[t])
        print(f"{pl_code[:2]}/{pl_code[2:]}: promoted {promoted}")
        print(f"        relegated {sorted(relegated)}  "
              f"→ {len(rel_prom)}/{len(promoted)} promoted went down "
              f"({', '.join(rel_prom) or 'none'})")
    rate = promoted_rel / promoted_total * 100
    print(f"\nPromoted-team relegation base rate: {promoted_rel}/{promoted_total} "
          f"= {rate:.0f}%")
    print(f"Relegated teams averaged {sum(rel_points)/len(rel_points):.0f} pts, "
          f"GD {sum(rel_gd)/len(rel_gd):+.0f}")
    return rate, sum(rel_points) / len(rel_points)


# ── model pre-season simulation ─────────────────────────────────────────────
def _priors_from_index():
    s = (ROOT / "index.html").read_text()
    out = {}
    for m in re.finditer(r"id: '([a-z]+)',.*?baseAtk: ([0-9.]+), baseDef: ([0-9.]+)", s):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def _poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1; p *= random.random()
        if p <= L:
            return k - 1


def simulate(priors, sims=8000):
    teams = list(priors)
    fixtures = [(h, a) for h in teams for a in teams if h != a]  # double round-robin
    rel = {t: 0 for t in teams}
    pts_sum = {t: 0.0 for t in teams}
    for _ in range(sims):
        r = {}
        for t in teams:
            mult = 1.6 if t in PROMOTED_IDS else 1.0
            form = math.exp(random.gauss(0, SIGMA_FORM * mult))
            atk, dfn = priors[t]
            r[t] = (atk * form * math.exp(random.gauss(0, SIGMA_IND * mult)),
                    dfn / form * math.exp(random.gauss(0, SIGMA_IND * mult)))
        pts, gf, ga = {t: 0 for t in teams}, {t: 0 for t in teams}, {t: 0 for t in teams}
        for h, a in fixtures:
            ha, hd = r[h]; aa, ad = r[a]
            hg = _poisson(max(0.05, (ha / LEAGUE_AVG) * ad * HOME_ATK))
            ag = _poisson(max(0.05, (aa / LEAGUE_AVG) * hd * AWAY_ATK))
            gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
            if hg > ag: pts[h] += 3
            elif ag > hg: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
        order = sorted(teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), random.random()))
        for t in teams: pts_sum[t] += pts[t]
        for t in order[-3:]: rel[t] += 1
    return {t: (rel[t] / sims * 100, pts_sum[t] / sims) for t in teams}


def main():
    base_rate, rel_pts = historical()

    print("\n══ Model pre-season prediction (current priors) ══\n")
    res = simulate(_priors_from_index())
    ranked = sorted(res, key=lambda t: -res[t][0])
    print(f"{'team':<14}{'rel%':>7}{'xPts':>7}")
    for t in ranked[:8]:
        star = " ◄ promoted" if t in PROMOTED_IDS else ""
        print(f"{t:<14}{res[t][0]:>6.1f}{res[t][1]:>7.0f}{star}")

    prom = {t: res[t] for t in PROMOTED_IDS}
    avg_prom_rel = sum(v[0] for v in prom.values()) / 3
    print(f"\n══ Verdict ══")
    print(f"Historical promoted base rate : {base_rate:.0f}%")
    print(f"Model avg for our 3 promoted  : {avg_prom_rel:.0f}%  "
          f"(hull {prom['hull'][0]:.0f}%, ipswich {prom['ipswich'][0]:.0f}%, "
          f"coventry {prom['coventry'][0]:.0f}%)")
    print(f"Relegated teams historically averaged ~{rel_pts:.0f} pts; "
          f"our promoted project to {prom['hull'][1]:.0f}/{prom['ipswich'][1]:.0f}/"
          f"{prom['coventry'][1]:.0f} pts (hull/ips/cov).")


if __name__ == "__main__":
    main()
