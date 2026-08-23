"""
promoted_analyzer.py — derive data-driven Premier League priors
(baseAtk/baseDef) for the promoted clubs (Hull, Ipswich, Coventry) by
scaling their previous Championship season up to PL level.

The scaling metric (the crux)
─────────────────────────────
Raw goal tallies translate badly: their absolute level is dominated by the
league gap and they swallow finishing luck. So instead we scale a team's
strength RELATIVE TO ITS OWN LEAGUE, measured on UNDERLYING xG rather than
goals:

    champ_attack_rating  = team xGF/game ÷ Championship mean xGF/game   (1.0 = avg)
    champ_defence_rating = team xGA/game ÷ Championship mean xGA/game   (1.0 = avg)

Why this form:
  • Relative-to-league ratings are league-neutral and far less noisy than
    raw rates, and the unknown xG conversion constant cancels in the ratio.
  • Using xG (from championship_scraper's shots-on-target model) means a
    team is judged on chances created/allowed, not on a goal count it may
    not repeat — this is how the system absorbs Hull's strange 2025/26
    (out-shot by opponents, yet 6th: results ≫ process).

The league-strength transform itself is fitted empirically from every team
that bridged the divide — promoted from the Championship one year, in the
PL the next — by regressing their PL rating on their prior Championship
rating:

    PL_rating = slope · Championship_rating + intercept

The intercept captures the demotion (an average Championship side is well
below PL average); the slope captures compression (rating gaps shrink a
division up). A projected prior is then:

    baseAtk = (slope_A · champ_attack_rating  + intercept_A) · PL_AVG
    baseDef = (slope_D · champ_defence_rating + intercept_D) · PL_AVG

Nothing is written automatically; the script prints recommendations and the
goals-vs-xG contrast that drives Hull's number.
"""

import sys

from championship_scraper import (fetch_csv, parse_matches, fit_xg,
                                   team_table, half_matches)

# (Championship season, following PL season) — each pair bridges the divide.
COHORTS = [("2122", "2223"), ("2223", "2324"),
           ("2324", "2425"), ("2425", "2526")]
TARGET_CHAMP = "2526"
TARGET_TEAMS = {"Hull": "hull", "Ipswich": "ipswich", "Coventry": "coventry"}

PL_AVG = 1.40   # model's league-average goals per team per game (index.html)

CURRENT = {"hull": (1.05, 2.10), "ipswich": (1.06, 1.98), "coventry": (1.07, 1.97)}
ATK_RANGE = (0.60, 1.50)
DEF_RANGE = (1.20, 2.20)


# ── small stats helpers ────────────────────────────────────────────────────
def linreg(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return b, a, (1 - ssr / sst if sst else 0.0)


def clamp(x, rng):
    return max(rng[0], min(rng[1], x))


# ── league-relative ratings ────────────────────────────────────────────────
def league_ratings(code, league, a, b, metric, half="full"):
    """team → (rel_atk, rel_def) where 1.0 = league-average; plus the table.
    `half` restricts to a window of the season ('2nd' = recent form)."""
    matches = half_matches(parse_matches(fetch_csv(code, league)), half)
    table = team_table(matches, a, b)
    fa = "xgf" if metric == "xg" else "gf"
    fd = "xga" if metric == "xg" else "ga"
    atk = {t: s[fa] / s["gp"] for t, s in table.items()}
    dfn = {t: s[fd] / s["gp"] for t, s in table.items()}
    ma = sum(atk.values()) / len(atk)
    md = sum(dfn.values()) / len(dfn)
    return ({t: atk[t] / ma for t in atk},
            {t: dfn[t] / md for t in dfn}, table)


# ── league-strength scaler ─────────────────────────────────────────────────
def fit_scaler(a, b, metric, champ_half="2nd"):
    """Regress full-season PL relative rating on the promoted side's
    Championship relative rating (over `champ_half`) for every bridge team.
    Predicting a full PL season from recent (2nd-half) Championship form.
    Returns (atk_reg, def_reg, observations)."""
    cx_a, py_a, cx_d, py_d, obs = [], [], [], [], []
    for champ_code, pl_code in COHORTS:
        cra, crd, ctab = league_ratings(champ_code, "E1", a, b, metric, half=champ_half)
        pra, prd, ptab = league_ratings(pl_code, "E0", a, b, metric, half="full")
        for team in sorted(set(ctab) & set(ptab)):
            # champ window may be a half-season (~23 games); PL is full (~38)
            if ctab[team]["gp"] < 15 or ptab[team]["gp"] < 20:
                continue
            cx_a.append(cra[team]); py_a.append(pra[team])
            cx_d.append(crd[team]); py_d.append(prd[team])
            obs.append((team, f"{champ_code}->{pl_code}",
                        cra[team], pra[team], crd[team], prd[team]))
    return linreg(cx_a, py_a), linreg(cx_d, py_d), obs


def project(rel, reg, rng):
    slope, intercept, _ = reg
    return clamp((slope * rel + intercept) * PL_AVG, rng)


# ── pooled, league-neutral xG model ────────────────────────────────────────
def pooled_xg():
    seen, allm = set(), []
    pairs = {("E1", c) for c, _ in COHORTS} | {("E0", p) for _, p in COHORTS}
    pairs.add(("E1", TARGET_CHAMP))
    for league, code in pairs:
        if (league, code) in seen:
            continue
        seen.add((league, code))
        allm += parse_matches(fetch_csv(code, league))
    return fit_xg(allm)


# ── report ──────────────────────────────────────────────────────────────────
def recommend(half="2nd"):
    """Return {tid: (baseAtk, baseDef)} from xG ratings scaled to the PL,
    using the given Championship window. Also returns supporting context."""
    a, b = pooled_xg()
    atk_x, def_x, obs = fit_scaler(a, b, "xg", champ_half=half)
    xa, xd, xtab = league_ratings(TARGET_CHAMP, "E1", a, b, "xg", half=half)
    priors = {}
    for name, tid in TARGET_TEAMS.items():
        priors[tid] = (round(project(xa[name], atk_x, ATK_RANGE), 2),
                       round(project(xd[name], def_x, DEF_RANGE), 2))
    return priors, (atk_x, def_x, obs, xa, xd, xtab, a, b)


def main():
    a, b = pooled_xg()
    print(f"\nPooled xG model: {a:.3f}·SoT + {b:.3f}·offTarget per shot.")

    # Is recent (2nd-half) Championship form a better PL predictor than the
    # full season? Compare the scaler fit on each.
    full_a, full_d, _ = fit_scaler(a, b, "xg", champ_half="full")
    half_a, half_d, obs = fit_scaler(a, b, "xg", champ_half="2nd")
    print("\n══ Does 2nd-half Championship form predict the PL better? (R²) ══")
    print(f"              {'attack R²':>11}{'defence R²':>12}")
    print(f"full season   {full_a[2]:>11.2f}{full_d[2]:>12.2f}")
    print(f"2nd half  ★   {half_a[2]:>11.2f}{half_d[2]:>12.2f}")

    # Recommendation built on 2nd-half, process-based (xG) ratings.
    priors, (atk_x, def_x, _o, xa, xd, xtab, _a, _b) = recommend("2nd")
    full_priors, _ = recommend("full")

    print(f"\n══ 2nd-half Championship xG ratings (1.00 = league avg) ══")
    print(f"{'team':<10}{'atk':>7}{'def':>7}")
    for name in TARGET_TEAMS:
        print(f"{name:<10}{xa[name]:>7.2f}{xd[name]:>7.2f}")

    print(f"\n══ Recommended 2026/27 priors  (baseAtk / baseDef) ══")
    print(f"{'team':<10}{'current':>14}{'full-szn xG':>14}{'2nd-half xG ★':>15}")
    for name, tid in TARGET_TEAMS.items():
        c, f, h = CURRENT[tid], full_priors[tid], priors[tid]
        print(f"{name:<10}{c[0]:>5.2f}/{c[1]:<7.2f}{f[0]:>5.2f}/{f[1]:<7.2f}"
              f"{h[0]:>6.2f}/{h[1]:<8.2f}")

    print("\n══ Hull spotlight (2nd half) ══")
    h = "Hull"
    over = (xtab[h]['gf'] - xtab[h]['xgf']) / xtab[h]['gp']
    print(f"  2nd-half xG rating: attack {xa[h]:.2f}× / leak {xd[h]:.2f}× of league avg; "
          f"out-finished xG by +{over:.2f} goals/g.")
    print(f"  Process-based prior: {priors['hull'][0]:.2f}/{priors['hull'][1]:.2f} "
          f"— still the weakest of the three (finishing excluded by construction).")
    print("\nThese 2nd-half xG priors are the recommendation. Apply to index.html "
          "TEAMS (+ optional Hull style-nudge on top).\n")


if __name__ == "__main__":
    main()
