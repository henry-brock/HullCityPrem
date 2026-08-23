"""
backtest.py — walk-forward validation of the match model against the
completed 2025/26 season (cache/2526.json: 272 real PL matches among the
17 teams that carry into 2026/27, each pair home + away, with bookmaker
odds attached).

What it measures
────────────────
For every match, in chronological order, the model predicts P(home win) /
P(draw) / P(away win) using ONLY the matches played before it, then the
real result is scored with:

  • log-loss  (−log of the probability assigned to the actual outcome)
  • RPS       (ranked probability score — rewards being "close" on the
              ordered H/D/A scale)

The same two metrics are computed for the bookmaker's average odds, which
acts as a near-optimal reference: a model that approaches the market's
log-loss is about as sharp as publicly available information allows.

It then:
  1. sweeps home advantage × Dixon-Coles ρ to find the log-loss minimum,
  2. ablates opponent-adjustment and Dixon-Coles to quantify each gain,
  3. calibrates the season-level noise (correlated log-normal) so the
     simulated final-points intervals actually cover reality.

The model formulas here mirror index.html exactly; tuned values found by
this script are copied back into the MODEL config there.
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache" / "2526.json"

LEAGUE_AVG = 1.40
BLEND_CAP  = 0.65    # tuned (tune.py — maximises on-field skill)
BLEND_GAMES = 12     # tuned
RECENCY_MAX = 2      # newest game weight (oldest = 1) — tuned
MAXG       = 10      # score-matrix truncation

# Pre-season priors — MUST mirror the TEAMS array in index.html (atk, def).
PRIORS = {
    'arsenal':      (1.76, 0.69),
    'astonvilla':   (1.4, 1.36),
    'bournemouth':  (1.51, 1.43),
    'brentford':    (1.33, 1.64),
    'brighton':     (1.54, 1.28),
    'chelsea':      (1.61, 1.27),
    'crystalpalace':(1.31, 1.32),
    'everton':      (1.16, 1.46),
    'fulham':       (1.36, 1.32),
    'leeds':        (1.27, 1.32),
    'liverpool':    (2.04, 1.1),
    'mancity':      (1.97, 0.88),
    'manutd':       (1.6, 1.39),
    'newcastle':    (1.5, 1.4),
    'nottmforest':  (1.23, 1.45),
    'sunderland':   (0.99, 1.52),
    'tottenham':    (1.45, 1.43),
}

# Mean priors of the team set — the opponent-strength normaliser, so an
# exactly average opponent yields an adjustment factor of 1.0 (no bias).
MEAN_ATK = sum(a for a, _ in PRIORS.values()) / len(PRIORS)
MEAN_DEF = sum(d for _, d in PRIORS.values()) / len(PRIORS)


# ── helpers ──────────────────────────────────────────────────────────────
def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _outcome(hg, ag):
    return 'H' if hg > ag else 'A' if ag > hg else 'D'


# ── model ────────────────────────────────────────────────────────────────
def get_ratings(team, history, opp_adjust):
    """Blended (atk, def) from prior matches — mirrors getTeamRatings()."""
    games = [m for m in history if m['homeId'] == team or m['awayId'] == team]
    n = len(games)
    base_atk, base_def = PRIORS[team]
    if n < 3:
        return base_atk, base_def

    atk_sum = def_sum = w_tot = 0.0
    for i, m in enumerate(games):
        home     = m['homeId'] == team
        scored   = m['homeGoals'] if home else m['awayGoals']
        conceded = m['awayGoals'] if home else m['homeGoals']
        if opp_adjust:
            opp = m['awayId'] if home else m['homeId']
            opp_atk, opp_def = PRIORS[opp]
            # Goals vs a strong defence count for more; goals conceded
            # against a strong attack are partly excused. Normalised by the
            # league-mean prior so an average opponent gives factor 1.0.
            scored   *= _clamp(MEAN_DEF / opp_def, 0.6, 1.7)
            conceded *= _clamp(MEAN_ATK / opp_atk, 0.6, 1.7)
        w = 1 + (i / (n - 1)) * (RECENCY_MAX - 1)
        atk_sum += scored   * w
        def_sum += conceded * w
        w_tot   += w

    actual_atk = atk_sum / w_tot
    actual_def = def_sum / w_tot
    blend = min(n / BLEND_GAMES, BLEND_CAP)
    return (base_atk * (1 - blend) + actual_atk * blend,
            base_def * (1 - blend) + actual_def * blend)


def score_matrix(hxg, axg, rho):
    """Dixon-Coles-adjusted joint score distribution, normalised."""
    ph = [_pmf(i, hxg) for i in range(MAXG + 1)]
    pa = [_pmf(j, axg) for j in range(MAXG + 1)]
    m = [[ph[i] * pa[j] for j in range(MAXG + 1)] for i in range(MAXG + 1)]
    if rho:
        m[0][0] *= max(0.0, 1 - hxg * axg * rho)
        m[0][1] *= max(0.0, 1 + hxg * rho)
        m[1][0] *= max(0.0, 1 + axg * rho)
        m[1][1] *= max(0.0, 1 - rho)
    s = sum(sum(row) for row in m)
    return [[c / s for c in row] for row in m]


def match_probs(home, away, history, cfg):
    h_atk, h_def = get_ratings(home, history, cfg['opp_adjust'])
    a_atk, a_def = get_ratings(away, history, cfg['opp_adjust'])
    hxg = max(0.05, (h_atk / LEAGUE_AVG) * a_def * cfg['home_adv'])
    axg = max(0.05, (a_atk / LEAGUE_AVG) * h_def)
    m = score_matrix(hxg, axg, cfg['rho'])
    hw = dr = 0.0
    for i in range(MAXG + 1):
        for j in range(MAXG + 1):
            if i > j:   hw += m[i][j]
            elif i == j: dr += m[i][j]
    return hw, dr, 1 - hw - dr


# ── scoring ──────────────────────────────────────────────────────────────
def _logloss(p):
    return -math.log(max(p, 1e-12))


def _rps(ph, pd, pa, outcome):
    pred = [ph, pd, pa]
    obs  = {'H': [1, 0, 0], 'D': [0, 1, 0], 'A': [0, 0, 1]}[outcome]
    c_pred = c_obs = 0.0
    total = 0.0
    for k in range(2):           # cumulative over first 2 of 3 categories
        c_pred += pred[k]
        c_obs  += obs[k]
        total  += (c_pred - c_obs) ** 2
    return total / 2.0


def walk_forward(matches, cfg):
    history, ll, rps, n = [], 0.0, 0.0, 0
    for m in matches:
        ph, pd, pa = match_probs(m['homeId'], m['awayId'], history, cfg)
        out = _outcome(m['homeGoals'], m['awayGoals'])
        p_actual = {'H': ph, 'D': pd, 'A': pa}[out]
        ll  += _logloss(p_actual)
        rps += _rps(ph, pd, pa, out)
        n   += 1
        history.append(m)
    return ll / n, rps / n


def bookmaker_baseline(matches):
    ll = rps = 0.0
    n = 0
    for m in matches:
        o = (m.get('odds') or {}).get('avg') or (m.get('odds') or {}).get('b365')
        if not o:
            continue
        inv = [1 / o['h'], 1 / o['d'], 1 / o['a']]
        s = sum(inv)
        ph, pd, pa = (x / s for x in inv)
        out = _outcome(m['homeGoals'], m['awayGoals'])
        ll  += _logloss({'H': ph, 'D': pd, 'A': pa}[out])
        rps += _rps(ph, pd, pa, out)
        n   += 1
    return (ll / n, rps / n) if n else (float('nan'), float('nan'))


# ── season-noise calibration ─────────────────────────────────────────────
def _poisson(lam):
    """Knuth sampler — fine for the small λ in football."""
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def calibrate_noise(matches, sigma_form, sigma_ind, sims=2000):
    """
    Simulate the 272-match schedule `sims` times from pre-season priors with
    correlated log-normal team noise, and report what fraction of teams' real
    final points land inside the simulated 5–95% interval (target ≈ 0.90).
    """
    teams = sorted(PRIORS)
    fixtures = [(m['homeId'], m['awayId']) for m in matches]

    # actual points among these 272 matches
    actual = {t: 0 for t in teams}
    for m in matches:
        out = _outcome(m['homeGoals'], m['awayGoals'])
        if out == 'H':   actual[m['homeId']] += 3
        elif out == 'A': actual[m['awayId']] += 3
        else:            actual[m['homeId']] += 1; actual[m['awayId']] += 1

    dist = {t: [] for t in teams}
    for _ in range(sims):
        r = {}
        for t in teams:
            ba, bd = PRIORS[t]
            form = math.exp(random.gauss(0, sigma_form))
            r[t] = (ba * form * math.exp(random.gauss(0, sigma_ind)),
                    bd / form * math.exp(random.gauss(0, sigma_ind)))
        pts = {t: 0 for t in teams}
        for home, away in fixtures:
            ha, hd = r[home]
            aa, ad = r[away]
            hg = _poisson(max(0.05, (ha / LEAGUE_AVG) * ad * 1.12))
            ag = _poisson(max(0.05, (aa / LEAGUE_AVG) * hd))
            if hg > ag:   pts[home] += 3
            elif ag > hg: pts[away] += 3
            else:         pts[home] += 1; pts[away] += 1
        for t in teams:
            dist[t].append(pts[t])

    covered = 0
    spread_sum = 0.0
    for t in teams:
        s = sorted(dist[t])
        lo = s[int(0.05 * sims)]
        hi = s[int(0.95 * sims)]
        spread_sum += hi - lo
        if lo <= actual[t] <= hi:
            covered += 1
    return covered / len(teams), spread_sum / len(teams)


# ── main ─────────────────────────────────────────────────────────────────
def _load_matches():
    data = json.loads(CACHE.read_text())
    matches = data['matches']

    def key(m):
        try:
            return datetime.strptime(f"{m['date']} {m.get('time','00:00')}",
                                     "%d/%m/%Y %H:%M")
        except ValueError:
            return datetime.strptime(m['date'], "%d/%m/%Y")

    return sorted(matches, key=key)


def main():
    matches = _load_matches()
    print(f"Loaded {len(matches)} matches from 2025/26 "
          f"(season {json.loads(CACHE.read_text()).get('season')})\n")

    # On-field reference points (the betting market is NOT a target — see
    # `bench` / `--bench` for the floor↔oracle skill scale).
    floor_ll = skill_floor(matches)[0]
    print("── On-field reference ────────────────────────")
    print(f"Skill floor (base rates)  log-loss {floor_ll:.4f}")
    base_ll, base_rps = bookmaker_baseline(matches)
    print(f"Bookmaker (external only) log-loss {base_ll:.4f}  ← reference, not a goal")

    cur = walk_forward(matches, {'home_adv': 1.12, 'rho': 0.0, 'opp_adjust': False})
    print(f"Pre-tuning model          log-loss {cur[0]:.4f}   RPS {cur[1]:.4f}\n")

    # ── sweep home advantage × Dixon-Coles ρ × opponent-adjust ──
    print("── Sweep: home_adv × Dixon-Coles ρ × opp-adjust ──")
    best = None
    for opp in (False, True):
        for ha in [1.05, 1.075, 1.10, 1.125, 1.15, 1.175, 1.20]:
            for rho in [-0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.03]:
                ll, rps = walk_forward(
                    matches, {'home_adv': ha, 'rho': rho, 'opp_adjust': opp})
                if best is None or ll < best[0]:
                    best = (ll, rps, ha, rho, opp)
    bll, brps, bha, brho, bopp = best
    print(f"Best: home_adv {bha}   ρ {brho:+.2f}   opp-adjust {bopp}   "
          f"log-loss {bll:.4f}   RPS {brps:.4f}\n")

    # ── ablations at the best config ──
    print("── Ablations (at best home_adv) ──")
    full   = walk_forward(matches, {'home_adv': bha, 'rho': brho, 'opp_adjust': bopp})
    no_dc  = walk_forward(matches, {'home_adv': bha, 'rho': 0.0,  'opp_adjust': bopp})
    flip   = walk_forward(matches, {'home_adv': bha, 'rho': brho, 'opp_adjust': not bopp})
    print(f"full model             log-loss {full[0]:.4f}   RPS {full[1]:.4f}")
    print(f"  − Dixon-Coles        log-loss {no_dc[0]:.4f}   RPS {no_dc[1]:.4f}"
          f"   (DC gain {no_dc[0]-full[0]:+.4f})")
    print(f"  opp-adjust {str(not bopp):5}       log-loss {flip[0]:.4f}   RPS {flip[1]:.4f}"
          f"   (opp gain {flip[0]-full[0]:+.4f})")
    print(f"  vs pre-tuning model  Δlog-loss {full[0]-cur[0]:+.4f}"
          f"   ({(cur[0]-full[0])/cur[0]*100:+.1f}% better)")
    print(f"  skill vs floor       {(floor_ll-full[0])/floor_ll*100:+.1f}% below "
          f"no-skill baseline (run --bench for floor↔oracle skill %)\n")

    # ── season-noise calibration ──
    print("── Season-noise calibration (target coverage ≈ 0.90) ──")
    # contrast with the old uniform ±35% noise, expressed as ~equivalent σ
    for sf, si in [(0.0, 0.0), (0.08, 0.05), (0.12, 0.08),
                   (0.16, 0.10), (0.20, 0.12)]:
        cov, spread = calibrate_noise(matches, sf, si, sims=1500)
        tag = "  (no noise)" if sf == 0 else ""
        print(f"σ_form {sf:.2f}  σ_ind {si:.2f}   "
              f"coverage {cov:.2f}   mean 90% width {spread:4.1f} pts{tag}")
    print("\n(Old model used uniform ±35% on atk AND def independently — "
          "far wider, flat-topped, and uncorrelated.)")


# ── full-season PL backtest (ESPN 380 games) ──────────────────────────────
# A complete, unfiltered validation set (the 272-match set above only covers
# the 17 retained teams). Neutral priors for every team; the recency blend
# learns ratings during the season, so this tests the match MODEL and lets us
# measure the team-specific home/away split (#1).
ESPN_GAMES = ROOT / "cache" / "espn_games_eng.1_2526.json"


def _espn_matches():
    g = json.loads(ESPN_GAMES.read_text())["games"]
    out = []
    for r in g.values():
        ts, sc = r.get("teamStats") or {}, r.get("score") or {}
        h = (ts.get("home") or {}).get("team")
        a = (ts.get("away") or {}).get("team")
        try:
            hg, ag = int(sc.get("home")), int(sc.get("away"))
        except (TypeError, ValueError):
            continue
        if not h or not a:
            continue
        out.append({"homeId": h, "awayId": a, "homeGoals": hg,
                    "awayGoals": ag, "date": r.get("date", "")})
    out.sort(key=lambda m: m["date"])
    return out


def _venue_ratings(team, history):
    """Neutral-prior blend, split into overall + home/away multipliers.
    Returns (atk, def, homeAtkMult, homeDefMult, awayAtkMult, awayDefMult)."""
    games = [m for m in history if team in (m["homeId"], m["awayId"])]
    n = len(games)
    if n < 3:
        return LEAGUE_AVG, LEAGUE_AVG, 1.0, 1.0, 1.0, 1.0
    atk = dfn = w_tot = 0.0
    h_gf = h_ga = h_n = a_gf = a_ga = a_n = 0.0
    for i, m in enumerate(games):
        home = m["homeId"] == team
        gf = m["homeGoals"] if home else m["awayGoals"]
        ga = m["awayGoals"] if home else m["homeGoals"]
        w = 1 + (i / (n - 1)) * (RECENCY_MAX - 1)
        atk += gf * w; dfn += ga * w; w_tot += w
        if home: h_gf += gf; h_ga += ga; h_n += 1
        else:    a_gf += gf; a_ga += ga; a_n += 1
    aa, ad = atk / w_tot, dfn / w_tot
    blend = min(n / BLEND_GAMES, BLEND_CAP)
    atkR = LEAGUE_AVG * (1 - blend) + aa * blend
    defR = LEAGUE_AVG * (1 - blend) + ad * blend
    # Home/away multipliers vs the team's own overall rate, regularised toward
    # 1.0 by a pseudo-count so small samples don't swing wildly.
    ovf = (h_gf + a_gf) / n or 1e-9
    ova = (h_ga + a_ga) / n or 1e-9
    K = 6
    def mult(s, cnt, ov):
        if cnt == 0: return 1.0
        raw = (s / cnt) / ov
        return (raw * cnt + 1.0 * K) / (cnt + K)
    return (atkR, defR,
            mult(h_gf, h_n, ovf), mult(h_ga, h_n, ova),
            mult(a_gf, a_n, ovf), mult(a_ga, a_n, ova))


def walk_forward_espn(cfg):
    """cfg: home_adv, rho, home_away(bool). Returns (logloss, rps, n)."""
    matches = _espn_matches()
    hist, ll, rps, n = [], 0.0, 0.0, 0
    for m in matches:
        ha_, hd_, hHa, hHd, _, _ = _venue_ratings(m["homeId"], hist)
        aa_, ad_, _, _, aAa, aAd = _venue_ratings(m["awayId"], hist)
        if cfg.get("shrink") is not None:
            # blend league-average split (κ=0) ↔ team-specific (κ=1)
            k = cfg["shrink"]
            hxg = (1-k)*((ha_/LEAGUE_AVG)*ad_*1.11) + k*((ha_*hHa/LEAGUE_AVG)*(ad_*aAd))
            axg = (1-k)*((aa_/LEAGUE_AVG)*hd_*0.89) + k*((aa_*aAa/LEAGUE_AVG)*(hd_*hHd))
        elif cfg.get("home_away"):
            hxg = (ha_ * hHa / LEAGUE_AVG) * (ad_ * aAd)
            axg = (aa_ * aAa / LEAGUE_AVG) * (hd_ * hHd)
        elif cfg.get("league_ha"):
            hxg = (ha_ / LEAGUE_AVG) * ad_ * cfg["ha"]
            axg = (aa_ / LEAGUE_AVG) * hd_ * cfg["aa"]
        else:
            hxg = (ha_ / LEAGUE_AVG) * ad_ * cfg["home_adv"]
            axg = (aa_ / LEAGUE_AVG) * hd_
        mat = score_matrix(max(0.05, hxg), max(0.05, axg), cfg["rho"])
        hw = sum(mat[i][j] for i in range(MAXG + 1) for j in range(MAXG + 1) if i > j)
        dr = sum(mat[i][i] for i in range(MAXG + 1))
        out = _outcome(m["homeGoals"], m["awayGoals"])
        p = {"H": hw, "D": dr, "A": 1 - hw - dr}[out]
        ll += _logloss(p)
        rps += _rps(hw, dr, 1 - hw - dr, out)
        n += 1
        hist.append(m)
    return ll / n, rps / n, n


# ── on-field benchmarks (replace the betting-market yardstick) ─────────────
def skill_floor(matches):
    """No-skill baseline: predict every match with the league base H/D/A rate
    (computed from the matches themselves). Pure on-field, zero team knowledge.
    The model must clear this to show any skill at all."""
    from collections import Counter
    c = Counter(_outcome(m["homeGoals"], m["awayGoals"]) for m in matches)
    n = len(matches)
    p = {k: c[k] / n for k in "HDA"}
    ll = sum(_logloss(p[_outcome(m["homeGoals"], m["awayGoals"])]) for m in matches) / n
    return ll, p


def oracle(matches, rho=-0.13):
    """On-field skill ceiling: fit opponent-adjusted attack/defence to the WHOLE
    season (perfect hindsight on team strength), then score each match. This is
    the best this model class can do from on-field data — the real target,
    not the market. Its residual gap to 0 is irreducible match randomness."""
    import ratings
    games = [({"home": m["homeId"], "away": m["awayId"],
               "hg": m["homeGoals"], "ag": m["awayGoals"]}, 1.0) for m in matches]
    atk, dfn, mu, H, A = ratings.solve(games)
    ll = n = 0
    for m in matches:
        h, a = m["homeId"], m["awayId"]
        hxg = max(0.05, atk[h] * dfn[a] * H * mu)
        axg = max(0.05, atk[a] * dfn[h] * A * mu)
        mat = score_matrix(hxg, axg, rho)
        hw = sum(mat[i][j] for i in range(MAXG + 1) for j in range(MAXG + 1) if i > j)
        dr = sum(mat[i][i] for i in range(MAXG + 1))
        out = _outcome(m["homeGoals"], m["awayGoals"])
        ll += _logloss({"H": hw, "D": dr, "A": 1 - hw - dr}[out]); n += 1
    return ll / n


def bench():
    """Where the model sits between the on-field floor and ceiling."""
    matches = _espn_matches()
    floor, base = skill_floor(matches)
    model, _, _ = walk_forward_espn({"league_ha": True, "ha": 1.11, "aa": 0.89, "rho": -0.13}), None, None
    model = model[0]
    ceil = oracle(matches)
    skill = (floor - model) / (floor - ceil) * 100
    print("On-field benchmarks (380-game PL season)\n")
    print(f"  base rate H/D/A           {base['H']:.0%}/{base['D']:.0%}/{base['A']:.0%}")
    print(f"  skill FLOOR (no team info)  log-loss {floor:.4f}")
    print(f"  our model (walk-forward)    log-loss {model:.4f}")
    print(f"  on-field CEILING (oracle)   log-loss {ceil:.4f}")
    print(f"\n  skill captured = (floor−model)/(floor−ceiling) = {skill:.1f}%")
    print(f"  remaining learnable gap to ceiling: {model-ceil:.4f}")
    print("  (ceiling's gap to 0 is irreducible football randomness — "
          "unbeatable from on-field data.)")


def main_espn():
    matches = _espn_matches()
    print(f"Full-season PL backtest — {len(matches)} ESPN games, "
          f"neutral priors, walk-forward.\n")

    # League-average home/away factors (robust, not per-team)
    hg = sum(m["homeGoals"] for m in matches) / len(matches)
    ag = sum(m["awayGoals"] for m in matches) / len(matches)
    lg = (hg + ag) / 2
    ha_f, aa_f = hg / lg, ag / lg
    print(f"League home goals/g {hg:.2f}  away {ag:.2f}  "
          f"→ home ×{ha_f:.3f} / away ×{aa_f:.3f}\n")

    flat   = walk_forward_espn({"home_adv": 1.16, "rho": -0.10})
    league = walk_forward_espn({"league_ha": True, "ha": ha_f, "aa": aa_f, "rho": -0.10})
    team   = walk_forward_espn({"home_away": True, "rho": -0.10})
    print(f"{'model':<30}{'log-loss':>10}{'RPS':>9}")
    print(f"{'flat HOME_ADV 1.16 (current)':<30}{flat[0]:>10.4f}{flat[1]:>9.4f}")
    print(f"{'league home/away split':<30}{league[0]:>10.4f}{league[1]:>9.4f}"
          f"   ({flat[0]-league[0]:+.4f})")
    print(f"{'team-specific home/away':<30}{team[0]:>10.4f}{team[1]:>9.4f}"
          f"   ({flat[0]-team[0]:+.4f})")
    best = min([("flat", flat[0]), ("league split", league[0]),
                ("team-specific", team[0])], key=lambda x: x[1])
    print(f"\nBest: {best[0]} ({best[1]:.4f}). (+ = better than current flat)")


# ── table-level simulation backtest (#3) ──────────────────────────────────
# Tests the SEASON simulation, not just single matches: at mid-season
# checkpoints take the real results so far, learn ratings, simulate the rest
# many times, and score the predicted FINAL TABLE against what actually
# happened (position ranked-probability score + points-interval coverage).
def _table_order(results, teams):
    pts = {t: 0 for t in teams}; gf = {t: 0 for t in teams}; ga = {t: 0 for t in teams}
    for m in results:
        h, a, hg, ag = m["homeId"], m["awayId"], m["homeGoals"], m["awayGoals"]
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:   pts[h] += 3
        elif ag > hg: pts[a] += 3
        else:         pts[h] += 1; pts[a] += 1
    order = sorted(teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t], random.random()))
    return {t: i + 1 for i, t in enumerate(order)}, pts


def _pos_rps(dist, actual):
    """Ranked probability score between a predicted position distribution
    (list over 1..20) and the actual position. Lower = sharper & righter."""
    cp = total = 0.0
    for k in range(1, 20):                 # cumulative over first 19 of 20
        cp += dist[k]
        co = 1.0 if actual <= k else 0.0   # observed CDF (one-hot), not a sum
        total += (cp - co) ** 2
    return total / 19


def backtest_table(checkpoints=(100, 190, 280), sims=800, sf=0.13, si=0.08,
                   hi=1.3, lo=0.85):
    # Dynamic σ (#1): σ scales by the fraction of the season still unplayed —
    # wide when little is known, tight late on. hi=lo=1 → constant (old).
    matches = _espn_matches()
    teams = sorted({m["homeId"] for m in matches} | {m["awayId"] for m in matches})
    actual_pos, actual_pts = _table_order(matches, teams)

    print(f"Table-level backtest — {len(matches)} games, {len(teams)} teams, "
          f"{sims} sims/checkpoint.\n")
    print(f"{'checkpoint':<14}{'sim posRPS':>12}{'freeze posRPS':>15}{'pts cover':>11}")

    for cut in checkpoints:
        played, remaining = matches[:cut], matches[cut:]
        ratings = {t: _venue_ratings(t, played)[:2] for t in teams}
        frozen, _ = _table_order(played, teams)
        rem_frac = len(remaining) / len(matches)
        scale = lo + (hi - lo) * rem_frac
        sf_e, si_e = sf * scale, si * scale

        pos_count = {t: [0] * 21 for t in teams}
        pts_samples = {t: [] for t in teams}
        for _ in range(sims):
            r = {}
            for t in teams:
                atk, dfn = ratings[t]
                form = math.exp(random.gauss(0, sf_e))
                r[t] = (atk * form * math.exp(random.gauss(0, si_e)),
                        dfn / form * math.exp(random.gauss(0, si_e)))
            res = list(played)
            for m in remaining:
                ha, hd = r[m["homeId"]]; aa, ad = r[m["awayId"]]
                hxg = max(0.05, (ha / LEAGUE_AVG) * ad * 1.11)
                axg = max(0.05, (aa / LEAGUE_AVG) * hd * 0.89)
                res.append({"homeId": m["homeId"], "awayId": m["awayId"],
                            "homeGoals": _poisson(hxg), "awayGoals": _poisson(axg)})
            order, pts = _table_order(res, teams)
            for t in teams:
                pos_count[t][order[t]] += 1
                pts_samples[t].append(pts[t])

        sim_rps = freeze_rps = covered = 0.0
        for t in teams:
            dist = [c / sims for c in pos_count[t]]
            sim_rps += _pos_rps(dist, actual_pos[t])
            frozen_dist = [0.0] * 21; frozen_dist[frozen[t]] = 1.0
            freeze_rps += _pos_rps(frozen_dist, actual_pos[t])
            s = sorted(pts_samples[t])
            lo, hi = s[int(0.05 * sims)], s[int(0.95 * sims)]
            if lo <= actual_pts[t] <= hi:
                covered += 1
        n = len(teams)
        mw = cut // 10
        print(f"after MW{mw:<10}{sim_rps/n:>12.4f}{freeze_rps/n:>15.4f}"
              f"{covered/n:>10.2f}")
    print("\nposRPS: lower = better (sim should beat the naive 'freeze current "
          "table' baseline, especially early). pts cover target ≈ 0.90.")


if __name__ == "__main__":
    import sys
    if "--espn" in sys.argv:
        main_espn()
    elif "--bench" in sys.argv:
        bench()
    elif "--table" in sys.argv:
        backtest_table()
    else:
        main()
