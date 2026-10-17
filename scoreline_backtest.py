"""
scoreline_backtest.py — does a Premier League-specific scoreline correction
beat the generic Dixon-Coles one?

Dixon-Coles adjusts four cells (0-0, 1-0, 0-1, 1-1) with one parameter, rho,
from a 1997 study of English football. The question here: fit a separate
ratio for EVERY scoreline from 0-0 to 3-3 to how the modern Premier League
actually finishes games, and see whether that predicts unseen matches better.

Variants, all on identical team ratings so only the scoreline shape differs:
  production   rho -0.13 — what the app runs today
  poisson      rho 0 — no correction at all, the reference point
  refit rho    rho re-tuned on the training seasons
  PL ratios    plain Poisson, then a fitted multiplier on each 0-0…3-3 cell

Folds — "a single fold is not evidence", and 2026/27 has only ~51 matches:
  validation   fit 2024/25            → test 2025/26 (380 matches; also picks
                                         the ratios' shrinkage strength)
  final        fit 2024/25 + 2025/26  → test 2026/27 (the season the app is
                                         actually predicting)

Ratings mirror the app's getTeamRatings exactly: pre-season prior, blended
with recency-weighted actual goals on a smoothstep ramp to 65% by game ten
(80% for promoted sides). 2026/27 uses the app's real priors from index.html;
past seasons use the 8-stat composite from the seasons before them, with
promoted sides given the average of the previous season's bottom three.

Usage:  python3 scoreline_backtest.py           four-variant report
        python3 scoreline_backtest.py --sweep   100 ratio options
        python3 scoreline_backtest.py --attribute   which groups of the winner help
        python3 scoreline_backtest.py --candidates  4+, 0-1, 1-0, 2-1/1-2 on top of 2-2
        python3 scoreline_backtest.py --dampen      shrink expected mismatches (alpha)
"""

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from prior_backtest import build_priors_composite, load

ROOT       = Path(__file__).parent
LEAGUE_AVG = 1.40
HOME, AWAY = 1.11, 0.89
PROD_RHO   = -0.13
MAXG       = 10
CELLS      = [(i, j) for i in range(4) for j in range(4)]       # 0-0 … 3-3
_FACT      = [math.factorial(k) for k in range(MAXG + 1)]


# ── ratings, walk-forward ───────────────────────────────────────────────
def _rating(games, prior, promoted):
    n = len(games)
    if not n:
        return prior
    atk = dfn = wsum = 0.0
    for i, (scored, conceded) in enumerate(games):
        w = 1 + (i / (n - 1) if n > 1 else 0)
        atk += scored * w
        dfn += conceded * w
        wsum += w
    cap = 0.80 if promoted else 0.65
    t = min(1.0, n / 10)
    blend = cap * t * t * (3 - 2 * t)
    return (prior[0] * (1 - blend) + atk / wsum * blend,
            prior[1] * (1 - blend) + dfn / wsum * blend)


def _season_rows(matches, priors, promoted):
    """(base home xG, base away xG, home goals, away goals) per match, each
    rated only on what came before it. The home/away multipliers and the
    scoreline shape are applied later, so one walk serves every variant."""
    hist, rows = defaultdict(list), []
    for h, a, hg, ag in matches:
        H = _rating(hist[h], priors[h], h in promoted)
        A = _rating(hist[a], priors[a], a in promoted)
        rows.append(((H[0] / LEAGUE_AVG) * A[1], (A[0] / LEAGUE_AVG) * H[1], hg, ag))
        hist[h].append((hg, ag))
        hist[a].append((ag, hg))
    return rows


def _bottom3(season):
    pts, gf, ga, gp = Counter(), Counter(), Counter(), Counter()
    for m in load(season):
        for t, f, a in ((m["homeName"], m["homeGoals"], m["awayGoals"]),
                        (m["awayName"], m["awayGoals"], m["homeGoals"])):
            gp[t] += 1; gf[t] += f; ga[t] += a
            pts[t] += 3 if f > a else (1 if f == a else 0)
    worst = sorted(pts, key=lambda t: (pts[t], gf[t] - ga[t]))[:3]
    return [(gf[t] / gp[t], ga[t] / gp[t]) for t in worst]


def historical(season, train):
    ms = sorted(load(season), key=lambda m: m["ms"])
    priors = build_priors_composite(train)
    prev = {m[k] for m in load(train[-1]) for k in ("homeName", "awayName")}
    teams = {m[k] for m in ms for k in ("homeName", "awayName")}
    promoted = teams - prev
    rel = [x for s in train for x in _bottom3(s)]
    promo_prior = (sum(a for a, _ in rel) / len(rel), sum(d for _, d in rel) / len(rel))
    full = {t: priors.get(t, promo_prior) for t in teams}
    for t in promoted:
        full[t] = promo_prior
    return _season_rows([(m["homeName"], m["awayName"], int(m["homeGoals"]), int(m["awayGoals"]))
                         for m in ms], full, promoted)


def current():
    """2026/27 with the app's own priors, straight from index.html."""
    html = (ROOT / "index.html").read_text()
    priors = {m[0]: (float(m[1]), float(m[2])) for m in
              re.findall(r"id:\s*'(\w+)'.*?baseAtk:\s*([\d.]+),\s*baseDef:\s*([\d.]+)", html)}
    promoted = set(re.findall(r"(\w+):", re.search(r"PROMOTED_SIGMA = \{([^}]*)\}", html).group(1)))
    res = json.loads((ROOT / "cache" / "pulse_2627.json").read_text())["matches"]
    when = lambda r: datetime.strptime(f"{r['date']} {r['time']}", "%d/%m/%Y %H:%M")
    ms = [(r["homeId"], r["awayId"], int(r["homeGoals"]), int(r["awayGoals"]))
          for r in sorted(res, key=when)]
    return _season_rows(ms, priors, promoted)


# ── score distributions ─────────────────────────────────────────────────
def _matrix(hxg, axg, rho, ratios):
    ph = [math.exp(-hxg) * hxg ** k / _FACT[k] for k in range(MAXG + 1)]
    pa = [math.exp(-axg) * axg ** k / _FACT[k] for k in range(MAXG + 1)]
    m = [[ph[i] * pa[j] for j in range(MAXG + 1)] for i in range(MAXG + 1)]
    if rho:
        m[0][0] *= max(0.0, 1 - hxg * axg * rho)
        m[0][1] *= max(0.0, 1 + hxg * rho)
        m[1][0] *= max(0.0, 1 + axg * rho)
        m[1][1] *= max(0.0, 1 - rho)
    if ratios:
        for (i, j), r in ratios.items():
            m[i][j] *= r
    s = sum(map(sum, m))
    return [[c / s for c in row] for row in m]


def _mats(rows, rho, ratios=None):
    return [_matrix(max(0.05, bh * HOME), max(0.05, ba * AWAY), rho, ratios) for bh, ba, _, _ in rows]


def score(rows, mats):
    """Mean per-match losses: exact scoreline, H/D/A log-loss, H/D/A RPS."""
    ex = out = rps = 0.0
    per_ex, per_out = [], []
    for (_, _, hg, ag), m in zip(rows, mats):
        p_ex = m[min(hg, MAXG)][min(ag, MAXG)]
        hw = sum(m[i][j] for i in range(MAXG + 1) for j in range(i))
        dr = sum(m[i][i] for i in range(MAXG + 1))
        aw = 1 - hw - dr
        o = "H" if hg > ag else ("D" if hg == ag else "A")
        p_o = {"H": hw, "D": dr, "A": aw}[o]
        e, l = -math.log(max(p_ex, 1e-12)), -math.log(max(p_o, 1e-12))
        per_ex.append(e); per_out.append(l)
        ex += e; out += l
        cum, obs = [hw, hw + dr], [1 if o == "H" else 0, 0 if o == "A" else 1]
        rps += sum((c - b) ** 2 for c, b in zip(cum, obs)) / 2
    n = len(rows)
    return {"exact": ex / n, "outcome": out / n, "rps": rps / n, "_ex": per_ex, "_out": per_out}


def fit_rho(rows):
    best = min((score(rows, _mats(rows, r / 100))["exact"], r / 100) for r in range(-30, 21))
    return best[1]


def fit_ratios(rows, shrink):
    """Multiplier per 0-0…3-3 cell so predicted counts match observed ones.

    Fitted by iterative proportional scaling: renormalising each match after
    a cell is boosted pulls probability off the other cells, so one pass of
    observed/expected is not the answer and it is repeated until it settles.
    `shrink` is that many pseudo-matches at the uncorrected model's own rate,
    blended into each cell's TARGET count before fitting — so thin cells
    (3-3 is ~1% of games) are pulled toward a ratio of 1.0 rather than
    trusting eight matches' worth of evidence outright. (Shrinking inside the
    update instead would wash out: iterated to convergence, it fits the raw
    counts regardless.)
    """
    n = len(rows)
    obs = Counter((hg, ag) for _, _, hg, ag in rows)
    base = _mats(rows, 0)
    target = {c: (obs[c] + shrink * sum(m[c[0]][c[1]] for m in base) / n) * n / (n + shrink)
              for c in CELLS}
    ratios = {c: 1.0 for c in CELLS}
    for _ in range(40):
        mats = _mats(rows, 0, ratios)
        for c in CELLS:
            exp = sum(m[c[0]][c[1]] for m in mats)
            ratios[c] *= target[c] / exp if exp else 1.0
    return ratios


# ── report ──────────────────────────────────────────────────────────────
def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mu = sum(d) / len(d)
    sd = (sum((x - mu) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    return mu, 1.96 * sd / len(d) ** 0.5


def _table(label, rows, variants):
    print(f"\n{label}  ({len(rows)} matches)")
    print(f"  {'variant':<22}{'exact-score LL':>15}{'H/D/A LL':>10}{'RPS':>8}")
    base = None
    for name, mats in variants:
        s = score(rows, mats)
        base = base or s
        print(f"  {name:<22}{s['exact']:>15.4f}{s['outcome']:>10.4f}{s['rps']:>8.4f}")
    return base


def _calibration(rows, variants):
    obs = Counter((hg, ag) for _, _, hg, ag in rows)
    n = len(rows)
    names = [v[0] for v in variants]
    print(f"\n  {'score':<6}{'actual':>8}" + "".join(f"{nm[:12]:>14}" for nm in names))
    for c in CELLS:
        line = f"  {c[0]}-{c[1]:<4}{obs[c] / n * 100:>7.1f}%"
        for _, mats in variants:
            line += f"{sum(m[c[0]][c[1]] for m in mats) / n * 100:>13.1f}%"
        print(line)


# ── 100-option sweep (numpy) ────────────────────────────────────────────
# 2 base models × 5 ways of grouping the 0-0…3-3 cells × 10 shrink strengths.
# Picking the best of 100 on the data it is scored against would flatter the
# winner, so the 100 compete ONLY on the validation fold (fit 2024/25, score
# 2025/26); the single winner is then refitted on 2024/25 + 2025/26 and
# judged once on 2026/27, which none of the 100 ever saw. Very high shrink
# collapses onto the base model, so "keep production" is inside the grid and
# wins if nothing beats it.
GROUPINGS = {
    "all 16 cells":    {c: c for c in CELLS},
    "mirror pairs":    {c: tuple(sorted(c)) for c in CELLS},       # 2-1 shares with 1-2
    "draws only":      {(i, i): (i, i) for i in range(4)},
    "low scores only": {c: c for c in ((0, 0), (1, 0), (0, 1), (1, 1))},
    "by total goals":  {c: (c[0] + c[1], c[0] == c[1]) for c in CELLS},
}
SHRINKS = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
BASES = {"poisson": 0.0, "dixon-coles": PROD_RHO}


def _np_base(rows, rho, alpha=1.0):
    """alpha < 1 compresses each side's xG toward that side's league mean
    (geometrically, so the mean stays put): see dampen()."""
    import numpy as np
    arr = np.asarray(rows, dtype=float)
    hx = np.maximum(0.05, arr[:, 0] * HOME)
    ax = np.maximum(0.05, arr[:, 1] * AWAY)
    if alpha != 1.0:
        mh, ma = LEAGUE_AVG * HOME, LEAGUE_AVG * AWAY
        hx = mh * (hx / mh) ** alpha
        ax = ma * (ax / ma) ** alpha
    g, fact = np.arange(MAXG + 1), np.asarray(_FACT, dtype=float)
    m = (np.exp(-hx)[:, None] * hx[:, None] ** g / fact)[:, :, None] * \
        (np.exp(-ax)[:, None] * ax[:, None] ** g / fact)[:, None, :]
    if rho:
        m[:, 0, 0] *= np.maximum(0, 1 - hx * ax * rho)
        m[:, 0, 1] *= np.maximum(0, 1 + hx * rho)
        m[:, 1, 0] *= np.maximum(0, 1 + ax * rho)
        m[:, 1, 1] *= max(0.0, 1 - rho)
    return m / m.sum(axis=(1, 2))[:, None, None]


def _goals(rows):
    import numpy as np
    return (np.minimum([r[2] for r in rows], MAXG), np.minimum([r[3] for r in rows], MAXG))


def _np_fit(base, rows, grouping, shrink, iters=60):
    import numpy as np
    n = len(rows)
    hg, ag = _goals(rows)
    obs = np.zeros((MAXG + 1, MAXG + 1))
    np.add.at(obs, (hg, ag), 1)
    rate = base.sum(axis=0) / n
    groups = defaultdict(list)
    for cell, key in grouping.items():
        groups[key].append(cell)
    target = {k: sum(obs[c] + shrink * rate[c] for c in cells) * n / (n + shrink)
              for k, cells in groups.items()}
    R = np.ones((MAXG + 1, MAXG + 1))
    for _ in range(iters):
        m = base * R
        exp = (m / m.sum(axis=(1, 2))[:, None, None]).sum(axis=0)
        for k, cells in groups.items():
            e = sum(exp[c] for c in cells)
            if e > 0:
                for c in cells:
                    R[c] *= target[k] / e
    return R


def _np_score(base, rows, R=None):
    import numpy as np
    m = base if R is None else base * R
    m = m / m.sum(axis=(1, 2))[:, None, None]
    hg, ag = _goals(rows)
    idx = np.arange(len(rows))
    ex = -np.log(np.maximum(m[idx, hg, ag], 1e-12))
    below = np.tril(np.ones((MAXG + 1, MAXG + 1)), -1)              # home > away
    hw = (m * below).sum(axis=(1, 2))
    dr = np.trace(m, axis1=1, axis2=2)
    aw = 1 - hw - dr
    p_out = np.where(hg > ag, hw, np.where(hg == ag, dr, aw))
    out = -np.log(np.maximum(p_out, 1e-12))
    oh, od = (hg > ag).astype(float), (hg == ag).astype(float)
    rps = ((hw - oh) ** 2 + (hw + dr - oh - od) ** 2) / 2
    return {"exact": ex.mean(), "outcome": out.mean(), "rps": rps.mean(),
            "_ex": ex.tolist(), "_out": out.tolist()}


def sweep():
    s2425 = historical("2425", ["2324"])
    s2526 = historical("2526", ["2324", "2425"])
    s2627 = current()
    train_v = {b: _np_base(s2425, r) for b, r in BASES.items()}
    test_v  = {b: _np_base(s2526, r) for b, r in BASES.items()}

    results = []
    for b in BASES:
        for gname, grouping in GROUPINGS.items():
            for k in SHRINKS:
                R = _np_fit(train_v[b], s2425, grouping, k)
                results.append((b, gname, k, _np_score(test_v[b], s2526, R)))
    prod_v = _np_score(test_v["dixon-coles"], s2526)
    results.sort(key=lambda r: r[3]["exact"])

    print("=" * 78)
    print(f"SWEEP — {len(results)} ratio options, fit 2024/25, scored on 2025/26 (380 matches)")
    print(f"\n  {'rank':<5}{'base':<13}{'grouping':<17}{'shrink':>7}{'exact LL':>10}{'H/D/A LL':>10}{'RPS':>8}")
    for i, (b, g, k, s) in enumerate(results[:12], 1):
        print(f"  {i:<5}{b:<13}{g:<17}{k:>7}{s['exact']:>10.4f}{s['outcome']:>10.4f}{s['rps']:>8.4f}")
    beat = sum(1 for r in results if r[3]["exact"] < prod_v["exact"])
    print(f"  {'—':<5}{'production (no ratios)':<37}{prod_v['exact']:>10.4f}{prod_v['outcome']:>10.4f}{prod_v['rps']:>8.4f}")
    print(f"\n  {beat} of {len(results)} options beat production on exact score in validation;"
          f" {sum(1 for r in results if r[3]['outcome'] < prod_v['outcome'])} on H/D/A.")
    by_group = defaultdict(list)
    for b, g, k, s in results:
        by_group[(b, g)].append((k, s["exact"]))
    print("\n  best shrink per base × grouping (validation exact LL):")
    for (b, g), v in sorted(by_group.items(), key=lambda kv: min(x[1] for x in kv[1])):
        k, e = min(v, key=lambda x: x[1])
        print(f"    {b:<13}{g:<17} shrink {k:>5}  {e:.4f}")

    b, g, k, sv = results[0]
    mu, ci = _paired(sv["_ex"], prod_v["_ex"])
    print(f"\n  winner vs production in validation: {mu:+.4f} ± {ci:.4f} exact-score LL (95%)")

    print("\n" + "=" * 78)
    print(f"HELD-OUT — winner ({b}, {g}, shrink {k}) refitted on 2024/25 + 2025/26, scored on 2026/27")
    train_f = s2425 + s2526
    R = _np_fit(_np_base(train_f, BASES[b]), train_f, GROUPINGS[g], k)
    win = _np_score(_np_base(s2627, BASES[b]), s2627, R)
    prod = _np_score(_np_base(s2627, PROD_RHO), s2627)
    print(f"\n  {'':<14}{'exact LL':>10}{'H/D/A LL':>10}{'RPS':>8}")
    for nm, s in (("production", prod), ("winner", win)):
        print(f"  {nm:<14}{s['exact']:>10.4f}{s['outcome']:>10.4f}{s['rps']:>8.4f}")
    for key, lab in (("_ex", "exact-score"), ("_out", "H/D/A")):
        mu, ci = _paired(win[key], prod[key])
        print(f"  winner minus production, {lab} LL: {mu:+.4f} ± {ci:.4f} (95%)")
    print("\n  winner's ratios (relative to its base; only cells in its grouping move):")
    print("          " + "".join(f"away {j:<4}" for j in range(4)))
    for i in range(4):
        print(f"    home {i} " + "".join(f"{R[i, j]:>9.2f}" for j in range(4)))


# ── attribution: which of the winner's groups actually help ─────────────
# The sweep winner fits one ratio per (total goals, draw?) group of the
# 0-0…3-3 cells. Every on/off combination of those groups is fitted (2^9 =
# 512), and each group is credited with its average effect on the loss when
# added to a combination that lacked it — averaged over all the others, so
# a group that only helps alongside one particular partner is not mistaken
# for one that helps on its own. Measured on both folds independently; only
# a group that improves both is kept.
WIN_SHRINK = 400


def _groups_of(grouping):
    out = defaultdict(list)
    for cell, key in grouping.items():
        out[key].append(cell)
    return dict(out)


def _label(cells):
    return "/".join(f"{i}-{j}" for i, j in sorted(cells, key=lambda c: (-c[0], c[1])))


def _subset_losses(train, test, keys, groups):
    import itertools
    base_tr, base_te = _np_base(train, PROD_RHO), _np_base(test, PROD_RHO)
    out = {}
    for r in range(len(keys) + 1):
        for subset in itertools.combinations(keys, r):
            grouping = {c: k for k in subset for c in groups[k]}
            R = _np_fit(base_tr, train, grouping, WIN_SHRINK) if subset else None
            s = _np_score(base_te, test, R)
            out[frozenset(subset)] = s
    return out


def _marginals(losses, keys, metric):
    eff = {}
    for k in keys:
        d = [losses[s | {k}][metric] - losses[s][metric] for s in losses if k not in s]
        eff[k] = sum(d) / len(d)
    return eff


def attribution():
    s2425 = historical("2425", ["2324"])
    s2526 = historical("2526", ["2324", "2425"])
    s2627 = current()
    groups = _groups_of(GROUPINGS["by total goals"])
    keys = sorted(groups, key=lambda k: (k[0], not k[1]))

    folds = {"val": _subset_losses(s2425, s2526, keys, groups),
             "held": _subset_losses(s2425 + s2526, s2627, keys, groups)}
    print("=" * 78)
    print(f"ATTRIBUTION — {len(folds['val'])} on/off combinations of the winner's 9 groups, shrink {WIN_SHRINK}")
    print("Average change in loss from switching each group on (negative = it helps)\n")
    print(f"  {'group':<20}{'exact, 25/26':>13}{'exact, 26/27':>13}{'H/D/A, 25/26':>14}{'H/D/A, 26/27':>14}  verdict")
    mv = {m: {f: _marginals(folds[f], keys, m) for f in folds} for m in ("exact", "outcome")}
    keep = []
    for k in keys:
        ev, eh = mv["exact"]["val"][k], mv["exact"]["held"][k]
        ov, oh = mv["outcome"]["val"][k], mv["outcome"]["held"][k]
        good = ev < 0 and eh < 0
        verdict = "keep — helps both" if good else ("mixed" if (ev < 0) != (eh < 0) else "drop — hurts both")
        if good:
            keep.append(k)
        print(f"  {_label(groups[k]):<20}{ev:>+13.4f}{eh:>+13.4f}{ov:>+14.4f}{oh:>+14.4f}  {verdict}")

    full, none, narrow = frozenset(keys), frozenset(), frozenset(keep)
    print("\nWhole models:")
    print(f"  {'':<34}{'exact 25/26':>12}{'exact 26/27':>12}{'H/D/A 25/26':>12}{'H/D/A 26/27':>12}")
    for nm, s in (("production (no ratios)", none), ("sweep winner (all 9 groups)", full),
                  (f"narrowed ({len(keep)} groups)", narrow)):
        v, h = folds["val"][s], folds["held"][s]
        print(f"  {nm:<34}{v['exact']:>12.4f}{h['exact']:>12.4f}{v['outcome']:>12.4f}{h['outcome']:>12.4f}")
    for f, lab in (("val", "2025/26"), ("held", "2026/27")):
        for key, m in (("_ex", "exact"), ("_out", "H/D/A")):
            mu, ci = _paired(folds[f][narrow][key], folds[f][none][key])
            print(f"  narrowed minus production, {lab} {m} LL: {mu:+.4f} ± {ci:.4f} (95%)")

    best = {f: min(folds[f], key=lambda s: folds[f][s]["exact"]) for f in folds}
    for f, lab in (("val", "2025/26"), ("held", "2026/27")):
        print(f"\n  single best combination on {lab} alone: "
              f"{', '.join(_label(groups[k]) for k in keys if k in best[f]) or '(none)'}")
    print("  (each fold's own best is shown for context only — it was chosen on that fold,"
          " so it is not evidence)")

    train = s2425 + s2526
    R = _np_fit(_np_base(train, PROD_RHO), train, {c: k for k in keep for c in groups[k]}, WIN_SHRINK)
    print("\n  narrowed model's ratios, fitted on 2024/25 + 2025/26 (cells outside it stay 1.00 before renormalising):")
    print("          " + "".join(f"away {j:<4}" for j in range(4)))
    for i in range(4):
        print(f"    home {i} " + "".join(f"{R[i, j]:>9.2f}" for j in range(4)))


# ── candidates on top of the shipped 2-2 boost ──────────────────────────
# The calibration check after shipping 2-2 found four suspects: one-sided
# hammerings (a side scoring 4+) over-predicted, 0-1 under and 1-0 over
# (opposite ways — the sweep's mirror grouping lumped them together, which
# hides exactly that), and 2-1/1-2 under. Each is tested on top of the
# current predictor, 2-2 always on. 2-2 is refitted per fold on that fold's
# own training data, so the validation fold never sees 2025/26 through it.
CANDIDATES = {
    "hammerings (4+)": [(i, j) for i in range(MAXG + 1) for j in range(MAXG + 1) if max(i, j) >= 4],
    "0-1":             [(0, 1)],
    "1-0":             [(1, 0)],
    "2-1/1-2":         [(2, 1), (1, 2)],
}


def candidates():
    import itertools
    s2425 = historical("2425", ["2324"])
    s2526 = historical("2526", ["2324", "2425"])
    s2627 = current()
    names = list(CANDIDATES)
    folds = {"val": (s2425, s2526), "held": (s2425 + s2526, s2627)}
    res = {}
    for f, (train, test) in folds.items():
        btr, bte = _np_base(train, PROD_RHO), _np_base(test, PROD_RHO)
        res[f] = {"production": _np_score(bte, test)}
        for r in range(len(names) + 1):
            for combo in itertools.combinations(names, r):
                grouping = {(2, 2): "2-2"}
                for nm in combo:
                    grouping.update({c: nm for c in CANDIDATES[nm]})
                R = _np_fit(btr, train, grouping, WIN_SHRINK)
                res[f][frozenset(combo)] = (_np_score(bte, test, R), R)

    print("=" * 80)
    print(f"CANDIDATES — {2 ** len(names)} combinations on top of DC + 2-2, shrink {WIN_SHRINK}")
    print("Average change in loss from switching each one on (negative = it helps)\n")
    print(f"  {'candidate':<18}{'exact 25/26':>12}{'exact 26/27':>12}{'H/D/A 25/26':>13}{'H/D/A 26/27':>13}  verdict")
    keep = []
    for nm in names:
        eff = {}
        for f in folds:
            for m in ("exact", "outcome"):
                d = [res[f][s | {nm}][0][m] - res[f][s][0][m]
                     for s in res[f] if s != "production" and nm not in s]
                eff[(f, m)] = sum(d) / len(d)
        ev, eh = eff[("val", "exact")], eff[("held", "exact")]
        ov, oh = eff[("val", "outcome")], eff[("held", "outcome")]
        verdict = ("keep — helps both" if ev < 0 and eh < 0
                   else "drop — hurts both" if ev > 0 and eh > 0 else "mixed")
        if ev < 0 and eh < 0:
            keep.append(nm)
        print(f"  {nm:<18}{ev:>+12.4f}{eh:>+12.4f}{ov:>+13.4f}{oh:>+13.4f}  {verdict}")

    print("\nWhole models:")
    print(f"  {'':<34}{'exact 25/26':>12}{'exact 26/27':>12}{'H/D/A 25/26':>12}{'H/D/A 26/27':>12}")
    models = [("production (no ratios)", "production"), ("current: DC + 2-2", frozenset()),
              ("all four added", frozenset(names)), (f"kept: {', '.join(keep) or 'none'}", frozenset(keep))]
    for label, key in models:
        v = res["val"][key] if key == "production" else res["val"][key][0]
        h = res["held"][key] if key == "production" else res["held"][key][0]
        print(f"  {label[:34]:<34}{v['exact']:>12.4f}{h['exact']:>12.4f}{v['outcome']:>12.4f}{h['outcome']:>12.4f}")
    if keep:
        for f, lab in (("val", "2025/26"), ("held", "2026/27")):
            for key, m in (("_ex", "exact"), ("_out", "H/D/A")):
                mu, ci = _paired(res[f][frozenset(keep)][0][key], res[f][frozenset()][0][key])
                print(f"  kept minus current (DC + 2-2), {lab} {m} LL: {mu:+.4f} ± {ci:.4f} (95%)")
        R = res["held"][frozenset(keep)][1]
        print("\n  ratios fitted on 2024/25 + 2025/26 for the kept set (plus 2-2):")
        for nm in ["2-2"] + keep:
            cells = [(2, 2)] if nm == "2-2" else CANDIDATES[nm]
            print(f"    {nm:<18}{float(R[cells[0]]):.3f}")


# ── dampening: the model expects bigger mismatches than reality delivers ─
# Calibration by mismatch size showed favourites winning less often than
# predicted at every confidence level, and in the biggest mismatches the
# stronger side scoring 2.09 against 2.62 expected — hammerings at 12.5%
# against 27.2%. One parameter, alpha, pulls each side's expected goals
# toward its side's league mean: xG' = mean × (xG / mean)^alpha. alpha = 1 is
# the current model; below 1 shrinks mismatches. Shipped 2-2/0-1 boosts on.
SHIPPED = {(2, 2): 1.42, (0, 1): 1.18}


def _shipped_R():
    import numpy as np
    R = np.ones((MAXG + 1, MAXG + 1))
    for c, v in SHIPPED.items():
        R[c] = v
    return R


def dampen():
    s2425 = historical("2425", ["2324"])
    s2526 = historical("2526", ["2324", "2425"])
    s2627 = current()
    R = _shipped_R()
    grid = [a / 100 for a in range(50, 111)]

    def fit(rows, metric):
        return min(grid, key=lambda a: _np_score(_np_base(rows, PROD_RHO, a), rows, R)[metric])

    print("=" * 78)
    print("DAMPENING — xG' = mean × (xG/mean)^alpha, on top of DC + shipped 2-2/0-1")
    for label, train, test in (("fit 2024/25 → test 2025/26", s2425, s2526),
                               ("fit 2024/25 + 2025/26 → test 2026/27", s2425 + s2526, s2627)):
        a_ex, a_out = fit(train, "exact"), fit(train, "outcome")
        cur = _np_score(_np_base(test, PROD_RHO), test, R)
        print(f"\n{label}   (alpha fitted on training: {a_ex:.2f} by exact score, {a_out:.2f} by W/D/L)")
        print(f"  {'':<22}{'exact LL':>10}{'W/D/L LL':>10}{'RPS':>8}")
        print(f"  {'current (alpha 1.00)':<22}{cur['exact']:>10.4f}{cur['outcome']:>10.4f}{cur['rps']:>8.4f}")
        for a in sorted({a_ex, a_out}):
            s = _np_score(_np_base(test, PROD_RHO, a), test, R)
            print(f"  {f'alpha {a:.2f}':<22}{s['exact']:>10.4f}{s['outcome']:>10.4f}{s['rps']:>8.4f}")
            for key, m in (("_out", "W/D/L"), ("_ex", "exact")):
                mu, ci = _paired(s[key], cur[key])
                sig = "  ← beyond noise" if mu + ci < 0 else ""
                print(f"     alpha {a:.2f} minus current, {m} LL: {mu:+.4f} ± {ci:.4f}{sig}")
    # how the loss moves across alpha on the big validation season (shape check)
    print("\n  2025/26 W/D/L LL across alpha (fitted on nothing — just the curve):")
    for a in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05):
        s = _np_score(_np_base(s2526, PROD_RHO, a), s2526, R)
        print(f"    alpha {a:.2f}: W/D/L {s['outcome']:.4f}   exact {s['exact']:.4f}")


def main():
    s2425 = historical("2425", ["2324"])
    s2526 = historical("2526", ["2324", "2425"])
    s2627 = current()

    # Validation fold: pick the shrinkage on 2025/26, fitting on 2024/25 alone.
    print("=" * 72)
    print("VALIDATION — fit 2024/25, test 2025/26")
    rho_v = fit_rho(s2425)
    variants = [("production (rho -0.13)", _mats(s2526, PROD_RHO)),
                ("poisson (rho 0)", _mats(s2526, 0)),
                (f"refit rho ({rho_v:+.2f})", _mats(s2526, rho_v))]
    shrinks = {}
    for k in (0, 10, 30, 100):
        r = fit_ratios(s2425, k)
        variants.append((f"PL ratios, shrink {k}", _mats(s2526, 0, r)))
        shrinks[k] = score(s2526, variants[-1][1])["exact"]
    _table("2025/26", s2526, variants)
    best_k = min(shrinks, key=shrinks.get)
    prod = score(s2526, variants[0][1])
    mine = score(s2526, _mats(s2526, 0, fit_ratios(s2425, best_k)))
    for key, lab in (("_ex", "exact-score"), ("_out", "H/D/A")):
        mu, ci = _paired(mine[key], prod[key])
        print(f"  PL ratios (shrink {best_k}) minus production, {lab} LL: {mu:+.4f} ± {ci:.4f} (95%)")

    # Final fold: everything fitted on the two seasons before, tested on now.
    print("\n" + "=" * 72)
    print(f"FINAL — fit 2024/25 + 2025/26, test 2026/27 (shrink {best_k}, chosen above)")
    train = s2425 + s2526
    rho_f = fit_rho(train)
    ratios = fit_ratios(train, best_k)
    print("\nFitted PL ratios (multiplier on plain Poisson; >1 = the PL produces it more often):")
    print("        " + "".join(f"away {j:<4}" for j in range(4)))
    for i in range(4):
        print(f"  home {i} " + "".join(f"{ratios[(i, j)]:>9.2f}" for j in range(4)))
    final = [("production (rho -0.13)", _mats(s2627, PROD_RHO)),
             ("poisson (rho 0)", _mats(s2627, 0)),
             (f"refit rho ({rho_f:+.2f})", _mats(s2627, rho_f)),
             ("PL ratios", _mats(s2627, 0, ratios))]
    _table("2026/27", s2627, final)
    prod = score(s2627, final[0][1]); mine = score(s2627, final[3][1])
    for key, lab in (("_ex", "exact-score"), ("_out", "H/D/A")):
        mu, ci = _paired(mine[key], prod[key])
        print(f"  PL ratios minus production, {lab} LL: {mu:+.4f} ± {ci:.4f} (95%)")
    print("\nPredicted vs actual frequency, 2026/27:")
    _calibration(s2627, [final[0], final[3]])
    print("\nPredicted vs actual frequency, 2025/26 (validation fold, 380 matches):")
    _calibration(s2526, [variants[0], (f"PL ratios", _mats(s2526, 0, fit_ratios(s2425, best_k)))])
    print("\nLower is better for every loss. Exact-score LL is the one this tests;")
    print("H/D/A LL and RPS are the model's day job and must not get worse.")


if __name__ == "__main__":
    import sys
    (dampen() if "--dampen" in sys.argv else candidates() if "--candidates" in sys.argv
     else attribution() if "--attribute" in sys.argv
     else sweep() if "--sweep" in sys.argv else main())
