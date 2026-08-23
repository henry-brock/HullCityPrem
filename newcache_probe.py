"""
newcache_probe.py
Ranks candidate team-strength metrics from the Newcache Opta archive by how
well they actually predict FUTURE goals.

Method (same split-season logic derive_priors.py used to pick xG over goals,
but pointed at the decision-relevant target):
  • split each team's season into its first 19 and last 19 matches
  • for each candidate metric, take the team's first-half per-game value
  • correlate that against the team's SECOND-half per-game goals

A prior is only worth using if it forecasts goals a team has not scored yet,
so corr(H1 metric → H2 goals) is the number that matters. Repeatability
corr(H1 → H2 same metric) is reported alongside as context: a metric can be
highly repeatable yet not predict goals (that gap is exactly what separates
process from finishing).

Attack uses a team's own stats; defence uses what it conceded.
Pooled across every season in Newcache, so n = seasons × 20 teams.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT     = Path(__file__).parent
NEWCACHE = ROOT / "Newcache"

HALF = 19   # PL season split point


# ── stats helpers ───────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def _ols2(rows):
    """Fit y = a·x1 + b·x2 (no intercept) via 2×2 normal equations."""
    s11 = sum(x1 * x1 for x1, _, _ in rows)
    s22 = sum(x2 * x2 for _, x2, _ in rows)
    s12 = sum(x1 * x2 for x1, x2, _ in rows)
    s1y = sum(x1 * y for x1, _, y in rows)
    s2y = sum(x2 * y for _, x2, y in rows)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-9:
        return 0.0, 0.0
    return (s1y * s22 - s2y * s12) / det, (s11 * s2y - s12 * s1y) / det


def _g(block, name):
    """Opta omits zero-valued stats, so a missing key means 0."""
    v = (block or {}).get(name)
    return 0.0 if v is None else float(v)


# ── load ────────────────────────────────────────────────────────────────

def load_matches():
    """Yield (season, side_stats, opp_stats, goals_for, goals_against, team)."""
    out = []
    for path in sorted(NEWCACHE.glob("pl_*.json")):
        data = json.loads(path.read_text())
        if data.get("season") == "2627":
            continue                     # current season: too few matches
        for m in data.get("matches", []):
            if m.get("homeGoals") is None or not m["stats"].get("home"):
                continue
            hs, as_ = m["stats"]["home"], m["stats"]["away"]
            hg, ag  = float(m["homeGoals"]), float(m["awayGoals"])
            out.append((data["season"], m["ms"], m["homeName"], hs, as_, hg, ag))
            out.append((data["season"], m["ms"], m["awayName"], as_, hs, ag, hg))
    return out


# ── candidate metrics ───────────────────────────────────────────────────
# Each maps (own_stats, opp_stats) → a per-match attacking value. The
# defensive version of a metric is the same function applied to what the
# opponent produced, so one definition covers both sides.

def build_candidates(ibox_w, obox_w):
    """name → (fn, required raw stat keys).

    The keys drive a coverage check: Opta added several of these stats
    partway through the archive, and a stat present in only part of a season
    produces a real-looking but meaningless correlation.
    """
    return {
        "goals":              (lambda s, o, gf, ga: gf, []),
        "shots":              (lambda s, o, gf, ga: _g(s, "total_scoring_att"),
                               ["total_scoring_att"]),
        "SoT":                (lambda s, o, gf, ga: _g(s, "ontarget_scoring_att"),
                               ["ontarget_scoring_att"]),
        "xG proxy (current)": (lambda s, o, gf, ga: (
            0.33 * _g(s, "ontarget_scoring_att")
            - 0.01 * (_g(s, "total_scoring_att") - _g(s, "ontarget_scoring_att"))),
                               ["ontarget_scoring_att", "total_scoring_att"]),
        "xG location-fit":    (lambda s, o, gf, ga: (
            ibox_w * _g(s, "attempts_ibox") + obox_w * _g(s, "attempts_obox")),
                               ["attempts_ibox", "attempts_obox"]),
        "shots in box":       (lambda s, o, gf, ga: _g(s, "attempts_ibox"),
                               ["attempts_ibox"]),
        "touches in opp box": (lambda s, o, gf, ga: _g(s, "touches_in_opp_box"),
                               ["touches_in_opp_box"]),
        "pen-area entries":   (lambda s, o, gf, ga: _g(s, "successful_pen_area_entries"),
                               ["successful_pen_area_entries"]),
        "shots created":      (lambda s, o, gf, ga: _g(s, "shot_created"),
                               ["shot_created"]),
        "big chances scored": (lambda s, o, gf, ga: _g(s, "big_chance_scored"),
                               ["big_chance_scored"]),
        "final-third entries": (lambda s, o, gf, ga: _g(s, "successful_final_third_entries"),
                                ["successful_final_third_entries"]),
        "possession %":       (lambda s, o, gf, ga: _g(s, "possession_percentage"),
                               ["possession_percentage"]),
    }


def coverage(rows, keys) -> float:
    """Share of match-sides carrying every required key."""
    if not keys:
        return 1.0
    have = sum(1 for _se, _ms, _t, own, _o, _gf, _ga in rows
               if all(k in (own or {}) for k in keys))
    return have / len(rows) if rows else 0.0


def evaluate(rows, candidates):
    """Return {metric: (attack_pred, attack_rep, def_pred, def_rep)}."""
    # team-season → chronological match list
    by_team = defaultdict(list)
    for season, ms, team, own, opp, gf, ga in rows:
        by_team[(season, team)].append((ms, own, opp, gf, ga))
    for key in by_team:
        by_team[key].sort(key=lambda r: r[0])

    results = {}
    for name, (fn, _keys) in candidates.items():
        a_h1, a_h2g, a_h2m = [], [], []
        d_h1, d_h2g, d_h2m = [], [], []

        for (_season, _team), games in by_team.items():
            if len(games) < HALF * 2:
                continue
            h1, h2 = games[:HALF], games[HALF:]

            # Attack: own stats → own future goals
            a_h1.append(_mean([fn(own, opp, gf, ga) for _, own, opp, gf, ga in h1]))
            a_h2m.append(_mean([fn(own, opp, gf, ga) for _, own, opp, gf, ga in h2]))
            a_h2g.append(_mean([gf for *_x, gf, _ga in [(m, o, p, gf, ga) for m, o, p, gf, ga in h2]]))

            # Defence: what the opponent produced → own future goals conceded
            d_h1.append(_mean([fn(opp, own, ga, gf) for _, own, opp, gf, ga in h1]))
            d_h2m.append(_mean([fn(opp, own, ga, gf) for _, own, opp, gf, ga in h2]))
            d_h2g.append(_mean([ga for _m, _o, _p, _gf, ga in h2]))

        results[name] = (
            _corr(a_h1, a_h2g), _corr(a_h1, a_h2m),
            _corr(d_h1, d_h2g), _corr(d_h1, d_h2m),
            len(a_h1),
        )
    return results


def main():
    rows = load_matches()
    if not rows:
        print("Newcache is empty — run newcache_builder.py first.", file=sys.stderr)
        sys.exit(1)

    # Fit shot-location weights: goals ~ a·ibox + b·obox, at match level.
    fit_rows = [(_g(s, "attempts_ibox"), _g(s, "attempts_obox"), gf)
                for _se, _ms, _t, s, _o, gf, _ga in rows]
    ibox_w, obox_w = _ols2(fit_rows)

    seasons = sorted({r[0] for r in rows})
    print(f"Newcache probe — seasons {', '.join(seasons)}  ({len(rows)//2} matches)")
    print(f"Fitted shot-location value:  in-box {ibox_w:.4f} goals/shot   "
          f"out-of-box {obox_w:.4f}   (ratio {ibox_w / obox_w:.1f}×)"
          if obox_w else "")
    print()

    candidates = build_candidates(ibox_w, obox_w)
    results    = evaluate(rows, candidates)
    n          = next(iter(results.values()))[4]

    print(f"corr(first-half metric → SECOND-half goals), n={n} team-seasons")
    print(f"{'metric':22} {'ATTACK':>8} {'(repeat)':>9}   {'DEFENCE':>8} {'(repeat)':>9}  {'cover':>6}")
    print("-" * 72)
    for name, (ap, ar, dp, dr, _n) in sorted(
            results.items(), key=lambda kv: -kv[1][0]):
        cov  = coverage(rows, candidates[name][1])
        flag = "" if cov > 0.95 else "  ← partial coverage, ignore"
        print(f"{name:22} {ap:8.3f} {ar:9.3f}   {dp:8.3f} {dr:9.3f}  {cov:6.0%}{flag}")


if __name__ == "__main__":
    main()
