"""
promoted_profile.py — multi-dimensional ranking & analysis of the promoted
clubs (Hull, Ipswich, Coventry) using the full ESPN team-stat catalogue
(cache/espn_teamstats_eng.2_2526.json), not just goals.

For each promoted team it builds a profile across four dimensions, scored as
PERCENTILE RANK within the 24-team Championship (0 = worst, 100 = best):

  ATTACK   — shot volume, shots on target, shot accuracy, conversion,
             chance creation (shot assists), goals
  DEFENCE  — goals conceded (inverted), clean sheets  [outcome]
             + tackles / interceptions / clearances   [workload, descriptive]
  CONTROL  — possession, pass accuracy                 [how much they dictate]
  DIRECTNESS — long-ball share, low possession         [style, not quality]

The point: results flattered Hull (6th on a negative xG). Ranking on the
underlying process instead separates genuine quality (Coventry, Ipswich)
from a side that over-achieved its numbers (Hull) — while the CONTROL /
DIRECTNESS axes quantify the style question (is Hull already PL-adapted?).

A composite QUALITY score (mean of attack & defence percentiles) gives the
underlying ranking; the style axes are reported alongside, never folded into
quality.
"""

import argparse

from half_season import slice_stats

PROMOTED = {"Hull City": "hull", "Ipswich Town": "ipswich", "Coventry City": "coventry"}

# metric → (label, higher_is_better, per_game)   per_game=True ⇒ divide by games
# CREATION = repeatable chance generation (feeds quality).
CREATION = {
    "totalShots":     ("shots",       True,  True),
    "shotsOnTarget":  ("SoT",         True,  True),
}
# FINISHING = conversion efficiency. NOT repeatable season-to-season, so it
# is reported as a sustainability flag, never folded into quality. High
# finishing + low creation = over-achiever riding variance (e.g. Hull).
FINISHING = {
    "shotPct":        ("SoT%",        True,  False),
    "goalConversion": ("conversion",  True,  False),
}
DEFENCE = {
    "goalsConceded":  ("conceded",    False, True),
    "cleanSheet":     ("clean sheets", True, True),
}
CONTROL = {
    "possessionPct":  ("possession",  True,  False),
    "passPct":        ("pass acc",    True,  False),
}
DIRECTNESS = {
    "longballPct":    ("long-ball%",  True,  False),
}


def load(half="2nd", league="eng.2", code="2526"):
    # Aggregates recomputed from the per-match cache over the chosen window.
    return slice_stats(league, code, half=half)


def _value(stats, key, per_game, games):
    v = stats.get(key) or 0
    return v / games if per_game else v


def percentiles(teams, games, key, higher, per_game):
    """team → percentile (0-100) for one metric across the league."""
    vals = {t: _value(s, key, per_game, games) for t, s in teams.items()}
    order = sorted(vals.values())
    n = len(order)
    out = {}
    for t, v in vals.items():
        rank = sum(1 for x in order if x < v)          # how many below
        pct = rank / (n - 1) * 100
        out[t] = pct if higher else 100 - pct
    return out, vals


def dimension(teams, games, spec):
    """Return per-team {metric: (pctile, raw)} and the mean percentile."""
    pcts = {t: {} for t in teams}
    raws = {t: {} for t in teams}
    for key, (label, higher, per_game) in spec.items():
        p, v = percentiles(teams, games, key, higher, per_game)
        for t in teams:
            pcts[t][label] = p[t]
            raws[t][label] = v[t]
    means = {t: sum(pcts[t].values()) / len(spec) for t in teams}
    return pcts, raws, means


def bar(pct, width=10):
    fill = round(pct / 100 * width)
    return "█" * fill + "·" * (width - fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="2nd", choices=["2nd", "1st", "full"])
    args = ap.parse_args()

    teams, games = load(half=args.half)
    cre_p, cre_r, cre_m = dimension(teams, games, CREATION)
    fin_p, fin_r, fin_m = dimension(teams, games, FINISHING)
    def_p, def_r, def_m = dimension(teams, games, DEFENCE)
    ctl_p, ctl_r, ctl_m = dimension(teams, games, CONTROL)
    dir_p, dir_r, dir_m = dimension(teams, games, DIRECTNESS)

    # QUALITY = repeatable process only (creation + defence). Finishing excluded.
    quality = {t: (cre_m[t] + def_m[t]) / 2 for t in teams}
    league_rank = {t: i + 1 for i, (t, _) in
                   enumerate(sorted(quality.items(), key=lambda kv: -kv[1]))}

    window = {"2nd": "SECOND HALF", "1st": "first half", "full": "full season"}[args.half]
    print(f"\n══ Promoted-team profiles — {window} of 2025/26 Championship "
          f"(~{games} games), percentile rank within the league ══\n")
    for name in sorted(PROMOTED, key=lambda n: -quality[n]):
        # finishing flag: clinical (high finishing) on top of weak creation = luck
        flag = ""
        if fin_m[name] >= 70 and cre_m[name] <= 45:
            flag = "  ⚠ over-achieving (clinical finishing masking weak creation)"
        elif fin_m[name] <= 30 and cre_m[name] >= 55:
            flag = "  ↑ under-finishing (creation better than results suggest)"
        print(f"▌ {name}   QUALITY {quality[name]:.0f}/100  "
              f"(league quality rank {league_rank[name]}/24){flag}")
        print(f"    CREATION  {bar(cre_m[name])} {cre_m[name]:>3.0f}   "
              f"shots/g {cre_r[name]['shots']:.1f}  SoT/g {cre_r[name]['SoT']:.1f}")
        print(f"    DEFENCE   {bar(def_m[name])} {def_m[name]:>3.0f}   "
              f"conceded/g {def_r[name]['conceded']:.2f}  "
              f"clean-sheet/g {def_r[name]['clean sheets']:.2f}")
        print(f"    FINISHING {bar(fin_m[name])} {fin_m[name]:>3.0f}   "
              f"SoT% {fin_r[name]['SoT%']*100:.0f}  conversion {fin_r[name]['conversion']*100:.0f}%"
              f"   (luck axis — excluded from quality)")
        print(f"    CONTROL   {bar(ctl_m[name])} {ctl_m[name]:>3.0f}   "
              f"possession {ctl_r[name]['possession']:.0f}%  "
              f"pass-acc {ctl_r[name]['pass acc']*100:.0f}%  "
              f"long-ball {dir_r[name]['long-ball%']*100:.0f}%")
        print()

    print(f"══ Underlying QUALITY ranking ({window}) ══")
    for i, (name, q) in enumerate(
            sorted(((n, quality[n]) for n in PROMOTED), key=lambda kv: -kv[1]), 1):
        style = ("controlling" if ctl_m[name] >= 55 else
                 "reactive/direct" if ctl_m[name] <= 35 else "balanced")
        print(f"  {i}. {name:<14} quality {q:>3.0f}/100   "
              f"creation {cre_m[name]:>3.0f}  defence {def_m[name]:>3.0f}  "
              f"finishing {fin_m[name]:>3.0f}  style: {style}")

    print("\nQUALITY = repeatable process (creation + defence), finishing excluded. "
          "Compare with the league finish (Coventry 1st, Ipswich 2nd, Hull 6th) to "
          "see who over/under-achieved. CONTROL/long-ball describe style for the "
          "PL-transfer question.")


if __name__ == "__main__":
    main()
