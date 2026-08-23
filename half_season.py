"""
half_season.py — recompute team aggregates from a slice of the season,
using the per-match game cache (cache/espn_games_<league>_<code>.json).

The season-aggregate stat catalogue can't be split, but the game cache can:
for each team we sort its matches by date and keep a window (default the
SECOND HALF — each team's last 23 of 46 Championship games), then re-derive
the stats the profiler needs. Recent form is a better guide to the step up
than a full season that includes a stale opening run.

Returns ({team: stats}, games_per_team) with the same keys promoted_profile
consumes, so it is a drop-in replacement for the catalogue loader. Rates are
rebuilt from raw components (the per-match rate fields are rounded);
possession is the per-match mean.
"""

import json
from pathlib import Path

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"


def _f(stats, key):
    try:
        return float(str(stats.get(key, 0)).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _aggregate(rows):
    """rows: list of (stats, gf, ga). Returns one stats dict."""
    n = len(rows)
    sums = lambda k: sum(_f(s, k) for s, _, _ in rows)
    shots   = sums("totalShots")
    sot     = sums("shotsOnTarget")
    goals   = sum(gf for _, gf, _ in rows)
    passes  = sums("totalPasses")
    accPass = sums("accuratePasses")
    longs   = sums("totalLongBalls")
    return {
        # volume (profiler divides by games)
        "totalShots":     shots,
        "shotsOnTarget":  sot,
        "totalGoals":     goals,
        "goalsConceded":  sum(ga for _, _, ga in rows),
        "cleanSheet":     sum(1 for _, _, ga in rows if ga == 0),
        "totalTackles":   sums("totalTackles"),
        "interceptions":  sums("interceptions"),
        "totalClearance": sums("totalClearance"),
        # rates (rebuilt from components / means)
        "shotPct":        sot / shots if shots else 0.0,
        "goalConversion": goals / shots if shots else 0.0,
        "possessionPct":  sum(_f(s, "possessionPct") for s, _, _ in rows) / n if n else 0.0,
        "passPct":        accPass / passes if passes else 0.0,
        "longballPct":    longs / passes if passes else 0.0,
    }


def slice_stats(league="eng.2", code="2526", half="2nd"):
    """half ∈ {'2nd','1st','full'}. Returns ({team: stats}, games_per_team)."""
    games = json.loads((CACHE / f"espn_games_{league}_{code}.json").read_text())["games"]

    by_team = {}
    for r in games.values():
        score = r.get("score") or {}
        for side, s in (r.get("teamStats") or {}).items():
            team = s.get("team")
            if not team:
                continue
            other = "away" if side == "home" else "home"
            try:
                gf = int(score.get(side) or 0)
                ga = int(score.get(other) or 0)
            except ValueError:
                gf = ga = 0
            by_team.setdefault(team, []).append((r.get("date") or "", s["stats"], gf, ga))

    out, gcount = {}, 0
    for team, rows in by_team.items():
        rows.sort(key=lambda x: x[0])
        m = len(rows)
        if half == "2nd":
            sel = rows[m // 2:]
        elif half == "1st":
            sel = rows[:m // 2]
        else:
            sel = rows
        gcount = max(gcount, len(sel))
        out[team] = _aggregate([(s, gf, ga) for _, s, gf, ga in sel])
    return out, gcount


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--half", default="2nd", choices=["2nd", "1st", "full"])
    p.add_argument("--league", default="eng.2")
    args = p.parse_args()
    stats, n = slice_stats(args.league, half=args.half)
    print(f"{args.half} half — {len(stats)} teams, ~{n} games each\n")
    for t in ["Hull City", "Ipswich Town", "Coventry City"]:
        s = stats[t]
        print(f"{t:<14} shots/g {s['totalShots']/n:.1f}  SoT/g {s['shotsOnTarget']/n:.1f}"
              f"  poss {s['possessionPct']:.0f}%  goals/g {s['totalGoals']/n:.2f}"
              f"  conceded/g {s['goalsConceded']/n:.2f}")
