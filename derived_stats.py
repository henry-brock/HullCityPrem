"""
derived_stats.py — ratio metrics that raw Opta counts hide.

Built to test a specific claim: that a side comfortable without the ball
(Hull) converts limited possession into more threat, and funnels opponents
into worse shots, than a side used to controlling games (Coventry/Ipswich).

Design rules
────────────
• Every metric is a RATIO, never a count. Counts mostly measure how much of
  the ball a team had; ratios measure what they did with it.
• `lower_better` is declared per metric so a ranking always reads best-first.
• Metrics that would flatter the thesis are NOT cherry-picked — the
  territorial ones currently favour Coventry and are included anyway, because
  a table of only favourable numbers proves nothing.
• Denominators are guarded. Note in particular why "big chances against per
  GOAL conceded" is not here: it is undefined on a clean sheet — the very game
  you would want it for — and it measures goalkeeping and opponent finishing,
  which this project measured at r=0.038 repeatability. The rate version
  (per 100 shots faced) is used instead.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def _g(block, name):
    """Opta omits zero-valued stats, so a missing key means 0."""
    v = (block or {}).get(name)
    return 0.0 if v is None else float(v)


def _safe(num, den):
    return num / den if den else None


# name, group, description, lower_is_better, fn(own, opp, gf, ga)
METRICS = [
    # ── Chance suppression ────────────────────────────────────────────
    ("Box-shot smother rate", "Chance suppression",
     "Share of opponent shots from inside the box that actually reach the goal. "
     "Low means shots are being taken under pressure, at bad angles, or rushed.",
     True,
     lambda s, o, gf, ga: _safe((_g(o, "att_ibox_target") + _g(o, "att_ibox_goal")) * 100,
                                _g(o, "attempts_ibox"))),
    ("Opponent shot accuracy allowed", "Chance suppression",
     "Share of all opponent shots that hit the target.", True,
     lambda s, o, gf, ga: _safe(_g(o, "ontarget_scoring_att") * 100, _g(o, "total_scoring_att"))),
    ("xG conceded per shot faced", "Chance suppression",
     "Chance quality per attempt, using the project's shots-based xG proxy — "
     "strips out how MANY shots were faced and asks how good they were.", True,
     lambda s, o, gf, ga: _safe(0.33 * _g(o, "ontarget_scoring_att")
                                - 0.01 * (_g(o, "total_scoring_att") - _g(o, "ontarget_scoring_att")),
                                _g(o, "total_scoring_att"))),
    ("Block rate", "Chance suppression",
     "Opponent shots blocked, as a share of shots faced — bodies in the way.", False,
     lambda s, o, gf, ga: _safe(_g(o, "blocked_scoring_att") * 100, _g(o, "total_scoring_att"))),
    ("Big chances allowed per 100 shots", "Chance suppression",
     "The rate version of 'big chances against'. Unlike per-goal-conceded it is "
     "always defined, and measures suppression rather than goalkeeping luck.", True,
     lambda s, o, gf, ga: _safe(_g(o, "big_chance_created") * 100, _g(o, "total_scoring_att"))),

    # ── Engagement ────────────────────────────────────────────────────
    ("Ground duel win %", "Engagement",
     "Duels won excluding aerials. Isolates willingness to engage from physique — "
     "aerial rates were near-identical between Hull and Coventry, ground rates were not.", False,
     lambda s, o, gf, ga: _safe((_g(s, "duel_won") - _g(s, "aerial_won")) * 100,
                                (_g(s, "duel_won") - _g(s, "aerial_won"))
                                + (_g(s, "duel_lost") - _g(s, "aerial_lost")))),
    ("Take-ons stopped %", "Engagement",
     "Share of opponent dribbles that failed — 1v1 defending.", False,
     lambda s, o, gf, ga: _safe((_g(o, "total_contest") - _g(o, "won_contest")) * 100,
                                _g(o, "total_contest"))),
    ("Tackle success %", "Engagement",
     "Tackles won as a share of tackles attempted.", False,
     lambda s, o, gf, ga: _safe(_g(s, "won_tackle") * 100, _g(s, "total_tackle"))),
    ("Aerial win %", "Engagement",
     "Included as a control: if this matches but ground duels differ, the gap is "
     "about engagement, not size.", False,
     lambda s, o, gf, ga: _safe(_g(s, "aerial_won") * 100,
                                _g(s, "aerial_won") + _g(s, "aerial_lost"))),

    # ── Value from limited possession ─────────────────────────────────
    ("Shots per 10% possession", "Value from the ball",
     "Attacking output relative to how much of the ball you actually had.", False,
     lambda s, o, gf, ga: _safe(_g(s, "total_scoring_att") * 10, _g(s, "possession_percentage"))),
    ("Box shots per 10 final-third entries", "Value from the ball",
     "Penetration. Coventry reached the final third MORE often than Hull and "
     "produced a fifth as many box shots.", False,
     lambda s, o, gf, ga: _safe(_g(s, "attempts_ibox") * 10, _g(s, "final_third_entries"))),
    ("Dangerous touch share", "Value from the ball",
     "Touches in the opposition box as a share of all touches.", False,
     lambda s, o, gf, ga: _safe(_g(s, "touches_in_opp_box") * 100, _g(s, "touches"))),
    ("Set-piece shot share", "Value from the ball",
     "Share of shots from set plays. A side that rarely holds the ball has to "
     "manufacture chances some other way.", False,
     lambda s, o, gf, ga: _safe(_g(s, "att_setpiece") * 100, _g(s, "total_scoring_att"))),
    ("Fast-break shot share", "Value from the ball",
     "Share of shots from transitions — the other route for a low-possession side.", False,
     lambda s, o, gf, ga: _safe(_g(s, "shot_fastbreak") * 100, _g(s, "total_scoring_att"))),

    # ── Territory & risk (included even where unflattering) ───────────
    ("Shots forced outside the box", "Territory & risk",
     "Share of opponent shots taken from distance. Hull currently rank POORLY "
     "here — their mechanism is smothering shots in the box, not preventing them.", False,
     lambda s, o, gf, ga: _safe(_g(o, "attempts_obox") * 100,
                                _g(o, "attempts_ibox") + _g(o, "attempts_obox"))),
    ("Opponent passes per shot allowed", "Territory & risk",
     "How much build-up the opposition needed to manufacture one attempt.", False,
     lambda s, o, gf, ga: _safe(_g(o, "total_pass"), _g(o, "total_scoring_att"))),
    ("Opponent box touches per shot", "Territory & risk",
     "How often opponents touched the ball in your box per shot they managed.", False,
     lambda s, o, gf, ga: _safe(_g(o, "touches_in_opp_box"), _g(o, "total_scoring_att"))),
    ("Cross completion allowed", "Territory & risk",
     "Share of opponent crosses that found a team-mate.", True,
     lambda s, o, gf, ga: _safe(_g(o, "accurate_cross") * 100, _g(o, "total_cross"))),
    ("Long-ball accuracy", "Territory & risk",
     "Whether a direct team's long passes actually stick. Hull currently rank "
     "poorly — worth showing rather than hiding.", False,
     lambda s, o, gf, ga: _safe(_g(s, "accurate_long_balls") * 100, _g(s, "total_long_balls"))),
    ("Errors leading to a shot per game", "Territory & risk",
     "Composure under sustained pressure.", True,
     lambda s, o, gf, ga: _g(s, "error_lead_to_shot")),
]

GROUPS = ["Chance suppression", "Engagement", "Value from the ball", "Territory & risk"]


def compute(season: str = "2627") -> dict:
    """Per-team values for every derived metric, aggregated over the season."""
    path = ROOT / "Newcache" / f"pl_{season}.json"
    if not path.exists():
        return {"season": season, "teams": {}, "metrics": [], "matches": 0}

    data = json.loads(path.read_text())
    # Sum the raw components across matches first, then take the ratio once —
    # averaging per-match ratios would weight a 4-shot game as heavily as a
    # 20-shot one.
    acc = {}
    for m in data.get("matches", []):
        for side, other in (("home", "away"), ("away", "home")):
            tid = m.get(f"{side}Id")
            own, opp = m["stats"].get(side) or {}, m["stats"].get(other) or {}
            if not tid or not own:
                continue
            a = acc.setdefault(tid, {"own": {}, "opp": {}, "gf": 0, "ga": 0, "games": 0})
            for k, v in own.items():
                a["own"][k] = a["own"].get(k, 0.0) + float(v or 0)
            for k, v in opp.items():
                a["opp"][k] = a["opp"].get(k, 0.0) + float(v or 0)
            a["gf"] += int(m.get(f"{side}Goals") or 0)
            a["ga"] += int(m.get(f"{other}Goals") or 0)
            a["games"] += 1

    teams = {}
    for tid, a in acc.items():
        vals = {}
        for name, _grp, _desc, _low, fn in METRICS:
            try:
                v = fn(a["own"], a["opp"], a["gf"], a["ga"])
            except Exception:
                v = None
            if v is not None and name == "Errors leading to a shot per game":
                v = v / max(a["games"], 1)
            vals[name] = None if v is None else round(v, 2)
        teams[tid] = {"games": a["games"], "values": vals}

    return {
        "season":  season,
        "matches": data.get("count", 0),
        "teams":   teams,
        "groups":  GROUPS,
        "metrics": [{"name": n, "group": g, "desc": d, "lowerBetter": low}
                    for n, g, d, low, _ in METRICS],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Derived style metrics")
    ap.add_argument("--season", default="2627")
    args = ap.parse_args()
    d = compute(args.season)
    ids = sorted(d["teams"])
    print(f"{d['matches']} matches · {len(ids)} teams\n")
    for grp in d["groups"]:
        print(f"── {grp} ──")
        for m in d["metrics"]:
            if m["group"] != grp:
                continue
            row = [(t, d["teams"][t]["values"][m["name"]]) for t in ids
                   if d["teams"][t]["values"][m["name"]] is not None]
            row.sort(key=lambda x: x[1], reverse=not m["lowerBetter"])
            arrow = "↓" if m["lowerBetter"] else "↑"
            print(f"  {m['name']:<36}{arrow} " +
                  "  ".join(f"{t}:{v}" for t, v in row))
        print()
