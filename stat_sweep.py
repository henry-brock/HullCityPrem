"""
stat_sweep.py — search every Opta stat in Newcache for attack/defence signal.

Two stages, deliberately separated so stat selection never sees the test data:

  SCREEN    fit on 2324, rank stats by how well they predict 2425 team goals.
  VALIDATE  take the winners, fit on 2324+2425, and score real match-level
            log-loss on 2526 via prior_backtest's harness.

2526 is untouched during screening, so the validation number is honest.

Three relations are tested per stat, because a defensive prior can come from
either side of the ball:
  atk      own stat      → own goals scored
  def(own) own stat      → own goals conceded   (e.g. defensive_actions)
  def(opp) opponent stat → own goals conceded   (e.g. shots faced)
"""

import argparse
from collections import defaultdict

from prior_backtest import (build_priors_oppadj, evaluate, load, _ols)
from newcache_probe import _corr, _mean

MIN_GAMES = 20


def aggregates(seasons):
    """
    (team, season) → {stat: per-game mean, ...} for own and opponent stats.

    Opta OMITS a stat when its value is zero, so an absent key means 0 — not
    "unknown". Averaging only over matches where the key appears would inflate
    every sparse stat (att_sv_low_centre appears in ~66% of matches, so
    skipping the blanks overstates it by half). Values are therefore
    zero-filled across a fixed per-season vocabulary.

    A stat that is genuinely not collected in a season (defensive_actions
    before 2025/26) zero-fills to a constant, giving it no variance and so no
    spurious correlation; `stat_coverage` screens those out explicitly.
    """
    matches = [(s, m) for s in seasons for m in load(s)]
    vocab   = set()
    for _s, m in matches:
        for side in ("home", "away"):
            vocab |= set(m["stats"][side] or {})

    own = defaultdict(lambda: defaultdict(list))
    opp = defaultdict(lambda: defaultdict(list))
    gf  = defaultdict(list)
    ga  = defaultdict(list)

    for season, m in matches:
        for side, other, g_for, g_against in (
                ("home", "away", m["homeGoals"], m["awayGoals"]),
                ("away", "home", m["awayGoals"], m["homeGoals"])):
            key    = (m[f"{side}Name"], season)
            s_own  = m["stats"][side]  or {}
            s_opp  = m["stats"][other] or {}
            for k in vocab:
                own[key][k].append(float(s_own.get(k) or 0.0))
                opp[key][k].append(float(s_opp.get(k) or 0.0))
            gf[key].append(float(g_for))
            ga[key].append(float(g_against))

    out = {}
    for key in gf:
        if len(gf[key]) < MIN_GAMES:
            continue
        out[key] = {
            "own": {k: _mean(v) for k, v in own[key].items()},
            "opp": {k: _mean(v) for k, v in opp[key].items()},
            "gf":  _mean(gf[key]),
            "ga":  _mean(ga[key]),
        }
    return out


def stat_coverage(seasons):
    """Per-MATCH presence share, used to drop stats a season never collected."""
    matches = [m for s in seasons for m in load(s)]
    counts  = defaultdict(int)
    total   = 0
    for m in matches:
        for side in ("home", "away"):
            block = m["stats"][side] or {}
            if not block:
                continue
            total += 1
            for k in block:
                counts[k] += 1
    return {k: c / total for k, c in counts.items()} if total else {}


def common_stats(seasons, min_share=0.20):
    """
    Stats the season actually collects. The bar is low on purpose: a rare
    event like att_sv_low_centre is legitimately absent from most matches, so
    a high bar would wrongly discard it. What must be excluded is a stat that
    is essentially never recorded (defensive_actions at 0-1% before 25/26).
    """
    return {k for k, share in stat_coverage(seasons).items() if share >= min_share}


def by_team(agg):
    """Collapse (team, season) → team, averaging across seasons."""
    grouped = defaultdict(list)
    for (team, _season), v in agg.items():
        grouped[team].append(v)
    out = {}
    for team, vs in grouped.items():
        keys_own = set().union(*[set(v["own"]) for v in vs])
        keys_opp = set().union(*[set(v["opp"]) for v in vs])
        out[team] = {
            "own": {k: _mean([v["own"][k] for v in vs if k in v["own"]]) for k in keys_own},
            "opp": {k: _mean([v["opp"][k] for v in vs if k in v["opp"]]) for k in keys_opp},
            "gf":  _mean([v["gf"] for v in vs]),
            "ga":  _mean([v["ga"] for v in vs]),
        }
    return out


def screen(train_seasons, probe_season):
    """Rank stats by corr(train stat → probe-season goals). No test data used."""
    tr = by_team(aggregates(train_seasons))
    pr = by_team(aggregates([probe_season]))
    teams = sorted(set(tr) & set(pr))
    stats = common_stats(train_seasons) & common_stats([probe_season])

    rows = []
    for stat in sorted(stats):
        xs_own = [tr[t]["own"].get(stat, 0.0) for t in teams]
        xs_opp = [tr[t]["opp"].get(stat, 0.0) for t in teams]
        ygf    = [pr[t]["gf"] for t in teams]
        yga    = [pr[t]["ga"] for t in teams]
        rows.append((stat,
                     _corr(xs_own, ygf),    # attack
                     _corr(xs_own, yga),    # defence from own actions
                     _corr(xs_opp, yga)))   # defence from what opponents did
    return rows, len(teams)


def build_from_stats(train_agg, atk_spec, def_spec):
    """
    Fit goals ~ stat on the training teams and return {team: (atk, def)}.
    Specs are (side, stat) where side is 'own' or 'opp'.
    """
    teams = sorted(train_agg)
    out   = {}
    for target, spec, key in (("gf", atk_spec, 0), ("ga", def_spec, 1)):
        side, stat = spec
        X = [[train_agg[t][side].get(stat, 0.0)] for t in teams]
        y = [train_agg[t][target] for t in teams]
        w = _ols(X, y)
        for t in teams:
            v = max(0.15, w[0] * train_agg[t][side].get(stat, 0.0) + w[-1])
            cur = out.get(t, [1.4, 1.4])
            cur[key] = v
            out[t] = cur
    # rescale each side to the league mean, as the model's Poisson expects
    ma = _mean([v[0] for v in out.values()]) or 1.0
    md = _mean([v[1] for v in out.values()]) or 1.0
    return {t: (max(0.15, v[0] * 1.40 / ma), max(0.15, v[1] * 1.40 / md))
            for t, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2324,2425")
    ap.add_argument("--test",  default="2526")
    ap.add_argument("--probe", default="2425", help="screening target inside train")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    train = [s.strip() for s in args.train.split(",")]
    screen_train = [s for s in train if s != args.probe] or [train[0]]

    rows, n_teams = screen(screen_train, args.probe)
    print(f"SCREEN — fit {'+'.join(screen_train)} → predict {args.probe} goals "
          f"({n_teams} teams, {len(rows)} stats with full coverage)\n")

    atk = sorted(rows, key=lambda r: -abs(r[1]))[:args.top]
    print(f"{'top ATTACK stats':34}{'r vs goals-for':>16}")
    for s, a, _d1, _d2 in atk:
        print(f"  {s:32}{a:>16.3f}")

    dfn = sorted(rows, key=lambda r: -max(abs(r[2]), abs(r[3])))[:args.top]
    print(f"\n{'top DEFENCE stats':34}{'own→GA':>10}{'opp→GA':>10}")
    for s, _a, d_own, d_opp in dfn:
        print(f"  {s:32}{d_own:>10.3f}{d_opp:>10.3f}")

    # ── VALIDATE on the untouched test season ──────────────────────────
    print(f"\n\nVALIDATE — fit {'+'.join(train)} → score {args.test} match log-loss")
    test_matches = [m for s in [args.test] for m in load(s)]
    train_agg    = by_team(aggregates(train))

    ll, rps, _ = evaluate({}, test_matches)
    print(f"\n{'priors':46}{'log-loss':>10}{'RPS':>9}")
    print("-" * 65)
    print(f"{'neutral (all 1.40) — floor':46}{ll:>10.4f}{rps:>9.4f}")

    prod = build_priors_oppadj(train, "goals")
    ll_p, rps_p, _ = evaluate(prod, test_matches)
    print(f"{'opp-adj goals (ratings.py) ★ PRODUCTION':46}{ll_p:>10.4f}{rps_p:>9.4f}")

    best_atk = ("own", atk[0][0])
    d0 = dfn[0]
    best_def = ("own", d0[0]) if abs(d0[2]) >= abs(d0[3]) else ("opp", d0[0])

    combos = [
        (f"atk={best_atk[1]} / def={best_def[0]}:{best_def[1]}", best_atk, best_def),
        (f"atk={best_atk[1]} / def=opp:total_scoring_att", best_atk, ("opp", "total_scoring_att")),
        ("atk=own:total_scoring_att / def=opp:total_scoring_att",
         ("own", "total_scoring_att"), ("opp", "total_scoring_att")),
    ]
    for label, a_spec, d_spec in combos:
        priors = build_from_stats(train_agg, a_spec, d_spec)
        ll_c, rps_c, _ = evaluate(priors, test_matches)
        print(f"{label:46}{ll_c:>10.4f}{rps_c:>9.4f}")

    print("\n★ production. Screening used only "
          f"{'+'.join(screen_train)}→{args.probe}; {args.test} was untouched.")


if __name__ == "__main__":
    main()
