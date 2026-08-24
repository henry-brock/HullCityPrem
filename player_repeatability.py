"""
player_repeatability.py — how much of each player metric carries from one
season to the next, and therefore how much weight it deserves in a rating.

Why
───
The player rater's concept weights are currently a judgement call. The
principled alternative — the same standard used for the team priors and for
the Sunderland question — is to weight a concept by how *repeatable* it is:
a metric that does not persist year to year describes what happened, not what
a player is, and should not drive a rating.

Two things this has to get right
────────────────────────────────
1. MATCH ON PLAYER ID, not name or name+team. Ids follow a player through a
   transfer; names collide and re-spell.

2. SEPARATE MOVERS FROM STAYERS. A player who changes club has his counting
   stats reshaped by the new side's style regardless of ability — move a
   full-back from a 60%-possession team to a 40% one and his touches,
   crosses and progressive passes fall off a cliff without his ability
   changing at all. Lumping movers in makes team-dependent metrics look less
   repeatable than they are, which would systematically under-weight passing
   and progression against intrinsic things like aerial duels. Stayers are
   the primary sample; movers are reported alongside to quantify the effect.

Counting stats are compared per-90 so minutes differences do not drive the
correlation; rate stats are compared as-is.
"""

import json
from pathlib import Path

from player_rater import CONCEPTS_GK, CONCEPTS_OUT, RATE_STATS
from newcache_probe import _corr, _mean

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"

MIN_MINUTES = 600     # in BOTH seasons, so a per-90 means something


def _load(season):
    p = CACHE / f"pl_players_{season}.json"
    if not p.exists():
        return {}
    return {pl["id"]: pl for pl in json.loads(p.read_text()).get("players", [])}


def _per90(stats, key):
    v = stats.get(key)
    if v is None:
        return None
    if key in RATE_STATS:
        return float(v)
    mins = stats.get("minutes") or 0
    return float(v) / mins * 90 if mins > 0 else None


def pairs(s1: str, s2: str):
    """Players present in both seasons with enough minutes in each."""
    a, b = _load(s1), _load(s2)
    out = []
    for pid, p1 in a.items():
        p2 = b.get(pid)
        if not p2:
            continue
        if (p1["stats"].get("minutes", 0) < MIN_MINUTES
                or p2["stats"].get("minutes", 0) < MIN_MINUTES):
            continue
        out.append((p1, p2, p1.get("team") == p2.get("team")))
    return out


def collected_in_both(sample, key, drop=0.5):
    """
    Was this stat actually recorded in BOTH seasons?

    Opta introduced several metrics partway through the archive — carries,
    progressive carries, final-third entries, shot-creating actions and
    pen-area entries are all near-absent in 2024/25 and universal in 2025/26.
    Correlating a real season against a column of zeros produces a low number
    that looks like "this doesn't repeat" when it means "this wasn't measured".
    A stat is rejected when its non-zero rate in the earlier season is less
    than `drop` of the later one; stats that are legitimately zero for many
    players (a defender's big chances scored) have similar rates in both and
    survive.
    """
    def rate(idx):
        vals = [(p[idx]["stats"].get(key) or 0) for p in sample]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0
    r1, r2 = rate(0), rate(1)
    if r2 == 0:
        return r1 > 0
    return (r1 / r2) >= drop


def stat_repeatability(sample, keys):
    """key → (year-over-year correlation, n). None when not comparable."""
    out = {}
    for key in keys:
        if not collected_in_both(sample, key):
            out[key] = (None, 0)          # not measured in both seasons
            continue
        xs, ys = [], []
        for p1, p2, _same in sample:
            v1, v2 = _per90(p1["stats"], key), _per90(p2["stats"], key)
            if v1 is not None and v2 is not None:
                xs.append(v1); ys.append(v2)
        out[key] = (_corr(xs, ys), len(xs)) if len(xs) >= 8 else (None, len(xs))
    return out


def concept_repeatability(sample, concepts):
    """Average the repeatability of a concept's member stats."""
    keys = sorted({k for c in concepts.values() for k in c["stats"]})
    per_stat = stat_repeatability(sample, keys)
    out = {}
    for name, c in concepts.items():
        usable = [k for k in c["stats"] if per_stat[k][0] is not None]
        vals = [per_stat[k][0] for k in usable]
        # None means "cannot be measured", which is NOT the same as "does not
        # repeat" — the caller must leave such a concept's weight alone rather
        # than zeroing it.
        out[name] = (_mean(vals), len(usable), len(c["stats"])) if vals else (None, 0, len(c["stats"]))
    return out, per_stat


def suggested_weights(concept_rep, current, scale=5.0):
    """
    Turn repeatability into weights on the same 0-5 scale the rater uses.

    Negative or missing correlations fall to zero — a concept that does not
    persist should not push a rating. Weights are scaled so the most
    repeatable concept sits at `scale`.
    """
    usable = {k: max(0.0, v[0]) for k, v in concept_rep.items() if v[0] is not None}
    if not usable:
        return {}
    top = max(usable.values()) or 1.0
    return {k: round(v / top * scale, 1) for k, v in usable.items()}


def main(s1="2425", s2="2526"):
    sample = pairs(s1, s2)
    if not sample:
        print(f"No player pairs between {s1} and {s2} — is both cache present?")
        return
    stay = [p for p in sample if p[2]]
    move = [p for p in sample if not p[2]]
    print(f"{len(sample)} players in both {s1} and {s2} with ≥{MIN_MINUTES} mins each")
    print(f"  same club: {len(stay)}   changed club: {len(move)}\n")

    for label, concepts in (("OUTFIELD", CONCEPTS_OUT), ("GOALKEEPER", CONCEPTS_GK)):
        pool = [p for p in stay if (p[1].get("pos") == "G") == (label == "GOALKEEPER")]
        pool_m = [p for p in move if (p[1].get("pos") == "G") == (label == "GOALKEEPER")]
        if len(pool) < 8:
            print(f"── {label}: only {len(pool)} same-club players, skipping\n")
            continue
        rep, per_stat = concept_repeatability(pool, concepts)
        rep_m, _ = concept_repeatability(pool_m, concepts) if len(pool_m) >= 8 else ({}, {})
        weights = suggested_weights(rep, None)

        print(f"── {label}  (n={len(pool)} same-club"
              + (f", {len(pool_m)} movers" if rep_m else "") + ")")
        print(f"   {'concept':<16}{'repeat r':>10}{'movers r':>10}{'stats':>8}{'suggested':>11}")
        for name in sorted(rep, key=lambda k: -(rep[k][0] if rep[k][0] is not None else -1)):
            r, used, tot = rep[name]
            rm = rep_m.get(name, (None,0,0))[0]
            note = "" if r is not None else "  not measurable in 24/25"
            cov  = f"{used}/{tot}"
            print(f"   {name:<16}{(f'{r:.3f}' if r is not None else '—'):>10}"
                  f"{(f'{rm:.3f}' if rm is not None else '—'):>10}"
                  f"{cov:>8}{(weights.get(name) if r is not None else '—'):>11}{note}")
        print()

    print("Most and least repeatable individual stats (same-club outfielders):")
    pool = [p for p in stay if p[1].get("pos") != "G"]
    keys = sorted({k for c in CONCEPTS_OUT.values() for k in c["stats"]})
    ps = stat_repeatability(pool, keys)
    ranked = sorted([(v[0], k, v[1]) for k, v in ps.items() if v[0] is not None], reverse=True)
    for r, k, n in ranked[:6]:
        print(f"   {k:<24}{r:+.3f}  (n={n})")
    print("   …")
    for r, k, n in ranked[-6:]:
        print(f"   {k:<24}{r:+.3f}  (n={n})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Year-over-year player stat repeatability")
    ap.add_argument("--from", dest="s1", default="2425")
    ap.add_argument("--to",   dest="s2", default="2526")
    a = ap.parse_args()
    main(a.s1, a.s2)
