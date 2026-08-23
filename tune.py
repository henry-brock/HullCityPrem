"""
tune.py — search the match-model parameters for a better fit.

The objective is ON-FIELD, not the betting market: out-of-sample walk-forward
log-loss against real match results, expressed as SKILL % — the share of the
gap between the no-skill floor (league base rates) and the on-field ceiling
(oracle: best this model class can do knowing true season strengths) that the
model captures. 0% = no better than base rates, 100% = matches the oracle.

Levers searched: Dixon-Coles ρ, home/away spread, blend (cap, games, recency).
Apply the winners to index.html + backtest.py by hand.
"""

import backtest as bt

_MATCHES = bt._espn_matches()
_FLOOR = bt.skill_floor(_MATCHES)[0]
_CEIL = bt.oracle(_MATCHES)


def score(rho=-0.13, s=0.11, cap=0.70, games=12, recency=3):
    bt.BLEND_CAP, bt.BLEND_GAMES, bt.RECENCY_MAX = cap, games, recency
    ll, rps, _ = bt.walk_forward_espn(
        {"league_ha": True, "ha": 1 + s, "aa": 1 - s, "rho": rho})
    skill = (_FLOOR - ll) / (_FLOOR - _CEIL) * 100
    return ll, skill


def main():
    print(f"on-field floor {_FLOOR:.4f}  ceiling(oracle) {_CEIL:.4f}\n")
    base = score()
    print(f"current (ρ-0.13 s0.11 cap0.70 g12 r3): "
          f"log-loss {base[0]:.4f}  skill {base[1]:.1f}%\n")

    def sweep(name, vals, fn):
        best = None
        for v in vals:
            ll, sk = fn(v)
            if best is None or sk > best[2]:
                best = (v, ll, sk)
        print(f"{name:<14} best {best[0]}  log-loss {best[1]:.4f}  skill {best[2]:.1f}%")
        return best[0]

    # one-at-a-time coordinate search (cheap, stable) — maximise on-field skill
    r = sweep("rho", [-0.16, -0.13, -0.10, -0.07, -0.04, 0.0],
              lambda v: score(rho=v))
    s = sweep("spread", [0.08, 0.10, 0.11, 0.12, 0.14, 0.16],
              lambda v: score(rho=r, s=v))
    cap = sweep("blend cap", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
                lambda v: score(rho=r, s=s, cap=v))
    g = sweep("blend games", [6, 8, 10, 12, 15],
              lambda v: score(rho=r, s=s, cap=cap, games=v))
    rec = sweep("recency", [1, 2, 3, 5, 7],
                lambda v: score(rho=r, s=s, cap=cap, games=g, recency=v))

    final = score(rho=r, s=s, cap=cap, games=g, recency=rec)
    print(f"\nTUNED (ρ{r} s{s} cap{cap} g{g} r{rec}): "
          f"log-loss {final[0]:.4f}  skill {final[1]:.1f}%")
    print(f"gain vs current: {final[1]-base[1]:+.1f} skill points")


if __name__ == "__main__":
    main()
