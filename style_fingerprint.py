"""
style_fingerprint.py — playing-style profile for each 2026/27 team.

Reflects the LIVE season. A team's fingerprint is built from its actual
2026/27 matches as soon as it has played; teams yet to kick off fall back to
last season's basis so the table is never empty, and every row declares which
source it used. Previously the whole tab was backward-looking — retained sides
on 2025/26 Premier League data and promoted sides on Championship data — which
described teams as they were, not as they are.

Style axes
  possession   — share of the ball
  passAcc      — pass completion %
  directness   — long balls as a share of all passes
  crosses      — crosses per game
  tempo        — passes per game
  ppda         — opponent passes allowed per defensive action (LOW = presses hard)

Labels
  identity — Possession / Direct / Counter / Balanced
  press    — High press / Mid block / Low block (from PPDA, live data only:
             Opta did not record PPDA before 2025/26)
"""

import json
from pathlib import Path

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"
NEWCACHE = ROOT / "Newcache"

RETAINED = {  # ESPN name → internal id (Premier League cache)
    "Arsenal": "arsenal", "Aston Villa": "astonvilla",
    "AFC Bournemouth": "bournemouth", "Brentford": "brentford",
    "Brighton & Hove Albion": "brighton", "Chelsea": "chelsea",
    "Crystal Palace": "crystalpalace", "Everton": "everton",
    "Fulham": "fulham", "Leeds United": "leeds", "Liverpool": "liverpool",
    "Manchester City": "mancity", "Manchester United": "manutd",
    "Newcastle United": "newcastle", "Nottingham Forest": "nottmforest",
    "Sunderland": "sunderland", "Tottenham Hotspur": "tottenham",
}
PROMOTED = {  # ESPN name → internal id (Championship cache)
    "Hull City": "hull", "Ipswich Town": "ipswich", "Coventry City": "coventry",
}


def _label(poss, direct, direct_avg):
    if poss >= 53:
        return "Possession"
    if poss <= 47 and direct >= direct_avg * 1.12:
        return "Direct"
    if poss <= 47:
        return "Counter"
    return "Balanced"


def _press_label(ppda, ppda_avg):
    """PPDA is inverted — a LOW value means more defensive actions per
    opponent pass, i.e. a more aggressive press."""
    if ppda is None or not ppda_avg:
        return None
    if ppda <= ppda_avg * 0.85:
        return "High press"
    if ppda >= ppda_avg * 1.15:
        return "Low block"
    return "Mid block"


def _g(block, name):
    v = (block or {}).get(name)
    return 0.0 if v is None else float(v)


def _live_rows(season: str) -> dict:
    """Per-team style axes from this season's actual matches."""
    path = NEWCACHE / f"pl_{season}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())

    acc = {}
    for m in data.get("matches", []):
        for side in ("home", "away"):
            tid = m.get(f"{side}Id")
            own = m["stats"].get(side) or {}
            if not tid or not own:
                continue
            a = acc.setdefault(tid, {"g": 0, "poss": 0.0, "pass": 0.0, "acc": 0.0,
                                     "long": 0.0, "cross": 0.0, "ppda": 0.0, "ppdaN": 0})
            a["g"]     += 1
            a["poss"]  += _g(own, "possession_percentage")
            a["pass"]  += _g(own, "total_pass")
            a["acc"]   += _g(own, "accurate_pass")
            a["long"]  += _g(own, "total_long_balls")
            a["cross"] += _g(own, "total_cross")
            if own.get("ppda") is not None:
                a["ppda"] += float(own["ppda"]); a["ppdaN"] += 1

    out = {}
    for tid, a in acc.items():
        g = a["g"] or 1
        passes = a["pass"] or 1
        out[tid] = {
            "id": tid,
            "poss":       round(a["poss"] / g, 1),
            "passAcc":    round(a["acc"] / passes * 100, 1),
            "directness": round(a["long"] / passes * 100, 1),
            "crosses":    round(a["cross"] / g, 1),
            "tempo":      round(passes / g),
            "ppda":       round(a["ppda"] / a["ppdaN"], 1) if a["ppdaN"] else None,
            "games":      a["g"],
            "source":     "live",
        }
    return out


def _fallback_rows(cache_name, name_map, games, source):
    path = CACHE / cache_name
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for t in data.get("teams", []):
        tid = name_map.get(t["team"])
        if not tid:
            continue
        s = t["stats"]
        passes = s.get("totalPasses", 0) or 1
        out[tid] = {
            "id": tid,
            "poss":       round(s.get("possessionPct", 0), 1),
            "passAcc":    round((s.get("passPct", 0) or 0) * 100, 1),
            "directness": round(s.get("totalLongBalls", 0) / passes * 100, 1),
            "crosses":    round(s.get("totalCrosses", 0) / games, 1),
            "tempo":      round(passes / games),
            "ppda":       None,          # not recorded before 2025/26
            "games":      0,
            "source":     source,
        }
    return out


def fingerprints(season: str = "2627"):
    """Live-season fingerprints, backfilled from last season where needed."""
    rows = {}
    rows.update(_fallback_rows("espn_teamstats_eng.1_2526.json", RETAINED, 38, "2025/26 PL"))
    rows.update(_fallback_rows("espn_teamstats_eng.2_2526.json", PROMOTED, 46, "2025/26 Champ"))
    # Live data wins wherever it exists.
    rows.update(_live_rows(season))

    out = list(rows.values())
    if not out:
        return []
    direct_avg = sum(r["directness"] for r in out) / len(out)
    ppdas = [r["ppda"] for r in out if r["ppda"] is not None]
    ppda_avg = sum(ppdas) / len(ppdas) if ppdas else None

    for r in out:
        r["promoted"] = r["id"] in PROMOTED.values()
        r["style"] = _label(r["poss"], r["directness"], direct_avg)
        r["press"] = _press_label(r["ppda"], ppda_avg)
    out.sort(key=lambda r: -r["poss"])
    return out


if __name__ == "__main__":
    rows = fingerprints()
    live = sum(1 for r in rows if r["source"] == "live")
    print(f"{'team':<14}{'poss':>6}{'pass%':>7}{'direct%':>8}{'cross/g':>8}{'ppda':>7}"
          f"  {'style':<11}{'press':<11}source")
    for r in rows:
        tag = "*" if r["promoted"] else " "
        print(f"{r['id']:<13}{tag}{r['poss']:>6}{r['passAcc']:>7}{r['directness']:>8}"
              f"{r['crosses']:>8}{(r['ppda'] if r['ppda'] is not None else '—'):>7}"
              f"  {r['style']:<11}{(r['press'] or '—'):<11}{r['source']}")
    print(f"\n{live}/{len(rows)} teams on live 2026/27 data; the rest fall back to "
          f"last season until they play.")
