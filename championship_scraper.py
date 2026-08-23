"""
championship_scraper.py — build a "backlog" of a completed Championship
season and attach an underlying-performance (xG-proxy) layer.

football-data.co.uk's free feed has no possession or Opta xG, but it does
carry shots and shots-on-target for every match. From those we fit a
league-neutral expected-goals proxy and produce, per team:

    GF/GA   — goals for / against per game (results)
    xGF/xGA — expected goals for / against per game (process)
    over    — goals minus xG (finishing / variance / luck)

The point of the xG layer is that PROCESS travels across a promotion better
than RESULTS do: a team that out-finished its chances (e.g. Hull 2025/26 —
out-shot by its opponents yet 6th) should be rated on what it created, not
on a goal tally it is unlikely to repeat a division up. promoted_analyzer.py
consumes this module to scale Championship form to the Premier League.

This file also exports the shared data helpers (fetch_csv, parse_matches,
fit_xg, team_table) so the scaler uses exactly the same xG model.
"""

import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
URL       = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"


# ── raw data ─────────────────────────────────────────────────────────────
def fetch_csv(code: str, league: str) -> str:
    """Download (and cache) a season CSV. league = 'E0' (PL) or 'E1' (Champ)."""
    cache = CACHE_DIR / f"{league}_{code}.csv"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    CACHE_DIR.mkdir(exist_ok=True)
    url = URL.format(code=code, league=league)
    print(f"[championship] downloading {url}", file=sys.stderr)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    raw = resp.content.decode("utf-8-sig")
    cache.write_text(raw, encoding="utf-8")
    return raw


def parse_matches(raw: str) -> list[dict]:
    """Rows with a played result and shot data → flat match dicts."""
    out = []
    for r in csv.DictReader(io.StringIO(raw)):
        h = (r.get("HomeTeam") or "").strip()
        a = (r.get("AwayTeam") or "").strip()
        if not h or not a:
            continue
        try:
            m = {
                "home": h, "away": a, "date": (r.get("Date") or "").strip(),
                "hg": int(r["FTHG"]), "ag": int(r["FTAG"]),
                "hs": int(r["HS"]),   "as": int(r["AS"]),
                "hst": int(r["HST"]), "ast": int(r["AST"]),
            }
        except (KeyError, ValueError):
            continue
        out.append(m)
    return out


def _parse_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.min


def half_matches(matches: list[dict], half: str = "2nd") -> list[dict]:
    """Calendar split: return the later (or earlier) half of the season by
    date. 'full' returns everything. A global split keeps every team measured
    over the same window, which is what relative-to-league ratings need."""
    if half == "full":
        return matches
    ordered = sorted(matches, key=lambda m: _parse_date(m.get("date", "")))
    mid = len(ordered) // 2
    return ordered[mid:] if half == "2nd" else ordered[:mid]


# ── xG proxy ──────────────────────────────────────────────────────────────
def fit_xg(matches: list[dict]) -> tuple[float, float]:
    """
    Fit goals ≈ a·(shots on target) + b·(shots off target) by ordinary least
    squares with no intercept (no shot ⇒ no goal), pooled over every
    team-match. Returns (a, b): a ≈ conversion of an on-target shot, b the
    small residual value of an off-target shot. League-neutral by design so
    the same lens is applied to the Championship and the Premier League.
    """
    s_oo = s_of = s_ff = s_go = s_gf = 0.0
    for m in matches:
        for sot, sh, g in ((m["hst"], m["hs"], m["hg"]),
                           (m["ast"], m["as"], m["ag"])):
            off = max(0, sh - sot)
            s_oo += sot * sot
            s_of += sot * off
            s_ff += off * off
            s_go += g * sot
            s_gf += g * off
    det = s_oo * s_ff - s_of * s_of
    if abs(det) < 1e-9:
        return (s_go / s_oo if s_oo else 0.0), 0.0
    a = (s_go * s_ff - s_gf * s_of) / det
    b = (s_gf * s_oo - s_go * s_of) / det
    return a, b


def xg(sot: int, shots: int, a: float, b: float) -> float:
    return a * sot + b * max(0, shots - sot)


# ── season table with process metrics ─────────────────────────────────────
def team_table(matches: list[dict], a: float, b: float) -> dict:
    """team → gp, gf, ga, xgf, xga, pts (totals; divide by gp for per-game)."""
    t = {}
    def row(name):
        return t.setdefault(name, {"gp": 0, "gf": 0, "ga": 0,
                                   "xgf": 0.0, "xga": 0.0, "pts": 0})
    for m in matches:
        hx = xg(m["hst"], m["hs"], a, b)
        ax = xg(m["ast"], m["as"], a, b)
        H, A = row(m["home"]), row(m["away"])
        H["gp"] += 1; A["gp"] += 1
        H["gf"] += m["hg"]; H["ga"] += m["ag"]; H["xgf"] += hx; H["xga"] += ax
        A["gf"] += m["ag"]; A["ga"] += m["hg"]; A["xgf"] += ax; A["xga"] += hx
        if   m["hg"] > m["ag"]: H["pts"] += 3
        elif m["ag"] > m["hg"]: A["pts"] += 3
        else:                   H["pts"] += 1; A["pts"] += 1
    return t


def build_backlog(code: str = "2526") -> dict:
    """Full Championship season backlog with the xG layer; cached as JSON."""
    matches = parse_matches(fetch_csv(code, "E1"))
    a, b = fit_xg(matches)
    table = team_table(matches, a, b)

    rows = []
    for name, s in table.items():
        gp = s["gp"]
        rows.append({
            "team": name, "gp": gp, "pts": s["pts"],
            "gf": round(s["gf"] / gp, 2), "ga": round(s["ga"] / gp, 2),
            "xgf": round(s["xgf"] / gp, 2), "xga": round(s["xga"] / gp, 2),
            "xgd": round((s["xgf"] - s["xga"]) / gp, 2),
            "over": round((s["gf"] - s["xgf"]) / gp, 2),   # finishing/luck
            "overAg": round((s["ga"] - s["xga"]) / gp, 2),
        })
    rows.sort(key=lambda r: (-r["pts"], -(r["gf"] - r["ga"])))
    for i, r in enumerate(rows):
        r["pos"] = i + 1

    backlog = {
        "season": code, "league": "Championship",
        "xgModel": {"perShotOnTarget": round(a, 4), "perShotOffTarget": round(b, 4)},
        "table": rows,
        "matches": matches,
    }
    out = CACHE_DIR / f"championship_{code}.json"
    out.write_text(json.dumps(backlog, indent=2))
    print(f"[championship] backlog → {out}  "
          f"({len(matches)} matches, xG = {a:.3f}·SoT + {b:.3f}·off)",
          file=sys.stderr)
    return backlog


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a Championship season backlog")
    p.add_argument("--season", default="2526")
    args = p.parse_args()

    bl = build_backlog(args.season)
    print(f"\n══ Championship {args.season} — results vs underlying (xG proxy) ══")
    print(f"xG model: {bl['xgModel']['perShotOnTarget']:.3f}·SoT "
          f"+ {bl['xgModel']['perShotOffTarget']:.3f}·offTarget per shot\n")
    print(f"{'#':>3} {'team':<16}{'pts':>4}{'GF':>6}{'GA':>6}"
          f"{'xGF':>6}{'xGA':>6}{'xGD':>6}{'over':>7}")
    for r in bl["table"]:
        print(f"{r['pos']:>3} {r['team']:<16}{r['pts']:>4}"
              f"{r['gf']:>6}{r['ga']:>6}{r['xgf']:>6}{r['xga']:>6}"
              f"{r['xgd']:>6}{r['over']:>+7.2f}")
    print("\n'over' = goals − xG per game (＋ = out-finished its chances, "
          "likely unsustainable; − = under-finished).")
