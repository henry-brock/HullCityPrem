"""
possession_scraper.py — real possession & passing-style metrics for a
Championship season, from ESPN's core API (the football-data.co.uk feed has
no possession, and FBref/Sofascore are Cloudflare-walled).

ESPN core API (no key, same source family as squad_scraper's eng.2 feed):
  teams:  …/leagues/eng.2/seasons/{yr}/teams
  stats:  …/leagues/eng.2/seasons/{yr}/types/1/teams/{id}/statistics
The team statistics carry "Possession Percentage" plus passing volume /
accuracy — enough to characterise whether a side plays a controlling or a
reactive, direct style. Cached to cache/championship_possession_<code>.json.
"""

import json
import sys
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CORE      = "https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.2/seasons/{yr}"
UA        = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

# Season code (football-data style) → ESPN season year (start year).
CODE_TO_YEAR = {"2526": 2025, "2425": 2024, "2324": 2023, "2223": 2022, "2122": 2021}

# Wanted stats → output key (ESPN displayName, case-insensitive substring).
WANT = {
    "possession percentage": "poss",
    "total passes":          "passes",
    "accurate passes":       "accPasses",
    "pass percentage":       "passPct",
    "total long balls":      "longBalls",
}


def _get(url):
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch(code: str = "2526", force: bool = False) -> dict:
    cache = CACHE_DIR / f"championship_possession_{code}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    yr = CODE_TO_YEAR.get(code)
    if yr is None:
        raise ValueError(f"No ESPN season-year mapping for code {code!r}")
    CACHE_DIR.mkdir(exist_ok=True)
    base = CORE.format(yr=yr)

    team_refs = [it["$ref"] for it in _get(f"{base}/teams?limit=50").get("items", [])]
    teams = []
    for ref in team_refs:
        tid = ref.split("/teams/")[1].split("?")[0].rstrip("/")
        meta = _get(f"{base}/teams/{tid}".split("?")[0])
        name = meta.get("displayName") or meta.get("name") or tid
        stats = _get(f"{base}/types/1/teams/{tid}/statistics")
        row = {"id": tid, "team": name}
        for cat in stats.get("splits", {}).get("categories", []):
            for s in cat.get("stats", []):
                dn = (s.get("displayName") or "").lower()
                for needle, key in WANT.items():
                    if needle == dn:
                        row[key] = s.get("value")
        teams.append(row)
        print(f"[possession] {name}: {row.get('poss')}% possession", file=sys.stderr)

    teams.sort(key=lambda t: -(t.get("poss") or 0))
    avg = sum(t.get("poss") or 0 for t in teams) / len(teams)
    # ESPN only carries possession for the live season; older seasons return
    # zeros. Don't cache an empty result (it would poison future reads).
    if avg == 0:
        raise RuntimeError(
            f"ESPN has no possession data for season {code} "
            f"(only the current season is populated) — not caching.")
    payload = {"season": code, "source": "ESPN core API",
               "leagueAvgPoss": round(avg, 1), "teams": teams}
    cache.write_text(json.dumps(payload, indent=2))
    print(f"[possession] cached {len(teams)} teams → {cache}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Championship possession via ESPN")
    p.add_argument("--season", default="2526")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    data = fetch(args.season, force=args.force)
    print(f"\n══ Championship {args.season} possession & passing style "
          f"(ESPN) ══   league avg {data['leagueAvgPoss']}%\n")
    print(f"{'#':>3} {'team':<22}{'Poss%':>7}{'passes/g?':>11}{'pass%':>7}")
    promoted = {"Hull City", "Coventry City", "Ipswich Town"}
    for i, t in enumerate(data["teams"], 1):
        star = " ◄" if t["team"] in promoted else ""
        passes = t.get("passes")
        pass_acc = t.get("passPct")
        print(f"{i:>3} {t['team']:<22}{t.get('poss','—'):>7}"
              f"{passes if passes is not None else '—':>11}"
              f"{pass_acc if pass_acc is not None else '—':>7}{star}")
    print("\n('passes' is the ESPN season total; pass% = accuracy. Low possession "
          "+ low pass volume/accuracy = a direct, reactive style.)")
