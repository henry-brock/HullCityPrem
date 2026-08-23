"""
espn_scraper.py — cache full team season statistics from ESPN's core API
for the Premier League (eng.1) and Championship (eng.2).

Each team carries ~109 Opta-grade stats (possession, passing, shots, big
chances created, goal conversion, duels, tackles, interceptions, GK save%,
ratings, discipline …) across defensive / general / goalKeeping / offensive
categories. One request per team — small and fast — cached as
cache/espn_teamstats_<league>_<code>.json.

Player and game-level data are also available on this API (34-man rosters
with 107 stats each; per-match boxscores with key events) — not cached here
because of volume; add when needed.
"""

import json
import re
import sys
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
BASE      = "https://sports.core.api.espn.com/v2/sports/soccer/leagues/{lg}/seasons/{yr}"
UA        = {"User-Agent": "Mozilla/5.0 (X11) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

CODE_TO_YEAR = {"2526": 2025, "2425": 2024, "2324": 2023, "2223": 2022, "2122": 2021}
LEAGUES = {"eng.1": "Premier League", "eng.2": "Championship"}


def _get(url):
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch(league: str, code: str = "2526", force: bool = False) -> dict:
    cache = CACHE_DIR / f"espn_teamstats_{league}_{code}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    yr = CODE_TO_YEAR.get(code)
    if yr is None:
        raise ValueError(f"No ESPN year mapping for code {code!r}")
    CACHE_DIR.mkdir(exist_ok=True)
    base = BASE.format(lg=league, yr=yr)

    refs = [it["$ref"] for it in _get(f"{base}/teams?limit=50").get("items", [])]
    teams = []
    for ref in refs:
        tid = re.search(r"/teams/(\d+)", ref).group(1)
        meta = _get(f"{base}/teams/{tid}".split("?")[0])
        name = meta.get("displayName") or tid
        stats = {}
        cats = _get(f"{base}/types/1/teams/{tid}/statistics") \
            .get("splits", {}).get("categories", [])
        for c in cats:
            for s in c.get("stats", []):
                stats[s.get("name")] = s.get("value")
        teams.append({"id": tid, "team": name, "stats": stats})
        print(f"[espn] {league} {name}: {len(stats)} stats", file=sys.stderr)

    payload = {"league": league, "leagueName": LEAGUES.get(league, league),
               "season": code, "source": "ESPN core API", "teams": teams}
    cache.write_text(json.dumps(payload, indent=2))
    print(f"[espn] cached {len(teams)} teams → {cache}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cache ESPN team season stats")
    p.add_argument("--league", default="all", choices=["eng.1", "eng.2", "all"])
    p.add_argument("--season", default="2526")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    leagues = ["eng.1", "eng.2"] if args.league == "all" else [args.league]
    for lg in leagues:
        d = fetch(lg, args.season, force=args.force)
        nstats = len(d["teams"][0]["stats"]) if d["teams"] else 0
        print(f"{d['leagueName']:<15} {len(d['teams'])} teams × {nstats} stats "
              f"→ cache/espn_teamstats_{lg}_{args.season}.json")
