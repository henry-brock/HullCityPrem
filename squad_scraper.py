"""
Scraper for Hull City squad data from the ESPN API.
Returns JSON directly — no HTML parsing or API key required.

Workflow:
  1. Tries the Premier League endpoint first (for when Hull are in the PL)
  2. Falls back to the Championship endpoint
  3. Caches parsed JSON to cache/hull_squad.json (24-hour TTL)
  4. force=True bypasses cache — use during a transfer window

ESPN team ID for Hull City: 306
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / "cache"
SQUAD_CACHE = CACHE_DIR / "hull_squad.json"

CACHE_TTL   = timedelta(hours=24)
HULL_ID     = 306

_BASE    = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_LEAGUES = ["eng.1", "eng.2"]          # PL first (Hull promoted for 2026/27); Championship fallback
_URL     = f"{_BASE}/{{league}}/teams/{HULL_ID}/roster"

# ESPN position abbreviation → our four-value position code
_POS = {
    "G":   "GK", "GK":  "GK",
    "D":   "DEF", "DF": "DEF", "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "WB":  "DEF", "LWB": "DEF", "RWB": "DEF",
    "M":   "MID", "MF": "MID", "CM": "MID", "DM": "MID", "AM": "MID",
    "CAM": "MID", "CDM": "MID", "LM": "MID", "RM": "MID",
    "F":   "FWD", "FW": "FWD", "CF": "FWD", "ST": "FWD",
    "LW":  "FWD", "RW": "FWD", "SS": "FWD",
}


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) < CACHE_TTL


def _parse(athletes: list) -> list[dict]:
    players      = []
    seen_numbers = set()   # shirt numbers are unique per player

    for a in athletes:
        raw_pos  = a.get("position", {}).get("abbreviation", "")
        pos      = _POS.get(raw_pos.upper(), "MID")

        num_str  = (a.get("jersey") or "").strip()
        try:
            number = int(num_str)
        except ValueError:
            number = None

        # Guard: shirt numbers must be unique
        if number is not None:
            if number in seen_numbers:
                continue
            seen_numbers.add(number)

        name = (a.get("fullName") or a.get("displayName") or "").strip()
        if not name:
            continue

        players.append({
            "id":     f"p_espn_{a.get('id', len(players))}",
            "name":   name,
            "number": number,
            "pos":    pos,
            "nationality": (a.get("citizenship") or "").strip(),
            "born":        (a.get("birthDate")   or "").strip(),
        })

    # Sort by shirt number; unnumbered players go to the end
    players.sort(key=lambda p: (p["number"] is None, p["number"] or 999, p["name"]))
    return players


def fetch(force: bool = False) -> list[dict]:
    """
    Return Hull City's squad sorted by shirt number.
    Tries the Premier League endpoint first, falls back to Championship.
    Results are cached for 24 hours; pass force=True to re-fetch.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if not force and _is_fresh(SQUAD_CACHE):
        print(f"[squad] Using cached squad: {SQUAD_CACHE}", file=sys.stderr)
        return json.loads(SQUAD_CACHE.read_text())["players"]

    athletes = []
    source   = None

    for league in _LEAGUES:
        url = _URL.format(league=league)
        print(f"[squad] Trying {url}", file=sys.stderr)
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data     = resp.json()
                athletes = data.get("athletes", [])
                if athletes:
                    source = url
                    print(f"[squad] Found {len(athletes)} players via {league}", file=sys.stderr)
                    break
        except Exception as exc:
            print(f"[squad] {league} failed: {exc}", file=sys.stderr)

    if not athletes:
        raise RuntimeError(
            "Hull City squad not found — ESPN may not list this team yet."
        )

    players = _parse(athletes)

    payload = {
        "team":    "Hull City",
        "fetched": datetime.now().isoformat(),
        "source":  source,
        "players": players,
    }
    SQUAD_CACHE.write_text(json.dumps(payload, indent=2))
    print(f"[squad] Cached {len(players)} players → {SQUAD_CACHE}", file=sys.stderr)
    return players


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Hull City squad from ESPN")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    squad = fetch(force=args.force)
    print(json.dumps(squad, indent=2))
    print(f"\n{len(squad)} player(s) returned.", file=sys.stderr)
