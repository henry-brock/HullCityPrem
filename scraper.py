"""
Scraper for Premier League match data from football-data.co.uk.

Workflow:
  1. Downloads season CSV once and caches it to cache/<season>.csv
  2. Parses into structured JSON at cache/<season>.json
  3. Subsequent calls read from the cache — no internet hit
  4. Pass force=True to re-download (e.g. mid-season update)

Target season: 2026/27 (code "2627")
  football-data.co.uk publishes the CSV for a season once it begins.
  Until then fetch() raises SeasonNotAvailable with a clear message.

Source: https://www.football-data.co.uk  (free, no API key required)
"""

import csv
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"

# Base URL pattern — season code is e.g. "2526" for 2025/26
_BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

CACHE_TTL = timedelta(hours=6)   # re-download after 6 h when force=True

# ─── Custom exception ─────────────────────────────────────────────────
class SeasonNotAvailable(RuntimeError):
    """Raised when the season CSV has not yet been published."""


# ─── Team name mapping: football-data.co.uk → our app IDs ────────────
# All 20 clubs in the 2026/27 Premier League.
# Confirmed naming conventions from football-data.co.uk:
#   Hull (verified from 2016/17), Ipswich (from 2024/25),
#   Coventry (from Championship 2025/26 E1 file).
# Burnley, West Ham and Wolves (relegated 2025/26) are absent so their
# rows are silently skipped if importing historical 2025/26 data.
TEAM_MAP = {
    # ── Retained from 2025/26 ─────────────────────────────────────────
    "Arsenal":        "arsenal",
    "Aston Villa":    "astonvilla",
    "Bournemouth":    "bournemouth",
    "Brentford":      "brentford",
    "Brighton":       "brighton",
    "Chelsea":        "chelsea",
    "Crystal Palace": "crystalpalace",
    "Everton":        "everton",
    "Fulham":         "fulham",
    "Leeds":          "leeds",
    "Liverpool":      "liverpool",
    "Man City":       "mancity",
    "Man United":     "manutd",
    "Newcastle":      "newcastle",
    "Nott'm Forest":  "nottmforest",
    "Sunderland":     "sunderland",
    "Tottenham":      "tottenham",
    # ── Promoted from Championship 2025/26 ───────────────────────────
    "Hull":           "hull",
    "Ipswich":        "ipswich",
    "Coventry":       "coventry",
}


# ─── Helpers ──────────────────────────────────────────────────────────
def _int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _flt(val) -> float | None:
    try:
        return round(float(val), 3)
    except (TypeError, ValueError):
        return None


def _cache_paths(season: str):
    return CACHE_DIR / f"{season}_raw.csv", CACHE_DIR / f"{season}.json"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < CACHE_TTL


# ─── Core ─────────────────────────────────────────────────────────────
def fetch(season: str = "2627", force: bool = False) -> list[dict]:
    """
    Return a list of parsed match dicts for the given season code.

    Each dict contains the result plus available in-game stats.
    Teams not in the 2026/27 roster are filtered out automatically.

    Args:
        season:  Season code, e.g. "2627" for 2026/27.
        force:   Re-download the CSV even if a cached copy exists.

    Raises:
        SeasonNotAvailable: if the CSV for this season has not yet been
            published by football-data.co.uk (HTTP 404).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    csv_path, json_path = _cache_paths(season)

    # Return cached JSON if fresh enough
    if not force and _is_fresh(json_path):
        print(f"[scraper] Using cached data: {json_path}", file=sys.stderr)
        return json.loads(json_path.read_text())["matches"]

    # Download CSV
    url = _BASE_URL.format(code=season)
    print(f"[scraper] Downloading {url}", file=sys.stderr)
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise SeasonNotAvailable(
            f"Season {season[:2]}/{season[2:]} data is not yet available on "
            f"football-data.co.uk. It will appear once the season has started."
        )

    raw = resp.content.decode("utf-8-sig")   # strip BOM if present
    csv_path.write_text(raw, encoding="utf-8")

    # Parse
    matches = _parse(raw)

    # Save structured cache
    payload = {
        "season":  season,
        "fetched": datetime.now().isoformat(),
        "source":  url,
        "matches": matches,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[scraper] Cached {len(matches)} matches → {json_path}", file=sys.stderr)

    return matches


def _parse(raw_csv: str) -> list[dict]:
    matches = []
    reader  = csv.DictReader(io.StringIO(raw_csv))

    for row in reader:
        home_name = row.get("HomeTeam", "").strip()
        away_name = row.get("AwayTeam", "").strip()

        home_id = TEAM_MAP.get(home_name)
        away_id = TEAM_MAP.get(away_name)

        # Skip if either team is not in our 2026/27 roster
        if not home_id or not away_id:
            continue

        # Skip if result is missing (fixture not yet played)
        home_goals = _int(row.get("FTHG"))
        away_goals = _int(row.get("FTAG"))
        if home_goals is None or away_goals is None:
            continue

        matches.append({
            # ── Identity ──────────────────────────────
            "date":     row.get("Date", ""),
            "time":     row.get("Time", ""),
            "referee":  row.get("Referee", ""),
            "homeId":   home_id,
            "awayId":   away_id,
            # ── Full-time result ──────────────────────
            "homeGoals":  home_goals,
            "awayGoals":  away_goals,
            "ftResult":   row.get("FTR", ""),   # H / D / A
            # ── Half-time result ──────────────────────
            "htHomeGoals": _int(row.get("HTHG")),
            "htAwayGoals": _int(row.get("HTAG")),
            "htResult":    row.get("HTR", ""),  # H / D / A
            # ── Shots ─────────────────────────────────
            "homeShots":         _int(row.get("HS")),
            "awayShots":         _int(row.get("AS")),
            "homeShotsOnTarget": _int(row.get("HST")),
            "awayShotsOnTarget": _int(row.get("AST")),
            # ── Fouls & cards ─────────────────────────
            "homeFouls":  _int(row.get("HF")),
            "awayFouls":  _int(row.get("AF")),
            "homeYellow": _int(row.get("HY")),
            "awayYellow": _int(row.get("AY")),
            "homeRed":    _int(row.get("HR")),
            "awayRed":    _int(row.get("AR")),
            # ── Corners ───────────────────────────────
            "homeCorners": _int(row.get("HC")),
            "awayCorners": _int(row.get("AC")),
            # ── Betting odds (separate sub-object) ────
            # Implicit probabilities: p = 1/odds (before margin)
            # Useful for pre-match strength estimates in the predictor
            "odds": {
                # 1X2 — three major markets
                "b365":    {"h": _flt(row.get("B365H")),  "d": _flt(row.get("B365D")),  "a": _flt(row.get("B365A"))},
                "pinnacle": {"h": _flt(row.get("PSH")),   "d": _flt(row.get("PSD")),    "a": _flt(row.get("PSA"))},
                "max":     {"h": _flt(row.get("MaxH")),   "d": _flt(row.get("MaxD")),   "a": _flt(row.get("MaxA"))},
                "avg":     {"h": _flt(row.get("AvgH")),   "d": _flt(row.get("AvgD")),   "a": _flt(row.get("AvgA"))},
                # Over / Under 2.5 goals
                "ou25": {
                    "b365over":  _flt(row.get("B365>2.5")), "b365under": _flt(row.get("B365<2.5")),
                    "maxover":   _flt(row.get("Max>2.5")),  "maxunder":  _flt(row.get("Max<2.5")),
                    "avgover":   _flt(row.get("Avg>2.5")),  "avgunder":  _flt(row.get("Avg<2.5")),
                },
                # Asian Handicap
                "ah": {
                    "line":      _flt(row.get("AHh")),
                    "b365home":  _flt(row.get("B365AHH")), "b365away": _flt(row.get("B365AHA")),
                    "maxhome":   _flt(row.get("MaxAHH")),  "maxaway":  _flt(row.get("MaxAHA")),
                    "avghome":   _flt(row.get("AvgAHH")),  "avgaway":  _flt(row.get("AvgAHA")),
                },
            },
        })

    return matches


# ─── CLI helper ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cache PL match data")
    parser.add_argument("--season", default="2627", help="Season code, e.g. 2627")
    parser.add_argument("--force",  action="store_true", help="Force re-download")
    parser.add_argument("--filter", default="",
                        help="Filter output to rows containing this team ID")
    args = parser.parse_args()

    matches = fetch(season=args.season, force=args.force)

    if args.filter:
        matches = [m for m in matches if args.filter in (m["homeId"], m["awayId"])]

    print(json.dumps(matches, indent=2))
    print(f"\n{len(matches)} match(es) returned.", file=sys.stderr)
