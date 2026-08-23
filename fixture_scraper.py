"""
fixture_scraper.py
Fetches the official 2026/27 Premier League fixture list from the PL
Pulse API (footballapi.pulselive.com) — the same source used by the
stats scraper.

Returns 380 [{homeId, awayId}] dicts ordered by gameweek + kickoff,
or raises FixturesNotAvailable when the schedule hasn't been released.

Caching strategy
────────────────
• Partial cache (< 380 fixtures): retried every 24 h until the full
  schedule is published.
• Full cache (380 fixtures): never expires — the fixture calendar does
  not change once released.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT       = Path(__file__).parent
CACHE_DIR  = ROOT / "cache"
FIX_CACHE  = CACHE_DIR / "fixtures_2627.json"

CACHE_TTL         = timedelta(hours=24)   # partial cache — retry daily
FULL_CACHE_TTL    = timedelta(days=7)     # full cache — re-check weekly for fixture changes
FULL_COUNT   = 380
PULSE_BASE   = "https://footballapi.pulselive.com/football"
PULSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
    "Origin":          "https://www.premierleague.com",
}
PL_COMP_ID  = 1
TARGET_YEAR = 2026


class FixturesNotAvailable(RuntimeError):
    """Raised when the official fixture list has not yet been published."""


# ── Team name fragments → internal ID (longer/more-specific first) ──────
# Match is case-insensitive substring; first hit wins.
_NAME_MAP = [
    ("manchester city",   "mancity"),
    ("manchester united", "manutd"),
    ("nottingham",        "nottmforest"),
    ("aston villa",       "astonvilla"),
    ("crystal palace",    "crystalpalace"),
    ("hull",              "hull"),
    ("ipswich",           "ipswich"),
    ("coventry",          "coventry"),
    ("arsenal",           "arsenal"),
    ("bournemouth",       "bournemouth"),
    ("brentford",         "brentford"),
    ("brighton",          "brighton"),
    ("chelsea",           "chelsea"),
    ("everton",           "everton"),
    ("fulham",            "fulham"),
    ("leeds",             "leeds"),
    ("liverpool",         "liverpool"),
    ("newcastle",         "newcastle"),
    ("sunderland",        "sunderland"),
    ("tottenham",         "tottenham"),
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _pulse_get(path: str, **params) -> dict:
    resp = requests.get(
        f"{PULSE_BASE}/{path}", headers=PULSE_HEADERS, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _season_start_year(label: str) -> int:
    # Label format varies: "2025/26" (year-first) vs.
    # "English Premier League Season 2026/2027" (year embedded). Search
    # anywhere for the first 4-digit year rather than anchoring to the start.
    m = re.search(r"(\d{4})", label or "")
    return int(m.group(1)) if m else 0


def _name_to_id(name: str) -> str | None:
    lower = (name or "").lower()
    for fragment, internal_id in _NAME_MAP:
        if fragment in lower:
            return internal_id
    return None


def _team_name(team_obj: dict) -> str:
    """Extract the best available name string from a Pulse team object."""
    return (
        team_obj.get("name")
        or team_obj.get("shortName")
        or team_obj.get("club", {}).get("name")
        or ""
    )


def _is_home(entry: dict, index: int) -> bool:
    """Determine whether a teams-array entry is the home side.

    Pulse fixture objects carry no explicit home/away marker on each entry —
    the convention is positional: the first team listed is home, the second
    is away. We still honour an explicit marker when one is present.
    """
    if "isHome" in entry:
        return bool(entry["isHome"])
    loc = (entry.get("teamLocation") or entry.get("location") or "").lower()
    if loc in ("home", "away"):
        return loc == "home"
    return index == 0


# ── Core ─────────────────────────────────────────────────────────────────

def _find_season_id() -> int:
    data = _pulse_get(f"competitions/{PL_COMP_ID}/compseasons", pageSize=100)
    for s in data.get("content", []):
        if _season_start_year(s.get("label", "")) == TARGET_YEAR:
            return int(s["id"])
    raise FixturesNotAvailable(
        "The 2026/27 Premier League season is not yet registered in the "
        "Pulse API — fixture list will appear once the season is set up."
    )


def _fetch_all(season_id: int) -> list[dict]:
    """Page through the fixtures endpoint and return all raw fixture dicts."""
    fixtures, page = [], 0
    while True:
        data = _pulse_get(
            "fixtures",
            comps=PL_COMP_ID,
            compSeasons=season_id,
            page=page,
            pageSize=100,
            sort="asc",
        )
        content = data.get("content", [])
        if not content:
            break
        fixtures.extend(content)
        total = data.get("pageInfo", {}).get("numEntries", 0)
        if len(fixtures) >= total:
            break
        page += 1
    return fixtures


def _parse(raw: list[dict]) -> list[dict]:
    """
    Convert raw Pulse fixture objects to [{homeId, awayId, gw, kickoff}].
    Fixtures where either team is outside our 2026/27 roster are dropped.
    """
    out = []
    for fix in raw:
        teams = fix.get("teams", [])
        home_id = away_id = None

        for index, entry in enumerate(teams):
            team_obj = entry.get("team", {})
            name = _team_name(team_obj)
            tid  = _name_to_id(name)
            if tid is None:
                continue
            if _is_home(entry, index):
                home_id = tid
            else:
                away_id = tid

        if not home_id or not away_id:
            continue

        gw      = (fix.get("gameweek") or {}).get("gameweek", 99)
        millis  = (fix.get("kickoff")  or {}).get("millis", 0) or 0
        out.append({"homeId": home_id, "awayId": away_id, "gw": gw, "ms": millis})

    return out


def fetch(force: bool = False) -> list[dict]:
    """
    Return the official 2026/27 fixture list as 380 [{homeId, awayId}] dicts
    ordered by gameweek then kickoff time.

    Raises FixturesNotAvailable when the Pulse API does not yet have the
    complete fixture schedule.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if FIX_CACHE.exists() and not force:
        try:
            cached = json.loads(FIX_CACHE.read_text())
            age    = datetime.now() - datetime.fromtimestamp(FIX_CACHE.stat().st_mtime)
            count  = len(cached.get("fixtures", []))
            # Full set: honour the longer TTL so modifications are caught weekly
            if count == FULL_COUNT and age < FULL_CACHE_TTL:
                print("[fixture_scraper] Using cached official fixtures.", file=sys.stderr)
                return cached["fixtures"]
        except (json.JSONDecodeError, KeyError):
            pass

        # Partial / empty cache — honour the shorter daily TTL before retrying
        age = datetime.now() - datetime.fromtimestamp(FIX_CACHE.stat().st_mtime)
        if age < CACHE_TTL:
            raise FixturesNotAvailable(
                "Official fixtures not yet published; cached partial result is still fresh."
            )

    print("[fixture_scraper] Fetching 2026/27 fixtures from Pulse API…", file=sys.stderr)

    season_id = _find_season_id()
    raw       = _fetch_all(season_id)
    parsed    = _parse(raw)

    if not parsed:
        raise FixturesNotAvailable(
            "Pulse API returned no fixtures for 2026/27 yet."
        )

    parsed.sort(key=lambda f: (f["gw"], f["ms"]))
    fixtures = [{"homeId": f["homeId"], "awayId": f["awayId"]} for f in parsed]

    # Save cache (partial or full)
    FIX_CACHE.write_text(json.dumps({
        "fetched":  datetime.now().isoformat(),
        "seasonId": season_id,
        "count":    len(fixtures),
        "fixtures": fixtures,
    }, indent=2))

    if len(fixtures) < FULL_COUNT:
        raise FixturesNotAvailable(
            f"Only {len(fixtures)}/{FULL_COUNT} fixtures published so far — "
            "waiting for the full release."
        )

    print(f"[fixture_scraper] Locked in {len(fixtures)} official fixtures.", file=sys.stderr)
    return fixtures


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch official 2026/27 PL fixture list")
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    try:
        result = fetch(force=args.force)
        print(json.dumps(result, indent=2))
        print(f"\n{len(result)} fixtures.", file=sys.stderr)
    except FixturesNotAvailable as exc:
        print(f"Not available: {exc}", file=sys.stderr)
        sys.exit(1)
