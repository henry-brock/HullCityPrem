"""
Hull City player stats — Pulse API first, Wikipedia fallback.

Preferred source: the official Premier League Pulse API
(footballapi.pulselive.com).  It only carries Hull City once they are an
active Premier League club with published stats, so until the 2026/27 PL
season is live this path finds nothing and we fall back to Wikipedia.

Fallback source: the Wikipedia season article, whose player statistics
table carries:
  shirt number · position · name · apps per competition · goals per competition

Pulse workflow:
  1. Walk Premier League comp-seasons newest-first
  2. Find the newest season whose team list contains Hull City
  3. Pull that team's players and each player's season appearances/goals
  4. Require >= 10 players with appearances to count as usable

Wikipedia workflow:
  1. Tries recent Hull City season articles in reverse order until one is found
  2. Locates the player statistics wikitable (Apps/Goals/Player headers)
  3. Parses total and Championship-specific appearances and goals

Both cache to cache/hull_stats.json (24-hour TTL).  Apps format "30+2"
means 30 starts + 2 sub appearances.  Cards and assists are not exposed by
Wikipedia season articles.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / "cache"
STATS_CACHE = CACHE_DIR / "hull_stats.json"

CACHE_TTL = timedelta(hours=24)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# ── Pulse API (official Premier League) ───────────────────────────────
PULSE_BASE    = "https://footballapi.pulselive.com/football"
PULSE_HEADERS = {**HEADERS, "Origin": "https://www.premierleague.com"}
PL_COMP_ID    = 1            # Premier League competition id in the Pulse API
HULL_NAME     = "hull city"  # matched case-insensitively in the team list

# Pulse single-letter position → our four-value code
_PULSE_POS = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

# En-dash (–) encoded for Wikipedia URLs
_EN_DASH = "%E2%80%93"

def _wiki_url(start_year: int) -> str:
    short = str(start_year + 1)[-2:]
    return (
        f"https://en.wikipedia.org/wiki/"
        f"{start_year}{_EN_DASH}{short}_Hull_City_A.F.C._season"
    )


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) < CACHE_TTL


def _parse_apps(s: str) -> int:
    """'30+2' → 32, '45' → 45, '' → 0."""
    s = s.strip()
    try:
        parts = [int(p) for p in re.findall(r"\d+", s)]
        return sum(parts)
    except (ValueError, TypeError):
        return 0


# ═══════════════════════════════════════════════════════════════════════
# PULSE API  (preferred — used once Hull are live in the Premier League)
# ═══════════════════════════════════════════════════════════════════════
def _pulse_get(path: str, **params) -> dict:
    resp = requests.get(
        f"{PULSE_BASE}/{path}", headers=PULSE_HEADERS, params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _season_start_year(label: str) -> int:
    """'2016/17' → 2016; unparseable → 0."""
    m = re.match(r"\s*(\d{4})", label or "")
    return int(m.group(1)) if m else 0


def _pulse_seasons() -> list[int]:
    """Recent Premier League comp-season (id, label) pairs, newest first.

    The Pulse API serialises ids as floats (e.g. 777.0); they must be cast
    to int before being sent back as params or the API returns 400.

    Only current/upcoming seasons are kept: Hull last played in the PL in
    2016/17, and that decade-old squad must never override the current
    Championship/Wikipedia proxy.  The threshold tracks the calendar so the
    live 2026/27 season is picked up automatically once published.
    """
    cutoff  = datetime.now().year - 1
    data    = _pulse_get(f"competitions/{PL_COMP_ID}/compseasons", pageSize=100)
    seasons = [
        (int(s["id"]), s.get("label", str(int(s["id"]))))
        for s in data.get("content", [])
        if s.get("id") is not None
        and _season_start_year(s.get("label", "")) >= cutoff
    ]
    seasons.sort(key=lambda s: s[0], reverse=True)
    return seasons


def _pulse_find_hull(season_id: int):
    """Return Hull City's (int) Pulse team id within a comp-season, or None."""
    data = _pulse_get(
        "teams", pageSize=100, compSeasons=season_id, comps=PL_COMP_ID,
        altIds="true", page=0,
    )
    for team in data.get("content", []):
        if HULL_NAME in (team.get("name", "") or "").lower():
            return int(team["id"])
    return None


def _pulse_player_goals(player_id: int, season_id: int) -> int:
    """Season goals for one player (absent in the stats feed means zero)."""
    data = _pulse_get(
        f"stats/player/{player_id}", comps=PL_COMP_ID, compSeasons=season_id
    )
    for s in data.get("stats", []):
        if s.get("name") == "goals":
            return int(s.get("value") or 0)
    return 0


def _pulse_team_players(team_id: int, season_id: int) -> list[dict]:
    """
    All players for a team in a comp-season.

    The staff feed already carries name, shirt number, position and season
    appearances; goals require one extra per-player stats call.
    """
    data    = _pulse_get(
        f"teams/{team_id}/compseasons/{season_id}/staff",
        pageSize=100, altIds="true",
    )
    players = []
    for p in data.get("players", []):
        info   = p.get("info", {}) or {}
        name   = (p.get("name", {}) or {}).get("display")
        pid    = p.get("id")
        if not name or pid is None:
            continue
        number = info.get("shirtNum")
        pos    = _PULSE_POS.get((info.get("position") or "").upper()[:1], "MID")
        apps   = int(p.get("appearances") or 0)

        try:
            goals = _pulse_player_goals(int(pid), season_id)
        except Exception:
            goals = 0

        players.append({
            "name":        name,
            "number":      int(number) if number else 0,
            "pos":         pos,
            "apps":        apps,         # PL appearances
            "total_apps":  apps,         # single competition (PL)
            "goals":       goals,        # PL goals
            "total_goals": goals,
        })

    players.sort(key=lambda p: (p["number"] or 999, p["name"]))
    return players


def _fetch_pulse():
    """
    Try the Premier League Pulse API.

    Returns (players, season_label, source_url) for the newest PL season
    that lists Hull City with usable stats, or None if Hull are not yet a
    live Premier League club (the normal case before 2026/27 kicks off).
    """
    try:
        seasons = _pulse_seasons()
    except Exception as exc:
        print(f"[stats] Pulse unavailable: {exc}", file=sys.stderr)
        return None

    for season_id, label in seasons:
        try:
            team_id = _pulse_find_hull(season_id)
            if team_id is None:
                continue
            print(f"[stats] Pulse: Hull found in PL {label} (team {team_id})",
                  file=sys.stderr)
            players = _pulse_team_players(team_id, season_id)
        except Exception as exc:
            print(f"[stats] Pulse season {label} error: {exc}", file=sys.stderr)
            continue

        meaningful = [p for p in players if p["apps"] > 0]
        if len(meaningful) < 10:
            print(f"[stats] Pulse {label}: only {len(meaningful)} players with "
                  f"apps — skipping", file=sys.stderr)
            continue

        print(f"[stats] Pulse: using {len(players)} players from PL {label}",
              file=sys.stderr)
        return (
            players,
            f"{label} (Premier League)",
            f"{PULSE_BASE}/teams/{team_id}/compseasons/{season_id} (Pulse API)",
        )

    print("[stats] Pulse: Hull City not in any live PL season — "
          "falling back to Wikipedia", file=sys.stderr)
    return None


# ═══════════════════════════════════════════════════════════════════════
# WIKIPEDIA  (fallback — pre-season / Championship proxy)
# ═══════════════════════════════════════════════════════════════════════
def _find_stats_table(soup: BeautifulSoup):
    """Return the wikitable that has player stats (Apps + Goals columns)."""
    for table in soup.find_all("table", class_="wikitable"):
        headers = {th.get_text(strip=True) for th in table.find_all("th")}
        if "Apps" in headers and "Goals" in headers and "Player" in headers:
            return table
    return None


def _pos_short(raw: str) -> str:
    p = (raw or "").upper()
    if p == "GK":                  return "GK"
    if p in ("DF", "D"):           return "DEF"
    if p in ("MF", "M", "AM", "DM"): return "MID"
    if p in ("FW", "F", "ST"):     return "FWD"
    return "MID"


def _parse_table(table, season_label: str) -> list[dict]:
    """
    Extract player rows from the stats wikitable.

    Column layout (based on current Wikipedia template):
      0  No.   1  Pos   2  Nat   3  Player
      4  Total Apps   5  Total Goals
      6  Champ Apps   7  Champ Goals
      (further columns: FA Cup, EFL Cup, Play-offs …)
    """
    players = []
    rows = table.find_all("tr")

    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 6:
            continue
        # Skip header rows (first cell is not a number)
        if not re.match(r"^\d+$", cells[0]):
            continue

        shirt = int(cells[0])
        pos   = _pos_short(cells[1])
        name  = cells[3]  # skip Nat at index 2
        if not name:
            continue

        total_apps  = _parse_apps(cells[4])
        total_goals = _parse_apps(cells[5])
        champ_apps  = _parse_apps(cells[6]) if len(cells) > 6 else total_apps
        champ_goals = _parse_apps(cells[7]) if len(cells) > 7 else total_goals

        players.append({
            "name":        name,
            "number":      shirt,
            "pos":         pos,
            "apps":        champ_apps,   # Championship appearances
            "total_apps":  total_apps,   # All-competition appearances
            "goals":       champ_goals,  # Championship goals
            "total_goals": total_goals,  # All-competition goals
        })

    players.sort(key=lambda p: (p["number"] or 999, p["name"]))
    return players


def _fetch_wikipedia():
    """
    Scrape Hull City player stats from Wikipedia, newest season first.

    Returns (players, season_label, source_url) or None.
    """
    current_year = datetime.now().year  # real calendar year

    for start_year in range(current_year, current_year - 4, -1):
        url   = _wiki_url(start_year)
        short = str(start_year + 1)[-2:]
        label = f"{start_year}/{short} (Wikipedia)"
        print(f"[stats] Trying {url}", file=sys.stderr)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue

            soup  = BeautifulSoup(resp.text, "html.parser")
            table = _find_stats_table(soup)
            if table is None:
                print(f"[stats]   No stats table found", file=sys.stderr)
                continue

            parsed = _parse_table(table, label)
            if not parsed:
                continue

            # Require at least 10 players with meaningful apps to count as usable
            meaningful = [p for p in parsed if p["apps"] > 0 or p["total_apps"] > 0]
            if len(meaningful) < 10:
                print(f"[stats]   Only {len(meaningful)} players — skipping",
                      file=sys.stderr)
                continue

            print(f"[stats]   Found {len(parsed)} players", file=sys.stderr)
            return parsed, label, url

        except Exception as exc:
            print(f"[stats]   Error: {exc}", file=sys.stderr)
            continue

    return None


def fetch(force: bool = False) -> dict:
    """
    Return Hull City player stats, preferring the Pulse API and falling back
    to Wikipedia.  Cached for 24 hours; pass force=True to re-fetch.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if not force and _is_fresh(STATS_CACHE):
        print(f"[stats] Using cached stats: {STATS_CACHE}", file=sys.stderr)
        return json.loads(STATS_CACHE.read_text())

    result = _fetch_pulse() or _fetch_wikipedia()

    if result is None:
        raise RuntimeError(
            "Could not find Hull City stats via the Pulse API or Wikipedia."
        )

    players, season_label, source_url = result
    is_pulse = "Pulse API" in source_url

    payload = {
        "team":    "Hull City",
        "fetched": datetime.now().isoformat(),
        "season":  season_label,
        "source":  source_url,
        "note":    ("Premier League stats via the official Pulse API."
                    if is_pulse else
                    "Cards and assists not available in Wikipedia season articles."),
        "players": {p["name"]: p for p in players},
    }
    STATS_CACHE.write_text(json.dumps(payload, indent=2))
    print(f"[stats] Cached stats for {len(players)} players → {STATS_CACHE}",
          file=sys.stderr)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Hull City season stats (Pulse API → Wikipedia fallback)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    data = fetch(force=args.force)
    print(f"\nSeason: {data['season']}\nSource: {data['source']}\n")
    header = f"{'#':<4} {'Player':<26} {'Pos':<4} {'Apps':>6} {'Gls':>5} {'All Apps':>8} {'All Gls':>7}"
    print(header)
    print("-" * len(header))
    for name, s in data["players"].items():
        print(f"{str(s['number']):<4} {name:<26} {s['pos']:<4} {s['apps']:>6} {s['goals']:>5} {s['total_apps']:>8} {s['total_goals']:>7}")
