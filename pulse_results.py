"""
pulse_results.py
Live 2026/27 Premier League results from the official PL Pulse API —
the fast-refresh counterpart to scraper.py (football-data.co.uk).

Why this exists
───────────────
football-data.co.uk only publishes its season CSV in batches, and creates
the file at all some way into the season (in 2026/27 it still had no
E0.csv the night the season kicked off). Pulse carries a full-time result
within minutes of the whistle, so it is the better source for keeping the
table current. scraper.py is retained and still preferred once it catches
up: it is the settled record and carries bookmaker odds, which Pulse has
no equivalent for.

Output shape is deliberately identical to scraper._parse() so the two are
drop-in interchangeable, with two differences:
  • no `odds` / `referee`  — Pulse does not carry them
  • adds `homePossession` / `awayPossession` — football-data has no equivalent

Caching
───────
• Result list  — short TTL (live source; new matches finish all the time).
• Match stats  — permanent, keyed by fixture id. A completed match's Opta
  stats never change, so each fixture's stat block is fetched exactly once.
  This keeps a full-season refresh to one request per *newly* played match
  rather than 380.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Reuse the Pulse plumbing + the canonical team-name map rather than
# duplicating them (championship_scraper follows the same shared-helper
# pattern). fixture_scraper stays untouched.
from fixture_scraper import (
    _fetch_all,
    _is_home,
    _name_to_id,
    _pulse_get,
    _season_start_year,
    _team_name,
)
from scraper import SeasonNotAvailable   # shared exception → app.py handles both alike

ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / "cache"

CACHE_TTL   = timedelta(minutes=15)   # live source — refresh often
SETTLE_DAYS = 7                       # a younger match may still be provisional
PL_COMP_ID  = 1

# Pulse fixture status codes. Only "C" (completed) may be imported —
# importing an in-play match would write a partial scoreline into the table.
STATUS_COMPLETED = "C"


def _cache_paths(season: str) -> tuple[Path, Path]:
    return (
        CACHE_DIR / f"pulse_{season}.json",
        CACHE_DIR / f"pulse_matchstats_{season}.json",
    )


def _season_year(season: str) -> int:
    """"2627" → 2026 (the calendar year the season starts in)."""
    return 2000 + int(season[:2])


def _find_season_id(year: int) -> int:
    """Locate the Pulse compSeason id for the PL season starting in `year`."""
    data = _pulse_get(f"competitions/{PL_COMP_ID}/compseasons", pageSize=100)
    for season in data.get("content", []):
        if _season_start_year(season.get("label", "")) == year:
            return int(season["id"])          # Pulse serialises ids as floats
    raise SeasonNotAvailable(
        f"The {year}/{str(year + 1)[2:]} Premier League season is not yet "
        f"registered in the Pulse API."
    )


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < CACHE_TTL


# ── Match stats ─────────────────────────────────────────────────────────

def _fetch_match_stats(fixture_id: int) -> dict[str, dict[str, float]]:
    """
    Return {pulse_team_id: {stat_name: value}} for one fixture.

    Returns {} when the stat feed is unavailable — the caller degrades to a
    result-only record rather than failing the whole import.
    """
    try:
        data = _pulse_get(f"stats/match/{fixture_id}")
    except (requests.RequestException, ValueError) as exc:
        print(f"[pulse_results] stats unavailable for {fixture_id}: {exc}", file=sys.stderr)
        return {}

    return {
        str(team_id): {s["name"]: s.get("value") for s in (block.get("M") or [])}
        for team_id, block in (data.get("data") or {}).items()
    }


def _count(stats: dict, name: str) -> int:
    """
    Read a counting stat. Opta omits a stat entirely when its value is zero
    (e.g. `total_red_card` is absent in a match with no red cards), so a
    missing key means 0 — not unknown.
    """
    value = stats.get(name)
    return 0 if value is None else int(value)


def _side_stats(stats: dict | None) -> dict:
    """Map one team's Opta block to the football-data field names."""
    if not stats:
        # No stat feed for this match — leave shot fields null so downstream
        # consumers (e.g. the xG table) skip it instead of reading zeros.
        return {
            "shots": None, "shotsOnTarget": None, "fouls": None,
            "yellow": None, "red": None, "corners": None, "possession": None,
        }

    possession = stats.get("possession_percentage")
    # PPDA (passes allowed per defensive action) only exists from 2025/26 on,
    # so it is None for older seasons rather than 0 — a missing PPDA means
    # "not recorded", never "pressed infinitely hard".
    ppda = stats.get("ppda")
    return {
        # total_scoring_att includes blocked shots, matching football-data's HS.
        "shots":         _count(stats, "total_scoring_att"),
        # ontarget_scoring_att includes goals, matching football-data's HST.
        "shotsOnTarget": _count(stats, "ontarget_scoring_att"),
        "fouls":         _count(stats, "fk_foul_lost"),
        "yellow":        _count(stats, "total_yel_card"),
        "red":           _count(stats, "total_red_card"),
        "corners":       _count(stats, "corner_taken") or _count(stats, "won_corners"),
        "possession":    float(possession) if possession is not None else None,
        # Forward-looking extras for 2026/27 analysis: pressing intensity and
        # the best single attack-quality signal found in the Newcache probe.
        "ppda":          float(ppda) if ppda is not None else None,
        "touchesInBox":  _count(stats, "touches_in_opp_box"),
    }


# ── Core ────────────────────────────────────────────────────────────────

def _settled(fixture: dict) -> bool:
    """
    True once Opta has stopped revising a match. Without this the first
    (provisional) stat block fetched after the whistle would be cached
    permanently, freezing possession and dropping late-arriving fields.
    """
    millis = (fixture.get("kickoff") or {}).get("millis") or 0
    if not millis:
        return True
    age = datetime.now(tz=timezone.utc) - datetime.fromtimestamp(millis / 1000,
                                                                 tz=timezone.utc)
    return age > timedelta(days=SETTLE_DAYS)


def _completed(raw: list[dict]) -> list[dict]:
    return [f for f in raw if f.get("status") == STATUS_COMPLETED]


def _parse(fixture: dict, stats_by_team: dict) -> dict | None:
    """Convert one completed Pulse fixture into a scraper.py-shaped record."""
    home = away = None

    for index, entry in enumerate(fixture.get("teams", [])):
        team = entry.get("team", {})
        internal = _name_to_id(_team_name(team))
        if internal is None:               # outside our 2026/27 roster
            continue
        side = {
            "id":      internal,
            "goals":   entry.get("score"),
            "pulseId": str(int(team["id"])),
        }
        if _is_home(entry, index):
            home = side
        else:
            away = side

    if not home or not away:
        return None
    if home["goals"] is None or away["goals"] is None:
        return None

    hs = _side_stats(stats_by_team.get(home["pulseId"]))
    as_ = _side_stats(stats_by_team.get(away["pulseId"]))

    home_goals, away_goals = int(home["goals"]), int(away["goals"])

    # Pulse gives kickoff as epoch millis plus the local UK offset. Apply the
    # offset so date/time match football-data's local-time convention — using
    # UTC would also roll a late kickoff back onto the previous day.
    ko         = fixture.get("kickoff") or {}
    kickoff_ms = ko.get("millis") or 0
    offset     = timedelta(hours=float(ko.get("gmtOffset") or 0))
    kickoff    = datetime.fromtimestamp(kickoff_ms / 1000, tz=timezone.utc) + offset

    return {
        # ── Identity ──────────────────────────────
        # football-data uses DD/MM/YYYY; keep that so both sources agree.
        "date":     kickoff.strftime("%d/%m/%Y"),
        "time":     kickoff.strftime("%H:%M"),
        "referee":  "",                    # not exposed by Pulse
        "homeId":   home["id"],
        "awayId":   away["id"],
        # ── Full-time result ──────────────────────
        "homeGoals": home_goals,
        "awayGoals": away_goals,
        "ftResult":  fixture.get("outcome")
                     or ("H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"),
        # ── Half-time result ──────────────────────
        # Pulse's goals[] identifies scorers by personId with no team marker,
        # so attributing them would cost an extra squad lookup per match.
        # Nothing consumes these fields, so they stay null.
        "htHomeGoals": None,
        "htAwayGoals": None,
        "htResult":    "",
        # ── Shots ─────────────────────────────────
        "homeShots":         hs["shots"],
        "awayShots":         as_["shots"],
        "homeShotsOnTarget": hs["shotsOnTarget"],
        "awayShotsOnTarget": as_["shotsOnTarget"],
        # ── Fouls & cards ─────────────────────────
        "homeFouls":  hs["fouls"],  "awayFouls":  as_["fouls"],
        "homeYellow": hs["yellow"], "awayYellow": as_["yellow"],
        "homeRed":    hs["red"],    "awayRed":    as_["red"],
        # ── Corners ───────────────────────────────
        "homeCorners": hs["corners"], "awayCorners": as_["corners"],
        # ── Pulse-only extras (no football-data equivalent) ──
        "homePossession": hs["possession"],
        "awayPossession": as_["possession"],
        "homePPDA":       hs["ppda"],
        "awayPPDA":       as_["ppda"],
        "homeTouchesInBox": hs["touchesInBox"],
        "awayTouchesInBox": as_["touchesInBox"],
        # No `odds` key — Pulse carries no betting markets. scraper.py fills
        # these in once football-data.co.uk publishes the season CSV.
    }


def fetch(season: str = "2627", force: bool = False) -> list[dict]:
    """
    Return completed matches for `season` from the Pulse API, in the same
    record shape as scraper.fetch().

    Raises SeasonNotAvailable when Pulse has no such season registered.
    An empty list means the season exists but no match has finished yet.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    results_path, stats_path = _cache_paths(season)

    if not force and _is_fresh(results_path):
        print(f"[pulse_results] Using cached results: {results_path}", file=sys.stderr)
        return json.loads(results_path.read_text())["matches"]

    season_id = _find_season_id(_season_year(season))
    print(f"[pulse_results] Fetching {season} results (compSeason {season_id})…", file=sys.stderr)

    played = _completed(_fetch_all(season_id))

    # Per-fixture stats cache. A finished match is NOT immutable: Opta keeps
    # revising it for a day or two, re-deriving possession and adding late
    # fields such as clean_sheet. So blocks are re-fetched until the match is
    # SETTLE_DAYS old, and only then treated as final.
    try:
        stats_cache = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    except json.JSONDecodeError:
        stats_cache = {}

    fetched = 0
    for fixture in played:
        key = str(int(fixture["id"]))
        if force or key not in stats_cache or not _settled(fixture):
            stats_cache[key] = _fetch_match_stats(int(fixture["id"]))
            fetched += 1

    if fetched:
        stats_path.write_text(json.dumps(stats_cache, indent=2), encoding="utf-8")
        print(f"[pulse_results] Fetched stats for {fetched} new match(es).", file=sys.stderr)

    matches = []
    for fixture in played:
        record = _parse(fixture, stats_cache.get(str(int(fixture["id"])), {}))
        if record:
            matches.append(record)

    results_path.write_text(json.dumps({
        "season":  season,
        "fetched": datetime.now().isoformat(),
        "source":  f"pulse compSeason {season_id}",
        "matches": matches,
    }, indent=2), encoding="utf-8")

    print(f"[pulse_results] {len(matches)} completed match(es) → {results_path}", file=sys.stderr)
    return matches


# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch live PL results from the Pulse API")
    parser.add_argument("--season", default="2627", help="Season code, e.g. 2627")
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    try:
        rows = fetch(season=args.season, force=args.force)
    except SeasonNotAvailable as exc:
        print(f"Not available: {exc}", file=sys.stderr)
        sys.exit(1)

    for m in rows:
        print(f"{m['date']}  {m['homeId']:>12} {m['homeGoals']}-{m['awayGoals']} "
              f"{m['awayId']:<12}  shots {m['homeShots']}-{m['awayShots']}  "
              f"SoT {m['homeShotsOnTarget']}-{m['awayShotsOnTarget']}")
    print(f"\n{len(rows)} completed match(es).", file=sys.stderr)
