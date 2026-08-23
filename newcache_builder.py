"""
newcache_builder.py
Builds `Newcache/` — a rich per-match Opta archive from the PL Pulse API.

Why
───
Every existing stat source in this project is limited in a way that caps
prediction quality:
  • football-data.co.uk — ~20 fields, no possession, no shot location
  • ESPN               — only populates advanced stats for the LIVE season
  • the xG proxy       — 0.33·SoT − 0.01·offTarget treats every shot alike

Pulse carries ~160 Opta stats per team per match and, unlike ESPN, keeps
them for past seasons. The two headline unlocks:
  • shot LOCATION (attempts_ibox / attempts_obox and the full
    att_ibox_goal/target/miss/blocked split) — an in-box shot is worth
    several times an out-of-box one, so this supports a far better xG
    proxy than the location-blind SoT formula.
  • real possession for past seasons, which ESPN could not provide.

Scope is the seasons that actually feed predictions: the three-season
window ratings.py uses, plus the current one.

Layout
──────
  Newcache/pl_<season>.json   one file per season, full stat blocks
  Newcache/index.json         manifest: counts + per-season stat coverage

Unlike the prediction-path scrapers this keeps EVERY match, including
teams outside the 2026/27 roster (Leicester, Southampton, …). Opponent-
adjusted ratings need those games to rate the teams that did stay up;
`homeId`/`awayId` are simply null for a team with no internal id.

Resumable: per-fixture stats are keyed by fixture id and never re-fetched,
and progress is flushed to disk as it goes, so an interrupted run resumes
where it stopped.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fixture_scraper import _fetch_all, _is_home, _name_to_id, _team_name
from pulse_results import _fetch_match_stats, _find_season_id, _season_year
from scraper import SeasonNotAvailable

ROOT     = Path(__file__).parent
NEWCACHE = ROOT / "Newcache"

# The window that feeds predictions: ratings.py's three-season lookback
# plus the live season.
DEFAULT_SEASONS = ["2324", "2425", "2526", "2627"]

REQUEST_DELAY = 0.15   # be polite — this walks ~380 fixtures per season
FLUSH_EVERY   = 25     # matches between incremental saves


def _season_path(season: str) -> Path:
    return NEWCACHE / f"pl_{season}.json"


def _load_season(season: str) -> dict:
    path = _season_path(season)
    if not path.exists():
        return {}
    try:
        cached = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {str(m["fixtureId"]): m for m in cached.get("matches", [])}


def _save_season(season: str, comp_season_id: int, by_id: dict) -> None:
    matches = sorted(by_id.values(), key=lambda m: (m["gw"] or 99, m["ms"] or 0))
    _season_path(season).write_text(json.dumps({
        "season":        season,
        "compSeasonId":  comp_season_id,
        "fetched":       datetime.now().isoformat(),
        "count":         len(matches),
        "matches":       matches,
    }, indent=1), encoding="utf-8")


def _sides(fixture: dict) -> tuple[dict | None, dict | None]:
    """Return (home, away) descriptors, or (None, None) if malformed."""
    home = away = None
    for index, entry in enumerate(fixture.get("teams", [])):
        team = entry.get("team", {})
        name = _team_name(team)
        side = {
            "name":    name,
            "id":      _name_to_id(name),        # null when outside the 26/27 roster
            "pulseId": str(int(team["id"])),
            "goals":   entry.get("score"),
        }
        if _is_home(entry, index):
            home = side
        else:
            away = side
    return home, away


def _record(fixture: dict, stats_by_team: dict) -> dict | None:
    home, away = _sides(fixture)
    if not home or not away:
        return None

    ko      = fixture.get("kickoff") or {}
    millis  = ko.get("millis") or 0
    # Apply the local UK offset, as pulse_results does — UTC would roll a
    # late kickoff onto the previous date.
    kickoff = (datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
               + timedelta(hours=float(ko.get("gmtOffset") or 0)))

    return {
        "fixtureId": int(fixture["id"]),
        "gw":        (fixture.get("gameweek") or {}).get("gameweek"),
        "ms":        millis,
        "date":      kickoff.strftime("%d/%m/%Y"),
        "time":      kickoff.strftime("%H:%M"),
        "homeId":    home["id"],   "awayId":    away["id"],
        "homeName":  home["name"], "awayName":  away["name"],
        "homeGoals": home["goals"], "awayGoals": away["goals"],
        "outcome":   fixture.get("outcome"),
        # Full raw Opta blocks, unfiltered — this is an archive, so keep
        # everything rather than guessing now what a model will want later.
        "stats": {
            "home": stats_by_team.get(home["pulseId"], {}),
            "away": stats_by_team.get(away["pulseId"], {}),
        },
    }


def build_season(season: str, force: bool = False) -> dict:
    """Fetch/refresh one season into Newcache. Returns a summary dict."""
    NEWCACHE.mkdir(exist_ok=True)

    comp_season_id = _find_season_id(_season_year(season))
    cached = {} if force else _load_season(season)

    fixtures = [f for f in _fetch_all(comp_season_id) if f.get("status") == "C"]
    print(f"[newcache] {season}: {len(fixtures)} completed, "
          f"{len(cached)} already cached", file=sys.stderr)

    fetched = 0
    for i, fixture in enumerate(fixtures, 1):
        key = str(int(fixture["id"]))
        if key in cached and cached[key].get("stats", {}).get("home"):
            continue

        record = _record(fixture, _fetch_match_stats(int(fixture["id"])))
        if record:
            cached[key] = record
        fetched += 1
        time.sleep(REQUEST_DELAY)

        if fetched % FLUSH_EVERY == 0:
            _save_season(season, comp_season_id, cached)
            print(f"[newcache] {season}: {i}/{len(fixtures)} "
                  f"({fetched} fetched)", file=sys.stderr)

    _save_season(season, comp_season_id, cached)
    print(f"[newcache] {season}: done — {len(cached)} matches "
          f"({fetched} newly fetched)", file=sys.stderr)

    return {
        "season":       season,
        "compSeasonId": comp_season_id,
        "matches":      len(cached),
        "coverage":     _coverage(cached),
    }


def _coverage(by_id: dict) -> dict:
    """
    Share of matches in which each stat is present, so it is obvious which
    fields are actually modellable for a given season (ppda, for instance,
    only exists in recent seasons).
    """
    if not by_id:
        return {}
    seen  = Counter()
    total = 0
    for match in by_id.values():
        for side in ("home", "away"):
            block = match.get("stats", {}).get(side) or {}
            if not block:
                continue
            total += 1
            seen.update(block.keys())
    if not total:
        return {}
    return {name: round(count / total, 3) for name, count in seen.most_common()}


def _write_index(summaries: list[dict]) -> None:
    (NEWCACHE / "index.json").write_text(json.dumps({
        "built":   datetime.now().isoformat(),
        "source":  "footballapi.pulselive.com (official PL Opta feed)",
        "seasons": summaries,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Newcache Opta archive")
    parser.add_argument("--seasons", default=",".join(DEFAULT_SEASONS),
                        help=f"Comma-separated season codes (default: {','.join(DEFAULT_SEASONS)})")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    summaries = []
    for season in [s.strip() for s in args.seasons.split(",") if s.strip()]:
        try:
            summaries.append(build_season(season, force=args.force))
        except SeasonNotAvailable as exc:
            print(f"[newcache] {season}: skipped — {exc}", file=sys.stderr)

    _write_index(summaries)
    print(f"\n[newcache] {sum(s['matches'] for s in summaries)} matches "
          f"across {len(summaries)} season(s) → {NEWCACHE}", file=sys.stderr)
