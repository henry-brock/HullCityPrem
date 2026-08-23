"""
pl_players.py — season player stats for EVERY Premier League team, 2026/27.

Needed because a player rating is only meaningful against positional peers,
and ranking within Hull's own 16 players says almost nothing. This pulls all
20 squads so percentiles are league-wide.

Cost control: the squad endpoint already reports appearances, so per-player
stat calls are made only for players who have actually played. Early in a
season that is a handful of requests; by May it is ~500, cached and
incremental (a player's block is refetched only when their appearance count
has changed).

Cache: cache/pl_players_<season>.json
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from fixture_scraper import _name_to_id, _pulse_get, _team_name
from pulse_players import _map_stats, _player_stats
from pulse_results import _find_season_id, _season_year

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_TTL = timedelta(hours=6)
PL_COMP_ID = 1
DELAY = 0.08


def _cache_path(season: str) -> Path:
    return CACHE_DIR / f"pl_players_{season}.json"


def _teams(season_id: int) -> dict[int, str]:
    """{pulseTeamId: internal_id} for the season, taken from its fixtures."""
    out, page = {}, 0
    while True:
        data = _pulse_get("fixtures", comps=PL_COMP_ID, compSeasons=season_id,
                          page=page, pageSize=100, sort="asc")
        content = data.get("content", [])
        if not content:
            break
        for f in content:
            for entry in f.get("teams", []):
                team = entry.get("team", {})
                tid = _name_to_id(_team_name(team))
                if tid:
                    out[int(team["id"])] = tid
        if len(out) >= 20:
            break
        page += 1
        if page > 8:
            break
    return out


def fetch(season: str = "2627", force: bool = False) -> dict:
    """Return {"players": [...]} across all PL teams, one record per player."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(season)

    cached = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
            age  = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            if not force and age < CACHE_TTL:
                print(f"[pl_players] using cached {path}", file=sys.stderr)
                return prev
            cached = {p["id"]: p for p in prev.get("players", [])}
        except (json.JSONDecodeError, KeyError):
            pass

    season_id = _find_season_id(_season_year(season))
    teams = _teams(season_id)
    print(f"[pl_players] {len(teams)} teams in {season}", file=sys.stderr)

    players, fetched, reused = [], 0, 0
    for pulse_id, internal in sorted(teams.items(), key=lambda kv: kv[1]):
        try:
            squad = _pulse_get(f"teams/{pulse_id}/compseasons/{season_id}/staff",
                               pageSize=100, compSeasons=season_id).get("players", [])
        except Exception as exc:
            print(f"[pl_players] squad failed for {internal}: {exc}", file=sys.stderr)
            continue

        for p in squad:
            pid  = int(p["id"])
            info = p.get("info") or {}
            apps = int(p.get("appearances") or 0)
            base = {
                "id":    pid,
                "name":  (p.get("name") or {}).get("display") or "",
                "team":  internal,
                "pos":   info.get("position") or "",
                "shirt": info.get("shirtNum"),
            }
            if apps == 0:
                # Never played — no stat block exists, so don't spend a request.
                players.append({**base, "stats": {"appearances": 0, "minutes": 0}})
                continue

            old = cached.get(pid)
            if old and not force and (old.get("stats", {}).get("appearances") == apps):
                players.append({**base, "stats": old["stats"]})
                reused += 1
                continue

            players.append({**base, "stats": _map_stats(_player_stats(pid, season_id))})
            fetched += 1
            time.sleep(DELAY)

    payload = {
        "season":  season,
        "fetched": datetime.now().isoformat(),
        "teams":   len(teams),
        "players": players,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[pl_players] {len(players)} players ({fetched} fetched, {reused} reused) "
          f"→ {path}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="All-PL player season stats")
    ap.add_argument("--season", default="2627")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    d = fetch(season=args.season, force=args.force)
    played = [p for p in d["players"] if p["stats"].get("minutes", 0) > 0]
    print(f"\n{len(d['players'])} players, {len(played)} with minutes")
    for p in sorted(played, key=lambda x: -x["stats"]["minutes"])[:12]:
        s = p["stats"]
        print(f"  {p['name']:<24}{p['team']:<12}{p['pos']:<3}{s['minutes']:>5}m "
              f"{s['totalGoals']}g {s['goalAssists']}a")
