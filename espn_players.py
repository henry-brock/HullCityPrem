"""
espn_players.py — cache per-player season statistics for both leagues.

Roster comes 34-at-once from the site API (names/positions/ids); each
player's ~107 season stats come from the core API. Resumable: re-running
skips teams already present in the output file, so an interrupted run
continues where it left off. One file per league:
    cache/espn_players_<league>_<code>.json
"""

import json
import sys
import time
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
SITE      = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}"
CORE      = "https://sports.core.api.espn.com/v2/sports/soccer/leagues/{lg}/seasons/{yr}"
UA        = {"User-Agent": "Mozilla/5.0 (X11) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
CODE_TO_YEAR = {"2526": 2025, "2425": 2024, "2324": 2023, "2223": 2022, "2122": 2021}
LEAGUES = ["eng.1", "eng.2"]


def _get(url):
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json()


def _player_stats(lg, yr, aid):
    try:
        cats = _get(f"{CORE.format(lg=lg, yr=yr)}/types/1/athletes/{aid}/statistics") \
            .get("splits", {}).get("categories", [])
    except Exception:
        return {}
    return {s.get("name"): s.get("value") for c in cats for s in c.get("stats", [])}


def fetch(league: str, code: str = "2526"):
    yr = CODE_TO_YEAR[code]
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"espn_players_{league}_{code}.json"
    data = json.loads(out.read_text()) if out.exists() else \
        {"league": league, "season": code, "source": "ESPN", "teams": {}}

    team_refs = _get(f"{CORE.format(lg=league, yr=yr)}/teams?limit=50").get("items", [])
    for ref in team_refs:
        tid = ref["$ref"].split("/teams/")[1].split("?")[0].rstrip("/")
        if tid in data["teams"]:
            continue
        try:
            roster = _get(f"{SITE.format(lg=league)}/teams/{tid}/roster").get("athletes", [])
        except Exception as e:
            print(f"[players] {league} team {tid} roster failed: {e}", file=sys.stderr)
            continue
        players = []
        for a in roster:
            aid = a.get("id")
            players.append({
                "id": aid,
                "name": a.get("displayName"),
                "pos": (a.get("position") or {}).get("abbreviation"),
                "stats": _player_stats(league, yr, aid),
            })
            time.sleep(0.05)
        if not players:
            # Empty roster (transient) — don't mark done, retry next run.
            print(f"[players] {league} {tid}: empty roster, will retry", file=sys.stderr)
            continue
        tname = roster[0].get("team", {}).get("displayName") if roster else tid
        data["teams"][tid] = {"team": tname, "players": players}
        out.write_text(json.dumps(data, indent=2))   # flush after each team
        print(f"[players] {league} {tid}: {len(players)} players cached "
              f"({len(data['teams'])} teams done)", file=sys.stderr)
    return data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--league", default="all", choices=["eng.1", "eng.2", "all"])
    p.add_argument("--season", default="2526")
    args = p.parse_args()
    for lg in (LEAGUES if args.league == "all" else [args.league]):
        d = fetch(lg, args.season)
        n = sum(len(t["players"]) for t in d["teams"].values())
        print(f"{lg}: {len(d['teams'])} teams, {n} players "
              f"→ cache/espn_players_{lg}_{args.season}.json")
