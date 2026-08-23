"""
espn_games.py — cache per-match data for both leagues.

All event IDs for a season come from the core API in one call; each match's
boxscore (team match stats incl. possession, shots, passes) and key events
(goals/cards) come from the site API summary. A compact record is stored per
game (the full summaries are ~100 KB each — too big to keep wholesale).
Resumable: cached event IDs are skipped, and progress is flushed every few
games. One file per league:
    cache/espn_games_<league>_<code>.json
"""

import json
import re
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


def _summary(lg, eid):
    """Compact record: identity, score, per-team match stats, key events."""
    d = _get(f"{SITE.format(lg=lg)}/summary?event={eid}")
    bs = d.get("boxscore", {})
    rec = {"id": eid, "teamStats": {}, "score": {}}
    for t in bs.get("teams", []):
        side = t.get("homeAway") or t.get("team", {}).get("id")
        info = t.get("team", {})
        rec["teamStats"][side] = {
            "team": info.get("displayName"),
            "id": info.get("id"),
            "stats": {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])},
        }
    comp = (d.get("header", {}).get("competitions") or [{}])[0]
    rec["date"] = comp.get("date")
    for c in comp.get("competitors", []):
        rec["score"][c.get("homeAway")] = c.get("score")
    key = []
    for k in d.get("keyEvents", []):
        key.append({
            "type": (k.get("type") or {}).get("text"),
            "clock": (k.get("clock") or {}).get("displayValue"),
            "team": (k.get("team") or {}).get("displayName"),
            "players": [p.get("displayName") for p in k.get("participants", []) if p.get("displayName")],
        })
    rec["keyEvents"] = key
    return rec


def fetch(league: str, code: str = "2526"):
    yr = CODE_TO_YEAR[code]
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"espn_games_{league}_{code}.json"
    data = json.loads(out.read_text()) if out.exists() else \
        {"league": league, "season": code, "source": "ESPN", "games": {}}

    items = _get(f"{CORE.format(lg=league, yr=yr)}/types/1/events?limit=1000").get("items", [])
    eids = [re.search(r"/events/(\d+)", it["$ref"]).group(1) for it in items]
    todo = [e for e in eids if e not in data["games"]]
    print(f"[games] {league}: {len(eids)} events, {len(todo)} to fetch", file=sys.stderr)

    for i, eid in enumerate(todo, 1):
        try:
            data["games"][eid] = _summary(league, eid)
        except Exception as e:
            print(f"[games] {league} event {eid} failed: {e}", file=sys.stderr)
        if i % 25 == 0:
            out.write_text(json.dumps(data, indent=2))
            print(f"[games] {league}: {i}/{len(todo)} done", file=sys.stderr)
        time.sleep(0.05)
    out.write_text(json.dumps(data, indent=2))
    print(f"[games] {league}: {len(data['games'])} games cached", file=sys.stderr)
    return data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--league", default="all", choices=["eng.1", "eng.2", "all"])
    p.add_argument("--season", default="2526")
    args = p.parse_args()
    for lg in (LEAGUES if args.league == "all" else [args.league]):
        d = fetch(lg, args.season)
        print(f"{lg}: {len(d['games'])} games "
              f"→ cache/espn_games_{lg}_{args.season}.json")
