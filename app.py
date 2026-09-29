"""Hull City PL 2026/27 — Flask backend for local data persistence."""

from flask import Flask, jsonify, request, send_file
from pathlib import Path
from scraper import fetch as scrape_season, SeasonNotAvailable
from pulse_results import fetch as scrape_live
from fixture_scraper import fetch as scrape_fixtures, FixturesNotAvailable
from collections import defaultdict
import json
import os
import sys
import threading
import time

app  = Flask(__name__)

ROOT = Path(__file__).parent
DB   = ROOT / "data.json"

# Team Stats and Derived Stats top Newcache up on every request; within this
# many seconds of the last top-up there is nothing new for them to find.
NEWCACHE_TOPUP_EVERY = 600


def _load() -> dict:
    try:
        return json.loads(DB.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"results": [], "playerStats": {}, "mcHistory": []}


def _save(data: dict) -> None:
    DB.write_text(json.dumps(data, indent=2))


@app.route("/")
def index():
    return send_file(ROOT / "index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(_load())


@app.route("/api/state", methods=["POST"])
def post_state():
    try:
        data = request.get_json(force=True, silent=True) or {}
        payload = {
            "results":   data.get("results",   []),
            "mcHistory": data.get("mcHistory", []),
        }
        if "fixtures" in data:
            payload["fixtures"] = data["fixtures"]
        if "fixturesSource" in data:
            payload["fixturesSource"] = data["fixturesSource"]
        _save(payload)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/deductions", methods=["GET"])
def get_deductions():
    """Read-only here — deductions are added and removed from admin.py."""
    import deductions
    entries = deductions.load()
    return jsonify({"deductions": entries, "totals": deductions.totals(entries)})


@app.route("/api/fixtures", methods=["GET"])
def get_fixtures():
    """
    Try to fetch the official 2026/27 fixture list from the Pulse API.
    Returns {"fixtures": [...380 items...], "source": "official"} when
    available, or {"fixtures": null, "source": null} while the schedule
    hasn't been published yet.
    Pass ?force=true to bypass the 24-hour partial cache and retry immediately.
    """
    force = request.args.get("force", "").lower() == "true"
    try:
        fixtures = scrape_fixtures(force=force)
        return jsonify({"fixtures": fixtures, "source": "official"})
    except FixturesNotAvailable:
        return jsonify({"fixtures": None, "source": None})
    except Exception as exc:
        return jsonify({"fixtures": None, "source": None, "error": str(exc)})


def _gather_results(season: str, force: bool) -> tuple[list[dict], list[str]]:
    """
    Collect a season's results from every available source, merged by fixture.

    Sources are applied cheapest-detail-first so the richer record wins:
      1. Pulse   — official PL API, carries a result minutes after full time,
                   but has no betting odds.
      2. football-data.co.uk — lags by days and only appears once the season
                   is under way, but is the settled record and carries odds.

    Raises SeasonNotAvailable only when *no* source has the season yet.
    """
    merged: dict[tuple[str, str], dict] = {}
    sources, problems = [], []

    for name, fetch_fn in (("pulse", scrape_live), ("football-data", scrape_season)):
        try:
            rows = fetch_fn(season=season, force=force)
        except SeasonNotAvailable as exc:
            problems.append(str(exc))
            continue
        except Exception as exc:                      # network/parse fault in one
            problems.append(f"{name}: {exc}")         # source must not sink the other
            continue

        for m in rows:
            key  = (m["homeId"], m["awayId"])
            base = merged.get(key)
            # Overlay non-null fields so a later source adds detail (odds,
            # referee) without blanking what an earlier one uniquely had
            # (Pulse possession has no football-data equivalent).
            merged[key] = {**base, **{k: v for k, v in m.items() if v is not None}} if base else m

        sources.append(name)

    if not sources:
        raise SeasonNotAvailable(" / ".join(problems) or "No result source available.")

    return list(merged.values()), sources


@app.route("/api/derived", methods=["GET"])
def derived():
    """Ratio metrics (chance suppression, engagement, value from the ball)."""
    season = request.args.get("season", "2627")
    if request.args.get("refresh", "1") == "1":
        try:
            import newcache_builder
            newcache_builder.build_season(season, min_interval=NEWCACHE_TOPUP_EVERY)
        except Exception as exc:
            print(f"[derived] refresh skipped: {exc}", file=sys.stderr)
    try:
        from derived_stats import compute
        return jsonify(compute(season))
    except Exception as exc:
        return jsonify({"teams": {}, "metrics": [], "error": str(exc)}), 500


@app.route("/api/players/rate", methods=["GET"])
def rate_players():
    """
    Position-aware player ratings for one team, scored against LEAGUE
    positional peers (not team-mates — see player_rater's docstring).
    """
    season = request.args.get("season", "2627")
    team   = request.args.get("team", "hull")
    force  = request.args.get("force") == "1"
    try:
        from pl_players import fetch as fetch_pool
        from player_rater import (CONCEPTS_GK, CONCEPTS_OUT, MIN_MINUTES_SHARE,
                                  ROLES, WEIGHTS, leaderboard, rate,
                                  player_derived)

        pool = fetch_pool(season=season, force=force)["players"]
        rows = rate(pool, target_team=team, season=season)
        return jsonify({
            "season":  season,
            "team":    team,
            "players": rows,
            "derived": player_derived(pool, target_team=team),
            # League top tens, so a squad rating has something to sit against.
            "leaders": leaderboard(pool, season=season, top=10),
            "minShare": MIN_MINUTES_SHARE,
            "pool":    sum(1 for p in pool if (p.get("stats") or {}).get("minutes", 0) > 0),
            "concepts": {r: (list(CONCEPTS_GK) if r == "G"
                             else [c for c in CONCEPTS_OUT if WEIGHTS[r].get(c)])
                         for r in ROLES},
            "negative": [c for c, d in {**CONCEPTS_OUT, **CONCEPTS_GK}.items() if d.get("neg")],
        })
    except Exception as exc:
        return jsonify({"players": [], "error": str(exc)}), 500


@app.route("/api/teamstats", methods=["GET"])
def team_stats():
    """
    Per-team, per-game averages of every Opta stat Pulse carries for a season.

    Reads the Newcache archive (which already holds full stat blocks keyed to
    internal team ids) and refreshes it incrementally first, so newly played
    matches appear without a manual rebuild. Only completed matches count.
    """
    season  = request.args.get("season", "2627")
    archive = ROOT / "Newcache" / f"pl_{season}.json"

    # Incremental top-up: build_season only fetches stats for fixtures it has
    # never seen, so this is one fixtures call plus any brand-new matches.
    if request.args.get("refresh", "1") == "1":
        try:
            import newcache_builder
            newcache_builder.build_season(season, min_interval=NEWCACHE_TOPUP_EVERY)
        except Exception as exc:                      # stale cache beats a 500
            print(f"[teamstats] refresh skipped: {exc}", file=sys.stderr)

    if not archive.exists():
        return jsonify({"season": season, "teams": {}, "stats": [], "matches": 0})

    data    = json.loads(archive.read_text())
    totals  = defaultdict(lambda: defaultdict(float))
    games   = defaultdict(int)
    vocab   = set()

    for m in data.get("matches", []):
        for side in ("home", "away"):
            vocab |= set(m["stats"].get(side) or {})

    for m in data.get("matches", []):
        for side in ("home", "away"):
            tid = m.get(f"{side}Id")
            block = m["stats"].get(side) or {}
            if not tid or not block:
                continue
            games[tid] += 1
            # Opta omits zero-valued stats, so absent means 0, not unknown.
            for name in vocab:
                totals[tid][name] += float(block.get(name) or 0.0)

    teams = {
        tid: {
            "games": games[tid],
            "stats": {n: round(v / games[tid], 3) for n, v in totals[tid].items()},
        }
        for tid in games
    }
    return jsonify({
        "season":  season,
        "matches": data.get("count", 0),
        "teams":   teams,
        "stats":   sorted(vocab),
    })


@app.route("/api/import", methods=["POST"])
def import_results():
    """
    Scrape (or read from local cache) a season's results and merge them
    into data.json.  Body: { "season": "2526", "force": false }
    """
    try:
        body   = request.get_json(force=True, silent=True) or {}
        season = body.get("season", "2627")
        force  = bool(body.get("force", False))

        scraped, sources = _gather_results(season, force)

        current = _load()
        by_key  = {(r["homeId"], r["awayId"]): r for r in current["results"]}

        added, enriched = [], 0
        for m in scraped:
            key      = (m["homeId"], m["awayId"])
            existing = by_key.get(key)

            if existing is None:
                # Store full stats record; table logic only needs homeId/awayId/goals
                current["results"].append(m)
                by_key[key] = m
                added.append(m)
                continue

            # Already stored — but a source may carry detail the stored copy
            # lacks: odds (football-data catching up on a match Pulse imported
            # live), shot data (a hand-entered result), or fields added to a
            # scraper after the match was first imported (PPDA, touches in
            # box). Backfill any field that is missing or null locally, rather
            # than naming specific ones — otherwise new fields never land on
            # matches already in data.json.
            missing = {k: v for k, v in m.items()
                       if v is not None and existing.get(k) is None}
            if missing:
                existing.update(missing)
                enriched += 1

        _save(current)

        # A new match makes every player total stale. Those caches carry a 6h
        # TTL, which is fine for idle refreshes but wrong here: a result that
        # lands mid-window leaves minutes and goals frozen at the previous
        # match for hours, and the "refetch when appearances changed" logic in
        # pl_players never gets a chance to run because the file-level TTL
        # short-circuits it first.
        #
        # So the cache is EXPIRED, not deleted: backdating its mtime fails the
        # TTL check while keeping the per-player appearance counts that let
        # pl_players refetch only the ~20 players who just played. Deleting
        # it threw those away and cost ~415 calls per new result.
        if added:
            stale = CACHE_DIR / f"pl_players_{season}.json"
            try:
                if stale.exists():
                    os.utime(stale, (0, 0))
            except OSError as exc:
                print(f"[import] could not expire {stale.name}: {exc}", file=sys.stderr)

        return jsonify({
            "ok":       True,
            "season":   season,
            "added":    len(added),
            "enriched": enriched,
            "sources":  sources,
            "total":    len(current["results"]),
            "matches":  added,
            "playerCacheCleared": bool(added),
        })

    except SeasonNotAvailable as exc:
        return jsonify({"error": str(exc), "not_available": True}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/import/status", methods=["GET"])
def import_status():
    """Return info about the local cache without triggering a download."""
    from scraper import CACHE_DIR
    results = []
    for f in sorted(CACHE_DIR.glob("*.json")):
        try:
            meta = json.loads(f.read_text())
            results.append({
                "file":    f.name,
                "season":  meta.get("season"),
                "fetched": meta.get("fetched"),
                "matches": len(meta.get("matches", [])),
                "source":  meta.get("source"),
            })
        except Exception:
            pass
    return jsonify(results)


HULL_ESPN_ID = "306"
CACHE_DIR = ROOT / "cache"


@app.route("/api/hull/analysis/<league>")
def hull_analysis(league):
    """Return Hull City player stats for squad analysis.

    league = 'eng.2' (Championship 25/26, ESPN cache)
           | 'eng.1' (Premier League 26/27, Pulse — live season)

    The PL side deliberately does NOT read the ESPN cache: that file covers
    2025/26, a season Hull spent in the Championship, so it contains no Hull
    rows at all — which is why this tab was empty. Pulse carries the live PL
    squad and ~70 Opta metrics per player.
    """
    if league == "eng.1":
        try:
            from pulse_players import fetch as scrape_pl_players
            data = scrape_pl_players(season="2627",
                                     force=request.args.get("force") == "1")
            return jsonify({
                "players": data.get("players", []),
                "league":  league,
                "season":  data.get("season", "2627"),
                "team":    data.get("team", "Hull City"),
                "source":  data.get("source"),
            })
        except Exception as exc:
            return jsonify({"players": [], "league": league, "error": str(exc)}), 500

    code = "2526"
    cache = CACHE_DIR / f"espn_players_{league}_{code}.json"
    if not cache.exists():
        return jsonify({"players": [], "league": league, "season": code})
    try:
        data = json.loads(cache.read_text())
        hull = data.get("teams", {}).get(HULL_ESPN_ID, {})
        players = hull.get("players", [])
        return jsonify({
            "players": players,
            "league": league,
            "season": code,
            "team": hull.get("team", "Hull City"),
        })
    except Exception as exc:
        return jsonify({"players": [], "error": str(exc)}), 500


def _live_detail(fixture_id: int, ids: list) -> dict:
    """Formations, starting lineups, events and a readable slice of the live
    Opta blocks — the same shape match_details() gives a finished match, so
    the frontend's player rater doesn't need to know live from finished."""
    from lineups import MATCH_STATS, _team_map
    from pulse_results import _fetch_match_stats
    from pl_players import _pulse_get

    out = {"stats": None, "formations": None, "lineup": None, "events": None}
    try:
        blocks = _fetch_match_stats(fixture_id)
        names  = _team_map("2627")
        by_id  = {names.get(int(t)): b for t, b in blocks.items() if names.get(int(t))}
        if by_id:
            out["stats"] = {
                "labels": [lab for _, lab in MATCH_STATS],
                "home":   [(by_id.get(ids[0]) or {}).get(k) for k, _ in MATCH_STATS],
                "away":   [(by_id.get(ids[1]) or {}).get(k) for k, _ in MATCH_STATS],
            }
    except Exception:
        pass
    try:
        fx = _pulse_get(f"fixtures/{fixture_id}")
        names = _team_map("2627")
        form, lineup, id_to_name, id_to_team = {}, {}, {}, {}
        for tl in fx.get("teamLists") or []:
            tid = names.get(int(tl.get("teamId")))
            if not tid:
                continue
            form[tid] = (tl.get("formation") or {}).get("label")
            for p in (tl.get("lineup") or []) + (tl.get("substitutes") or []):
                if p.get("id"):
                    id_to_name[str(int(p["id"]))] = (p.get("name") or {}).get("display")
                    id_to_team[str(int(p["id"]))] = tid
            # Same {id, name}, formation-row grouping match_details() uses for
            # a finished game, built here from the live payload instead of the
            # cache — a live lineup does not wait for a settle window, since
            # who started is fixed at kickoff and never revised.
            rows = (tl.get("formation") or {}).get("players") or []
            if rows:
                lineup[tid] = [[{"id": pid, "name": (id_to_name.get(str(pid)) or "?").split()[-1]}
                               for pid in row] for row in rows]
        out["formations"] = form or None
        out["lineup"] = lineup or None

        events = []
        for e in fx.get("events") or []:
            if e.get("type") not in ("G", "B", "S"):
                continue
            pid = e.get("personId")
            tid = id_to_team.get(str(pid))
            if not tid:
                continue
            etype = ("goal" if e["type"] == "G" else
                     "sub" if e["type"] == "S" else
                     ("red" if e.get("description") == "R" else "yellow"))
            aid = e.get("assistId") if e["type"] == "G" else None
            events.append({
                "type": etype, "team": tid, "playerId": pid,
                "player": (id_to_name.get(str(pid)) or "?").split()[-1],
                "assistId": aid,
                "assist": (id_to_name.get(str(aid)) or "?").split()[-1] if aid else None,
                "dir": e.get("description") if e["type"] == "S" else None,
                "clock": (e.get("clock") or {}).get("label"),
            })
        out["events"] = events or None
    except Exception:
        pass
    return out


@app.route("/api/live")
def live_matches():
    """
    Matches in progress right now, with the score as it stands.

    Read-only and deliberately so. Nothing here is written to data.json or to
    any cache: an unfinished match has provisional stats, and the whole point
    of the settle window elsewhere is that provisional data must never be
    stored as if it were final. This exists to be shown, then thrown away.
    """
    season = request.args.get("season", "2627")
    detail = request.args.get("detail") == "1"
    key = (season, detail)
    # Every open tab polls once a minute, and the Table and Live tabs each ask
    # again when opened. Within LIVE_TTL they all get the same answer from one
    # Pulse round trip; the lock stops simultaneous callers each fetching it.
    with _live_lock:
        hit = _live_cache.get(key)
        if hit and time.monotonic() - hit[0] < LIVE_TTL:
            return jsonify(hit[1])
        try:
            payload = {"live": _live_now(season, detail)}
        except Exception as exc:
            return jsonify({"live": [], "error": str(exc)}), 500
        _live_cache[key] = (time.monotonic(), payload)
        return jsonify(payload)


LIVE_TTL    = 20            # seconds; a minute-by-minute clock loses nothing
_live_cache = {}
_live_lock  = threading.Lock()


def _live_now(season: str, detail: bool) -> list[dict]:
    from fixture_scraper import _fetch_all, _name_to_id, _team_name
    from pulse_results import _find_season_id, _season_year

    out = []
    # statuses="L" filters on Pulse's side: one small request, empty on the
    # (vast majority of) days with nothing being played.
    for f in _fetch_all(_find_season_id(_season_year(season)), statuses="L"):
        if f.get("status") != "L":
            continue
        sides = f.get("teams") or []
        if len(sides) != 2:
            continue
        ids = [_name_to_id(_team_name(s.get("team", {}))) for s in sides]
        if not all(ids):
            continue
        rec = {
            "homeId":    ids[0],
            "awayId":    ids[1],
            "homeGoals": int(sides[0].get("score") or 0),
            "awayGoals": int(sides[1].get("score") or 0),
            "clock":     (f.get("clock") or {}).get("label", ""),
            "clockSecs": (f.get("clock") or {}).get("secs"),
            "phase":     f.get("phase"),
        }
        # The stat blocks and shape cost an extra request per match, so
        # they are opt-in: the table only needs the score, Scout wants the
        # game as it unfolds.
        if detail:
            rec.update(_live_detail(int(f["id"]), ids))
        out.append(rec)
    return out


@app.route("/api/matchdetails")
def match_details_route():
    """Formations, goals/cards and a full stat comparison for every finished
    match — everything a full-time fixture card needs, in one call."""
    season = request.args.get("season", "2627")
    try:
        import lineups
        if request.args.get("refresh", "1") == "1":
            try:
                lineups.fetch(season)
            except Exception as exc:
                print(f"[matchdetails] refresh skipped: {exc}", file=sys.stderr)
        return jsonify({"matches": lineups.match_details(season)})
    except Exception as exc:
        return jsonify({"matches": {}, "error": str(exc)}), 500


@app.route("/api/formations")
def formations():
    """Starting shapes each team has used, and what they have faced."""
    season = request.args.get("season", "2627")
    try:
        import lineups
        # Incremental top-up, same pattern as /api/teamstats: fetch() only
        # pulls fixtures it has never cached, so this costs one archive read
        # plus any brand-new matches, not a rebuild.
        if request.args.get("refresh", "1") == "1":
            try:
                lineups.fetch(season)
            except Exception as exc:
                print(f"[formations] refresh skipped: {exc}", file=sys.stderr)
        teams = lineups.formations(season)
        league = defaultdict(int)
        for r in teams.values():
            for shape, n in r["shapes"].items():
                league[shape] += n
        return jsonify({
            "season": season,
            "teams":  teams,
            "league": dict(sorted(league.items(), key=lambda kv: -kv[1])),
        })
    except Exception as exc:
        return jsonify({"teams": {}, "error": str(exc)}), 500


@app.route("/api/styles")
def team_styles():
    """Playing-style fingerprint per 2026/27 team from cached ESPN stats."""
    try:
        from style_fingerprint import fingerprints
        return jsonify({"teams": fingerprints()})
    except Exception as exc:
        return jsonify({"teams": [], "error": str(exc)}), 500


@app.route("/api/hull/analysis/refresh/<league>", methods=["POST"])
def hull_analysis_refresh(league):
    """Re-fetch Hull City player data from ESPN for the given league."""
    try:
        from espn_players import fetch
        fetch(league, "2526")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
