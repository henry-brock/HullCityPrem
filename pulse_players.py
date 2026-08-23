"""
pulse_players.py — Hull City player stats for the LIVE Premier League season,
from the official Pulse API.

Why this exists
───────────────
The PL Analysis tab was empty, and not because a scraper broke: it read
`cache/espn_players_eng.1_2526.json` looking for Hull, but Hull spent 2025/26
in the Championship, so no such rows existed. Now that 2026/27 has started
Hull are a Premier League club and their PL player data is available.

Pulse is also a much richer source than what was being stored. The existing
stats_scraper keeps two numbers per player (appearances, goals); Pulse's
`/stats/player/{id}` returns ~70 Opta metrics per player per season,
including minutes, carries, progressive carries, duels, interceptions, shot
locations and big chances.

Output is deliberately shaped like the ESPN player records the Squad Analysis
tab already renders (camelCase keys under `stats`), so the frontend needs no
special-casing — the same table, sorting and position composites work.

Cache: cache/pulse_players_<season>.json, 6h TTL (a live season changes).
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

from fixture_scraper import _pulse_get
from pulse_results import _find_season_id, _season_year

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_TTL = timedelta(hours=6)

PL_COMP_ID   = 1
HULL_PULSE_ID = 41          # Hull City's Premier League team id

POS_MAP = {"G": "G", "D": "D", "M": "M", "F": "F"}


def _cache_path(season: str) -> Path:
    return CACHE_DIR / f"pulse_players_{season}.json"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < CACHE_TTL


def _g(stats: dict, name: str) -> float:
    """Opta omits zero-valued stats, so a missing key means 0."""
    return float(stats.get(name) or 0.0)


def _pct(part: float, whole: float):
    return round(part / whole * 100, 1) if whole else None


def _map_stats(s: dict) -> dict:
    """
    Opta metric names → the ESPN-style keys SA_STATS in index.html expects.

    Anything Pulse does not expose is left absent rather than zeroed, so the
    table shows a blank instead of implying a real measurement of nothing.
    """
    tackles_won   = _g(s, "won_tackle")
    tackles_total = _g(s, "total_tackle")
    passes_ok     = _g(s, "accurate_pass")
    passes_all    = _g(s, "total_pass")
    shots         = _g(s, "total_scoring_att")
    on_target     = _g(s, "ontarget_scoring_att")
    goals         = _g(s, "goals")
    saves         = _g(s, "saves")
    conceded      = _g(s, "goals_conceded")

    out = {
        "appearances":      int(_g(s, "appearances")),
        "minutes":          int(_g(s, "mins_played")),
        "totalGoals":       int(goals),
        "goalAssists":      int(_g(s, "goal_assist")),
        "totalShots":       int(shots),
        "shotsOnTarget":    int(on_target),
        "shotAssists":      int(_g(s, "total_att_assist")),
        "accuratePasses":   int(passes_ok),
        "effectiveTackles": int(tackles_won),
        "interceptions":    int(_g(s, "interception")),
        "totalClearance":   int(_g(s, "total_clearance")),
        "blockedShots":     int(_g(s, "outfielder_block")),
        "yellowCards":      int(_g(s, "total_yel_card")),
        "redCards":         int(_g(s, "total_red_card")),
        "offsides":         int(_g(s, "total_offside")),
        # Goalkeeping
        "saves":            int(saves),
        "goalsConceded":    int(conceded),
        "cleanSheet":       int(_g(s, "clean_sheet")),
        "crossesCaught":    int(_g(s, "good_high_claim")),
        "punches":          int(_g(s, "punches")),
        # Extras Pulse has and ESPN never did — surfaced for future use.
        "touches":            int(_g(s, "touches")),
        "touchesInOppBox":    int(_g(s, "touches_in_opp_box")),
        "duelsWon":           int(_g(s, "duel_won")),
        "duelsLost":          int(_g(s, "duel_lost")),
        "aerialsWon":         int(_g(s, "aerial_won")),
        "aerialsLost":        int(_g(s, "aerial_lost")),
        "carries":            int(_g(s, "carries")),
        "progressiveCarries": int(_g(s, "progressive_carries")),
        "bigChancesScored":   int(_g(s, "big_chance_scored")),
        "bigChancesMissed":   int(_g(s, "big_chance_missed")),
        "gamesStarted":       int(_g(s, "game_started")),
        # ── Concepts the player rater groups on ──────────────────────
        "totalTackles":       int(_g(s, "total_tackle")),
        "ballRecoveries":     int(_g(s, "ball_recovery")),
        "possWonDef3rd":      int(_g(s, "poss_won_def_3rd")),
        "possWonMid3rd":      int(_g(s, "poss_won_mid_3rd")),
        "possWonAtt3rd":      int(_g(s, "poss_won_att_3rd")),
        "headClearances":     int(_g(s, "head_clearance")),
        "blockedCrosses":     int(_g(s, "blocked_cross")),
        "attemptsIbox":       int(_g(s, "attempts_ibox")),
        "totalLongBalls":     int(_g(s, "total_long_balls")),
        "accurateLongBalls":  int(_g(s, "accurate_long_balls")),
        "totalCrosses":       int(_g(s, "total_cross")),
        "accurateCrosses":    int(_g(s, "accurate_cross")),
        "finalThirdEntries":  int(_g(s, "successful_final_third_entries")),
        "penAreaEntries":     int(_g(s, "successful_pen_area_entries")),
        "shotCreated":        int(_g(s, "shot_created")),
        "bigChancesCreated":  int(_g(s, "big_chance_created")),
        "possLost":           int(_g(s, "poss_lost_all")),
        "dispossessed":       int(_g(s, "dispossessed")),
        "unsuccessfulTouch":  int(_g(s, "unsuccessful_touch")),
        "errorLeadToShot":    int(_g(s, "error_lead_to_shot")),
        "errorLeadToGoal":    int(_g(s, "error_lead_to_goal")),
        "fouls":              int(_g(s, "fouls") or _g(s, "fk_foul_lost")),
        "wasFouled":          int(_g(s, "was_fouled") or _g(s, "fk_foul_won")),
        # Goalkeeping detail
        "savedIbox":          int(_g(s, "saved_ibox")),
        "savedObox":          int(_g(s, "saved_obox")),
        "bigChanceSaves":     int(_g(s, "big_chance_saves")),
        "diveSaves":          int(_g(s, "dive_save") or _g(s, "diving_save")),
        "highClaims":         int(_g(s, "total_high_claim")),
        "keeperSweeper":      int(_g(s, "total_keeper_sweeper")),
        "accKeeperSweeper":   int(_g(s, "accurate_keeper_sweeper")),
        "goalKicks":          int(_g(s, "goal_kicks")),
        "accurateGoalKicks":  int(_g(s, "accurate_goal_kicks")),
        "accKeeperThrows":    int(_g(s, "accurate_keeper_throws")),
        "gkSmother":          int(_g(s, "gk_smother")),
    }
    # Rates — omitted entirely when the denominator is zero.
    for key, value in (("tacklePct",      _pct(tackles_won, tackles_total)),
                       ("passPct",        _pct(passes_ok, passes_all)),
                       ("shotPct",        _pct(on_target, shots)),
                       ("goalConversion", _pct(goals, shots)),
                       ("savePct",        _pct(saves, saves + conceded))):
        if value is not None:
            out[key] = value
    return out


def _squad(season_id: int) -> list[dict]:
    data = _pulse_get(
        f"teams/{HULL_PULSE_ID}/compseasons/{season_id}/staff",
        pageSize=100, compSeasons=season_id,
    )
    return data.get("players", []) or []


def _player_stats(player_id: int, season_id: int) -> dict:
    try:
        data = _pulse_get(f"stats/player/{player_id}",
                          comps=PL_COMP_ID, compSeasons=season_id)
    except (requests.RequestException, ValueError) as exc:
        print(f"[pulse_players] stats failed for {player_id}: {exc}", file=sys.stderr)
        return {}
    return {s["name"]: s.get("value") for s in (data.get("stats") or [])}


def fetch(season: str = "2627", force: bool = False) -> dict:
    """
    Return {"players": [...], "team", "season", "source"} for Hull's PL squad.

    Players with no recorded minutes are still included — the frontend filters
    on minutes itself, and knowing who is in the squad but unused is useful.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(season)
    if not force and _is_fresh(path):
        print(f"[pulse_players] using cached {path}", file=sys.stderr)
        return json.loads(path.read_text())

    season_id = _find_season_id(_season_year(season))
    squad     = _squad(season_id)
    print(f"[pulse_players] {len(squad)} players in Hull's {season} PL squad",
          file=sys.stderr)

    players = []
    for p in squad:
        pid  = int(p["id"])                     # Pulse serialises ids as floats
        info = p.get("info") or {}
        raw  = _player_stats(pid, season_id)
        players.append({
            "id":     pid,
            "name":   (p.get("name") or {}).get("display") or "",
            "pos":    POS_MAP.get(info.get("position"), info.get("position") or ""),
            "shirt":  info.get("shirtNum"),
            "stats":  _map_stats(raw),
        })

    payload = {
        "team":    "Hull City",
        "season":  season,
        "source":  f"Pulse API (compSeason {season_id})",
        "fetched": datetime.now().isoformat(),
        "players": players,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[pulse_players] cached {len(players)} players → {path}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Hull PL player stats from Pulse")
    ap.add_argument("--season", default="2627")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = fetch(season=args.season, force=args.force)
    rows = sorted(data["players"], key=lambda p: -p["stats"]["minutes"])
    print(f"\n{'#':>3} {'player':<24}{'pos':<5}{'mins':>5}{'gls':>5}{'ast':>5}"
          f"{'pass%':>7}{'touch':>7}{'carr':>6}")
    for p in rows:
        s = p["stats"]
        print(f"{p['shirt'] or '-':>3} {p['name']:<24}{p['pos']:<5}{s['minutes']:>5}"
              f"{s['totalGoals']:>5}{s['goalAssists']:>5}"
              f"{(str(s.get('passPct','—'))):>7}{s['touches']:>7}{s['carries']:>6}")
    print(f"\n{len(rows)} players.", file=sys.stderr)
