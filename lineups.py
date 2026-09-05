"""
lineups.py — per-match starting formations, and the positions they imply.

Why
───
Every role in the Player Rater came from Pulse's `positionInfo`, which is a
SEASON-LONG label, while a player's stats are accumulated wherever he actually
played. Phil Foden is filed as a "Centre/Right Attacking Midfielder" and
started 2026/27 in the left slot of a 4-2-3-1's attacking three — a wide role —
so he was rated against central midfielders on numbers he built as a winger.
Splitting the label more finely cannot fix a label that is wrong.

Pulse gives the fix away in the fixture payload:

    "formation": {"label": "5-4-1",
                  "players": [[gk], [5 defenders], [4 midfielders], [1 forward]]}

Rows run back to front and players run ACROSS each row, so a starter's line and
his slot within it are both recoverable — enough to separate a left winger from
a No. 10, which is the distinction the season label loses.

Mapped by SHAPE, never by the formation's label
───────────────────────────────────────────────
A 4-4-2 and a 4-2-3-1 both field four midfielders and they are not the same
job: the first is W CM CM W, the second DM DM behind W AM W. What matters is
which line a player occupied, how wide that line was, and whether he stood on
the end of it. Two consequences worth naming:

  • Two midfield lines mean the deeper one is holding. One midfield line
    leaves depth unknowable from shape, so it stays central midfield.
  • A back THREE turns the wide players in front of it into wing-backs, not
    wingers — they defend a flank rather than attack one, and rating them on
    creation would ignore most of their work.

Limits: starters only, since substitutes are absent from the formation rows;
and a player moved at half time is recorded where he started.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from pl_players import _pulse_get

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"
REQUEST_DELAY = 0.12


def _cache_path(season: str) -> Path:
    return CACHE / f"lineups_{season}.json"


def fetch(season: str = "2627", force: bool = False) -> dict:
    """{fixtureId: [{teamId, formation, rows}]} for every completed match."""
    CACHE.mkdir(exist_ok=True)
    path = _cache_path(season)
    cached = {}
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text()).get("fixtures", {})
        except json.JSONDecodeError:
            cached = {}

    archive = ROOT / "Newcache" / f"pl_{season}.json"
    if not archive.exists():
        raise FileNotFoundError(f"no Newcache archive for {season}")
    fixtures = [m["fixtureId"] for m in json.loads(archive.read_text())["matches"]]

    fetched = 0
    for i, fid in enumerate(fixtures, 1):
        key = str(fid)
        if key in cached:
            continue
        try:
            fx = _pulse_get(f"fixtures/{fid}")
        except Exception as exc:
            print(f"[lineups] {fid} failed: {exc}", file=sys.stderr)
            continue
        sides, players = [], {}
        for tl in fx.get("teamLists") or []:
            form = tl.get("formation") or {}
            for pl in (tl.get("lineup") or []) + (tl.get("substitutes") or []):
                if pl.get("id"):
                    players[str(int(pl["id"]))] = (pl.get("name") or {}).get("display")
            if not form.get("players"):
                continue
            sides.append({
                "teamId":    tl.get("teamId"),
                "formation": form.get("label"),
                "rows":      [[int(p) for p in row] for row in form["players"]],
                # Opta's own coarse position for that match (G/D/M/F). Kept so
                # the shape mapping can be audited without re-fetching: it is
                # the only independent check on whether a slot was read right.
                "pos":       {str(int(pl["id"])): pl.get("matchPosition")
                              for pl in (tl.get("lineup") or []) if pl.get("id")},
            })
        # Goals and bookings — the fixture payload already carries a full
        # events feed and this endpoint is being fetched anyway for the
        # formation, so capturing it costs nothing extra. A finished match's
        # events do not change, unlike its live stat block, so this is fetched
        # once and never revisited.
        events = [
            {"type": e.get("type"), "teamId": e.get("teamId"),
             "personId": e.get("personId"), "clock": (e.get("clock") or {}).get("label"),
             "desc": e.get("description")}
            for e in (fx.get("events") or []) if e.get("type") in ("G", "B")
        ]
        cached[key] = {"sides": sides, "events": events, "players": players}
        fetched += 1
        time.sleep(REQUEST_DELAY)
        if fetched % 50 == 0:
            path.write_text(json.dumps({"season": season, "fixtures": cached}))
            print(f"[lineups] {season}: {i}/{len(fixtures)}", file=sys.stderr)

    path.write_text(json.dumps({"season": season, "fixtures": cached}))
    print(f"[lineups] {season}: {len(cached)} matches ({fetched} fetched)", file=sys.stderr)
    return cached


def _sides(rec) -> list:
    """A fixture's per-side formation records, from either cache shape.

    Older cache files stored a bare list; fetch() now stores a dict alongside
    events and player names. Reading through this means the format could
    change again without a forced rebuild being the only way forward.
    """
    return rec if isinstance(rec, list) else (rec or {}).get("sides", [])


def slot_role(rows: list[list[int]], row_i: int, slot_i: int) -> str:
    """Map a starter's place in the formation to a rating role."""
    if row_i == 0:
        return "G"

    line  = row_i - 1                     # 0 = defensive line
    lines = len(rows) - 1                 # outfield lines
    width = len(rows[row_i])
    back  = len(rows[1]) if len(rows) > 1 else 4
    edge  = slot_i == 0 or slot_i == width - 1

    if line == 0:
        # A back four or five has full-backs outside it; a back three does not.
        return "FB" if (width >= 4 and edge) else "CB"

    if line == lines - 1:
        # A front three is a winger either side of a centre-forward; a front
        # one or two is central.
        return "W" if (width >= 3 and edge) else "ST"

    # Wing-backs: the wide players in the line immediately ahead of a back
    # three. Their own role, not full-backs: one man covers the whole flank
    # with three centre-backs behind him rather than two, so he starts higher
    # and is asked for more of the attacking width. Opta files them as
    # midfielders, which is the single systematic disagreement between this
    # mapping and its coarse position field.
    if line == 1 and back == 3 and width >= 4 and edge:
        return "WB"

    mid_lines = lines - 2
    if mid_lines >= 2 and line == 1:
        return "DM"
    if mid_lines >= 2 and line == lines - 2:
        return "W" if (width >= 3 and edge) else "AM"

    # One midfield line. Depth is not directly observable, so the line BEHIND
    # it decides: a back five means the midfield is screening it rather than
    # playing box to box, which is the whole point of a 5-4-1 — those two
    # central players are holding players, not shuttlers.
    if width >= 4 and edge:
        return "W"
    return "DM" if back >= 5 else "CM"


def derive(season: str = "2627") -> dict:
    """player id -> {"role": commonest start role, "starts": n, "share", "mix"}."""
    try:
        data = json.loads(_cache_path(season).read_text()).get("fixtures", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    tally = {}
    for rec in data.values():
        for side in _sides(rec):
            rows = side["rows"]
            for r, row in enumerate(rows):
                for s, pid in enumerate(row):
                    tally.setdefault(pid, Counter())[slot_role(rows, r, s)] += 1
    out = {}
    for pid, c in tally.items():
        role, n = c.most_common(1)[0]
        total = sum(c.values())
        out[pid] = {"role": role, "starts": total,
                    "share": round(n / total, 2), "mix": dict(c)}
    return out


def _team_map(season: str) -> dict:
    """{pulseTeamId: internal id}, cached — the fixtures walk is not free."""
    path = CACHE / f"team_ids_{season}.json"
    if path.exists():
        try:
            return {int(k): v for k, v in json.loads(path.read_text()).items()}
        except (json.JSONDecodeError, ValueError):
            pass
    from pl_players import _teams
    from pulse_results import _find_season_id, _season_year
    out = _teams(_find_season_id(_season_year(season)))
    path.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def formations(season: str = "2627") -> dict:
    """
    Per-team shape usage: what each side lined up in, and what it faced.

    The shapes a team picks are a blunter statement of intent than any derived
    metric — Hull have started both matches in a 5-4-1, which is the same fact
    the possession and PPDA numbers report at one remove.
    """
    try:
        data = json.loads(_cache_path(season).read_text()).get("fixtures", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    names = _team_map(season)

    teams = {}
    for rec in data.values():
        sides = _sides(rec)
        for i, side in enumerate(sides):
            tid = names.get(side.get("teamId"))
            if not tid:
                continue
            rec = teams.setdefault(tid, {"used": Counter(), "faced": Counter(),
                                         "back": [], "front": [], "matches": 0})
            rec["used"][side["formation"]] += 1
            rec["matches"] += 1
            rows = side["rows"]
            if len(rows) > 2:
                rec["back"].append(len(rows[1]))
                rec["front"].append(len(rows[-1]))
            for j, other in enumerate(sides):
                if j != i:
                    rec["faced"][other["formation"]] += 1

    out = {}
    for tid, r in teams.items():
        common, n = r["used"].most_common(1)[0]
        out[tid] = {
            "matches":  r["matches"],
            "shapes":   dict(r["used"].most_common()),
            "faced":    dict(r["faced"].most_common()),
            "common":   common,
            "commonN":  n,
            "variety":  len(r["used"]),
            "avgBack":  round(sum(r["back"]) / len(r["back"]), 2) if r["back"] else None,
            "avgFront": round(sum(r["front"]) / len(r["front"]), 2) if r["front"] else None,
        }
    return out


# Same stat set /api/live shows for a match in progress. Centralised here
# rather than duplicated in app.py, since a finished match's detail panel and
# a live one are the same nine numbers off the same kind of Opta block —
# Newcache's for a finished game, a fresh Pulse call for one still being played.
MATCH_STATS = [
    ("possession_percentage", "Possession %"),
    ("total_scoring_att",     "Shots"),
    ("ontarget_scoring_att",  "On target"),
    ("attempts_ibox",         "Shots in box"),
    ("big_chance_created",    "Big chances"),
    ("touches_in_opp_box",    "Touches in box"),
    ("total_cross",           "Crosses"),
    ("won_corners",           "Corners"),
    ("total_pass",            "Passes"),
    ("total_tackle",          "Tackles"),
    ("total_clearance",       "Clearances"),
]


def _clock_key(event: dict) -> int:
    """Sort key for a clock label like "45+3'00" — stoppage time sorts after
    the minute it extends, not before it as a plain string sort would."""
    label = (event.get("clock") or "0'00").split("'")[0]
    base, _, extra = label.partition("+")
    try:
        return int(base or 0) * 100 + int(extra or 0)
    except ValueError:
        return 0


def match_details(season: str = "2627") -> dict:
    """
    Formations, goals/cards and a full stat comparison for every completed
    match this season, keyed "homeId:awayId" — everything a full-time fixture
    card needs, in one call rather than one round trip per finished game.

    The stats come straight from Newcache's own per-match Opta block, already
    fetched and cached for the team-stats and derived-stats pages, so this
    adds no network cost beyond the formations/events walk fetch() already
    does. A match missing from the lineup cache (not yet fetched, or from
    before lineups.py existed) still gets its stats, just no formations or
    events — degrading gracefully rather than dropping the whole card.
    """
    archive = ROOT / "Newcache" / f"pl_{season}.json"
    if not archive.exists():
        return {}
    matches = json.loads(archive.read_text())["matches"]
    try:
        cache = json.loads(_cache_path(season).read_text()).get("fixtures", {})
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    names = _team_map(season)

    out = {}
    for m in matches:
        rec = cache.get(str(m["fixtureId"]))
        formations_by_team, events = {}, []
        if rec:
            for side in _sides(rec):
                tid = names.get(side.get("teamId"))
                if tid:
                    formations_by_team[tid] = side.get("formation")
            if isinstance(rec, dict):
                players = rec.get("players", {})
                for e in rec.get("events", []):
                    tid = names.get(e.get("teamId"))
                    if not tid:
                        continue
                    events.append({
                        "type":   "goal" if e["type"] == "G" else
                                  ("red" if e.get("desc") == "R" else "yellow"),
                        "team":   tid,
                        "player": players.get(str(e.get("personId"))) or "?",
                        "clock":  e.get("clock"),
                    })
                events.sort(key=_clock_key)

        home_stats = (m["stats"] or {}).get("home") or {}
        away_stats = (m["stats"] or {}).get("away") or {}
        out[f"{m['homeId']}:{m['awayId']}"] = {
            "formations": formations_by_team,
            "events":     events,
            "stats": {
                "labels": [lab for _, lab in MATCH_STATS],
                "home":   [home_stats.get(k) for k, _ in MATCH_STATS],
                "away":   [away_stats.get(k) for k, _ in MATCH_STATS],
            },
        }
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the per-match lineup cache")
    ap.add_argument("--seasons", default="2627")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in a.seasons.split(","):
        fetch(s.strip(), force=a.force)
