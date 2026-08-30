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
        sides = []
        for tl in fx.get("teamLists") or []:
            form = tl.get("formation") or {}
            if not form.get("players"):
                continue
            sides.append({
                "teamId":    tl.get("teamId"),
                "formation": form.get("label"),
                "rows":      [[int(p) for p in row] for row in form["players"]],
            })
        cached[key] = sides
        fetched += 1
        time.sleep(REQUEST_DELAY)
        if fetched % 50 == 0:
            path.write_text(json.dumps({"season": season, "fixtures": cached}))
            print(f"[lineups] {season}: {i}/{len(fixtures)}", file=sys.stderr)

    path.write_text(json.dumps({"season": season, "fixtures": cached}))
    print(f"[lineups] {season}: {len(cached)} matches ({fetched} fetched)", file=sys.stderr)
    return cached


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

    # Wing-backs: the wide players in the line immediately ahead of a back three.
    if line == 1 and back == 3 and width >= 4 and edge:
        return "FB"

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
    for sides in data.values():
        for side in sides:
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the per-match lineup cache")
    ap.add_argument("--seasons", default="2627")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in a.seasons.split(","):
        fetch(s.strip(), force=a.force)
