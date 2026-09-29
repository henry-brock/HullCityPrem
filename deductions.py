"""
deductions.py — points deductions, applied on top of results.

Stored in deductions.json, deliberately apart from data.json: the browser
rewrites data.json wholesale on every save, so a deduction kept there would be
silently dropped the first time a result was entered from a tab that loaded
before the deduction existed. Here there is one writer (the admin panel) and
any number of readers.

A deduction is a sanction, not a result, so it never touches data.json's
matches and never affects ATK/DEF — a team docked points is exactly as good at
football as it was the day before. It only moves the table.

Totals are allowed to go below zero. That is real (clubs have finished seasons
on negative points), and the frontend is written to cope with it; what is NOT
allowed is a deduction of zero or less, because a reduced sanction on appeal
is recorded by editing the original entry, not by stacking a negative one on
top of it — two entries that cancel out would read as two punishments.
"""

import json
import os
import secrets
from datetime import date
from pathlib import Path

ROOT   = Path(__file__).parent
PATH   = ROOT / "deductions.json"
SEASON = "2627"

MAX_POINTS = 120        # a season is worth 114; above that is a typo, not a sanction
MAX_REASON = 200


def team_names(season: str = SEASON) -> dict:
    """{internal id: display name} for the season's 20 clubs."""
    archive = ROOT / "Newcache" / f"pl_{season}.json"
    names = {}
    try:
        for m in json.loads(archive.read_text())["matches"]:
            if m.get("homeId"):
                names[m["homeId"]] = m["homeName"]
            if m.get("awayId"):
                names[m["awayId"]] = m["awayName"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    if len(names) < 20:
        # Early in a season Newcache may not have seen every club at home and
        # away yet; the Pulse team map always has all twenty.
        from lineups import _team_map
        for tid in _team_map(season).values():
            names.setdefault(tid, tid)
    return dict(sorted(names.items(), key=lambda kv: kv[1]))


def load() -> list[dict]:
    """Every stored deduction that still validates. A hand-edited file with a
    bad entry loses that entry, not the whole list."""
    try:
        raw = json.loads(PATH.read_text()).get("deductions", [])
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return []
    known = set(team_names())
    out = []
    for d in raw if isinstance(raw, list) else []:
        try:
            _check(d.get("team"), d.get("points"), d.get("reason", ""), known)
        except ValueError:
            continue
        out.append({"id": str(d.get("id") or secrets.token_hex(4)), "team": d["team"],
                    "points": int(d["points"]), "reason": str(d.get("reason", "")).strip(),
                    "added": d.get("added")})
    return out


def totals(entries: list[dict] | None = None) -> dict:
    out = {}
    for d in entries if entries is not None else load():
        out[d["team"]] = out.get(d["team"], 0) + d["points"]
    return out


def _check(team, points, reason, known) -> None:
    if team not in known:
        raise ValueError(f"unknown team '{team}'")
    if isinstance(points, bool) or not isinstance(points, (int, float)) or points != int(points):
        raise ValueError("points must be a whole number")
    if not 1 <= int(points) <= MAX_POINTS:
        raise ValueError(f"points must be between 1 and {MAX_POINTS}")
    if not isinstance(reason, str) or len(reason.strip()) > MAX_REASON:
        raise ValueError(f"reason must be text of at most {MAX_REASON} characters")


def _save(entries: list[dict]) -> None:
    # Write-then-rename so a crash mid-write leaves the old file intact rather
    # than a truncated one that load() would read as "no deductions".
    tmp = PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"deductions": entries}, indent=2))
    os.replace(tmp, PATH)


def add(team: str, points, reason: str = "") -> dict:
    _check(team, points, reason, set(team_names()))
    entry = {"id": secrets.token_hex(4), "team": team, "points": int(points),
             "reason": reason.strip(), "added": date.today().isoformat()}
    _save(load() + [entry])
    return entry


def update(entry_id: str, points=None, reason: str | None = None) -> dict:
    entries = load()
    for d in entries:
        if d["id"] == entry_id:
            new_points = d["points"] if points is None else points
            new_reason = d["reason"] if reason is None else reason
            _check(d["team"], new_points, new_reason, set(team_names()))
            d["points"], d["reason"] = int(new_points), new_reason.strip()
            _save(entries)
            return d
    raise KeyError(entry_id)


def remove(entry_id: str) -> None:
    entries = load()
    kept = [d for d in entries if d["id"] != entry_id]
    if len(kept) == len(entries):
        raise KeyError(entry_id)
    _save(kept)
