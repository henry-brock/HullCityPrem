"""
player_rater.py — position-aware player ratings for Hull City.

Design notes (the reasoning matters more than the weights)
─────────────────────────────────────────────────────────
1. TWO-LEVEL, never a flat average of raw stats. Opta's columns are wildly
   uneven — ~33 passing variants against 4 clearance columns — so averaging
   raw stats would weight passing about eight times more than defending purely
   because Opta subdivides passing more finely. Raw stats are therefore first
   averaged WITHIN a concept, and only concepts are weighted against each
   other. (Note this is the opposite of the team-level composite, where a flat
   average worked precisely because all eight stats measured one thing.)

2. GOALKEEPERS ARE SCORED ON THEIR OWN AXES. 18 stats are GK-only and 41 are
   outfield-only, so the two cannot share a scale at all.

3. RATED AGAINST LEAGUE POSITIONAL PEERS, not team-mates. Ranking a defender
   against 15 Hull players says who is Hull's best; ranking against every PL
   defender says whether that is any good.

4. TEAM CONTEXT IS DIVIDED OUT. Hull's defenders lead the league in clearances
   because Hull defend ~70% of the time — that rates the team, not the player.
   Defensive volume is scaled by the defensive load a team faces, attacking
   volume by the team's attacking volume.

5. MINUTES SHRINKAGE INSTEAD OF A CUTOFF. Percentiles are pulled toward 50 by
   m/(m+K), so a player with 60 minutes reads as "roughly average, unknown"
   rather than being either wrongly extreme or hidden behind an N/A.

What this can and cannot claim: there is no ground truth for player quality,
so a composite is a defensible opinion. It supports "does more of X than
positional peers", not "is better".
"""

import json
from pathlib import Path

ROOT  = Path(__file__).parent
CACHE = ROOT / "cache"

SHRINK_K = 450          # minutes at which a percentile keeps half its distance from 50

# ── Concepts: raw stat keys grouped by what they actually measure ──────────
# `neg` concepts are ones where MORE is worse; their percentile is inverted.
CONCEPTS_OUT = {
    # iboxShare is "does he shoot from good positions", which repeats (0.75).
    # Conversion — whether those shots go in — does not (0.15) and is absent.
    "Shooting":     {"stats": ["totalShots", "shotsOnTarget", "attemptsIbox",
                               "totalGoals", "bigChancesScored", "touchesInOppBox",
                               "iboxShare"],
                     "ctx": "att"},
    "Creation":     {"stats": ["shotAssists", "goalAssists", "bigChancesCreated",
                               "shotCreated", "penAreaEntries", "accurateCrosses"],
                     "ctx": "att"},
    "Passing":      {"stats": ["accuratePasses", "passPct", "accurateLongBalls"],
                     "ctx": "pass"},
    "Progression":  {"stats": ["carries", "progressiveCarries", "finalThirdEntries"],
                     "ctx": "att"},
    "Ball-winning": {"stats": ["effectiveTackles", "interceptions", "ballRecoveries",
                               "possWonDef3rd", "possWonMid3rd", "possWonAtt3rd"],
                     "ctx": "def"},
    "Defending":    {"stats": ["totalClearance", "headClearances", "blockedShots",
                               "blockedCrosses"], "ctx": "def"},
    # Volume AND success rate: how often he contests, and whether he wins.
    "Duels":        {"stats": ["duelsWon", "aerialsWon",
                               "duelWinPct", "aerialWinPct"], "ctx": "def"},
    "Security":     {"stats": ["possLost", "dispossessed", "unsuccessfulTouch",
                               "errorLeadToShot", "fouls", "turnoverRate"],
                     "ctx": "att", "neg": True},
}

CONCEPTS_GK = {
    "Shot-stopping": {"stats": ["saves", "savePct", "savedIbox", "bigChanceSaves",
                                "diveSaves"], "ctx": "sot"},
    "Claiming":      {"stats": ["crossesCaught", "highClaims", "punches", "gkSmother"],
                      "ctx": "cross"},
    "Sweeping":      {"stats": ["keeperSweeper", "accKeeperSweeper"], "ctx": "def"},
    "Distribution":  {"stats": ["passPct", "accurateLongBalls", "accurateGoalKicks",
                                "accKeeperThrows"], "ctx": "pass"},
    "Conceding":     {"stats": ["goalsConceded"], "ctx": "sot", "neg": True},
}

# Positional relevance — how much a concept matters to doing THAT job well.
# This part is judgement and is meant to be: a striker is judged on shooting,
# a centre-back on defending.
BASE_WEIGHTS = {
    "G": {"Shot-stopping": 5, "Claiming": 2, "Sweeping": 1, "Distribution": 2,
          "Conceding": 3},
    # Centre-backs and full-backs are rated apart, because they are different
    # jobs: the league's full-backs cross 2.5 times a game at the median and
    # its centre-backs cross zero. Judging them together scored a full-back on
    # heading and a centre-back on crossing, and neither is the work.
    #
    # A centre-back is rated on stopping the ball — clearances, blocks, duels,
    # winning it back. Security is small but present: an error leading to a
    # shot and a needless foul are defensive failures, not on-ball ones.
    # Passing is kept deliberately light. A centre-back who can play does have
    # value, but weighting it heavily meant one at a 60%-possession side
    # out-rated one at a 30%-possession side on the strength of touches he was
    # simply given rather than anything he did defending.
    "CB": {"Defending": 5, "Duels": 4, "Ball-winning": 3, "Security": 1,
           "Passing": 1},
    # A full-back defends and attacks, so the weight is split. Creation covers
    # the crossing and overlapping that is genuinely part of the job.
    "FB": {"Defending": 3, "Duels": 3, "Ball-winning": 3, "Creation": 3,
           "Progression": 2, "Security": 1, "Passing": 1},
    # A midfielder's job is control and creation: keep it, move it forward,
    # make chances, win it back.
    # Central midfield — the job is control: keep it, move it forward, make
    # chances, win it back. Defensive, central and attacking central mids are
    # one group for now: only 8 defensive midfielders clear the minutes bar,
    # and a peer group that small gives percentiles in 12-point steps.
    "CM": {"Passing": 4, "Creation": 4, "Progression": 3, "Ball-winning": 3,
           "Duels": 2, "Security": 2},
    # A winger creates and scores, and is judged much less on winning the ball
    # back than a central midfielder is.
    "W":  {"Creation": 4, "Shooting": 3, "Progression": 3, "Passing": 1,
           "Duels": 1, "Security": 1, "Ball-winning": 0.5},
    # A centre-forward is judged on goals, and on the chances he makes for
    # others. Almost nothing else is the job.
    "ST": {"Shooting": 5, "Creation": 2, "Progression": 1, "Security": 0.5},
}

# How much of a concept carries from one season to the next, measured on 167
# outfielders who stayed at the same club across 2024/25 → 2025/26 with ≥600
# minutes in each (player_repeatability.py). Normalised so the most repeatable
# concept scores 1.0.
#
# These MODULATE the positional weights rather than replacing them —
# repeatability is not importance. Passing repeats at r=0.86 for everybody,
# which does not make it the right way to judge a striker. Multiplying keeps
# the positional logic while trusting flaky concepts less.
#
# Progression is deliberately 1.0 (no adjustment): carries, progressive
# carries and final-third entries were not collected in 2024/25, so it is
# UNMEASURED, not unrepeatable. Penalising it would be penalising missing data.
#
# Goalkeepers are left unmodulated — only 9 kept at the same club cleared the
# minutes bar, which is too thin to reweight on. Worth noting their
# shot-stopping came out at r=0.46, pointing the same way as the team-level
# finding that save efficiency barely persists (r=0.038); revisit with more
# seasons.
REPEATABILITY = {
    "Passing": 1.00, "Duels": 0.93, "Shooting": 0.92, "Defending": 0.89,
    "Ball-winning": 0.87, "Security": 0.82, "Creation": 0.74,
    "Progression": 1.00,
}

WEIGHTS = {
    pos: {c: round(w * (1.0 if pos == "G" else REPEATABILITY.get(c, 1.0)), 2)
          for c, w in base.items()}
    for pos, base in BASE_WEIGHTS.items()
}

RATING_LABEL = {"G": "Goalkeeping", "CB": "Defending", "FB": "Full-back play",
                "CM": "Control", "W": "Wing play", "ST": "Goal threat"}

# The rating groups. Pulse reports a coarse position (G/D/M/F) and a detailed
# role ("Left Full Back", "Centre Central Defender"); the detailed one is what
# separates the two defensive jobs.
ROLES = ("G", "CB", "FB", "CM", "W", "ST")

# Full-backs cross; centre-backs do not. Used only for the handful of players
# Pulse files as a bare "Defender" — the league medians are 2.5 crosses per 90
# for a full-back and 0.0 for a centre-back, so this is a wide gap to fall in.
FB_CROSS_PER90 = 1.5


def _role(player: dict) -> str:
    """
    Which peer group a player is rated against.

    Driven by Pulse's detailed role, because the coarse position is not
    reliable for this: of the league's wingers, 17 are filed under midfield
    and 67 under forward. Grouping on the coarse field would rate a winger
    against Haaland or against a holding midfielder depending on which way
    the feed happened to file him.
    """
    pos  = player.get("pos") or ""
    if pos == "G":
        return "G"
    info = player.get("posInfo") or ""
    # Order matters: "Left/Centre/Right Attacking Midfielder" must land in
    # midfield, and "Second Striker" up front.
    if "Full Back" in info or "Wing Back" in info:
        return "FB"
    if "Central Defender" in info or "Centre Back" in info:
        return "CB"
    if "Winger" in info:
        return "W"
    if "Striker" in info or "Forward" in info:
        return "ST"
    if "Midfielder" in info:
        return "CM"

    # Bare "Defender"/"Midfielder"/"Forward" with no detail — fall back to the
    # coarse position, splitting defenders on crossing volume.
    if pos == "D":
        stats = player.get("stats") or {}
        mins  = stats.get("minutes") or 0
        if mins:
            per90 = (stats.get("totalCrosses") or 0) / mins * 90
            return "FB" if per90 >= FB_CROSS_PER90 else "CB"
        return "CB"
    return "ST" if pos == "F" else "CM"

# Stats already expressed as a rate — must not be divided by minutes again.
RATE_STATS = {"passPct", "savePct", "tacklePct", "shotPct", "goalConversion",
               "duelWinPct", "aerialWinPct", "turnoverRate", "iboxShare"}


# Derived per-opportunity rates, injected into every player's stat block.
#
# The concepts were built almost entirely from VOLUME: duelsWon but never
# duelsLost, so a defender who contests ten duels and wins eight scores the
# same as one who contests ten and wins three. That measures how often a
# player engages, which is largely a function of how his team plays, rather
# than whether he wins when he does.
#
# Only rates that survive the repeatability test are here (player_repeatability
# on 176 same-club players, 2024/25 → 2025/26):
#
#   turnoverRate   r=0.92     duelWinPct    r=0.85     iboxShare  r=0.75
#   aerialWinPct   r=0.72
#
# Deliberately EXCLUDED, having failed the same test: tackle success (0.13),
# goal conversion (0.15), shot accuracy (0.30) and cross accuracy (-0.05).
# These are the player-level echo of the team-level finding that volume and
# territory persist while conversion efficiency does not — a defender's 100%
# tackle record over two games is noise, and weighting it would be building
# the rating out of luck.
DERIVED_RATES = {
    "duelWinPct":   lambda s: _share(s.get("duelsWon"),
                                     (s.get("duelsWon") or 0) + (s.get("duelsLost") or 0)),
    "aerialWinPct": lambda s: _share(s.get("aerialsWon"),
                                     (s.get("aerialsWon") or 0) + (s.get("aerialsLost") or 0)),
    # Expressed as loss rather than retention so it sits in Security, whose
    # percentiles are inverted; the two are the same measurement.
    "turnoverRate": lambda s: _share(s.get("possLost"), s.get("touches")),
    "iboxShare":    lambda s: _share(s.get("attemptsIbox"), s.get("totalShots")),
}


def _share(num, den):
    if num is None or not den:
        return None
    return 100.0 * float(num) / float(den)


def _augment(players: list[dict]) -> None:
    """Add the derived rates to each player's stat block, in place."""
    for p in players:
        st = p.get("stats")
        if not st:
            continue
        for key, fn in DERIVED_RATES.items():
            st[key] = fn(st)


def _per90(stats: dict, key: str):
    v = stats.get(key)
    if v is None:
        return None
    if key in RATE_STATS:
        return float(v)
    mins = stats.get("minutes") or 0
    return float(v) / mins * 90 if mins > 0 else None


def _percentile(value, pool) -> float:
    """Share of the pool this value beats, 0-100."""
    vals = [v for v in pool if v is not None]
    if value is None or len(vals) < 3:
        return 50.0
    below = sum(1 for v in vals if v < value)
    equal = sum(1 for v in vals if v == value)
    return (below + equal / 2) / len(vals) * 100


# What a concept's counting stats should be normalised BY. One blanket
# "defensive load" figure is wrong for several of them: a keeper's claims
# should be judged per cross faced, not per shot faced, and Hull face roughly
# double the league's crosses while facing only slightly more shots — so a
# shots-based divisor barely corrects the volume and leaves claims looking
# inflated. Each key maps to the opponent (or own) stat that represents the
# opportunity count.
CONTEXT_STATS = {
    "att":   ("own", "total_scoring_att"),      # team attacking volume
    "def":   ("opp", "total_scoring_att"),      # shots faced
    "sot":   ("opp", "ontarget_scoring_att"),   # shots on target faced
    "cross": ("opp", "total_cross"),            # crosses faced
    "pass":  ("own", "total_pass"),             # team passing volume
}


def team_context(season: str = "2627") -> dict:
    """
    Per-team volume on each context axis, relative to the league mean.

    A value > 1 means the team sees more of that thing than average, so its
    players' related counts are deflated to compensate. Missing teams and
    missing axes fall back to 1.0 (no adjustment).
    """
    path = ROOT / "Newcache" / f"pl_{season}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())

    acc = {axis: {} for axis in CONTEXT_STATS}
    for m in data.get("matches", []):
        for side, other in (("home", "away"), ("away", "home")):
            tid = m.get(f"{side}Id")
            if not tid:
                continue
            blocks = {"own": m["stats"].get(side) or {},
                      "opp": m["stats"].get(other) or {}}
            for axis, (which, stat) in CONTEXT_STATS.items():
                acc[axis].setdefault(tid, []).append(float(blocks[which].get(stat) or 0))

    if not acc["def"]:
        return {}
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    out = {}
    for axis, per_team in acc.items():
        league = mean([mean(v) for v in per_team.values()]) or 1.0
        for tid, vals in per_team.items():
            out.setdefault(tid, {})[axis] = (mean(vals) / league) or 1.0
    return out


# A leaderboard needs a floor: with two games played, a 20-minute substitute
# who touched the ball twice can post a spectacular per-90. Expressed as a
# share of the most-used player's minutes so it scales with the season instead
# of needing to be re-tuned — 40% is roughly "a regular", not "an ever-present".
MIN_MINUTES_SHARE = 0.4


def _rate_pool(players: list[dict], season: str = "2627",
               adjust_context: bool = True) -> dict:
    """
    Rate EVERY player in the pool against their league positional peers.

    The percentile work was always league-wide; the previous version computed
    it across all peers and then discarded everyone outside the target team.
    That discarded figure is exactly what a league ranking needs, so this
    keeps it and lets the callers filter.
    """
    ctx = team_context(season) if adjust_context else {}
    _augment(players)

    all_mins = [(p.get("stats", {}).get("minutes") or 0) for p in players]
    floor    = max(all_mins or [0]) * MIN_MINUTES_SHARE

    by_pos = {}
    for pos in ROLES:
        concepts = CONCEPTS_GK if pos == "G" else CONCEPTS_OUT
        weights  = WEIGHTS[pos]

        peers = [p for p in players
                 if _role(p) == pos and (p.get("stats", {}).get("minutes") or 0) > 0]
        if not peers:
            continue

        # Per-90, context-adjusted values for every peer, per stat.
        vals = {}
        for cname, cdef in concepts.items():
            for key in cdef["stats"]:
                col = []
                for p in peers:
                    v = _per90(p.get("stats", {}), key)
                    if v is not None and key not in RATE_STATS:
                        scale = (ctx.get(p.get("team"), {}) or {}).get(cdef["ctx"], 1.0)
                        v = v / scale if scale else v
                    col.append(v)
                vals[key] = col

        rows = []
        for i, p in enumerate(peers):
            mins   = p.get("stats", {}).get("minutes") or 0
            shrink = mins / (mins + SHRINK_K)

            # Concepts keep their RAW percentile — they describe what the player
            # has actually done, and flattening them would hide it. The
            # confidence penalty is applied once, to the headline rating.
            scores, wsum, wtot = {}, 0.0, 0.0
            for cname, cdef in concepts.items():
                pcts = []
                for key in cdef["stats"]:
                    pct = _percentile(vals[key][i], vals[key])
                    if cdef.get("neg"):
                        pct = 100 - pct
                    pcts.append(pct)
                if not pcts:
                    continue
                raw = sum(pcts) / len(pcts)
                scores[cname] = round(raw, 1)
                w = weights.get(cname, 0)
                if w:
                    wsum += raw * w
                    wtot += w

            raw_rating = wsum / wtot if wtot else None
            rows.append({
                "id":        p.get("id"),
                "name":      p.get("name"),
                "team":      p.get("team"),
                "pos":       pos,
                "posInfo":   p.get("posInfo") or "",
                "shirt":     p.get("shirt"),
                "minutes":   mins,
                # Headline: pulled toward average by how little we have seen.
                "rating":    round(50 + (raw_rating - 50) * shrink, 1) if raw_rating is not None else None,
                "ratingRaw": round(raw_rating, 1) if raw_rating is not None else None,
                "label":     RATING_LABEL[pos],
                "concepts":  scores,
                "peers":     len(peers),
                "confidence": round(shrink * 100),
                "qualified": mins >= floor and raw_rating is not None,
            })

        # League order: the SHRUNK rating, so a player cannot top the table on
        # 20 minutes. Within the squad view the raw figure still does the
        # sorting, because there the whole point is to compare team-mates who
        # have all played a similar amount.
        rows.sort(key=lambda r: -(r["rating"] if r["rating"] is not None else -1))
        rank = 0
        for r in rows:
            if r["qualified"]:
                rank += 1
                r["leagueRank"] = rank
            else:
                r["leagueRank"] = None
        qualified = rank
        for r in rows:
            r["leagueOf"] = qualified
            # Percentile against every peer who has played at all, qualified
            # or not — "better than N% of the league's midfielders".
            r["leaguePct"] = round(_percentile(
                r["ratingRaw"], [q["ratingRaw"] for q in rows]), 1) \
                if r["ratingRaw"] is not None else None

        by_pos[pos] = rows

    return by_pos


def rate(players: list[dict], target_team: str = "hull", season: str = "2627",
         adjust_context: bool = True) -> list[dict]:
    """
    Rate `target_team`'s players against league positional peers.

    `players` is the whole-league pool (pl_players.fetch()["players"]). Returns
    the target team's players with per-concept percentiles, an overall rating,
    and where that rating places them in the league at their position.
    """
    out = [dict(r)
           for rows in _rate_pool(players, season, adjust_context).values()
           for r in rows if r.get("team") == target_team]

    # Rank on the unshrunk figure: early in a season the shrunk ratings are all
    # ~50 by design, so sorting on them would order the squad by minutes played.
    out.sort(key=lambda p: -(p["ratingRaw"] if p["ratingRaw"] is not None else 0))
    # Rank within the squad, alongside the league-relative rating.
    for i, p in enumerate(out, 1):
        p["squadRank"] = i
    return out


def leaderboard(players: list[dict], season: str = "2627", top: int = 10,
                adjust_context: bool = True) -> dict:
    """Top `top` qualified players at each position, league-wide."""
    return {pos: [r for r in rows if r["qualified"]][:top]
            for pos, rows in _rate_pool(players, season, adjust_context).items()}


# ── Per-opportunity player metrics ────────────────────────────────────────
# Counting stats mostly measure how much a player was involved. These divide
# each one by the involvement that produced it, so a full-back who touches the
# ball 90 times is not automatically ahead of one who touches it 40. Every
# metric applies to every outfield position — a centre-back's shot rate is as
# legitimate a number as a striker's, it is just smaller.
#
# `pct` marks a ratio that is already a percentage; `neg` marks one where a
# high value is worse.
PLAYER_DERIVED = [
    ("Touches / 90",            lambda s: _rate(s.get("touches"), s.get("minutes"), 90), {}),
    ("Pass completion %",       lambda s: s.get("passPct"), {"pct": True}),
    ("Poss. lost / 100 touches", lambda s: _ratio(s.get("possLost"), s.get("touches"), 100), {"neg": True}),
    ("Duel win %",              lambda s: _ratio(s.get("duelsWon"),
                                                 (s.get("duelsWon") or 0) + (s.get("duelsLost") or 0), 100), {"pct": True}),
    ("Aerial win %",            lambda s: _ratio(s.get("aerialsWon"),
                                                 (s.get("aerialsWon") or 0) + (s.get("aerialsLost") or 0), 100), {"pct": True}),
    ("Tackle success %",        lambda s: s.get("tacklePct"), {"pct": True}),
    ("Clearances / 100 touches", lambda s: _ratio(s.get("totalClearance"), s.get("touches"), 100), {}),
    ("Ball wins / 100 touches", lambda s: _ratio((s.get("interceptions") or 0) + (s.get("ballRecoveries") or 0),
                                                 s.get("touches"), 100), {}),
    ("Shots / 100 touches",     lambda s: _ratio(s.get("totalShots"), s.get("touches"), 100), {}),
    ("Box touches / 100 touches", lambda s: _ratio(s.get("touchesInOppBox"), s.get("touches"), 100), {}),
    ("Chances / 100 passes",    lambda s: _ratio(s.get("shotAssists"), s.get("accuratePasses"), 100), {}),
    ("Goals / 10 shots on target", lambda s: _ratio(s.get("totalGoals"), s.get("shotsOnTarget"), 10), {}),
]


def _rate(num, mins, per):
    if num is None or not mins:
        return None
    return round(float(num) / mins * per, 2)


def _ratio(num, den, scale):
    if num is None or not den:
        return None
    return round(float(num) / float(den) * scale, 2)


def player_derived(players: list[dict], target_team: str = "hull") -> dict:
    """
    Per-opportunity ratios for a team's players, with each value's percentile
    against league peers IN THE SAME ROLE so the numbers are comparable — a
    winger's crossing judged against wingers, not against centre-backs.
    """
    by_pos = {}
    for p in players:
        if (p.get("stats", {}).get("minutes") or 0) > 0:
            by_pos.setdefault(_role(p), []).append(p)

    rows = []
    for p in players:
        if p.get("team") != target_team or (p.get("stats", {}).get("minutes") or 0) <= 0:
            continue
        peers = by_pos.get(_role(p), [])
        vals, pcts = {}, {}
        for name, fn, opts in PLAYER_DERIVED:
            v = fn(p["stats"])
            vals[name] = v
            pool = [fn(q["stats"]) for q in peers]
            pct = _percentile(v, pool)
            if opts.get("neg"):
                pct = 100 - pct
            pcts[name] = round(pct) if v is not None else None
        rows.append({
            "id": p.get("id"), "name": p.get("name"), "pos": _role(p),
            "shirt": p.get("shirt"), "minutes": p["stats"].get("minutes"),
            "values": vals, "pcts": pcts, "peers": len(peers),
        })
    rows.sort(key=lambda r: -(r["minutes"] or 0))
    return {
        "players": rows,
        "metrics": [{"name": n, **o} for n, _f, o in PLAYER_DERIVED],
    }


if __name__ == "__main__":
    import argparse
    from pl_players import fetch as fetch_pl

    ap = argparse.ArgumentParser(description="Rate Hull players vs PL peers")
    ap.add_argument("--season", default="2627")
    ap.add_argument("--team", default="hull")
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()

    pool = fetch_pl(season=args.season)["players"]
    rows = rate(pool, target_team=args.team, season=args.season,
                adjust_context=not args.no_context)
    print(f"\n{'#':>2} {'player':<24}{'pos':<4}{'min':>4}{'rating':>7}  concepts (percentile vs PL peers)")
    for p in rows:
        cs = " ".join(f"{k[:4]} {v:>4.0f}" for k, v in sorted(p["concepts"].items(),
                                                              key=lambda kv: -kv[1])[:4])
        print(f"{p['squadRank']:>2} {p['name']:<24}{p['pos']:<4}{p['minutes']:>4}"
              f"{p['rating'] if p['rating'] is not None else '—':>7}  {cs}")
    print(f"\n{len(rows)} players rated against league positional peers.")
