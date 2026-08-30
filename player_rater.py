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
import sys
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
    # A holding midfielder's job is to break play up and give it to someone
    # else. Attacking output is a bonus, not the brief, so Creation and
    # Progression together are held under 10% — the previous single midfield
    # group gave them 37%, which judged a destroyer on a playmaker's work.
    # Defending is here and absent from CM: shielding the back four means
    # blocking shots and clearing, which a No. 10 is not asked to do.
    "DM": {"Ball-winning": 5, "Duels": 4, "Passing": 3, "Defending": 2,
           "Security": 2, "Progression": 0.5, "Creation": 0.5},
    # Central and attacking midfield — the job is control: keep it, move it
    # forward, make chances, win it back. The possession weighting is right
    # here in a way it was not for holding players, now they are rated apart.
    # Defending is here for the same reason it is in DM: a central midfielder
    # blocks shots and clears in front of his back four, and the concept was
    # simply absent. Adding it is what improved the rating's season-to-season
    # stability (0.600 -> 0.639); merely reweighting toward duels and
    # ball-winning without it did not (0.593), which is a different claim from
    # "more defensive" and the one the evidence actually supports.
    #
    # The attacking side was then cut further (27% -> 20%), because the old
    # weighting had been set for a pool that still contained attacking
    # midfielders. Within their shared pool the genuine central midfielders
    # sit at 67 for Defending and 59 for Duels against 46 and 43 for the
    # attacking ones, so scoring them at 10% and 16% undersold the job.
    # Repeatability 0.678 -> 0.706. It stops there rather than going further:
    # at 62% defensive the group becomes indistinguishable from DM's 64%, and
    # two roles with the same weighting are not two roles.
    "CM": {"Ball-winning": 4, "Duels": 3.5, "Defending": 3, "Passing": 3,
           "Progression": 2, "Creation": 2, "Security": 2},
    # A No. 10 — the central player in an advanced midfield band. Creates
    # first, scores second, and is asked for less defensive work than the
    # players behind him.
    "AM": {"Creation": 4, "Passing": 3, "Progression": 3, "Shooting": 2,
           "Security": 2, "Duels": 1.5, "Ball-winning": 1.5},
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
                "DM": "Screening", "CM": "Control", "AM": "Creating",
                "W": "Wing play", "ST": "Goal threat"}

# The rating groups. Pulse reports a coarse position (G/D/M/F) and a detailed
# role ("Left Full Back", "Centre Central Defender"); the detailed one is what
# separates the two defensive jobs.
ROLES = ("G", "CB", "FB", "DM", "CM", "AM", "W", "ST")

# Roles the formation mapping distinguishes but the rating pools cannot.
#
# A wing-back in a back three is a real and separate job — he starts higher
# than a full-back because there are three centre-backs behind him instead of
# two — and lineups.py labels him correctly, which the Formations view uses.
# The rating cannot follow: only four wing-backs a season clear the minutes
# floor, giving percentiles in 25-point steps, well below the 16-19 that was
# already too thin to justify an attacking-midfield group.
#
# Note this never touched a back FIVE. There the wide players sit in the
# defensive line and have always been full-backs, which is the right answer:
# a 5-4-1's wide men defend where a 3-4-3's attack.
ROLE_ALIASES = {"WB": "FB"}

# Full-backs cross; centre-backs do not. Used only for the handful of players
# Pulse files as a bare "Defender" — the league medians are 2.5 crosses per 90
# for a full-back and 0.0 for a centre-back, so this is a wide gap to fall in.
FB_CROSS_PER90 = 1.5


# Reclassifying a midfielder by how he plays rather than how he is filed.
#
# The label is the prior and it is a good one: across 24/25 -> 25/26 a
# player's Pulse role was identical 100% of the time, while a purely
# data-driven classification agreed only 83-91% of the time. Overriding on
# every disagreement would therefore swap a stable signal for a noisier one,
# and it does not pay for itself — rating repeatability was 0.607 on labels
# alone and 0.603-0.622 across every override threshold tried, all inside the
# +/-0.09 noise band on 60-odd players.
#
# So the override fires only when the data is emphatic. Matt Crooks scores
# +2.20 on this index, ahead of every player the feed actually files as a
# defensive midfielder, while being listed as a central midfielder; rating him
# against playmakers is simply wrong. At these thresholds the classification
# still agrees with itself across seasons 92% of the time.
ROLE_INDEX_DEF = ["effectiveTackles", "interceptions", "totalClearance",
                  "blockedShots", "duelsWon", "aerialsWon"]
ROLE_INDEX_ATT = ["shotAssists", "bigChancesCreated", "totalShots",
                  "goalAssists", "touchesInOppBox"]
ROLE_PROMOTE = 1.5     # labelled CM, plays as a screener -> DM
ROLE_DEMOTE  = -1.0    # labelled DM, plays as a creator  -> CM


def _role_index(players: list[dict], floor: float) -> None:
    """
    Store a defensive-minus-attacking z index on each qualified midfielder.

    Only players over the same minutes floor the leaderboards use take part:
    a per-90 built from a twenty-minute cameo says nothing about a role.
    """
    mids = [p for p in players
            if "Midfielder" in (p.get("posInfo") or "")
            and (p.get("stats", {}).get("minutes") or 0) >= floor]
    for p in players:
        p["_roleIndex"] = None
    if len(mids) < 8:
        return

    def per90(stats, key):
        mins = stats.get("minutes") or 0
        return (stats.get(key) or 0) / mins * 90 if mins else 0.0

    zs = {}
    for key in ROLE_INDEX_DEF + ROLE_INDEX_ATT:
        col = [per90(p["stats"], key) for p in mids]
        mu  = _mean(col)
        sd  = (_mean([(v - mu) ** 2 for v in col])) ** 0.5 or 1.0
        zs[key] = [(v - mu) / sd for v in col]

    for i, p in enumerate(mids):
        d = _mean([zs[k][i] for k in ROLE_INDEX_DEF])
        a = _mean([zs[k][i] for k in ROLE_INDEX_ATT])
        p["_roleIndex"] = d - a


# Before where a player LINED UP overrides how he is filed, it has to be where
# he USUALLY lines up. The threshold is not there to catch bad data: Marc Guéhi
# really did play a midfield role in one odd game, and the feed recorded it
# correctly. It is there because one unusual afternoon should not decide which
# peer group rates his whole season — on two starts a 1-1 split is a coin flip,
# and it put a centre-back top of the defensive-midfield table.
#
# Requiring two starts and a clear majority keeps the genuine cases, where a
# player is being used somewhere his label does not say: Foden started twice on
# the left of a 4-2-3-1's three, Gakpo twice wide while filed as a striker.
# The share resolves itself as a season fills out, so this is strictest now and
# loosens naturally.
MIN_STARTS_FOR_ROLE = 2
MIN_ROLE_SHARE      = 0.6
# A role has to account for this share of a player's starts before it is
# blended in — otherwise one emergency afternoon at left-back colours a whole
# season. 15% is about four or five starts across a full one. Nothing caps the
# number of roles: at this threshold 23 players in 25/26 are rated across three
# or four. A fifth was too strict, discarding 11% of a hybrid's starts against
# 7% here, and finding only half as many genuine three-role players.
MIN_BLEND_SHARE     = 0.15

_DERIVED_CACHE: dict = {}


def _derived_roles(season: str) -> dict:
    """player id -> role, from the starting formations he appeared in."""
    if season not in _DERIVED_CACHE:
        try:
            from lineups import derive
            _DERIVED_CACHE[season] = derive(season)
        except Exception as exc:                       # no lineup cache yet
            print(f"[player_rater] derived roles unavailable: {exc}", file=sys.stderr)
            _DERIVED_CACHE[season] = {}
    return _DERIVED_CACHE[season]


def _attach_derived(players: list[dict], season: str) -> None:
    """
    Attach the role a player is rated in, and — for genuine utility players —
    the mix he is rated ACROSS.

    Forcing one role on a man who played four is the flaw the whole exercise
    was meant to fix. Over a full season 47% of regular starters play a single
    role and 84% have a clear majority, but the rest are properly positionless:
    one 25/26 regular started 36 games as FB 8, AM 14, W 6, DM 6, CM 1, ST 1.
    Picking his commonest role would rate two thirds of his football in the
    wrong company, and falling back to the label just swaps one single answer
    for another. Those players are scored against every role they actually
    played and the results blended by how often they played it — which matches
    the fact that their STATS are already a blend.
    """
    derived = _derived_roles(season)
    for p in players:
        d = derived.get(p.get("id"))
        p["_derivedRole"] = p["_roleMix"] = None
        p["_hybrid"] = False
        p["_aliasRoles"] = []
        if not d or d["starts"] < MIN_STARTS_FOR_ROLE:
            continue
        merged = {}
        for r, n in d["mix"].items():
            merged[ROLE_ALIASES.get(r, r)] = merged.get(ROLE_ALIASES.get(r, r), 0) + n
        total = sum(merged.values())
        # Which aliased roles were folded away, for players who did enough of
        # one to be worth saying so. A full-back who spent a third of his
        # season as a wing-back is rated among full-backs because that is the
        # only pool big enough, but the fact should not vanish with the label.
        p["_aliasRoles"] = sorted(
            r for r, n in d["mix"].items()
            if r in ROLE_ALIASES and n / total >= MIN_BLEND_SHARE)
        mix = {r: n / total for r, n in merged.items()
               if n / total >= MIN_BLEND_SHARE}
        if not mix:
            continue
        top = max(mix, key=lambda r: mix[r])
        p["_derivedRole"] = top
        p["_hybrid"] = mix[top] < MIN_ROLE_SHARE and len(mix) > 1
        p["_roleMix"] = mix if p["_hybrid"] else None


def _role(player: dict) -> str:
    """
    Which peer group a player is rated against.

    Driven by Pulse's detailed role, because the coarse position is not
    reliable for this: of the league's wingers, 17 are filed under midfield
    and 67 under forward. Grouping on the coarse field would rate a winger
    against Haaland or against a holding midfielder depending on which way
    the feed happened to file him.
    """
    # Where he actually started, when we know it. A season label describes one
    # position; the stats being rated were accumulated across every position he
    # was picked in, which is why Foden — filed "Centre/Right Attacking
    # Midfielder", started twice on the left of a 4-2-3-1's three — was being
    # rated against central midfielders on a winger's numbers.
    if player.get("_derivedRole"):
        return player["_derivedRole"]

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
        labelled_dm = "Defensive Midfielder" in info
        idx = player.get("_roleIndex")
        if idx is not None:
            if not labelled_dm and idx >= ROLE_PROMOTE:
                return "DM"
            if labelled_dm and idx <= ROLE_DEMOTE:
                return "CM"
        return "DM" if labelled_dm else "CM"

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


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


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
    _role_index(players, floor)
    _attach_derived(players, season)

    # Per-90, context-adjusted value of one stat for one player. Pulled out so
    # a player can be scored against a group he is not a member of, which is
    # what blending a hybrid requires.
    def adjusted(p, key, cdef):
        v = _per90(p.get("stats", {}), key)
        if v is not None and key not in RATE_STATS:
            scale = (ctx.get(p.get("team"), {}) or {}).get(cdef["ctx"], 1.0)
            v = v / scale if scale else v
        return v

    # Peer groups, keyed on each player's PRIMARY role. A hybrid belongs to the
    # pool of the role he played most, so the pools stay clean, but he is
    # scored against several.
    groups = {}
    for pos in ROLES:
        concepts = CONCEPTS_GK if pos == "G" else CONCEPTS_OUT
        peers = [p for p in players
                 if _role(p) == pos and (p.get("stats", {}).get("minutes") or 0) > 0]
        if not peers:
            continue
        cols = {}
        for cdef in concepts.values():
            for key in cdef["stats"]:
                cols[key] = [adjusted(p, key, cdef) for p in peers]
        groups[pos] = {"peers": peers, "cols": cols, "concepts": concepts}

    def score_against(p, pos):
        """Concept percentiles and a weighted rating for `p` in role `pos`."""
        g, weights = groups[pos], WEIGHTS[pos]
        scores, wsum, wtot = {}, 0.0, 0.0
        for cname, cdef in g["concepts"].items():
            pcts = []
            for key in cdef["stats"]:
                pct = _percentile(adjusted(p, key, cdef), g["cols"][key])
                if cdef.get("neg"):
                    pct = 100 - pct
                pcts.append(pct)
            if not pcts:
                continue
            raw = sum(pcts) / len(pcts)
            scores[cname] = raw
            w = weights.get(cname, 0)
            if w:
                wsum += raw * w
                wtot += w
        return scores, (wsum / wtot if wtot else None)

    by_pos = {}
    for pos in groups:
        rows = []
        for p in groups[pos]["peers"]:
            mins   = p.get("stats", {}).get("minutes") or 0
            shrink = mins / (mins + SHRINK_K)

            # Which roles to score him in, and how heavily.
            mix = {r: w for r, w in (p.get("_roleMix") or {}).items() if r in groups}
            # Goalkeeping has its own concepts entirely, so it never blends.
            if pos == "G" or "G" in mix:
                mix = {}
            if not mix:
                mix = {pos: 1.0}
            total = sum(mix.values())

            parts = {r: score_against(p, r) for r in mix}
            names = set().union(*[set(sc) for sc, _ in parts.values()])
            # Concepts keep their RAW percentile — they describe what the player
            # has actually done, and flattening them would hide it. The
            # confidence penalty is applied once, to the headline rating.
            scores = {n: round(sum(w * parts[r][0].get(n, 50.0)
                                   for r, w in mix.items()) / total, 1)
                      for n in names}
            got = [(w, parts[r][1]) for r, w in mix.items() if parts[r][1] is not None]
            raw_rating = (sum(w * v for w, v in got) / sum(w for w, _ in got)
                          if got else None)

            hybrid = len(mix) > 1
            label  = ("HYBRID (" + ", ".join(sorted(mix, key=lambda r: -mix[r])) + ")"
                      if hybrid else RATING_LABEL[pos])

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
                "label":     label,
                # The group's own name, so a section heading is not taken from
                # whichever player happens to top it.
                "roleLabel": RATING_LABEL[pos],
                "hybrid":    hybrid,
                # Roles merged into this one for pool-size reasons, e.g. a
                # full-back who plays a good share of his football at wing-back.
                "aliasRoles": p.get("_aliasRoles") or None,
                "roleMix":   {r: round(w / total, 2) for r, w in mix.items()} if hybrid else None,
                "concepts":  scores,
                "peers":     len(groups[pos]["peers"]),
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
    _augment(players)
    all_mins = [(p.get("stats", {}).get("minutes") or 0) for p in players]
    _role_index(players, max(all_mins or [0]) * MIN_MINUTES_SHARE)
    _attach_derived(players, "2627")

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
