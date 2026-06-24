"""Group-stage standings: tables and third-place ranking under FIFA tiebreakers."""

from dataclasses import dataclass

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

WIN_POINTS, DRAW_POINTS = 3, 1


@dataclass
class Game:
    home: str
    away: str
    home_score: int
    away_score: int
    group: str
    source: str  # "actual" or "predicted"


@dataclass
class TeamStanding:
    team: str
    group: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    gf: int = 0
    ga: int = 0
    gd: int = 0
    points: int = 0
    rank: int = 0


def _blank_table(group: str, teams: list[str]) -> dict[str, "TeamStanding"]:
    return {t: TeamStanding(team=t, group=group) for t in teams}


def _apply_game(table: dict[str, "TeamStanding"], g: "Game") -> None:
    """Fold one game's result into a group's standings (in place)."""
    h, a = table[g.home], table[g.away]
    h.played += 1
    a.played += 1
    h.gf += g.home_score
    h.ga += g.away_score
    a.gf += g.away_score
    a.ga += g.home_score
    if g.home_score > g.away_score:
        h.won += 1
        a.lost += 1
        h.points += WIN_POINTS
    elif g.home_score < g.away_score:
        a.won += 1
        h.lost += 1
        a.points += WIN_POINTS
    else:
        h.drawn += 1
        a.drawn += 1
        h.points += DRAW_POINTS
        a.points += DRAW_POINTS
    for s in (h, a):
        s.gd = s.gf - s.ga


def _h2h_points(team: str, tied: set[str], games: list["Game"]) -> tuple[int, int, int]:
    """(points, goal difference, goals for) for `team` in games among `tied` only."""
    pts = gf = ga = 0
    for g in games:
        if g.home in tied and g.away in tied and team in (g.home, g.away):
            ts, os_ = (g.home_score, g.away_score) if g.home == team else (g.away_score, g.home_score)
            gf += ts
            ga += os_
            if ts > os_:
                pts += WIN_POINTS
            elif ts == os_:
                pts += DRAW_POINTS
    return pts, gf - ga, gf


def _ordered(standings: list["TeamStanding"], games: list["Game"]) -> list["TeamStanding"]:
    """Order a group's standings by the FIFA tiebreak chain.

    Overall points -> GD -> GF, then head-to-head points -> GD -> GF among the
    teams still tied, then a deterministic fallback (team name) in place of
    FIFA's fair-play/drawing-of-lots, with any such ties logged.
    """
    # Stage 1: overall criteria (points -> GD -> GF), best first.
    by_overall = sorted(standings, key=lambda s: (s.points, s.gd, s.gf), reverse=True)
    # Stage 2: within blocks tied on (points, gd, gf), apply head-to-head then name.
    # Head-to-head is a single pass here, not FIFA's full recursive re-application
    # of the whole chain within each still-tied sub-group. At four teams per group
    # the difference is rare, and the deterministic name fallback keeps any
    # remaining ties stable.
    result: list[TeamStanding] = []
    i = 0
    while i < len(by_overall):
        j = i + 1
        key = (by_overall[i].points, by_overall[i].gd, by_overall[i].gf)
        while j < len(by_overall) and (by_overall[j].points, by_overall[j].gd, by_overall[j].gf) == key:
            j += 1
        block = by_overall[i:j]
        if len(block) > 1:
            tied = {s.team for s in block}
            block.sort(key=lambda s: s.team)  # deterministic fallback last
            block.sort(key=lambda s: _h2h_points(s.team, tied, games), reverse=True)
            h2h = {s.team: _h2h_points(s.team, tied, games) for s in block}
            if len({h2h[s.team] for s in block}) < len(block):
                logger.info("Group %s: teams broken only by deterministic fallback: %s",
                            block[0].group, [s.team for s in block])
        result.extend(block)
        i = j
    for rank, s in enumerate(result, start=1):
        s.rank = rank
    return result


def build_group_tables(games: list["Game"], groups: dict[str, list[str]]) -> dict[str, list["TeamStanding"]]:
    """Build ordered, rank-filled standings for every group."""
    tables: dict[str, list[TeamStanding]] = {}
    for letter, teams in groups.items():
        group_games = [g for g in games if g.group == letter]
        table = _blank_table(letter, teams)
        for g in group_games:
            _apply_game(table, g)
        tables[letter] = _ordered(list(table.values()), group_games)
    return tables


def rank_third_placed(tables: dict[str, list["TeamStanding"]]) -> list["TeamStanding"]:
    """The twelve third-placed teams, best first (points -> GD -> GF -> name)."""
    thirds = [tables[letter][2] for letter in sorted(tables)]
    thirds.sort(key=lambda s: s.team)
    thirds.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
    if len({(s.points, s.gd, s.gf) for s in thirds}) < len(thirds):
        logger.info("Third-place ranking has ties broken by deterministic fallback")
    return thirds
