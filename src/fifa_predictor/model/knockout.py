"""Resolve group standings into a filled Round of 32 bracket."""

import argparse
import json
from pathlib import Path

import pandas as pd

from fifa_predictor.model.standings import (
    Game,
    TeamStanding,
    build_group_tables,
    rank_third_placed,
)
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

GROUPS_PATH = "data/reference/groups_world_cup_2026.json"
BRACKET_PATH = "data/reference/knockout_bracket.json"


def load_groups(path: str = GROUPS_PATH) -> dict[str, list[str]]:
    """Load the group-letter -> four-team mapping."""
    return json.loads(Path(path).read_text())["groups"]


def load_bracket(path: str = BRACKET_PATH) -> dict:
    """Load the predetermined bracket structure and allocation table."""
    return json.loads(Path(path).read_text())


def _parse_match(label: str) -> tuple[str, str]:
    """'Home vs Away' -> ('Home', 'Away')."""
    home, away = label.split(" vs ")
    return home.strip(), away.strip()


def resolve_group_games(
    results: pd.DataFrame, predictions: pd.DataFrame, groups: dict[str, list[str]]
) -> list[Game]:
    """Build the 72 group games, preferring actual results over predicted fills.

    The predictions frame ('match', 'score') is the complete fixture list. For
    each fixture we use the actual score from `results` when that (home, away)
    game has been played, else the predicted score. Each game is tagged with its
    group and a source of 'actual' or 'predicted'.
    """
    team_group = {t: letter for letter, teams in groups.items() for t in teams}
    actual = {
        (r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
        for r in results.itertuples()
        if pd.notna(r.home_score) and pd.notna(r.away_score)
    }
    games: list[Game] = []
    for row in predictions.itertuples():
        home, away = _parse_match(row.match)
        grp = team_group[home]
        if team_group[away] != grp:
            raise ValueError(f"cross-group fixture: {home} ({grp}) vs {away} ({team_group[away]})")
        if (home, away) in actual:
            hs, as_ = actual[(home, away)]
            source = "actual"
        else:
            hs, as_ = (int(x) for x in str(row.score).split("-"))
            source = "predicted"
        games.append(Game(home, away, hs, as_, grp, source))
    return games


def _group_complete(tables: dict[str, list[TeamStanding]], games: list[Game]) -> dict[str, bool]:
    """Per-group flag: True when all of that group's games are actual results."""
    complete: dict[str, bool] = {}
    for letter in tables:
        gg = [g for g in games if g.group == letter]
        complete[letter] = bool(gg) and all(g.source == "actual" for g in gg)
    return complete


def resolve_round_of_32(
    tables: dict[str, list[TeamStanding]],
    thirds: list[TeamStanding],
    bracket: dict,
    games: list[Game] | None = None,
) -> list[dict]:
    """Fill every Round of 32 slot with a concrete team.

    `1X`/`2X` come from a group's rank-1/rank-2 team; each `3rd:GROUPS` slot is
    resolved via the 495-row allocation table keyed on the eight groups whose
    third-placed team qualifies. When `games` is given, each slot gets a status
    of 'final' or 'projected'.
    """
    winners = {letter: rows[0].team for letter, rows in tables.items()}
    runners = {letter: rows[1].team for letter, rows in tables.items()}
    third_of = {letter: rows[2].team for letter, rows in tables.items()}

    best8 = thirds[:8]
    qualifying = sorted(s.group for s in best8)
    key = "".join(qualifying)
    alloc = bracket["third_place_allocations"][key]  # {match_number_str: group_letter}

    complete = _group_complete(tables, games) if games is not None else None
    all_complete = complete is not None and all(complete.values())

    def slot(token: str, match: int) -> tuple[str, str]:
        # Status is intentionally asymmetric: a winner/runner-up slot is final once
        # its own group is complete, but a 3rd-place slot depends on the global
        # best-eight ranking, so it is only final when every group is complete.
        if token.startswith("3rd:"):
            src_group = alloc[str(match)]
            return third_of[src_group], ("final" if all_complete else "projected")
        rank_char, letter = token[0], token[1]
        team = winners[letter] if rank_char == "1" else runners[letter]
        status = "final" if (complete is not None and complete[letter]) else "projected"
        return team, status

    resolved: list[dict] = []
    for m in bracket["bracket"]["round_of_32"]:
        home, home_status = slot(m["home"], m["match"])
        away, away_status = slot(m["away"], m["match"])
        resolved.append({
            "match": m["match"], "home": home, "away": away,
            "home_status": home_status, "away_status": away_status,
        })
    return resolved


def _standings_frame(tables: dict[str, list[TeamStanding]], games: list[Game]) -> pd.DataFrame:
    complete = _group_complete(tables, games)
    records = []
    for letter in sorted(tables):
        for s in tables[letter]:
            records.append({
                "group": s.group, "rank": s.rank, "team": s.team,
                "played": s.played, "won": s.won, "drawn": s.drawn, "lost": s.lost,
                "gf": s.gf, "ga": s.ga, "gd": s.gd, "points": s.points,
                "status": "final" if complete[letter] else "projected",
            })
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Resolve standings and the Round of 32 from saved results + predictions."""
    parser = argparse.ArgumentParser(description="Resolve group standings and fill the Round of 32.")
    parser.add_argument("competition", nargs="?", default="world_cup_2026")
    args = parser.parse_args()
    comp = args.competition

    groups = load_groups()
    bracket = load_bracket()
    results = pd.read_csv(f"data/raw/results_{comp}.csv")
    predictions = pd.read_csv(f"data/processed/predictions_{comp}.csv")

    games = resolve_group_games(results, predictions, groups)
    n_actual = sum(g.source == "actual" for g in games)
    logger.info("Resolved %d group games (%d actual, %d predicted)",
                len(games), n_actual, len(games) - n_actual)

    tables = build_group_tables(games, groups)
    thirds = rank_third_placed(tables)
    r32 = resolve_round_of_32(tables, thirds, bracket, games)

    standings_path = Path(f"data/processed/group_standings_{comp}.csv")
    standings_path.parent.mkdir(parents=True, exist_ok=True)
    _standings_frame(tables, games).to_csv(standings_path, index=False)
    logger.info("Wrote standings -> %s", standings_path)

    qualifying = sorted(s.group for s in thirds[:8])
    out = {
        "competition": bracket["competition"],
        "qualifying_third_place_groups": qualifying,
        "allocation_key": "".join(qualifying),
        "round_of_32": r32,
    }
    bracket_path = Path(f"data/processed/knockout_bracket_resolved_{comp}.json")
    bracket_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote resolved bracket -> %s", bracket_path)


if __name__ == "__main__":
    main()
