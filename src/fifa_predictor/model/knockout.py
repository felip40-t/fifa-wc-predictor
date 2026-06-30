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

    The predictions frame ('match', 'score') may carry knockout-stage rows in
    addition to the group fixtures, so we keep only the same-group pairings: a
    fixture whose two teams sit in different groups is a knockout prediction and
    is skipped. For each group fixture we use the actual score from `results`
    when that (home, away) game has been played, else the predicted score. Each
    game is tagged with its group and a source of 'actual' or 'predicted'.
    """
    team_group = {t: letter for letter, teams in groups.items() for t in teams}
    actual = {
        (r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
        for r in results.itertuples()
        if pd.notna(r.home_score) and pd.notna(r.away_score)
    }
    games: list[Game] = []
    skipped = 0
    for row in predictions.itertuples():
        home, away = _parse_match(row.match)
        grp = team_group[home]
        if team_group[away] != grp:
            # Cross-group pairing -> knockout-stage prediction, not a group game.
            skipped += 1
            continue
        if (home, away) in actual:
            hs, as_ = actual[(home, away)]
            source = "actual"
        else:
            hs, as_ = (int(x) for x in str(row.score).split("-"))
            source = "predicted"
        games.append(Game(home, away, hs, as_, grp, source))
    if skipped:
        logger.info("Skipped %d non-group (knockout) prediction rows", skipped)
    return games


def _decide_winner(home: str, away: str, hs: int, as_: int, winner_cell: str | None) -> str:
    """Resolve the advancing team for one played knockout match.

    A decisive score advances the higher scorer; a `winner_cell` that contradicts
    the score is an error. A level score (the only outcome that goes to extra time
    or penalties) must name the advancing team in `winner_cell`, which has to be
    one of the two teams.
    """
    match = f"{home} {hs}-{as_} {away}"
    if hs != as_:
        scored = home if hs > as_ else away
        if winner_cell and winner_cell != scored:
            raise ValueError(
                f"winner '{winner_cell}' contradicts decisive score for {match}"
            )
        return scored
    if not winner_cell:
        raise ValueError(f"level knockout match needs a recorded winner: {match}")
    if winner_cell not in (home, away):
        raise ValueError(f"winner '{winner_cell}' is not in match {match}")
    return winner_cell


def attach_r32_results(r32: list[dict], results: pd.DataFrame) -> list[dict]:
    """Fill each resolved Round of 32 slot with its actual result, when played.

    Slots are matched to result rows by (home, away) team pairing. A matched slot
    gains `home_score`, `away_score`, `winner`, and `result_status` of 'played';
    an unmatched slot gets `result_status` of 'scheduled' and no score. Winners
    come from the score, except a level (extra time / penalties) match whose
    advancing team is read from the optional `winner` column.
    """
    has_winner_col = "winner" in results.columns
    played = {}
    for r in results.itertuples():
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        winner_cell = getattr(r, "winner", None) if has_winner_col else None
        if winner_cell is not None and (pd.isna(winner_cell) or str(winner_cell).strip() == ""):
            winner_cell = None
        played[(r.home_team, r.away_team)] = (int(r.home_score), int(r.away_score), winner_cell)

    resolved: list[dict] = []
    for slot in r32:
        out = dict(slot)
        key = (slot["home"], slot["away"])
        if key in played:
            hs, as_, winner_cell = played[key]
            out["home_score"] = hs
            out["away_score"] = as_
            out["winner"] = _decide_winner(slot["home"], slot["away"], hs, as_, winner_cell)
            out["result_status"] = "played"
        else:
            out["result_status"] = "scheduled"
        resolved.append(out)
    return resolved


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
    r32 = attach_r32_results(r32, results)
    n_played = sum(m["result_status"] == "played" for m in r32)
    logger.info("Filled %d of %d Round of 32 matches from results", n_played, len(r32))

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
