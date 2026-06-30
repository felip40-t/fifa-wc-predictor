"""Human-readable rendering of the predictor's processed outputs.

Turns the hard-to-scan processed files into aligned, at-a-glance tables:
the simulated_outcomes CSV (goal rates and scoreline probabilities), the
group_standings CSV, and the resolved knockout-bracket JSON.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fifa_predictor.model.knockout import BRACKET_PATH
from fifa_predictor.model.simulate import select_headline_score
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

# Round numbers occupied by the Round of 32 (the bracket's team-bearing leaves).
_R32_MATCHES = range(73, 89)
# Vertical rows between consecutive team leaves. Even spacing keeps every parent
# match's mid-point row a whole number, so connectors land on exact rows.
_LEAF_SPACING = 2

# Above this 1X2 + over/under inversion residual, the implied goal rates fit the
# bookmaker prices poorly; such rows are flagged so readers treat them with care.
POOR_FIT_THRESHOLD = 0.05

# How many ranked scoreline columns (score_1..score_N) the CSV carries.
_TOP_SCORELINES = 3


def _pct(value: float) -> str:
    """Render a 0-1 probability as a one-decimal percentage, e.g. 0.688 -> '68.8%'."""
    return f"{value * 100:.1f}%"


def _top_scores(row: "pd.Series") -> str:
    """Join a row's ranked scorelines into 'score freq' parts, skipping blanks."""
    parts = []
    for rank in range(1, _TOP_SCORELINES + 1):
        scoreline = row.get(f"score_{rank}")
        if pd.isna(scoreline) or scoreline == "":
            continue
        parts.append(f"{scoreline} {_pct(row[f'score_{rank}_freq'])}")
    return "   ".join(parts)


def format_simulated_outcomes(
    source: "str | Path | pd.DataFrame", played_ids: "set | None" = None
) -> str:
    """Build an aligned, human-readable table of simulated match outcomes.

    Args:
        source: Either an already-loaded summary DataFrame (as returned by
            simulate_games_from_odds) or a path to the simulated_outcomes CSV.
        played_ids: Optional set of game_ids already played. When supplied, those
            rows are dropped so the table shows only games yet to be played.

    Returns:
        A multi-line string: one row per game with the matchup, implied goal
        rates, win/draw/away percentages, the result-consistent most likely
        score (LIKELY), the expected rounded-goal-rate score (EXP), and the
        three most likely scorelines each with their frequency. Poorly fit
        games are marked with a trailing
        '*' and explained in a footnote. The game-id hash is intentionally
        omitted.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if played_ids:
        df = df[~df["game_id"].isin(played_ids)].reset_index(drop=True)

    matchups = [f"{home} vs {away}" for home, away in zip(df["home_team"], df["away_team"])]
    flagged = [residual > POOR_FIT_THRESHOLD for residual in df["residual_norm"]]
    labels = [matchup + (" *" if flag else "") for matchup, flag in zip(matchups, flagged)]

    match_w = max([len("MATCHUP"), *(len(label) for label in labels)])

    header = (
        f"{'MATCHUP':<{match_w}}  {'λH':>4}  {'λA':>4}  "
        f"{'HOME':>6}  {'DRAW':>6}  {'AWAY':>6}  {'LIKELY':>6}  {'EXP':>4}  TOP 3 SCORES"
    )
    lines = [header, "-" * len(header)]

    for label, (_, row) in zip(labels, df.iterrows()):
        lines.append(
            f"{label:<{match_w}}  {row['lh']:>4.2f}  {row['la']:>4.2f}  "
            f"{_pct(row['sim_p_home']):>6}  {_pct(row['sim_p_draw']):>6}  "
            f"{_pct(row['sim_p_away']):>6}  {row['likely_score']:>6}  "
            f"{row['expected_score']:>4}  {_top_scores(row)}"
        )

    if any(flagged):
        lines.append("")
        lines.append(
            f"* implied goal rates fit the odds poorly "
            f"(residual_norm > {POOR_FIT_THRESHOLD}); treat with caution"
        )

    return "\n".join(lines)


def export_predictions(
    source: "str | Path | pd.DataFrame",
    output_path: "str | Path",
) -> pd.DataFrame:
    """Write a two-column predictions CSV from a simulated-outcomes frame.

    Applies select_headline_score to each game and writes a CSV with exactly two
    columns: ``match`` ("Home vs Away") and ``score`` (the chosen "h-a" string).

    Args:
        source: A summary DataFrame (as returned by simulate_games_from_odds) or
            a path to the simulated_outcomes CSV.
        output_path: Where to write the predictions CSV.

    Returns:
        The two-column predictions DataFrame (also written to output_path).
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    matches = [f"{home} vs {away}" for home, away in zip(df["home_team"], df["away_team"])]
    scores = [
        select_headline_score(
            row["sim_p_home"],
            row["sim_p_draw"],
            row["sim_p_away"],
            row["likely_score"],
            row["score_1"],
        )
        for _, row in df.iterrows()
    ]

    predictions = pd.DataFrame({"match": matches, "score": scores})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions -> %s", len(predictions), output_path)
    return predictions


def display_simulated_outcomes(
    source: "str | Path | pd.DataFrame", played_ids: "set | None" = None
) -> None:
    """Log the formatted simulated-outcomes table as a single message.

    Args:
        source: A summary DataFrame or a path to the simulated_outcomes CSV.
        played_ids: Optional set of game_ids to drop so only games still to be
            played are shown.
    """
    logger.info("Simulated outcomes:\n%s", format_simulated_outcomes(source, played_ids))


def _played_game_ids(results_path: "str | Path") -> set:
    """Game_ids of already-played games, read from the results CSV.

    Returns an empty set when the results file is absent, so the report falls
    back to showing every game.
    """
    path = Path(results_path)
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["game_id"])


def format_group_standings(source: "str | Path | pd.DataFrame") -> str:
    """Build an aligned, group-by-group table of the resolved standings.

    Args:
        source: A standings DataFrame (as written by the knockout resolver) or a
            path to the group_standings CSV. Expected columns: group, rank, team,
            played, won, drawn, lost, gf, ga, gd, points, status.

    Returns:
        A multi-line string with one block per group, teams ordered by rank, a
        dashed line after rank 2 marking the automatic-qualification cut, and
        each group's status (final once all its games are played, else
        projected) in the block header.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    blocks = []
    for letter in sorted(df["group"].unique()):
        group = df[df["group"] == letter].sort_values("rank")
        team_w = max(len("TEAM"), *(len(team) for team in group["team"]))
        status = group["status"].iloc[0]

        header = f"GROUP {letter}   ({status})"
        columns = (
            f"{'#':>2}  {'TEAM':<{team_w}}  {'P':>2} {'W':>2} {'D':>2} {'L':>2}  "
            f"{'GF':>3} {'GA':>3} {'GD':>3}  {'PTS':>3}"
        )
        lines = [header, columns]
        for _, row in group.iterrows():
            lines.append(
                f"{row['rank']:>2}  {row['team']:<{team_w}}  "
                f"{row['played']:>2} {row['won']:>2} {row['drawn']:>2} {row['lost']:>2}  "
                f"{row['gf']:>3} {row['ga']:>3} {row['gd']:>+3}  {row['points']:>3}"
            )
            if row["rank"] == 2:
                lines.append("   " + "-" * (len(columns) - 3))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _slot_label(team: str, status: str) -> str:
    """A bracket team name, marked with ' *' while its slot is still projected."""
    return team if status == "final" else f"{team} *"


def format_round_of_32(source: "str | Path | dict") -> str:
    """Build an aligned table of the resolved Round of 32.

    Args:
        source: A resolved-bracket dict (as written by the knockout resolver) or
            a path to the knockout_bracket_resolved JSON. Expected keys:
            round_of_32 (list of match dicts with home/away and *_status) and
            optionally qualifying_third_place_groups.

    Returns:
        A multi-line string, one row per match (`M73  Home  vs  Away`). Slots
        whose team is not yet locked are marked with a trailing '*', explained in
        a footnote; the qualifying third-place groups are shown in the header
        when present.
    """
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    matches = data["round_of_32"]

    homes = [_slot_label(m["home"], m["home_status"]) for m in matches]
    aways = [_slot_label(m["away"], m["away_status"]) for m in matches]
    home_w = max(len("HOME"), *(len(home) for home in homes))

    title = "ROUND OF 32"
    groups = data.get("qualifying_third_place_groups")
    if groups:
        title += f"   (best third-placed groups: {' '.join(groups)})"
    lines = [title, "-" * len(title)]

    for match, home, away in zip(matches, homes, aways):
        lines.append(f"  M{match['match']:<3} {home:>{home_w}}  vs  {away}")

    projected = any(m["home_status"] != "final" or m["away_status"] != "final" for m in matches)
    if projected:
        lines.append("")
        lines.append("* slot not yet locked (depends on games still to be played)")

    return "\n".join(lines)


def _bracket_children(structure: dict) -> dict[int, tuple[int, int]]:
    """Map each non-R32 match to its two feeder match numbers (home, away).

    Reads the predetermined connections (round_of_16 .. final), whose slots are
    "W##" tokens, into {match: (home_feeder, away_feeder)}.
    """
    children: dict[int, tuple[int, int]] = {}
    for rnd in ("round_of_16", "quarter_finals", "semi_finals", "final"):
        for match in structure["bracket"][rnd]:
            children[match["match"]] = (int(match["home"][1:]), int(match["away"][1:]))
    return children


class _Canvas:
    """A fixed-size grid of characters with simple text/glyph placement."""

    def __init__(self, height: int, width: int) -> None:
        self._grid = [[" "] * width for _ in range(height)]

    def text(self, row: int, col: int, value: str) -> None:
        for offset, char in enumerate(value):
            self._set(row, col + offset, char)

    def _set(self, row: int, col: int, char: str) -> None:
        if 0 <= row < len(self._grid) and 0 <= col < len(self._grid[0]):
            self._grid[row][col] = char

    def render(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self._grid).rstrip("\n")


def format_bracket(
    resolved_source: "str | Path | dict",
    structure_source: "str | Path | dict" = BRACKET_PATH,
) -> str:
    """Draw the knockout as a single-sided left-to-right bracket tree.

    All 32 Round-of-32 teams stack vertically on the left; each match's winner
    feeds rightward through R16, QF, SF, to the Final. Later rounds are not yet
    decided, so their slots show "W##" (winner of match ##). Round-of-32 teams
    whose slot is not yet locked are marked with a trailing "*".

    Args:
        resolved_source: A resolved-bracket dict (as written by the knockout
            resolver) or a path to the knockout_bracket_resolved JSON. Supplies
            the Round-of-32 team names and their final/projected statuses.
        structure_source: The reference knockout_bracket.json (dict or path) that
            carries the round-to-round connections. Defaults to the packaged file.

    Returns:
        A multi-line string holding the bracket, a column header per round, and a
        footnote when any slot is still projected.
    """
    resolved = (
        resolved_source
        if isinstance(resolved_source, dict)
        else json.loads(Path(resolved_source).read_text())
    )
    structure = (
        structure_source
        if isinstance(structure_source, dict)
        else json.loads(Path(structure_source).read_text())
    )

    children = _bracket_children(structure)
    final_match = structure["bracket"]["final"][0]["match"]

    teams = {}
    projected = False
    advanced = False
    # Leftmost winner column: show the team that advanced from a played result,
    # else the abstract "W##" token for a match still to be decided.
    r32_winner_label: dict[int, str] = {}
    for match in resolved["round_of_32"]:
        labels = []
        for side in ("home", "away"):
            locked = match[f"{side}_status"] == "final"
            projected = projected or not locked
            labels.append(match[side] if locked else f"{match[side]} *")
        teams[match["match"]] = tuple(labels)
        winner = match.get("winner")
        if winner:
            r32_winner_label[match["match"]] = winner
            advanced = True
        else:
            r32_winner_label[match["match"]] = f"W{match['match']}"

    # Column index per round, left (teams) to right (champion).
    col_of = {"r32": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}
    round_of = {}
    for match in range(73, 89):
        round_of[match] = "r32"
    for rnd, name in (("round_of_16", "r16"), ("quarter_finals", "qf"),
                      ("semi_finals", "sf"), ("final", "final")):
        for match in structure["bracket"][rnd]:
            round_of[match["match"]] = name

    team_w = max(len(label) for pair in teams.values() for label in pair)
    node_w = len(f"W{final_match}")
    # The R32-winner column can hold team names, so it is sized to its widest
    # label; later columns only ever hold "W##" nodes of width node_w.
    r32win_w = max(node_w, *(len(label) for label in r32_winner_label.values()))
    width_of_col = [team_w, r32win_w, node_w, node_w, node_w, node_w]
    pad = 4
    x_of_col = [0]
    for prev in range(5):
        x_of_col.append(x_of_col[-1] + width_of_col[prev] + pad)

    # Assign rows: leaves get evenly spaced rows; each match sits at the midpoint
    # of its two children.
    next_leaf_row = [0]
    node_row: dict[int, int] = {}
    leaf_rows: dict[int, tuple[int, int]] = {}

    def layout(match: int) -> int:
        if match in _R32_MATCHES:
            row_home = next_leaf_row[0]
            row_away = row_home + _LEAF_SPACING
            next_leaf_row[0] = row_away + _LEAF_SPACING
            leaf_rows[match] = (row_home, row_away)
            row = (row_home + row_away) // 2
        else:
            home, away = children[match]
            row = (layout(home) + layout(away)) // 2
        node_row[match] = row
        return row

    layout(final_match)

    height = next_leaf_row[0]
    canvas = _Canvas(height, x_of_col[5] + node_w)

    def connect(row_top: int, row_bottom: int, row_parent: int,
                child_right: int, parent_x: int) -> None:
        elbow = child_right + 2
        for row in (row_top, row_bottom):
            for col in range(child_right, elbow):
                canvas._set(row, col, "─")
        canvas._set(row_top, elbow, "┐")
        canvas._set(row_bottom, elbow, "┘")
        for row in range(row_top + 1, row_bottom):
            canvas._set(row, elbow, "│")
        canvas._set(row_parent, elbow, "├")
        for col in range(elbow + 1, parent_x):
            canvas._set(row_parent, col, "─")

    # Draw teams (right-aligned to the R32 column) and their match connectors.
    for match, (row_home, row_away) in leaf_rows.items():
        home, away = teams[match]
        canvas.text(row_home, team_w - len(home), home)
        canvas.text(row_away, team_w - len(away), away)
        connect(row_home, row_away, node_row[match], team_w, x_of_col[1])
        canvas.text(node_row[match], x_of_col[1], r32_winner_label[match])

    # Draw the winner nodes for R16, QF, SF, and the Final, with their connectors.
    for match, col_name in round_of.items():
        if col_name == "r32":
            continue
        col = col_of[col_name]
        home, away = children[match]
        child_right = x_of_col[col - 1] + width_of_col[col - 1]
        connect(node_row[home], node_row[away], node_row[match], child_right, x_of_col[col])
        canvas.text(node_row[match], x_of_col[col], f"W{match}")

    # Each column header names the round its teams are IN: the leaf teams played
    # the Round of 32, the winner beside them has reached the Round of 16, and so
    # on through to the final's winner (the champion).
    header_titles = [
        (max(0, team_w - len("R32")), "R32"),  # right-aligned over the team column
        (x_of_col[1], "R16"),
        (x_of_col[2], "QF"),
        (x_of_col[3], "SF"),
        (x_of_col[4], "FINAL"),
        (x_of_col[5], "CHAMPION"),
    ]
    header = ""
    for x, title in header_titles:
        header = header.ljust(x) + title

    lines = [header, canvas.render()]
    notes = []
    if advanced:
        notes.append("named team in the R32 column = advanced from a result")
    if projected or advanced:
        notes.append("W## = winner of match ## (still to be decided)")
    if projected:
        notes.append("* = slot not yet locked")
    if notes:
        lines.append("")
        lines.append("; ".join(notes))
    return "\n".join(lines)


def display_bracket(
    resolved_source: "str | Path | dict",
    structure_source: "str | Path | dict" = BRACKET_PATH,
) -> None:
    """Log the formatted bracket tree as a single message."""
    logger.info("Knockout bracket:\n%s", format_bracket(resolved_source, structure_source))


def display_group_standings(source: "str | Path | pd.DataFrame") -> None:
    """Log the formatted group-standings table as a single message."""
    logger.info("Group standings:\n%s", format_group_standings(source))


def display_round_of_32(source: "str | Path | dict") -> None:
    """Log the formatted Round of 32 table as a single message."""
    logger.info("Round of 32:\n%s", format_round_of_32(source))


def main() -> None:
    """Pretty-print a competition's processed outputs, or export predictions.

    Entry point for `python -m fifa_predictor.utils.display`. By default renders
    the still-to-be-played games from
    `data/processed/simulated_outcomes_<competition>.csv`, hiding games already
    in `data/raw/results_<competition>.csv` (pass `--all` to show every game).
    Flags switch the view: `--predict` writes
    `data/processed/predictions_<competition>.csv` (match, score); `--standings`
    renders `group_standings_<competition>.csv`; `--bracket` renders
    `knockout_bracket_resolved_<competition>.json`.
    """
    parser = argparse.ArgumentParser(
        description="Render simulated outcomes, standings, or the bracket; or export predictions."
    )
    parser.add_argument("competition", nargs="?", default="world_cup_2026")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Write the two-column predictions CSV instead of printing the table.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include games already played in the outcomes table (default hides them).",
    )
    parser.add_argument(
        "--standings",
        action="store_true",
        help="Render the resolved group_standings CSV instead of simulated outcomes.",
    )
    parser.add_argument(
        "--bracket",
        action="store_true",
        help="Render the resolved Round of 32 JSON instead of simulated outcomes.",
    )
    args = parser.parse_args()

    processed = "data/processed"
    if args.predict:
        export_predictions(
            f"{processed}/simulated_outcomes_{args.competition}.csv",
            f"{processed}/predictions_{args.competition}.csv",
        )
    elif args.standings:
        display_group_standings(f"{processed}/group_standings_{args.competition}.csv")
    elif args.bracket:
        display_bracket(f"{processed}/knockout_bracket_resolved_{args.competition}.json")
    else:
        played = set()
        if not args.all:
            played = _played_game_ids(f"data/raw/results_{args.competition}.csv")
        display_simulated_outcomes(
            f"{processed}/simulated_outcomes_{args.competition}.csv", played
        )


if __name__ == "__main__":
    main()
