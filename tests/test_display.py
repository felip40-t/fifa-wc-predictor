"""Tests for the human-readable simulated-outcomes display helper."""

import pandas as pd

from fifa_predictor.model.knockout import load_bracket
from fifa_predictor.utils.display import (
    POOR_FIT_THRESHOLD,
    format_bracket,
    format_group_standings,
    format_round_of_32,
    format_simulated_outcomes,
)


def _resolved_from_structure(projected_first: bool = False) -> dict:
    """A resolved-bracket dict for all 16 R32 matches, teams named by match id."""
    rows = load_bracket()["bracket"]["round_of_32"]
    return {
        "round_of_32": [
            {
                "match": m["match"],
                "home": f"H{m['match']}",
                "away": f"A{m['match']}",
                "home_status": "projected" if (projected_first and i == 0) else "final",
                "away_status": "final",
            }
            for i, m in enumerate(rows)
        ]
    }


def _frame() -> pd.DataFrame:
    """A one-row frame mirroring the simulated_outcomes CSV schema."""
    return pd.DataFrame(
        [
            {
                "game_id": "abc123hashvalue",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "lh": 1.9016,
                "la": 0.5520,
                "residual_norm": 0.018,
                "sim_p_home": 0.6876,
                "sim_p_draw": 0.226,
                "sim_p_away": 0.0864,
                "likely_score": "2-0",
                "expected_score": "2-1",
                "score_1": "2-0",
                "score_1_freq": 0.1577,
                "score_2": "1-0",
                "score_2_freq": 0.1203,
                "score_3": "3-0",
                "score_3_freq": 0.0981,
            }
        ]
    )


def test_format_includes_teams_and_rounded_values() -> None:
    out = format_simulated_outcomes(_frame())
    assert "Mexico vs South Africa" in out
    assert "68.8%" in out  # sim_p_home as a percentage
    assert "1.90" in out  # lh rounded to two places


def test_format_shows_point_estimate_columns() -> None:
    out = format_simulated_outcomes(_frame())
    assert "LIKELY" in out  # result-consistent mode header
    assert "EXP" in out  # expected (rounded goal-rate) score header
    assert "2-1" in out  # expected_score value, distinct from any top scoreline


def test_format_shows_all_three_scorelines_with_frequencies() -> None:
    out = format_simulated_outcomes(_frame())
    assert "2-0 15.8%" in out
    assert "1-0 12.0%" in out
    assert "3-0 9.8%" in out


def test_format_skips_blank_scoreline_ranks() -> None:
    df = _frame()
    df.loc[0, "score_3"] = ""
    df.loc[0, "score_3_freq"] = 0.0
    out = format_simulated_outcomes(df)
    assert "2-0 15.8%" in out
    assert "3-0" not in out


def test_format_hides_the_game_id_hash() -> None:
    out = format_simulated_outcomes(_frame())
    assert "abc123hashvalue" not in out


def _two_game_frame() -> pd.DataFrame:
    """A two-row frame for testing the played-game filter (distinct game_ids)."""
    frame = pd.concat([_frame(), _frame()], ignore_index=True)
    frame.loc[1, "game_id"] = "def456"
    frame.loc[1, "home_team"] = "Brazil"
    frame.loc[1, "away_team"] = "Croatia"
    return frame


def test_format_hides_played_games() -> None:
    out = format_simulated_outcomes(_two_game_frame(), played_ids={"abc123hashvalue"})
    assert "Mexico vs South Africa" not in out
    assert "Brazil vs Croatia" in out


def test_format_without_played_ids_shows_all_games() -> None:
    out = format_simulated_outcomes(_two_game_frame())
    assert "Mexico vs South Africa" in out
    assert "Brazil vs Croatia" in out


def test_poor_fit_rows_are_flagged() -> None:
    df = _frame()
    df.loc[0, "residual_norm"] = POOR_FIT_THRESHOLD + 1
    out = format_simulated_outcomes(df)
    assert "Mexico vs South Africa *" in out
    assert "caution" in out


def test_well_fit_rows_have_no_flag_footnote() -> None:
    out = format_simulated_outcomes(_frame())
    assert "caution" not in out


def _standings_frame() -> pd.DataFrame:
    """A two-team group mirroring the group_standings CSV schema."""
    return pd.DataFrame(
        [
            {"group": "A", "rank": 1, "team": "Mexico", "played": 3, "won": 3,
             "drawn": 0, "lost": 0, "gf": 4, "ga": 0, "gd": 4, "points": 9,
             "status": "projected"},
            {"group": "A", "rank": 2, "team": "South Korea", "played": 3, "won": 2,
             "drawn": 0, "lost": 1, "gf": 3, "ga": 2, "gd": 1, "points": 6,
             "status": "projected"},
        ]
    )


def test_format_group_standings_shows_group_teams_and_signed_gd() -> None:
    out = format_group_standings(_standings_frame())
    assert "GROUP A   (projected)" in out
    assert "Mexico" in out and "South Korea" in out
    assert "+4" in out  # goal difference is signed
    assert "----" in out  # qualification cut line drawn after rank 2


def test_format_round_of_32_marks_projected_slots() -> None:
    data = {
        "qualifying_third_place_groups": ["B", "C", "D", "F", "I", "J", "K", "L"],
        "round_of_32": [
            {"match": 73, "home": "South Korea", "away": "Switzerland",
             "home_status": "final", "away_status": "projected"},
        ],
    }
    out = format_round_of_32(data)
    assert "M73" in out
    assert "South Korea" in out  # final slot: no marker
    assert "Switzerland *" in out  # projected slot: trailing marker
    assert "best third-placed groups: B C D F I J K L" in out
    assert "not yet locked" in out


def test_format_bracket_renders_all_rounds_as_a_tree() -> None:
    out = format_bracket(_resolved_from_structure(), load_bracket())
    assert "H74" in out and "A74" in out  # an R32 team pair (leaves)
    assert "W74" in out and "W89" in out  # R32 and R16 winner nodes
    assert "W101" in out and "W104" in out  # SF and final nodes
    assert "├" in out and "┐" in out  # bracket connector glyphs
    assert "not yet locked" not in out  # every slot final -> no footnote


def test_format_bracket_marks_projected_slots() -> None:
    out = format_bracket(_resolved_from_structure(projected_first=True), load_bracket())
    assert "*" in out
    assert "not yet locked" in out


def test_format_bracket_round_labels_sit_over_the_round_they_name() -> None:
    """A team's column header is the round it is IN, not the round it won.

    The leaf teams played the Round of 32, so 'R32' labels the team column; the
    winner that emerged has reached the Round of 16, so 'R16' sits over the
    winner column.
    """
    out = format_bracket(_resolved_from_structure(), load_bracket())
    lines = out.splitlines()
    header = lines[0]
    winner_line = next(line for line in lines if "W73" in line)

    assert header.index("R16") == winner_line.index("W73")  # R16 over R32 winners
    assert header.index("R32") < header.index("R16")  # R32 labels the team column
    assert "CHAMPION" in header  # the final's winner column is labelled


def test_format_bracket_shows_advanced_team_for_decided_r32_match() -> None:
    resolved = _resolved_from_structure()
    resolved["round_of_32"][0]["winner"] = "Canada"  # match 73 decided
    resolved["round_of_32"][0]["result_status"] = "played"

    out = format_bracket(resolved, load_bracket())

    assert "Canada" in out  # advancing team shown in the winner column
    assert "W73" not in out  # its abstract winner token is replaced
    assert "W74" in out  # an undecided R32 winner stays abstract


def test_reads_from_a_csv_path(tmp_path) -> None:
    path = tmp_path / "sim.csv"
    _frame().to_csv(path, index=False)
    out = format_simulated_outcomes(str(path))
    assert "Mexico vs South Africa" in out


def test_export_predictions_writes_two_column_csv(tmp_path) -> None:
    from fifa_predictor.utils.display import export_predictions

    df = pd.DataFrame(
        {
            "home_team": ["Mexico", "Spain"],
            "away_team": ["South Africa", "Brazil"],
            "sim_p_home": [0.69, 0.36],
            "sim_p_draw": [0.20, 0.32],
            "sim_p_away": [0.11, 0.32],
            "likely_score": ["2-0", "1-0"],
            "score_1": ["1-0", "1-1"],
        }
    )
    out = tmp_path / "predictions.csv"
    export_predictions(df, out)

    result = pd.read_csv(out)
    assert list(result.columns) == ["match", "score"]
    assert len(result) == 2
    # Clear favorite -> result-consistent score; flat race -> most likely exact.
    assert result.iloc[0]["match"] == "Mexico vs South Africa"
    assert result.iloc[0]["score"] == "2-0"
    assert result.iloc[1]["score"] == "1-1"
