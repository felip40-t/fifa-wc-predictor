"""Tests for the human-readable simulated-outcomes display helper."""

import pandas as pd

from fifa_predictor.utils.display import (
    POOR_FIT_THRESHOLD,
    format_simulated_outcomes,
)


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


def test_poor_fit_rows_are_flagged() -> None:
    df = _frame()
    df.loc[0, "residual_norm"] = POOR_FIT_THRESHOLD + 1
    out = format_simulated_outcomes(df)
    assert "Mexico vs South Africa *" in out
    assert "caution" in out


def test_well_fit_rows_have_no_flag_footnote() -> None:
    out = format_simulated_outcomes(_frame())
    assert "caution" not in out


def test_reads_from_a_csv_path(tmp_path) -> None:
    path = tmp_path / "sim.csv"
    _frame().to_csv(path, index=False)
    out = format_simulated_outcomes(str(path))
    assert "Mexico vs South Africa" in out
