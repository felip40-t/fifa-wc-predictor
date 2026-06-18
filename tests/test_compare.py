"""Tests for the compare module (prediction vs actual result).

The comparison joins the predictions frame (predicted) to the results frame
(actual) by the "Home vs Away" match string, with no live data involved.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from fifa_predictor.utils import compare
from fifa_predictor.utils.compare import (
    _COMPARISON_COLUMNS,
    _score_outcome,
    build_comparison,
)


def _predictions_row(match: str, score: str) -> dict:
    """A predictions-CSV row: the published two columns build_comparison reads."""
    return {"match": match, "score": score}


def _results_row(game_id: str, home: str, away: str, home_score: int, away_score: int) -> dict:
    return {
        "game_id": game_id,
        "commence_time": "2026-06-11T19:00:00Z",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    }


def _outcomes_row(home: str, away: str, score_1: str, score_2: str, score_3: str) -> dict:
    """A simulated-outcomes-CSV row, trimmed to the columns compare reads."""
    return {
        "home_team": home,
        "away_team": away,
        "score_1": score_1,
        "score_2": score_2,
        "score_3": score_3,
    }


def _outcomes_row_full(
    home: str,
    away: str,
    *,
    sim_p_home: float,
    sim_p_draw: float,
    sim_p_away: float,
    lh: float,
    la: float,
    rho: float,
    score_1: str = "1-0",
    score_2: str = "1-1",
    score_3: str = "2-1",
) -> dict:
    """A simulated-outcomes row carrying the marginal probabilities and DC rates."""
    return {
        "home_team": home,
        "away_team": away,
        "sim_p_home": sim_p_home,
        "sim_p_draw": sim_p_draw,
        "sim_p_away": sim_p_away,
        "lh": lh,
        "la": la,
        "rho": rho,
        "score_1": score_1,
        "score_2": score_2,
        "score_3": score_3,
    }


def test_score_outcome_classifies_home_draw_away():
    assert _score_outcome(2, 1) == "H"
    assert _score_outcome(1, 1) == "D"
    assert _score_outcome(0, 2) == "A"


def test_build_comparison_marks_exact_and_result_hits():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 2, 1)])

    df = build_comparison(predictions, results)

    assert list(df.columns) == _COMPARISON_COLUMNS
    row = df.iloc[0]
    assert row["match"] == "Mexico vs Poland"
    assert row["predicted"] == "2-1"
    assert row["actual"] == "2-1"
    assert bool(row["exact_hit"]) is True
    assert bool(row["result_hit"]) is True


def test_build_comparison_result_hit_without_exact_hit():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 3, 0)])  # home win, wrong score

    row = build_comparison(predictions, results).iloc[0]

    assert row["predicted"] == "2-1"
    assert row["actual"] == "3-0"
    assert bool(row["exact_hit"]) is False
    assert bool(row["result_hit"]) is True


def test_build_comparison_total_miss():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 0, 2)])  # away win

    row = build_comparison(predictions, results).iloc[0]

    assert bool(row["exact_hit"]) is False
    assert bool(row["result_hit"]) is False


def test_build_comparison_only_includes_played_games():
    predictions = pd.DataFrame([
        _predictions_row("A vs B", "1-0"),
        _predictions_row("C vs D", "1-1"),
    ])
    results = pd.DataFrame([_results_row("played", "A", "B", 1, 0)])

    df = build_comparison(predictions, results)

    assert list(df["match"]) == ["A vs B"]


def test_build_comparison_empty_results_returns_typed_empty():
    predictions = pd.DataFrame([_predictions_row("A vs B", "1-0")])
    results = pd.DataFrame(columns=["game_id", "home_team", "away_team", "home_score", "away_score"])

    df = build_comparison(predictions, results)

    assert df.empty
    assert list(df.columns) == _COMPARISON_COLUMNS


def test_build_comparison_flags_top3_hit_when_actual_is_a_runner_up():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "1-0")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 1, 1)])  # the score_2 case
    outcomes = pd.DataFrame([_outcomes_row("Mexico", "Poland", "1-0", "1-1", "2-1")])

    row = build_comparison(predictions, results, outcomes).iloc[0]

    assert row["top3"] == "1-0 / 1-1 / 2-1"
    assert bool(row["top3_hit"]) is True
    # The headline still missed the exact score; top3 catches the runner-up.
    assert bool(row["exact_hit"]) is False


def test_build_comparison_top3_miss_when_actual_outside_top_three():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "1-0")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 3, 2)])
    outcomes = pd.DataFrame([_outcomes_row("Mexico", "Poland", "1-0", "1-1", "2-1")])

    row = build_comparison(predictions, results, outcomes).iloc[0]

    assert bool(row["top3_hit"]) is False


def test_build_comparison_without_outcomes_leaves_top3_empty():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 2, 1)])

    df = build_comparison(predictions, results)

    assert list(df.columns) == _COMPARISON_COLUMNS
    row = df.iloc[0]
    assert row["top3"] == ""
    assert bool(row["top3_hit"]) is False


def test_build_comparison_result_prob_uses_actual_outcome_marginal():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 0, 2)])  # away win
    outcomes = pd.DataFrame([
        _outcomes_row_full(
            "Mexico", "Poland",
            sim_p_home=0.5, sim_p_draw=0.3, sim_p_away=0.2,
            lh=1.5, la=1.2, rho=0.0,
        )
    ])

    row = build_comparison(predictions, results, outcomes).iloc[0]

    # Away team won, so the result probability is the away marginal.
    assert row["result_prob"] == pytest.approx(0.2)


def test_build_comparison_score_prob_reads_exact_matrix_cell():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 2, 1)])
    outcomes = pd.DataFrame([
        _outcomes_row_full(
            "Mexico", "Poland",
            sim_p_home=0.5, sim_p_draw=0.3, sim_p_away=0.2,
            lh=1.5, la=1.2, rho=0.0,
        )
    ])

    # With rho=0 the DC correction is identity, so the matrix is the plain
    # independent-Poisson outer product; the 2-1 cell is poisson(2;1.5)*poisson(1;1.2).
    goals = np.arange(11)
    matrix = np.outer(poisson.pmf(goals, 1.5), poisson.pmf(goals, 1.2))
    expected = matrix[2, 1] / matrix.sum()

    row = build_comparison(predictions, results, outcomes).iloc[0]

    assert row["score_prob"] == pytest.approx(expected)


def test_build_comparison_without_outcomes_leaves_probs_blank():
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 2, 1)])

    row = build_comparison(predictions, results).iloc[0]

    assert pd.isna(row["result_prob"])
    assert pd.isna(row["score_prob"])


def test_format_comparison_renders_probabilities_as_percentages():
    df = pd.DataFrame(
        {
            "match": ["A vs B"],
            "predicted": ["2-1"],
            "actual": ["0-2"],
            "result_hit": [False],
            "exact_hit": [False],
            "top3": ["1-0 / 1-1 / 2-1"],
            "top3_hit": [False],
            "result_prob": [0.2],
            "score_prob": [0.123],
        }
    )

    text = compare.format_comparison(df)

    assert "20.0%" in text
    assert "12.3%" in text


def test_format_comparison_top3_column_shows_matching_rank():
    # actual 8-8 is the 2nd of the three most likely scores; the digit 2 appears
    # nowhere else in the row, so finding it proves the rank (not "OK") is shown.
    df = pd.DataFrame(
        {
            "match": ["A vs B"],
            "predicted": ["0-0"],
            "actual": ["8-8"],
            "result_hit": [False],
            "exact_hit": [False],
            "top3": ["9-9 / 8-8 / 7-7"],
            "top3_hit": [True],
            "result_prob": [float("nan")],
            "score_prob": [float("nan")],
        }
    )

    text = compare.format_comparison(df)

    assert "OK" not in text
    assert "2" in text


def test_format_comparison_top3_column_blank_on_miss():
    df = pd.DataFrame(
        {
            "match": ["A vs B"],
            "predicted": ["0-0"],
            "actual": ["8-8"],
            "result_hit": [False],
            "exact_hit": [False],
            "top3": ["9-9 / 7-7 / 6-6"],
            "top3_hit": [False],
            "result_prob": [float("nan")],
            "score_prob": [float("nan")],
        }
    )

    data_line = compare.format_comparison(df).splitlines()[2]

    assert "1" not in data_line
    assert "2" not in data_line
    assert "3" not in data_line


def test_format_comparison_reports_top3_count():
    df = pd.DataFrame(
        {
            "match": ["A vs B", "C vs D", "E vs F"],
            "predicted": ["2-1", "1-0", "0-0"],
            "actual": ["2-1", "0-2", "1-1"],
            "result_hit": [True, False, True],
            "exact_hit": [True, False, False],
            "top3": ["2-1 / 1-1 / 2-0", "1-0 / 0-1 / 1-1", "0-0 / 1-1 / 1-0"],
            "top3_hit": [True, False, True],
        }
    )

    text = compare.format_comparison(df)

    assert "top3 2/3" in text


def test_format_comparison_reports_summary_counts():
    df = pd.DataFrame(
        {
            "match": ["A vs B", "C vs D", "E vs F"],
            "predicted": ["2-1", "1-0", "0-0"],
            "actual": ["2-1", "0-2", "1-1"],
            "result_hit": [True, False, True],
            "exact_hit": [True, False, False],
            "top3": ["2-1 / 1-1 / 2-0", "1-0 / 0-1 / 1-1", "0-0 / 1-1 / 1-0"],
            "top3_hit": [True, False, True],
        }
    )

    text = compare.format_comparison(df)

    assert "exact 1/3" in text
    assert "result 2/3" in text


def test_export_comparison_writes_csv(tmp_path):
    predictions = pd.DataFrame([_predictions_row("Mexico vs Poland", "2-1")])
    results = pd.DataFrame([_results_row("g1", "Mexico", "Poland", 2, 1)])
    out = tmp_path / "comparison.csv"

    compare.export_comparison(predictions, results, out)

    written = pd.read_csv(out)
    assert list(written.columns) == _COMPARISON_COLUMNS
    assert written.iloc[0]["predicted"] == "2-1"
