"""Tests for the compare module (prediction vs actual result).

The comparison joins the predictions frame (predicted) to the results frame
(actual) by the "Home vs Away" match string, with no live data involved.
"""

import pandas as pd

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
