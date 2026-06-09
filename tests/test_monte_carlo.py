"""Tests for the Monte Carlo simulation module."""

import math

import numpy as np
import pandas as pd
import pytest

from fifa_predictor.model import monte_carlo
from fifa_predictor.model import poisson_inversion
from fifa_predictor.model.monte_carlo import _implied_rates_from_odds_row, _row_value
from fifa_predictor.model.poisson_inversion import (
    poisson_probs_dc,
    scoreline_probabilities_dc,
)


def test_simulate_match_degenerate_matrix_returns_that_scoreline() -> None:
    """A matrix with a single nonzero cell always returns that scoreline."""
    matrix = np.zeros((4, 4))
    matrix[2, 1] = 1.0
    rng = np.random.default_rng(0)

    for _ in range(20):
        assert monte_carlo.simulate_match(matrix, rng) == (2, 1)


def test_simulate_match_empirical_distribution_matches_matrix() -> None:
    """Sampling many matches should reproduce the input probabilities."""
    matrix = np.array(
        [
            [0.30, 0.10, 0.05],
            [0.15, 0.20, 0.05],
            [0.05, 0.03, 0.07],
        ]
    )
    rng = np.random.default_rng(42)

    counts = np.zeros_like(matrix)
    n = 50_000
    for _ in range(n):
        h, a = monte_carlo.simulate_match(matrix, rng)
        counts[h, a] += 1

    empirical = counts / n
    np.testing.assert_allclose(empirical, matrix, atol=0.01)


def test_simulate_match_returns_python_ints() -> None:
    """Returned goals should be plain Python ints, not numpy scalars."""
    matrix = np.zeros((3, 3))
    matrix[1, 2] = 1.0
    h, a = monte_carlo.simulate_match(matrix, np.random.default_rng(1))

    assert type(h) is int and type(a) is int


def test_goals_round_rounds_up_only_at_seven_tenths() -> None:
    """goals_round credits an extra goal only when the fraction reaches 0.7."""
    assert monte_carlo.goals_round(0.5) == 0
    assert monte_carlo.goals_round(0.69) == 0
    assert monte_carlo.goals_round(0.7) == 1
    assert monte_carlo.goals_round(0.71) == 1
    assert monte_carlo.goals_round(0.45) == 0
    assert monte_carlo.goals_round(2.7) == 3
    assert monte_carlo.goals_round(3.12) == 3


def test_result_consistent_mode_favourite_never_returns_a_draw() -> None:
    """For a clear home favourite the headline score is a home win, not 1-1."""
    # Belgium-like rates: 1-1 is the joint mode, but home win is the result.
    matrix = scoreline_probabilities_dc(1.76, 0.83, rho=-0.13, max_goals=10)
    score = monte_carlo.result_consistent_mode(matrix)
    home, away = (int(x) for x in score.split("-"))
    assert home > away


def test_result_consistent_mode_away_favourite_returns_away_win() -> None:
    """For a clear away favourite the headline score is an away win."""
    matrix = scoreline_probabilities_dc(0.7, 2.9, rho=-0.13, max_goals=10)
    score = monte_carlo.result_consistent_mode(matrix)
    home, away = (int(x) for x in score.split("-"))
    assert away > home


def test_result_consistent_mode_low_scoring_even_game_returns_a_draw() -> None:
    """When the draw is the most likely result the headline score is a draw.

    Equal but low goal rates make the draw the plurality result (~41% vs ~29%
    each side), so the result-consistent score is a draw.
    """
    matrix = scoreline_probabilities_dc(0.7, 0.7, rho=-0.13, max_goals=10)
    score = monte_carlo.result_consistent_mode(matrix)
    home, away = (int(x) for x in score.split("-"))
    assert home == away


def test_game_outcomes_keys_and_analytical_probabilities() -> None:
    """Outcomes are read exactly off the DC matrix, with the new point estimates."""
    lh, la, rho = 1.76, 0.83, -0.13
    result = monte_carlo._game_outcomes(lh=lh, la=la, rho=rho, max_goals=10)

    assert set(result) == {
        "sim_p_home",
        "sim_p_draw",
        "sim_p_away",
        "likely_score",
        "expected_score",
        "score_1",
        "score_1_freq",
        "score_2",
        "score_2_freq",
        "score_3",
        "score_3_freq",
    }
    assert result["sim_p_home"] + result["sim_p_draw"] + result["sim_p_away"] == pytest.approx(
        1.0, abs=1e-9
    )
    # Exact region sums match poisson_probs_dc (normalised to sum to one).
    raw_home, raw_draw, raw_away = poisson_probs_dc(lh, la, rho, max_goals=10)
    total = raw_home + raw_draw + raw_away
    assert result["sim_p_home"] == pytest.approx(raw_home / total, abs=1e-9)
    # Top scorelines ranked most to least likely.
    assert result["score_1_freq"] >= result["score_2_freq"] >= result["score_3_freq"]
    # This favourite's expected score is 2-1 (1.76 -> 2, 0.83 -> 1).
    assert result["expected_score"] == "2-1"
    # The result-consistent headline is a home win, not the 1-1 joint mode.
    assert result["likely_score"] == "1-0"


def _synthetic_odds_row() -> dict:
    """A single odds row with known, well-behaved DC-consistent odds."""
    return {
        "game_id": "g1",
        "home_team": "Alpha",
        "away_team": "Beta",
        "pinnacle_h2h_home": 1.8,
        "pinnacle_h2h_draw": 3.6,
        "pinnacle_h2h_away": 4.5,
        "pinnacle_ou_line": 2.5,
        "pinnacle_ou_over": 1.95,
        "pinnacle_ou_under": 1.95,
        "odds_source": "pinnacle",
    }


def test_implied_rates_from_odds_row_returns_positive_rates() -> None:
    """A well-formed odds row should invert to positive goal rates."""
    row = _synthetic_odds_row()

    lh, la, rho, residual_norm = monte_carlo._implied_rates_from_odds_row(
        row, rho_init=-0.13, max_goals=10
    )

    assert lh > 0
    assert la > 0
    low, high = poisson_inversion.RHO_BOUNDS
    assert low <= rho <= high
    assert residual_norm == pytest.approx(0.0, abs=0.05)


def _two_row_odds_frame() -> pd.DataFrame:
    """A small synthetic 2-game odds frame, decoupled from the live CSV."""
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "commence_time": "2026-06-11 19:00:00+00:00",
                "pinnacle_h2h_home": 1.8,
                "pinnacle_h2h_draw": 3.6,
                "pinnacle_h2h_away": 4.5,
                "pinnacle_ou_line": 2.5,
                "pinnacle_ou_over": 1.95,
                "pinnacle_ou_under": 1.95,
                "odds_source": "pinnacle",
            },
            {
                "game_id": "g2",
                "home_team": "Gamma",
                "away_team": "Delta",
                "commence_time": "2026-06-12 02:00:00+00:00",
                "pinnacle_h2h_home": 2.6,
                "pinnacle_h2h_draw": 3.3,
                "pinnacle_h2h_away": 2.7,
                "pinnacle_ou_line": 2.5,
                "pinnacle_ou_over": 2.05,
                "pinnacle_ou_under": 1.85,
                "odds_source": "pinnacle",
            },
        ]
    )


def test_simulate_games_from_odds_shape_columns_and_probabilities(tmp_path) -> None:
    """Output frame has one row per game, expected columns, and valid probabilities."""
    odds_csv = tmp_path / "odds.csv"
    _two_row_odds_frame().to_csv(odds_csv, index=False)
    output_csv = tmp_path / "out.csv"

    result = monte_carlo.simulate_games_from_odds(
        str(odds_csv),
        output_csv_path=str(output_csv),
    )

    expected_columns = [
        "game_id",
        "home_team",
        "away_team",
        "lh",
        "la",
        "rho",
        "residual_norm",
        "sim_p_home",
        "sim_p_draw",
        "sim_p_away",
        "likely_score",
        "expected_score",
        "score_1",
        "score_1_freq",
        "score_2",
        "score_2_freq",
        "score_3",
        "score_3_freq",
    ]
    assert list(result.columns) == expected_columns
    assert len(result) == 2
    assert output_csv.exists()

    prob_sums = result[["sim_p_home", "sim_p_draw", "sim_p_away"]].sum(axis=1)
    np.testing.assert_allclose(prob_sums.to_numpy(), 1.0, atol=1e-9)


def test_simulate_games_from_odds_is_deterministic(tmp_path) -> None:
    """The analytic summary is deterministic: two runs produce identical results."""
    odds_csv = tmp_path / "odds.csv"
    _two_row_odds_frame().to_csv(odds_csv, index=False)

    first = monte_carlo.simulate_games_from_odds(
        str(odds_csv), output_csv_path=str(tmp_path / "a.csv")
    )
    second = monte_carlo.simulate_games_from_odds(
        str(odds_csv), output_csv_path=str(tmp_path / "b.csv")
    )

    pd.testing.assert_frame_equal(first, second)


def test_simulate_games_from_odds_progress_flag_runs(tmp_path) -> None:
    """The progress=True path should run and return the same shape as without it."""
    odds_csv = tmp_path / "odds.csv"
    _two_row_odds_frame().to_csv(odds_csv, index=False)

    result = monte_carlo.simulate_games_from_odds(
        str(odds_csv),
        output_csv_path=str(tmp_path / "out.csv"),
        progress=True,
    )

    assert len(result) == 2


def _base_row():
    return {
        "pinnacle_h2h_home": 1.8,
        "pinnacle_h2h_draw": 3.5,
        "pinnacle_h2h_away": 4.5,
        "pinnacle_ou_line": 2.5,
        "pinnacle_ou_over": 1.95,
        "pinnacle_ou_under": 1.90,
    }


def test_row_value_missing_key_returns_nan():
    assert math.isnan(_row_value({}, "pinnacle_ou2_line"))


def test_implied_rates_without_secondary_columns_still_works():
    lh, la, rho, residual = _implied_rates_from_odds_row(_base_row(), -0.13, 10)
    assert lh > 0 and la > 0
    assert -0.2 <= rho <= 0.2


def test_implied_rates_uses_secondary_line_when_present():
    row = _base_row()
    row.update({
        "pinnacle_ou2_line": 2.25,
        "pinnacle_ou2_over": 1.98,
        "pinnacle_ou2_under": 1.86,
    })
    lh, la, rho, residual = _implied_rates_from_odds_row(row, -0.13, 10)
    assert lh > 0 and la > 0
    assert -0.2 <= rho <= 0.2
    assert math.isfinite(residual)


def test_implied_rates_nan_secondary_line_is_ignored():
    row = _base_row()
    row.update({
        "pinnacle_ou2_line": float("nan"),
        "pinnacle_ou2_over": float("nan"),
        "pinnacle_ou2_under": float("nan"),
    })
    with_nan = _implied_rates_from_odds_row(row, -0.13, 10)
    without = _implied_rates_from_odds_row(_base_row(), -0.13, 10)
    assert with_nan == pytest.approx(without)


def test_headline_clear_favorite_uses_result_consistent() -> None:
    # Home leads draw by 45pp >> 8pp: commit to the result-consistent score.
    out = monte_carlo.select_headline_score(0.65, 0.20, 0.15, "2-0", "1-0")
    assert out == "2-0"


def test_headline_flat_race_uses_most_likely_exact() -> None:
    # 36 / 32 / 32: top-minus-runner-up = 4pp < 8pp -> fall back to exact score.
    out = monte_carlo.select_headline_score(0.36, 0.32, 0.32, "1-0", "1-1")
    assert out == "1-1"


def test_headline_boundary_at_margin_commits_to_result() -> None:
    # Gap exactly 8pp is inclusive (>=), so commit to the result-consistent score.
    out = monte_carlo.select_headline_score(0.40, 0.32, 0.28, "2-1", "1-1")
    assert out == "2-1"


def test_headline_away_favorite_symmetric() -> None:
    # Away clearly favored: same behavior as a home favorite.
    out = monte_carlo.select_headline_score(0.15, 0.20, 0.65, "0-2", "0-1")
    assert out == "0-2"
