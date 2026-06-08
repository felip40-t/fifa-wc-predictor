"""Tests for the Monte Carlo simulation module."""

import numpy as np
import pandas as pd
import pytest

from fifa_predictor.model import monte_carlo


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

    lh, la, residual_norm = monte_carlo._implied_rates_from_odds_row(
        row, rho=-0.13, max_goals=10
    )

    assert lh > 0
    assert la > 0
    assert residual_norm == pytest.approx(0.0, abs=0.05)


def test_simulate_game_outcomes_keys_and_probabilities_sum_to_one() -> None:
    """The summary dict should have the expected keys and outcome probs summing to ~1."""
    rng = np.random.default_rng(7)
    result = monte_carlo._simulate_game_outcomes(
        lh=1.4, la=1.1, rho=-0.13, max_goals=10, n_simulations=5_000, rng=rng
    )

    assert set(result) == {
        "sim_p_home",
        "sim_p_draw",
        "sim_p_away",
        "most_likely_scoreline",
        "most_likely_scoreline_freq",
    }
    assert result["sim_p_home"] + result["sim_p_draw"] + result["sim_p_away"] == pytest.approx(
        1.0, abs=1e-9
    )
    assert 0.0 < result["most_likely_scoreline_freq"] <= 1.0
    assert "-" in result["most_likely_scoreline"]


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
        n_simulations=2_000,
        seed=123,
    )

    expected_columns = [
        "game_id",
        "home_team",
        "away_team",
        "lh",
        "la",
        "residual_norm",
        "sim_p_home",
        "sim_p_draw",
        "sim_p_away",
        "most_likely_scoreline",
        "most_likely_scoreline_freq",
    ]
    assert list(result.columns) == expected_columns
    assert len(result) == 2
    assert output_csv.exists()

    prob_sums = result[["sim_p_home", "sim_p_draw", "sim_p_away"]].sum(axis=1)
    np.testing.assert_allclose(prob_sums.to_numpy(), 1.0, atol=1e-9)


def test_simulate_games_from_odds_is_reproducible_with_seed(tmp_path) -> None:
    """Two runs with the same seed should produce identical results."""
    odds_csv = tmp_path / "odds.csv"
    _two_row_odds_frame().to_csv(odds_csv, index=False)

    first = monte_carlo.simulate_games_from_odds(
        str(odds_csv), output_csv_path=str(tmp_path / "a.csv"), n_simulations=2_000, seed=99
    )
    second = monte_carlo.simulate_games_from_odds(
        str(odds_csv), output_csv_path=str(tmp_path / "b.csv"), n_simulations=2_000, seed=99
    )

    pd.testing.assert_frame_equal(first, second)
