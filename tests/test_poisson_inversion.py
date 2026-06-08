"""Tests for the Poisson inversion model module."""

import numpy as np
import pytest

from fifa_predictor.model import poisson_inversion


def test_poisson_probabilities_sums_to_one() -> None:
    """Home/draw/away probabilities should be non-negative and sum to ~1."""
    home_p, draw_p, away_p = poisson_inversion.poisson_probabilities(1.5, 1.2, max_goals=10)

    assert home_p >= 0
    assert draw_p >= 0
    assert away_p >= 0
    assert home_p + draw_p + away_p == pytest.approx(1.0, abs=1e-6)


def test_poisson_probabilities_symmetry() -> None:
    """Swapping home and away rates should swap home and away win probabilities."""
    home_p, draw_p, away_p = poisson_inversion.poisson_probabilities(1.8, 1.1, max_goals=10)
    away_p2, draw_p2, home_p2 = poisson_inversion.poisson_probabilities(1.1, 1.8, max_goals=10)

    assert home_p == pytest.approx(home_p2)
    assert draw_p == pytest.approx(draw_p2)
    assert away_p == pytest.approx(away_p2)


def test_poisson_probabilities_equal_rates_favor_draw_symmetrically() -> None:
    """Equal goal rates should give equal home/away win probabilities."""
    home_p, draw_p, away_p = poisson_inversion.poisson_probabilities(1.4, 1.4, max_goals=10)

    assert home_p == pytest.approx(away_p)
    assert draw_p > 0


def test_equations_returns_inf_for_non_positive_rates() -> None:
    """The solver objective should reject non-positive goal rates."""
    assert poisson_inversion.equations((0.0, 1.0), 0.45, 0.27) == (np.inf, np.inf)
    assert poisson_inversion.equations((1.0, -0.5), 0.45, 0.27) == (np.inf, np.inf)


def test_equations_zero_at_true_rates() -> None:
    """The objective should be ~zero when evaluated at the rates that produced the targets."""
    home_rate, away_rate = 1.5, 1.1
    target_home_p, target_draw_p, _ = poisson_inversion.poisson_probabilities(
        home_rate, away_rate, max_goals=10
    )

    home_residual, draw_residual = poisson_inversion.equations(
        (home_rate, away_rate), target_home_p, target_draw_p
    )

    assert home_residual == pytest.approx(0.0, abs=1e-9)
    assert draw_residual == pytest.approx(0.0, abs=1e-9)


def test_implied_goal_rates_round_trips_through_poisson_probabilities() -> None:
    """Rates implied from outcome probabilities should reproduce those probabilities."""
    home_rate, away_rate = 1.5, 1.1
    target_home_p, target_draw_p, target_away_p = poisson_inversion.poisson_probabilities(
        home_rate, away_rate, max_goals=10
    )

    implied_home_rate, implied_away_rate = poisson_inversion.implied_goal_rates(
        target_home_p, target_draw_p, target_away_p
    )
    home_p, draw_p, away_p = poisson_inversion.poisson_probabilities(
        implied_home_rate, implied_away_rate, max_goals=10
    )

    assert home_p == pytest.approx(target_home_p, abs=1e-6)
    assert draw_p == pytest.approx(target_draw_p, abs=1e-6)
    assert away_p == pytest.approx(target_away_p, abs=1e-6)


def test_implied_goal_rates_returns_positive_rates() -> None:
    """Implied goal rates should be positive Poisson parameters."""
    home_rate, away_rate = poisson_inversion.implied_goal_rates(0.45, 0.27, 0.28)

    assert home_rate > 0
    assert away_rate > 0


def test_scoreline_probabilities_shape_and_sum() -> None:
    """The scoreline matrix should have the expected shape and sum close to 1."""
    matrix = poisson_inversion.scoreline_probabilities(1.5, 1.2, max_goals=10)

    assert matrix.shape == (11, 11)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(matrix >= 0)


def test_scoreline_probabilities_matches_outer_product_of_marginals() -> None:
    """Each entry should equal the product of the independent home/away goal probabilities."""
    home_rate, away_rate, max_goals = 1.5, 1.2, 6
    matrix = poisson_inversion.scoreline_probabilities(home_rate, away_rate, max_goals)

    goals = np.arange(max_goals + 1)
    from scipy.stats import poisson

    expected = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))

    np.testing.assert_allclose(matrix, expected)


def test_implied_goal_rates_dc_round_trips_through_dc_probabilities() -> None:
    """Rates implied from DC 1X2 + over/under probabilities should recover the originals."""
    home_rate, away_rate, rho = 1.4, 1.1, -0.13
    ou_line = 2.5
    target_home_p, target_draw_p, target_away_p = poisson_inversion.poisson_probs_dc(
        home_rate, away_rate, rho, max_goals=10
    )
    target_over_p = poisson_inversion.poisson_over(home_rate, away_rate, rho, ou_line, max_goals=10)

    implied_home_rate, implied_away_rate, residual_norm = poisson_inversion.implied_goal_rates_dc(
        target_home_p, target_draw_p, target_away_p, ou_line, target_over_p, rho=rho, max_goals=10
    )

    assert implied_home_rate == pytest.approx(home_rate, abs=1e-3)
    assert implied_away_rate == pytest.approx(away_rate, abs=1e-3)
    assert residual_norm == pytest.approx(0.0, abs=1e-6)


def test_scoreline_probabilities_aggregates_to_outcome_probabilities() -> None:
    """Summing the scoreline matrix by outcome should match poisson_probabilities."""
    home_rate, away_rate, max_goals = 1.5, 1.2, 10
    matrix = poisson_inversion.scoreline_probabilities(home_rate, away_rate, max_goals)
    home_p, draw_p, away_p = poisson_inversion.poisson_probabilities(home_rate, away_rate, max_goals)

    assert np.tril(matrix, -1).sum() == pytest.approx(home_p)
    assert np.trace(matrix) == pytest.approx(draw_p)
    assert np.triu(matrix, 1).sum() == pytest.approx(away_p)


def test_scoreline_probabilities_dc_shape_and_sum() -> None:
    """The DC scoreline matrix should have the expected shape and sum close to 1."""
    matrix = poisson_inversion.scoreline_probabilities_dc(1.4, 1.1, rho=-0.13, max_goals=10)

    assert matrix.shape == (11, 11)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-3)
    assert np.all(matrix >= 0)


def test_scoreline_probabilities_dc_aggregates_to_dc_outcome_probabilities() -> None:
    """Summing the DC matrix by outcome should match poisson_probs_dc."""
    lh, la, rho, max_goals = 1.4, 1.1, -0.13, 10
    matrix = poisson_inversion.scoreline_probabilities_dc(lh, la, rho, max_goals)
    p_home, p_draw, p_away = poisson_inversion.poisson_probs_dc(lh, la, rho, max_goals)

    assert np.tril(matrix, -1).sum() == pytest.approx(p_home, abs=1e-9)
    assert np.trace(matrix) == pytest.approx(p_draw, abs=1e-9)
    assert np.triu(matrix, 1).sum() == pytest.approx(p_away, abs=1e-9)


def test_scoreline_probabilities_dc_differs_from_plain_only_in_low_cells() -> None:
    """DC correction should touch only the four low-score cells."""
    lh, la, rho, max_goals = 1.4, 1.1, -0.13, 10
    dc = poisson_inversion.scoreline_probabilities_dc(lh, la, rho, max_goals)
    plain = poisson_inversion.scoreline_probabilities(lh, la, max_goals)

    diff = ~np.isclose(dc, plain)
    changed = set(zip(*np.where(diff)))
    assert changed.issubset({(0, 0), (0, 1), (1, 0), (1, 1)})
