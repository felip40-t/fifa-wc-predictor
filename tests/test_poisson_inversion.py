"""Tests for the Poisson inversion model module."""

import numpy as np
import pytest

from fifa_predictor.model import poisson_inversion
from fifa_predictor.model.poisson_inversion import (
    fair_over_probability,
    implied_goal_rates_dc,
    poisson_over,
    poisson_probs_dc,
    _dc_residuals,
    _dc_total_pmf,
)


def _fair_under_probability(lh, la, rho, line, max_goals=10):
    """Mirror of fair_over_probability for Under, used only to assert over+under==1."""
    return 1.0 - fair_over_probability(lh, la, rho, line, max_goals)


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

    implied_home_rate, implied_away_rate, implied_rho, residual_norm = (
        poisson_inversion.implied_goal_rates_dc(
            target_home_p, target_draw_p, target_away_p, ou_line, target_over_p, max_goals=10
        )
    )

    assert implied_home_rate == pytest.approx(home_rate, abs=1e-3)
    assert implied_away_rate == pytest.approx(away_rate, abs=1e-3)
    assert implied_rho == pytest.approx(rho, abs=1e-3)
    assert residual_norm == pytest.approx(0.0, abs=1e-6)


def test_implied_goal_rates_dc_matches_all_three_markets() -> None:
    """With rho free, the fit reproduces 1X2 AND over/under, not just a compromise.

    These targets are not DC-consistent at a fixed rho: the draw and over/under
    markets pull the total in different directions. The two-rate (fixed-rho) fit
    cannot satisfy all three and sacrifices the draw; freeing rho as a third
    parameter makes the system exactly determined, so every market is matched.
    """
    p_home, p_draw, p_away = 0.45, 0.27, 0.28
    ou_line, p_over = 2.5, 0.52

    lh, la, rho, residual_norm = poisson_inversion.implied_goal_rates_dc(
        p_home, p_draw, p_away, ou_line, p_over, max_goals=10
    )

    fit_home, fit_draw, _ = poisson_inversion.poisson_probs_dc(lh, la, rho, max_goals=10)
    fit_over = poisson_inversion.poisson_over(lh, la, rho, ou_line, max_goals=10)

    assert fit_home == pytest.approx(p_home, abs=1e-3)
    assert fit_draw == pytest.approx(p_draw, abs=1e-3)
    assert fit_over == pytest.approx(p_over, abs=1e-3)
    assert residual_norm == pytest.approx(0.0, abs=1e-3)
    # rho stays within the Dixon-Coles sensible band.
    low, high = poisson_inversion.RHO_BOUNDS
    assert low <= rho <= high


def test_implied_goal_rates_dc_respects_rho_bounds() -> None:
    """The fitted rho never escapes the Dixon-Coles bounds, even under strain.

    A heavy favourite with a draw/over tension that would want rho below the
    lower bound must clip to the bound rather than running off to keep tau valid.
    """
    lh, la, rho, _ = poisson_inversion.implied_goal_rates_dc(
        0.18, 0.30, 0.52, 2.5, 0.60, max_goals=10
    )
    low, high = poisson_inversion.RHO_BOUNDS
    assert low <= rho <= high


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


def test_total_pmf_sums_to_dc_mass():
    """The DC pmf mass is preserved by the tau correction and is within ~1e-7 of 1.0."""
    pmf = _dc_total_pmf(1.4, 1.1, -0.05, max_goals=10)
    assert pmf.shape == (21,)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-4)


def test_fair_over_half_line_matches_normalized_poisson_over():
    lh, la, rho = 1.4, 1.1, -0.05
    pmf = _dc_total_pmf(lh, la, rho, max_goals=10)
    expected = pmf[3:].sum() / pmf.sum()
    assert fair_over_probability(lh, la, rho, 2.5) == pytest.approx(expected)
    assert fair_over_probability(lh, la, rho, 2.5) == pytest.approx(
        poisson_over(lh, la, rho, 2.5, 10), abs=5e-4
    )


def test_fair_over_integer_line_removes_push_mass():
    lh, la, rho = 1.4, 1.1, -0.05
    pmf = _dc_total_pmf(lh, la, rho, max_goals=10)
    expected = pmf[3:].sum() / (pmf.sum() - pmf[2])
    assert fair_over_probability(lh, la, rho, 2.0) == pytest.approx(expected)


def test_fair_over_quarter_line_is_the_blend():
    lh, la, rho = 1.4, 1.1, -0.05
    over_225 = fair_over_probability(lh, la, rho, 2.25)
    over_20 = fair_over_probability(lh, la, rho, 2.0)
    over_25 = fair_over_probability(lh, la, rho, 2.5)
    assert over_225 == pytest.approx(0.5 * over_20 + 0.5 * over_25)


def test_fair_over_quarter_line_75_is_the_blend():
    lh, la, rho = 1.4, 1.1, -0.05
    over_275 = fair_over_probability(lh, la, rho, 2.75)
    over_25 = fair_over_probability(lh, la, rho, 2.5)
    over_30 = fair_over_probability(lh, la, rho, 3.0)
    assert over_275 == pytest.approx(0.5 * over_25 + 0.5 * over_30)


@pytest.mark.parametrize("line", [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5])
def test_fair_over_in_unit_interval_and_over_plus_under_is_one(line):
    lh, la, rho = 1.6, 1.3, -0.08
    over = fair_over_probability(lh, la, rho, line)
    assert 0.0 <= over <= 1.0
    under = _fair_under_probability(lh, la, rho, line)
    assert over + under == pytest.approx(1.0)


def test_fair_over_is_monotone_decreasing_in_line():
    lh, la, rho = 1.6, 1.3, -0.08
    overs = [fair_over_probability(lh, la, rho, x) for x in [1.5, 2.0, 2.25, 2.5, 3.0]]
    assert all(a >= b for a, b in zip(overs, overs[1:]))


def test_fair_over_rejects_off_grid_line():
    with pytest.raises(ValueError):
        fair_over_probability(1.4, 1.1, -0.05, 2.1)


def test_dc_residuals_length_without_secondary():
    res = _dc_residuals((1.4, 1.1, -0.05), 0.4, 0.27, 0.5, 2.5, 10)
    assert res.shape == (3,)


def test_dc_residuals_length_with_secondary():
    res = _dc_residuals(
        (1.4, 1.1, -0.05), 0.4, 0.27, 0.5, 2.5, 10,
        secondary_line=2.25, target_over_secondary=0.55, secondary_weight=0.5,
    )
    assert res.shape == (4,)


def test_dc_residuals_secondary_term_is_weighted():
    params = (1.4, 1.1, -0.05)
    model_secondary = fair_over_probability(*params, 2.25, 10)
    res = _dc_residuals(
        params, 0.4, 0.27, 0.5, 2.5, 10,
        secondary_line=2.25, target_over_secondary=0.0, secondary_weight=0.5,  # target 0 so the residual is exactly weight * model_value
    )
    assert res[3] == pytest.approx(0.5 * model_secondary)


def test_inversion_round_trips_with_secondary_line():
    lh_true, la_true, rho_true = 1.5, 1.0, -0.06
    p_home, p_draw, p_away = poisson_probs_dc(lh_true, la_true, rho_true, 10)
    p_over_primary = poisson_over(lh_true, la_true, rho_true, 2.5, 10)
    p_over_secondary = fair_over_probability(lh_true, la_true, rho_true, 2.25, 10)

    lh, la, rho, residual_norm = implied_goal_rates_dc(
        p_home, p_draw, p_away, 2.5, p_over_primary,
        secondary_line=2.25, p_over_secondary=p_over_secondary,
    )
    assert lh == pytest.approx(lh_true, abs=1e-3)
    assert la == pytest.approx(la_true, abs=1e-3)
    assert rho == pytest.approx(rho_true, abs=1e-3)
    assert residual_norm < 1e-3


def test_inversion_without_secondary_is_unchanged_path():
    lh_true, la_true, rho_true = 1.5, 1.0, -0.06
    p_home, p_draw, p_away = poisson_probs_dc(lh_true, la_true, rho_true, 10)
    p_over = poisson_over(lh_true, la_true, rho_true, 2.5, 10)
    lh, la, rho, _ = implied_goal_rates_dc(p_home, p_draw, p_away, 2.5, p_over)
    assert lh == pytest.approx(lh_true, abs=0.02)
    assert la == pytest.approx(la_true, abs=0.02)
    assert rho == pytest.approx(rho_true, abs=0.02)


def test_dc_residuals_nan_secondary_treated_as_absent():
    """A NaN secondary line (no second line in the CSV) yields length-3 residuals."""
    res = _dc_residuals(
        (1.4, 1.1, -0.05), 0.4, 0.27, 0.5, 2.5, 10,
        secondary_line=float("nan"), target_over_secondary=0.55,
    )
    assert res.shape == (3,)


def test_inversion_requires_secondary_prob_when_line_given():
    with pytest.raises(ValueError):
        implied_goal_rates_dc(0.4, 0.27, 0.33, 2.5, 0.5, secondary_line=2.25)
