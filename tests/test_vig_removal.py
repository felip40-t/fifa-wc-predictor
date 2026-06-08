"""Tests for the vig removal model module."""

import numpy as np
import pytest

from fifa_predictor.model import vig_removal


def test_odds_to_implied_probabilities_inverts_odds() -> None:
    """Each implied probability should be the reciprocal of its decimal odd."""
    odds = np.array([2.0, 3.5, 4.0])

    imp_probs = vig_removal.odds_to_implied_probabilities(odds)

    np.testing.assert_allclose(imp_probs, 1 / odds)


def test_odds_to_implied_probabilities_sums_above_one() -> None:
    """Raw implied probabilities should sum to more than 1 due to the overround."""
    imp_probs = vig_removal.odds_to_implied_probabilities(np.array([2.0, 3.5, 4.0]))

    assert imp_probs.sum() > 1.0


def test_remove_vig_normalizes_to_one() -> None:
    """Fair probabilities should be proportional to the input and sum to 1."""
    raw_probs = np.array([0.5, 0.3, 0.3])

    fair_probs = vig_removal.remove_vig(raw_probs)

    assert fair_probs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(fair_probs, raw_probs / raw_probs.sum())


def test_remove_vig_handles_non_positive_total() -> None:
    """A non-positive total should be handled gracefully rather than dividing by zero."""
    raw_probs = np.array([0.0, 0.0, 0.0])

    fair_probs = vig_removal.remove_vig(raw_probs)

    np.testing.assert_array_equal(fair_probs, raw_probs)
