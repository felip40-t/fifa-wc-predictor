"""Tests for vig removal (power method)."""

import numpy as np
import pytest

from fifa_predictor.model.vig_removal import odds_to_implied_probabilities, remove_vig


def test_fair_probabilities_sum_to_one():
    raw = odds_to_implied_probabilities(np.array([1.9, 3.8, 3.8]))
    fair = remove_vig(raw)
    assert fair.sum() == pytest.approx(1.0)


def test_symmetric_market_is_unchanged_by_power():
    raw = odds_to_implied_probabilities(np.array([1.9, 1.9]))
    fair = remove_vig(raw)
    assert fair == pytest.approx(np.array([0.5, 0.5]))


def test_power_gives_favorite_more_than_proportional():
    odds = np.array([1.5, 4.5, 7.0])
    raw = odds_to_implied_probabilities(odds)
    proportional = raw / raw.sum()
    fair = remove_vig(raw)
    assert fair[0] > proportional[0]
    assert fair[2] < proportional[2]
    assert fair.sum() == pytest.approx(1.0)


def test_already_fair_odds_are_returned_unchanged():
    fair_probs = np.array([0.5, 0.3, 0.2])
    raw = fair_probs.copy()
    out = remove_vig(raw)
    assert out == pytest.approx(fair_probs)


def test_ordering_is_preserved():
    raw = odds_to_implied_probabilities(np.array([1.6, 4.0, 5.0]))
    fair = remove_vig(raw)
    assert fair[0] > fair[1] > fair[2]


def test_two_way_market_with_margin_sums_to_one():
    raw = odds_to_implied_probabilities(np.array([1.8, 2.1]))
    fair = remove_vig(raw)
    assert fair.sum() == pytest.approx(1.0)
    assert fair[0] > fair[1]


def test_nonpositive_total_returns_input():
    raw = np.array([0.0, 0.0])
    out = remove_vig(raw)
    assert np.array_equal(out, raw)
