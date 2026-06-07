"""Tests for the vig removal model module."""

import numpy as np
import pytest

from fifa_predictor.model import vig_removal


def test_odds_to_implied_probabilities_not_implemented() -> None:
    """odds_to_implied_probabilities is a stub and should raise until implemented."""
    with pytest.raises(NotImplementedError):
        vig_removal.odds_to_implied_probabilities(np.array([2.0, 3.5, 4.0]))


def test_remove_vig_not_implemented() -> None:
    """remove_vig is a stub and should raise until implemented."""
    with pytest.raises(NotImplementedError):
        vig_removal.remove_vig(np.array([0.5, 0.3, 0.3]))
