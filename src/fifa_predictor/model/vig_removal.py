"""Removes bookmaker overround (vigorish) from odds to recover implied probabilities."""

import numpy as np
from scipy.optimize import brentq

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def odds_to_implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to raw implied probabilities.

    Args:
        odds: Array of decimal odds (e.g. [home, draw, away]).

    Returns:
        Array of implied probabilities, one per odd, summing to more than 1
        due to the bookmaker's overround.
    """
    imp_probs = 1 / odds
    return imp_probs


def remove_vig(probabilities: np.ndarray) -> np.ndarray:
    """Remove the bookmaker's overround with the power method.

    Takes raw implied probabilities (1 / odds), whose sum (the "booksum") exceeds
    1 by the overround, and returns fair probabilities summing to 1. The power
    method finds the exponent k with sum(p_i ** k) == 1 and returns p_i ** k.
    Because outcomes near probability 1 shrink slowest under exponentiation,
    favorites gain probability relative to the proportional method, correcting the
    favorite-longshot bias.

    When the booksum is <= 1 (no margin, e.g. already-fair odds) the method has
    nothing to remove and returns the proportionally normalized input, so applying
    remove_vig to fair odds is an identity.

    Args:
        probabilities: Raw implied probabilities (1 / odds), normally summing to
            more than 1.

    Returns:
        Fair probabilities summing to 1 (or the input unchanged if its total is
        non-positive).
    """
    total = probabilities.sum()
    if total <= 0:
        logger.warning("Total implied probability is non-positive, cannot remove vig.")
        return probabilities
    if total <= 1.0 + 1e-9:
        return probabilities / total

    def overround(k: float) -> float:
        return float(np.sum(probabilities ** k) - 1.0)

    k = brentq(overround, 1.0, 100.0)
    fair = probabilities ** k
    return fair / fair.sum()
