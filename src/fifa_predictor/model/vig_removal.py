"""Removes bookmaker overround (vigorish) from odds to recover implied probabilities."""

import numpy as np

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
    """Normalize raw implied probabilities to remove the bookmaker's overround.

    Args:
        probabilities: Array of raw implied probabilities that sum to more than 1.

    Returns:
        Array of fair (vig-free) probabilities summing to 1.
    """
    total = probabilities.sum()
    if total <= 0:
        logger.warning("Total implied probability is non-positive, cannot remove vig.")
        return probabilities  # Return as-is to avoid division by zero
    fair_probs = probabilities / total
    return fair_probs
