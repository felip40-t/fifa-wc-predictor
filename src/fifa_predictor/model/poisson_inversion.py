"""Inverts match outcome probabilities into implied Poisson goal-scoring rates."""

import numpy as np

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def implied_goal_rates(
    home_win_prob: float, draw_prob: float, away_win_prob: float
) -> tuple[float, float]:
    """Derive implied home and away goal-scoring rates from outcome probabilities.

    Args:
        home_win_prob: Fair probability of a home win.
        draw_prob: Fair probability of a draw.
        away_win_prob: Fair probability of an away win.

    Returns:
        A tuple of (home_rate, away_rate) Poisson goal-scoring rates that
        reproduce the given outcome probabilities.
    """
    raise NotImplementedError


def scoreline_probabilities(
    home_rate: float, away_rate: float, max_goals: int
) -> np.ndarray:
    """Compute a probability matrix over scorelines from Poisson goal rates.

    Args:
        home_rate: Expected number of goals for the home team.
        away_rate: Expected number of goals for the away team.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A (max_goals + 1) x (max_goals + 1) matrix where entry [i, j] is the
        probability of a final score of i-j.
    """
    raise NotImplementedError
