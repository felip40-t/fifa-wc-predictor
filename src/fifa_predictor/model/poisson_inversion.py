"""Inverts match outcome probabilities into implied Poisson goal-scoring rates."""

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import poisson

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

def poisson_probabilities(home_rate: float, away_rate: float, max_goals: int) -> np.ndarray:
    """Compute a probability matrix over scorelines from Poisson goal rates.

    Args:
        home_rate: Expected number of goals for the home team.
        away_rate: Expected number of goals for the away team.
        max_goals: Maximum number of goals to consider per team.
    Returns:
        
    """
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, lh)  # shape (11,)
    pa = poisson.pmf(goals, la)  # shape (11,)
    joint = np.outer(ph, pa)     # joint[h, a] = P(H=h, A=a)
    
    p_home = np.tril(joint, -1).sum()  # h > a
    p_draw = np.trace(joint)
    p_away = np.triu(joint, 1).sum()   # a > h
    return p_home, p_draw, p_away


def equations(rates: tuple[float, float], target_home_p: float, target_draw_p: float) -> tuple[float, float]:
    """System of equations relating Poisson rates to outcome probabilities."""
    lh, la = rates
    if lh <= 0 or la <= 0:
        return (np.inf, np.inf)
    p_home, p_draw, _ = poisson_probabilities(lh, la, max_goals=10)
    return (p_home - target_home_p, p_draw - target_draw_p)
    

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
    implied_total = -np.log(draw_prob) * 1.3  # rough guess for total goals
    ratio = away_win_prob / home_win_prob  
    x0 = (implied_total / (1 + ratio), implied_total * ratio / (1 + ratio))

    # Use fsolve to find the rates that satisfy the equations
    initial_guess = x0
    solution = fsolve(equations, initial_guess)
    return tuple(solution)


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
