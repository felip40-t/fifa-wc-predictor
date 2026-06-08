"""Inverts match outcome probabilities into implied Poisson goal-scoring rates."""

import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.stats import poisson

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

def poisson_probabilities(home_rate: float, away_rate: float, max_goals: int) -> tuple[float, float, float]:
    """Compute match outcome probabilities from Poisson goal-scoring rates.

    Args:
        home_rate: Expected number of goals for the home team.
        away_rate: Expected number of goals for the away team.
        max_goals: Maximum number of goals to consider per team.
    Returns:
        A tuple of (home_win_prob, draw_prob, away_win_prob).
    """
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, home_rate)
    pa = poisson.pmf(goals, away_rate)
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
    solution = fsolve(equations, initial_guess, args=(home_win_prob, draw_prob))
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
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, home_rate)
    pa = poisson.pmf(goals, away_rate)
    return np.outer(ph, pa)


def scoreline_probabilities_dc(
    lh: float, la: float, rho: float, max_goals: int
) -> np.ndarray:
    """Compute a Dixon-Coles adjusted scoreline probability matrix.

    Starts from the independent-Poisson outer product and reweights each cell
    by the DC correction tau(h, a, lh, la, rho). Only the four low-score cells
    differ from the plain matrix. Keeping this matrix DC-consistent matters
    because implied_goal_rates_dc fits (lh, la) under the DC model; simulating
    from a non-DC matrix would not match the fitted rates.

    Args:
        lh: Expected number of goals for the home team.
        la: Expected number of goals for the away team.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A (max_goals + 1) x (max_goals + 1) matrix where entry [h, a] is the
        DC-adjusted probability of a final score of h-a.
    """
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, lh)
    pa = poisson.pmf(goals, la)
    matrix = np.outer(ph, pa)
    for h, a in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[h, a] *= tau(h, a, lh, la, rho)
    return matrix

def tau(h, a, lh, la, rho):
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    elif h == 0 and a == 1:
        return 1 + lh * rho
    elif h == 1 and a == 0:
        return 1 + la * rho
    elif h == 1 and a == 1:
        return 1 - rho
    else:
        return 1.0

def poisson_probs_dc(lh, la, rho=0.0, max_goals=10):
    p_home, p_draw, p_away = 0.0, 0.0, 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = tau(h, a, lh, la, rho) * poisson.pmf(h, lh) * poisson.pmf(a, la)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

    return p_home, p_draw, p_away


def poisson_over(lh: float, la: float, rho: float, ou_line: float, max_goals: int) -> float:
    """Compute P(total goals > ou_line) under the Dixon-Coles adjusted Poisson model.

    Args:
        lh: Expected number of goals for the home team.
        la: Expected number of goals for the away team.
        rho: Dixon-Coles low-score correlation parameter.
        ou_line: Over/under goals line (e.g. 2.5).
        max_goals: Maximum number of goals to consider per team.

    Returns:
        The probability that total goals (home + away) exceed ou_line.
    """
    if lh <= 0 or la <= 0:
        raise ValueError("Goal rates lh and la must be positive.")
    if not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1.")

    p_over = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > ou_line:
                p_over += tau(h, a, lh, la, rho) * poisson.pmf(h, lh) * poisson.pmf(a, la)

    return p_over


def _dc_residuals(
    rates: tuple[float, float],
    target_home_p: float,
    target_draw_p: float,
    target_over_p: float,
    rho: float,
    ou_line: float,
    max_goals: int,
) -> np.ndarray:
    """Residuals between DC-model 1X2/over probabilities and their market targets."""
    lh, la = rates
    p_home, p_draw, _ = poisson_probs_dc(lh, la, rho, max_goals)
    p_over = poisson_over(lh, la, rho, ou_line, max_goals)
    return np.array([p_home - target_home_p, p_draw - target_draw_p, p_over - target_over_p])


def implied_goal_rates_dc(
    p_home: float,
    p_draw: float,
    p_away: float,
    ou_line: float,
    p_over: float,
    rho: float = -0.13,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Derive implied DC-adjusted goal rates from 1X2 and over/under probabilities.

    Solves the overdetermined system (P(home) = p_home, P(draw) = p_draw,
    P(over ou_line) = p_over) for the two unknowns (lh, la) in a least-squares
    sense, since the over/under market gives one constraint more than needed.

    Args:
        p_home: Vig-free implied probability of a home win.
        p_draw: Vig-free implied probability of a draw.
        p_away: Vig-free implied probability of an away win.
        ou_line: Over/under goals line (e.g. 2.5).
        p_over: Vig-free implied probability that total goals exceed ou_line.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A tuple of (home_rate, away_rate, residual_norm), where residual_norm
        is the Euclidean norm of the final residual vector and indicates how
        well the solution fits all three market constraints.
    """
    if not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1.")

    # Initial guess: total expected goals ~ ou_line, split by the home/away
    # implied-probability ratio as a proxy for the lh/la ratio. Computed via
    # the probability shares rather than a raw ratio so it stays finite even
    # when one side's win probability is near zero.
    win_p_total = p_home + p_away
    lh0 = max(ou_line * p_home / win_p_total, 1e-3)
    la0 = max(ou_line * p_away / win_p_total, 1e-3)

    result = least_squares(
        _dc_residuals,
        x0=(lh0, la0),
        args=(p_home, p_draw, p_over, rho, ou_line, max_goals),
        bounds=((1e-6, 1e-6), (np.inf, np.inf)),
    )
    lh, la = result.x
    residual_norm = float(np.linalg.norm(result.fun))
    return lh, la, residual_norm