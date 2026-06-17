"""Inverts match outcome probabilities into implied Poisson goal-scoring rates."""

import math

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

def scoreline_probabilities_dc(
    lh: float, la: float, rho: float, max_goals: int
) -> np.ndarray:
    """Compute a Dixon-Coles adjusted scoreline probability matrix.

    Starts from the independent-Poisson outer product and reweights each cell
    by the DC correction tau(h, a, lh, la, rho). Only the four low-score cells
    differ from the plain matrix. Keeping this matrix DC-consistent matters
    because implied_goal_rates_dc fits (lh, la, rho) under the DC model; simulating
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


def _dc_total_pmf(lh: float, la: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Un-normalized Dixon-Coles pmf over total goals (home + away).

    Sums the tau-reweighted independent-Poisson scoreline matrix along its
    anti-diagonals. Not normalized: the DC tau correction reweights four cells,
    so the total mass is near but not exactly 1. Callers that need a probability
    normalize by the returned mass.

    Args:
        lh: Expected number of goals for the home team.
        la: Expected number of goals for the away team.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A length (2 * max_goals + 1) array where entry k is the un-normalized
        probability mass of a total of k goals.
    """
    matrix = scoreline_probabilities_dc(lh, la, rho, max_goals)

    pmf = np.zeros(2 * max_goals + 1)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            pmf[h + a] += matrix[h, a]
    return pmf


def fair_over_probability(
    lh: float, la: float, rho: float, line: float, max_goals: int = 10
) -> float:
    """Money-weighted fair Over probability for any 0.25-grid totals line.

    Decomposes the line into the components a bookmaker actually prices and
    normalizes so Over + Under == 1 for every line type:

    - Half line (e.g. 2.5): P(total >= ceil(line)) / mass.
    - Integer line (e.g. 2.0, has a push): P(total > line) / (mass - P(total = line)),
      i.e. the push mass is removed from the normalization.
    - Quarter line (e.g. 2.25 / 2.75): the equal blend of the two half-stake
      components it splits into.

    On a half line this equals the normalized form of poisson_over; the primary
    constraint in the inversion still calls poisson_over directly, so single-line
    games are unaffected by this function.

    Args:
        lh: Expected number of goals for the home team.
        la: Expected number of goals for the away team.
        rho: Dixon-Coles low-score correlation parameter.
        line: Over/under line on the 0.25 grid (e.g. 2.0, 2.25, 2.5, 2.75).
        max_goals: Maximum number of goals to consider per team.

    Returns:
        The fair probability that the bet settles Over, in [0, 1].
    """
    # Validate the line is on the 0.25 grid before dispatching.
    quarter_steps = round(line * 4)
    if not math.isclose(line, quarter_steps / 4, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Unsupported line {line!r}; expected a value on the 0.25 grid.")

    pmf = _dc_total_pmf(lh, la, rho, max_goals)
    mass = pmf.sum()

    def half_over(half_line: float) -> float:
        threshold = math.ceil(half_line)  # 2.5 -> 3, i.e. P(total >= 3)
        return pmf[threshold:].sum() / mass

    def integer_over(n: int) -> float:
        over = pmf[n + 1:].sum()
        return over / (mass - pmf[n])  # remove the push mass at total == n

    base = math.floor(line)
    frac = quarter_steps % 4  # 0, 1, 2, 3 for .0, .25, .5, .75
    if frac == 0:
        return integer_over(base)
    if frac == 2:
        return half_over(line)
    if frac == 1:
        return 0.5 * integer_over(base) + 0.5 * half_over(base + 0.5)
    if frac == 3:
        return 0.5 * half_over(base + 0.5) + 0.5 * integer_over(base + 1)
    raise ValueError(f"Unsupported line {line!r}; expected a value on the 0.25 grid.")


def _dc_residuals(
    params: tuple[float, float, float],
    target_home_p: float,
    target_draw_p: float,
    target_over_p: float,
    ou_line: float,
    max_goals: int,
) -> np.ndarray:
    """Residuals between DC-model 1X2/over probabilities and their market targets.

    Solves for the three parameters (lh, la, rho) against the three core market
    constraints (home, draw, primary over). Three unknowns for three constraints
    makes the system exactly determined, so a clean fit drives every residual to
    ~0 and reproduces the de-vigged market prices, including the draw.
    """
    lh, la, rho = params
    p_home, p_draw, _ = poisson_probs_dc(lh, la, rho, max_goals)
    p_over = poisson_over(lh, la, rho, ou_line, max_goals)
    return np.array(
        [
            p_home - target_home_p,
            p_draw - target_draw_p,
            p_over - target_over_p,
        ]
    )


# Bounds on the fitted Dixon-Coles correlation. Wide enough to cover the
# market-implied values seen across a full slate (audit: mean ~-0.04, range
# roughly [-0.28, +0.16]) while keeping the four tau cells positive for normal
# goal rates. A solve that lands on a bound is rejected for the fixed-rho fallback.
RHO_BOUNDS = (-0.35, 0.20)


def _tau_cells_positive(lh: float, la: float, rho: float) -> bool:
    """True if all four Dixon-Coles tau cells are strictly positive.

    A non-positive tau would make a low-score cell non-positive, so a fitted rho
    that violates this is rejected in favour of the fixed-rho fallback.
    """
    return all(tau(h, a, lh, la, rho) > 0 for h, a in ((0, 0), (0, 1), (1, 0), (1, 1)))


def implied_goal_rates_dc(
    p_home: float,
    p_draw: float,
    p_away: float,
    ou_line: float,
    p_over: float,
    rho: float = -0.13,
    max_goals: int = 10,
) -> tuple[float, float, float, float]:
    """Derive implied DC goal rates and correlation from 1X2 and over/under probs.

    Solves the system (P(home) = p_home, P(draw) = p_draw, P(over ou_line) =
    p_over) for the three unknowns (lh, la, rho). Three unknowns for three market
    constraints makes the system exactly determined, so a clean fit reproduces all
    three de-vigged market prices with a ~0 residual and the draw is no longer
    sacrificed to fit the home and over prices. rho is fitted per game, seeded from
    the rho argument and bounded to RHO_BOUNDS.

    If the free-rho fit fails, lands on a rho bound, or produces a non-positive
    Dixon-Coles tau cell, the solve falls back to holding rho fixed at the seed and
    fitting only (lh, la); the returned residual_norm (~0 on a clean free fit) is
    then non-zero, flagging the fallback.

    Args:
        p_home: Vig-free implied probability of a home win.
        p_draw: Vig-free implied probability of a draw.
        p_away: Vig-free implied probability of an away win.
        ou_line: Over/under goals line (e.g. 2.5).
        p_over: Vig-free implied probability that total goals exceed ou_line.
        rho: Starting seed for the fitted correlation, and the value held fixed if
            the free fit falls back (default -0.13).
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A tuple of (home_rate, away_rate, rho, residual_norm). rho is the fitted
        correlation (or the seed if the fit fell back); residual_norm is the
        Euclidean norm of the final residual vector.
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

    rho_lo, rho_hi = RHO_BOUNDS
    rho_seed = min(max(rho, rho_lo), rho_hi)
    result = least_squares(
        _dc_residuals,
        x0=(lh0, la0, rho_seed),
        args=(p_home, p_draw, p_over, ou_line, max_goals),
        bounds=((1e-6, 1e-6, rho_lo), (np.inf, np.inf, rho_hi)),
    )
    lh, la, rho_fit = result.x
    at_bound = abs(rho_fit - rho_lo) < 1e-6 or abs(rho_fit - rho_hi) < 1e-6
    if result.success and not at_bound and _tau_cells_positive(lh, la, rho_fit):
        return lh, la, rho_fit, float(np.linalg.norm(result.fun))

    # Fallback: hold rho fixed at the seed and fit only (lh, la), the original
    # over-determined behaviour. The non-zero residual flags the affected game.
    logger.warning(
        "Free-rho fit rejected (success=%s, rho_fit=%.3f); holding rho fixed at %.3f.",
        result.success,
        rho_fit,
        rho,
    )
    fixed = least_squares(
        lambda rates: _dc_residuals(
            (rates[0], rates[1], rho), p_home, p_draw, p_over, ou_line, max_goals
        ),
        x0=(lh0, la0),
        bounds=((1e-6, 1e-6), (np.inf, np.inf)),
    )
    lh, la = fixed.x
    return lh, la, rho, float(np.linalg.norm(fixed.fun))