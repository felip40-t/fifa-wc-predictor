"""Inverts match outcome probabilities into implied Poisson goal-scoring rates."""

import math

import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.stats import poisson

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

# Bounds on the Dixon-Coles low-score correlation rho when it is fitted per game.
# DC's hypothesis is that the correction is a small adjustment to the four
# low-score cells (their empirical estimate is ~-0.13). The lower bound also
# keeps tau non-negative for realistic rates: tau(0,1) = 1 + lh*rho and
# tau(1,0) = 1 + la*rho stay >= 0 as long as rho >= -1/max_rate, and -0.2
# covers home/away rates up to 5 goals. A symmetric band keeps it well away
# from the degenerate tau region while still letting rho move enough to make
# the 1X2 + over/under system exactly determined.
RHO_BOUNDS = (-0.2, 0.2)


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
    secondary_line: float | None = None,
    target_over_secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> np.ndarray:
    """Residuals between DC-model 1X2/over probabilities and their market targets.

    Fits three parameters (lh, la, rho) against the three core market constraints
    (home, draw, primary over), so that system is exactly determined. When a
    secondary over/under line is supplied, its residual is appended at
    secondary_weight (< 1) so the extra total sharpens the goal-distribution shape
    without out-voting the full-weight draw constraint. The primary over still uses
    poisson_over, so with no secondary the residuals are identical to before.
    """
    lh, la, rho = params
    p_home, p_draw, _ = poisson_probs_dc(lh, la, rho, max_goals)
    p_over = poisson_over(lh, la, rho, ou_line, max_goals)
    residuals = [
        p_home - target_home_p,
        p_draw - target_draw_p,
        p_over - target_over_p,
    ]
    if secondary_line is not None and not np.isnan(secondary_line):
        model_secondary = fair_over_probability(lh, la, rho, secondary_line, max_goals)
        residuals.append(secondary_weight * (model_secondary - target_over_secondary))
    return np.array(residuals)


def implied_goal_rates_dc(
    p_home: float,
    p_draw: float,
    p_away: float,
    ou_line: float,
    p_over: float,
    rho_init: float = -0.13,
    max_goals: int = 10,
    secondary_line: float | None = None,
    p_over_secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> tuple[float, float, float, float]:
    """Derive implied DC-adjusted goal rates from 1X2 and over/under probabilities.

    Solves the system (P(home) = p_home, P(draw) = p_draw, P(over ou_line) =
    p_over) for the three unknowns (lh, la, rho). With rho held fixed there are
    only two free parameters for three market constraints, so the system is
    over-determined and the least-squares compromise systematically sacrifices
    the draw. Fitting rho as well makes the system exactly determined, so all
    three markets are honoured. rho is bounded to RHO_BOUNDS to stay within the
    Dixon-Coles sensible band and keep the tau correction valid.

    Args:
        p_home: Vig-free implied probability of a home win.
        p_draw: Vig-free implied probability of a draw.
        p_away: Vig-free implied probability of an away win.
        ou_line: Over/under goals line (e.g. 2.5).
        p_over: Vig-free implied probability that total goals exceed ou_line.
        rho_init: Initial guess for the fitted Dixon-Coles correlation.
        max_goals: Maximum number of goals to consider per team.
        secondary_line: Optional second totals line (e.g. 2.25) to sharpen the
            goal-distribution shape; if None (default) the fit uses only the three
            core market constraints and behaviour is identical to the no-secondary path.
            p_over_secondary is required when secondary_line is given; both are
            ignored otherwise.
        p_over_secondary: Vig-free implied probability that total goals exceed
            secondary_line; required when secondary_line is supplied.
        secondary_weight: Weight applied to the secondary-line residual (default 0.5)
            so it sharpens the shape without out-voting the full-weight draw constraint.

    Returns:
        A tuple of (home_rate, away_rate, rho, residual_norm), where rho is the
        fitted Dixon-Coles correlation and residual_norm is the Euclidean norm
        of the final residual vector, indicating how well the solution fits all
        three market constraints.
    """
    rho_low, rho_high = RHO_BOUNDS
    if not rho_low <= rho_init <= rho_high:
        raise ValueError(f"rho_init must lie within RHO_BOUNDS {RHO_BOUNDS}.")
    if secondary_line is not None and p_over_secondary is None:
        raise ValueError("p_over_secondary is required when secondary_line is provided.")
    if not 0 <= secondary_weight <= 1:
        raise ValueError("secondary_weight must lie in [0, 1].")

    # Initial guess: total expected goals ~ ou_line, split by the home/away
    # implied-probability ratio as a proxy for the lh/la ratio. Computed via
    # the probability shares rather than a raw ratio so it stays finite even
    # when one side's win probability is near zero.
    win_p_total = p_home + p_away
    lh0 = max(ou_line * p_home / win_p_total, 1e-3)
    la0 = max(ou_line * p_away / win_p_total, 1e-3)

    result = least_squares(
        _dc_residuals,
        x0=(lh0, la0, rho_init),
        args=(
            p_home,
            p_draw,
            p_over,
            ou_line,
            max_goals,
            secondary_line,
            p_over_secondary,
            secondary_weight,
        ),
        bounds=((1e-6, 1e-6, rho_low), (np.inf, np.inf, rho_high)),
    )
    lh, la, rho = result.x
    residual_norm = float(np.linalg.norm(result.fun))
    return lh, la, rho, residual_norm