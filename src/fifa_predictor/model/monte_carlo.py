"""Monte Carlo tournament simulation built on top of scoreline prediction models."""

import numpy as np
import pandas as pd

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def simulate_match(scoreline_matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a single match scoreline from a probability matrix.

    Args:
        scoreline_matrix: Matrix where entry [i, j] is the probability of
            a final score of i-j.
        rng: NumPy random generator used for sampling.

    Returns:
        A tuple of (home_goals, away_goals) sampled for the match.
    """
    raise NotImplementedError


def simulate_tournament(
    fixtures: pd.DataFrame, params: dict[str, np.ndarray], n_simulations: int
) -> pd.DataFrame:
    """Run repeated Monte Carlo simulations of a tournament bracket.

    Args:
        fixtures: DataFrame describing the tournament structure and fixtures.
        params: Fitted model parameters used to generate scoreline matrices.
        n_simulations: Number of full tournament simulations to run.

    Returns:
        A DataFrame summarizing simulation outcomes, such as each team's
        probability of reaching each stage and winning the tournament.
    """
    raise NotImplementedError
