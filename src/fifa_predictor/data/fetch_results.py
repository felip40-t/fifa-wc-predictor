"""Fetches historical international match results used to train the prediction model."""

from pathlib import Path

import pandas as pd

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def fetch_results(start_year: int, end_year: int) -> pd.DataFrame:
    """Retrieve historical match results for the given year range.

    Args:
        start_year: First year (inclusive) to fetch results for.
        end_year: Last year (inclusive) to fetch results for.

    Returns:
        A DataFrame of match results with columns such as date, home_team,
        away_team, home_score, and away_score.
    """
    raise NotImplementedError


def save_results(results: pd.DataFrame, destination: Path) -> None:
    """Persist fetched match results to disk.

    Args:
        results: DataFrame of match results to save.
        destination: Path to the output file.
    """
    raise NotImplementedError
