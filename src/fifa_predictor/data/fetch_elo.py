"""Fetches national team Elo ratings used as model features."""

from pathlib import Path

import pandas as pd

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def fetch_elo_ratings(as_of_date: str) -> pd.DataFrame:
    """Retrieve national team Elo ratings as of a given date.

    Args:
        as_of_date: ISO-format date string (YYYY-MM-DD) to fetch ratings for.

    Returns:
        A DataFrame of Elo ratings with columns such as team and rating.
    """
    raise NotImplementedError


def save_elo_ratings(ratings: pd.DataFrame, destination: Path) -> None:
    """Persist fetched Elo ratings to disk.

    Args:
        ratings: DataFrame of Elo ratings to save.
        destination: Path to the output file.
    """
    raise NotImplementedError
