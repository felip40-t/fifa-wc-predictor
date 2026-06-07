"""Fetches bookmaker odds for upcoming and historical international matches."""

from pathlib import Path

import pandas as pd

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def fetch_odds(competition: str) -> pd.DataFrame:
    """Retrieve bookmaker odds for matches in the given competition.

    Args:
        competition: Name or identifier of the competition (e.g. "world_cup_2026").

    Returns:
        A DataFrame of odds with columns such as match_id, home_odds,
        draw_odds, and away_odds.
    """
    raise NotImplementedError


def save_odds(odds: pd.DataFrame, destination: Path) -> None:
    """Persist fetched odds data to disk.

    Args:
        odds: DataFrame of odds to save.
        destination: Path to the output file.
    """
    raise NotImplementedError
