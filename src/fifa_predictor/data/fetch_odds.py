"""Fetches bookmaker odds for upcoming and historical international matches."""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

_API_BASE = "https://api.the-odds-api.com/v4"
_REGIONS = "eu"
_MARKETS = "h2h"
_REQUEST_TIMEOUT = 30

# Maps the project's competition identifiers to The Odds API sport keys.
_SPORT_KEYS = {
    "world_cup_2026": "soccer_fifa_world_cup",
}

_COLUMNS = [
    "match_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "home_odds",
    "draw_odds",
    "away_odds",
]

_EVENT_COLUMNS = ["match_id", "commence_time", "home_team", "away_team"]


def _request_json(competition: str, endpoint: str, params: dict | None = None):
    """Call an Odds API endpoint for a competition and return the parsed JSON.

    Handles the concerns shared by every fetch: the ODDS_API_KEY check (fail fast,
    before any request), the competition-to-sport_key mapping, and non-200
    logging followed by raise_for_status.

    Args:
        competition: Project competition identifier (e.g. "world_cup_2026").
        endpoint: Path segment under the sport key (e.g. "odds" or "events").
        params: Extra query params; apiKey is added automatically.

    Returns:
        The decoded JSON body of the response.
    """
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Add it to your project-root .env file."
        )

    try:
        sport_key = _SPORT_KEYS[competition]
    except KeyError:
        raise ValueError(
            f"Unknown competition {competition!r}; known: {sorted(_SPORT_KEYS)}"
        ) from None

    url = f"{_API_BASE}/sports/{sport_key}/{endpoint}"
    response = requests.get(
        url,
        params={"apiKey": api_key, **(params or {})},
        timeout=_REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        logger.error(
            "Odds API returned %s: %s",
            response.status_code,
            response.text[:500],
        )
    response.raise_for_status()
    return response.json()


def fetch_odds(competition: str) -> pd.DataFrame:
    """Retrieve bookmaker odds for the earliest-kickoff match in the competition.

    Args:
        competition: Name or identifier of the competition (e.g. "world_cup_2026").

    Returns:
        A tidy DataFrame with one row per bookmaker for the selected match, with
        columns match_id, commence_time, home_team, away_team, bookmaker,
        home_odds, draw_odds, away_odds. Returns an empty, correctly-typed
        DataFrame if the API reports no matches.
    """
    logger.info("Requesting odds for %s", competition)
    matches = _request_json(
        competition,
        "odds",
        {"regions": _REGIONS, "markets": _MARKETS, "oddsFormat": "decimal"},
    )
    logger.info("API returned %d match(es)", len(matches))

    if not matches:
        logger.warning("No matches returned for %s; returning empty frame", competition)
        return _build_frame([])

    match = min(matches, key=lambda m: m["commence_time"])
    rows = _parse_match(match)
    logger.info("Parsed %d bookmaker row(s) for match %s", len(rows), match["id"])
    return _build_frame(rows)


def list_events(competition: str) -> pd.DataFrame:
    """List the scheduled fixtures for a competition (no odds, free API call).

    Args:
        competition: Name or identifier of the competition (e.g. "world_cup_2026").

    Returns:
        A DataFrame with one row per fixture, columns match_id, commence_time,
        home_team, away_team.
    """
    logger.info("Requesting events for %s", competition)
    events = _request_json(competition, "events")
    logger.info("API returned %d event(s)", len(events))

    if not events:
        logger.warning("No events returned for %s; returning empty frame", competition)
        return _build_event_frame([])

    rows = [
        {
            "match_id": event["id"],
            "commence_time": event["commence_time"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
        }
        for event in events
    ]
    return _build_event_frame(rows)


def _build_event_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble fixture rows into a typed frame sorted by ascending commence_time."""
    df = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    return df.sort_values("commence_time", ignore_index=True)


def _parse_match(match: dict) -> list[dict]:
    """Flatten one match's bookmakers into tidy row dicts, skipping malformed ones."""
    rows: list[dict] = []
    home_team = match["home_team"]
    away_team = match["away_team"]
    for bookmaker in match.get("bookmakers", []):
        prices = _h2h_prices(bookmaker, home_team, away_team)
        if prices is None:
            continue
        rows.append(
            {
                "match_id": match["id"],
                "commence_time": match["commence_time"],
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker["key"],
                **prices,
            }
        )
    return rows


def _h2h_prices(bookmaker: dict, home_team: str, away_team: str) -> dict | None:
    """Return home/draw/away odds for a bookmaker, or None if h2h data is missing."""
    market = next(
        (m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None
    )
    if market is None:
        logger.warning(
            "Bookmaker %s has no h2h market; skipping", bookmaker.get("key")
        )
        return None

    by_name = {o["name"]: o["price"] for o in market.get("outcomes", [])}
    try:
        return {
            "home_odds": float(by_name[home_team]),
            "draw_odds": float(by_name["Draw"]),
            "away_odds": float(by_name[away_team]),
        }
    except KeyError as missing:
        logger.warning(
            "Bookmaker %s missing outcome %s; skipping",
            bookmaker.get("key"),
            missing,
        )
        return None


def _build_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble parsed rows into a typed DataFrame with the canonical columns."""
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    for col in ("home_odds", "draw_odds", "away_odds"):
        df[col] = df[col].astype(float)
    return df


def save_odds(odds: pd.DataFrame, destination: Path) -> None:
    """Persist fetched odds data to disk.

    Args:
        odds: DataFrame of odds to save.
        destination: Path to the output CSV file. Parent directories are created
            if they do not already exist.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(destination, index=False)
    logger.info("Wrote %d row(s) to %s", len(odds), destination)
