"""Fetches bookmaker odds for upcoming and historical international matches."""

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from fifa_predictor.model import vig_removal
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

_API_BASE = "https://api.the-odds-api.com/v4"
_REGIONS = "eu"
_MARKETS = "h2h,totals"
_REQUEST_TIMEOUT = 30

# Maps the project's competition identifiers to The Odds API sport keys.
_SPORT_KEYS = {
    "world_cup_2026": "soccer_fifa_world_cup",
}

_EVENT_COLUMNS = ["match_id", "commence_time", "home_team", "away_team"]

_GAME_COLUMNS = [
    "game_id",
    "home_team",
    "away_team",
    "commence_time",
    "pinnacle_h2h_home",
    "pinnacle_h2h_draw",
    "pinnacle_h2h_away",
    "pinnacle_ou_line",
    "pinnacle_ou_over",
    "pinnacle_ou_under",
    "pinnacle_ou2_line",
    "pinnacle_ou2_over",
    "pinnacle_ou2_under",
    "odds_source",
]

_FLOAT_COLUMNS = [
    "pinnacle_h2h_home",
    "pinnacle_h2h_draw",
    "pinnacle_h2h_away",
    "pinnacle_ou_line",
    "pinnacle_ou_over",
    "pinnacle_ou_under",
    "pinnacle_ou2_line",
    "pinnacle_ou2_over",
    "pinnacle_ou2_under",
]


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
    """Retrieve Pinnacle (or consensus) h2h + totals odds for every game.

    Fetches both the match-result (h2h) and over/under (totals) markets in a
    single API call. Odds are taken from Pinnacle when available, otherwise from
    the median consensus of all books, resolved independently per market. All
    stored odds are vig-free (de-vigged), so every row is on the same footing
    regardless of odds_source.

    Args:
        competition: Name or identifier of the competition (e.g. "world_cup_2026").

    Returns:
        A wide DataFrame, one row per game, with the columns in _GAME_COLUMNS.
        Missing markets are filled with NaN; odds_source records provenance
        (pinnacle / consensus / mixed). Returns an empty, correctly-typed frame
        if no games are priced.
    """
    logger.info("Requesting odds (h2h, totals) for %s", competition)
    games = _request_json(
        competition,
        "odds",
        {"regions": _REGIONS, "markets": _MARKETS, "oddsFormat": "decimal"},
    )
    logger.info("API returned %d game(s)", len(games))

    if not games:
        logger.warning("No games returned for %s; returning empty frame", competition)
        return _build_games_frame([])

    return _build_games_frame([_parse_game(g) for g in games])


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


def _pinnacle(game: dict) -> dict | None:
    return next((b for b in game.get("bookmakers", []) if b.get("key") == "pinnacle"), None)


def _market(bookmaker: dict, key: str) -> dict | None:
    return next((m for m in bookmaker.get("markets", []) if m.get("key") == key), None)


def _h2h_odds(market: dict, home_team: str, away_team: str) -> tuple | None:
    by_name = {o["name"]: o["price"] for o in market.get("outcomes", [])}
    try:
        return (float(by_name[home_team]), float(by_name["Draw"]), float(by_name[away_team]))
    except KeyError:
        return None


def _totals_lines(market: dict) -> list[tuple]:
    """Group a totals market's outcomes by line into (point, over, under) tuples."""
    by_point: dict = {}
    for o in market.get("outcomes", []):
        point = o.get("point")
        if point is None:
            continue
        by_point.setdefault(point, {})[o["name"]] = o["price"]
    return [(p, d.get("Over"), d.get("Under")) for p, d in by_point.items()]


def _on_grid(point: float) -> bool:
    """True if the line sits on the 0.5 grid (excludes quarter/Asian lines)."""
    return abs(point * 2 - round(point * 2)) < 1e-9


def _score_line(over: float, under: float) -> float:
    """Absolute gap between vig-removed Over and Under probabilities (0 = even)."""
    fair = vig_removal.remove_vig(vig_removal.odds_to_implied_probabilities(np.array([over, under])))
    return abs(fair[0] - fair[1])


def _select_line(lines: list[tuple]) -> tuple | None:
    """Pick the 0.5-grid totals line closest to 50/50 after vig removal.

    Lines missing either price or off the 0.5 grid are excluded. Ties break to
    the line closest to 2.5, then to the lower line.
    """
    eligible = [
        (p, o, u) for (p, o, u) in lines
        if o is not None and u is not None and _on_grid(p)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda t: (_score_line(t[1], t[2]), abs(t[0] - 2.5), t[0]))


def _on_quarter_grid(point: float) -> bool:
    """True if the line sits on the 0.25 grid (allows quarter/Asian lines)."""
    return abs(point * 4 - round(point * 4)) < 1e-9


def _select_pinnacle_secondary(market: dict) -> tuple | None:
    """Pinnacle's most balanced totals line on the strict quarter grid.

    Admits only lines that are on the 0.25 grid but NOT on the 0.5 grid (i.e.
    quarter/Asian lines like 2.25 or 2.75 that the primary 0.5-grid filter
    discards). Falls back to any 0.25-grid line (including 0.5-grid lines) when
    no strict quarter line exists.
    Ties break to the line closest to 2.5, then to the lower line.
    """
    lines = _totals_lines(market)
    quarter_only = [
        (p, o, u) for (p, o, u) in lines
        if o is not None and u is not None and _on_quarter_grid(p) and not _on_grid(p)
    ]
    eligible = quarter_only or [
        (p, o, u) for (p, o, u) in lines
        if o is not None and u is not None and _on_quarter_grid(p)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda t: (_score_line(t[1], t[2]), abs(t[0] - 2.5), t[0]))


def _resolve_secondary_totals(game: dict) -> tuple:
    """Resolve the secondary totals line: Pinnacle's sharpest line, else NaN.

    Sourced from Pinnacle only (the sharp book). Returns vig-free odds. NaN when
    Pinnacle posts no usable totals market.
    """
    pin = _pinnacle(game)
    if pin:
        market = _market(pin, "totals")
        if market:
            selected = _select_pinnacle_secondary(market)
            if selected:
                point, over, under = selected
                fair_over, fair_under = _fair_odds([over, under])
                return (point, fair_over, fair_under)
    return (float("nan"), float("nan"), float("nan"))


def _fair_probs(odds: list) -> np.ndarray:
    """De-vig a set of decimal odds into fair probabilities summing to 1."""
    return vig_removal.remove_vig(vig_removal.odds_to_implied_probabilities(np.array(odds, dtype=float)))


def _probs_to_odds(probs) -> tuple:
    """Convert fair probabilities into decimal odds (1 / p)."""
    return tuple(float(1.0 / p) for p in probs)


def _fair_odds(odds: list) -> tuple:
    """Strip vig from a set of decimal odds, returning fair decimal odds."""
    return _probs_to_odds(_fair_probs(odds))


def _consensus_h2h(bookmakers: list[dict], home_team: str, away_team: str) -> tuple | None:
    """Consensus h2h as vig-free odds.

    De-vig each book's prices into fair probabilities, median those probabilities
    per outcome across books, renormalize, then convert back to decimal odds. This
    aggregates in probability space rather than medianing raw (vig-laden) odds.
    """
    fair_triples = []
    for book in bookmakers:
        market = _market(book, "h2h")
        if not market:
            continue
        odds = _h2h_odds(market, home_team, away_team)
        if odds:
            fair_triples.append(_fair_probs(odds))
    if not fair_triples:
        return None
    probs = np.median(np.array(fair_triples), axis=0)
    probs = probs / probs.sum()
    return _probs_to_odds(probs)


def _consensus_totals(bookmakers: list[dict]) -> tuple | None:
    """Consensus totals as vig-free odds per line, then the balanced-line pick.

    For each line, de-vig each book's Over/Under into fair probabilities, median
    them across books, renormalize, and convert back to decimal odds before
    selecting the most balanced 0.5-grid line.
    """
    fair_by_line: dict = defaultdict(list)
    for book in bookmakers:
        market = _market(book, "totals")
        if not market:
            continue
        for point, over, under in _totals_lines(market):
            if over is None or under is None:
                continue
            fair = _fair_probs([over, under])
            fair_by_line[point].append((float(fair[0]), float(fair[1])))
    consensus_lines = []
    for point, probs in fair_by_line.items():
        median = np.median(np.array(probs), axis=0)
        median = median / median.sum()
        over_odds, under_odds = _probs_to_odds(median)
        consensus_lines.append((point, over_odds, under_odds))
    return _select_line(consensus_lines)


def _resolve_h2h(game: dict) -> tuple:
    """Resolve h2h odds: Pinnacle if available, else median consensus, else NaN."""
    pin = _pinnacle(game)
    if pin:
        market = _market(pin, "h2h")
        if market:
            odds = _h2h_odds(market, game["home_team"], game["away_team"])
            if odds:
                return (*_fair_odds(odds), "pinnacle")
    consensus = _consensus_h2h(game.get("bookmakers", []), game["home_team"], game["away_team"])
    if consensus is None:
        logger.warning("No h2h market for game %s", game["id"])
        return (float("nan"), float("nan"), float("nan"), "consensus")
    logger.warning("Pinnacle h2h unavailable for game %s; using consensus", game["id"])
    return (*consensus, "consensus")


def _resolve_totals(game: dict) -> tuple:
    """Resolve totals: Pinnacle's best line if available, else consensus, else NaN."""
    pin = _pinnacle(game)
    if pin:
        market = _market(pin, "totals")
        if market:
            selected = _select_line(_totals_lines(market))
            if selected:
                point, over, under = selected
                fair_over, fair_under = _fair_odds([over, under])
                return (point, fair_over, fair_under, "pinnacle")
    consensus = _consensus_totals(game.get("bookmakers", []))
    if consensus is None:
        logger.warning("No totals market for game %s", game["id"])
        return (float("nan"), float("nan"), float("nan"), "consensus")
    logger.warning("Pinnacle totals unavailable for game %s; using consensus", game["id"])
    return (*consensus, "consensus")


def _combine_source(src_h2h: str, src_ou: str) -> str:
    if src_h2h == "pinnacle" and src_ou == "pinnacle":
        return "pinnacle"
    if src_h2h == "pinnacle" or src_ou == "pinnacle":
        return "mixed"
    return "consensus"


def _parse_game(game: dict) -> dict:
    """Flatten one game into a wide row dict, resolving each market independently."""
    h_home, h_draw, h_away, src_h2h = _resolve_h2h(game)
    ou_line, ou_over, ou_under, src_ou = _resolve_totals(game)
    ou2_line, ou2_over, ou2_under = _resolve_secondary_totals(game)
    if not math.isnan(ou2_line) and ou2_line == ou_line:
        logger.debug("Game %s secondary totals line duplicates the primary (%s)", game["id"], ou_line)
    odds_source = _combine_source(src_h2h, src_ou)
    if odds_source != "pinnacle":
        logger.warning("Game %s using non-Pinnacle odds (%s)", game["id"], odds_source)
    return {
        "game_id": game["id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "commence_time": game["commence_time"],
        "pinnacle_h2h_home": h_home,
        "pinnacle_h2h_draw": h_draw,
        "pinnacle_h2h_away": h_away,
        "pinnacle_ou_line": ou_line,
        "pinnacle_ou_over": ou_over,
        "pinnacle_ou_under": ou_under,
        "pinnacle_ou2_line": ou2_line,
        "pinnacle_ou2_over": ou2_over,
        "pinnacle_ou2_under": ou2_under,
        "odds_source": odds_source,
    }


def _build_games_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble parsed game rows into a typed wide DataFrame."""
    df = pd.DataFrame(rows, columns=_GAME_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    for col in _FLOAT_COLUMNS:
        df[col] = df[col].astype(float)
    return df


def load_odds(source: Path) -> pd.DataFrame:
    """Load previously saved odds, or an empty typed frame if the file is absent.

    Args:
        source: Path to a CSV written by save_odds.

    Returns:
        A wide DataFrame in _GAME_COLUMNS shape with commence_time as a UTC
        datetime and float odds columns. Missing files yield an empty frame so
        callers can merge unconditionally.
    """
    source = Path(source)
    if not source.exists():
        return _build_games_frame([])
    df = pd.read_csv(source)
    df = df.reindex(columns=_GAME_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    for col in _FLOAT_COLUMNS:
        df[col] = df[col].astype(float)
    return df


def merge_odds(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert freshly fetched odds onto previously stored odds, keyed by game_id.

    Games only in `existing` (already played, so dropped from the /odds feed) are
    retained; games in both take the `new` fetch's values, but any odds cell that
    is NaN in `new` falls back to the stored value so a late empty fetch never
    blanks out historical odds. Games only in `new` are added.

    Args:
        existing: Previously stored odds (may be empty).
        new: Odds from the latest fetch (may be empty).

    Returns:
        A wide DataFrame in _GAME_COLUMNS shape, sorted by ascending
        commence_time.
    """
    old_idx = existing.set_index("game_id")
    new_idx = new.set_index("game_id")
    merged = new_idx.combine_first(old_idx).reset_index()
    merged = merged.reindex(columns=_GAME_COLUMNS)
    merged["commence_time"] = pd.to_datetime(merged["commence_time"], utc=True)
    for col in _FLOAT_COLUMNS:
        merged[col] = merged[col].astype(float)
    return merged.sort_values("commence_time", ignore_index=True)


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


def main() -> None:
    """Fetch odds for the default competition and write them under data/raw/.

    Entry point for `python -m fifa_predictor.data.fetch_odds` (see the Makefile
    `odds` target). The competition can be overridden with the first CLI argument.
    """
    competition = sys.argv[1] if len(sys.argv) > 1 else "world_cup_2026"
    destination = Path("data/raw") / f"odds_{competition}.csv"
    existing = load_odds(destination)
    new = fetch_odds(competition)
    odds = merge_odds(existing, new)
    logger.info(
        "Merged %d fetched game(s) onto %d stored; %d total after merge",
        len(new),
        len(existing),
        len(odds),
    )
    save_odds(odds, destination)
    breakdown = odds["odds_source"].value_counts().to_dict() if not odds.empty else {}
    logger.info("Stored %d game(s); odds_source breakdown: %s", len(odds), breakdown)


if __name__ == "__main__":
    main()
