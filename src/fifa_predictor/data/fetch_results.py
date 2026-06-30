"""Fetches international match results.

Two distinct concerns live here:

* ``fetch_results`` / ``save_results`` (year-range) are the historical-results
  path used to train the model. These remain stubs.
* ``fetch_scores`` and friends pull *completed* games for a single competition
  from The Odds API ``/scores`` endpoint, keyed by the same event id that
  ``fetch_odds`` stores as ``game_id``. Because that endpoint only reaches back
  ~3 days, results are accumulated into a stored CSV (the same load/merge/save
  pattern ``fetch_odds`` uses) by fetching regularly.
"""

import sys
from pathlib import Path

import pandas as pd

from fifa_predictor.data.fetch_odds import _request_json
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

# How many days back the /scores endpoint should report completed games for.
# The Odds API caps this at 3; older results survive via the accumulated CSV.
_DAYS_FROM = 3

_RESULT_COLUMNS = [
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "winner",
]


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


def fetch_scores(competition: str) -> pd.DataFrame:
    """Retrieve completed-game scores for a competition from The Odds API.

    Calls the ``/scores`` endpoint (which reports games completed within the last
    few days) and keeps only completed games that carry a usable score. Each row
    is keyed by ``game_id`` (the Odds API event id), so results join directly to
    the odds and simulated-outcomes frames.

    Args:
        competition: Project competition identifier (e.g. "world_cup_2026").

    Returns:
        A DataFrame in ``_RESULT_COLUMNS`` shape, one row per completed game,
        sorted by ascending commence_time. Empty (but typed) when nothing has
        finished.
    """
    logger.info("Requesting scores for %s", competition)
    events = _request_json(competition, "scores", {"daysFrom": _DAYS_FROM})
    logger.info("API returned %d event(s)", len(events))

    rows = [row for row in (_parse_score_event(e) for e in events) if row is not None]
    logger.info("%d completed game(s) with scores", len(rows))
    return _build_results_frame(rows)


def _parse_score_event(event: dict) -> dict | None:
    """Flatten one /scores event into a result row, or None if not usable.

    Returns None when the game is not yet completed or when no score is posted
    for one of the two teams.
    """
    if not event.get("completed"):
        return None
    scores = event.get("scores")
    if not scores:
        return None

    by_name = {s["name"]: s["score"] for s in scores}
    home_team = event["home_team"]
    away_team = event["away_team"]
    try:
        home_score = int(by_name[home_team])
        away_score = int(by_name[away_team])
    except (KeyError, TypeError, ValueError):
        logger.warning("Game %s completed but scores unparseable: %s", event.get("id"), scores)
        return None

    return {
        "game_id": event["id"],
        "commence_time": event["commence_time"],
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }


def _resolve_winner(row: dict) -> object:
    """Winner for one result row: keep an entered value, else derive from the score.

    A decisive score names its own winner; a level score (the only outcome that
    goes to extra time or penalties) leaves the winner blank to be recorded by
    hand, and that hand-entered value is preserved here on later rebuilds.
    """
    existing = row.get("winner")
    if existing is not None and not pd.isna(existing) and str(existing).strip():
        return existing
    hs, as_ = row.get("home_score"), row.get("away_score")
    if pd.isna(hs) or pd.isna(as_) or hs == as_:
        return pd.NA
    return row["home_team"] if hs > as_ else row["away_team"]


def _build_results_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble result rows into a typed frame sorted by ascending commence_time."""
    df = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    for col in ("home_score", "away_score"):
        df[col] = df[col].astype("Int64")
    df["winner"] = [_resolve_winner(r) for r in df.to_dict("records")]
    df["winner"] = df["winner"].astype("object")
    return df.sort_values("commence_time", ignore_index=True)


def load_results(source: Path) -> pd.DataFrame:
    """Load previously saved results, or an empty typed frame if absent.

    Args:
        source: Path to a CSV written by save_results.

    Returns:
        A DataFrame in ``_RESULT_COLUMNS`` shape with a UTC commence_time and
        nullable-integer scores. Missing files yield an empty frame so callers
        can merge unconditionally.
    """
    source = Path(source)
    if not source.exists():
        return _build_results_frame([])
    df = pd.read_csv(source)
    df = df.reindex(columns=_RESULT_COLUMNS)
    return _build_results_frame(df.to_dict("records"))


def merge_results(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert freshly fetched results onto stored ones, keyed by game_id.

    Games only in ``existing`` (older than the /scores window) are retained;
    games in both take the new fetch's values; games only in ``new`` are added.

    Args:
        existing: Previously stored results (may be empty).
        new: Results from the latest fetch (may be empty).

    Returns:
        A DataFrame in ``_RESULT_COLUMNS`` shape, sorted by ascending
        commence_time.
    """
    old_idx = existing.set_index("game_id")
    new_idx = new.set_index("game_id")
    merged = new_idx.combine_first(old_idx).reset_index()
    merged = merged.reindex(columns=_RESULT_COLUMNS)
    return _build_results_frame(merged.to_dict("records"))


def save_results(results: pd.DataFrame, destination: Path) -> None:
    """Persist fetched match results to disk.

    Args:
        results: DataFrame of match results to save.
        destination: Path to the output CSV file. Parent directories are created
            if they do not already exist.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
    logger.info("Wrote %d row(s) to %s", len(results), destination)


def main() -> None:
    """Fetch completed-game scores and accumulate them under data/raw/.

    Entry point for ``python -m fifa_predictor.data.fetch_results`` (see the
    Makefile ``results`` target). The competition can be overridden with the
    first CLI argument.
    """
    competition = sys.argv[1] if len(sys.argv) > 1 else "world_cup_2026"
    destination = Path("data/raw") / f"results_{competition}.csv"
    existing = load_results(destination)
    new = fetch_scores(competition)
    results = merge_results(existing, new)
    logger.info(
        "Merged %d fetched game(s) onto %d stored; %d total after merge",
        len(new),
        len(existing),
        len(results),
    )
    save_results(results, destination)


if __name__ == "__main__":
    main()
