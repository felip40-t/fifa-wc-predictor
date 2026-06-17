"""Tests for the fetch_results data module (competition score fetching).

All network access is stubbed; the suite never makes a live API call. The HTTP
call is delegated to fetch_odds._request_json, so the stub patches
fetch_odds.requests.get.
"""

import pandas as pd
import pytest

from fifa_predictor.data import fetch_odds, fetch_results
from fifa_predictor.data.fetch_results import (
    _RESULT_COLUMNS,
    _parse_score_event,
)


class FakeResponse:
    """Minimal stand-in for a requests.Response used by the stubbed requests.get."""

    def __init__(self, json_data, status_code: int = 200, text: str = "") -> None:
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            err = requests.HTTPError(f"{self.status_code} Error")
            err.response = self
            raise err


def _score_event(
    event_id: str,
    commence_time: str,
    home_team: str,
    away_team: str,
    home_score=None,
    away_score=None,
    completed: bool = True,
) -> dict:
    """A single event in The Odds API /scores shape."""
    scores = None
    if home_score is not None and away_score is not None:
        scores = [
            {"name": home_team, "score": str(home_score)},
            {"name": away_team, "score": str(away_score)},
        ]
    return {
        "id": event_id,
        "sport_key": "soccer_fifa_world_cup",
        "commence_time": commence_time,
        "completed": completed,
        "home_team": home_team,
        "away_team": away_team,
        "scores": scores,
        "last_update": commence_time,
    }


def _stub_get(monkeypatch, response: FakeResponse, captured: dict | None = None) -> None:
    """Patch the requests.get used by the shared API helper (lives in fetch_odds)."""

    def fake_get(url, params=None, timeout=None):
        if captured is not None:
            captured["url"] = url
            captured["params"] = params
        return response

    monkeypatch.setattr(fetch_odds.requests, "get", fake_get)


# ---------------------------------------------------------------------------
# _parse_score_event
# ---------------------------------------------------------------------------


def test_parse_score_event_extracts_scores_by_team_name():
    event = _score_event("g1", "2026-06-11T19:00:00Z", "Mexico", "Poland", 2, 1)

    row = _parse_score_event(event)

    assert row["game_id"] == "g1"
    assert row["home_team"] == "Mexico"
    assert row["away_team"] == "Poland"
    assert row["home_score"] == 2
    assert row["away_score"] == 1


def test_parse_score_event_skips_uncompleted_game():
    event = _score_event("g1", "2026-06-11T19:00:00Z", "A", "B", completed=False)

    assert _parse_score_event(event) is None


def test_parse_score_event_skips_completed_without_scores():
    event = _score_event("g1", "2026-06-11T19:00:00Z", "A", "B", completed=True)

    assert _parse_score_event(event) is None


# ---------------------------------------------------------------------------
# fetch_scores
# ---------------------------------------------------------------------------


def test_fetch_scores_missing_api_key_raises_before_request(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("requests.get should not be called without an API key")

    monkeypatch.setattr(fetch_odds.requests, "get", fail_if_called)

    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        fetch_results.fetch_scores("world_cup_2026")


def test_fetch_scores_returns_completed_games_sorted(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        _score_event("later", "2026-06-12T16:00:00Z", "Spain", "Brazil", 0, 0),
        _score_event("earliest", "2026-06-11T19:00:00Z", "Mexico", "Poland", 2, 1),
        _score_event("pending", "2026-06-13T19:00:00Z", "France", "Japan", completed=False),
    ]
    _stub_get(monkeypatch, FakeResponse(payload))

    df = fetch_results.fetch_scores("world_cup_2026")

    assert list(df.columns) == _RESULT_COLUMNS
    assert list(df["game_id"]) == ["earliest", "later"]  # pending dropped, sorted
    assert str(df["commence_time"].dt.tz) == "UTC"
    assert df.iloc[0]["home_score"] == 2
    assert df.iloc[0]["away_score"] == 1


def test_fetch_scores_targets_scores_endpoint_with_days_from(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    captured: dict = {}
    _stub_get(monkeypatch, FakeResponse([]), captured)

    fetch_results.fetch_scores("world_cup_2026")

    assert captured["url"].endswith("soccer_fifa_world_cup/scores")
    assert captured["params"]["apiKey"] == "test-key"
    assert captured["params"]["daysFrom"] == 3


def test_fetch_scores_empty_returns_typed_frame(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    _stub_get(monkeypatch, FakeResponse([]))

    df = fetch_results.fetch_scores("world_cup_2026")

    assert df.empty
    assert list(df.columns) == _RESULT_COLUMNS


# ---------------------------------------------------------------------------
# load_results / merge_results / save_results
# ---------------------------------------------------------------------------


def _result_row(game_id: str, commence_time: str, home_score: int = 1, away_score: int = 0) -> dict:
    return {
        "game_id": game_id,
        "commence_time": commence_time,
        "home_team": "Home",
        "away_team": "Away",
        "home_score": home_score,
        "away_score": away_score,
    }


def _frame(*rows: dict) -> pd.DataFrame:
    return fetch_results._build_results_frame(list(rows))


def test_merge_results_accumulates_new_games():
    existing = _frame(_result_row("g1", "2026-06-11T19:00:00Z"))
    new = _frame(_result_row("g2", "2026-06-12T19:00:00Z"))

    merged = fetch_results.merge_results(existing, new)

    assert set(merged["game_id"]) == {"g1", "g2"}


def test_merge_results_retains_old_games_absent_from_new_fetch():
    """The 3-day window means old games drop from new fetches; they must persist."""
    existing = _frame(
        _result_row("old", "2026-06-01T19:00:00Z"),
        _result_row("recent", "2026-06-12T19:00:00Z"),
    )
    new = _frame(_result_row("recent", "2026-06-12T19:00:00Z"))

    merged = fetch_results.merge_results(existing, new)

    assert set(merged["game_id"]) == {"old", "recent"}


def test_merge_results_sorts_by_commence_time():
    existing = _frame(_result_row("late", "2026-06-20T19:00:00Z"))
    new = _frame(_result_row("early", "2026-06-11T19:00:00Z"))

    merged = fetch_results.merge_results(existing, new)

    assert list(merged["game_id"]) == ["early", "late"]
    assert list(merged.columns) == _RESULT_COLUMNS


def test_load_results_missing_file_returns_empty_typed_frame(tmp_path):
    df = fetch_results.load_results(tmp_path / "nope.csv")

    assert df.empty
    assert list(df.columns) == _RESULT_COLUMNS


def test_load_results_round_trips_saved_results(tmp_path):
    path = tmp_path / "results.csv"
    fetch_results.save_results(_frame(_result_row("g", "2026-06-14T19:00:00Z", 3, 2)), path)

    df = fetch_results.load_results(path)

    assert list(df["game_id"]) == ["g"]
    assert str(df["commence_time"].dt.tz) == "UTC"
    assert int(df.iloc[0]["home_score"]) == 3
    assert int(df.iloc[0]["away_score"]) == 2
