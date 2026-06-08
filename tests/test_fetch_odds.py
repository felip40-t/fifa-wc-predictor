"""Tests for the fetch_odds data module.

All network access is stubbed; the suite never makes a live API call.
"""

import pandas as pd
import pytest

from fifa_predictor.data import fetch_odds


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
            raise requests_http_error(self)


def requests_http_error(response: "FakeResponse"):
    """Build an HTTPError mirroring requests' own, carrying the response."""
    import requests

    err = requests.HTTPError(f"{response.status_code} Error")
    err.response = response
    return err


def _bookmaker(key: str, home_team: str, away_team: str, home: float, draw: float, away: float) -> dict:
    """A single bookmaker entry with an h2h market in The Odds API shape."""
    return {
        "key": key,
        "title": key.title(),
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home},
                    {"name": away_team, "price": away},
                    {"name": "Draw", "price": draw},
                ],
            }
        ],
    }


def _event(event_id: str, commence_time: str, home_team: str, away_team: str) -> dict:
    """A single event in The Odds API /events shape (no odds)."""
    return {
        "id": event_id,
        "sport_key": "soccer_fifa_world_cup",
        "sport_title": "FIFA World Cup",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
    }


def _stub_get(monkeypatch, response: FakeResponse, captured: dict | None = None) -> None:
    """Patch fetch_odds.requests.get to return the given response."""

    def fake_get(url, params=None, timeout=None):
        if captured is not None:
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
        return response

    monkeypatch.setattr(fetch_odds.requests, "get", fake_get)


def test_missing_api_key_raises_before_request(monkeypatch) -> None:
    """With no ODDS_API_KEY set, fetch_odds fails fast without hitting the network."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("requests.get should not be called without an API key")

    monkeypatch.setattr(fetch_odds.requests, "get", fail_if_called)

    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        fetch_odds.fetch_odds("world_cup_2026")


def test_happy_path_parses_one_match_multiple_bookmakers(monkeypatch) -> None:
    """A single match with two bookmakers yields one tidy row per bookmaker."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        {
            "id": "match-001",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "Poland",
            "bookmakers": [
                _bookmaker("pinnacle", "Mexico", "Poland", 2.10, 3.40, 3.80),
                _bookmaker("bet365", "Mexico", "Poland", 2.05, 3.50, 3.90),
            ],
        }
    ]
    _stub_get(monkeypatch, FakeResponse(payload))

    df = fetch_odds.fetch_odds("world_cup_2026")

    assert list(df.columns) == [
        "match_id",
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]
    assert len(df) == 2
    assert set(df["bookmaker"]) == {"pinnacle", "bet365"}
    assert (df["match_id"] == "match-001").all()
    assert (df["home_team"] == "Mexico").all()
    assert (df["away_team"] == "Poland").all()

    assert pd.api.types.is_datetime64_any_dtype(df["commence_time"])
    assert str(df["commence_time"].dt.tz) == "UTC"

    for col in ("home_odds", "draw_odds", "away_odds"):
        assert pd.api.types.is_float_dtype(df[col])
        assert (df[col] > 1.0).all()

    pinnacle = df.set_index("bookmaker").loc["pinnacle"]
    assert pinnacle["home_odds"] == 2.10
    assert pinnacle["draw_odds"] == 3.40
    assert pinnacle["away_odds"] == 3.80


def test_request_targets_correct_sport_key_and_params(monkeypatch) -> None:
    """The competition maps to the soccer_fifa_world_cup sport key with eu/h2h params."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        {
            "id": "match-001",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "Poland",
            "bookmakers": [_bookmaker("pinnacle", "Mexico", "Poland", 2.10, 3.40, 3.80)],
        }
    ]
    captured: dict = {}
    _stub_get(monkeypatch, FakeResponse(payload), captured)

    fetch_odds.fetch_odds("world_cup_2026")

    assert "soccer_fifa_world_cup/odds" in captured["url"]
    assert captured["params"]["apiKey"] == "test-key"
    assert captured["params"]["regions"] == "eu"
    assert captured["params"]["markets"] == "h2h"


def test_empty_match_list_returns_empty_typed_frame(monkeypatch, caplog) -> None:
    """No matches → empty DataFrame with the canonical columns and a warning."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    _stub_get(monkeypatch, FakeResponse([]))

    with caplog.at_level("WARNING"):
        df = fetch_odds.fetch_odds("world_cup_2026")

    assert df.empty
    assert list(df.columns) == [
        "match_id",
        "commence_time",
        "home_team",
        "away_team",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]
    assert any("no match" in r.message.lower() for r in caplog.records)


def test_non_200_logs_and_raises(monkeypatch, caplog) -> None:
    """A non-200 response logs the status/body and raises requests.HTTPError."""
    import requests

    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    _stub_get(
        monkeypatch,
        FakeResponse({"message": "bad"}, status_code=401, text="Invalid API key"),
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(requests.HTTPError):
            fetch_odds.fetch_odds("world_cup_2026")

    logged = " ".join(r.message for r in caplog.records)
    assert "401" in logged
    assert "Invalid API key" in logged


def test_bookmaker_without_h2h_is_skipped(monkeypatch, caplog) -> None:
    """A bookmaker lacking the h2h market is skipped with a warning; others remain."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    broken = {
        "key": "brokenbook",
        "title": "BrokenBook",
        "markets": [{"key": "totals", "outcomes": []}],
    }
    payload = [
        {
            "id": "match-001",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "Poland",
            "bookmakers": [
                _bookmaker("pinnacle", "Mexico", "Poland", 2.10, 3.40, 3.80),
                broken,
            ],
        }
    ]
    _stub_get(monkeypatch, FakeResponse(payload))

    with caplog.at_level("WARNING"):
        df = fetch_odds.fetch_odds("world_cup_2026")

    assert list(df["bookmaker"]) == ["pinnacle"]
    assert any("brokenbook" in r.message.lower() for r in caplog.records)


def test_selects_match_with_earliest_commence_time(monkeypatch) -> None:
    """When several matches are returned, only the earliest-kickoff one is parsed."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        {
            "id": "later",
            "commence_time": "2026-06-12T16:00:00Z",
            "home_team": "Spain",
            "away_team": "Brazil",
            "bookmakers": [_bookmaker("pinnacle", "Spain", "Brazil", 2.0, 3.3, 3.6)],
        },
        {
            "id": "earliest",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "Poland",
            "bookmakers": [_bookmaker("pinnacle", "Mexico", "Poland", 2.1, 3.4, 3.8)],
        },
    ]
    _stub_get(monkeypatch, FakeResponse(payload))

    df = fetch_odds.fetch_odds("world_cup_2026")

    assert (df["match_id"] == "earliest").all()
    assert (df["home_team"] == "Mexico").all()


def test_save_odds_round_trip(monkeypatch, tmp_path) -> None:
    """save_odds writes a CSV (creating parent dirs) that reads back identically."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        {
            "id": "match-001",
            "commence_time": "2026-06-11T19:00:00Z",
            "home_team": "Mexico",
            "away_team": "Poland",
            "bookmakers": [
                _bookmaker("pinnacle", "Mexico", "Poland", 2.10, 3.40, 3.80),
                _bookmaker("bet365", "Mexico", "Poland", 2.05, 3.50, 3.90),
            ],
        }
    ]
    _stub_get(monkeypatch, FakeResponse(payload))
    df = fetch_odds.fetch_odds("world_cup_2026")

    destination = tmp_path / "raw" / "odds.csv"
    fetch_odds.save_odds(df, destination)
    assert destination.exists()

    reloaded = pd.read_csv(destination, parse_dates=["commence_time"])
    pd.testing.assert_frame_equal(reloaded, df)


def test_list_events_missing_api_key_raises_before_request(monkeypatch) -> None:
    """With no ODDS_API_KEY set, list_events fails fast without hitting the network."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("requests.get should not be called without an API key")

    monkeypatch.setattr(fetch_odds.requests, "get", fail_if_called)

    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        fetch_odds.list_events("world_cup_2026")


def test_list_events_parses_and_sorts_fixtures(monkeypatch) -> None:
    """Events become a typed fixture frame sorted by ascending commence_time."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    payload = [
        _event("later", "2026-06-12T16:00:00Z", "Spain", "Brazil"),
        _event("earliest", "2026-06-11T19:00:00Z", "Mexico", "Poland"),
        _event("middle", "2026-06-12T01:00:00Z", "France", "Japan"),
    ]
    _stub_get(monkeypatch, FakeResponse(payload))

    df = fetch_odds.list_events("world_cup_2026")

    assert list(df.columns) == ["match_id", "commence_time", "home_team", "away_team"]
    assert list(df["match_id"]) == ["earliest", "middle", "later"]
    assert pd.api.types.is_datetime64_any_dtype(df["commence_time"])
    assert str(df["commence_time"].dt.tz) == "UTC"
    first = df.iloc[0]
    assert first["home_team"] == "Mexico"
    assert first["away_team"] == "Poland"


def test_list_events_targets_events_endpoint_with_only_api_key(monkeypatch) -> None:
    """list_events hits /events for the right sport key, sending apiKey but no odds params."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    captured: dict = {}
    _stub_get(
        monkeypatch,
        FakeResponse([_event("e1", "2026-06-11T19:00:00Z", "Mexico", "Poland")]),
        captured,
    )

    fetch_odds.list_events("world_cup_2026")

    assert captured["url"].endswith("soccer_fifa_world_cup/events")
    assert captured["params"]["apiKey"] == "test-key"
    assert "regions" not in captured["params"]
    assert "markets" not in captured["params"]


def test_list_events_empty_returns_typed_frame(monkeypatch, caplog) -> None:
    """No scheduled events → empty DataFrame with the fixture columns and a warning."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    _stub_get(monkeypatch, FakeResponse([]))

    with caplog.at_level("WARNING"):
        df = fetch_odds.list_events("world_cup_2026")

    assert df.empty
    assert list(df.columns) == ["match_id", "commence_time", "home_team", "away_team"]
    assert any("no event" in r.message.lower() for r in caplog.records)
