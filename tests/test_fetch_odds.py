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


def _h2h_market(home_team: str, away_team: str, home: float, draw: float, away: float) -> dict:
    """An h2h market in The Odds API shape."""
    return {
        "key": "h2h",
        "outcomes": [
            {"name": home_team, "price": home},
            {"name": "Draw", "price": draw},
            {"name": away_team, "price": away},
        ],
    }


def _totals_market(lines: list[tuple]) -> dict:
    """A totals market from (point, over, under) tuples; a None price is omitted."""
    outcomes = []
    for point, over, under in lines:
        if over is not None:
            outcomes.append({"name": "Over", "price": over, "point": point})
        if under is not None:
            outcomes.append({"name": "Under", "price": under, "point": point})
    return {"key": "totals", "outcomes": outcomes}


def _book(key: str, markets: list[dict]) -> dict:
    """A bookmaker entry holding the given markets."""
    return {"key": key, "title": key.title(), "markets": markets}


def _game(game_id: str, commence_time: str, home_team: str, away_team: str, bookmakers: list[dict]) -> dict:
    """A game/event in The Odds API /odds shape."""
    return {
        "id": game_id,
        "sport_key": "soccer_fifa_world_cup",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": bookmakers,
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


_GAME_COLS = [
    "game_id", "home_team", "away_team", "commence_time",
    "pinnacle_h2h_home", "pinnacle_h2h_draw", "pinnacle_h2h_away",
    "pinnacle_ou_line", "pinnacle_ou_over", "pinnacle_ou_under", "odds_source",
]


def test_fetch_odds_pinnacle_balance_and_grid(monkeypatch) -> None:
    """Pinnacle path: quarter line excluded, most-balanced grid line selected."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    # 2.75 is perfectly even but a quarter line -> excluded. Among grid lines,
    # 3.0 (1.95/1.95) is more balanced than 2.5 (1.50/2.60) -> pick 3.0.
    game = _game("g1", "2026-06-11T19:00:00Z", "Mexico", "South Africa", [
        _book("pinnacle", [
            _h2h_market("Mexico", "South Africa", 2.10, 3.40, 3.80),
            _totals_market([(2.5, 1.50, 2.60), (2.75, 1.90, 1.90), (3.0, 1.95, 1.95)]),
        ]),
    ])
    _stub_get(monkeypatch, FakeResponse([game]))
    df = fetch_odds.fetch_odds("world_cup_2026")
    assert list(df.columns) == _GAME_COLS
    row = df.iloc[0]
    assert row["game_id"] == "g1"
    # Pinnacle odds are de-vigged (fair) before storage.
    assert row["pinnacle_h2h_home"] == pytest.approx(2.170279, abs=1e-5)
    assert row["pinnacle_h2h_draw"] == pytest.approx(3.513784, abs=1e-5)
    assert row["pinnacle_h2h_away"] == pytest.approx(3.927171, abs=1e-5)
    assert row["pinnacle_ou_line"] == 3.0
    assert row["pinnacle_ou_over"] == 2.0   # de-vig of even 1.95/1.95
    assert row["pinnacle_ou_under"] == 2.0
    assert row["odds_source"] == "pinnacle"
    assert str(df["commence_time"].dt.tz) == "UTC"


def test_fetch_odds_consensus_when_pinnacle_absent(monkeypatch) -> None:
    """No Pinnacle -> median consensus across books for both markets."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    game = _game("g1", "2026-06-11T19:00:00Z", "A", "B", [
        _book("bet365", [_h2h_market("A", "B", 2.0, 3.0, 4.0), _totals_market([(2.5, 1.90, 1.90)])]),
        _book("williamhill", [_h2h_market("A", "B", 2.2, 3.2, 3.6), _totals_market([(2.5, 2.00, 1.80)])]),
    ])
    _stub_get(monkeypatch, FakeResponse([game]))
    row = fetch_odds.fetch_odds("world_cup_2026").iloc[0]
    # Principled consensus: de-vig each book, median the fair probabilities,
    # then convert back to (vig-free) decimal odds.
    assert row["pinnacle_h2h_home"] == pytest.approx(2.230689, abs=1e-5)
    assert row["pinnacle_h2h_draw"] == pytest.approx(3.296055, abs=1e-5)
    assert row["pinnacle_h2h_away"] == pytest.approx(4.027141, abs=1e-5)
    assert row["pinnacle_ou_line"] == 2.5
    assert row["pinnacle_ou_over"] == pytest.approx(2.054054, abs=1e-5)
    assert row["pinnacle_ou_under"] == pytest.approx(1.948718, abs=1e-5)
    assert row["odds_source"] == "consensus"


def test_fetch_odds_empty_returns_typed_frame(monkeypatch) -> None:
    """No games -> empty DataFrame with the wide columns."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    _stub_get(monkeypatch, FakeResponse([]))
    df = fetch_odds.fetch_odds("world_cup_2026")
    assert df.empty
    assert list(df.columns) == _GAME_COLS


def test_fetch_odds_mixed_source_when_pinnacle_missing_totals(monkeypatch) -> None:
    """Pinnacle has h2h but no totals -> h2h from Pinnacle, totals from consensus."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    game = _game("g1", "2026-06-11T19:00:00Z", "A", "B", [
        _book("pinnacle", [_h2h_market("A", "B", 2.10, 3.40, 3.80)]),  # no totals
        _book("bet365", [_totals_market([(2.5, 1.90, 1.90)])]),        # totals only
    ])
    _stub_get(monkeypatch, FakeResponse([game]))
    row = fetch_odds.fetch_odds("world_cup_2026").iloc[0]
    assert row["pinnacle_h2h_home"] == pytest.approx(2.170279, abs=1e-5)  # Pinnacle, de-vigged
    assert row["pinnacle_ou_line"] == 2.5     # from consensus
    assert row["pinnacle_ou_over"] == 2.0     # de-vig of even 1.90/1.90 -> fair 2.0/2.0
    assert row["pinnacle_ou_under"] == 2.0
    assert row["odds_source"] == "mixed"


def test_fetch_odds_totals_absent_everywhere_yields_nan(monkeypatch, caplog) -> None:
    """No book offers totals -> ou_* are NaN, the row is kept, and a warning logs."""
    import math
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    game = _game("g1", "2026-06-11T19:00:00Z", "A", "B", [
        _book("pinnacle", [_h2h_market("A", "B", 2.10, 3.40, 3.80)]),
    ])
    _stub_get(monkeypatch, FakeResponse([game]))
    with caplog.at_level("WARNING"):
        df = fetch_odds.fetch_odds("world_cup_2026")
    row = df.iloc[0]
    assert row["pinnacle_h2h_home"] == pytest.approx(2.170279, abs=1e-5)  # de-vigged
    assert math.isnan(row["pinnacle_ou_line"])
    assert math.isnan(row["pinnacle_ou_over"])
    assert len(df) == 1  # row retained
    assert any("totals" in r.message.lower() for r in caplog.records)
