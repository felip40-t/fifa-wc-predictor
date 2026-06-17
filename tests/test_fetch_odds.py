"""Tests for the fetch_odds data module.

All network access is stubbed; the suite never makes a live API call.
"""

import math

import pandas as pd
import pytest

from fifa_predictor.data import fetch_odds
from fifa_predictor.data.fetch_odds import (
    _GAME_COLUMNS,
    _build_games_frame,
    _parse_game,
    _resolve_secondary_totals,
    _select_pinnacle_secondary,
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
    "pinnacle_ou_line", "pinnacle_ou_over", "pinnacle_ou_under",
    "pinnacle_ou2_line", "pinnacle_ou2_over", "pinnacle_ou2_under",
    "odds_source",
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
    # Power method: home (favorite) gained prob vs proportional (2.1703 -> 2.1504).
    assert row["pinnacle_h2h_home"] == pytest.approx(2.150444, abs=1e-5)  # power-method de-vig
    assert row["pinnacle_h2h_draw"] == pytest.approx(3.535758, abs=1e-5)  # power-method de-vig
    assert row["pinnacle_h2h_away"] == pytest.approx(3.965817, abs=1e-5)  # power-method de-vig
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
    # Power method: each book de-vigged independently; home (favorite) gained prob.
    assert row["pinnacle_h2h_home"] == pytest.approx(2.193274, abs=1e-5)  # power-method de-vig
    assert row["pinnacle_h2h_draw"] == pytest.approx(3.322353, abs=1e-5)  # power-method de-vig
    assert row["pinnacle_h2h_away"] == pytest.approx(4.114055, abs=1e-5)  # power-method de-vig
    assert row["pinnacle_ou_line"] == 2.5
    assert row["pinnacle_ou_over"] == pytest.approx(2.058763, abs=1e-5)   # power-method de-vig
    assert row["pinnacle_ou_under"] == pytest.approx(1.944499, abs=1e-5)  # power-method de-vig
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
    assert row["pinnacle_h2h_home"] == pytest.approx(2.150444, abs=1e-5)  # power-method de-vig (was 2.170279 proportional)
    assert row["pinnacle_ou_line"] == 2.5     # from consensus
    assert row["pinnacle_ou_over"] == 2.0     # de-vig of even 1.90/1.90 -> fair 2.0/2.0
    assert row["pinnacle_ou_under"] == 2.0
    assert row["odds_source"] == "mixed"


def test_fetch_odds_totals_absent_everywhere_yields_nan(monkeypatch, caplog) -> None:
    """No book offers totals -> ou_* are NaN, the row is kept, and a warning logs."""
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    game = _game("g1", "2026-06-11T19:00:00Z", "A", "B", [
        _book("pinnacle", [_h2h_market("A", "B", 2.10, 3.40, 3.80)]),
    ])
    _stub_get(monkeypatch, FakeResponse([game]))
    with caplog.at_level("WARNING"):
        df = fetch_odds.fetch_odds("world_cup_2026")
    row = df.iloc[0]
    assert row["pinnacle_h2h_home"] == pytest.approx(2.150444, abs=1e-5)  # power-method de-vig (was 2.170279 proportional)
    assert math.isnan(row["pinnacle_ou_line"])
    assert math.isnan(row["pinnacle_ou_over"])
    assert len(df) == 1  # row retained
    assert any("totals" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Secondary totals (ou2) tests
# ---------------------------------------------------------------------------


def _pinnacle_game(totals_outcomes):
    """A minimal /odds game dict with only Pinnacle, h2h + the given totals."""
    return {
        "id": "g1",
        "commence_time": "2026-06-11T19:00:00Z",
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home", "price": 2.0},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Away", "price": 4.0},
                        ],
                    },
                    {"key": "totals", "outcomes": totals_outcomes},
                ],
            }
        ],
    }


def _totals(*lines):
    """Build totals outcomes from (point, over, under) tuples."""
    out = []
    for point, over, under in lines:
        out.append({"name": "Over", "price": over, "point": point})
        out.append({"name": "Under", "price": under, "point": point})
    return out


def test_select_pinnacle_secondary_picks_most_balanced_quarter_line():
    market = {"key": "totals", "outcomes": _totals(
        (2.0, 1.55, 2.45),
        (2.25, 1.95, 1.90),
        (3.0, 3.10, 1.37),
    )}
    point, over, under = _select_pinnacle_secondary(market)
    assert point == 2.25


def test_resolve_secondary_totals_returns_nan_without_pinnacle_totals():
    game = {
        "id": "g2", "commence_time": "2026-06-11T19:00:00Z",
        "home_team": "Home", "away_team": "Away",
        "bookmakers": [{"key": "betfair_ex_eu", "markets": []}],
    }
    point, over, under = _resolve_secondary_totals(game)
    assert math.isnan(point) and math.isnan(over) and math.isnan(under)


def test_parse_game_emits_secondary_trio_columns():
    game = _pinnacle_game(_totals((2.5, 1.95, 1.90), (2.25, 1.98, 1.86)))
    row = _parse_game(game)
    for col in ("pinnacle_ou2_line", "pinnacle_ou2_over", "pinnacle_ou2_under"):
        assert col in row
        assert col in _GAME_COLUMNS
    assert row["pinnacle_ou2_line"] == 2.25
    assert 1 / row["pinnacle_ou2_over"] + 1 / row["pinnacle_ou2_under"] == pytest.approx(1.0)


def test_parse_game_single_pinnacle_line_becomes_secondary():
    game = _pinnacle_game(_totals((2.5, 1.95, 1.90)))
    row = _parse_game(game)
    frame = _build_games_frame([row])
    assert "pinnacle_ou2_line" in frame.columns
    assert row["pinnacle_ou2_line"] == 2.5


def test_resolve_secondary_totals_nan_when_pinnacle_has_no_totals():
    game = {
        "id": "g3", "commence_time": "2026-06-11T19:00:00Z",
        "home_team": "Home", "away_team": "Away",
        "bookmakers": [{"key": "pinnacle", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Home", "price": 2.0},
                {"name": "Draw", "price": 3.4},
                {"name": "Away", "price": 4.0},
            ]},
        ]}],
    }
    point, over, under = _resolve_secondary_totals(game)
    assert math.isnan(point) and math.isnan(over) and math.isnan(under)


# ---------------------------------------------------------------------------
# load_odds / merge_odds (preserve historical odds across re-fetches)
# ---------------------------------------------------------------------------


def _row(game_id: str, commence_time: str, home: float = 2.0, **overrides) -> dict:
    """A game row dict in _GAME_COLUMNS shape with sensible numeric defaults."""
    row = {
        "game_id": game_id,
        "home_team": "Home",
        "away_team": "Away",
        "commence_time": commence_time,
        "pinnacle_h2h_home": home,
        "pinnacle_h2h_draw": 3.4,
        "pinnacle_h2h_away": 4.0,
        "pinnacle_ou_line": 2.5,
        "pinnacle_ou_over": 1.95,
        "pinnacle_ou_under": 1.95,
        "pinnacle_ou2_line": 2.25,
        "pinnacle_ou2_over": 1.98,
        "pinnacle_ou2_under": 1.86,
        "odds_source": "pinnacle",
    }
    row.update(overrides)
    return row


def _frame(*rows: dict) -> pd.DataFrame:
    """Build a typed games frame from row dicts."""
    return fetch_odds._build_games_frame(list(rows))


def test_merge_odds_keeps_completed_games_absent_from_new_fetch():
    """A game present only in the existing frame (already played) is retained."""
    existing = _frame(
        _row("played", "2026-06-11T19:00:00Z"),
        _row("upcoming", "2026-06-14T19:00:00Z"),
    )
    new = _frame(_row("upcoming", "2026-06-14T19:00:00Z"))

    merged = fetch_odds.merge_odds(existing, new)

    assert set(merged["game_id"]) == {"played", "upcoming"}


def test_merge_odds_refreshes_upcoming_game_with_new_odds():
    """A game in both frames takes the new fetch's odds."""
    existing = _frame(_row("g", "2026-06-14T19:00:00Z", home=2.0))
    new = _frame(_row("g", "2026-06-14T19:00:00Z", home=1.7))

    merged = fetch_odds.merge_odds(existing, new)

    assert len(merged) == 1
    assert merged.iloc[0]["pinnacle_h2h_home"] == pytest.approx(1.7)


def test_merge_odds_coalesces_nan_in_new_from_existing():
    """A NaN odds cell in the new row falls back to the existing value."""
    existing = _frame(_row("g", "2026-06-14T19:00:00Z", pinnacle_ou_line=2.5, pinnacle_ou_over=1.95))
    new = _frame(_row("g", "2026-06-14T19:00:00Z", pinnacle_ou_line=float("nan"), pinnacle_ou_over=float("nan")))

    merged = fetch_odds.merge_odds(existing, new)

    row = merged.iloc[0]
    assert row["pinnacle_ou_line"] == 2.5
    assert row["pinnacle_ou_over"] == pytest.approx(1.95)


def test_merge_odds_adds_brand_new_game():
    """A game only in the new fetch is added to the result."""
    existing = _frame(_row("old", "2026-06-11T19:00:00Z"))
    new = _frame(_row("fresh", "2026-06-20T19:00:00Z"))

    merged = fetch_odds.merge_odds(existing, new)

    assert set(merged["game_id"]) == {"old", "fresh"}


def test_merge_odds_empty_existing_returns_new():
    """With no prior odds, the merge is just the new frame."""
    new = _frame(_row("g", "2026-06-14T19:00:00Z"))

    merged = fetch_odds.merge_odds(_frame(), new)

    assert list(merged["game_id"]) == ["g"]


def test_merge_odds_sorts_by_commence_time_and_keeps_columns():
    """Result carries the canonical columns, sorted by ascending commence_time."""
    existing = _frame(_row("late", "2026-06-20T19:00:00Z"))
    new = _frame(_row("early", "2026-06-11T19:00:00Z"))

    merged = fetch_odds.merge_odds(existing, new)

    assert list(merged.columns) == _GAME_COLUMNS
    assert list(merged["game_id"]) == ["early", "late"]


def test_load_odds_missing_file_returns_empty_typed_frame(tmp_path):
    """Loading a path that does not exist yields an empty frame with the columns."""
    df = fetch_odds.load_odds(tmp_path / "nope.csv")

    assert df.empty
    assert list(df.columns) == _GAME_COLUMNS


def test_load_odds_round_trips_saved_odds(tmp_path):
    """A saved frame loads back with UTC commence_time and float odds."""
    path = tmp_path / "odds.csv"
    fetch_odds.save_odds(_frame(_row("g", "2026-06-14T19:00:00Z")), path)

    df = fetch_odds.load_odds(path)

    assert list(df["game_id"]) == ["g"]
    assert str(df["commence_time"].dt.tz) == "UTC"
    assert df["pinnacle_h2h_home"].dtype == float
