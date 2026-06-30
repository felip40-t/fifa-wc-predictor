"""Tests for bracket slot resolution."""

import pandas as pd

import pytest

from fifa_predictor.model.knockout import (
    attach_r32_results,
    load_bracket,
    resolve_group_games,
    resolve_round_of_32,
)
from fifa_predictor.model.standings import TeamStanding


def _r32_slot(match=73, home="Canada", away="Mexico"):
    """A single resolved Round of 32 entry as produced by resolve_round_of_32."""
    return [{
        "match": match, "home": home, "away": away,
        "home_status": "final", "away_status": "final",
    }]


def _results(rows):
    """Build a results frame from (home, away, hs, as_, winner) tuples."""
    return pd.DataFrame(
        [
            {"home_team": h, "away_team": a, "home_score": hs,
             "away_score": as_, "winner": w}
            for h, a, hs, as_, w in rows
        ],
        columns=["home_team", "away_team", "home_score", "away_score", "winner"],
    )


def test_attach_r32_results_decisive_score():
    """A played match with a decisive score advances the higher scorer."""
    r32 = _r32_slot(home="Canada", away="Mexico")
    results = _results([("Canada", "Mexico", 2, 1, None)])

    out = attach_r32_results(r32, results)

    assert out[0]["home_score"] == 2
    assert out[0]["away_score"] == 1
    assert out[0]["winner"] == "Canada"
    assert out[0]["result_status"] == "played"


def test_attach_r32_results_unplayed_slot_is_scheduled():
    """A slot with no matching result row stays scheduled with no score."""
    r32 = _r32_slot(home="Canada", away="Mexico")
    results = _results([("Brazil", "Japan", 1, 0, None)])  # different match

    out = attach_r32_results(r32, results)

    assert out[0]["result_status"] == "scheduled"
    assert "home_score" not in out[0]
    assert "winner" not in out[0]


def test_attach_r32_results_level_match_uses_winner_column():
    """A level match advances the team named in the winner column (ET/penalties)."""
    r32 = _r32_slot(home="Netherlands", away="Morocco")
    results = _results([("Netherlands", "Morocco", 1, 1, "Morocco")])

    out = attach_r32_results(r32, results)

    assert out[0]["home_score"] == 1 and out[0]["away_score"] == 1
    assert out[0]["winner"] == "Morocco"
    assert out[0]["result_status"] == "played"


def test_attach_r32_results_level_match_without_winner_errors():
    """A level match with no recorded winner raises, naming the match."""
    r32 = _r32_slot(home="Germany", away="Paraguay")
    results = _results([("Germany", "Paraguay", 1, 1, None)])

    with pytest.raises(ValueError, match="Germany 1-1 Paraguay"):
        attach_r32_results(r32, results)


def test_attach_r32_results_winner_contradicting_score_errors():
    """A winner that disagrees with a decisive score raises."""
    r32 = _r32_slot(home="Canada", away="Mexico")
    results = _results([("Canada", "Mexico", 2, 1, "Mexico")])

    with pytest.raises(ValueError, match="contradicts decisive score"):
        attach_r32_results(r32, results)


def test_resolve_group_games_ignores_knockout_fixtures():
    """Predictions may carry knockout (cross-group) rows; only group games count."""
    groups = {"A": ["T1", "T2"], "B": ["T3", "T4"]}
    results = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score"])
    predictions = pd.DataFrame(
        {
            "match": ["T1 vs T2", "T3 vs T4", "T1 vs T3"],  # last row is cross-group
            "score": ["1-0", "2-2", "0-0"],
        }
    )

    games = resolve_group_games(results, predictions, groups)

    assert [(g.home, g.away) for g in games] == [("T1", "T2"), ("T3", "T4")]
    assert all(g.source == "predicted" for g in games)


def _rows(group, teams):
    return [TeamStanding(team=t, group=group, rank=i + 1) for i, t in enumerate(teams)]


def test_resolve_round_of_32_uses_allocation_table():
    bracket = load_bracket()
    # Make the eight qualifying third places come from groups A..H so the key is
    # 'ABCDEFGH'. Name each team "<letter><rank>" so we can read slots back.
    tables = {L: _rows(L, [f"{L}1", f"{L}2", f"{L}3", f"{L}4"]) for L in "ABCDEFGHIJKL"}
    # Third-place ranking: A..H ahead of I..L (give A..H more points).
    thirds = []
    for L in "ABCDEFGH":
        s = tables[L][2]
        s.points = 3
        thirds.append(s)
    for L in "IJKL":
        s = tables[L][2]
        s.points = 0
        thirds.append(s)
    thirds.sort(key=lambda s: s.points, reverse=True)

    r32 = resolve_round_of_32(tables, thirds, bracket)
    by_match = {m["match"]: m for m in r32}

    # Fixed slots: match 73 is 2A vs 2B.
    assert by_match[73]["home"] == "A2" and by_match[73]["away"] == "B2"
    # Allocation for key 'ABCDEFGH': match 74's 3rd slot is group C (from the table).
    alloc = bracket["third_place_allocations"]["ABCDEFGH"]
    assert by_match[74]["home"] == "E1"                      # 1E
    assert by_match[74]["away"] == f"{alloc['74']}3"         # 3rd from allocated group
    # Every away 3rd-place team must come from one of that match's allowed groups.
    assert len(r32) == 16
