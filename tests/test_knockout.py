"""Tests for bracket slot resolution."""

from fifa_predictor.model.knockout import load_bracket, resolve_round_of_32
from fifa_predictor.model.standings import TeamStanding


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
