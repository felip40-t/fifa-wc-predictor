"""Tests for group standings and third-place ranking."""

from fifa_predictor.model.standings import (
    Game, build_group_tables, rank_third_placed,
)


def _g(h, a, hs, as_, grp):
    return Game(h, a, hs, as_, grp, "actual")


def test_group_order_uses_h2h_when_points_and_gd_tie():
    # A, B, C all finish 1-1-1 on points with identical GD/GF; the A>B>C head-to-head
    # results decide the order. D loses everything.
    games = [
        _g("A", "B", 2, 1, "X"),  # A beats B
        _g("B", "C", 2, 1, "X"),  # B beats C
        _g("C", "A", 2, 1, "X"),  # C beats A  -> A,B,C cyclic, all GD 0 overall after D games
        _g("A", "D", 3, 0, "X"),
        _g("B", "D", 3, 0, "X"),
        _g("C", "D", 3, 0, "X"),
    ]
    groups = {"X": ["A", "B", "C", "D"]}
    table = build_group_tables(games, groups)["X"]
    assert [s.team for s in table][3] == "D"          # D is clearly last
    assert table[0].rank == 1 and table[3].rank == 4   # ranks are filled 1..4
    assert {s.team for s in table[:3]} == {"A", "B", "C"}


def test_rank_third_placed_orders_by_points_then_gd():
    # Two trivial groups; the better third-placed team must come first.
    groups = {"X": ["A", "B", "C", "D"], "Y": ["E", "F", "G", "H"]}
    games = [
        # Group X: A wins all (9 pts); C edges B on goals for 2nd, so B is 3rd on 4 pts.
        _g("A", "B", 1, 0, "X"), _g("C", "D", 2, 1, "X"),
        _g("A", "C", 1, 0, "X"), _g("B", "D", 1, 0, "X"),
        _g("A", "D", 1, 0, "X"), _g("B", "C", 0, 0, "X"),
        # Group Y: E/F take 1st/2nd; G and H tie on 1 pt, G is 3rd by name fallback.
        _g("E", "F", 1, 0, "Y"), _g("G", "H", 0, 0, "Y"),
        _g("E", "G", 2, 0, "Y"), _g("F", "H", 2, 0, "Y"),
        _g("E", "H", 2, 0, "Y"), _g("F", "G", 2, 0, "Y"),
    ]
    tables = build_group_tables(games, groups)
    thirds = rank_third_placed(tables)
    assert len(thirds) == 2
    assert thirds[0].points >= thirds[1].points
