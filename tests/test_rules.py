"""Tests for Renju rules: win detection, forbidden moves (overline, double-four, double-three)."""

import pytest

from gomonova.game.board import BLACK, WHITE, Board, rc_to_pos
from gomonova.game.rules import (
    check_winner_at,
    count_fours,
    count_open_threes,
    get_winner,
    has_overline,
    is_forbidden,
    is_legal,
    makes_five,
)


def _place(board: Board, positions: list[tuple[int, int]], start_black: bool = True) -> None:
    """Place stones alternating colors starting from the given positions."""
    for r, c in positions:
        board.play(rc_to_pos(r, c))


def _set_stones(board: Board, black: list[tuple[int, int]], white: list[tuple[int, int]] = None):
    """Directly set stones on the board without alternating (for test setups)."""
    for r, c in black:
        board.cells[r, c] = BLACK
    if white:
        for r, c in white:
            board.cells[r, c] = WHITE


class TestWinDetection:
    def test_horizontal_five_black(self):
        b = Board()
        _set_stones(b, [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)])
        assert makes_five(b, rc_to_pos(7, 5), BLACK)

    def test_vertical_five(self):
        b = Board()
        _set_stones(b, [(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)])
        assert makes_five(b, rc_to_pos(5, 7), BLACK)

    def test_diagonal_five(self):
        b = Board()
        _set_stones(b, [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)])
        assert makes_five(b, rc_to_pos(5, 5), BLACK)

    def test_anti_diagonal_five(self):
        b = Board()
        _set_stones(b, [(3, 11), (4, 10), (5, 9), (6, 8), (7, 7)])
        assert makes_five(b, rc_to_pos(5, 9), BLACK)

    def test_six_is_not_five_for_black(self):
        b = Board()
        _set_stones(b, [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)])
        assert not makes_five(b, rc_to_pos(7, 4), BLACK)
        assert has_overline(b, rc_to_pos(7, 4), BLACK)

    def test_white_wins_with_overline(self):
        b = Board()
        _set_stones(b, [], [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)])
        assert check_winner_at(b, rc_to_pos(7, 4)) == WHITE

    def test_get_winner_black(self):
        b = Board()
        moves = [(7, 3), (0, 0), (7, 4), (0, 1), (7, 5), (0, 2), (7, 6), (0, 3), (7, 7)]
        _place(b, moves)
        assert get_winner(b) == BLACK

    def test_no_winner(self):
        b = Board()
        _place(b, [(7, 7), (7, 8), (8, 7)])
        assert get_winner(b) is None


class TestOverlineForbidden:
    def test_overline_forbidden(self):
        b = Board()
        _set_stones(b, [(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert is_forbidden(b, pos)

    def test_exactly_five_not_forbidden(self):
        b = Board()
        _set_stones(b, [(7, 3), (7, 4), (7, 6), (7, 7)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert not is_forbidden(b, pos)

    def test_five_overrides_overline_in_other_dir(self):
        """If a move makes exactly 5 in one direction and overline in another, it's legal."""
        b = Board()
        _set_stones(b, [
            (7, 3), (7, 4), (7, 6), (7, 7),  # horizontal: placing (7,5) makes 5
            (5, 5), (6, 5), (8, 5), (9, 5), (10, 5),  # vertical: placing (7,5) makes 6
        ])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert not is_forbidden(b, pos)


class TestDoubleFourForbidden:
    def test_double_four(self):
        """Black forms two fours simultaneously → forbidden."""
        b = Board()
        _set_stones(b, [
            (7, 3), (7, 4), (7, 6),  # horizontal: X X . X → four at (7,5)
            (5, 5), (6, 5), (8, 5),  # vertical: X X . X → four at (7,5)
        ])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert count_fours(b, pos, BLACK) >= 2
        assert is_forbidden(b, pos)

    def test_single_four_not_forbidden(self):
        b = Board()
        _set_stones(b, [(7, 3), (7, 4), (7, 6)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert count_fours(b, pos, BLACK) == 1
        assert not is_forbidden(b, pos)

    def test_five_overrides_double_four(self):
        """If the move makes exactly 5, it's legal even with double four."""
        b = Board()
        _set_stones(b, [
            (7, 3), (7, 4), (7, 6), (7, 7),  # horizontal five at (7,5)
            (5, 5), (6, 5), (8, 5),          # vertical four at (7,5)
        ])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert makes_five(b, pos, BLACK)
        assert not is_forbidden(b, pos)


class TestDoubleThreeForbidden:
    def test_double_three(self):
        """Black forms two open threes simultaneously → forbidden."""
        b = Board()
        _set_stones(b, [
            (7, 4), (7, 6),  # horizontal: . X . X . → open three through (7,5)
            (6, 5), (8, 5),  # vertical: . X . X . → open three through (7,5)
        ])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        threes = count_open_threes(b, pos, BLACK)
        assert threes >= 2
        assert is_forbidden(b, pos)

    def test_single_three_not_forbidden(self):
        b = Board()
        _set_stones(b, [(7, 4), (7, 6)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert count_open_threes(b, pos, BLACK) == 1
        assert not is_forbidden(b, pos)

    def test_blocked_three_not_open(self):
        """A three blocked by opponent on one side is not an open three."""
        b = Board()
        _set_stones(b, [(7, 4), (7, 6)], [(7, 3)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert count_open_threes(b, pos, BLACK) == 0


class TestIsLegal:
    def test_occupied_cell_illegal(self):
        b = Board()
        b.play(rc_to_pos(7, 7))
        assert not is_legal(b, rc_to_pos(7, 7))

    def test_white_no_forbidden(self):
        """White can play anywhere empty, even if it would be forbidden for Black."""
        b = Board()
        _set_stones(b, [], [(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)])
        b.current = WHITE
        pos = rc_to_pos(7, 5)
        assert is_legal(b, pos)

    def test_black_forbidden_position(self):
        b = Board()
        _set_stones(b, [(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)])
        b.current = BLACK
        pos = rc_to_pos(7, 5)
        assert not is_legal(b, pos)


class TestEdgeCases:
    def test_corner_five(self):
        b = Board()
        _set_stones(b, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
        assert makes_five(b, rc_to_pos(0, 2), BLACK)

    def test_edge_five(self):
        b = Board()
        _set_stones(b, [(14, 10), (14, 11), (14, 12), (14, 13), (14, 14)])
        assert makes_five(b, rc_to_pos(14, 12), BLACK)

    def test_forbidden_at_edge(self):
        b = Board()
        _set_stones(b, [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5)])
        b.current = BLACK
        pos = rc_to_pos(0, 3)
        assert is_forbidden(b, pos)
