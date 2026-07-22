import numpy as np
import pytest

from gomonova.game.board import BOARD_SIZE, BLACK, WHITE, EMPTY, Board, rc_to_pos


class TestBoardBasics:
    def test_initial_state(self):
        b = Board()
        assert b.current == BLACK
        assert b.move_count() == 0
        assert b.is_full() is False
        assert b.legal_moves().size == 225

    def test_play_and_undo(self):
        b = Board()
        pos = rc_to_pos(7, 7)
        b.play(pos)
        assert b.cells[7, 7] == BLACK
        assert b.current == WHITE
        assert b.move_count() == 1

        undone = b.undo()
        assert undone == pos
        assert b.cells[7, 7] == EMPTY
        assert b.current == BLACK
        assert b.move_count() == 0

    def test_alternating_colors(self):
        b = Board()
        b.play(rc_to_pos(7, 7))
        assert b.current == WHITE
        b.play(rc_to_pos(7, 8))
        assert b.current == BLACK
        assert b.cells[7, 8] == WHITE

    def test_hash_consistency(self):
        b1 = Board()
        b2 = Board()
        moves = [rc_to_pos(7, 7), rc_to_pos(7, 8), rc_to_pos(8, 7)]
        for m in moves:
            b1.play(m)
            b2.play(m)
        assert b1.hash == b2.hash

        b1.undo()
        b2.undo()
        assert b1.hash == b2.hash

    def test_hash_differs_for_different_positions(self):
        b1 = Board()
        b2 = Board()
        b1.play(rc_to_pos(7, 7))
        b2.play(rc_to_pos(7, 8))
        assert b1.hash != b2.hash

    def test_copy(self):
        b = Board()
        b.play(rc_to_pos(7, 7))
        c = b.copy()
        c.play(rc_to_pos(0, 0))
        assert b.move_count() == 1
        assert c.move_count() == 2
        assert b.cells[0, 0] == EMPTY

    def test_is_empty(self):
        b = Board()
        pos = rc_to_pos(3, 4)
        assert b.is_empty(pos)
        b.play(pos)
        assert not b.is_empty(pos)

    def test_last_moves_for(self):
        b = Board()
        b.play(rc_to_pos(7, 7))   # BLACK
        b.play(rc_to_pos(0, 0))   # WHITE
        b.play(rc_to_pos(8, 8))   # BLACK
        b.play(rc_to_pos(1, 1))   # WHITE

        black_last2 = b.last_moves_for(BLACK, 2)
        assert black_last2 == [rc_to_pos(8, 8), rc_to_pos(7, 7)]

        white_last1 = b.last_moves_for(WHITE, 1)
        assert white_last1 == [rc_to_pos(1, 1)]

    def test_full_board(self):
        b = Board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                b.play(rc_to_pos(r, c))
        assert b.is_full()
        assert b.legal_moves().size == 0
