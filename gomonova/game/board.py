from __future__ import annotations

import numpy as np

BOARD_SIZE = 15
NUM_CELLS = BOARD_SIZE * BOARD_SIZE

EMPTY, BLACK, WHITE = 0, 1, 2

_rng = np.random.RandomState(42)
_ZOBRIST = _rng.randint(0, 2**63, size=(2, BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
_ZOBRIST_TURN = _rng.randint(0, 2**63, dtype=np.int64)


def pos_to_rc(pos: int) -> tuple[int, int]:
    return divmod(pos, BOARD_SIZE)


def rc_to_pos(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


class Board:
    __slots__ = ("cells", "current", "history", "_hash")

    def __init__(self) -> None:
        self.cells = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current = BLACK
        self.history: list[int] = []
        self._hash: int = 0

    def play(self, pos: int) -> None:
        r, c = pos_to_rc(pos)
        self.cells[r, c] = self.current
        self._hash ^= int(_ZOBRIST[self.current - 1, r, c])
        self.history.append(pos)
        self._hash ^= int(_ZOBRIST_TURN)
        self.current = WHITE if self.current == BLACK else BLACK

    def undo(self) -> int:
        pos = self.history.pop()
        self._hash ^= int(_ZOBRIST_TURN)
        self.current = WHITE if self.current == BLACK else BLACK
        r, c = pos_to_rc(pos)
        self._hash ^= int(_ZOBRIST[self.current - 1, r, c])
        self.cells[r, c] = EMPTY
        return pos

    @property
    def hash(self) -> int:
        return self._hash

    def is_empty(self, pos: int) -> bool:
        r, c = pos_to_rc(pos)
        return self.cells[r, c] == EMPTY

    def at(self, r: int, c: int) -> int:
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return int(self.cells[r, c])
        return -1

    def legal_moves(self) -> np.ndarray:
        return np.flatnonzero(self.cells == EMPTY)

    def move_count(self) -> int:
        return len(self.history)

    def is_full(self) -> bool:
        return len(self.history) == NUM_CELLS

    def last_move(self) -> int | None:
        return self.history[-1] if self.history else None

    def last_moves_for(self, player: int, count: int = 2) -> list[int]:
        result = []
        for pos in reversed(self.history):
            r, c = pos_to_rc(pos)
            if self.cells[r, c] == player:
                result.append(pos)
                if len(result) >= count:
                    break
        return result

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.cells = self.cells.copy()
        b.current = self.current
        b.history = list(self.history)
        b._hash = self._hash
        return b

    def __repr__(self) -> str:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for r in range(BOARD_SIZE):
            rows.append(" ".join(symbols[int(self.cells[r, c])] for c in range(BOARD_SIZE)))
        return "\n".join(rows)
