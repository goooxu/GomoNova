"""Renju rules: win detection and forbidden-move checking for Black.

Forbidden moves (Black only):
  - Overline: 6+ consecutive stones
  - Double-four: two or more fours created simultaneously
  - Double-three: two or more open threes created simultaneously

A move that makes exactly five in a row is ALWAYS legal, even if it
would otherwise trigger a forbidden pattern in another direction.
White has no restrictions; overline counts as a win for White.

Public helpers (`count_fours`, `count_open_threes`, `is_forbidden`) all
accept a board WITHOUT the candidate stone placed; they place it
temporarily and restore it before returning.
"""

from __future__ import annotations

from .board import BOARD_SIZE, BLACK, WHITE, Board, pos_to_rc

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _consecutive(board: Board, r: int, c: int, dr: int, dc: int, player: int) -> int:
    count = 0
    cr, cc = r + dr, c + dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board.cells[cr, cc] == player:
        count += 1
        cr += dr
        cc += dc
    return count


def _line_run(board: Board, r: int, c: int, dr: int, dc: int, player: int) -> int:
    return 1 + _consecutive(board, r, c, dr, dc, player) + _consecutive(board, r, c, -dr, -dc, player)


def makes_five(board: Board, pos: int, player: int) -> bool:
    """Exactly five in a row through *pos* (stone assumed already placed)."""
    r, c = pos_to_rc(pos)
    for dr, dc in DIRS:
        if _line_run(board, r, c, dr, dc, player) == 5:
            return True
    return False


def makes_five_or_more(board: Board, pos: int, player: int) -> bool:
    r, c = pos_to_rc(pos)
    for dr, dc in DIRS:
        if _line_run(board, r, c, dr, dc, player) >= 5:
            return True
    return False


def has_overline(board: Board, pos: int, player: int) -> bool:
    r, c = pos_to_rc(pos)
    for dr, dc in DIRS:
        if _line_run(board, r, c, dr, dc, player) >= 6:
            return True
    return False


def _four_threats_placed(board: Board, pos: int, player: int, dr: int, dc: int) -> set[int]:
    """With *pos* already occupied, find empty cells in direction (dr,dc)
    that complete a five through *pos*.

    Scans length-5 windows containing *pos*.  A window with 4 stones of
    *player*, 1 empty, and no opponent stones yields that empty cell.
    For Black, threats that would form an overline (6+) are discarded,
    since an overline is not a winning four under Renju.
    """
    r, c = pos_to_rc(pos)
    opponent = WHITE if player == BLACK else BLACK
    threats: set[int] = set()

    for start in range(-4, 1):
        window: list[tuple[int, int]] = []
        valid = True
        for i in range(5):
            cr, cc = r + (start + i) * dr, c + (start + i) * dc
            if not (0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
                valid = False
                break
            window.append((cr, cc))
        if not valid:
            continue

        p_count = 0
        o_count = 0
        empty_cell: tuple[int, int] | None = None
        for cr, cc in window:
            v = board.cells[cr, cc]
            if v == player:
                p_count += 1
            elif v == opponent:
                o_count += 1
            else:
                empty_cell = (cr, cc)

        if o_count == 0 and p_count == 4 and empty_cell is not None:
            er, ec = empty_cell
            if player == BLACK:
                board.cells[er, ec] = BLACK
                run = _line_run(board, er, ec, dr, dc, BLACK)
                board.cells[er, ec] = 0
                if run != 5:
                    continue
            threats.add(er * BOARD_SIZE + ec)

    return threats


def count_fours(board: Board, pos: int, player: int) -> int:
    """Number of directions in which placing at *pos* creates a four.

    A four (冲四/活四) is a pattern with a threat to make five.
    The board must NOT have a stone at *pos*.
    """
    r, c = pos_to_rc(pos)
    board.cells[r, c] = player
    count = 0
    for dr, dc in DIRS:
        if _four_threats_placed(board, pos, player, dr, dc):
            count += 1
    board.cells[r, c] = 0
    return count


def _is_open_three_dir_placed(board: Board, pos: int, player: int, dr: int, dc: int) -> bool:
    """With *pos* already occupied, is there an open three (活三) here?

    An open three can become an open four (>=2 distinct five-threats) with
    one more stone.  Try every nearby empty cell in this direction.
    """
    r, c = pos_to_rc(pos)

    for offset in range(-5, 6):
        er, ec = r + offset * dr, c + offset * dc
        if not (0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE):
            continue
        if board.cells[er, ec] != 0:
            continue

        board.cells[er, ec] = player
        threats = _four_threats_placed(board, er * BOARD_SIZE + ec, player, dr, dc)
        board.cells[er, ec] = 0

        if len(threats) >= 2:
            return True

    return False


def count_open_threes(board: Board, pos: int, player: int) -> int:
    """Number of directions in which placing at *pos* creates an open three.

    The board must NOT have a stone at *pos*.
    """
    r, c = pos_to_rc(pos)
    board.cells[r, c] = player
    count = 0
    for dr, dc in DIRS:
        if _is_open_three_dir_placed(board, pos, player, dr, dc):
            count += 1
    board.cells[r, c] = 0
    return count


def is_forbidden(board: Board, pos: int) -> bool:
    """Whether *pos* is a forbidden move for Black (Renju).

    The board must NOT have a stone at *pos*.
    """
    r, c = pos_to_rc(pos)
    if board.cells[r, c] != 0:
        return False

    board.cells[r, c] = BLACK

    if makes_five(board, pos, BLACK):
        board.cells[r, c] = 0
        return False

    forbidden = False
    if has_overline(board, pos, BLACK):
        forbidden = True
    if not forbidden and count_fours(board, pos, BLACK) >= 2:
        forbidden = True
    if not forbidden and count_open_threes(board, pos, BLACK) >= 2:
        forbidden = True

    board.cells[r, c] = 0
    return forbidden


def is_legal(board: Board, pos: int) -> bool:
    if not board.is_empty(pos):
        return False
    if board.current == WHITE:
        return True
    return not is_forbidden(board, pos)


def get_winner(board: Board) -> int | None:
    for pos in board.history:
        r, c = pos_to_rc(pos)
        player = int(board.cells[r, c])
        if player == 0:
            continue
        for dr, dc in DIRS:
            run = _line_run(board, r, c, dr, dc, player)
            if player == BLACK and run == 5:
                return BLACK
            if player == WHITE and run >= 5:
                return WHITE
    return None


def check_winner_at(board: Board, pos: int) -> int | None:
    r, c = pos_to_rc(pos)
    player = int(board.cells[r, c])
    if player == 0:
        return None
    for dr, dc in DIRS:
        run = _line_run(board, r, c, dr, dc, player)
        if player == BLACK and run == 5:
            return BLACK
        if player == WHITE and run >= 5:
            return WHITE
    return None
