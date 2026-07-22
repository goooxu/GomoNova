"""Rich-based terminal board rendering."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc

console = Console()

COL_LABELS = "ABCDEFGHIJKLMNO"


def render_board(board: Board, last_move: int | None = None) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("", width=2)
    for c in range(BOARD_SIZE):
        table.add_column(COL_LABELS[c], width=2, justify="center")

    for r in range(BOARD_SIZE):
        row = [f"{r+1:2d}"]
        for c in range(BOARD_SIZE):
            pos = r * BOARD_SIZE + c
            cell = board.cells[r, c]
            if cell == BLACK:
                sym = Text("●", style="bold white")
            elif cell == WHITE:
                sym = Text("○", style="bold yellow")
            else:
                sym = Text("·", style="dim")
            if pos == last_move:
                sym.stylize("bold red underline")
            row.append(sym)
        table.add_row(*row)

    console.print(table)


def render_move_info(move_str: str, confidence: float, value: float) -> None:
    console.print(f"  AI plays [bold green]{move_str}[/] (conf={confidence:.3f}, value={value:+.3f})")


def render_top_k(top: list[tuple[int, float]]) -> None:
    parts = []
    for i, (pos, prob) in enumerate(top, 1):
        r, c = pos_to_rc(pos)
        parts.append(f"{i}. {COL_LABELS[c]}{r+1} ({prob:.3f})")
    console.print(f"  Top-3: {' | '.join(parts)}")


def pos_from_str(s: str) -> int | None:
    s = s.strip().upper()
    if len(s) < 2:
        return None
    col_char = s[0]
    row_str = s[1:]
    if col_char not in COL_LABELS:
        return None
    try:
        row = int(row_str) - 1
    except ValueError:
        return None
    col = COL_LABELS.index(col_char)
    if not (0 <= row < BOARD_SIZE):
        return None
    return row * BOARD_SIZE + col


def pos_to_str(pos: int) -> str:
    r, c = pos_to_rc(pos)
    return f"{COL_LABELS[c]}{r+1}"
