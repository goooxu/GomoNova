"""Interactive CLI for human vs AI Gomoku (Renju rules) with arrow key controls.

Controls:
  Arrow keys / WASD  Move cursor
  Enter / Space      Place stone
  u                  Undo (human + AI move)
  h                  Hint (AI top-3)
  r                  Resign
  n                  New game
  q                  Quit
"""

from __future__ import annotations

import curses
import os
import time

import click
import numpy as np
import torch

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc
from ..game.rules import check_winner_at, is_forbidden, is_legal
from ..inference.player import InferencePlayer
from ..nn.network import GomoNovaNet
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config

COL_LABELS = "ABCDEFGHIJKLMNO"


def _create_player(config_path: str, checkpoint_path: str | None, device: torch.device) -> InferencePlayer:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    network = GomoNovaNet(
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        policy_channels=model_cfg["policy_channels"],
        value_channels=model_cfg["value_channels"],
    ).bfloat16().to(device)
    for m in network.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.float()

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, network, device)
    return InferencePlayer(network, device, temperature=0.0)


def _draw_board(win, board: Board, cursor_r: int, cursor_c: int, last_move: int | None,
                message: str, ai_top: list[tuple[int, float]] | None, human_color: int):
    """Draw the board with cursor using curses."""
    win.clear()
    h, w = win.getmaxyx()

    # Title
    title = " GomoNova - Arrow keys to move, Enter to place "
    win.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_BOLD)

    # Color indicator
    color_str = "You: ● Black" if human_color == BLACK else "You: ○ White"
    win.addstr(1, 2, color_str, curses.A_BOLD)

    # Board offset
    top = 3
    left = 4

    # Column headers
    header = "    " + " ".join(f"{c:>2}" for c in COL_LABELS)
    win.addstr(top, left, header, curses.color_pair(3))

    for r in range(BOARD_SIZE):
        row_y = top + 1 + r
        row_str = f"{r+1:2d} "
        win.addstr(row_y, left, row_str, curses.color_pair(3))

        for c in range(BOARD_SIZE):
            pos = r * BOARD_SIZE + c
            cell = board.cells[r, c]
            x = left + 4 + c * 3

            if pos == last_move:
                if cell == BLACK:
                    win.addstr(row_y, x, "●", curses.color_pair(1) | curses.A_BOLD | curses.A_UNDERLINE)
                else:
                    win.addstr(row_y, x, "○", curses.color_pair(2) | curses.A_BOLD | curses.A_UNDERLINE)
            elif cell == BLACK:
                win.addstr(row_y, x, "●", curses.color_pair(1) | curses.A_BOLD)
            elif cell == WHITE:
                win.addstr(row_y, x, "○", curses.color_pair(2) | curses.A_BOLD)
            elif r == cursor_r and c == cursor_c:
                win.addstr(row_y, x, "◆", curses.color_pair(4) | curses.A_BOLD | curses.A_BLINK)
            else:
                win.addstr(row_y, x, "·", curses.color_pair(5))

    # Row numbers on right
    for r in range(BOARD_SIZE):
        row_y = top + 1 + r
        win.addstr(row_y, left + 4 + BOARD_SIZE * 3, f" {r+1:2d}", curses.color_pair(3))

    # AI top-3 info
    info_y = top + BOARD_SIZE + 2
    if ai_top:
        parts = []
        for i, (pos, prob) in enumerate(ai_top[:3], 1):
            r, c = pos_to_rc(pos)
            parts.append(f"{i}.{COL_LABELS[c]}{r+1}({prob:.2f})")
        win.addstr(info_y, left, "AI top-3: " + "  ".join(parts), curses.color_pair(3))

    # Message
    if message:
        win.addstr(info_y + 1, left, message[:w - left - 1], curses.A_BOLD)

    # Controls help
    help_y = min(info_y + 3, h - 2)
    controls = " [↑↓←→/WASD] Move  [Enter] Place  [u]Undo  [h]Hint  [r]Resign  [n]New  [q]Quit "
    win.addstr(help_y, max(0, (w - len(controls)) // 2), controls, curses.A_DIM)

    win.refresh()


def _game_loop(stdscr, player: InferencePlayer, human_color: int):
    """Main curses game loop."""
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)    # Black stones
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # White stones
    curses.init_pair(3, curses.COLOR_CYAN, -1)     # Labels
    curses.init_pair(4, curses.COLOR_GREEN, -1)    # Cursor
    curses.init_pair(5, curses.COLOR_WHITE, -1)    # Empty dots

    board = Board()
    cursor_r, cursor_c = 7, 7
    last_move = None
    message = "Your turn. Use arrow keys to select position."
    ai_top = None
    game_over = False

    # If AI goes first (human is White)
    if human_color == WHITE:
        move, policy, value = player.get_move(board)
        board.play(move)
        last_move = move
        r, c = pos_to_rc(move)
        ai_top = [(move, float(policy[move]))]
        message = f"AI plays {COL_LABELS[c]}{r+1}. Your turn."

    while True:
        _draw_board(stdscr, board, cursor_r, cursor_c, last_move, message, ai_top, human_color)

        key = stdscr.getch()

        # Movement
        if key in (curses.KEY_UP, ord('w'), ord('W')):
            cursor_r = max(0, cursor_r - 1)
        elif key in (curses.KEY_DOWN, ord('s'), ord('S')):
            cursor_r = min(BOARD_SIZE - 1, cursor_r + 1)
        elif key in (curses.KEY_LEFT, ord('a'), ord('A')):
            cursor_c = max(0, cursor_c - 1)
        elif key in (curses.KEY_RIGHT, ord('d'), ord('D')):
            cursor_c = min(BOARD_SIZE - 1, cursor_c + 1)

        # Place stone
        elif key in (curses.KEY_ENTER, 10, 13, ord(' ')):
            if game_over:
                board = Board()
                game_over = False
                last_move = None
                ai_top = None
                cursor_r, cursor_c = 7, 7
                message = "New game. Your turn."
                if human_color == WHITE:
                    move, policy, value = player.get_move(board)
                    if board.current == BLACK and is_forbidden(board, move):
                        r, c = pos_to_rc(move)
                        message = f"AI plays {COL_LABELS[c]}{r+1} — forbidden! You win!"
                        game_over = True
                    else:
                        board.play(move)
                        last_move = move
                        r, c = pos_to_rc(move)
                        ai_top = [(move, float(policy[move]))]
                        message = f"AI plays {COL_LABELS[c]}{r+1}. Your turn."
                continue

            if board.current != human_color:
                message = "Not your turn!"
                continue

            pos = cursor_r * BOARD_SIZE + cursor_c
            if not board.is_empty(pos):
                message = "Position occupied! Choose another."
                continue
            if not is_legal(board, pos):
                message = "Forbidden move (Renju rule)!"
                continue

            board.play(pos)
            last_move = pos
            winner = check_winner_at(board, pos)
            if winner is not None:
                message = "★ You win! Press Enter for new game, q to quit."
                game_over = True
                continue
            if board.is_full():
                message = "Draw! Press Enter for new game, q to quit."
                game_over = True
                continue

            # AI move
            message = "AI thinking..."
            _draw_board(stdscr, board, cursor_r, cursor_c, last_move, message, ai_top, human_color)
            time.sleep(0.1)

            move, policy, value = player.get_move(board)
            r, c = pos_to_rc(move)

            if board.current == BLACK and is_forbidden(board, move):
                message = f"AI plays {COL_LABELS[c]}{r+1} — forbidden! You win!"
                game_over = True
                continue

            board.play(move)
            last_move = move
            ai_top = player.top_k(board, k=3)
            message = f"AI plays {COL_LABELS[c]}{r+1} (value={value:+.2f}). Your turn."

            winner = check_winner_at(board, move)
            if winner is not None:
                message = "AI wins! Press Enter for new game, q to quit."
                game_over = True
            elif board.is_full():
                message = "Draw! Press Enter for new game, q to quit."
                game_over = True

        # Undo
        elif key == ord('u') or key == ord('U'):
            if board.move_count() >= 2:
                board.undo()
                board.undo()
                last_move = board.last_move()
                message = "Undone. Your turn."
                ai_top = None
                game_over = False
            else:
                message = "Nothing to undo."

        # Hint
        elif key == ord('h') or key == ord('H'):
            if not game_over and board.current == human_color:
                ai_top = player.top_k(board, k=3)
                message = "Hint: AI suggests the highlighted positions."
            else:
                message = "No hint available."

        # Resign
        elif key == ord('r') or key == ord('R'):
            message = "You resigned. AI wins. Press Enter for new game, q to quit."
            game_over = True

        # New game
        elif key == ord('n') or key == ord('N'):
            board = Board()
            game_over = False
            last_move = None
            ai_top = None
            cursor_r, cursor_c = 7, 7
            message = "New game. Your turn."
            if human_color == WHITE:
                move, policy, value = player.get_move(board)
                board.play(move)
                last_move = move
                r, c = pos_to_rc(move)
                ai_top = [(move, float(policy[move]))]
                message = f"AI plays {COL_LABELS[c]}{r+1}. Your turn."

        # Quit
        elif key == ord('q') or key == ord('Q'):
            break


@click.command()
@click.option("--config", default="configs/inference.yaml", help="Config file path")
@click.option("--checkpoint", default=None, help="Model checkpoint path")
@click.option("--device", default="auto", help="Device: auto/cuda/cpu")
@click.option("--color", default="b", help="Your color: b(lack) or w(hite)")
def main(config: str, checkpoint: str | None, device: str, color: str):
    """GomoNova - Play Gomoku against an AI with arrow key controls."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    player = _create_player(config, checkpoint, dev)
    human_color = BLACK if color.lower().startswith("b") else WHITE

    print(f"GomoNova | Device: {dev} | You: {'● Black' if human_color == BLACK else '○ White'}")
    print("Starting interactive board...")
    time.sleep(0.5)

    curses.wrapper(lambda stdscr: _game_loop(stdscr, player, human_color))
    print("Thanks for playing GomoNova!")


if __name__ == "__main__":
    main()
