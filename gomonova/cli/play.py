"""Interactive CLI for human vs AI Gomoku (Renju rules)."""

from __future__ import annotations

import os
import sys

import click
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

from ..game.board import BLACK, WHITE, Board, pos_to_rc
from ..game.rules import check_winner_at, is_legal
from ..inference.player import InferencePlayer
from ..nn.network import GomoNovaNet
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config
from .render import console, pos_from_str, pos_to_str, render_board, render_top_k

HELP_TEXT = """
Commands:
  <coord>       Place a stone (e.g. H8, h8)
  undo          Undo last move pair (human + AI)
  resign        Resign the game
  hint          Show AI top-3 suggestions for your position
  newgame       Start a new game
  color <b|w>   Choose your color (restarts game)
  temperature   Set AI randomness (0=greedy)
  help          Show this help
  quit          Exit
"""


def _create_player(config_path: str, checkpoint_path: str | None, device: torch.device, temperature: float) -> InferencePlayer:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    network = GomoNovaNet(
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        policy_channels=model_cfg["policy_channels"],
        value_channels=model_cfg["value_channels"],
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, network, device)
        console.print(f"[green]Loaded checkpoint: {checkpoint_path}[/]")
    else:
        console.print("[yellow]No checkpoint found, using random weights.[/]")

    return InferencePlayer(network, device, temperature=temperature)


@click.command()
@click.option("--config", default="configs/inference.yaml", help="Config file path")
@click.option("--checkpoint", default=None, help="Model checkpoint path")
@click.option("--device", default="auto", help="Device: auto/cuda/cpu")
def main(config: str, checkpoint: str | None, device: str):
    """GomoNova - Play Gomoku against an AI (Renju rules)."""
    console.print(Panel("[bold]GomoNova[/] - Gomoku AI (Renju Rules)", style="cyan"))

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    console.print(f"Device: {dev}")

    player = _create_player(config, checkpoint, dev, temperature=0.0)
    human_color = BLACK
    board = Board()
    game_over = False

    console.print(f"You are [bold]{'● Black' if human_color == BLACK else '○ White'}[/]. Type 'help' for commands.\n")
    render_board(board)

    while True:
        try:
            if not game_over and board.current == human_color:
                user_input = console.input("\n[bold green]Your move[/] (or command): ").strip()
            elif not game_over:
                user_input = "__ai_move__"
            else:
                user_input = console.input("\n[bold]Game over.[/] New game? (y/n): ").strip()
                if user_input.lower() in ("y", "yes", ""):
                    board = Board()
                    game_over = False
                    console.print()
                    render_board(board)
                    continue
                else:
                    break

            if user_input == "__ai_move__":
                move, policy, value = player.get_move(board)
                r, c = pos_to_rc(move)
                board.play(move)
                render_board(board, last_move=move)
                top = player.top_k(board.copy(), k=3)
                render_top_k([(move, policy[move])] + [(p, v) for p, v in top if p != move][:2])
                winner = check_winner_at(board, move)
                if winner is not None:
                    console.print(f"\n[bold red]AI wins![/]")
                    game_over = True
                elif board.is_full():
                    console.print("\n[yellow]Draw![/]")
                    game_over = True
                continue

            cmd = user_input.lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                console.print(HELP_TEXT)
            elif cmd == "resign":
                console.print("[red]You resigned. AI wins.[/]")
                game_over = True
            elif cmd == "newgame":
                board = Board()
                game_over = False
                console.print()
                render_board(board)
            elif cmd == "undo":
                if board.move_count() >= 2:
                    board.undo()
                    board.undo()
                    console.print("[dim]Undone 2 moves.[/]")
                    render_board(board, last_move=board.last_move())
                else:
                    console.print("[yellow]Nothing to undo.[/]")
            elif cmd == "hint":
                top = player.top_k(board, k=3)
                render_top_k(top)
            elif cmd.startswith("color"):
                parts = cmd.split()
                if len(parts) > 1 and parts[1] in ("b", "w", "black", "white"):
                    human_color = BLACK if parts[1].startswith("b") else WHITE
                    board = Board()
                    game_over = False
                    console.print(f"You are now [bold]{'● Black' if human_color == BLACK else '○ White'}[/].")
                    render_board(board)
            elif cmd.startswith("temperature"):
                parts = cmd.split()
                if len(parts) > 1:
                    try:
                        t = float(parts[1])
                        player.temperature = t
                        console.print(f"AI temperature set to {t}")
                    except ValueError:
                        console.print("[red]Invalid temperature value.[/]")
            else:
                pos = pos_from_str(user_input)
                if pos is None:
                    console.print(f"[red]Invalid input: '{user_input}'. Try H8 or type 'help'.[/]")
                    continue
                if not board.is_empty(pos):
                    console.print("[red]Position occupied.[/]")
                    continue
                if not is_legal(board, pos):
                    console.print("[red]Forbidden move (Renju rule).[/]")
                    continue

                board.play(pos)
                render_board(board, last_move=pos)
                winner = check_winner_at(board, pos)
                if winner is not None:
                    console.print(f"\n[bold green]You win![/]")
                    game_over = True

        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

    console.print("[dim]Thanks for playing GomoNova![/]")


if __name__ == "__main__":
    main()
