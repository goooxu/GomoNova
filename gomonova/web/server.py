"""Web server for playing against the trained GomoNova model.

Run:
    python -m gomonova.web.server --checkpoint checkpoints/best.pt --port 8000

The frontend is a single self-contained HTML page served at "/".
Game logic (Renju rules, forbidden-move adjudication) runs server-side;
the AI move is the raw model output with no extra intervention.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc, rc_to_pos
from ..game.rules import check_winner_at, is_forbidden
from ..inference.player import InferencePlayer
from ..nn.network import GomoNovaNet
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

app = FastAPI(title="GomoNova Web")

# ---- globals filled at startup ----
_player: InferencePlayer | None = None
_device: torch.device | None = None


def _winning_line(board: Board, pos: int) -> list[int]:
    """Return the 5+ positions forming the winning run through *pos*."""
    r, c = pos_to_rc(pos)
    player = int(board.cells[r, c])
    for dr, dc in _DIRS:
        line = [(r, c)]
        cr, cc = r + dr, c + dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board.cells[cr, cc] == player:
            line.append((cr, cc))
            cr += dr
            cc += dc
        cr, cc = r - dr, c - dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board.cells[cr, cc] == player:
            line.append((cr, cc))
            cr -= dr
            cc -= dc
        if len(line) >= 5:
            return sorted(rc_to_pos(rr, cc) for rr, cc in line)
    return []


class PlayRequest(BaseModel):
    moves: list[int]
    human_move: int | None = None
    human_color: int = BLACK


class HintRequest(BaseModel):
    moves: list[int]


def _rebuild(moves: list[int]) -> Board:
    board = Board()
    for m in moves:
        board.play(m)
    return board


def _ai_turn(board: Board, human_color: int) -> dict:
    """Let the model play one move for the AI.  Returns result payload."""
    ai_color = WHITE if human_color == BLACK else BLACK
    move, policy, value = _player.get_move(board)
    r, c = pos_to_rc(move)

    top_k = [
        [int(i), float(policy[i])]
        for i in np.argsort(policy)[::-1][:3]
        if policy[i] > 0
    ]

    # Raw model output — if the model picks a forbidden spot, it loses.
    if ai_color == BLACK and is_forbidden(board, move):
        return {
            "moves": board.history,
            "ai_move": move,
            "status": "ai_forbidden",
            "forbidden_pos": move,
            "black_value": None,
            "top_k": top_k,
            "win_line": [],
        }

    board.play(move)
    black_value = value if ai_color == BLACK else -value

    winner = check_winner_at(board, move)
    if winner is not None:
        return {
            "moves": board.history,
            "ai_move": move,
            "status": "ai_win",
            "forbidden_pos": None,
            "black_value": float(black_value),
            "top_k": top_k,
            "win_line": _winning_line(board, move),
        }
    if board.is_full():
        return {
            "moves": board.history,
            "ai_move": move,
            "status": "draw",
            "forbidden_pos": None,
            "black_value": float(black_value),
            "top_k": top_k,
            "win_line": [],
        }
    return {
        "moves": board.history,
        "ai_move": move,
        "status": "playing",
        "forbidden_pos": None,
        "black_value": float(black_value),
        "top_k": top_k,
        "win_line": [],
    }


@app.post("/api/play")
def play(req: PlayRequest) -> dict:
    board = _rebuild(req.moves)
    human_color = req.human_color

    if req.human_move is not None:
        if board.current != human_color:
            raise HTTPException(400, "Not the human player's turn")
        pos = req.human_move
        if not (0 <= pos < BOARD_SIZE * BOARD_SIZE) or not board.is_empty(pos):
            raise HTTPException(400, "Illegal position")

        # Human plays a forbidden spot -> immediate loss (no move placed).
        if human_color == BLACK and is_forbidden(board, pos):
            return {
                "moves": board.history,
                "ai_move": None,
                "status": "human_forbidden",
                "forbidden_pos": pos,
                "black_value": None,
                "top_k": [],
                "win_line": [],
            }

        board.play(pos)
        winner = check_winner_at(board, pos)
        if winner is not None:
            return {
                "moves": board.history,
                "ai_move": None,
                "status": "human_win",
                "forbidden_pos": None,
                "black_value": 1.0 if winner == BLACK else -1.0,
                "top_k": [],
                "win_line": _winning_line(board, pos),
            }
        if board.is_full():
            return {
                "moves": board.history,
                "ai_move": None,
                "status": "draw",
                "forbidden_pos": None,
                "black_value": 0.0,
                "top_k": [],
                "win_line": [],
            }

    return _ai_turn(board, human_color)


@app.post("/api/hint")
def hint(req: HintRequest) -> dict:
    board = _rebuild(req.moves)
    policy = _player.get_policy(board)
    top_k = [
        [int(i), float(policy[i])]
        for i in np.argsort(policy)[::-1][:3]
        if policy[i] > 0
    ]
    return {"top_k": top_k}


@app.get("/")
def index() -> FileResponse:
    html = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html)


def main() -> None:
    global _player, _device

    parser = argparse.ArgumentParser(description="GomoNova web server")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(args.device)

    cfg = load_config(args.config)
    m = cfg["model"]
    network = GomoNovaNet(
        channels=m["channels"],
        num_blocks=m["num_blocks"],
        policy_channels=m["policy_channels"],
        value_channels=m["value_channels"],
    ).bfloat16().to(_device)
    for mod in network.modules():
        if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            mod.float()
    load_checkpoint(args.checkpoint, network, _device)
    network.eval()

    _player = InferencePlayer(network, _device, temperature=0.0)
    print(f"Model loaded from {args.checkpoint} on {_device}")
    print(f"Open http://localhost:{args.port} in your browser")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
