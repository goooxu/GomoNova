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
_network: GomoNovaNet | None = None
_ckpt_path: str | None = None
_ckpt_mtime: float = 0.0


def _maybe_reload() -> None:
    """Reload weights if the checkpoint on disk is newer than what we hold.

    Training rewrites best.pt periodically; this lets a long-running server
    track the latest model without a process restart.  A partially written
    file fails torch.load before any weight is touched, so on failure we keep
    the current weights and retry on the next request.
    """
    global _player, _ckpt_mtime
    if _network is None or _ckpt_path is None:
        return
    try:
        mtime = os.path.getmtime(_ckpt_path)
    except OSError:
        return
    if mtime <= _ckpt_mtime:
        return
    try:
        load_checkpoint(_ckpt_path, _network, _device)
        _network.eval()
        _player = InferencePlayer(_network, _device, temperature=0.0)
        _ckpt_mtime = mtime
        print(f"[hot-reload] reloaded {_ckpt_path}", flush=True)
    except Exception as e:
        print(f"[hot-reload] skipped ({e}); keeping current weights", flush=True)


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


def _result(
    board: Board,
    status: str,
    ai_move: int | None = None,
    forbidden_pos: int | None = None,
    black_value: float | None = None,
    top_k: list | None = None,
    win_line: list | None = None,
) -> dict:
    return {
        "moves": board.history,
        "ai_move": ai_move,
        "status": status,
        "forbidden_pos": forbidden_pos,
        "black_value": black_value,
        "top_k": top_k or [],
        "win_line": win_line or [],
    }


def _ai_turn(board: Board, human_color: int) -> dict:
    """Let the model play one move for the AI.  Returns result payload."""
    ai_color = WHITE if human_color == BLACK else BLACK
    move, policy, value = _player.get_move(board)

    top_k = [
        [int(i), float(policy[i])]
        for i in np.argsort(policy)[::-1][:3]
        if policy[i] > 0
    ]

    # Raw model output — if the model picks a forbidden spot, it loses.
    if ai_color == BLACK and is_forbidden(board, move):
        return _result(board, "ai_forbidden", ai_move=move, forbidden_pos=move, top_k=top_k)

    board.play(move)
    black_value = float(value if ai_color == BLACK else -value)

    winner = check_winner_at(board, move)
    if winner is not None:
        return _result(board, "ai_win", ai_move=move, black_value=black_value,
                       top_k=top_k, win_line=_winning_line(board, move))
    if board.is_full():
        return _result(board, "draw", ai_move=move, black_value=black_value, top_k=top_k)
    return _result(board, "playing", ai_move=move, black_value=black_value, top_k=top_k)


@app.post("/api/play")
def play(req: PlayRequest) -> dict:
    _maybe_reload()
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
            return _result(board, "human_forbidden", forbidden_pos=pos)

        board.play(pos)
        winner = check_winner_at(board, pos)
        if winner is not None:
            return _result(board, "human_win",
                           black_value=1.0 if winner == BLACK else -1.0,
                           win_line=_winning_line(board, pos))
        if board.is_full():
            return _result(board, "draw", black_value=0.0)

    return _ai_turn(board, human_color)


@app.post("/api/hint")
def hint(req: HintRequest) -> dict:
    _maybe_reload()
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
    global _player, _device, _network, _ckpt_path, _ckpt_mtime

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
    _network = GomoNovaNet(
        channels=m["channels"],
        num_blocks=m["num_blocks"],
        policy_channels=m["policy_channels"],
        value_channels=m["value_channels"],
    ).bfloat16().to(_device)
    for mod in _network.modules():
        if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            mod.float()
    _ckpt_path = args.checkpoint
    _ckpt_mtime = os.path.getmtime(_ckpt_path)
    load_checkpoint(_ckpt_path, _network, _device)
    _network.eval()

    _player = InferencePlayer(_network, _device, temperature=0.0)
    print(f"Model loaded from {_ckpt_path} on {_device} (hot-reload enabled)")
    print(f"Open http://localhost:{args.port} in your browser")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
