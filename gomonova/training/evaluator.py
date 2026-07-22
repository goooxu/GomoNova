"""Model evaluator: batched parallel match (new vs best), pure inference."""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc
from ..nn.encoder import board_to_planes

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _check_winner_freestyle(board: Board, pos: int) -> int | None:
    r, c = pos_to_rc(pos)
    player = int(board.cells[r, c])
    if player == 0:
        return None
    for dr, dc in _DIRS:
        count = 1
        cr, cc = r + dr, c + dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board.cells[cr, cc] == player:
            count += 1
            cr += dr
            cc += dc
        cr, cc = r - dr, c - dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board.cells[cr, cc] == player:
            count += 1
            cr -= dr
            cc -= dc
        if count >= 5:
            return player
    return None


@torch.no_grad()
def _batch_move(network: torch.nn.Module, boards: list[Board], device: torch.device) -> list[int]:
    """Get greedy moves for a batch of boards in one forward pass."""
    if not boards:
        return []
    planes = np.stack([board_to_planes(b) for b in boards])
    x = torch.from_numpy(planes).to(device)
    logits, _ = network(x)
    policies = torch.softmax(logits, dim=1).cpu().numpy()

    moves = []
    for i, board in enumerate(boards):
        legal = board.legal_moves()
        masked = np.zeros(225, dtype=np.float32)
        masked[legal] = policies[i][legal]
        moves.append(int(np.argmax(masked)))
    return moves


def play_match(
    net_a: torch.nn.Module,
    net_b: torch.nn.Module,
    device: torch.device,
    num_games: int = 100,
) -> dict:
    """Batched parallel match between two networks. Returns stats for net_a."""
    net_a.eval()
    net_b.eval()

    boards = [Board() for _ in range(num_games)]
    # Even games: A plays Black (first), B plays White
    # Odd games: B plays Black (first), A plays White
    a_is_first = [i % 2 == 0 for i in range(num_games)]
    done = [False] * num_games
    results = [0.0] * num_games  # +1 = A wins, -1 = B wins, 0 = draw

    while not all(done):
        # Determine which boards need A's move and which need B's move
        a_indices = []
        b_indices = []
        for i in range(num_games):
            if done[i]:
                continue
            is_a_turn = (boards[i].current == BLACK) == a_is_first[i]
            if is_a_turn:
                a_indices.append(i)
            else:
                b_indices.append(i)

        # Batch evaluate for net_a
        if a_indices:
            a_boards = [boards[i] for i in a_indices]
            a_moves = _batch_move(net_a, a_boards, device)
            for idx, move in zip(a_indices, a_moves):
                boards[idx].play(move)
                winner = _check_winner_freestyle(boards[idx], move)
                if winner is not None:
                    results[idx] = 1.0 if a_is_first[idx] == (winner == BLACK) else -1.0
                    done[idx] = True
                elif boards[idx].is_full():
                    done[idx] = True

        # Batch evaluate for net_b
        if b_indices:
            b_boards = [boards[i] for i in b_indices]
            b_moves = _batch_move(net_b, b_boards, device)
            for idx, move in zip(b_indices, b_moves):
                boards[idx].play(move)
                winner = _check_winner_freestyle(boards[idx], move)
                if winner is not None:
                    results[idx] = -1.0 if a_is_first[idx] == (winner == BLACK) else 1.0
                    done[idx] = True
                elif boards[idx].is_full():
                    done[idx] = True

    wins_a = sum(1 for r in results if r > 0)
    wins_b = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)
    total = num_games
    return {
        "wins": wins_a,
        "losses": wins_b,
        "draws": draws,
        "winrate": (wins_a + 0.5 * draws) / total,
    }
