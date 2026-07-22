"""Fast self-play via policy network (no MCTS).

Games are played using the policy network directly with temperature
sampling.  Training uses REINFORCE for policy and MSE for value.
This is orders of magnitude faster than MCTS-based self-play while
still learning purely from game rules.
"""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc
from ..game.symmetry import NUM_TRANSFORMS, transform_board, transform_policy
from ..nn.encoder import board_to_planes

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _check_winner_freestyle(board: Board, pos: int) -> int | None:
    """Freestyle win check: 5+ in a row wins for both colors."""
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


def _get_legal_mask(board: Board) -> np.ndarray:
    """Training uses freestyle rules (no forbidden moves) for speed."""
    mask = np.zeros(225, dtype=np.float32)
    mask[board.legal_moves()] = 1.0
    return mask


@torch.no_grad()
def play_games_fast(
    network: torch.nn.Module,
    device: torch.device,
    num_games: int,
    temperature: float = 1.0,
    temp_decay_move: int = 30,
    batch_size: int = 256,
) -> list[dict]:
    """Play games using policy network only. Returns game records."""
    network.eval()
    records = []
    games_remaining = num_games

    while games_remaining > 0:
        n = min(batch_size, games_remaining)
        games_remaining -= n

        boards = [Board() for _ in range(n)]
        histories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n)]
        active = list(range(n))

        while active:
            # Batch evaluate all active boards
            active_boards = [boards[i] for i in active]
            planes = np.stack([board_to_planes(b) for b in active_boards])
            x = torch.from_numpy(planes).to(device)
            logits, _ = network(x)
            policies = torch.softmax(logits, dim=1).cpu().numpy()

            # Select moves
            next_active = []
            for idx, game_idx in enumerate(active):
                board = boards[game_idx]
                move_num = len(histories[game_idx])
                temp = temperature if move_num < temp_decay_move else 0.1

                legal_mask = _get_legal_mask(board)
                masked_policy = policies[idx] * legal_mask
                total = masked_policy.sum()
                if total > 0:
                    masked_policy /= total
                else:
                    masked_policy[legal_mask > 0] = 1.0 / legal_mask.sum()

                planes_snapshot = board_to_planes(board)
                histories[game_idx].append((planes_snapshot, masked_policy.copy(), board.current))

                if temp < 1e-8:
                    move = int(np.argmax(masked_policy))
                else:
                    tempered = masked_policy ** (1.0 / temp)
                    tempered /= tempered.sum()
                    move = int(np.random.choice(225, p=tempered))

                board.play(move)
                winner = _check_winner_freestyle(board, move)

                if winner is not None:
                    result = 1.0 if winner == BLACK else -1.0
                    records.append({"history": histories[game_idx], "result": result})
                elif board.is_full():
                    records.append({"history": histories[game_idx], "result": 0.0})
                else:
                    next_active.append(game_idx)

            active = next_active

    return records


def records_to_samples(
    records: list[dict],
    augment: int = 4,
) -> list[tuple[np.ndarray, int, float]]:
    """Convert game records to (planes, move, outcome) training samples."""
    samples = []
    for rec in records:
        result = rec["result"]
        for planes, policy, player in rec["history"]:
            outcome = result if player == BLACK else -result
            move = int(np.argmax(policy))
            samples.append((planes, move, outcome))

            if augment > 1:
                rng = np.random.default_rng()
                for _ in range(augment - 1):
                    t = int(rng.integers(1, NUM_TRANSFORMS))
                    aug_planes = np.stack([transform_board(planes[i], t) for i in range(planes.shape[0])])
                    aug_move = int(transform_policy(policy, t).argmax())
                    samples.append((aug_planes, aug_move, outcome))

    return samples


def generate_games(
    network: torch.nn.Module,
    device: torch.device,
    num_games: int,
    num_simulations: int = 400,
    c_puct: float = 2.0,
    temp_threshold: int = 30,
    augment: int = 4,
    parallel_games: int = 256,
) -> list[tuple[np.ndarray, int, float]]:
    """Generate self-play games (API-compatible signature)."""
    records = play_games_fast(
        network, device,
        num_games=num_games,
        temperature=1.0,
        temp_decay_move=temp_threshold,
        batch_size=parallel_games,
    )
    return records_to_samples(records, augment=augment)
