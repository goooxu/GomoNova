"""Self-play with optional MCTS guidance for training.

Supports two modes:
  - Pure policy: fast lockstep batched games (Phase 1 warmup)
  - MCTS-guided: MCTS for opening moves, pure policy for rest (Phase 2/3)

Training targets:
  - MCTS moves: visit distribution (KL divergence loss)
  - Pure-policy moves: played move (outcome-weighted CE loss)
"""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, WHITE, BOARD_SIZE, Board, pos_to_rc, rc_to_pos
from ..game.symmetry import NUM_TRANSFORMS, transform_board, transform_policy
from ..mcts.search import MCTSSearch
from ..nn.encoder import board_to_planes

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

TENGAN = rc_to_pos(7, 7)   # 天元 (H8)：连珠规则规定黑棋第一手必须下在这里


def _open_game() -> tuple[Board, tuple]:
    """Create a board with the Renju opening: Black plays tengen (center).

    Returns the board (Black's first move already played, White to move) and
    the opening record ``(planes, mcts_policy, move, player)`` for training.
    The opening move is forced (mcts_policy=None), so it is learned via
    outcome-weighted CE — reinforcing "first move = tengen" in every game.
    """
    board = Board()
    planes = board_to_planes(board)   # empty board, Black to move
    board.play(TENGAN)
    return board, (planes, None, TENGAN, BLACK)


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
    mask = np.zeros(225, dtype=np.float32)
    mask[board.legal_moves()] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Pure-policy self-play (Phase 1: fast warmup)
# ---------------------------------------------------------------------------

def _net_dtype(network: torch.nn.Module) -> torch.dtype:
    return next(network.parameters()).dtype


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
    dtype = _net_dtype(network)
    records = []
    games_remaining = num_games

    while games_remaining > 0:
        n = min(batch_size, games_remaining)
        games_remaining -= n

        boards = []
        histories: list[list[tuple]] = []
        for _ in range(n):
            board, opening = _open_game()
            boards.append(board)
            histories.append([opening])
        active = list(range(n))

        while active:
            active_boards = [boards[i] for i in active]
            planes = np.stack([board_to_planes(b) for b in active_boards])
            x = torch.from_numpy(planes).to(device=device, dtype=dtype)
            logits, _ = network(x)
            policies = torch.softmax(logits.float(), dim=1).cpu().numpy()

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
                move = _sample_move(masked_policy, temp)
                # (planes, mcts_policy, move, player)
                histories[game_idx].append(
                    (planes_snapshot, None, move, board.current)
                )

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


# ---------------------------------------------------------------------------
# MCTS-guided self-play (Phase 2/3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def play_games_with_mcts(
    network: torch.nn.Module,
    device: torch.device,
    num_games: int,
    mcts_searcher: MCTSSearch,
    mcts_moves: int = 10,
    temperature: float = 1.0,
    temp_decay_move: int = 20,
    batch_size: int = 256,
) -> list[dict]:
    """Play games with batched MCTS for opening, then batched pure policy."""
    network.eval()
    dtype = _net_dtype(network)

    boards = []
    histories: list[list[tuple]] = []
    for _ in range(num_games):
        board, opening = _open_game()
        boards.append(board)
        histories.append([opening])
    results: list[float | None] = [None] * num_games

    # Phase A: batched MCTS moves
    for move_num in range(mcts_moves):
        active = [i for i in range(num_games) if results[i] is None]
        if not active:
            break

        active_boards = [boards[i] for i in active]
        temp = temperature if move_num < temp_decay_move else 0.1
        if len(active_boards) >= 32:
            from .parallel_mcts import parallel_batch_search
            visit_dists = parallel_batch_search(
                active_boards, network, device,
                num_workers=min(36, len(active_boards)),
                num_simulations=mcts_searcher.num_simulations,
                c_puct=mcts_searcher.c_puct,
                dirichlet_alpha=mcts_searcher.dirichlet_alpha,
                dirichlet_epsilon=mcts_searcher.dirichlet_epsilon,
                use_renju=mcts_searcher.use_renju,
            )
        else:
            visit_dists = mcts_searcher.flat_batch_search(active_boards, add_noise=True)

        for idx, game_idx in enumerate(active):
            board = boards[game_idx]
            mcts_policy = visit_dists[idx]
            planes_snapshot = board_to_planes(board)

            legal_mask = _get_legal_mask(board)
            masked = mcts_policy * legal_mask
            total = masked.sum()
            if total > 0:
                masked /= total
            else:
                masked[legal_mask > 0] = 1.0 / legal_mask.sum()

            move = _sample_move(masked, temp)
            histories[game_idx].append(
                (planes_snapshot, mcts_policy.copy(), move, board.current)
            )
            board.play(move)
            winner = mcts_searcher._check_winner(board, move)
            if winner is not None:
                results[game_idx] = 1.0 if winner == BLACK else -1.0
            elif board.is_full():
                results[game_idx] = 0.0

    # Phase B: batched pure policy moves (lockstep)
    while True:
        active = [i for i in range(num_games) if results[i] is None]
        if not active:
            break

        active_boards = [boards[i] for i in active]
        planes = np.stack([board_to_planes(b) for b in active_boards])
        x = torch.from_numpy(planes).to(device=device, dtype=dtype)
        logits, _ = network(x)
        policies = torch.softmax(logits.float(), dim=1).cpu().numpy()

        for idx, game_idx in enumerate(active):
            board = boards[game_idx]
            move_num = len(histories[game_idx])
            temp = temperature if move_num < temp_decay_move else 0.1

            legal_mask = _get_legal_mask(board)
            masked = policies[idx] * legal_mask
            total = masked.sum()
            if total > 0:
                masked /= total
            else:
                masked[legal_mask > 0] = 1.0 / legal_mask.sum()

            planes_snapshot = board_to_planes(board)
            move = _sample_move(masked, temp)
            histories[game_idx].append((planes_snapshot, None, move, board.current))

            board.play(move)
            winner = mcts_searcher._check_winner(board, move)
            if winner is not None:
                results[game_idx] = 1.0 if winner == BLACK else -1.0
            elif board.is_full():
                results[game_idx] = 0.0

    return [
        {"history": histories[i], "result": results[i]}
        for i in range(num_games)
    ]


def _sample_move(policy: np.ndarray, temp: float) -> int:
    if temp < 1e-8:
        return int(np.argmax(policy))
    tempered = policy ** (1.0 / temp)
    tempered /= tempered.sum()
    return int(np.random.choice(len(tempered), p=tempered))


# ---------------------------------------------------------------------------
# Convert records to training samples
# ---------------------------------------------------------------------------

def records_to_samples(
    records: list[dict],
    augment: int = 4,
) -> list[tuple[np.ndarray, int, float, np.ndarray | None]]:
    """Convert game records to (planes, move, outcome, mcts_policy) samples."""
    samples = []
    rng = np.random.default_rng()
    for rec in records:
        result = rec["result"]
        for planes, mcts_pol, move, player in rec["history"]:
            outcome = result if player == BLACK else -result
            samples.append((planes, move, outcome, mcts_pol))

            if augment > 1:
                for _ in range(augment - 1):
                    t = int(rng.integers(1, NUM_TRANSFORMS))
                    aug_planes = np.stack([
                        transform_board(planes[i], t)
                        for i in range(planes.shape[0])
                    ])
                    aug_move = int(transform_policy(
                        _move_to_policy(move), t
                    ).argmax())
                    aug_mcts = None
                    if mcts_pol is not None:
                        aug_mcts = transform_policy(mcts_pol, t)
                    samples.append((aug_planes, aug_move, outcome, aug_mcts))

    return samples


def _move_to_policy(move: int) -> np.ndarray:
    p = np.zeros(225, dtype=np.float32)
    p[move] = 1.0
    return p


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def generate_games(
    network: torch.nn.Module,
    device: torch.device,
    num_games: int,
    temp_threshold: int = 30,
    augment: int = 4,
    parallel_games: int = 256,
    use_mcts: bool = False,
    mcts_sims: int = 25,
    mcts_moves: int = 10,
    use_renju: bool = False,
    dirichlet_alpha: float = 0.3,
) -> list[tuple[np.ndarray, int, float, np.ndarray | None]]:
    """Generate self-play games. Returns training samples."""
    if use_mcts:
        searcher = MCTSSearch(
            network, device,
            num_simulations=mcts_sims,
            dirichlet_alpha=dirichlet_alpha,
            use_renju=use_renju,
        )
        records = play_games_with_mcts(
            network, device,
            num_games=num_games,
            mcts_searcher=searcher,
            mcts_moves=mcts_moves,
            temperature=1.0,
            temp_decay_move=temp_threshold,
        )
    else:
        records = play_games_fast(
            network, device,
            num_games=num_games,
            temperature=1.0,
            temp_decay_move=temp_threshold,
            batch_size=parallel_games,
        )
    return records_to_samples(records, augment=augment)
