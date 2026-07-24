"""Parallel MCTS with round-synchronized shared memory.

All workers traverse trees in lockstep rounds.  Each round:
  1. Workers write leaf positions to shared memory
  2. Workers increment ready_count, poll for gen advance
  3. Main process waits for ready_count == n_games
  4. Main evaluates ALL positions in one GPU batch
  5. Main writes results, advances gen
  6. Workers read results, update trees, next round

This keeps GPU batches large (= n_games) and CPU work fully parallel.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import torch

from ..game.board import BLACK, Board
from ..game.rules import check_winner_at, is_legal
from ..mcts.flat_tree import FlatMCTSTree
from ..mcts.search import _check_winner_freestyle
from ..nn.encoder import INPUT_CHANNELS, board_to_planes

_CELLS = 225
_PLANE_SIZE = INPUT_CHANNELS * _CELLS  # 3600
_RESULT_SIZE = _CELLS + 1              # 226


def _worker(
    wid: int,
    boards: list[Board],
    num_sims: int,
    c_puct: float,
    d_alpha: float,
    d_eps: float,
    use_renju: bool,
    planes_raw,
    results_raw,
    ready_count,
    gen,
    out_q: mp.Queue,
    n_total: int,
) -> None:
    n = len(boards)
    offset = wid  # will be set by caller via boards slicing

    planes = np.frombuffer(planes_raw, dtype=np.float32).reshape(n_total, _PLANE_SIZE)
    results = np.frombuffer(results_raw, dtype=np.float32).reshape(n_total, _RESULT_SIZE)

    check_win = check_winner_at if use_renju else _check_winner_freestyle

    def _legal(board: Board) -> np.ndarray:
        moves = board.legal_moves()
        if use_renju and board.current == BLACK:
            leg = np.array([m for m in moves if is_legal(board, int(m))], dtype=np.int64)
            return leg if len(leg) > 0 else moves
        return moves

    def _sync(round_gen: int) -> None:
        with ready_count.get_lock():
            ready_count.value += n
        while True:
            with gen.get_lock():
                if gen.value > round_gen:
                    break
            time.sleep(0.00002)

    trees = [FlatMCTSTree(num_sims + 2) for _ in range(n)]

    # Root expansion
    for i, b in enumerate(boards):
        planes[offset + i, :] = board_to_planes(b).ravel()
    _sync(0)
    for i, b in enumerate(boards):
        pol = results[offset + i, :_CELLS].copy()
        trees[i].expand_node(0, pol, _legal(b))
        trees[i].add_dirichlet(0, d_alpha, d_eps)

    # Simulations
    for sim in range(num_sims):
        rg = sim + 1
        need: list[tuple[int, int]] = []

        for i in range(n):
            tree, board = trees[i], boards[i]
            node, slots, terminal = 0, [], False

            while tree.is_expanded[node] and tree.num_children[node] > 0:
                slot, action = tree.best_child(node, c_puct)
                board.play(action)
                slots.append(slot)
                w = check_win(board, action)
                if w is not None:
                    tree.backup(slots, -1.0 if board.current == BLACK else 1.0)
                    terminal = True
                    break
                if board.is_full():
                    tree.backup(slots, 0.0)
                    terminal = True
                    break
                node = tree.alloc_node()

            for _ in slots:
                board.undo()

            if not terminal:
                planes[offset + i, :] = board_to_planes(board).ravel()
                need.append((i, node))
            else:
                planes[offset + i, :] = 0.0  # dummy

        _sync(rg)

        for i, node in need:
            pol = results[offset + i, :_CELLS].copy()
            val = float(results[offset + i, _CELLS])
            leg = _legal(boards[i])
            if len(leg) == 0:
                trees[i].backup([node], 0.0)
            else:
                trees[i].expand_node(node, pol, leg)
                trees[i].backup([node], val)

    out_q.put((wid, [t.get_visit_distribution(1.0) for t in trees]))


def parallel_batch_search(
    boards: list[Board],
    network: torch.nn.Module,
    device: torch.device,
    num_workers: int = 36,
    num_simulations: int = 25,
    c_puct: float = 2.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    use_renju: bool = False,
) -> list[np.ndarray]:
    """Run MCTS with parallel CPU tree traversal + centralized GPU eval."""
    n = len(boards)
    if n == 0:
        return []

    num_workers = min(num_workers, n)
    chunk = (n + num_workers - 1) // num_workers
    actual_workers = (n + chunk - 1) // chunk

    planes_raw = mp.RawArray("f", n * _PLANE_SIZE)
    results_raw = mp.RawArray("f", n * _RESULT_SIZE)
    ready_count = mp.Value("i", 0)
    gen = mp.Value("i", 0)
    out_q: mp.Queue = mp.Queue()

    # Worker w handles boards[w*chunk : (w+1)*chunk], offset = w*chunk
    worker_offsets = []
    procs = []
    for w in range(actual_workers):
        s = w * chunk
        e = min(s + chunk, n)
        if s >= n:
            break
        worker_offsets.append(s)
        p = mp.Process(
            target=_worker,
            args=(
                s, boards[s:e], num_simulations, c_puct,
                dirichlet_alpha, dirichlet_epsilon, use_renju,
                planes_raw, results_raw, ready_count, gen,
                out_q, n,
            ),
        )
        p.start()
        procs.append(p)

    planes_np = np.frombuffer(planes_raw, dtype=np.float32).reshape(n, _PLANE_SIZE)
    results_np = np.frombuffer(results_raw, dtype=np.float32).reshape(n, _RESULT_SIZE)
    dtype = next(network.parameters()).dtype

    total_rounds = num_simulations + 1
    for rd in range(total_rounds):
        # Wait for all workers to write positions
        while True:
            with ready_count.get_lock():
                if ready_count.value >= n:
                    ready_count.value = 0
                    break
            time.sleep(0.00002)

        # GPU batch evaluation (all n positions)
        planes_batch = planes_np.reshape(n, INPUT_CHANNELS, 15, 15).copy()
        x = torch.from_numpy(planes_batch).to(device=device, dtype=dtype)
        with torch.no_grad():
            logits, values = network(x)
        pols = torch.softmax(logits.float(), dim=1).cpu().numpy()
        vals = values.float().squeeze(-1).cpu().numpy()

        results_np[:, :_CELLS] = pols
        results_np[:, _CELLS] = vals

        with gen.get_lock():
            gen.value = rd + 1

    # Collect results
    raw = {}
    for _ in range(len(worker_offsets)):
        wid, pols = out_q.get()
        raw[wid] = pols

    result: list[np.ndarray] = []
    for s in worker_offsets:
        result.extend(raw[s])

    for p in procs:
        p.join(timeout=5)

    return result
