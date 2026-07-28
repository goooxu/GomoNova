"""MCTS search using neural network evaluation (training only).

This module is imported ONLY by training/selfplay.py.  The inference path
(gomonova/inference/) must never import from here.
"""

from __future__ import annotations

import numpy as np
import torch

from ..game.board import BLACK, BOARD_SIZE, Board, pos_to_rc
from ..game.rules import check_winner_at, is_legal
from ..nn.encoder import board_to_planes
from .flat_tree import FlatMCTSTree
from .node import MCTSNode

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


def candidate_moves(board: Board, apply_forbidden: bool) -> np.ndarray:
    """Empty cells within distance-2 of any stone (standard gomoku MCTS pruning).

    Restricting candidates to the neighbourhood of existing stones drastically
    cuts opening branching, so MCTS concentrates its simulations on plausible
    moves instead of spreading them over all 225 cells — which otherwise lets a
    weak policy prior trap the search on far-away edge/corner moves.  Falls back
    to all legal moves on an empty board or if pruning leaves nothing.
    """
    moves = board.legal_moves()
    occupied = np.argwhere(board.cells != 0)
    if len(occupied) == 0:
        candidates = moves
    else:
        cand: set[int] = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and board.cells[nr, nc] == 0):
                        cand.add(nr * BOARD_SIZE + nc)
        candidates = np.array(sorted(cand), dtype=np.int64)
    if apply_forbidden:
        candidates = np.array(
            [m for m in candidates if is_legal(board, int(m))], dtype=np.int64
        )
    if len(candidates) == 0:
        return moves
    return candidates


class MCTSSearch:
    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 400,
        c_puct: float = 2.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_renju: bool = True,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_renju = use_renju

    @torch.no_grad()
    def _evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        planes = board_to_planes(board)
        dtype = next(self.network.parameters()).dtype
        x = torch.from_numpy(planes).unsqueeze(0).to(
            device=self.device, dtype=dtype
        )
        policy_logits, value = self.network(x)
        policy = torch.softmax(policy_logits.float(), dim=1).cpu().numpy()[0]
        v = value.item()
        return policy, v

    def _get_legal_actions(self, board: Board) -> np.ndarray:
        apply_forbidden = self.use_renju and board.current == BLACK
        return candidate_moves(board, apply_forbidden)

    def _check_winner(self, board: Board, pos: int) -> int | None:
        if self.use_renju:
            return check_winner_at(board, pos)
        return _check_winner_freestyle(board, pos)

    def search(self, board: Board, add_noise: bool = True) -> MCTSNode:
        """Run MCTS from the current board state and return the root node."""
        root = MCTSNode()
        policy, _ = self._evaluate(board)
        legal = self._get_legal_actions(board)
        root.expand(policy, legal)
        if add_noise:
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()

            while node.is_expanded and node.children:
                node = node.best_child(self.c_puct)
                sim_board.play(node.action)
                winner = self._check_winner(sim_board, node.action)
                if winner is not None:
                    # 终局节点轮走方=输家，negamax 价值恒为 -1（与 backup「先加后翻」
                    # 及 best_child 取 -q 的约定一致；旧实现按黑/白胜给异号是错的）。
                    node.backup(-1.0)
                    break
                if sim_board.is_full():
                    node.backup(0.0)
                    break
            else:
                if not node.is_expanded:
                    policy, v = self._evaluate(sim_board)
                    legal = self._get_legal_actions(sim_board)
                    if len(legal) == 0:
                        node.backup(0.0)
                    else:
                        node.expand(policy, legal)
                        node.backup(v)

        return root

    def get_move(self, board: Board, temperature: float = 1.0) -> tuple[int, np.ndarray]:
        """Run search and select a move. Returns (move, visit_policy)."""
        root = self.search(board, add_noise=True)
        policy = root.get_visit_distribution(temperature)
        move = int(np.argmax(policy)) if temperature < 1e-8 else int(
            np.random.choice(len(policy), p=policy)
        )
        return move, policy

    @torch.no_grad()
    def batch_search(
        self, boards: list[Board], add_noise: bool = True,
    ) -> list[np.ndarray]:
        """Run MCTS on multiple boards simultaneously with batched NN eval.

        Key optimizations over per-game search:
          - NN evaluations are batched across all games (1 GPU call per round)
          - Uses board.undo() instead of board.copy() per simulation
        """
        n = len(boards)
        if n == 0:
            return []
        dtype = next(self.network.parameters()).dtype

        roots: list[MCTSNode] = []
        for board in boards:
            root = MCTSNode()
            policy, _ = self._evaluate(board)
            legal = self._get_legal_actions(board)
            root.expand(policy, legal)
            if add_noise:
                root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)
            roots.append(root)

        for _ in range(self.num_simulations):
            expand_info: list[tuple[int, MCTSNode, np.ndarray, np.ndarray]] = []

            for i in range(n):
                node = roots[i]
                board = boards[i]
                moves_played: list[int] = []
                terminal = False

                while node.is_expanded and node.children:
                    node = node.best_child(self.c_puct)
                    board.play(node.action)
                    moves_played.append(node.action)
                    winner = self._check_winner(board, node.action)
                    if winner is not None:
                        node.backup(-1.0)   # 终局轮走方=输家，negamax 价值恒 -1
                        terminal = True
                        break
                    if board.is_full():
                        node.backup(0.0)
                        terminal = True
                        break

                expand_this = (not terminal) and (not node.is_expanded)
                if expand_this:
                    # undo 之前捕获叶子局面（planes + 合法动作）：undo 后 board 回到
                    # 根局面，评估它会让所有叶子拿到同一个根价值，MCTS 失去叶估值信号。
                    leaf_planes = board_to_planes(board)
                    leaf_legal = self._get_legal_actions(board)

                for _ in moves_played:
                    board.undo()

                if expand_this:
                    expand_info.append((i, node, leaf_planes, leaf_legal))

            if expand_info:
                planes = np.stack([lp for _, _, lp, _ in expand_info])
                x = torch.from_numpy(planes).to(device=self.device, dtype=dtype)
                logits, values = self.network(x)
                policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
                vals = values.float().squeeze(-1).cpu().numpy()

                for j, (i, node, _lp, legal) in enumerate(expand_info):
                    if len(legal) == 0:
                        node.backup(0.0)
                    else:
                        node.expand(policies[j], legal)
                        node.backup(float(vals[j]))

        return [root.get_visit_distribution(1.0) for root in roots]

    @torch.no_grad()
    def flat_batch_search(
        self, boards: list[Board], add_noise: bool = True,
    ) -> list[np.ndarray]:
        """Batched MCTS using array-based trees + batched NN eval.

        Combines two optimizations:
          1. FlatMCTSTree: numpy arrays instead of Python objects (~10× faster node ops)
          2. Batched NN eval: one GPU call per simulation round
        """
        n = len(boards)
        if n == 0:
            return []
        dtype = next(self.network.parameters()).dtype
        max_nodes = self.num_simulations + 2

        trees = [FlatMCTSTree(max_nodes) for _ in range(n)]
        for i, board in enumerate(boards):
            policy, _ = self._evaluate(board)
            legal = self._get_legal_actions(board)
            trees[i].expand_node(0, policy, legal)
            if add_noise:
                trees[i].add_dirichlet(0, self.dirichlet_alpha, self.dirichlet_epsilon)

        for _ in range(self.num_simulations):
            expand_info: list[tuple[int, int, np.ndarray, np.ndarray, list[int]]] = []

            for i in range(n):
                tree = trees[i]
                board = boards[i]
                node = 0
                slots: list[int] = []
                terminal = False

                while tree.is_expanded[node] and tree.num_children[node] > 0:
                    slot, action = tree.best_child(node, self.c_puct)
                    board.play(action)
                    slots.append(slot)
                    winner = self._check_winner(board, action)
                    if winner is not None:
                        tree.backup(slots, -1.0)   # 终局轮走方=输家，价值恒 -1
                        terminal = True
                        break
                    if board.is_full():
                        tree.backup(slots, 0.0)
                        terminal = True
                        break
                    node = tree.alloc_node()

                if not terminal:
                    # undo 之前捕获叶子局面（planes + 合法动作）：undo 后 board 回到
                    # 根局面，评估它会让所有叶子拿到同一个根价值，MCTS 失去叶估值信号。
                    leaf_planes = board_to_planes(board)
                    leaf_legal = self._get_legal_actions(board)
                else:
                    leaf_planes = leaf_legal = None

                for _ in slots:
                    board.undo()

                if not terminal:
                    expand_info.append((i, node, leaf_planes, leaf_legal, slots))

            if expand_info:
                planes = np.stack([lp for _, _, lp, _, _ in expand_info])
                x = torch.from_numpy(planes).to(device=self.device, dtype=dtype)
                logits, values = self.network(x)
                policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
                vals = values.float().squeeze(-1).cpu().numpy()

                for j, (i, node, _lp, legal, slots) in enumerate(expand_info):
                    if len(legal) == 0:
                        trees[i].backup(slots, 0.0)
                    else:
                        trees[i].expand_node(node, policies[j], legal)
                        # 回传完整路径 slots（含进入叶子的边）。
                        trees[i].backup(slots, float(vals[j]))

        return [trees[i].get_visit_distribution(1.0) for i in range(n)]
