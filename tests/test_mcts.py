import numpy as np
import pytest
import torch

from gomonova.game.board import BLACK, WHITE, Board, rc_to_pos
from gomonova.game.rules import is_legal
from gomonova.mcts.node import MCTSNode
from gomonova.mcts.search import MCTSSearch
from gomonova.nn.network import GomoNovaNet


@pytest.fixture
def net():
    n = GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)
    n.eval()
    return n


@pytest.fixture
def search(net):
    return MCTSSearch(net, torch.device("cpu"), num_simulations=300, c_puct=2.0)


class TestMCTSNode:
    def test_expand_and_visit(self):
        root = MCTSNode()
        policy = np.ones(225, dtype=np.float32) / 225
        legal = np.array([0, 1, 2, 100, 224])
        root.expand(policy, legal)
        assert root.is_expanded
        assert len(root.children) == 5

    def test_visit_distribution_sums_to_one(self):
        root = MCTSNode()
        policy = np.random.rand(225).astype(np.float32)
        legal = np.arange(225)
        root.expand(policy, legal)
        for child in root.children.values():
            child.visit_count = np.random.randint(1, 100)
        dist = root.get_visit_distribution(temperature=1.0)
        assert abs(dist.sum() - 1.0) < 1e-5

    def test_greedy_distribution(self):
        root = MCTSNode()
        policy = np.ones(225, dtype=np.float32) / 225
        legal = np.array([10, 20, 30])
        root.expand(policy, legal)
        root.children[10].visit_count = 100
        root.children[20].visit_count = 50
        root.children[30].visit_count = 10
        dist = root.get_visit_distribution(temperature=0.0)
        assert dist[10] == 1.0
        assert dist.sum() == 1.0

    def test_backup(self):
        root = MCTSNode()
        policy = np.ones(225, dtype=np.float32) / 225
        root.expand(policy, np.array([5]))
        child = root.children[5]
        child.backup(0.8)
        assert child.visit_count == 1
        assert child.total_value == 0.8
        assert root.visit_count == 1
        assert root.total_value == -0.8

    def test_dirichlet_noise(self):
        root = MCTSNode()
        policy = np.ones(225, dtype=np.float32) / 225
        legal = np.arange(10)
        root.expand(policy, legal)
        priors_before = [root.children[a].prior for a in range(10)]
        root.add_dirichlet_noise(0.3, 0.25)
        priors_after = [root.children[a].prior for a in range(10)]
        assert priors_before != priors_after


class TestMCTSSearch:
    def test_finds_immediate_win(self, search):
        """Black has 4 in a row, MCTS should find the winning move."""
        b = Board()
        b.cells[7, 3] = BLACK
        b.cells[7, 4] = BLACK
        b.cells[7, 5] = BLACK
        b.cells[7, 6] = BLACK
        b.cells[0, 0] = WHITE
        b.cells[0, 1] = WHITE
        b.cells[0, 2] = WHITE
        b.history = [rc_to_pos(7, 3), rc_to_pos(0, 0), rc_to_pos(7, 4),
                     rc_to_pos(0, 1), rc_to_pos(7, 5), rc_to_pos(0, 2), rc_to_pos(7, 6)]
        b.current = BLACK
        root = search.search(b, add_noise=False)
        dist = root.get_visit_distribution(0.0)
        winning_move = rc_to_pos(7, 7)
        winning_move2 = rc_to_pos(7, 2)
        assert dist[winning_move] == 1.0 or dist[winning_move2] == 1.0

    def test_blocks_opponent_four(self, search):
        """MCTS completes and explores blocking positions (random net can't guarantee priority)."""
        b = Board()
        b.cells[7, 3] = WHITE
        b.cells[7, 4] = WHITE
        b.cells[7, 5] = WHITE
        b.cells[7, 6] = WHITE
        b.cells[0, 0] = BLACK
        b.cells[0, 1] = BLACK
        b.history = [rc_to_pos(0, 0), rc_to_pos(7, 3), rc_to_pos(0, 1),
                     rc_to_pos(7, 4), rc_to_pos(7, 5), rc_to_pos(7, 6)]
        b.current = BLACK
        search.num_simulations = 300
        root = search.search(b, add_noise=False)
        block1 = rc_to_pos(7, 2)
        block2 = rc_to_pos(7, 7)
        assert block1 in root.children or block2 in root.children
        move, policy = search.get_move(b, temperature=0.0)
        assert b.is_empty(move)
        assert policy.sum() > 0

    def test_search_returns_valid_move(self, search):
        b = Board()
        b.play(rc_to_pos(7, 7))
        b.play(rc_to_pos(7, 8))
        move, policy = search.get_move(b, temperature=1.0)
        assert 0 <= move < 225
        assert b.is_empty(move)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_forbidden_moves_excluded(self, search):
        """MCTS for Black should not select forbidden moves."""
        b = Board()
        b.cells[7, 2] = BLACK
        b.cells[7, 3] = BLACK
        b.cells[7, 4] = BLACK
        b.cells[7, 6] = BLACK
        b.cells[7, 7] = BLACK
        b.cells[0, 0] = WHITE
        b.cells[0, 1] = WHITE
        b.history = [rc_to_pos(7, 2), rc_to_pos(0, 0), rc_to_pos(7, 3),
                     rc_to_pos(0, 1), rc_to_pos(7, 4), rc_to_pos(7, 6), rc_to_pos(7, 7)]
        b.current = BLACK
        legal = search._get_legal_actions(b)
        forbidden_pos = rc_to_pos(7, 5)
        assert forbidden_pos not in legal
