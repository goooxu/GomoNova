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


class _CenterValueNet(torch.nn.Module):
    """测试替身：均匀策略先验；价值头（轮走方视角）在 H8(7,7) 被占时返回 -0.9，
    否则 +0.9——即"占据中心让对手难受"，占中心是好棋。正确 MCTS（叶估值以正确符号
    回传）应优先占中心；历史符号 bug 会回避中心，叶子回传 bug 会完全无视价值。
    用于端到端回归守护这两个 bug。"""

    def __init__(self):
        super().__init__()
        # 哑参数：让 .parameters() 非空，供 MCTSSearch._evaluate 推断 dtype。
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        logits = torch.zeros(n, 225)
        occupied_center = x[:, 14, 7, 7] > 0.5   # 通道14=占位
        value = torch.where(
            occupied_center, torch.full((n,), -0.9), torch.full((n,), 0.9)
        ).view(n, 1)
        return logits, value


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

    def test_best_child_prefers_parent_favorable(self):
        # negamax 约定：子节点 q_value 是子节点轮走方（父节点的对手）视角，
        # 父节点为自己选子应取 -q 最大（对手最差）的子。回归守护选子符号 bug。
        root = MCTSNode()
        root.visit_count = 100
        policy = np.ones(225, dtype=np.float32) / 225
        root.expand(policy, np.array([10, 20]))
        good = root.children[10]
        bad = root.children[20]
        good.visit_count, good.total_value = 50, -45.0   # q=-0.9 对手差→对父有利
        bad.visit_count, bad.total_value = 50, 45.0      # q=+0.9 对手好→对父不利
        assert root.best_child(c_puct=0.0) is good

    def test_flat_best_child_prefers_parent_favorable(self):
        from gomonova.mcts.flat_tree import FlatMCTSTree
        tree = FlatMCTSTree(4)
        policy = np.ones(225, dtype=np.float32) / 225
        tree.expand_node(0, policy, np.array([10, 20]))
        tree.child_visit[0], tree.child_value[0] = 50, -45.0   # action10 q=-0.9 好
        tree.child_visit[1], tree.child_value[1] = 50, 45.0    # action20 q=+0.9 坏
        _, action = tree.best_child(0, c_puct=0.0)
        assert action == 10


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

    def test_finds_immediate_win_white(self, search):
        """White has 4 in a row; MCTS (correct terminal sign) finds the win."""
        b = Board()
        b.cells[7, 3] = WHITE
        b.cells[7, 4] = WHITE
        b.cells[7, 5] = WHITE
        b.cells[7, 6] = WHITE
        b.cells[0, 0] = BLACK
        b.cells[0, 1] = BLACK
        b.cells[0, 2] = BLACK
        b.history = [rc_to_pos(0, 0), rc_to_pos(7, 3), rc_to_pos(0, 1),
                     rc_to_pos(7, 4), rc_to_pos(0, 2), rc_to_pos(7, 5), rc_to_pos(7, 6)]
        b.current = WHITE
        root = search.search(b, add_noise=False)
        dist = root.get_visit_distribution(0.0)
        assert dist[rc_to_pos(7, 7)] == 1.0 or dist[rc_to_pos(7, 2)] == 1.0

    def test_search_uses_leaf_values_with_correct_sign(self):
        """端到端回归：占中心是好棋（假价值头），MCTS 必须选中心。
        同时守护选子符号 bug（会回避中心）与叶子回传 bug（会无视价值）。"""
        fake = _CenterValueNet().eval()
        s = MCTSSearch(fake, torch.device("cpu"), num_simulations=300,
                       c_puct=2.0, use_renju=False)
        center = rc_to_pos(7, 7)

        b = Board()
        b.play(rc_to_pos(7, 8))   # 黑下 I8，白先；H8(7,7) 为空且是候选
        root = s.search(b, add_noise=False)
        assert int(np.argmax(root.get_visit_distribution(0.0))) == center

        b2 = Board()
        b2.play(rc_to_pos(7, 8))
        flat = s.flat_batch_search([b2], add_noise=False)[0]
        assert int(np.argmax(flat)) == center

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
        """MCTS for Black should not select forbidden moves (Renju mode)."""
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

    def test_freestyle_allows_all_empty(self, net):
        """Freestyle MCTS: 空盘无子→不裁剪、无禁手过滤，返回全部 225 点。
        （有子时双方都按邻接区裁剪，见 candidate_moves。）"""
        fs_search = MCTSSearch(net, torch.device("cpu"), num_simulations=10, use_renju=False)
        legal = fs_search._get_legal_actions(Board())
        assert len(legal) == 225

    def test_freestyle_winner_detection(self, net):
        """Freestyle: 5+ in a row wins for both colors."""
        fs_search = MCTSSearch(net, torch.device("cpu"), num_simulations=10, use_renju=False)
        b = Board()
        b.cells[7, 3] = WHITE
        b.cells[7, 4] = WHITE
        b.cells[7, 5] = WHITE
        b.cells[7, 6] = WHITE
        b.cells[7, 7] = WHITE
        b.history = [rc_to_pos(7, 3), rc_to_pos(7, 4), rc_to_pos(7, 5),
                     rc_to_pos(7, 6), rc_to_pos(7, 7)]
        winner = fs_search._check_winner(b, rc_to_pos(7, 7))
        assert winner == WHITE
