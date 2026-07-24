import numpy as np
import pytest
import torch

from gomonova.game.board import Board, rc_to_pos
from gomonova.nn.encoder import INPUT_CHANNELS
from gomonova.nn.network import GomoNovaNet
from gomonova.training.replay import ReplayBuffer
from gomonova.training.selfplay import generate_games


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        planes = np.zeros((INPUT_CHANNELS, 15, 15), dtype=np.float32)
        buf.add(planes, move=10, outcome=1.0)
        assert len(buf) == 1
        p, m, o, mp = buf.sample(1)
        assert p.shape == (1, INPUT_CHANNELS, 15, 15)
        assert m[0] == 10
        assert o[0] == 1.0
        assert mp.shape == (1, 225)
        assert mp[0].sum() == 0.0  # no MCTS policy

    def test_add_with_mcts_policy(self):
        buf = ReplayBuffer(capacity=100)
        planes = np.zeros((INPUT_CHANNELS, 15, 15), dtype=np.float32)
        mcts = np.zeros(225, dtype=np.float32)
        mcts[10] = 0.7
        mcts[20] = 0.3
        buf.add(planes, move=10, outcome=1.0, mcts_policy=mcts)
        _, _, _, mp = buf.sample(1)
        assert abs(mp[0].sum() - 1.0) < 1e-5
        assert mp[0, 10] == pytest.approx(0.7)

    def test_ring_buffer_wrap(self):
        buf = ReplayBuffer(capacity=10)
        planes = np.zeros((INPUT_CHANNELS, 15, 15), dtype=np.float32)
        for i in range(15):
            buf.add(planes, move=i, outcome=0.0)
        assert len(buf) == 10

    def test_add_batch(self):
        buf = ReplayBuffer(capacity=100)
        planes = np.zeros((INPUT_CHANNELS, 15, 15), dtype=np.float32)
        samples = [(planes, i, 0.5, None) for i in range(5)]
        buf.add_batch(samples)
        assert len(buf) == 5

    def test_sample_returns_correct_types(self):
        buf = ReplayBuffer(capacity=100)
        planes = np.zeros((INPUT_CHANNELS, 15, 15), dtype=np.float32)
        buf.add(planes, move=42, outcome=-1.0)
        p, m, o, mp = buf.sample(1)
        assert p.dtype == np.float32
        assert m.dtype == np.int64
        assert o.dtype == np.float32
        assert mp.dtype == np.float32


class TestSelfplay:
    @pytest.fixture
    def small_net(self):
        net = GomoNovaNet(
            channels=32, num_blocks=2, policy_channels=16, value_channels=8
        )
        net.eval()
        return net

    def test_generate_games_pure_policy(self, small_net):
        samples = generate_games(
            small_net, torch.device("cpu"),
            num_games=2, augment=1, use_mcts=False,
        )
        assert len(samples) > 0
        planes, move, outcome, mcts_pol = samples[0]
        assert planes.shape == (INPUT_CHANNELS, 15, 15)
        assert 0 <= move < 225
        assert outcome in (-1.0, 0.0, 1.0)
        assert mcts_pol is None

    def test_generate_games_with_mcts(self, small_net):
        samples = generate_games(
            small_net, torch.device("cpu"),
            num_games=1, augment=1,
            use_mcts=True, mcts_sims=5, mcts_moves=3,
        )
        assert len(samples) > 0
        has_mcts = any(s[3] is not None for s in samples)
        assert has_mcts, "MCTS moves should produce non-None mcts_policy"

    def test_augmentation_preserves_outcome(self, small_net):
        samples = generate_games(
            small_net, torch.device("cpu"),
            num_games=1, augment=4, use_mcts=False,
        )
        outcomes = {s[2] for s in samples}
        assert len(outcomes) <= 3  # -1, 0, 1
