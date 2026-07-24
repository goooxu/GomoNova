import torch
import pytest

from gomonova.game.board import Board, rc_to_pos
from gomonova.nn.encoder import INPUT_CHANNELS, board_to_planes, planes_to_tensor
from gomonova.nn.network import GomoNovaNet
from gomonova.nn.losses import total_loss


class TestEncoder:
    def test_empty_board_planes(self):
        b = Board()
        planes = board_to_planes(b)
        assert planes.shape == (INPUT_CHANNELS, 15, 15)
        assert planes[:INPUT_CHANNELS - 1].sum() == 0.0
        assert planes[INPUT_CHANNELS - 1].sum() == 225.0

    def test_planes_after_moves(self):
        b = Board()
        b.play(rc_to_pos(7, 7))   # Black plays
        b.play(rc_to_pos(7, 8))   # White plays → Black to move
        planes = board_to_planes(b)
        # current=Black, opponent=White
        assert planes[0, 7, 7] == 1.0   # current (Black) stone
        assert planes[1, 7, 8] == 1.0   # opponent (White) stone
        assert planes[2, 7, 7] == 1.0   # Black's last move (ch2)
        assert planes[8, 7, 8] == 1.0   # White's last move (ch 2+6=8)
        assert planes[14, 7, 7] == 1.0  # occupancy
        assert planes[14, 7, 8] == 1.0
        assert planes[15].sum() == 225.0  # Black turn bias

    def test_white_turn_bias(self):
        b = Board()
        b.play(rc_to_pos(7, 7))  # Black plays, White to move
        planes = board_to_planes(b)
        assert planes[15].sum() == 0.0

    def test_history_depth(self):
        b = Board()
        for i in range(12):
            b.play(rc_to_pos(i // 15, i % 15))
        planes = board_to_planes(b)
        assert planes.shape == (INPUT_CHANNELS, 15, 15)


class TestNetwork:
    @pytest.fixture
    def small_net(self):
        return GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)

    def test_output_shapes(self, small_net):
        x = torch.randn(4, INPUT_CHANNELS, 15, 15)
        policy, value = small_net(x)
        assert policy.shape == (4, 225)
        assert value.shape == (4, 1)

    def test_value_range(self, small_net):
        x = torch.randn(8, INPUT_CHANNELS, 15, 15)
        _, value = small_net(x)
        assert (value >= -1.0).all() and (value <= 1.0).all()

    def test_deterministic(self, small_net):
        small_net.eval()
        x = torch.randn(2, INPUT_CHANNELS, 15, 15)
        p1, v1 = small_net(x)
        p2, v2 = small_net(x)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_param_count_small(self, small_net):
        n = small_net.num_params()
        assert 100_000 < n < 5_000_000

    def test_param_count_main(self):
        net = GomoNovaNet()  # default: 192ch, 10 blocks
        n = net.num_params()
        assert 30_000_000 < n < 80_000_000, f"Expected 30-80M params, got {n:,}"

    def test_gradient_flow(self, small_net):
        x = torch.randn(2, INPUT_CHANNELS, 15, 15)
        policy, value = small_net(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, p in small_net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_per_position_gate(self, small_net):
        """Policy gate should be per-position (B, 225), not scalar."""
        x = torch.randn(2, INPUT_CHANNELS, 15, 15)
        small_net.eval()
        policy, _ = small_net(x)
        assert policy.shape == (2, 225)


class TestLosses:
    def test_total_loss_runs(self):
        net = GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)
        x = torch.randn(4, INPUT_CHANNELS, 15, 15)
        logits, value = net(x)
        moves = torch.tensor([10, 50, 100, 200])
        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])
        loss, p_loss, v_loss = total_loss(logits, value, moves, outcomes, net)
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_total_loss_with_mcts_policy(self):
        net = GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)
        x = torch.randn(4, INPUT_CHANNELS, 15, 15)
        logits, value = net(x)
        moves = torch.tensor([10, 50, 100, 200])
        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])
        mcts = torch.zeros(4, 225)
        mcts[0, 10] = 0.7
        mcts[0, 20] = 0.3
        mcts[1, 50] = 1.0
        # samples 2,3 have zero mcts_policy → use CE
        loss, p_loss, v_loss = total_loss(
            logits, value, moves, outcomes, net, mcts_policy=mcts
        )
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_kl_divergence_nonnegative(self):
        from gomonova.nn.losses import policy_kl_divergence
        logits = torch.randn(8, 225)
        target = torch.softmax(torch.randn(8, 225), dim=1)
        kl = policy_kl_divergence(logits, target)
        assert kl.item() >= -1e-6
