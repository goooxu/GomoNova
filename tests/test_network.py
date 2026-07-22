import torch
import pytest

from gomonova.game.board import Board, rc_to_pos
from gomonova.nn.encoder import board_to_planes, planes_to_tensor
from gomonova.nn.network import GomoNovaNet
from gomonova.nn.losses import total_loss


class TestEncoder:
    def test_empty_board_planes(self):
        b = Board()
        planes = board_to_planes(b)
        assert planes.shape == (8, 15, 15)
        assert planes[:7].sum() == 0.0
        assert planes[7].sum() == 225.0

    def test_planes_after_moves(self):
        b = Board()
        b.play(rc_to_pos(7, 7))
        b.play(rc_to_pos(7, 8))
        planes = board_to_planes(b)
        assert planes[0, 7, 7] == 1.0
        assert planes[1, 7, 8] == 1.0
        assert planes[2, 7, 7] == 1.0
        assert planes[4, 7, 8] == 1.0
        assert planes[6, 7, 7] == 1.0
        assert planes[6, 7, 8] == 1.0
        assert planes[7].sum() == 225.0

    def test_white_turn_bias(self):
        b = Board()
        b.play(rc_to_pos(7, 7))
        planes = board_to_planes(b)
        assert planes[7].sum() == 0.0


class TestNetwork:
    @pytest.fixture
    def small_net(self):
        return GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)

    def test_output_shapes(self, small_net):
        x = torch.randn(4, 8, 15, 15)
        policy, value = small_net(x)
        assert policy.shape == (4, 225)
        assert value.shape == (4, 1)

    def test_value_range(self, small_net):
        x = torch.randn(8, 8, 15, 15)
        _, value = small_net(x)
        assert (value >= -1.0).all() and (value <= 1.0).all()

    def test_deterministic(self, small_net):
        small_net.eval()
        x = torch.randn(2, 8, 15, 15)
        p1, v1 = small_net(x)
        p2, v2 = small_net(x)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_param_count_small(self, small_net):
        n = small_net.num_params()
        assert 100_000 < n < 2_000_000

    def test_param_count_main(self):
        net = GomoNovaNet(channels=128, num_blocks=12, policy_channels=64, value_channels=32)
        n = net.num_params()
        assert 5_000_000 < n < 50_000_000, f"Expected 5-50M params, got {n:,}"

    def test_gradient_flow(self, small_net):
        x = torch.randn(2, 8, 15, 15)
        policy, value = small_net(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, p in small_net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


class TestLosses:
    def test_total_loss_runs(self):
        net = GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)
        x = torch.randn(4, 8, 15, 15)
        logits, value = net(x)
        moves = torch.tensor([10, 50, 100, 200])
        outcomes = torch.tensor([1.0, -1.0, 0.0, 1.0])
        loss, p_loss, v_loss = total_loss(logits, value, moves, outcomes, net)
        assert loss.requires_grad
        assert not torch.isnan(loss)
