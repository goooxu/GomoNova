import numpy as np
import pytest

from gomonova.game.board import BOARD_SIZE, NUM_CELLS
from gomonova.game.symmetry import (
    NUM_TRANSFORMS,
    TRANSFORMS,
    inverse_transform,
    transform_board,
    transform_policy,
    transform_pos,
)


class TestTransforms:
    def test_eight_transforms(self):
        assert NUM_TRANSFORMS == 8

    def test_all_are_permutations(self):
        for i, t in enumerate(TRANSFORMS):
            assert len(t) == NUM_CELLS
            assert set(t.tolist()) == set(range(NUM_CELLS)), f"Transform {i} is not a valid permutation"

    def test_identity(self):
        for pos in range(NUM_CELLS):
            assert transform_pos(pos, 0) == pos

    def test_inverse_roundtrip(self):
        for t in range(NUM_TRANSFORMS):
            inv = inverse_transform(t)
            for pos in range(NUM_CELLS):
                assert transform_pos(transform_pos(pos, t), inv) == pos

    def test_closure(self):
        """Composing any two transforms should yield another transform in the group."""
        for i in range(NUM_TRANSFORMS):
            for j in range(NUM_TRANSFORMS):
                composed = TRANSFORMS[j][TRANSFORMS[i]]
                found = any(np.array_equal(composed, TRANSFORMS[k]) for k in range(NUM_TRANSFORMS))
                assert found, f"Composition of {i} and {j} not in group"

    def test_rot90_center(self):
        center = 7 * BOARD_SIZE + 7
        assert transform_pos(center, 1) == center

    def test_rot90_corner(self):
        top_left = 0
        assert transform_pos(top_left, 1) == (BOARD_SIZE - 1)


class TestPolicyTransform:
    def test_policy_roundtrip(self):
        rng = np.random.default_rng(42)
        policy = rng.random(NUM_CELLS)
        policy /= policy.sum()

        for t in range(NUM_TRANSFORMS):
            inv = inverse_transform(t)
            transformed = transform_policy(policy, t)
            restored = transform_policy(transformed, inv)
            np.testing.assert_allclose(restored, policy, atol=1e-12)

    def test_policy_preserves_sum(self):
        rng = np.random.default_rng(123)
        policy = rng.random(NUM_CELLS)
        policy /= policy.sum()

        for t in range(NUM_TRANSFORMS):
            transformed = transform_policy(policy, t)
            assert abs(transformed.sum() - 1.0) < 1e-10


class TestBoardTransform:
    def test_board_identity(self):
        rng = np.random.default_rng(7)
        board = rng.integers(0, 3, size=(BOARD_SIZE, BOARD_SIZE))
        result = transform_board(board, 0)
        np.testing.assert_array_equal(result, board)

    def test_board_rot90(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[0, 0] = 1
        result = transform_board(board, 1)
        assert result[0, BOARD_SIZE - 1] == 1

    def test_board_flip_horizontal(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[3, 5] = 2
        result = transform_board(board, 4)
        assert result[3, BOARD_SIZE - 1 - 5] == 2
