"""Tests for the prioritized experience replay buffer."""

import numpy as np
import pytest

from src.replay_buffer import Experience, PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def setup_method(self):
        self.buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    def _make_board(self, value: float = 0.0) -> np.ndarray:
        """Create a simple 6x7 board for testing."""
        board = np.zeros((6, 7), dtype=np.float32)
        board[0, 0] = value
        return board

    def _make_policy(self, action_size: int = 7) -> np.ndarray:
        """Create a uniform policy."""
        return np.ones(action_size, dtype=np.float32) / action_size

    def test_empty_buffer(self):
        assert len(self.buffer) == 0
        assert not self.buffer.is_full

    def test_add_single(self):
        self.buffer.add(self._make_board(), self._make_policy(), 1.0)
        assert len(self.buffer) == 1

    def test_add_batch(self):
        boards = [self._make_board(i) for i in range(5)]
        policies = [self._make_policy() for _ in range(5)]
        values = [1.0, -1.0, 0.0, 1.0, -1.0]
        self.buffer.add_batch(boards, policies, values, generation=1)
        assert len(self.buffer) == 5

    def test_capacity_limit(self):
        buffer = PrioritizedReplayBuffer(capacity=10)
        for i in range(20):
            buffer.add(self._make_board(i), self._make_policy(), 1.0)
        assert len(buffer) == 10
        assert buffer.is_full

    def test_circular_overwrite(self):
        buffer = PrioritizedReplayBuffer(capacity=3)
        for i in range(5):
            buffer.add(self._make_board(float(i)), self._make_policy(), float(i))
        assert len(buffer) == 3
        # The buffer should contain the last 3 entries (wrapped around)
        values = [exp.value for exp in buffer._buffer]
        # After 5 inserts into capacity 3:
        # positions: 0->exp0, 1->exp1, 2->exp2, 0->exp3, 1->exp4
        # So buffer should have: [exp3, exp4, exp2]
        assert 2.0 in values
        assert 3.0 in values
        assert 4.0 in values

    def test_sample_returns_correct_shapes(self):
        for i in range(20):
            self.buffer.add(self._make_board(i), self._make_policy(), 1.0)

        boards, policies, values, indices = self.buffer.sample(5)
        assert boards.shape == (5, 6, 7)
        assert policies.shape == (5, 7)
        assert values.shape == (5,)
        assert indices.shape == (5,)

    def test_sample_batch_size_capped(self):
        for i in range(3):
            self.buffer.add(self._make_board(i), self._make_policy(), 1.0)

        boards, policies, values, indices = self.buffer.sample(10)
        assert boards.shape[0] == 3  # Only 3 available

    def test_sample_empty_raises(self):
        with pytest.raises(ValueError):
            self.buffer.sample(1)

    def test_priority_affects_sampling(self):
        """High-priority items should be sampled more frequently."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # Add 10 low-priority items
        for i in range(10):
            buffer.add(
                self._make_board(0),
                self._make_policy(),
                0.0,
                priority=0.01,
            )

        # Add 1 high-priority item
        high_board = self._make_board(99.0)
        buffer.add(high_board, self._make_policy(), 1.0, priority=100.0)

        # Sample many times and count how often the high-priority item appears
        high_count = 0
        num_samples = 500
        for _ in range(num_samples):
            boards, _, _, _ = buffer.sample(1)
            if boards[0, 0, 0] == 99.0:
                high_count += 1

        # With alpha=1.0, the high-priority item should appear very frequently
        assert high_count > num_samples * 0.5

    def test_update_priorities(self):
        for i in range(5):
            self.buffer.add(self._make_board(i), self._make_policy(), 1.0)

        indices = np.array([0, 2, 4])
        new_priorities = np.array([10.0, 20.0, 30.0])
        self.buffer.update_priorities(indices, new_priorities)

        assert self.buffer._buffer[0].priority == 10.0
        assert self.buffer._buffer[2].priority == 20.0
        assert self.buffer._buffer[4].priority == 30.0

    def test_min_priority_enforced(self):
        buffer = PrioritizedReplayBuffer(capacity=10, min_priority=0.1)
        buffer.add(self._make_board(), self._make_policy(), 1.0, priority=0.0001)
        assert buffer._buffer[0].priority >= 0.1

    def test_new_items_get_max_priority(self):
        # Add item with high priority
        self.buffer.add(
            self._make_board(), self._make_policy(), 1.0, priority=50.0
        )
        # Add item without specifying priority
        self.buffer.add(self._make_board(), self._make_policy(), 1.0)
        # New item should get the max priority
        assert self.buffer._buffer[1].priority == 50.0

    def test_clear(self):
        for i in range(5):
            self.buffer.add(self._make_board(i), self._make_policy(), 1.0)
        assert len(self.buffer) == 5
        self.buffer.clear()
        assert len(self.buffer) == 0
        assert not self.buffer.is_full

    def test_get_stats(self):
        for i in range(5):
            self.buffer.add(
                self._make_board(i), self._make_policy(), 1.0, generation=i % 3
            )
        stats = self.buffer.get_stats()
        assert stats["size"] == 5
        assert stats["capacity"] == 100
        assert stats["utilization"] == pytest.approx(0.05)
        assert stats["mean_priority"] > 0
        assert stats["max_priority"] > 0
        assert sorted(stats["generations"]) == [0, 1, 2]

    def test_get_stats_empty(self):
        stats = self.buffer.get_stats()
        assert stats["size"] == 0
        assert stats["mean_priority"] == 0.0

    def test_generation_tracking(self):
        self.buffer.add(self._make_board(), self._make_policy(), 1.0, generation=5)
        assert self.buffer._buffer[0].generation == 5


class TestExperience:
    """Tests for the Experience dataclass."""

    def test_creation(self):
        exp = Experience(
            board=np.zeros((6, 7), dtype=np.float32),
            policy=np.ones(7, dtype=np.float32) / 7,
            value=1.0,
            priority=0.5,
            generation=3,
        )
        assert exp.value == 1.0
        assert exp.priority == 0.5
        assert exp.generation == 3

    def test_default_priority(self):
        exp = Experience(
            board=np.zeros((6, 7), dtype=np.float32),
            policy=np.ones(7, dtype=np.float32) / 7,
            value=0.0,
        )
        assert exp.priority == 1.0
        assert exp.generation == 0
