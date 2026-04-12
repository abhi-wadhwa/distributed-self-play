"""
Prioritized experience replay buffer.

Stores (board, policy, value) training examples with TD-error based
priorities. Supports circular capacity and weighted sampling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Experience:
    """A single training example from self-play."""

    board: np.ndarray          # canonical board state
    policy: np.ndarray         # MCTS visit-count policy
    value: float               # game outcome from this player's perspective
    priority: float = 1.0      # priority for sampling (TD-error based)
    generation: int = 0        # model generation that produced this experience


class PrioritizedReplayBuffer:
    """
    Circular replay buffer with priority-based sampling.

    Priorities determine sampling probability. Higher-priority experiences
    (those the network finds surprising, measured by TD-error) are sampled
    more frequently.

    Args:
        capacity: maximum number of experiences to store.
        alpha: priority exponent controlling how strongly priorities
               affect sampling probability. 0 = uniform, 1 = fully prioritized.
        min_priority: minimum priority value to avoid zero probabilities.
    """

    def __init__(
        self,
        capacity: int = 500_000,
        alpha: float = 0.6,
        min_priority: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.min_priority = min_priority
        self._buffer: List[Experience] = []
        self._position: int = 0
        self._max_priority: float = 1.0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def add(
        self,
        board: np.ndarray,
        policy: np.ndarray,
        value: float,
        generation: int = 0,
        priority: Optional[float] = None,
    ) -> None:
        """
        Add an experience to the buffer.

        New experiences get the maximum observed priority to ensure
        they are sampled at least once.
        """
        if priority is None:
            priority = self._max_priority

        priority = max(priority, self.min_priority)

        exp = Experience(
            board=board.copy(),
            policy=policy.copy(),
            value=value,
            priority=priority,
            generation=generation,
        )

        if len(self._buffer) < self.capacity:
            self._buffer.append(exp)
        else:
            self._buffer[self._position] = exp

        self._position = (self._position + 1) % self.capacity
        self._max_priority = max(self._max_priority, priority)

    def add_batch(
        self,
        boards: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[float],
        generation: int = 0,
    ) -> None:
        """Add a batch of experiences (e.g., from a complete game)."""
        for board, policy, value in zip(boards, policies, values):
            self.add(board, policy, value, generation=generation)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample a batch of experiences with priority-based weighting.

        Returns:
            (boards, policies, values, indices):
                boards: (batch_size, rows, cols)
                policies: (batch_size, action_size)
                values: (batch_size,)
                indices: (batch_size,) indices into the buffer for priority updates
        """
        n = len(self._buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        batch_size = min(batch_size, n)

        # Compute sampling probabilities from priorities
        priorities = np.array(
            [exp.priority for exp in self._buffer], dtype=np.float64
        )
        priorities = priorities ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices without replacement
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        boards = np.array([self._buffer[i].board for i in indices])
        policies = np.array([self._buffer[i].policy for i in indices])
        values = np.array([self._buffer[i].value for i in indices], dtype=np.float32)

        return boards, policies, values, indices

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences (e.g., using new TD-errors).

        Args:
            indices: indices into the buffer (from sample()).
            new_priorities: new priority values.
        """
        for idx, prio in zip(indices, new_priorities):
            if 0 <= idx < len(self._buffer):
                self._buffer[idx].priority = max(float(prio), self.min_priority)
                self._max_priority = max(self._max_priority, float(prio))

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._position = 0
        self._max_priority = 1.0

    def get_stats(self) -> dict:
        """Return buffer statistics."""
        if not self._buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "mean_priority": 0.0,
                "max_priority": 0.0,
                "generations": [],
            }

        priorities = [exp.priority for exp in self._buffer]
        generations = list({exp.generation for exp in self._buffer})
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "utilization": len(self._buffer) / self.capacity,
            "mean_priority": float(np.mean(priorities)),
            "max_priority": float(np.max(priorities)),
            "generations": sorted(generations),
        }
