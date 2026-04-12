"""Tests for the learner process, using MockRedisInterface."""

import numpy as np
import pytest
import torch

from src.communication import MockRedisInterface
from src.games.connect4 import Connect4
from src.learner import Learner
from src.network import create_network


class TestLearner:
    """Tests for the training Learner."""

    def setup_method(self):
        self.game = Connect4()
        self.comm = MockRedisInterface()
        self.board_size = self.game.get_board_size()
        self.action_size = self.game.get_action_size()

    def _create_learner(self, **kwargs) -> Learner:
        defaults = {
            "game_name": "connect4",
            "board_size": self.board_size,
            "action_size": self.action_size,
            "comm": self.comm,
            "num_channels": 16,
            "num_res_blocks": 1,
            "batch_size": 8,
            "min_buffer_size": 8,
            "publish_interval": 5,
            "checkpoint_interval": 10,
            "checkpoint_dir": "test_checkpoints",
        }
        defaults.update(kwargs)
        return Learner(**defaults)

    def _push_fake_experience(self, n: int = 20):
        """Push fake experience data to MockRedis."""
        exps = []
        for _ in range(n):
            exps.append({
                "board": np.random.randn(*self.board_size).astype(np.float32),
                "policy": np.random.dirichlet(np.ones(self.action_size)).astype(np.float32),
                "value": np.random.choice([-1.0, 0.0, 1.0]),
                "generation": 0,
            })
        self.comm.push_experience(exps)

    def test_learner_creation(self):
        learner = self._create_learner()
        assert learner.training_step == 0
        assert learner.generation == 0
        assert len(learner.buffer) == 0

    def test_publish_initial_weights(self):
        learner = self._create_learner()
        learner.publish_weights()
        assert learner.generation == 1
        weights = self.comm.get_latest_weights()
        assert weights is not None
        assert weights["generation"] == 1

    def test_pull_experience(self):
        learner = self._create_learner()
        self._push_fake_experience(15)

        pulled = learner.pull_experience()
        assert pulled == 15
        assert len(learner.buffer) == 15

    def test_train_step(self):
        learner = self._create_learner(batch_size=4, min_buffer_size=4)

        # Fill buffer directly
        for _ in range(10):
            learner.buffer.add(
                board=np.random.randn(*self.board_size).astype(np.float32),
                policy=np.random.dirichlet(np.ones(self.action_size)).astype(np.float32),
                value=np.random.choice([-1.0, 0.0, 1.0]),
            )

        losses = learner.train_step()
        assert "total_loss" in losses
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert losses["total_loss"] > 0
        assert learner.training_step == 1

    def test_run_iteration_waits_for_buffer(self):
        learner = self._create_learner(min_buffer_size=100)
        result = learner.run_iteration()
        assert result is None  # Not enough data

    def test_run_iteration_trains(self):
        learner = self._create_learner(
            batch_size=4,
            min_buffer_size=4,
            publish_interval=100,
            checkpoint_interval=100,
        )
        self._push_fake_experience(20)

        result = learner.run_iteration()
        assert result is not None
        assert result["training_step"] == 1
        assert "total_loss" in result

    def test_weight_publishing_on_interval(self):
        learner = self._create_learner(
            batch_size=4,
            min_buffer_size=4,
            publish_interval=2,
            checkpoint_interval=1000,
        )

        # Fill buffer
        for _ in range(20):
            learner.buffer.add(
                board=np.random.randn(*self.board_size).astype(np.float32),
                policy=np.random.dirichlet(np.ones(self.action_size)).astype(np.float32),
                value=np.random.choice([-1.0, 0.0, 1.0]),
            )

        # Train 2 steps (publish_interval=2)
        learner.train_step()
        assert learner.generation == 0  # Not published yet
        learner.training_step = 0  # Reset for run_iteration test

        for _ in range(2):
            learner.run_iteration()

        assert learner.generation == 1  # Published after step 2

    def test_metrics_pushed(self):
        learner = self._create_learner(
            batch_size=4,
            min_buffer_size=4,
            publish_interval=1000,
            checkpoint_interval=1000,
        )
        self._push_fake_experience(20)
        learner.run_iteration()

        metrics = self.comm.get_metrics()
        assert len(metrics) >= 1
        assert "total_loss" in metrics[-1]

    def test_priorities_updated_after_training(self):
        learner = self._create_learner(batch_size=4, min_buffer_size=4)

        for _ in range(10):
            learner.buffer.add(
                board=np.random.randn(*self.board_size).astype(np.float32),
                policy=np.random.dirichlet(np.ones(self.action_size)).astype(np.float32),
                value=np.random.choice([-1.0, 0.0, 1.0]),
            )

        # Record initial priorities
        initial_priorities = [exp.priority for exp in learner.buffer._buffer]

        learner.train_step()

        # Some priorities should have been updated
        current_priorities = [exp.priority for exp in learner.buffer._buffer]
        # At least the sampled items should have new priorities
        assert initial_priorities != current_priorities


class TestWeightRoundtrip:
    """Test that model weights survive serialization through MockRedis."""

    def test_weights_roundtrip(self):
        game = Connect4()
        comm = MockRedisInterface()

        net1 = create_network(
            board_size=game.get_board_size(),
            action_size=game.get_action_size(),
            num_channels=16,
            num_res_blocks=1,
        )

        # Set specific parameter values
        with torch.no_grad():
            for p in net1.parameters():
                p.fill_(0.42)

        state_dict = {k: v.cpu() for k, v in net1.state_dict().items()}
        comm.publish_weights(state_dict, generation=10)

        # Load into a new network
        net2 = create_network(
            board_size=game.get_board_size(),
            action_size=game.get_action_size(),
            num_channels=16,
            num_res_blocks=1,
        )

        data = comm.get_latest_weights()
        assert data is not None
        assert data["generation"] == 10
        net2.load_state_dict(data["state_dict"])

        # Verify weights match
        for (name1, p1), (name2, p2) in zip(
            net1.named_parameters(), net2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2), f"Mismatch in {name1}"
