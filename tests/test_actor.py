"""Tests for the actor process, using MockRedisInterface to avoid real Redis."""

import numpy as np
import pytest
import torch

from src.actor import Actor
from src.communication import MockRedisInterface
from src.games.connect4 import Connect4
from src.network import create_network


class TestActor:
    """Tests for the self-play Actor."""

    def setup_method(self):
        self.game = Connect4()
        self.comm = MockRedisInterface()
        self.network_config = {
            "board_size": self.game.get_board_size(),
            "action_size": self.game.get_action_size(),
            "num_channels": 16,   # Small for speed
            "num_res_blocks": 1,  # Small for speed
        }

    def test_actor_creation(self):
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
            actor_id=42,
        )
        assert actor.actor_id == 42
        assert actor.generation == 0
        assert actor.games_played == 0

    def test_play_game_produces_experiences(self):
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        experiences = actor.play_game()

        assert len(experiences) > 0
        for exp in experiences:
            assert "board" in exp
            assert "policy" in exp
            assert "value" in exp
            assert "generation" in exp
            assert exp["board"].shape == (6, 7)
            assert exp["policy"].shape == (7,)
            assert exp["value"] in [-1.0, 0.0, 1.0]

    def test_experience_values_consistent(self):
        """All experiences from one game should have values from {-1, 0, 1}."""
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        experiences = actor.play_game()
        values = {exp["value"] for exp in experiences}
        assert values.issubset({-1.0, 0.0, 1.0})

    def test_run_episode_pushes_to_comm(self):
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        stats = actor.run_episode()

        assert stats["games_played"] == 1
        assert stats["num_experiences"] > 0
        assert stats["actor_id"] == 0

        # Check that experiences were pushed to the mock Redis
        queue_size = self.comm.experience_queue_size()
        assert queue_size == stats["num_experiences"]

    def test_load_weights_from_comm(self):
        # Create and publish weights
        net = create_network(
            board_size=self.game.get_board_size(),
            action_size=self.game.get_action_size(),
            num_channels=16,
            num_res_blocks=1,
        )
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        self.comm.publish_weights(state_dict, generation=5)

        # Actor should load them
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        loaded = actor.load_latest_weights()
        assert loaded
        assert actor.generation == 5

    def test_no_reload_same_generation(self):
        net = create_network(
            board_size=self.game.get_board_size(),
            action_size=self.game.get_action_size(),
            num_channels=16,
            num_res_blocks=1,
        )
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        self.comm.publish_weights(state_dict, generation=3)

        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        assert actor.load_latest_weights()  # First load
        assert not actor.load_latest_weights()  # Same gen, skip

    def test_symmetries_in_experience(self):
        """Connect4 has mirror symmetry, so experience count should be even."""
        actor = Actor(
            game=self.game,
            network_config=self.network_config,
            comm=self.comm,
            mcts_simulations=5,
        )
        experiences = actor.play_game()
        # Each move produces 2 experiences (original + mirror)
        assert len(experiences) % 2 == 0


class TestExperienceSerialization:
    """Test that experience can be serialized and deserialized through MockRedis."""

    def test_roundtrip(self):
        comm = MockRedisInterface()
        board = np.random.randn(6, 7).astype(np.float32)
        policy = np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05], dtype=np.float32)

        exp = {
            "board": board,
            "policy": policy,
            "value": 1.0,
            "generation": 7,
        }
        comm.push_experience([exp])

        retrieved = comm.pull_experience(1)
        assert len(retrieved) == 1
        r = retrieved[0]
        np.testing.assert_allclose(r["board"], board, atol=1e-5)
        np.testing.assert_allclose(r["policy"], policy, atol=1e-5)
        assert r["value"] == 1.0
        assert r["generation"] == 7

    def test_multiple_experiences(self):
        comm = MockRedisInterface()
        exps = []
        for i in range(10):
            exps.append({
                "board": np.full((6, 7), float(i), dtype=np.float32),
                "policy": np.ones(7, dtype=np.float32) / 7,
                "value": float(i % 2),
                "generation": i,
            })
        comm.push_experience(exps)

        assert comm.experience_queue_size() == 10

        batch = comm.pull_experience(5)
        assert len(batch) == 5
        assert comm.experience_queue_size() == 5

        batch2 = comm.pull_experience(10)
        assert len(batch2) == 5  # Only 5 remaining
        assert comm.experience_queue_size() == 0
