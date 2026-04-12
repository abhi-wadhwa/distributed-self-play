"""
Learner process for distributed self-play training.

Pulls experience from the replay buffer, trains the neural network,
and publishes updated weights to Redis for actors to consume.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.communication import MockRedisInterface, RedisInterface
from src.model_version import ModelVersionManager
from src.network import DualHeadedNet, create_network
from src.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class Learner:
    """
    Central learner that trains the shared neural network.

    The learner continuously:
      1. Pulls experience from Redis into the local replay buffer.
      2. Samples batches from the replay buffer.
      3. Updates the network parameters.
      4. Publishes new weights to Redis.
      5. Periodically saves checkpoints and updates metrics.

    Args:
        game_name: name of the game (for logging).
        board_size: (rows, cols) of the game board.
        action_size: number of possible actions.
        comm: Redis communication interface.
        checkpoint_dir: path to directory for saving checkpoints.
        num_channels: CNN channels in the network.
        num_res_blocks: number of residual blocks.
        batch_size: training batch size.
        learning_rate: optimizer learning rate.
        weight_decay: L2 regularization strength.
        buffer_capacity: maximum replay buffer size.
        min_buffer_size: minimum experiences before training starts.
        publish_interval: publish weights every N training steps.
        checkpoint_interval: save checkpoint every N training steps.
        pull_batch_size: number of experiences to pull from Redis per iteration.
        device: torch device.
    """

    def __init__(
        self,
        game_name: str,
        board_size: Tuple[int, int],
        action_size: int,
        comm: Union[RedisInterface, MockRedisInterface],
        checkpoint_dir: str = "checkpoints",
        num_channels: int = 128,
        num_res_blocks: int = 8,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 500_000,
        min_buffer_size: int = 1000,
        publish_interval: int = 100,
        checkpoint_interval: int = 1000,
        pull_batch_size: int = 512,
        device: str = "cpu",
    ) -> None:
        self.game_name = game_name
        self.comm = comm
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.publish_interval = publish_interval
        self.checkpoint_interval = checkpoint_interval
        self.pull_batch_size = pull_batch_size
        self.device = device

        # Network
        self.network = create_network(
            board_size=board_size,
            action_size=action_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            device=device,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

        # Model versioning
        self.version_manager = ModelVersionManager(checkpoint_dir=checkpoint_dir)

        # Training state
        self.generation = 0
        self.training_step = 0
        self.total_games = 0
        self.total_loss_history: list = []

    def pull_experience(self) -> int:
        """
        Pull experience from Redis into the local replay buffer.

        Returns:
            Number of experiences pulled.
        """
        experiences = self.comm.pull_experience(self.pull_batch_size)
        for exp in experiences:
            self.buffer.add(
                board=exp["board"],
                policy=exp["policy"],
                value=exp["value"],
                generation=exp["generation"],
            )
        return len(experiences)

    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.

        Returns:
            Dict with loss components: total_loss, policy_loss, value_loss.
        """
        self.network.train()

        # Sample from buffer
        boards, policies, values, indices = self.buffer.sample(self.batch_size)

        # Convert to tensors
        board_tensor = torch.from_numpy(boards).float().unsqueeze(1).to(self.device)
        policy_tensor = torch.from_numpy(policies).float().to(self.device)
        value_tensor = torch.from_numpy(values).float().unsqueeze(1).to(self.device)

        # Forward pass
        log_policy, predicted_value = self.network(board_tensor)

        # Policy loss: cross-entropy with MCTS policy
        policy_loss = -torch.sum(policy_tensor * log_policy) / policy_tensor.size(0)

        # Value loss: MSE between predicted and actual value
        value_loss = nn.functional.mse_loss(predicted_value, value_tensor)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities based on prediction error (TD-error proxy)
        with torch.no_grad():
            td_errors = torch.abs(predicted_value.squeeze() - value_tensor.squeeze())
            new_priorities = td_errors.cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        self.training_step += 1

        losses = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
        return losses

    def publish_weights(self) -> None:
        """Publish current model weights to Redis."""
        self.generation += 1
        state_dict = self.network.state_dict()
        # Move state_dict to CPU before publishing
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        self.comm.publish_weights(cpu_state_dict, self.generation)
        logger.info(f"Published weights generation {self.generation}")

    def save_checkpoint(self) -> None:
        """Save a model checkpoint to disk."""
        self.version_manager.save_checkpoint(
            network=self.network,
            generation=self.generation,
            training_steps=self.training_step,
            games_played=self.total_games,
        )

    def run_iteration(self) -> Optional[Dict[str, Any]]:
        """
        Run a single training iteration:
          1. Pull experience from Redis.
          2. If buffer is large enough, train.
          3. Periodically publish weights and save checkpoints.

        Returns:
            Training stats dict, or None if buffer is too small.
        """
        # Pull new experience
        num_pulled = self.pull_experience()

        # Wait for enough experience
        if len(self.buffer) < self.min_buffer_size:
            logger.info(
                f"Buffer size {len(self.buffer)} < {self.min_buffer_size}, "
                f"waiting for more experience..."
            )
            return None

        # Train
        losses = self.train_step()

        # Publish weights periodically
        if self.training_step % self.publish_interval == 0:
            self.publish_weights()

        # Save checkpoint periodically
        if self.training_step % self.checkpoint_interval == 0:
            self.save_checkpoint()

        # Push metrics
        metrics = {
            "training_step": self.training_step,
            "generation": self.generation,
            "buffer_size": len(self.buffer),
            "experiences_pulled": num_pulled,
            **losses,
        }
        self.comm.push_metrics(metrics)

        if self.training_step % 50 == 0:
            logger.info(
                f"Step {self.training_step}: loss={losses['total_loss']:.4f} "
                f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}), "
                f"buffer={len(self.buffer)}, gen={self.generation}"
            )

        return metrics

    def run(self, num_steps: int = 0, poll_interval: float = 0.1) -> None:
        """
        Main learner loop.

        Args:
            num_steps: number of training steps (0 = infinite).
            poll_interval: seconds to wait between iterations when buffer is empty.
        """
        logger.info(
            f"Learner starting (device={self.device}, game={self.game_name})"
        )

        # Publish initial weights so actors can start
        self.publish_weights()

        while True:
            result = self.run_iteration()
            if result is None:
                time.sleep(poll_interval)
                continue

            if num_steps > 0 and self.training_step >= num_steps:
                logger.info(f"Reached {num_steps} training steps, stopping.")
                self.save_checkpoint()
                break
