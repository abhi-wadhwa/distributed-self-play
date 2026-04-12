"""
Self-play actor process.

Runs self-play games using the latest model weights retrieved from Redis.
Generates training experience (board, policy, value) and pushes it to the
experience queue for the learner.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.communication import MockRedisInterface, RedisInterface
from src.games.game_base import GameBase
from src.mcts import MCTS
from src.network import DualHeadedNet, create_network

logger = logging.getLogger(__name__)


class Actor:
    """
    Self-play actor that generates training data.

    The actor continuously:
      1. Checks for new model weights from the learner.
      2. Plays a game of self-play using MCTS.
      3. Sends the resulting experience to the replay buffer via Redis.

    Args:
        game: game instance to play.
        network_config: dict with keys board_size, action_size, num_channels, num_res_blocks.
        comm: Redis communication interface.
        mcts_simulations: number of MCTS simulations per move.
        temperature_threshold: number of moves before switching from exploratory
                               (temperature=1) to greedy (temperature~0).
        device: torch device.
        actor_id: unique identifier for this actor.
    """

    def __init__(
        self,
        game: GameBase,
        network_config: Dict[str, Any],
        comm: Union[RedisInterface, MockRedisInterface],
        mcts_simulations: int = 100,
        temperature_threshold: int = 15,
        device: str = "cpu",
        actor_id: int = 0,
    ) -> None:
        self.game = game
        self.comm = comm
        self.mcts_simulations = mcts_simulations
        self.temperature_threshold = temperature_threshold
        self.device = device
        self.actor_id = actor_id
        self.generation = 0
        self.games_played = 0

        # Create the network
        self.network = create_network(
            board_size=network_config["board_size"],
            action_size=network_config["action_size"],
            num_channels=network_config.get("num_channels", 128),
            num_res_blocks=network_config.get("num_res_blocks", 8),
            device=device,
        )
        self.network.eval()

    def load_latest_weights(self) -> bool:
        """
        Attempt to load the latest model weights from Redis.

        Returns:
            True if weights were loaded, False if no new weights available.
        """
        data = self.comm.get_latest_weights()
        if data is None:
            return False

        new_gen = data["generation"]
        if new_gen <= self.generation:
            return False

        self.network.load_state_dict(data["state_dict"])
        self.network.eval()
        self.generation = new_gen
        logger.info(f"Actor {self.actor_id}: loaded weights gen={new_gen}")
        return True

    def play_game(self) -> List[Dict[str, Any]]:
        """
        Play a single game of self-play using MCTS.

        Returns:
            List of experience dicts, each with keys:
                board: canonical board state (np.ndarray)
                policy: MCTS visit-count policy (np.ndarray)
                value: game outcome from current player's perspective (float)
                generation: model generation that produced this data (int)
        """
        mcts = MCTS(
            game=self.game,
            network=self.network,
            num_simulations=self.mcts_simulations,
            device=self.device,
        )

        board = self.game.get_initial_board()
        player = 1
        move_number = 0
        history: List[Tuple[np.ndarray, int, np.ndarray]] = []

        while True:
            # Choose temperature based on move number
            temperature = 1.0 if move_number < self.temperature_threshold else 0.1

            # Run MCTS to get policy
            canonical = self.game.get_canonical_board(board, player)
            policy = mcts.search(board, player, temperature=temperature)

            # Store (canonical_board, current_player, policy) for training
            history.append((canonical.copy(), player, policy.copy()))

            # Sample action from policy
            action = int(np.random.choice(len(policy), p=policy))
            board, player = self.game.get_next_state(board, player, action)
            move_number += 1

            # Check for game end
            result = self.game.get_game_ended(board, player)
            if result is not None:
                break

        # Build training examples
        # result is from the perspective of `player` (the next player to move
        # when the game ended), so we flip signs accordingly.
        experiences: List[Dict[str, Any]] = []
        for canonical_board, hist_player, pi in history:
            # value from the perspective of hist_player
            # If hist_player == player, value = result
            # If hist_player != player, value = -result
            if hist_player == player:
                value = float(result)
            else:
                value = -float(result)

            # Apply symmetries for data augmentation
            symmetries = self.game.get_symmetries(canonical_board, pi)
            for sym_board, sym_pi in symmetries:
                experiences.append({
                    "board": sym_board,
                    "policy": sym_pi,
                    "value": value,
                    "generation": self.generation,
                })

        self.games_played += 1
        return experiences

    def run_episode(self) -> Dict[str, Any]:
        """
        Run one complete episode: check for new weights, play a game,
        and push the experience.

        Returns:
            Stats dict with keys: games_played, num_experiences, generation.
        """
        # Check for new weights
        self.load_latest_weights()

        # Play game
        start = time.time()
        experiences = self.play_game()
        duration = time.time() - start

        # Push experience to Redis
        self.comm.push_experience(experiences)

        stats = {
            "actor_id": self.actor_id,
            "games_played": self.games_played,
            "num_experiences": len(experiences),
            "generation": self.generation,
            "game_duration_s": duration,
        }
        logger.info(
            f"Actor {self.actor_id}: game {self.games_played} done, "
            f"{len(experiences)} experiences, {duration:.1f}s"
        )
        return stats

    def run(self, num_games: int = 0) -> None:
        """
        Main actor loop. Plays games continuously.

        Args:
            num_games: number of games to play (0 = infinite).
        """
        logger.info(f"Actor {self.actor_id} starting (device={self.device})")
        count = 0
        while True:
            self.run_episode()
            count += 1
            if num_games > 0 and count >= num_games:
                break
