"""
Evaluator process for distributed self-play.

Tests new checkpoints against previous model versions and baselines.
Updates ELO ratings and determines if the new model is stronger.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.communication import MockRedisInterface, RedisInterface
from src.games.game_base import GameBase
from src.mcts import MCTS
from src.model_version import ModelVersionManager
from src.network import DualHeadedNet, create_network

logger = logging.getLogger(__name__)


class RandomPlayer:
    """Baseline player that selects uniformly from valid moves."""

    def __init__(self, game: GameBase) -> None:
        self.game = game

    def get_action(self, board: np.ndarray, player: int) -> int:
        valid = self.game.get_valid_moves(board, player)
        valid_actions = np.where(valid > 0)[0]
        return int(np.random.choice(valid_actions))


class MCTSPlayer:
    """Player that uses MCTS with a neural network for move selection."""

    def __init__(
        self,
        game: GameBase,
        network: DualHeadedNet,
        num_simulations: int = 50,
        device: str = "cpu",
    ) -> None:
        self.game = game
        self.network = network
        self.mcts = MCTS(
            game=game,
            network=network,
            num_simulations=num_simulations,
            device=device,
        )

    def get_action(self, board: np.ndarray, player: int) -> int:
        policy = self.mcts.search(board, player, temperature=0.0, add_noise=False)
        return int(np.argmax(policy))


def play_match(
    game: GameBase,
    player1_fn,
    player2_fn,
) -> float:
    """
    Play a single game between two players.

    Args:
        game: game instance.
        player1_fn: callable(board, player) -> action for player 1.
        player2_fn: callable(board, player) -> action for player -1.

    Returns:
        Result from player 1's perspective: 1.0 (win), 0.0 (loss), 0.5 (draw).
    """
    board = game.get_initial_board()
    player = 1

    while True:
        if player == 1:
            action = player1_fn(board, player)
        else:
            action = player2_fn(board, player)

        board, player = game.get_next_state(board, player, action)
        result = game.get_game_ended(board, player)

        if result is not None:
            # result is from the perspective of the current player (who would move next)
            # We need to convert to player 1's perspective
            if player == 1:
                # Player 1 is about to move, result is from their perspective
                if result == 1.0:
                    return 1.0
                elif result == -1.0:
                    return 0.0
                else:
                    return 0.5
            else:
                # Player -1 is about to move, result is from their perspective
                if result == 1.0:
                    return 0.0  # player -1 won = player 1 lost
                elif result == -1.0:
                    return 1.0  # player -1 lost = player 1 won
                else:
                    return 0.5


class Evaluator:
    """
    Evaluates new model checkpoints against previous versions and baselines.

    Plays matches between the current model and opponents, tracking
    ELO ratings and win rates.

    Args:
        game: game instance.
        board_size: (rows, cols) of the game board.
        action_size: number of possible actions.
        comm: Redis communication interface.
        checkpoint_dir: path to model checkpoints.
        num_eval_games: number of games per evaluation matchup.
        mcts_simulations: MCTS simulations per move during evaluation.
        num_channels: network channels.
        num_res_blocks: network residual blocks.
        device: torch device.
    """

    def __init__(
        self,
        game: GameBase,
        board_size: Tuple[int, int],
        action_size: int,
        comm: Union[RedisInterface, MockRedisInterface],
        checkpoint_dir: str = "checkpoints",
        num_eval_games: int = 20,
        mcts_simulations: int = 50,
        num_channels: int = 128,
        num_res_blocks: int = 8,
        device: str = "cpu",
    ) -> None:
        self.game = game
        self.board_size = board_size
        self.action_size = action_size
        self.comm = comm
        self.num_eval_games = num_eval_games
        self.mcts_simulations = mcts_simulations
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.device = device

        self.version_manager = ModelVersionManager(checkpoint_dir=checkpoint_dir)
        self.random_player = RandomPlayer(game)
        self._last_evaluated_gen = 0

    def _create_network(self) -> DualHeadedNet:
        """Create a new network instance."""
        return create_network(
            board_size=self.board_size,
            action_size=self.action_size,
            num_channels=self.num_channels,
            num_res_blocks=self.num_res_blocks,
            device=self.device,
        )

    def evaluate_against_random(
        self, network: DualHeadedNet, generation: int
    ) -> Dict[str, float]:
        """
        Evaluate a model against the random baseline.

        Returns:
            Dict with win_rate, draw_rate, loss_rate.
        """
        network.eval()
        mcts_player = MCTSPlayer(
            self.game, network, self.mcts_simulations, self.device
        )

        wins = 0
        draws = 0
        losses = 0

        for i in range(self.num_eval_games):
            # Alternate colors
            if i % 2 == 0:
                score = play_match(
                    self.game,
                    mcts_player.get_action,
                    self.random_player.get_action,
                )
            else:
                score = 1.0 - play_match(
                    self.game,
                    self.random_player.get_action,
                    mcts_player.get_action,
                )

            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

        total = self.num_eval_games
        result = {
            "win_rate": wins / total,
            "draw_rate": draws / total,
            "loss_rate": losses / total,
        }
        logger.info(
            f"Gen {generation} vs Random: "
            f"W={result['win_rate']:.1%} D={result['draw_rate']:.1%} "
            f"L={result['loss_rate']:.1%}"
        )
        return result

    def evaluate_against_generation(
        self,
        current_net: DualHeadedNet,
        current_gen: int,
        opponent_gen: int,
    ) -> Dict[str, Any]:
        """
        Evaluate the current model against a specific past generation.

        Returns:
            Dict with win_rate, draw_rate, loss_rate, elo_current, elo_opponent.
        """
        opponent_net = self._create_network()
        version = self.version_manager.load_checkpoint(opponent_net, opponent_gen)
        if version is None:
            logger.warning(f"Cannot load opponent generation {opponent_gen}")
            return {}

        current_net.eval()
        opponent_net.eval()

        current_player = MCTSPlayer(
            self.game, current_net, self.mcts_simulations, self.device
        )
        opponent_player = MCTSPlayer(
            self.game, opponent_net, self.mcts_simulations, self.device
        )

        wins = 0
        draws = 0
        losses = 0

        for i in range(self.num_eval_games):
            if i % 2 == 0:
                score = play_match(
                    self.game,
                    current_player.get_action,
                    opponent_player.get_action,
                )
            else:
                score = 1.0 - play_match(
                    self.game,
                    opponent_player.get_action,
                    current_player.get_action,
                )

            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

            # Update ELO after each game
            game_score = score
            self.version_manager.record_match(current_gen, opponent_gen, game_score)

        total = self.num_eval_games
        result = {
            "opponent_gen": opponent_gen,
            "win_rate": wins / total,
            "draw_rate": draws / total,
            "loss_rate": losses / total,
            "elo_current": self.version_manager.elo_tracker.get_rating(current_gen),
            "elo_opponent": self.version_manager.elo_tracker.get_rating(opponent_gen),
        }
        logger.info(
            f"Gen {current_gen} vs Gen {opponent_gen}: "
            f"W={result['win_rate']:.1%} D={result['draw_rate']:.1%} "
            f"L={result['loss_rate']:.1%} "
            f"(ELO: {result['elo_current']:.0f} vs {result['elo_opponent']:.0f})"
        )
        return result

    def evaluate_generation(self, generation: int) -> Dict[str, Any]:
        """
        Run a full evaluation of a model generation.

        Evaluates against the random baseline and selected past versions.

        Returns:
            Dict with all evaluation results.
        """
        logger.info(f"Starting evaluation of generation {generation}")

        # Load the model
        network = self._create_network()
        loaded = self.version_manager.load_checkpoint(network, generation)
        if loaded is None:
            # Try loading from Redis
            data = self.comm.get_weights_by_generation(generation)
            if data is None:
                logger.warning(f"Cannot find weights for generation {generation}")
                return {}
            network.load_state_dict(data["state_dict"])

        results: Dict[str, Any] = {
            "generation": generation,
            "timestamp": time.time(),
        }

        # Evaluate vs random
        results["vs_random"] = self.evaluate_against_random(network, generation)

        # Evaluate vs past generations
        opponents = self.version_manager.get_opponent_generations(generation)
        results["vs_opponents"] = []
        for opp_gen in opponents:
            match_result = self.evaluate_against_generation(
                network, generation, opp_gen
            )
            if match_result:
                results["vs_opponents"].append(match_result)

        # Push ELO to Redis
        elo = self.version_manager.elo_tracker.get_rating(generation)
        self.comm.update_elo(generation, elo)
        results["elo"] = elo

        # Push metrics
        self.comm.push_metrics({
            "type": "evaluation",
            "generation": generation,
            "elo": elo,
            "vs_random_win_rate": results["vs_random"]["win_rate"],
        })

        return results

    def run(self, poll_interval: float = 10.0) -> None:
        """
        Main evaluator loop. Watches for new checkpoints and evaluates them.

        Args:
            poll_interval: seconds between polling for new checkpoints.
        """
        logger.info("Evaluator starting")

        while True:
            current_gen = self.comm.get_current_generation()

            if current_gen > self._last_evaluated_gen:
                self.evaluate_generation(current_gen)
                self._last_evaluated_gen = current_gen

            time.sleep(poll_interval)
