"""
Monte Carlo Tree Search (MCTS) for self-play.

Implements PUCT-based selection (as in AlphaZero) using the neural network
to provide prior probabilities and value estimates.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.games.game_base import GameBase
from src.network import DualHeadedNet, board_to_tensor


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ["visit_count", "value_sum", "prior", "children", "is_expanded"]

    def __init__(self, prior: float = 0.0) -> None:
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT formula for selection:
        U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        score  = Q(s,a) + U(s,a)

    Args:
        game: game instance implementing GameBase.
        network: neural network for policy and value prediction.
        num_simulations: number of MCTS simulations per move (default 100).
        c_puct: exploration constant (default 1.5).
        dirichlet_alpha: Dirichlet noise parameter for root (default 0.3).
        dirichlet_epsilon: fraction of noise to mix in at root (default 0.25).
        device: torch device string.
    """

    def __init__(
        self,
        game: GameBase,
        network: DualHeadedNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    @torch.no_grad()
    def _evaluate(
        self, board: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Use the neural network to evaluate a position.

        Returns:
            (policy, value): policy is a probability distribution over actions,
                             value is a scalar in [-1, 1].
        """
        tensor = board_to_tensor(board, self.device)
        log_policy, value = self.network(tensor)
        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        v = value.item()
        return policy, v

    def _expand(
        self, node: MCTSNode, board: np.ndarray, player: int
    ) -> float:
        """Expand a leaf node using the network."""
        canonical = self.game.get_canonical_board(board, player)
        policy, value = self._evaluate(canonical)

        # Mask invalid moves and renormalize
        valid_moves = self.game.get_valid_moves(board, player)
        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback: uniform over valid moves
            policy = valid_moves / valid_moves.sum()

        # Create child nodes
        for action in range(self.game.get_action_size()):
            if valid_moves[action] > 0:
                node.children[action] = MCTSNode(prior=float(policy[action]))

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select the child with the highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_child: Optional[MCTSNode] = None

        sqrt_parent = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            # PUCT formula
            u = self.c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count)
            score = child.q_value + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None
        return best_action, best_child

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root node priors for exploration."""
        actions = list(node.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for i, action in enumerate(actions):
            node.children[action].prior = (
                (1.0 - eps) * node.children[action].prior + eps * noise[i]
            )

    def search(
        self,
        board: np.ndarray,
        player: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS from the given position and return the visit-count policy.

        Args:
            board: current board state.
            player: current player (1 or -1).
            temperature: controls exploration in the final move selection.
                         0 = greedy (pick most visited), >0 = proportional to visit counts.
            add_noise: whether to add Dirichlet noise at the root.

        Returns:
            policy: np.ndarray of shape (action_size,) representing the move probabilities.
        """
        root = MCTSNode()
        self._expand(root, board, player)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            current_board = board.copy()
            current_player = player
            search_path = [node]

            # Selection: traverse tree using PUCT until we hit an unexpanded node
            while node.is_expanded and node.children:
                action, node = self._select_child(node)
                current_board, current_player = self.game.get_next_state(
                    current_board, current_player, action
                )
                search_path.append(node)

            # Check terminal state
            result = self.game.get_game_ended(current_board, current_player)
            if result is not None:
                value = result
            else:
                # Expansion + evaluation
                value = self._expand(node, current_board, current_player)

            # Backpropagation (flip value at each level)
            for i, path_node in enumerate(reversed(search_path)):
                # The value is from the perspective of the player at the leaf.
                # We alternate signs as we go up the tree.
                v = value if i % 2 == 0 else -value
                path_node.visit_count += 1
                path_node.value_sum += v

        # Build the policy from visit counts
        action_size = self.game.get_action_size()
        visits = np.zeros(action_size, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature == 0:
            # Greedy: pick the most visited action
            policy = np.zeros(action_size, dtype=np.float64)
            best = int(np.argmax(visits))
            policy[best] = 1.0
        else:
            # Proportional to visit counts raised to 1/temperature
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                policy = visits_temp / total
            else:
                policy = np.ones(action_size, dtype=np.float64) / action_size

        return policy.astype(np.float32)
