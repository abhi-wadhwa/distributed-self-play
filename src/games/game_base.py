"""Abstract base class for two-player board games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class GameBase(ABC):
    """
    Abstract base class defining the interface for two-player zero-sum board games.

    Players are represented as 1 (first player) and -1 (second player).
    The board state is a numpy array suitable for neural network input.
    """

    @abstractmethod
    def get_board_size(self) -> Tuple[int, int]:
        """Return (rows, cols) of the board."""

    @abstractmethod
    def get_action_size(self) -> int:
        """Return the total number of possible actions (including illegal ones)."""

    @abstractmethod
    def get_initial_board(self) -> np.ndarray:
        """Return the initial (empty) board state."""

    @abstractmethod
    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> Tuple[np.ndarray, int]:
        """
        Apply action for player on board.

        Returns:
            (new_board, next_player): The resulting board and the next player to move.
        """

    @abstractmethod
    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return a binary mask of valid moves for the given player.

        Returns:
            np.ndarray of shape (action_size,) with 1 for valid moves, 0 otherwise.
        """

    @abstractmethod
    def get_game_ended(self, board: np.ndarray, player: int) -> Optional[float]:
        """
        Check if the game has ended from the perspective of `player`.

        Returns:
            None if game is not over.
            1.0 if `player` has won.
            -1.0 if `player` has lost.
            0.0 for a draw.
        """

    @abstractmethod
    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return the board from the perspective of the given player.
        For player 1 this is the board as-is; for player -1 the board
        is flipped so the current player's pieces are always represented as 1.
        """

    @abstractmethod
    def get_symmetries(
        self, board: np.ndarray, pi: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return a list of (board, policy) tuples representing symmetrically
        equivalent positions. Used for data augmentation during training.
        """

    def string_representation(self, board: np.ndarray) -> str:
        """Return a hashable string representation of the board for MCTS caching."""
        return board.tobytes().hex()

    @abstractmethod
    def display(self, board: np.ndarray) -> str:
        """Return a human-readable string representation of the board."""
