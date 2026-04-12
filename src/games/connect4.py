"""Connect Four game implementation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.games.game_base import GameBase


class Connect4(GameBase):
    """
    Connect Four on a 6x7 board.

    Players drop discs into one of 7 columns. First to connect 4 in a row
    (horizontal, vertical, or diagonal) wins. The board is stored as a
    6x7 numpy array where 1 = player 1, -1 = player 2, 0 = empty.
    """

    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4

    def get_board_size(self) -> Tuple[int, int]:
        return (self.ROWS, self.COLS)

    def get_action_size(self) -> int:
        return self.COLS

    def get_initial_board(self) -> np.ndarray:
        return np.zeros((self.ROWS, self.COLS), dtype=np.float32)

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> Tuple[np.ndarray, int]:
        new_board = board.copy()
        # Find the lowest empty row in the chosen column
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row, action] == 0:
                new_board[row, action] = player
                break
        return new_board, -player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        valid = np.zeros(self.COLS, dtype=np.float32)
        for col in range(self.COLS):
            if board[0, col] == 0:  # Top row empty means column is playable
                valid[col] = 1.0
        return valid

    def _check_win(self, board: np.ndarray, player: int) -> bool:
        """Check if `player` has 4 in a row on the board."""
        rows, cols = self.ROWS, self.COLS
        w = self.WIN_LENGTH

        # Horizontal
        for r in range(rows):
            for c in range(cols - w + 1):
                if all(board[r, c + i] == player for i in range(w)):
                    return True

        # Vertical
        for r in range(rows - w + 1):
            for c in range(cols):
                if all(board[r + i, c] == player for i in range(w)):
                    return True

        # Diagonal (top-left to bottom-right)
        for r in range(rows - w + 1):
            for c in range(cols - w + 1):
                if all(board[r + i, c + i] == player for i in range(w)):
                    return True

        # Diagonal (bottom-left to top-right)
        for r in range(w - 1, rows):
            for c in range(cols - w + 1):
                if all(board[r - i, c + i] == player for i in range(w)):
                    return True

        return False

    def get_game_ended(self, board: np.ndarray, player: int) -> Optional[float]:
        if self._check_win(board, player):
            return 1.0
        if self._check_win(board, -player):
            return -1.0
        # Check for draw (board full)
        if np.all(board != 0):
            return 0.0
        return None

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        return board * player

    def get_symmetries(
        self, board: np.ndarray, pi: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Connect Four has left-right mirror symmetry."""
        return [
            (board, pi),
            (np.fliplr(board), pi[::-1]),
        ]

    def display(self, board: np.ndarray) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for row in range(self.ROWS):
            line = " ".join(symbols[int(board[row, col])] for col in range(self.COLS))
            lines.append(line)
        lines.append(" ".join(str(c) for c in range(self.COLS)))
        return "\n".join(lines)
