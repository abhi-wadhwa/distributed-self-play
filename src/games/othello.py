"""Othello (Reversi) game implementation on an 8x8 board."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.games.game_base import GameBase

# Eight directions: (row_delta, col_delta)
_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Othello(GameBase):
    """
    Othello (Reversi) on an 8x8 board.

    Board encoding: 1 = black (first player), -1 = white, 0 = empty.
    Action space is 64 (one per cell) + 1 pass action = 65.
    """

    SIZE = 8
    PASS_ACTION = 64  # action index for passing

    def get_board_size(self) -> Tuple[int, int]:
        return (self.SIZE, self.SIZE)

    def get_action_size(self) -> int:
        return self.SIZE * self.SIZE + 1  # 64 cells + 1 pass

    def get_initial_board(self) -> np.ndarray:
        board = np.zeros((self.SIZE, self.SIZE), dtype=np.float32)
        mid = self.SIZE // 2
        board[mid - 1, mid - 1] = -1
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        board[mid, mid] = -1
        return board

    def _get_flips(
        self, board: np.ndarray, row: int, col: int, player: int
    ) -> List[Tuple[int, int]]:
        """Return a list of (r, c) positions that would be flipped by placing at (row, col)."""
        if board[row, col] != 0:
            return []

        flips: List[Tuple[int, int]] = []
        for dr, dc in _DIRECTIONS:
            r, c = row + dr, col + dc
            line: List[Tuple[int, int]] = []
            while 0 <= r < self.SIZE and 0 <= c < self.SIZE and board[r, c] == -player:
                line.append((r, c))
                r += dr
                c += dc
            # Valid flip requires at least one opponent piece followed by own piece
            if line and 0 <= r < self.SIZE and 0 <= c < self.SIZE and board[r, c] == player:
                flips.extend(line)
        return flips

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> Tuple[np.ndarray, int]:
        new_board = board.copy()

        if action == self.PASS_ACTION:
            return new_board, -player

        row, col = divmod(action, self.SIZE)
        flips = self._get_flips(new_board, row, col, player)
        new_board[row, col] = player
        for r, c in flips:
            new_board[r, c] = player

        return new_board, -player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        valid = np.zeros(self.get_action_size(), dtype=np.float32)

        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self._get_flips(board, row, col, player):
                    valid[row * self.SIZE + col] = 1.0

        # If no moves, the player must pass
        if valid.sum() == 0:
            valid[self.PASS_ACTION] = 1.0

        return valid

    def get_game_ended(self, board: np.ndarray, player: int) -> Optional[float]:
        # Game ends when neither player can move
        p1_moves = np.zeros(self.get_action_size(), dtype=np.float32)
        p2_moves = np.zeros(self.get_action_size(), dtype=np.float32)

        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self._get_flips(board, row, col, 1):
                    p1_moves[row * self.SIZE + col] = 1.0
                if self._get_flips(board, row, col, -1):
                    p2_moves[row * self.SIZE + col] = 1.0

        # If either player has real moves (not just pass), game continues
        if p1_moves.sum() > 0 or p2_moves.sum() > 0:
            return None

        # Count pieces
        player_count = np.sum(board == player)
        opponent_count = np.sum(board == -player)

        if player_count > opponent_count:
            return 1.0
        elif player_count < opponent_count:
            return -1.0
        else:
            return 0.0

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        return board * player

    def get_symmetries(
        self, board: np.ndarray, pi: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Othello has 8-fold symmetry (4 rotations x 2 reflections).
        We reshape the policy (minus the pass action) to match board shape,
        apply transforms, then flatten back.
        """
        pi_board = pi[:-1].reshape(self.SIZE, self.SIZE)
        pass_prob = pi[-1]
        symmetries = []

        for rot in range(4):
            rotated_board = np.rot90(board, rot)
            rotated_pi = np.rot90(pi_board, rot)
            symmetries.append((
                rotated_board.copy(),
                np.append(rotated_pi.flatten(), pass_prob),
            ))
            # Mirror
            flipped_board = np.fliplr(rotated_board)
            flipped_pi = np.fliplr(rotated_pi)
            symmetries.append((
                flipped_board.copy(),
                np.append(flipped_pi.flatten(), pass_prob),
            ))

        return symmetries

    def display(self, board: np.ndarray) -> str:
        symbols = {0: ".", 1: "B", -1: "W"}
        lines = ["  " + " ".join(str(c) for c in range(self.SIZE))]
        for row in range(self.SIZE):
            line = str(row) + " " + " ".join(
                symbols[int(board[row, col])] for col in range(self.SIZE)
            )
            lines.append(line)
        black = int(np.sum(board == 1))
        white = int(np.sum(board == -1))
        lines.append(f"Black(B): {black}  White(W): {white}")
        return "\n".join(lines)
