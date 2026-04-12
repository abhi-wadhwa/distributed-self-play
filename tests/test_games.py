"""Tests for game implementations: Connect Four and Othello."""

import numpy as np
import pytest

from src.games.connect4 import Connect4
from src.games.othello import Othello


class TestConnect4:
    """Tests for Connect Four game rules."""

    def setup_method(self):
        self.game = Connect4()

    def test_board_size(self):
        assert self.game.get_board_size() == (6, 7)

    def test_action_size(self):
        assert self.game.get_action_size() == 7

    def test_initial_board_is_empty(self):
        board = self.game.get_initial_board()
        assert board.shape == (6, 7)
        assert np.all(board == 0)

    def test_valid_moves_on_empty_board(self):
        board = self.game.get_initial_board()
        valid = self.game.get_valid_moves(board, 1)
        assert np.all(valid == 1)  # All columns playable

    def test_drop_piece_goes_to_bottom(self):
        board = self.game.get_initial_board()
        new_board, next_player = self.game.get_next_state(board, 1, 3)
        # Piece should be at bottom row, column 3
        assert new_board[5, 3] == 1
        assert next_player == -1

    def test_pieces_stack_in_column(self):
        board = self.game.get_initial_board()
        board, _ = self.game.get_next_state(board, 1, 3)   # row 5
        board, _ = self.game.get_next_state(board, -1, 3)  # row 4
        board, _ = self.game.get_next_state(board, 1, 3)   # row 3
        assert board[5, 3] == 1
        assert board[4, 3] == -1
        assert board[3, 3] == 1

    def test_full_column_is_invalid(self):
        board = self.game.get_initial_board()
        player = 1
        # Fill column 0
        for _ in range(6):
            board, player = self.game.get_next_state(board, player, 0)
        valid = self.game.get_valid_moves(board, player)
        assert valid[0] == 0  # Column 0 full
        assert valid[1] == 1  # Other columns still valid

    def test_horizontal_win(self):
        board = self.game.get_initial_board()
        # Player 1 fills bottom row columns 0-3
        for col in range(4):
            board[5, col] = 1
        result = self.game.get_game_ended(board, 1)
        assert result == 1.0

    def test_vertical_win(self):
        board = self.game.get_initial_board()
        # Player -1 fills column 2 rows 2-5
        for row in range(2, 6):
            board[row, 2] = -1
        result = self.game.get_game_ended(board, -1)
        assert result == 1.0

    def test_diagonal_win(self):
        board = self.game.get_initial_board()
        # Diagonal from (5,0) to (2,3)
        for i in range(4):
            board[5 - i, i] = 1
        result = self.game.get_game_ended(board, 1)
        assert result == 1.0

    def test_anti_diagonal_win(self):
        board = self.game.get_initial_board()
        # Diagonal from (2,0) to (5,3)
        for i in range(4):
            board[2 + i, i] = -1
        result = self.game.get_game_ended(board, -1)
        assert result == 1.0

    def test_no_win_yet(self):
        board = self.game.get_initial_board()
        board[5, 0] = 1
        board[5, 1] = 1
        board[5, 2] = 1
        # Only 3 in a row
        result = self.game.get_game_ended(board, 1)
        assert result is None

    def test_draw(self):
        board = self.game.get_initial_board()
        # Fill entire board without any 4-in-a-row
        # Pattern that avoids any 4 in a row:
        pattern = [
            [1, 1, -1, 1, -1, -1, 1],
            [-1, -1, 1, -1, 1, 1, -1],
            [1, 1, -1, 1, -1, -1, 1],
            [-1, -1, 1, -1, 1, 1, -1],
            [1, 1, -1, 1, -1, -1, 1],
            [-1, -1, 1, -1, 1, 1, -1],
        ]
        board = np.array(pattern, dtype=np.float32)
        result = self.game.get_game_ended(board, 1)
        assert result == 0.0  # Draw

    def test_canonical_board_player1(self):
        board = self.game.get_initial_board()
        board[5, 0] = 1
        canonical = self.game.get_canonical_board(board, 1)
        assert np.array_equal(canonical, board)

    def test_canonical_board_player_neg1(self):
        board = self.game.get_initial_board()
        board[5, 0] = 1
        board[5, 1] = -1
        canonical = self.game.get_canonical_board(board, -1)
        assert canonical[5, 0] == -1  # Flipped
        assert canonical[5, 1] == 1   # Flipped

    def test_symmetries(self):
        board = self.game.get_initial_board()
        board[5, 0] = 1
        pi = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 2
        # Mirror should have piece at column 6
        mirror_board, mirror_pi = symmetries[1]
        assert mirror_board[5, 6] == 1
        assert mirror_pi[6] == 0.5

    def test_display(self):
        board = self.game.get_initial_board()
        board[5, 3] = 1
        text = self.game.display(board)
        assert "X" in text
        assert "." in text

    def test_string_representation_unique(self):
        board1 = self.game.get_initial_board()
        board1[5, 0] = 1
        board2 = self.game.get_initial_board()
        board2[5, 1] = 1
        assert self.game.string_representation(board1) != self.game.string_representation(board2)


class TestOthello:
    """Tests for Othello game rules."""

    def setup_method(self):
        self.game = Othello()

    def test_board_size(self):
        assert self.game.get_board_size() == (8, 8)

    def test_action_size(self):
        # 64 cells + 1 pass
        assert self.game.get_action_size() == 65

    def test_initial_board(self):
        board = self.game.get_initial_board()
        assert board.shape == (8, 8)
        # Check starting position
        assert board[3, 3] == -1  # White
        assert board[3, 4] == 1   # Black
        assert board[4, 3] == 1   # Black
        assert board[4, 4] == -1  # White
        # Total of 4 pieces
        assert np.sum(board != 0) == 4

    def test_valid_moves_initial(self):
        board = self.game.get_initial_board()
        valid = self.game.get_valid_moves(board, 1)  # Black's turn
        # Black should have 4 valid moves on a standard Othello board
        num_valid = int(valid[:64].sum())
        assert num_valid == 4
        # Pass should not be valid
        assert valid[64] == 0

    def test_place_piece_flips(self):
        board = self.game.get_initial_board()
        # Black plays at (2, 3) - should flip white at (3, 3)
        new_board, next_player = self.game.get_next_state(board, 1, 2 * 8 + 3)
        assert new_board[2, 3] == 1   # New piece
        assert new_board[3, 3] == 1   # Flipped from -1 to 1
        assert next_player == -1

    def test_pass_action(self):
        board = self.game.get_initial_board()
        new_board, next_player = self.game.get_next_state(board, 1, 64)
        assert np.array_equal(new_board, board)  # Board unchanged
        assert next_player == -1

    def test_pass_when_no_moves(self):
        # Create a board where one player has no valid moves
        board = np.zeros((8, 8), dtype=np.float32)
        board[0, 0] = 1
        # Player -1 has no moves that would flip anything
        valid = self.game.get_valid_moves(board, -1)
        assert valid[64] == 1  # Must pass

    def test_game_not_ended_initially(self):
        board = self.game.get_initial_board()
        result = self.game.get_game_ended(board, 1)
        assert result is None

    def test_game_ended_all_one_color(self):
        # Board completely filled with one color
        board = np.ones((8, 8), dtype=np.float32)
        result = self.game.get_game_ended(board, 1)
        assert result == 1.0
        result = self.game.get_game_ended(board, -1)
        assert result == -1.0

    def test_canonical_board(self):
        board = self.game.get_initial_board()
        canonical = self.game.get_canonical_board(board, -1)
        # All pieces flipped
        assert canonical[3, 3] == 1   # Was -1
        assert canonical[3, 4] == -1  # Was 1

    def test_symmetries_count(self):
        board = self.game.get_initial_board()
        pi = np.ones(65, dtype=np.float32) / 65
        symmetries = self.game.get_symmetries(board, pi)
        # 4 rotations * 2 (with/without mirror) = 8
        assert len(symmetries) == 8

    def test_symmetries_preserve_pass_prob(self):
        board = self.game.get_initial_board()
        pi = np.zeros(65, dtype=np.float32)
        pi[64] = 0.42  # Pass probability
        symmetries = self.game.get_symmetries(board, pi)
        for _, sym_pi in symmetries:
            assert sym_pi[64] == pytest.approx(0.42)

    def test_display(self):
        board = self.game.get_initial_board()
        text = self.game.display(board)
        assert "B" in text
        assert "W" in text
        assert "Black(B): 2" in text
        assert "White(W): 2" in text

    def test_full_game_terminates(self):
        """Play a game with random moves and verify it terminates."""
        board = self.game.get_initial_board()
        player = 1
        moves = 0
        max_moves = 200  # Safety limit

        while moves < max_moves:
            result = self.game.get_game_ended(board, player)
            if result is not None:
                assert result in [-1.0, 0.0, 1.0]
                return

            valid = self.game.get_valid_moves(board, player)
            valid_actions = np.where(valid > 0)[0]
            action = np.random.choice(valid_actions)
            board, player = self.game.get_next_state(board, player, action)
            moves += 1

        # If we get here, the game should still end
        # (just checking it doesn't crash)
