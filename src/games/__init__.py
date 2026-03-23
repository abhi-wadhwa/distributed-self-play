"""Game implementations for self-play training."""

from src.games.game_base import GameBase
from src.games.connect4 import Connect4
from src.games.othello import Othello

__all__ = ["GameBase", "Connect4", "Othello"]
