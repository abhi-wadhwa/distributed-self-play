"""
Model versioning and ELO tracking for self-play training.

Maintains a league of past model versions, tracks ELO ratings,
and manages checkpoint save/load to disk.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.network import DualHeadedNet

logger = logging.getLogger(__name__)

# ELO constants
ELO_K_FACTOR = 32.0
ELO_INITIAL = 1200.0


@dataclass
class ModelVersion:
    """Metadata for a saved model checkpoint."""

    generation: int
    elo: float = ELO_INITIAL
    games_played: int = 0
    training_steps: int = 0
    timestamp: float = field(default_factory=time.time)
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "elo": self.elo,
            "games_played": self.games_played,
            "training_steps": self.training_steps,
            "timestamp": self.timestamp,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelVersion:
        return cls(**data)


class EloTracker:
    """
    Tracks ELO ratings across model generations.

    Uses the standard ELO formula:
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))
        R_a_new = R_a + K * (S_a - E_a)

    where S_a is the actual score (1 for win, 0.5 for draw, 0 for loss).
    """

    def __init__(self, k_factor: float = ELO_K_FACTOR) -> None:
        self.k_factor = k_factor
        self.ratings: Dict[int, float] = {}

    def get_rating(self, generation: int) -> float:
        """Get ELO rating for a generation, initializing if necessary."""
        if generation not in self.ratings:
            self.ratings[generation] = ELO_INITIAL
        return self.ratings[generation]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update(
        self,
        gen_a: int,
        gen_b: int,
        score_a: float,
    ) -> Tuple[float, float]:
        """
        Update ELO ratings after a match.

        Args:
            gen_a: generation of player A.
            gen_b: generation of player B.
            score_a: actual score for player A (1.0 = win, 0.5 = draw, 0.0 = loss).

        Returns:
            (new_elo_a, new_elo_b): updated ELO ratings.
        """
        rating_a = self.get_rating(gen_a)
        rating_b = self.get_rating(gen_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        score_b = 1.0 - score_a

        new_a = rating_a + self.k_factor * (score_a - expected_a)
        new_b = rating_b + self.k_factor * (score_b - expected_b)

        self.ratings[gen_a] = new_a
        self.ratings[gen_b] = new_b

        return new_a, new_b

    def get_leaderboard(self) -> List[Tuple[int, float]]:
        """Return generations sorted by ELO rating (highest first)."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


class ModelVersionManager:
    """
    Manages model checkpoints on disk with versioning and ELO tracking.

    Directory structure:
        checkpoint_dir/
            versions.json          # metadata for all versions
            gen_000001.pt          # model weights
            gen_000002.pt
            ...

    Args:
        checkpoint_dir: path to directory for storing checkpoints.
        keep_last_n: number of recent checkpoints to keep on disk (0 = keep all).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.elo_tracker = EloTracker()
        self.versions: Dict[int, ModelVersion] = {}
        self._load_metadata()

    def _metadata_path(self) -> Path:
        return self.checkpoint_dir / "versions.json"

    def _checkpoint_path(self, generation: int) -> Path:
        return self.checkpoint_dir / f"gen_{generation:06d}.pt"

    def _load_metadata(self) -> None:
        """Load version metadata from disk."""
        meta_path = self._metadata_path()
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            for entry in data.get("versions", []):
                mv = ModelVersion.from_dict(entry)
                self.versions[mv.generation] = mv
                self.elo_tracker.ratings[mv.generation] = mv.elo
            logger.info(f"Loaded metadata for {len(self.versions)} model versions")

    def _save_metadata(self) -> None:
        """Save version metadata to disk."""
        data = {
            "versions": [v.to_dict() for v in self.versions.values()],
        }
        with open(self._metadata_path(), "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        network: DualHeadedNet,
        generation: int,
        training_steps: int = 0,
        games_played: int = 0,
    ) -> ModelVersion:
        """
        Save a model checkpoint and register the version.

        Args:
            network: the neural network to save.
            generation: generation number for this checkpoint.
            training_steps: total training steps so far.
            games_played: total games played so far.

        Returns:
            The created ModelVersion.
        """
        path = self._checkpoint_path(generation)
        torch.save({
            "state_dict": network.state_dict(),
            "generation": generation,
            "training_steps": training_steps,
            "games_played": games_played,
        }, path)

        version = ModelVersion(
            generation=generation,
            elo=self.elo_tracker.get_rating(generation),
            games_played=games_played,
            training_steps=training_steps,
            path=str(path),
        )
        self.versions[generation] = version
        self._save_metadata()

        # Cleanup old checkpoints if configured
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        logger.info(
            f"Saved checkpoint gen={generation}, steps={training_steps}, "
            f"games={games_played}, elo={version.elo:.1f}"
        )
        return version

    def load_checkpoint(
        self, network: DualHeadedNet, generation: int
    ) -> Optional[ModelVersion]:
        """
        Load a checkpoint into the network.

        Args:
            network: the network to load weights into.
            generation: generation number to load.

        Returns:
            ModelVersion if successful, None otherwise.
        """
        path = self._checkpoint_path(generation)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return None

        data = torch.load(path, map_location="cpu", weights_only=False)
        network.load_state_dict(data["state_dict"])
        logger.info(f"Loaded checkpoint generation {generation}")
        return self.versions.get(generation)

    def get_latest_generation(self) -> int:
        """Return the highest generation number, or 0 if no checkpoints exist."""
        if not self.versions:
            return 0
        return max(self.versions.keys())

    def get_version(self, generation: int) -> Optional[ModelVersion]:
        """Get metadata for a specific generation."""
        return self.versions.get(generation)

    def record_match(
        self, gen_a: int, gen_b: int, score_a: float
    ) -> Tuple[float, float]:
        """
        Record a match result and update ELO ratings.

        Returns:
            (new_elo_a, new_elo_b): updated ELO ratings.
        """
        new_a, new_b = self.elo_tracker.update(gen_a, gen_b, score_a)
        if gen_a in self.versions:
            self.versions[gen_a].elo = new_a
        if gen_b in self.versions:
            self.versions[gen_b].elo = new_b
        self._save_metadata()
        return new_a, new_b

    def get_opponent_generations(
        self, current_gen: int, num_opponents: int = 3
    ) -> List[int]:
        """
        Select opponent generations for evaluation.

        Returns a mix of recent and historically strong opponents.

        Args:
            current_gen: the current model generation.
            num_opponents: number of opponents to select.

        Returns:
            List of generation numbers.
        """
        available = [g for g in self.versions.keys() if g != current_gen]
        if not available:
            return []

        opponents = []

        # Always include the most recent previous generation
        recent = max(available)
        opponents.append(recent)

        # Add top-rated opponents
        leaderboard = self.elo_tracker.get_leaderboard()
        for gen, elo in leaderboard:
            if gen != current_gen and gen not in opponents:
                opponents.append(gen)
            if len(opponents) >= num_opponents:
                break

        return opponents[:num_opponents]

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint files, keeping only the most recent N."""
        generations = sorted(self.versions.keys())
        if len(generations) <= self.keep_last_n:
            return

        to_remove = generations[:-self.keep_last_n]
        for gen in to_remove:
            path = self._checkpoint_path(gen)
            if path.exists():
                path.unlink()
                logger.info(f"Removed old checkpoint: {path}")
            # Keep the metadata entry but clear the path
            if gen in self.versions:
                self.versions[gen].path = ""
        self._save_metadata()
