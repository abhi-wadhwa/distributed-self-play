"""
Redis communication interface for distributed self-play.

Provides abstractions for:
  - Publishing and subscribing to model weight updates (pub/sub)
  - Pushing and pulling experience data (lists/queues)
  - Storing and retrieving model checkpoints
  - Tracking training metrics
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import redis
import torch

logger = logging.getLogger(__name__)

# Redis key prefixes
WEIGHT_CHANNEL = "selfplay:weights"
EXPERIENCE_QUEUE = "selfplay:experience"
METRICS_KEY = "selfplay:metrics"
CHECKPOINT_PREFIX = "selfplay:checkpoint:"
GENERATION_KEY = "selfplay:generation"
ELO_KEY = "selfplay:elo"


class RedisInterface:
    """
    Redis-based communication layer for distributed self-play.

    Handles serialization/deserialization of model weights and
    experience data between actors and learners.

    Args:
        host: Redis server hostname.
        port: Redis server port.
        db: Redis database number.
        password: optional Redis password.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ) -> None:
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=False
        )
        self._pubsub: Optional[redis.client.PubSub] = None

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    # ---- Model Weights ----

    def publish_weights(self, state_dict: dict, generation: int) -> None:
        """
        Serialize model weights and publish to the weight channel.

        Args:
            state_dict: PyTorch model state dict.
            generation: model generation number.
        """
        buf = io.BytesIO()
        torch.save({"state_dict": state_dict, "generation": generation}, buf)
        payload = buf.getvalue()

        # Store as latest checkpoint
        self.client.set(f"{CHECKPOINT_PREFIX}latest", payload)
        self.client.set(f"{CHECKPOINT_PREFIX}{generation}", payload)
        self.client.set(GENERATION_KEY, str(generation).encode())

        # Publish notification (payload is too large for pub/sub, send gen only)
        self.client.publish(WEIGHT_CHANNEL, str(generation).encode())
        logger.info(f"Published weights for generation {generation}")

    def get_latest_weights(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest model weights from Redis.

        Returns:
            dict with 'state_dict' and 'generation', or None if not available.
        """
        payload = self.client.get(f"{CHECKPOINT_PREFIX}latest")
        if payload is None:
            return None
        buf = io.BytesIO(payload)
        return torch.load(buf, map_location="cpu", weights_only=False)

    def get_weights_by_generation(self, generation: int) -> Optional[Dict[str, Any]]:
        """Retrieve model weights for a specific generation."""
        payload = self.client.get(f"{CHECKPOINT_PREFIX}{generation}")
        if payload is None:
            return None
        buf = io.BytesIO(payload)
        return torch.load(buf, map_location="cpu", weights_only=False)

    def get_current_generation(self) -> int:
        """Get the current model generation number."""
        val = self.client.get(GENERATION_KEY)
        if val is None:
            return 0
        return int(val)

    def subscribe_weights(self, callback: Callable[[int], None]) -> None:
        """
        Subscribe to weight update notifications.

        Args:
            callback: function called with the new generation number
                      when weights are updated.
        """
        self._pubsub = self.client.pubsub()
        self._pubsub.subscribe(WEIGHT_CHANNEL)
        logger.info("Subscribed to weight updates")

        for message in self._pubsub.listen():
            if message["type"] == "message":
                generation = int(message["data"])
                callback(generation)

    # ---- Experience Data ----

    def push_experience(self, experience_batch: List[Dict[str, Any]]) -> None:
        """
        Push a batch of experience tuples to the experience queue.

        Each experience is a dict with keys: board, policy, value, generation.
        """
        for exp in experience_batch:
            serialized = pickle.dumps({
                "board": exp["board"].tolist(),
                "policy": exp["policy"].tolist(),
                "value": float(exp["value"]),
                "generation": int(exp.get("generation", 0)),
            })
            self.client.rpush(EXPERIENCE_QUEUE, serialized)

    def pull_experience(self, batch_size: int, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Pull a batch of experiences from the queue.

        Args:
            batch_size: number of experiences to pull.
            timeout: max seconds to wait for each experience.

        Returns:
            List of experience dicts.
        """
        experiences = []
        for _ in range(batch_size):
            result = self.client.lpop(EXPERIENCE_QUEUE)
            if result is None:
                break
            data = pickle.loads(result)
            experiences.append({
                "board": np.array(data["board"], dtype=np.float32),
                "policy": np.array(data["policy"], dtype=np.float32),
                "value": data["value"],
                "generation": data["generation"],
            })
        return experiences

    def experience_queue_size(self) -> int:
        """Return the number of experiences waiting in the queue."""
        return self.client.llen(EXPERIENCE_QUEUE)

    # ---- Metrics ----

    def push_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Push training metrics to Redis (stored as a time-series list).

        Args:
            metrics: dict of metric values (loss, games_played, etc.).
        """
        metrics["timestamp"] = time.time()
        self.client.rpush(METRICS_KEY, json.dumps(metrics).encode())
        # Keep only the last 10,000 entries
        self.client.ltrim(METRICS_KEY, -10000, -1)

    def get_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Retrieve the latest metrics entries."""
        raw = self.client.lrange(METRICS_KEY, -count, -1)
        return [json.loads(entry) for entry in raw]

    # ---- ELO Tracking ----

    def update_elo(self, generation: int, elo: float) -> None:
        """Store ELO rating for a model generation."""
        self.client.hset(ELO_KEY, str(generation), str(elo))

    def get_elo_history(self) -> Dict[int, float]:
        """Get all ELO ratings."""
        raw = self.client.hgetall(ELO_KEY)
        return {int(k): float(v) for k, v in raw.items()}

    def close(self) -> None:
        """Close the Redis connection."""
        if self._pubsub:
            self._pubsub.close()
        self.client.close()


class MockRedisInterface:
    """
    In-memory mock of RedisInterface for testing without a Redis server.

    Implements the same API using Python data structures.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lists: Dict[str, List[bytes]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._generation: int = 0

    def ping(self) -> bool:
        return True

    def publish_weights(self, state_dict: dict, generation: int) -> None:
        buf = io.BytesIO()
        torch.save({"state_dict": state_dict, "generation": generation}, buf)
        payload = buf.getvalue()
        self._store[f"{CHECKPOINT_PREFIX}latest"] = payload
        self._store[f"{CHECKPOINT_PREFIX}{generation}"] = payload
        self._generation = generation

    def get_latest_weights(self) -> Optional[Dict[str, Any]]:
        payload = self._store.get(f"{CHECKPOINT_PREFIX}latest")
        if payload is None:
            return None
        buf = io.BytesIO(payload)
        return torch.load(buf, map_location="cpu", weights_only=False)

    def get_weights_by_generation(self, generation: int) -> Optional[Dict[str, Any]]:
        payload = self._store.get(f"{CHECKPOINT_PREFIX}{generation}")
        if payload is None:
            return None
        buf = io.BytesIO(payload)
        return torch.load(buf, map_location="cpu", weights_only=False)

    def get_current_generation(self) -> int:
        return self._generation

    def push_experience(self, experience_batch: List[Dict[str, Any]]) -> None:
        if EXPERIENCE_QUEUE not in self._lists:
            self._lists[EXPERIENCE_QUEUE] = []
        for exp in experience_batch:
            serialized = pickle.dumps({
                "board": exp["board"].tolist(),
                "policy": exp["policy"].tolist(),
                "value": float(exp["value"]),
                "generation": int(exp.get("generation", 0)),
            })
            self._lists[EXPERIENCE_QUEUE].append(serialized)

    def pull_experience(self, batch_size: int, timeout: float = 1.0) -> List[Dict[str, Any]]:
        queue = self._lists.get(EXPERIENCE_QUEUE, [])
        experiences = []
        for _ in range(batch_size):
            if not queue:
                break
            data = pickle.loads(queue.pop(0))
            experiences.append({
                "board": np.array(data["board"], dtype=np.float32),
                "policy": np.array(data["policy"], dtype=np.float32),
                "value": data["value"],
                "generation": data["generation"],
            })
        return experiences

    def experience_queue_size(self) -> int:
        return len(self._lists.get(EXPERIENCE_QUEUE, []))

    def push_metrics(self, metrics: Dict[str, Any]) -> None:
        if METRICS_KEY not in self._lists:
            self._lists[METRICS_KEY] = []
        metrics["timestamp"] = time.time()
        self._lists[METRICS_KEY].append(json.dumps(metrics).encode())

    def get_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        entries = self._lists.get(METRICS_KEY, [])
        return [json.loads(e) for e in entries[-count:]]

    def update_elo(self, generation: int, elo: float) -> None:
        if ELO_KEY not in self._hashes:
            self._hashes[ELO_KEY] = {}
        self._hashes[ELO_KEY][str(generation)] = str(elo)

    def get_elo_history(self) -> Dict[int, float]:
        raw = self._hashes.get(ELO_KEY, {})
        return {int(k): float(v) for k, v in raw.items()}

    def close(self) -> None:
        pass
