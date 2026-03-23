# Distributed Self-Play Training Pipeline

A scalable distributed reinforcement learning system implementing AlphaZero-style self-play training with an actor-learner architecture, Redis-based communication, prioritized experience replay, model versioning with ELO tracking, and Docker orchestration.

## Architecture

```
                    +-------------------+
                    |   Redis Server    |
                    |  (Communication)  |
                    +--------+----------+
                             |
            +----------------+----------------+
            |                |                |
    +-------v------+  +-----v------+  +------v-------+
    |   Actor 1    |  |   Actor 2  |  |   Actor N    |
    |  (Self-Play) |  | (Self-Play)|  |  (Self-Play) |
    |   via MCTS   |  |  via MCTS  |  |   via MCTS   |
    +-------+------+  +-----+------+  +------+-------+
            |                |                |
            +-------Push Experience-----------+
                             |
                    +--------v----------+
                    |   Redis Queues    |
                    | (Experience Data) |
                    +--------+----------+
                             |
                    +--------v----------+
                    |     Learner       |
                    |  (GPU Training)   |
                    | Prioritized Replay|
                    +--------+----------+
                             |
                 +-----------+-----------+
                 |                       |
        +--------v----------+  +--------v----------+
        | Publish Weights   |  | Save Checkpoints  |
        | (Redis Pub/Sub)   |  | (Model Versioning)|
        +-------------------+  +--------+----------+
                                        |
                               +--------v----------+
                               |    Evaluator      |
                               | (ELO Tracking)    |
                               | vs Past Selves    |
                               | vs Random Baseline|
                               +-------------------+
```

## Key Components

### Actor (`src/actor.py`)
Self-play workers that continuously generate training data. Each actor:
- Loads the latest model weights from Redis
- Plays complete games using MCTS with neural network guidance
- Applies data augmentation via board symmetries
- Pushes experience tuples `(board, policy, value)` to the Redis queue
- Scales horizontally by running multiple actor containers

### Learner (`src/learner.py`)
Central training process that:
- Pulls experience from Redis into a local prioritized replay buffer
- Samples mini-batches weighted by TD-error priority
- Trains a dual-headed ResNet (policy + value) with cross-entropy and MSE losses
- Publishes updated weights to Redis for actors
- Saves periodic checkpoints with generation tracking

### Evaluator (`src/evaluator.py`)
Model evaluation service that:
- Tests new checkpoints against the random baseline
- Runs matches against previous model generations
- Maintains ELO ratings across the league of past selves
- Publishes evaluation metrics to the dashboard

### Neural Network (`src/network.py`)
AlphaZero-style architecture:
- Input convolutional layer
- Residual tower (configurable depth)
- Policy head: outputs action probabilities via softmax
- Value head: outputs position evaluation in [-1, 1] via tanh

### MCTS (`src/mcts.py`)
Monte Carlo Tree Search with PUCT selection:
- Neural network provides prior probabilities and value estimates
- Dirichlet noise at the root for exploration
- Temperature-controlled move selection
- Configurable simulation count

### Replay Buffer (`src/replay_buffer.py`)
Prioritized experience replay:
- Circular buffer with configurable capacity
- Priority-based sampling (TD-error weighted)
- New experiences receive maximum priority
- Priority exponent (alpha) controls sampling distribution

### Model Versioning (`src/model_version.py`)
- Saves checkpoints with generation numbers
- ELO rating system tracking strength progression
- League of past selves for evaluation matchups
- Configurable checkpoint retention policy

### Games (`src/games/`)
Pluggable game interface with two implementations:
- **Connect Four**: 6x7 board, 7 actions, mirror symmetry
- **Othello**: 8x8 board, 65 actions (64 cells + pass), 8-fold symmetry

## Scaling Properties

```
Throughput = N_actors x Games_per_actor_per_second

With 4 actors @ 100 MCTS sims, ~50 experiences/game:
  ~200 experiences/sec per actor
  ~800 experiences/sec total

Training processes 256 samples/step:
  ~3 training steps/sec on GPU
  Buffer fills faster than consumption => no idle learner
```

Horizontal scaling:
- Add more actors with `docker compose up -d --scale actor=N`
- Each actor runs on CPU; learner benefits from GPU
- Redis handles communication bottleneck up to ~10k actors

## Quick Start

### Local Demo (no Redis required)

```bash
pip install -e ".[dev]"
python -m src.cli demo --game connect4 --simulations 25
```

### Docker Compose (full distributed setup)

```bash
# Start all services (Redis + Learner + 4 Actors + Evaluator + Dashboard)
docker compose up -d

# Scale to 8 actors
docker compose up -d --scale actor=8

# View training dashboard
open http://localhost:5000

# View logs
docker compose logs -f learner

# Stop
docker compose down
```

### Manual Setup

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Learner
python -m src.cli learner --game connect4 --device auto

# Terminal 3+: Actors
python -m src.cli actor --game connect4 --actor-id 0
python -m src.cli actor --game connect4 --actor-id 1

# Terminal N: Evaluator
python -m src.cli evaluator --game connect4

# Terminal N+1: Dashboard
python -m src.cli dashboard
```

## Project Structure

```
distributed-self-play/
├── README.md
├── Makefile                    # Build/test/deploy commands
├── pyproject.toml              # Package configuration
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── .github/workflows/ci.yml   # CI pipeline
├── src/
│   ├── __init__.py
│   ├── actor.py                # Self-play actor process
│   ├── learner.py              # Training learner process
│   ├── evaluator.py            # Model evaluation service
│   ├── replay_buffer.py        # Prioritized experience replay
│   ├── model_version.py        # Checkpoint and ELO management
│   ├── communication.py        # Redis interface + mock
│   ├── network.py              # Dual-headed ResNet
│   ├── mcts.py                 # Monte Carlo Tree Search
│   ├── cli.py                  # CLI entry point
│   ├── games/
│   │   ├── __init__.py
│   │   ├── game_base.py        # Abstract game interface
│   │   ├── connect4.py         # Connect Four
│   │   └── othello.py          # Othello/Reversi
│   └── viz/
│       ├── __init__.py
│       └── app.py              # Monitoring dashboard
├── tests/
│   ├── test_actor.py           # Actor tests (mocked Redis)
│   ├── test_learner.py         # Learner tests (mocked Redis)
│   ├── test_replay_buffer.py   # Buffer priority/capacity tests
│   └── test_games.py           # Game rule correctness tests
├── examples/
│   └── demo.py                 # End-to-end demo script
└── LICENSE
```

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Lint
make lint
```

## Configuration

Key hyperparameters (configurable via CLI flags):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--simulations` | 100 | MCTS simulations per move |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--buffer-capacity` | 500,000 | Replay buffer size |
| `--channels` | 128 | Network convolutional channels |
| `--res-blocks` | 8 | Residual block count |

## Adding a New Game

Implement the `GameBase` abstract class:

```python
from src.games.game_base import GameBase

class MyGame(GameBase):
    def get_board_size(self): ...
    def get_action_size(self): ...
    def get_initial_board(self): ...
    def get_next_state(self, board, player, action): ...
    def get_valid_moves(self, board, player): ...
    def get_game_ended(self, board, player): ...
    def get_canonical_board(self, board, player): ...
    def get_symmetries(self, board, pi): ...
    def display(self, board): ...
```

Register it in `src/games/__init__.py` and `src/cli.py`.

## License

MIT
