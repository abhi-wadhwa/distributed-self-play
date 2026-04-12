"""
CLI entry point for the distributed self-play system.

Commands:
  - actor: Start a self-play actor
  - learner: Start the learner/trainer
  - evaluator: Start the model evaluator
  - dashboard: Start the monitoring web dashboard
  - demo: Run a quick local demo (no Redis required)
"""

from __future__ import annotations

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GAME_CHOICES = ["connect4", "othello"]


def _get_game(name: str):
    """Instantiate a game by name."""
    from src.games.connect4 import Connect4
    from src.games.othello import Othello

    games = {
        "connect4": Connect4,
        "othello": Othello,
    }
    return games[name]()


def _get_device(device: str) -> str:
    """Resolve device string, auto-detecting CUDA if requested."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Distributed self-play training pipeline."""


@main.command()
@click.option("--game", type=click.Choice(GAME_CHOICES), default="connect4", help="Game to play.")
@click.option("--redis-host", default="localhost", help="Redis hostname.")
@click.option("--redis-port", default=6379, type=int, help="Redis port.")
@click.option("--simulations", default=100, type=int, help="MCTS simulations per move.")
@click.option("--device", default="cpu", help="Torch device (cpu/cuda/auto).")
@click.option("--actor-id", default=0, type=int, help="Unique actor ID.")
@click.option("--num-games", default=0, type=int, help="Games to play (0 = infinite).")
@click.option("--channels", default=128, type=int, help="Network channels.")
@click.option("--res-blocks", default=8, type=int, help="Number of residual blocks.")
def actor(
    game: str,
    redis_host: str,
    redis_port: int,
    simulations: int,
    device: str,
    actor_id: int,
    num_games: int,
    channels: int,
    res_blocks: int,
) -> None:
    """Start a self-play actor."""
    from src.actor import Actor
    from src.communication import RedisInterface

    device = _get_device(device)
    game_instance = _get_game(game)
    comm = RedisInterface(host=redis_host, port=redis_port)

    if not comm.ping():
        logger.error(f"Cannot connect to Redis at {redis_host}:{redis_port}")
        sys.exit(1)

    network_config = {
        "board_size": game_instance.get_board_size(),
        "action_size": game_instance.get_action_size(),
        "num_channels": channels,
        "num_res_blocks": res_blocks,
    }

    act = Actor(
        game=game_instance,
        network_config=network_config,
        comm=comm,
        mcts_simulations=simulations,
        device=device,
        actor_id=actor_id,
    )
    act.run(num_games=num_games)


@main.command()
@click.option("--game", type=click.Choice(GAME_CHOICES), default="connect4", help="Game to train.")
@click.option("--redis-host", default="localhost", help="Redis hostname.")
@click.option("--redis-port", default=6379, type=int, help="Redis port.")
@click.option("--device", default="auto", help="Torch device (cpu/cuda/auto).")
@click.option("--batch-size", default=256, type=int, help="Training batch size.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate.")
@click.option("--buffer-capacity", default=500000, type=int, help="Replay buffer capacity.")
@click.option("--min-buffer", default=1000, type=int, help="Min buffer size before training.")
@click.option("--checkpoint-dir", default="checkpoints", help="Checkpoint directory.")
@click.option("--num-steps", default=0, type=int, help="Training steps (0 = infinite).")
@click.option("--channels", default=128, type=int, help="Network channels.")
@click.option("--res-blocks", default=8, type=int, help="Number of residual blocks.")
def learner(
    game: str,
    redis_host: str,
    redis_port: int,
    device: str,
    batch_size: int,
    lr: float,
    buffer_capacity: int,
    min_buffer: int,
    checkpoint_dir: str,
    num_steps: int,
    channels: int,
    res_blocks: int,
) -> None:
    """Start the training learner."""
    from src.communication import RedisInterface
    from src.learner import Learner

    device = _get_device(device)
    game_instance = _get_game(game)
    comm = RedisInterface(host=redis_host, port=redis_port)

    if not comm.ping():
        logger.error(f"Cannot connect to Redis at {redis_host}:{redis_port}")
        sys.exit(1)

    learn = Learner(
        game_name=game,
        board_size=game_instance.get_board_size(),
        action_size=game_instance.get_action_size(),
        comm=comm,
        checkpoint_dir=checkpoint_dir,
        num_channels=channels,
        num_res_blocks=res_blocks,
        batch_size=batch_size,
        learning_rate=lr,
        buffer_capacity=buffer_capacity,
        min_buffer_size=min_buffer,
        device=device,
    )
    learn.run(num_steps=num_steps)


@main.command()
@click.option("--game", type=click.Choice(GAME_CHOICES), default="connect4", help="Game to evaluate.")
@click.option("--redis-host", default="localhost", help="Redis hostname.")
@click.option("--redis-port", default=6379, type=int, help="Redis port.")
@click.option("--device", default="cpu", help="Torch device.")
@click.option("--num-games", default=20, type=int, help="Games per evaluation matchup.")
@click.option("--simulations", default=50, type=int, help="MCTS simulations for evaluation.")
@click.option("--checkpoint-dir", default="checkpoints", help="Checkpoint directory.")
@click.option("--channels", default=128, type=int, help="Network channels.")
@click.option("--res-blocks", default=8, type=int, help="Number of residual blocks.")
def evaluator(
    game: str,
    redis_host: str,
    redis_port: int,
    device: str,
    num_games: int,
    simulations: int,
    checkpoint_dir: str,
    channels: int,
    res_blocks: int,
) -> None:
    """Start the model evaluator."""
    from src.communication import RedisInterface
    from src.evaluator import Evaluator

    device = _get_device(device)
    game_instance = _get_game(game)
    comm = RedisInterface(host=redis_host, port=redis_port)

    if not comm.ping():
        logger.error(f"Cannot connect to Redis at {redis_host}:{redis_port}")
        sys.exit(1)

    evl = Evaluator(
        game=game_instance,
        board_size=game_instance.get_board_size(),
        action_size=game_instance.get_action_size(),
        comm=comm,
        checkpoint_dir=checkpoint_dir,
        num_eval_games=num_games,
        mcts_simulations=simulations,
        num_channels=channels,
        num_res_blocks=res_blocks,
        device=device,
    )
    evl.run()


@main.command()
@click.option("--host", default="0.0.0.0", help="Dashboard host.")
@click.option("--port", default=5000, type=int, help="Dashboard port.")
@click.option("--redis-host", default="localhost", help="Redis hostname.")
@click.option("--redis-port", default=6379, type=int, help="Redis port.")
def dashboard(host: str, port: int, redis_host: str, redis_port: int) -> None:
    """Start the monitoring dashboard."""
    from src.viz.app import run_dashboard

    run_dashboard(host=host, port=port, redis_host=redis_host, redis_port=redis_port)


@main.command()
@click.option("--game", type=click.Choice(GAME_CHOICES), default="connect4", help="Game to demo.")
@click.option("--simulations", default=25, type=int, help="MCTS simulations (low for speed).")
@click.option("--num-games", default=2, type=int, help="Number of demo games.")
def demo(game: str, simulations: int, num_games: int) -> None:
    """Run a quick local demo without Redis."""
    from src.actor import Actor
    from src.communication import MockRedisInterface
    from src.learner import Learner
    from src.network import create_network

    game_instance = _get_game(game)
    comm = MockRedisInterface()

    board_size = game_instance.get_board_size()
    action_size = game_instance.get_action_size()

    # Use a small network for the demo
    net_config = {
        "board_size": board_size,
        "action_size": action_size,
        "num_channels": 32,
        "num_res_blocks": 2,
    }

    click.echo(f"Running demo with {game} (simulations={simulations})")
    click.echo("=" * 50)

    # Create a learner to publish initial weights
    learn = Learner(
        game_name=game,
        board_size=board_size,
        action_size=action_size,
        comm=comm,
        num_channels=32,
        num_res_blocks=2,
        batch_size=32,
        min_buffer_size=32,
    )
    learn.publish_weights()
    click.echo("Initial weights published.")

    # Create actor and play games
    act = Actor(
        game=game_instance,
        network_config=net_config,
        comm=comm,
        mcts_simulations=simulations,
        actor_id=0,
    )

    for i in range(num_games):
        click.echo(f"\nPlaying game {i + 1}/{num_games}...")
        stats = act.run_episode()
        click.echo(
            f"  Experiences: {stats['num_experiences']}, "
            f"  Duration: {stats['game_duration_s']:.1f}s"
        )

    # Train on collected experience
    click.echo(f"\nBuffer size: {len(learn.buffer)}")
    click.echo(f"Experience queue: {comm.experience_queue_size()}")

    # Pull experience into learner buffer
    learn.pull_experience()
    click.echo(f"After pull - Buffer: {len(learn.buffer)}")

    if len(learn.buffer) >= learn.min_buffer_size:
        click.echo("\nTraining for 10 steps...")
        for step in range(10):
            losses = learn.train_step()
            if step % 5 == 0:
                click.echo(
                    f"  Step {step}: loss={losses['total_loss']:.4f} "
                    f"(policy={losses['policy_loss']:.4f}, "
                    f"value={losses['value_loss']:.4f})"
                )
        click.echo("Training complete!")
    else:
        click.echo(f"Not enough data to train (need {learn.min_buffer_size}).")

    click.echo("\nDemo finished.")


if __name__ == "__main__":
    main()
