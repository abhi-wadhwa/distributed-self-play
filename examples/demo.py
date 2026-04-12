"""
Quick demonstration of the distributed self-play pipeline.

Runs entirely in-process using MockRedisInterface (no Redis needed).
Demonstrates the full cycle: actor generates experience, learner trains,
evaluator measures improvement.

Usage:
    python examples/demo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from src.actor import Actor
from src.communication import MockRedisInterface
from src.evaluator import Evaluator, RandomPlayer, play_match
from src.games.connect4 import Connect4
from src.learner import Learner
from src.mcts import MCTS
from src.network import create_network


def run_demo():
    """Run an end-to-end demo of the self-play pipeline."""
    print("=" * 60)
    print("  Distributed Self-Play Pipeline Demo")
    print("=" * 60)
    print()

    game = Connect4()
    comm = MockRedisInterface()

    board_size = game.get_board_size()
    action_size = game.get_action_size()

    # Use a small network for fast demo
    num_channels = 32
    num_res_blocks = 2
    mcts_sims = 15

    network_config = {
        "board_size": board_size,
        "action_size": action_size,
        "num_channels": num_channels,
        "num_res_blocks": num_res_blocks,
    }

    # ---- Step 1: Initialize learner and publish initial weights ----
    print("[1/5] Initializing learner...")
    learner = Learner(
        game_name="connect4",
        board_size=board_size,
        action_size=action_size,
        comm=comm,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        batch_size=32,
        min_buffer_size=32,
        publish_interval=5,
        checkpoint_interval=50,
        checkpoint_dir="demo_checkpoints",
    )
    learner.publish_weights()
    print(f"  Initial weights published (generation {learner.generation})")
    print()

    # ---- Step 2: Actor generates self-play experience ----
    print("[2/5] Actor generating self-play data...")
    actor = Actor(
        game=game,
        network_config=network_config,
        comm=comm,
        mcts_simulations=mcts_sims,
        actor_id=0,
    )

    num_games = 5
    total_experiences = 0
    start_time = time.time()

    for i in range(num_games):
        stats = actor.run_episode()
        total_experiences += stats["num_experiences"]
        print(
            f"  Game {i + 1}/{num_games}: "
            f"{stats['num_experiences']} experiences, "
            f"{stats['game_duration_s']:.1f}s"
        )

    elapsed = time.time() - start_time
    print(f"  Total: {total_experiences} experiences in {elapsed:.1f}s")
    print(f"  Games/sec: {num_games / elapsed:.2f}")
    print()

    # ---- Step 3: Learner trains on collected data ----
    print("[3/5] Learner training...")
    learner.pull_experience()
    print(f"  Buffer size: {len(learner.buffer)}")

    num_train_steps = 20
    for step in range(num_train_steps):
        if len(learner.buffer) < learner.min_buffer_size:
            print(f"  Not enough data ({len(learner.buffer)} < {learner.min_buffer_size})")
            break
        losses = learner.train_step()
        if step % 5 == 0 or step == num_train_steps - 1:
            print(
                f"  Step {step + 1}/{num_train_steps}: "
                f"loss={losses['total_loss']:.4f} "
                f"(policy={losses['policy_loss']:.4f}, "
                f"value={losses['value_loss']:.4f})"
            )

    learner.publish_weights()
    print(f"  Published generation {learner.generation}")
    print()

    # ---- Step 4: Evaluate against random ----
    print("[4/5] Evaluating against random player...")
    network = create_network(
        board_size=board_size,
        action_size=action_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
    )
    weights = comm.get_latest_weights()
    network.load_state_dict(weights["state_dict"])
    network.eval()

    random_player = RandomPlayer(game)
    mcts_obj = MCTS(game=game, network=network, num_simulations=mcts_sims)

    wins = 0
    draws = 0
    eval_games = 10

    for i in range(eval_games):
        def mcts_action(board, player):
            policy = mcts_obj.search(board, player, temperature=0, add_noise=False)
            return int(np.argmax(policy))

        if i % 2 == 0:
            score = play_match(game, mcts_action, random_player.get_action)
        else:
            score = 1.0 - play_match(game, random_player.get_action, mcts_action)

        if score == 1.0:
            wins += 1
        elif score == 0.5:
            draws += 1

    print(f"  vs Random: {wins}W / {draws}D / {eval_games - wins - draws}L")
    print(f"  Win rate: {wins / eval_games:.0%}")
    print()

    # ---- Step 5: Display a sample game ----
    print("[5/5] Sample game (model vs random):")
    print("-" * 40)
    board = game.get_initial_board()
    player = 1
    move = 0

    while True:
        if player == 1:
            policy = mcts_obj.search(board, player, temperature=0, add_noise=False)
            action = int(np.argmax(policy))
            label = "Model"
        else:
            valid = game.get_valid_moves(board, player)
            valid_actions = np.where(valid > 0)[0]
            action = int(np.random.choice(valid_actions))
            label = "Random"

        board, player = game.get_next_state(board, player, action)
        move += 1

        result = game.get_game_ended(board, player)
        if result is not None:
            print(game.display(board))
            if result == 1.0:
                winner = "Model" if player == 1 else "Random"
            elif result == -1.0:
                winner = "Random" if player == 1 else "Model"
            else:
                winner = "Draw"
            print(f"\nResult after {move} moves: {winner}")
            break

    print()
    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("demo_checkpoints", ignore_errors=True)


if __name__ == "__main__":
    run_demo()
