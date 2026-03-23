"""
Shared neural network architecture for self-play training.

Uses a ResNet-style dual-headed network (policy + value) similar to AlphaZero.
The network takes a board state and outputs:
  - policy: probability distribution over actions
  - value: scalar evaluation of the position [-1, 1]
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with two convolutions and batch normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class DualHeadedNet(nn.Module):
    """
    AlphaZero-style dual-headed network.

    Architecture:
        Input -> Conv -> N x ResBlock -> Policy head
                                      -> Value head

    Args:
        board_size: (rows, cols) of the game board.
        action_size: number of possible actions.
        num_channels: number of convolutional channels (default 128).
        num_res_blocks: number of residual blocks (default 8).
    """

    def __init__(
        self,
        board_size: Tuple[int, int],
        action_size: int,
        num_channels: int = 128,
        num_res_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.board_rows, self.board_cols = board_size
        self.action_size = action_size
        self.num_channels = num_channels

        # Initial convolution
        self.conv_input = nn.Conv2d(1, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_rows * self.board_cols, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_rows * self.board_cols, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: board tensor of shape (batch, 1, rows, cols).

        Returns:
            (log_policy, value): log-probabilities over actions and scalar value in [-1, 1].
        """
        # Shared trunk
        out = F.relu(self.bn_input(self.conv_input(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        log_policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return log_policy, v


def board_to_tensor(board: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert a numpy board to a batched tensor for the network.

    Args:
        board: numpy array of shape (rows, cols).
        device: target torch device.

    Returns:
        Tensor of shape (1, 1, rows, cols).
    """
    t = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0)
    return t.to(device)


def create_network(
    board_size: Tuple[int, int],
    action_size: int,
    num_channels: int = 128,
    num_res_blocks: int = 8,
    device: str = "cpu",
) -> DualHeadedNet:
    """Create and initialize a DualHeadedNet on the specified device."""
    net = DualHeadedNet(board_size, action_size, num_channels, num_res_blocks)
    return net.to(device)
