"""
common.py
=========

Shared utility functions and classes for the robot control system.

This module provides helper functions for mathematical operations,
state discretization, reward calculation, plotting and physics resetting.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional

# Import logger
from common.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1 (list or tuple): Coordinates [x, y] of the first point.
        p2 (list or tuple): Coordinates [x, y] of the second point.

    Returns:
        float: The Euclidean distance.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-π, π].

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The normalized angle.
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def plot_q_learning_progress(
    rewards: List[float],
    window: int = 20,
    short_window: int = 5,
    ema_span: int = 20,
    title: str = "Q‑Learning Progress",
    filename: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plot episode rewards with:
      - raw per-episode rewards
      - short & long moving averages
      - cumulative average
      - exponential moving average (EMA)
    """
    if not rewards:
        logger.warning("No rewards to plot")
        return

    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))
    # raw rewards
    plt.plot(
        episodes, rewards, color="lightblue", alpha=0.4, label="Reward per Episode"
    )

    # short moving average
    if len(rewards) >= short_window:
        ma_s = np.convolve(rewards, np.ones(short_window) / short_window, mode="valid")
        ma_s_x = list(range(short_window, len(rewards) + 1))
        plt.plot(
            ma_s_x, ma_s, color="green", linewidth=2, label=f"{short_window}-Episode MA"
        )

    # long moving average
    if len(rewards) >= window:
        ma_l = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ma_l_x = list(range(window, len(rewards) + 1))
        plt.plot(ma_l_x, ma_l, color="red", linewidth=2, label=f"{window}-Episode MA")

    # cumulative average
    cumavg = list(np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
    plt.plot(
        episodes, cumavg, color="orange", linestyle="--", label="Cumulative Average"
    )

    # exponential moving average
    if ema_span > 1 and len(rewards) > 0:
        alpha = 2.0 / (ema_span + 1)
        ema = [rewards[0]]
        for r in rewards[1:]:
            ema.append(alpha * r + (1 - alpha) * ema[-1])
        plt.plot(
            episodes, ema, color="purple", linestyle=":", label=f"{ema_span}-Span EMA"
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()

    if filename:
        save_dir = save_dir or "."
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(path)

    plt.show()
    plt.close()
