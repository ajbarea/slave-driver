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
from typing import List, Optional, Any

# Import logger
from common.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def common_print(caller: str) -> None:
    """
    Print a common message indicating the caller.

    Args:
        caller (str): Identifier of the caller module.
    """
    logger.info(
        f"This module is common to both driver and slave controllers (called from {caller})."
    )


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


def calculate_heading_to_target(
    current_pos: List[float], target_pos: List[float]
) -> float:
    """
    Calculate the heading angle from the current position to the target position.

    Args:
        current_pos (list or tuple): The current [x, y] position.
        target_pos (list or tuple): The target [x, y] position.

    Returns:
        float: The heading angle in radians.
    """
    return math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])


def calculate_orientation_reward(
    robot_orientation: float,
    current_pos: List[float],
    target_pos: List[float],
    max_reward: float = 5.0,
) -> float:
    """
    Calculate reward based on how well the robot is oriented toward the target.

    Args:
        robot_orientation (float): Current orientation of robot in radians.
        current_pos (list): Current [x, y] position of the robot.
        target_pos (list): Target [x, y] position.
        max_reward (float): Maximum reward for perfect alignment.

    Returns:
        float: Reward value based on alignment quality.
    """
    # Calculate ideal heading to target
    ideal_heading = math.atan2(
        target_pos[1] - current_pos[1], target_pos[0] - current_pos[0]
    )

    # Calculate the angular difference
    angle_diff = normalize_angle(ideal_heading - robot_orientation)

    # Convert to absolute difference in range [0, π]
    abs_angle_diff = abs(angle_diff)

    # Use a cosine function for smoother rewards that drop more significantly as alignment worsens
    # This will provide more directional gradient for learning
    alignment_factor = (
        math.cos(abs_angle_diff) + 1
    ) / 2  # Range 0-1, 1 is perfect alignment

    # Scale to desired reward range
    reward = alignment_factor * max_reward

    return reward


def calculate_movement_reward(
    distance_improvement: float, current_distance: float, stop_action: bool = False
) -> float:
    """
    Calculate a movement-based reward that encourages progress toward the target.

    Args:
        distance_improvement (float): Change in distance to target (positive means closer)
        current_distance (float): Current distance to target
        stop_action (bool): Whether the robot is currently stopped

    Returns:
        float: The calculated reward
    """
    # Base reward for distance improvement
    if distance_improvement > 0:
        # Progressive reward for closing distance
        base_reward = 15.0 * distance_improvement
        # Bonus for significant progress
        if distance_improvement > 0.05:
            base_reward += 2.0
    else:
        # Strong penalty for moving away from target
        base_reward = -10.0 * abs(distance_improvement)

    # Apply proximity-based reward component
    proximity_factor = 1.0 / (current_distance + 0.2)  # Higher when closer to target
    proximity_reward = min(5.0, proximity_factor * 2)  # Cap at 5.0

    # Special handling for STOP action
    if stop_action:
        # Only reward stopping when very close to target
        if current_distance < 0.2:
            return proximity_reward * 3  # Encourage stopping near target
        else:
            return -5.0  # Discourage stopping away from target

    # Combine components
    return base_reward + proximity_reward


def safe_reset_physics(
    robot_controller: Any,
    robot_node: Any,
    translation_field: Any,
    position: List[float],
    time_step: int,
    emitter: Any = None,
) -> None:
    """
    Safely reset a robot's physics and position with stabilization.

    Args:
        robot_controller: Controller instance managing simulation steps.
        robot_node: The robot node whose physics is to be reset.
        translation_field: Field object for setting the robot's translation.
        position (list): New position [x, y, z] to set.
        time_step (int): Simulation time step.
        emitter (optional): Device used to send stop commands.
    """
    # Stop the robot.
    if emitter:
        emitter.send("stop".encode("utf-8"))

    for _ in range(3):
        robot_controller.step(time_step)

    # Ensure the robot is in an upright orientation.
    rotation_field = robot_node.getField("rotation")
    if rotation_field:
        rotation_field.setSFRotation([0, 1, 0, 0])

    translation_field.setSFVec3f(position)
    robot_node.resetPhysics()

    for _ in range(5):
        robot_controller.step(time_step)

    # Optionally reset velocity fields if they exist.
    try:
        velocity_field = robot_node.getField("velocity")
        if velocity_field:
            velocity_field.setSFVec3f([0, 0, 0])
        angular_velocity_field = robot_node.getField("angularVelocity")
        if angular_velocity_field:
            angular_velocity_field.setSFVec3f([0, 0, 0])
    except Exception:
        # Some fields might not be available.
        pass

    robot_controller.step(time_step)


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
