"""
RL Utilities module for centralizing reinforcement learning functions.

This module provides common functions for state discretization, reward
calculation, and other RL-related utilities used by both the driver
and slave controllers.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
from common.config import RLConfig
from common.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def get_discrete_state(
    position: List[float],
    target_position: List[float],
    orientation: float,
    left_sensor: float,
    right_sensor: float,
    wheel_velocities: List[float],
    angle_bins: int = 8,
) -> Optional[Tuple]:
    """
    Generate a refined discrete state representation for Q-learning.

    Args:
        position: Current [x, y] position
        target_position: Target [x, y] position
        orientation: Current robot orientation in radians
        left_sensor: Left distance sensor reading
        right_sensor: Right distance sensor reading
        wheel_velocities: [left_wheel_velocity, right_wheel_velocity]
        angle_bins: Number of angle bins to use

    Returns:
        A tuple representing the discrete state or None if inputs are invalid
    """
    if not position or not target_position:
        return None

    # Calculate distance to target
    distance = calculate_distance(position, target_position)

    # Calculate angle to target
    dx = target_position[0] - position[0]
    dy = target_position[1] - position[1]
    angle_to_target = math.atan2(dy, dx)

    # Normalize angle to target relative to robot's orientation
    relative_angle = normalize_angle(angle_to_target - orientation)

    # Simplified distance binning with fixed number of bins
    distance_bin = discretize_distance(distance)

    # Fixed angle binning
    angle_bin = (
        int((relative_angle + math.pi) / (2 * math.pi / angle_bins)) % angle_bins
    )

    # Obstacle detection
    left_obstacle = discretize_sensor(left_sensor)
    right_obstacle = discretize_sensor(right_sensor)

    # Velocity awareness
    velocity_state = discretize_velocity(wheel_velocities)

    # Create the state tuple
    state = (distance_bin, angle_bin, left_obstacle, right_obstacle, velocity_state)
    return state


def discretize_distance(distance: float) -> int:
    """
    Discretize a continuous distance value into a small number of bins.

    Args:
        distance: Distance to target

    Returns:
        Discretized distance bin (0-6)
    """
    if distance < 0.1:  # Very close - precise control needed
        return 0
    elif distance < 0.25:  # Close - approach carefully
        return 1
    elif distance < 0.5:  # Medium-close
        return 2
    elif distance < 0.75:  # Medium
        return 3
    elif distance < 1.25:  # Medium-far
        return 4
    elif distance < 2.0:  # Far
        return 5
    else:  # Very far
        return 6


def discretize_sensor(sensor_value: float) -> int:
    """
    Convert sensor readings to more granular obstacle detection states.

    Args:
        sensor_value: Distance sensor reading

    Returns:
        Discretized sensor state (0-3)
    """
    if sensor_value < 100:  # No obstacle detected
        return 0
    elif sensor_value < 400:  # Distant obstacle
        return 1
    elif sensor_value < 700:  # Medium-close obstacle
        return 2
    else:  # Very close obstacle
        return 3


def discretize_velocity(wheel_velocities: List[float]) -> int:
    """
    Discretize wheel velocities into movement states.

    Args:
        wheel_velocities: [left_wheel_velocity, right_wheel_velocity]

    Returns:
        Discretized velocity state (0-4)
    """
    left_vel = wheel_velocities[0]
    right_vel = wheel_velocities[1]
    avg_speed = (abs(left_vel) + abs(right_vel)) / 2

    # Check if turning (wheels moving in opposite directions)
    is_turning = left_vel * right_vel < 0

    # Define states:
    # 0: Stopped
    # 1: Slow forward
    # 2: Fast forward
    # 3: Backward
    # 4: Turning

    if is_turning:
        return 4
    elif avg_speed < 0.1:  # Stopped
        return 0
    elif left_vel > 0 and right_vel > 0:  # Forward
        return 2 if avg_speed > 5.0 else 1
    elif left_vel < 0 and right_vel < 0:  # Backward
        return 3
    else:  # Default/unexpected
        return 0


def calculate_reward(
    current_position: List[float],
    target_position: List[float],
    previous_distance: Optional[float] = None,
    target_threshold: float = 0.1,
    orientation: Optional[float] = None,
    wheel_velocities: Optional[List[float]] = None,
) -> float:
    """
    Calculate reward based on progress toward the target, with optional orientation bonus and penalties.

    Args:
        current_position: Current [x, y] position
        target_position: Target [x, y] position
        previous_distance: Previous distance to target
        target_threshold: Distance threshold to consider target reached
        orientation: Current robot orientation in radians (optional)
        wheel_velocities: [left_wheel_velocity, right_wheel_velocity] (optional)

    Returns:
        Calculated reward value
    """
    # Calculate current distance to target
    current_distance = calculate_distance(current_position[:2], target_position)

    # Target reached - very large reward
    if current_distance < target_threshold:
        return 100.0  # Significant reward for reaching target

    # First step - no previous distance for comparison
    if previous_distance is None:
        return 0.0

    # Calculate improvement (positive means getting closer)
    distance_improvement = previous_distance - current_distance

    # Main reward: progress toward target
    if distance_improvement > 0:
        base_reward = 10.0 * distance_improvement
    else:
        base_reward = -8.0 * abs(distance_improvement)

    # Proximity bonus - higher when closer to target
    proximity_bonus = 1.0 / (current_distance + 0.5)

    # Orientation bonus (small, only if orientation is provided)
    orientation_bonus = 0.0
    if orientation is not None:
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        angle_to_target = math.atan2(dy, dx)
        angle_diff = abs(normalize_angle(angle_to_target - orientation))
        # Bonus is highest when facing the target (angle_diff ~ 0)
        orientation_bonus = 0.5 * (1.0 - angle_diff / math.pi)  # Max 0.5

    # Penalty for unnecessary stops (if not close to target)
    stop_penalty = 0.0
    if wheel_velocities is not None and current_distance > target_threshold * 2:
        if abs(wheel_velocities[0]) < 0.05 and abs(wheel_velocities[1]) < 0.05:
            stop_penalty = -1.0  # Penalize stopping far from target

    # Penalty for spins (turning in place: wheels in opposite directions)
    spin_penalty = 0.0
    if wheel_velocities is not None:
        if wheel_velocities[0] * wheel_velocities[1] < 0:
            spin_penalty = -0.5  # Small penalty for spinning in place

    # Small step penalty to encourage efficiency
    step_penalty = -RLConfig.STEP_PENALTY

    # Combine all reward components
    total_reward = (
        base_reward
        + proximity_bonus
        + orientation_bonus
        + stop_penalty
        + spin_penalty
        + step_penalty
    )

    return total_reward


def get_action_name(action: int) -> str:
    """
    Convert an action index to a human-readable name.

    Args:
        action: Action index (0-4)

    Returns:
        Action name as string
    """
    action_names = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACKWARD", "STOP"]
    if 0 <= action < len(action_names):
        return action_names[action]
    return f"UNKNOWN_ACTION({action})"


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
