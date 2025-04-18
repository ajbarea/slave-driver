"""
RL Utilities module for centralizing reinforcement learning functions.

This module provides common functions for state discretization, reward
calculation, and other RL-related utilities used by both the driver
and slave controllers.
"""

import math
from typing import List, Tuple, Optional
from common.logger import get_logger
from common.common import calculate_distance, normalize_angle

# Set up logger
logger = get_logger(__name__)


def get_discrete_state(
    position: List[float],
    target_position: List[float],
    orientation: float,
    left_sensor: float,
    right_sensor: float,
    wheel_velocities: List[float],
    angle_bins: int = 6,
) -> Optional[Tuple]:
    """
    Generate a simplified discrete state representation for Q-learning.

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

    # Simplified obstacle detection - binary state per sensor
    left_obstacle = 1 if left_sensor > 500 else 0
    right_obstacle = 1 if right_sensor > 500 else 0

    # Simplified velocity awareness - just moving or not
    avg_velocity = (abs(wheel_velocities[0]) + abs(wheel_velocities[1])) / 2
    is_moving = 1 if avg_velocity > 0.1 else 0

    # Create the simplified state tuple
    state = (distance_bin, angle_bin, left_obstacle, right_obstacle, is_moving)
    return state


def discretize_distance(distance: float) -> int:
    """
    Discretize a continuous distance value into a small number of bins.

    Args:
        distance: Distance to target

    Returns:
        Discretized distance bin (0-3)
    """
    if distance < 0.2:  # Very close
        return 0
    elif distance < 0.5:  # Medium-close
        return 1
    elif distance < 1.0:  # Medium
        return 2
    else:  # Far
        return 3


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
    step_penalty = -0.1

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
