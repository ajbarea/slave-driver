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
from typing import List, Tuple, Optional, Any

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


class ReinforcementLearning:
    """Collection of reinforcement learning utility methods."""

    @staticmethod
    def calculate_reward(
        current_distance: float,
        previous_distance: Optional[float],
        target_threshold: float = 0.1,
    ) -> float:
        """
        Calculate the reward based on the change in distance to the target.

        Args:
            current_distance (float): The current distance to the target.
            previous_distance (float): The previous distance to the target.
            target_threshold (float): Distance below which the target is considered reached.

        Returns:
            float: The computed reward.
        """
        if current_distance < target_threshold:
            return 100.0  # Large positive reward for reaching the target

        if previous_distance is not None:
            improvement = previous_distance - current_distance
            progress_reward = improvement * 10.0  # Reward scaled by improvement
            step_penalty = -0.1  # Small penalty per step
            return progress_reward + step_penalty

        return 0.0  # Initial step

    @staticmethod
    def calculate_improved_reward(
        current_distance: float,
        previous_distance: Optional[float],
        target_threshold: float = 0.1,
        current_position: Optional[List[float]] = None,
        target_position: Optional[List[float]] = None,
    ) -> float:
        """Enhanced reward calculation incorporating distance and position."""
        if current_distance < target_threshold:
            return 100.0  # Target reached

        if previous_distance is None:
            return 0.0  # First step

        # Calculate basic progress reward
        improvement = previous_distance - current_distance
        reward = 0.0

        # Add sophisticated reward components
        # Progress component with diminishing returns
        if improvement > 0:
            progress_reward = 10.0 * math.sqrt(improvement)
            if improvement > 0.1:
                progress_reward += 5.0
            reward += progress_reward
        else:
            regression_penalty = 15.0 * abs(improvement)
            reward -= min(regression_penalty, 10.0)

        # Proximity bonus - higher reward when closer to target
        proximity = 3.0 / (current_distance + 0.5)
        reward += min(proximity, 5.0)

        # Bonus for consistent progress
        if improvement > 0:
            reward += 1.0

        # Small efficiency penalty per step
        reward -= 0.2

        # Alignment bonus if positional information is provided
        if current_position and target_position:
            # Add orientation-based reward if making significant progress
            if improvement > 0.05:
                reward += 2.0

            # Add additional position-based context
            if current_distance < 0.3:  # Very close to target
                reward += 3.0  # Extra encouragement for final approach

        # Apply potential-based shaping if configured
        if (
            hasattr(ReinforcementLearning, "use_shaping")
            and ReinforcementLearning.use_shaping
        ):
            if current_position and target_position:
                current_potential = calculate_state_potential(
                    current_position, target_position
                )
                if hasattr(ReinforcementLearning, "previous_potential"):
                    reward += apply_potential_based_shaping(
                        ReinforcementLearning.previous_potential, current_potential, 0.9
                    )
                ReinforcementLearning.previous_potential = current_potential

        return reward


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


def parse_target_position(message: str) -> Optional[List[float]]:
    """
    Parse a target position from a message string.

    The expected format is "prefix:x,y", for example "learn:1.0,2.0".

    Args:
        message (str): Message containing the target coordinates.

    Returns:
        list or None: A list [x, y] if parsing is successful; otherwise, None.
    """
    try:
        colon_pos = message.find(":")
        if colon_pos == -1:
            return None
        coords = message[colon_pos + 1 :].split(",")
        if len(coords) == 2:
            return [float(coords[0]), float(coords[1])]
    except ValueError:
        pass
    except Exception as e:
        logger.error(f"Error parsing target position: {e}")

    return None


def get_next_goal_position(
    current_position: List[float], position_list: List[List[float]]
) -> List[float]:
    """
    Retrieve the next target position from a list, cycling if necessary.

    Args:
        current_position (list): The current goal position [x, y].
        position_list (list): A list of available target positions.

    Returns:
        list: The next target position [x, y]. Returns a default if the list is empty.
    """
    if not position_list:
        return [0.5, 0.5]

    current_index = 0
    for i, pos in enumerate(position_list):
        if pos[0] == current_position[0] and pos[1] == current_position[1]:
            current_index = (i + 1) % len(position_list)
            break

    return position_list[current_index]


def calculate_smooth_turn(
    sensor_value: float, threshold: int = 500, max_value: int = 1000
) -> float:
    """
    Calculate a normalized smooth turning factor based on sensor input.

    Args:
        sensor_value (float): The sensor measurement.
        threshold (int): Value where turning adjustments begin.
        max_value (int): Value where maximum turning is reached.

    Returns:
        float: A turn factor between 0.0 (no turn) and 1.0 (maximum turn).
    """
    if sensor_value <= threshold:
        return 0.0
    raw_factor = (sensor_value - threshold) / (max_value - threshold)
    return min(1.0, raw_factor * raw_factor)


def calculate_smooth_speeds(
    left_value: float,
    right_value: float,
    max_speed: float,
    base_speed_factor: float = 0.7,
) -> List[float]:
    """
    Compute left and right wheel speeds for smooth obstacle avoidance.

    Args:
        left_value (float): Left sensor value.
        right_value (float): Right sensor value.
        max_speed (float): Maximum wheel speed.
        base_speed_factor (float): Factor defining base speed.

    Returns:
        list: A list containing [left_wheel_speed, right_wheel_speed].
    """
    delta = left_value - right_value
    base_speed = max_speed * base_speed_factor
    left_turn_factor = calculate_smooth_turn(right_value)
    right_turn_factor = calculate_smooth_turn(left_value)

    if abs(delta) < 50:
        speeds = [
            base_speed + (delta * 0.0001 * max_speed),
            base_speed - (delta * 0.0001 * max_speed),
        ]
    elif delta > 0:
        speeds = [base_speed * (1.0 - 0.3 * right_turn_factor), base_speed]
    else:
        speeds = [base_speed, base_speed * (1.0 - 0.3 * left_turn_factor)]

    speeds[0] = max(-max_speed, min(max_speed, speeds[0]))
    speeds[1] = max(-max_speed, min(max_speed, speeds[1]))
    return speeds


def calculate_success_percentage(successes: int, total: int) -> float:
    """
    Compute the success percentage.

    Args:
        successes (int): Number of successful attempts.
        total (int): Total number of attempts.

    Returns:
        float: The percentage value, 0 if total is zero.
    """
    if total <= 0:
        return 0.0
    return (successes / total) * 100.0


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


def calculate_state_potential(
    position: List[float],
    target_position: List[float],
    robot_orientation: float = None,
    obstacles: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculate a potential function value for the current state.

    This is used for potential-based reward shaping, which provides
    theoretical guarantees for policy invariance while accelerating learning.

    Args:
        position (list): Current [x, y] position of the robot.
        target_position (list): Target [x, y] position to reach.
        robot_orientation (float, optional): Current orientation of the robot in radians.
        obstacles (list, optional): List of obstacle positions to avoid.

    Returns:
        float: A potential value (higher means closer to desired outcomes).
    """
    # Distance component - higher potential when closer to target
    distance = calculate_distance(position, target_position)

    # Using a smoother, exponential-based potential that gives stronger gradients
    # near the target and provides meaningful signal even at longer distances
    distance_potential = 10.0 * math.exp(-distance)

    # Add orientation component if orientation is provided
    orientation_potential = 0.0
    if robot_orientation is not None:
        # Calculate ideal heading to target
        ideal_heading = math.atan2(
            target_position[1] - position[1], target_position[0] - position[0]
        )
        # Calculate angle difference
        angle_diff = normalize_angle(ideal_heading - robot_orientation)
        # Smoother orientation potential using cosine function
        # Maximum value when perfectly aligned (cos(0) = 1)
        orientation_potential = 3.0 * (math.cos(angle_diff) + 1) / 2

    # Obstacle avoidance component - if obstacles are provided
    obstacle_potential = 0.0
    if obstacles:
        for obstacle in obstacles:
            obstacle_dist = calculate_distance(position, obstacle)
            # Stronger repulsive field for very close obstacles
            if obstacle_dist < 0.5:  # Consider obstacles within 0.5 units
                # Exponential repulsion that grows quickly as distance decreases
                obstacle_potential -= 2.0 * math.exp(2.0 * (0.3 - obstacle_dist))

    # Overall potential combines all components
    return distance_potential + orientation_potential + obstacle_potential


def apply_potential_based_shaping(
    current_potential: float,
    next_potential: float,
    discount_factor: float = 0.9,
    shaping_factor: float = 1.0,
) -> float:
    """
    Calculate a reward shaping term using the potential-based approach.

    This implements F(s,s') = γΦ(s') - Φ(s) which preserves optimal policies
    while helping the agent learn more efficiently.

    Args:
        current_potential (float): Potential of the current state.
        next_potential (float): Potential of the next state.
        discount_factor (float): Discount factor for future rewards (gamma).
        shaping_factor (float): Factor to scale the shaping reward.

    Returns:
        float: The shaped reward component to be added to the original reward.
    """
    return shaping_factor * (discount_factor * next_potential - current_potential)


def get_adaptive_discount_factor(
    training_progress: float,
    base_gamma: float = 0.9,
    min_gamma: float = 0.6,
) -> float:
    """
    Calculate an adaptive discount factor based on training progress.

    Early in training, using a lower gamma (more myopic) helps with learning
    immediate rewards. As training progresses, gamma increases to optimize
    for long-term returns.

    Args:
        training_progress (float): Value between 0 and 1 indicating training progress.
        base_gamma (float): The target gamma value for late in training.
        min_gamma (float): The minimum gamma value for early training.

    Returns:
        float: Adapted gamma value between min_gamma and base_gamma.
    """
    # Bound training progress between 0 and 1
    progress = max(0.0, min(1.0, training_progress))

    # Calculate gamma that grows from min_gamma to base_gamma as training progresses
    # Using a smooth sigmoid-like transition
    return min_gamma + (base_gamma - min_gamma) * (1 - math.exp(-3 * progress))


def calculate_curiosity_bonus(
    state: Tuple,
    visit_counts: dict,
    max_bonus: float = 2.0,
    decay_factor: float = 100.0,
) -> float:
    """
    Calculate a curiosity-driven exploration bonus that encourages visiting
    less-explored states early in training.

    Args:
        state (tuple): The current state representation.
        visit_counts (dict): Dictionary mapping states to their visit counts.
        max_bonus (float): Maximum curiosity bonus value.
        decay_factor (float): Controls how quickly the bonus decays with visits.

    Returns:
        float: Exploration bonus that decays with repeated visits.
    """
    # Get visit count for this state (default 0 if not seen before)
    visit_count = visit_counts.get(state, 0)

    # Calculate bonus using a decaying function
    # Higher for less-visited states, approaches zero for frequently visited states
    bonus = max_bonus * math.exp(-visit_count / decay_factor)

    return bonus


def calculate_interactive_reward(
    previous_distance: float,
    current_distance: float,
    robot_orientation: float,
    target_position: List[float],
    current_position: List[float],
    is_moving: bool = True,
) -> float:
    """
    Calculate an immediate interactive reward component based on the robot's
    current movement and orientation relative to the target.

    This provides faster feedback about whether the robot is making progress.

    Args:
        previous_distance (float): Previous distance to target.
        current_distance (float): Current distance to target.
        robot_orientation (float): Current robot orientation in radians.
        target_position (list): Target [x, y] position.
        current_position (list): Current [x, y] position.
        is_moving (bool): Whether the robot is currently moving.

    Returns:
        float: Interactive reward component.
    """
    reward = 0.0

    # Calculate distance improvement
    distance_improvement = previous_distance - current_distance

    # Calculate ideal heading to target
    ideal_heading = math.atan2(
        target_position[1] - current_position[1],
        target_position[0] - current_position[0],
    )

    # Calculate how well the robot is aligned with the target
    angle_diff = abs(normalize_angle(ideal_heading - robot_orientation))
    alignment_quality = (math.pi - angle_diff) / math.pi  # 1.0 when perfectly aligned

    # Reward components:

    # 1. Distance improvement with progressive scaling
    if distance_improvement > 0:
        # Larger rewards for bigger improvements
        progress_reward = 8.0 * math.sqrt(distance_improvement)
        # Bonus for significant progress
        if distance_improvement > 0.05:
            progress_reward += 3.0
        reward += progress_reward
    else:
        # Penalty for moving away from target
        regression_penalty = 12.0 * abs(distance_improvement)
        reward -= min(regression_penalty, 10.0)  # Cap the penalty

    # 2. Alignment reward - encourage facing toward the target while moving
    if is_moving:
        # Higher reward when moving in the right direction
        alignment_reward = (
            4.0 * alignment_quality * alignment_quality
        )  # Squared for more contrast
        reward += alignment_reward

    # 3. Proximity bonus for being close to target
    if current_distance < 0.5:
        # Stronger reward as robot gets very close to target
        proximity_bonus = 5.0 * (1.0 - current_distance / 0.5)
        reward += proximity_bonus

    # 4. Efficiency penalty - small cost for each action
    reward -= 0.2

    return reward
