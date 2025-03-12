"""
common.py - Shared utilities for robot control system
"""

import math


def common_print(caller):
    print(
        "This module is common to both driver and slave controllers (called from "
        + caller
        + ")."
    )


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two spoints"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def normalize_angle(angle):
    """Normalize angle to range [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class ReinforcementLearning:
    """Common reinforcement learning utilities shared between controllers"""

    @staticmethod
    def calculate_reward(current_distance, previous_distance, target_threshold=0.1):
        """
        Calculate reward based on progress toward target

        Args:
            current_distance: Current distance to target
            previous_distance: Previous distance to target
            target_threshold: Distance threshold to consider target reached

        Returns:
            float: Calculated reward
        """
        # Check if target reached
        if current_distance < target_threshold:
            return 100.0  # Large positive reward for reaching target

        if previous_distance is not None:
            # Reward for getting closer, penalize for moving away
            distance_improvement = previous_distance - current_distance
            progress_reward = distance_improvement * 10.0

            # Small penalty for each step to encourage efficiency
            step_penalty = -0.1

            return progress_reward + step_penalty

        return 0.0  # First calculation with no previous distance


def safe_reset_physics(
    robot_controller, robot_node, translation_field, position, time_step, emitter=None
):
    """
    Safely reset a robot's position and physics

    Args:
        robot_controller: The controller handling the simulation step
        robot_node: The robot node to reset
        translation_field: The translation field to set
        position: The position to set
        time_step: The simulation time step
        emitter: Optional emitter to send commands to robot
    """
    # First stop the robot if we can
    if emitter:
        emitter.send("stop".encode("utf-8"))

    # Allow stop command to process
    robot_controller.step(time_step)

    # Reset position
    translation_field.setSFVec3f(position)

    # Reset physics
    robot_node.resetPhysics()

    # Allow physics to stabilize
    robot_controller.step(time_step)


def parse_target_position(message):
    """
    Parse target position from a message string

    Args:
        message: The message containing coordinates (format: "prefix:x,y")

    Returns:
        tuple or list: (x, y) coordinates or None if parsing fails
    """
    try:
        # Find the position after the colon
        colon_pos = message.find(":")
        if colon_pos == -1:
            return None

        # Extract and split the coordinate part
        coords = message[colon_pos + 1 :].split(",")
        if len(coords) == 2:
            x = float(coords[0])
            y = float(coords[1])
            return [x, y]  # Return as list to match target_position format
    except ValueError:
        pass
    except Exception as e:
        print(f"Error parsing target position: {e}")

    return None


def get_next_goal_position(current_position, position_list):
    """
    Get the next goal position from a list of positions

    Args:
        current_position: Current goal position
        position_list: List of available positions

    Returns:
        list: Next position from the list
    """
    if not position_list:
        return [0.5, 0.5]  # Default position

    # Find current position in the list
    current_index = 0
    for i, pos in enumerate(position_list):
        if pos[0] == current_position[0] and pos[1] == current_position[1]:
            current_index = (i + 1) % len(position_list)
            break

    return position_list[current_index]
