"""
Q-Learning Agent module for the Slave robot.

This module encapsulates the reinforcement learning logic, separating it
from the robot control code for better maintainability.
"""

import random
import pickle
import os
from typing import Dict, List, Tuple, Optional
from common.logger import get_logger
from common.rl_utils import get_discrete_state
from common.config import RLConfig

# Set up logger
logger = get_logger(__name__)


class QLearningAgent:
    """
    Q-Learning agent class that manages the reinforcement learning process.

    This class encapsulates state discretization, action selection,
    Q-table updates, and other reinforcement learning operations.
    """

    # Constants for action indices
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    BACKWARD = 3
    STOP = 4

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_learning_rate: float = 0.03,
        discount_factor: float = 0.9,
        min_discount_factor: float = 0.7,
        exploration_rate: float = 0.3,
        max_speed: float = 10.0,
        angle_bins: int = 8,
    ):
        """
        Initialize the Q-learning agent with learning parameters.

        Args:
            learning_rate: Alpha parameter for Q-learning updates
            min_learning_rate: Minimum learning rate after decay
            discount_factor: Gamma parameter for future reward weighting
            min_discount_factor: Minimum discount factor
            exploration_rate: Epsilon for exploration-exploitation balance
            max_speed: Maximum robot speed for action execution
            angle_bins: Number of bins for discretizing angles
        """
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.min_discount_factor = min_discount_factor
        self.exploration_rate = exploration_rate
        self.max_speed = max_speed
        self.angle_bins = angle_bins

        # Initialize Q-table and learning statistics
        self.q_table: Dict[Tuple, List[float]] = {}
        self.total_updates = 0

        # Track learning metrics
        self.td_errors: List[float] = []
        self.learning_rates: List[float] = []
        self.discount_factors: List[float] = []

        # Load existing Q-table if available
        try:
            self.load_q_table(RLConfig.Q_TABLE_PATH)
        except Exception as e:
            logger.warning(f"Could not load Q-table: {e}")

    def get_discrete_state(
        self,
        position: List[float],
        target_position: List[float],
        orientation: float,
        left_sensor: float,
        right_sensor: float,
        wheel_velocities: List[float],
    ) -> Optional[Tuple]:
        """
        Generate a discrete state representation for Q-learning using the centralized rl_utils function.

        Args:
            position: Current [x, y] position
            target_position: Target [x, y] position
            orientation: Current robot orientation in radians
            left_sensor: Left distance sensor reading
            right_sensor: Right distance sensor reading
            wheel_velocities: [left_wheel_velocity, right_wheel_velocity]

        Returns:
            A tuple representing the discrete state or None if inputs are invalid
        """
        return get_discrete_state(
            position,
            target_position,
            orientation,
            left_sensor,
            right_sensor,
            wheel_velocities,
            self.angle_bins,
        )

    def choose_action(self, state: Tuple, current_distance: float = None) -> int:
        """
        Select an action using an enhanced epsilon-greedy strategy.
        Only allow STOP if close to the target.

        Args:
            state: The current discrete state tuple
            current_distance: Current distance to target (float)

        Returns:
            The chosen action index
        """
        # Initialize Q-values for this state if not already done
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5  # Initialize Q-values for all actions

        # Determine allowed actions
        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]  # FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD
        if allow_stop:
            action_indices.append(4)  # STOP

        # Exploration: choose a random action based on exploration rate
        if random.random() < self.exploration_rate:
            # Random action selection - bias toward forward movement to speed up exploration of the environment
            if (
                random.random() < 0.5 and 0 in action_indices
            ):  # 50% chance to go forward
                return 0  # FORWARD
            return random.choice(action_indices)  # Otherwise random

        # Exploitation: choose the action with the highest Q-value among allowed actions
        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for i, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return random.choice(best_actions)

    def choose_best_action(self, state: Tuple, current_distance: float = None) -> int:
        """
        Select the best action from the Q-table without exploration.
        Only allow STOP if close to the target.

        Args:
            state: The current discrete state tuple
            current_distance: Current distance to target (float)

        Returns:
            The action index with the highest Q-value
        """
        # If state is unknown, initialize it
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5

            # Extract state components for basic heuristic
            distance_bin, angle_bin, left_obstacle, right_obstacle, is_moving = state

            # Simple fallback for unknown states
            if left_obstacle and right_obstacle:
                return self.BACKWARD
            elif left_obstacle:
                return self.TURN_RIGHT
            elif right_obstacle:
                return self.TURN_LEFT
            elif angle_bin < self.angle_bins // 2:
                return self.TURN_RIGHT
            else:
                return self.TURN_LEFT

        # Determine allowed actions
        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)

        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for i, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return random.choice(best_actions)

    def update_q_table(
        self, state: Tuple, action: int, reward: float, next_state: Tuple
    ) -> None:
        """
        Update the Q-table using standard Q-learning with adaptive parameters.

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
        """
        if state is None or next_state is None:
            return

        # Initialize Q-values if states don't exist
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5  # For 5 actions
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 5  # For 5 actions

        # Track total updates
        self.total_updates += 1

        # Use adaptive learning rate decay
        adaptive_learning_rate = max(
            self.min_learning_rate,
            self.learning_rate
            * (
                RLConfig.LEARNING_RATE_DECAY_BASE
                ** (self.total_updates / RLConfig.LEARNING_RATE_DECAY_DENOM)
            ),
        )

        # Simplified adaptive discount factor
        adaptive_discount = max(
            self.min_discount_factor,
            self.discount_factor * (0.9995 ** (self.total_updates / 5000)),
        )

        # Store the learning parameters for analysis
        self.learning_rates.append(adaptive_learning_rate)
        self.discount_factors.append(adaptive_discount)

        # Standard Q-learning update formula
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])

        # Calculate the TD error
        td_error = reward + adaptive_discount * next_max_q - current_q
        self.td_errors.append(td_error)

        # Update Q-value with bounds
        new_q = current_q + adaptive_learning_rate * td_error
        self.q_table[state][action] = max(-50.0, min(50.0, new_q))

    def execute_action(self, action: int) -> List[float]:
        """
        Execute the given action and return the motor speeds.

        Args:
            action: The action index

        Returns:
            Motor speeds [left_speed, right_speed]
        """
        # Standard action execution
        if action == self.FORWARD:
            return [self.max_speed, self.max_speed]
        elif action == self.TURN_LEFT:
            return [self.max_speed / 2, -self.max_speed / 2]
        elif action == self.TURN_RIGHT:
            return [-self.max_speed / 2, self.max_speed / 2]
        elif action == self.BACKWARD:
            return [-self.max_speed, -self.max_speed]
        elif action == self.STOP:
            return [0.0, 0.0]
        return [0.0, 0.0]

    def save_q_table(self, filepath: str) -> bool:
        """
        Save the Q-table to a file.

        Args:
            filepath: Path to save the Q-table

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save the Q-table
            with open(filepath, "wb") as f:
                pickle.dump(self.q_table, f)

            logger.info(f"Q-table saved to {filepath} with {len(self.q_table)} states")
            return True
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")
            return False

    def load_q_table(self, filepath: str) -> bool:
        """
        Load the Q-table from a file.

        Args:
            filepath: Path to the Q-table file

        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    self.q_table = pickle.load(f)

                logger.info(
                    f"Q-table loaded from {filepath} with {len(self.q_table)} states"
                )
                return True
            else:
                logger.warning(
                    f"Q-table file {filepath} not found. Starting with empty Q-table."
                )
                self.q_table = {}
                return False
        except Exception as e:
            logger.error(f"Error loading Q-table: {e}")
            self.q_table = {}
            return False
