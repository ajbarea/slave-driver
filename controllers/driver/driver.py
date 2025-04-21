"""Supervisor controller for robot simulation."""

from controller import Supervisor  # type: ignore
import logging
import os
from common.logger import get_logger
from common.rl_utils import calculate_distance, plot_q_learning_progress
from common.config import (
    SimulationConfig,
    RobotConfig,
    RLConfig,
)
from q_learning_controller import QLearningController


class Driver(Supervisor):
    TIME_STEP = RobotConfig.TIME_STEP

    def __init__(self):
        super(Driver, self).__init__()

        # Basic setup
        self.logger = get_logger(
            __name__, level=getattr(logging, SimulationConfig.LOG_LEVEL_DRIVER, "INFO")
        )
        self.emitter = self.getDevice("emitter")
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.TIME_STEP)

        # Robot reference
        self.robot = self.getFromDef("ROBOT1")
        self.translation_field = self.robot.getField("translation")

        # Navigation and positioning
        self.target_position = None
        self.previous_distance_to_target = None

        # Create RL controller
        self.rl_controller = QLearningController(self, self.logger)

        # Step counter for periodic tasks
        self.step_counter = 0

        self.logger.info("Driver initialization complete")
        self.logger.info("Press 'I' for help")

    def run(self):
        """Main control loop."""
        self.display_help()
        previous_message = ""

        while True:
            # Increment step counter
            self.step_counter += 1

            # Send robot position information periodically
            if self.step_counter % SimulationConfig.POSITION_UPDATE_FREQ == 0:
                position = self.translation_field.getSFVec3f()
                pos_message = f"position:{position[0]},{position[1]}"
                self.emitter.send(pos_message.encode("utf-8"))

            # Handle reinforcement learning training if active
            if self.rl_controller.training_active:
                position = self.translation_field.getSFVec3f()
                self.rl_controller.manage_training_step(position)

            # Handle goal seeking if active
            elif (
                hasattr(self.rl_controller, "goal_seeking_active")
                and self.rl_controller.goal_seeking_active
            ):
                position = self.translation_field.getSFVec3f()
                self.monitor_goal_seeking(position)

            # Handle keyboard input and basic robot control
            k = self.keyboard.getKey()
            message = ""

            if k == ord("A"):
                message = "avoid obstacles"
            elif k == ord("F"):
                message = "move forward"
            elif k == ord("S"):
                message = "stop"
            elif k == ord("T"):
                message = "turn"
            elif k == ord("G"):
                position = self.translation_field.getSFVec3f()
                self.logger.info(
                    f"ROBOT1 is located at ({position[0]:.2f}, {position[1]:.2f})"
                )
            elif k == ord("R"):
                self.safely_reset_robot()
            elif k == ord("I"):
                self.display_help()
            elif k == ord("L"):
                self.rl_controller.start_learning()

            if message and message != previous_message:
                previous_message = message
                self.logger.info(f"Command: {message}")
                self.emitter.send(message.encode("utf-8"))

            if self.step(self.TIME_STEP) == -1:
                self.rl_controller.save_q_table()
                break

    def clear_pending_commands(self):
        """Clear any pending commands in the message queue to ensure clean state."""
        # Just step the simulation a few times without sending commands
        for _ in range(5):
            self.step(self.TIME_STEP)
        return

    def monitor_goal_seeking(self, position):
        """
        Monitor the robot's progress toward the goal during goal-seeking behavior.
        Enhanced with insights from old_code.py.
        """
        if not self.target_position:
            return

        # Calculate current distance to target
        current_distance = calculate_distance(position[:2], self.target_position)

        # Check if the robot has reached the target
        if current_distance < RLConfig.TARGET_THRESHOLD:
            if not getattr(self.rl_controller, "goal_reached", False):
                self.rl_controller.goal_reached = True

                # Send stop command to the robot multiple times to ensure it stops
                for _ in range(3):
                    self.emitter.send("stop".encode("utf-8"))
                    self.step(self.TIME_STEP)

                # Report success time
                elapsed_time = (
                    self.getTime() - self.rl_controller.goal_seeking_start_time
                )
                self.logger.info(f"Goal reached in {elapsed_time:.1f} seconds")

                # Disable further goal seeking checks to prevent timeout
                self.rl_controller.goal_seeking_active = False
                return

        # Check for timeout
        current_time = self.getTime()
        elapsed_time = current_time - self.rl_controller.goal_seeking_start_time

        # Provide periodic progress updates during goal seeking
        if (
            not getattr(self.rl_controller, "goal_reached", False)
            and self.step_counter % 100 == 0
        ):
            self.logger.info(
                f"Goal seeking in progress - Distance: {current_distance:.2f}, "
                f"Time elapsed: {elapsed_time:.1f}s"
            )

            # Check if robot is stuck (not making progress)
            if hasattr(self, "last_goal_seeking_distance"):
                # If distance hasn't changed much in last check
                if abs(current_distance - self.last_goal_seeking_distance) < 0.05:
                    self.stuck_counter = getattr(self, "stuck_counter", 0) + 1
                    if self.stuck_counter >= 6:  # Stuck for 6 consecutive checks
                        self.logger.info(
                            f"Robot appears stuck at distance {current_distance:.2f}. Sending randomize command."
                        )
                        self.emitter.send("randomize".encode("utf-8"))
                        self.stuck_counter = 0
                else:
                    self.stuck_counter = 0

            # Update last distance
            self.last_goal_seeking_distance = current_distance

        # Check for timeout with extended time for goal seeking
        if elapsed_time > SimulationConfig.GOAL_SEEKING_TIMEOUT:
            self.logger.info(f"Goal seeking timed out after {elapsed_time:.1f} seconds")
            self.rl_controller.goal_seeking_active = False
            self.emitter.send("stop".encode("utf-8"))

    def display_help(self):
        """Display available keyboard commands."""
        self.logger.info(
            "\nCommands:\n"
            " I - Display this help message\n"
            " A - Avoid obstacles mode\n"
            " F - Move forward\n"
            " S - Stop\n"
            " T - Turn\n"
            " R - Reset robot position\n"
            " G - Get (x,y) position of ROBOT1\n"
            " L - Start reinforcement learning"
        )

    def safely_reset_robot(self):
        """Safely reset the robot to its default position."""
        self.emitter.send("stop".encode("utf-8"))
        self.step(self.TIME_STEP)
        self.robot.resetPhysics()
        self.translation_field.setSFVec3f(RobotConfig.DEFAULT_POSITION)
        for _ in range(5):
            self.step(self.TIME_STEP)
        self.logger.info("Robot reset to default position")

    def reset_robot_position(self, position):
        """Reset the robot to a specific position with proper physics reset.

        Args:
            position (list): The [x, y, z] position to reset to.
        """
        # Add small random offset for variability
        import random

        random_offset_x = random.uniform(-0.03, 0.03)
        random_offset_y = random.uniform(-0.03, 0.03)
        randomized_position = [
            position[0] + random_offset_x,
            position[1] + random_offset_y,
            position[2],
        ]

        # Send stop command first
        self.emitter.send("stop".encode("utf-8"))
        for _ in range(3):  # Multiple steps to ensure stop is processed
            self.step(self.TIME_STEP)

        # Reset orientation to upright
        rotation_field = self.robot.getField("rotation")
        if rotation_field:
            rotation_field.setSFRotation([0, 1, 0, 0])

        # Reset position and physics
        self.translation_field.setSFVec3f(randomized_position)
        self.robot.resetPhysics()

        # Reset velocities if fields exist
        try:
            velocity_field = self.robot.getField("velocity")
            if velocity_field:
                velocity_field.setSFVec3f([0, 0, 0])
            angular_velocity_field = self.robot.getField("angularVelocity")
            if angular_velocity_field:
                angular_velocity_field.setSFVec3f([0, 0, 0])
        except Exception:
            pass  # Fields might not exist

        # Give more time to stabilize
        for _ in range(5):
            self.step(self.TIME_STEP)

        # Initialize with obstacle avoidance before learning
        self.emitter.send("avoid obstacles".encode("utf-8"))
        self.step(self.TIME_STEP * 2)

        self.logger.debug(f"Robot reset to position: {randomized_position}")

    def set_target_position(self, target_position):
        """Set the target position for the robot."""
        self.target_position = target_position

    def plot_training_results(self, rewards):
        """Plot the training results."""
        if not rewards:
            self.logger.warning("No rewards to plot")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(SimulationConfig.PLOT_DIR, exist_ok=True)

            # Create episode numbers based on actual rewards length
            episodes = list(range(1, len(rewards) + 1))

            # Ensure episodes and rewards have same length
            if len(episodes) != len(rewards):
                self.logger.warning(
                    f"Length mismatch: episodes({len(episodes)}) != rewards({len(rewards)})"
                )
                # Truncate to shorter length to ensure they match
                min_len = min(len(episodes), len(rewards))
                episodes = episodes[:min_len]
                rewards = rewards[:min_len]

            # Plot Q‑learning progress
            plot_q_learning_progress(
                rewards=rewards,
                filename="training_results",
                save_dir=SimulationConfig.PLOT_DIR,
            )

            self.logger.info(
                f"Training results plotted to {SimulationConfig.PLOT_DIR}\\training_results.png"
            )
        except Exception as e:
            self.logger.error(f"Error plotting training results: {e}")
            self.logger.error(
                f"Episodes length: {len(episodes)}, Rewards length: {len(rewards)}"
            )


# Main entry point
if __name__ == "__main__":
    controller = Driver()
    controller.run()
