"""
driver.py - Supervisor controller for robot simulation
This controller handles keyboard input and sends commands to robots.
It also manages goal-based behavior and reinforcement learning.
"""

from controller import Supervisor
from common import calculate_distance, safe_reset_physics, ReinforcementLearning
import pickle
import os


class Driver(Supervisor):
    # Constants
    TIME_STEP = 64
    DEFAULT_POSITION = [-0.3, -0.1, 0]

    def __init__(self):
        super(Driver, self).__init__()

        # Get simulation time information
        self.world_time = self.getTime()
        self.world_time_step = int(self.getBasicTimeStep())

        # Initialize devices
        self.emitter = self.getDevice("emitter")
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.TIME_STEP)

        # Get reference to robot
        self.robot = self.getFromDef("ROBOT1")
        self.translation_field = self.robot.getField("translation")

        # Track current robot mode for restoring after teleport
        self.current_robot_mode = "avoid obstacles"

        # Initialize learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.load_q_table()

        # Current target position (for goal-seeking)
        self.target_position = [0.62, -0.61]  # Single specific target position
        self.previous_distance_to_target = None
        self.reached_target_threshold = (
            0.1  # Distance threshold to consider target reached
        )

        # Position sharing
        self.position_update_freq = 5  # Update position every N steps

        # Track episode for training
        self.episode_count = 0
        self.max_episodes = 100
        self.training_active = False
        self.episode_step = 0
        self.max_steps = 200  # Steps per episode
        self.reward_report_freq = 20  # Only log rewards occasionally

        print("Driver ready - Press 'I' for help")

    def run(self):
        """Main loop for the supervisor controller"""
        self.display_help()
        previous_message = ""
        step_counter = 0
        last_stats_time = self.getTime()

        while True:
            # Handle RL training if active
            if self.training_active:
                self.manage_training()

            # Send robot position information periodically
            step_counter += 1
            if step_counter % self.position_update_freq == 0:
                position = self.translation_field.getSFVec3f()
                pos_message = f"position:{position[0]},{position[1]}"
                self.emitter.send(pos_message.encode("utf-8"))

            # Process keyboard input
            k = self.keyboard.getKey()
            message = ""

            if k == ord("A"):
                message = "avoid obstacles"
                self.current_robot_mode = message
            elif k == ord("F"):
                message = "move forward"
                self.current_robot_mode = message
            elif k == ord("S"):
                message = "stop"
                self.current_robot_mode = message
            elif k == ord("T"):
                message = "turn"
                self.current_robot_mode = message
            elif k == ord("G"):
                position = self.translation_field.getSFVec3f()
                print(f"ROBOT1 is located at ({position[0]:.2f}, {position[1]:.2f})")
            elif k == ord("R"):
                # First stop the robot to prevent physics issues
                self.emitter.send("stop".encode("utf-8"))

                # Pause briefly to ensure the stop command is processed
                self.step(self.TIME_STEP)

                # Teleport the robot
                self.translation_field.setSFVec3f(self.DEFAULT_POSITION)
                print("Robot successfully reset to default position")

                # Reset physics to prevent damage or unexpected behavior
                self.robot.resetPhysics()

                # Allow physics to stabilize for one step
                self.step(self.TIME_STEP)

                # Restore previous movement mode
                if self.current_robot_mode != "stop":
                    self.emitter.send(self.current_robot_mode.encode("utf-8"))
            elif k == ord("I"):
                self.display_help()
            elif k == ord("L"):
                self.start_learning()
            elif k == ord("Q"):
                self.save_q_table()
                print("Q-table saved.")

            # Send message to robot if needed
            if message and message != previous_message:
                previous_message = message
                print(f"Command: {message}")
                self.emitter.send(message.encode("utf-8"))

            # Report statistics periodically
            current_time = self.getTime()
            if current_time - last_stats_time >= 60:  # Every minute
                self.report_simulation_stats()
                last_stats_time = current_time

            # Exit condition
            if self.step(self.TIME_STEP) == -1:
                self.save_q_table()
                break

    def display_help(self):
        """Display available keyboard commands"""
        print(
            "\nCommands:\n"
            " I - Display this help message\n"
            " A - Avoid obstacles mode\n"
            " F - Move forward\n"
            " S - Stop\n"
            " T - Turn\n"
            " R - Position ROBOT1 at (-0.3,-0.1)\n"
            " G - Get (x,y) position of ROBOT1\n"
            " L - Start reinforcement learning to reach the target\n"
            " Q - Save the Q-table"
        )

    def start_learning(self):
        """Start the reinforcement learning process"""
        print(
            f"Beginning reinforcement learning to reach target at {self.target_position}..."
        )
        self.training_active = True
        self.episode_count = 0
        self.episode_step = 0

        # First stop the robot to prevent physics issues
        self.emitter.send("stop".encode("utf-8"))
        print("Preparing robot for learning mode...")

        # Pause briefly to ensure the stop command is processed
        self.step(self.TIME_STEP)

        # Reset robot position
        self.translation_field.setSFVec3f(self.DEFAULT_POSITION)

        # Reset physics to prevent damage or unexpected behavior
        self.robot.resetPhysics()

        # Additional pause to allow physics stabilization
        self.step(self.TIME_STEP)

        # Initialize distance to target for reward calculation
        position = self.translation_field.getSFVec3f()
        self.previous_distance_to_target = self.calculate_distance_to_target(position)

        # Tell robot to enter learning mode with target information
        target_message = f"learn:{self.target_position[0]},{self.target_position[1]}"
        self.emitter.send(target_message.encode("utf-8"))
        print("Robot repositioned and learning mode activated with target information")

        # After initial setup, start the robot in avoid obstacles mode
        # to begin the learning process with movement
        self.step(self.TIME_STEP)  # Small delay before sending movement command
        self.emitter.send("avoid obstacles".encode("utf-8"))
        self.current_robot_mode = "avoid obstacles"
        print("Robot is now moving in learning mode")

    def calculate_distance_to_target(self, position):
        """Calculate distance from current position to target"""
        if self.target_position is None:
            return float("inf")
        return calculate_distance(position, self.target_position)

    def calculate_reward(self, current_position):
        """Calculate reward based on distance to target and progress"""
        current_distance = self.calculate_distance_to_target(current_position)

        # Use the common ReinforcementLearning utility
        reward = ReinforcementLearning.calculate_reward(
            current_distance,
            self.previous_distance_to_target,
            self.reached_target_threshold,
        )

        # Update previous distance for next calculation
        self.previous_distance_to_target = current_distance

        return reward

    def manage_training(self):
        """Manage the RL training process"""
        # Get robot state
        position = self.translation_field.getSFVec3f()

        # Calculate reward based on distance to target
        reward = self.calculate_reward(position)

        # Send reward to the robot (silently)
        self.emitter.send(f"reward:{reward}".encode("utf-8"))

        # Only log rewards occasionally to reduce console spam
        if self.episode_step % self.reward_report_freq == 0:
            current_distance = self.calculate_distance_to_target(position)
            print(
                f"Training progress: Step {self.episode_step}, Distance to target: {current_distance:.2f}"
            )

        # If target reached, end episode early with success
        current_distance = self.calculate_distance_to_target(position)
        if current_distance < self.reached_target_threshold:
            print(f"🎉 Target reached! Distance: {current_distance:.2f}")
            self.episode_count += 1
            self.episode_step = 0

            # Reset robot position for next episode
            self.reset_robot_position()

            # Report progress
            print(
                f"Completed episode {self.episode_count}/{self.max_episodes} with SUCCESS"
            )

            # Continue with regular episode management...
        else:
            # Increment step counter
            self.episode_step += 1

            # Check if episode is complete due to steps
            if self.episode_step >= self.max_steps:
                self.episode_count += 1
                self.episode_step = 0

                # Reset robot position for next episode
                self.reset_robot_position()

                # Report progress
                print(
                    f"Completed episode {self.episode_count}/{self.max_episodes} - Target not reached"
                )

        # Reduce exploration rate over time
        if self.episode_step == 0:  # At the start of each episode
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
            self.emitter.send(f"exploration:{self.exploration_rate}".encode("utf-8"))

            # Check if training is complete
            if self.episode_count >= self.max_episodes:
                self.training_active = False
                self.save_q_table()
                print("Training complete! Q-table saved.")
                self.emitter.send("stop learn".encode("utf-8"))
                # Stop command to halt the robot's movement
                self.step(self.TIME_STEP)
                self.emitter.send("stop".encode("utf-8"))
                print("Robot stopped after completing training.")

    def reset_robot_position(self):
        """Safely reset the robot's position for a new episode"""
        # Always use the default starting position for consistency
        start_pos = self.DEFAULT_POSITION

        # Use the common safe_reset_physics utility
        safe_reset_physics(
            self,
            self.robot,
            self.translation_field,
            start_pos,
            self.TIME_STEP,
            self.emitter,
        )

        # Reset distance tracking
        position = self.translation_field.getSFVec3f()
        self.previous_distance_to_target = self.calculate_distance_to_target(position)

        # Resume learning movement with avoid obstacles mode
        self.emitter.send("avoid obstacles".encode("utf-8"))

        # Send a randomize command to encourage exploration
        self.emitter.send("randomize".encode("utf-8"))

    def save_q_table(self):
        """Save the Q-table to a file"""
        try:
            # Ensure the directory exists
            save_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(save_dir, exist_ok=True)

            # Save the Q-table to a file
            q_table_path = os.path.join(save_dir, "q_table.pkl")
            with open(q_table_path, "wb") as f:
                pickle.dump(self.q_table, f)

            # Try to get Q-table from robot for update
            self.emitter.send("send q_table".encode("utf-8"))

            print(f"Q-table saved to {q_table_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self):
        """Load the Q-table from a file if it exists"""
        try:
            if os.path.exists("q_table.pkl"):
                with open("q_table.pkl", "rb") as f:
                    self.q_table = pickle.load(f)
                print("Q-table loaded from file")
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            self.q_table = {}

    def report_simulation_stats(self):
        """Report statistics about the simulation"""
        time = self.getTime()
        robot_position = self.translation_field.getSFVec3f()

        # Calculate distance to target if available
        distance_to_target = "N/A"
        if self.target_position:
            distance_to_target = (
                f"{calculate_distance(robot_position, self.target_position):.2f}"
            )

        print(f"\n--- Simulation Statistics at {time:.1f}s ---")
        print(f"  Robot position: ({robot_position[0]:.2f}, {robot_position[1]:.2f})")
        print(f"  Target position: {self.target_position}")
        print(f"  Distance to target: {distance_to_target}")
        print(f"  Learning active: {self.training_active}")
        if self.training_active:
            print(f"  Episode: {self.episode_count}/{self.max_episodes}")
            print(f"  Step: {self.episode_step}/{self.max_steps}")
            print(f"  Exploration rate: {self.exploration_rate:.2f}")
        print("-------------------------------------------\n")


# Main entry point
controller = Driver()
controller.run()
