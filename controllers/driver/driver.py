"""
driver.py - Supervisor controller for robot simulation
This controller handles keyboard input and sends commands to robots.
It also manages goal-based behavior and reinforcement learning.
"""

from controller import Supervisor
from common import common_print
import pickle
import os


class Driver(Supervisor):
    # Constants
    TIME_STEP = 64
    DEFAULT_POSITION = [-0.3, -0.1, 0]

    def __init__(self):
        super(Driver, self).__init__()

        # Initialize devices
        self.emitter = self.getDevice("emitter")
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.TIME_STEP)

        # Get reference to robot
        self.robot = self.getFromDef("ROBOT1")
        self.translation_field = self.robot.getField("translation")

        # Initialize learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.load_q_table()

        # Current target position (for goal-seeking)
        self.target_position = None

        # Track episode for training
        self.episode_count = 0
        self.max_episodes = 100
        self.training_active = False
        self.episode_step = 0
        self.max_steps = 200  # Steps per episode

        print("Driver initialized. Press 'I' for help.")

    def run(self):
        """Main loop for the supervisor controller"""
        self.display_help()
        previous_message = ""

        while True:
            # Handle RL training if active
            if self.training_active:
                self.manage_training()

            # Process keyboard input
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
                print(f"ROBOT1 is located at ({position[0]:.2f}, {position[1]:.2f})")
            elif k == ord("R"):
                print(f"Teleporting ROBOT1 to {self.DEFAULT_POSITION[:2]}")
                self.translation_field.setSFVec3f(self.DEFAULT_POSITION)
            elif k == ord("I"):
                self.display_help()
            elif k == ord("L"):
                self.start_learning()
            elif k == ord("P"):
                # Set a target position
                try:
                    target_x = float(input("Enter target X coordinate: "))
                    target_y = float(input("Enter target Y coordinate: "))
                    self.target_position = [target_x, target_y]
                    message = f"seek goal:{target_x},{target_y}"
                    print(f"Setting goal at ({target_x}, {target_y})")
                except ValueError:
                    print("Invalid coordinates. Please enter numeric values.")
            elif k == ord("Q"):
                self.save_q_table()
                print("Q-table saved.")

            # Send message to robot if needed
            if message and message != previous_message:
                previous_message = message
                print(f"Command: {message}")
                self.emitter.send(message.encode("utf-8"))

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
            " P - Set a goal position\n"
            " L - Start reinforcement learning\n"
            " Q - Save the Q-table"
        )

    def start_learning(self):
        """Start the reinforcement learning process"""
        print("Beginning reinforcement learning...")
        self.training_active = True
        self.episode_count = 0
        self.episode_step = 0

        # Reset robot position
        self.translation_field.setSFVec3f(self.DEFAULT_POSITION)

        # Tell robot to enter learning mode
        self.emitter.send("learn".encode("utf-8"))

    def manage_training(self):
        """Manage the RL training process"""
        # Get robot state
        # position = self.robot.getPosition()
        # velocity = self.robot.getVelocity()

        # Increment step counter
        self.episode_step += 1

        # Check if episode is complete
        if self.episode_step >= self.max_steps:
            self.episode_count += 1
            self.episode_step = 0

            # Reset robot position for next episode
            self.translation_field.setSFVec3f(self.DEFAULT_POSITION)

            # Report progress
            print(f"Completed episode {self.episode_count}/{self.max_episodes}")

            # Reduce exploration rate
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)

            # Send updated exploration rate to robot
            self.emitter.send(f"exploration:{self.exploration_rate}".encode("utf-8"))

            # Check if training is complete
            if self.episode_count >= self.max_episodes:
                self.training_active = False
                self.save_q_table()
                print("Training complete! Q-table saved.")
                self.emitter.send("stop learn".encode("utf-8"))

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


# Main entry point
controller = Driver()
common_print("driver")
controller.run()
