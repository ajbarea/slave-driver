"""
Enhanced slave controller with additional Webots robot API functionality

This module defines the Slave class, which extends the Robot class and provides
functionalities for obstacle avoidance, reinforcement learning, and goal seeking.
"""

from controller import AnsiCodes, Robot  # type: ignore
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from common.logger import get_logger
from common.config import RLConfig, RobotConfig, SimulationConfig
from common.rl_utils import calculate_distance, get_action_name
from q_learning_agent import QLearningAgent

# Set up logger
logger = get_logger(
    __name__, level=getattr(logging, SimulationConfig.LOG_LEVEL_SLAVE, "INFO")
)


class Enumerate(object):
    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)


class Slave(Robot):
    Mode = Enumerate("STOP MOVE_FORWARD AVOID_OBSTACLES TURN SEEK_GOAL LEARN")
    timeStep = RobotConfig.TIME_STEP
    maxSpeed = RobotConfig.MAX_SPEED
    mode = Mode.AVOID_OBSTACLES
    motors = []
    distanceSensors = []

    def boundSpeed(self, speed):
        """Clamp the speed value within [-maxSpeed, maxSpeed]."""
        return max(-self.maxSpeed, min(self.maxSpeed, speed))

    def __init__(self):
        """Initialize the slave robot, its sensors, devices and learning parameters."""
        super(Slave, self).__init__()

        # Get robot information
        self.robot_name = self.getName()
        logger.info(f"Initializing robot: {self.robot_name}")

        # Get basic time step from the world
        self.world_time_step = int(self.getBasicTimeStep())
        self.timeStep = self.world_time_step

        # Try to load custom data from previous runs (Q-learning data)
        try:
            custom_data = self.getCustomData()
            if custom_data and custom_data.strip():
                logger.info(f"Found custom data: {custom_data}")
        except Exception:
            pass

        self.mode = self.Mode.AVOID_OBSTACLES
        self.camera = self.getDevice("camera")
        self.camera.enable(4 * self.timeStep)
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)
        self.motors.append(self.getDevice("left wheel motor"))
        self.motors.append(self.getDevice("right wheel motor"))
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        for dsnumber in range(0, 2):
            self.distanceSensors.append(self.getDevice("ds" + str(dsnumber)))
            self.distanceSensors[-1].enable(self.timeStep)

        # Add GPS for position tracking
        self.gps = None
        try:
            gps_device = self.getDevice("gps")
            self.gps = gps_device
            self.gps.enable(self.timeStep)
        except Exception:
            logger.info("Using supervisor position updates (GPS not available)")
            self.gps = None

        # For positioning without GPS
        self.position = [0, 0]  # Default position when unknown
        self.orientation = 0.0  # Estimate of current orientation

        # Initialize the Q-learning agent
        self.q_agent = QLearningAgent(
            learning_rate=RLConfig.LEARNING_RATE,
            min_learning_rate=RLConfig.MIN_LEARNING_RATE,
            discount_factor=RLConfig.DISCOUNT_FACTOR,
            min_discount_factor=RLConfig.MIN_DISCOUNT_FACTOR,
            exploration_rate=RLConfig.EXPLORATION_RATE,
            max_speed=self.maxSpeed,
        )

        # Q-learning state
        self.learning_active = False
        self.target_position = None
        self.last_reward = 0
        self.current_state = None
        self.last_action = None

        # For tracking learning performance
        self.rewards_history = []

        # Flag to track if target reached message has been printed
        self.target_reached_reported = False

        # Add action persistence for smoother learning
        self.action_persistence = 0  # Counter for current action duration
        self.action_persistence_duration = RLConfig.ACTION_PERSISTENCE_INITIAL
        self.current_persistent_action = None  # Current action being executed

        logger.info(f"Slave robot initialization complete: {self.robot_name}")

    def run(self):
        """Main control loop for the slave robot."""
        logger.info("Starting slave robot control loop")

        while True:
            # Read the supervisor order.
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getString()
                self.receiver.nextPacket()

                # Process position updates
                if message.startswith("position:"):
                    try:
                        coords = message[9:].split(",")
                        if len(coords) == 2:
                            new_position = [float(coords[0]), float(coords[1])]
                            self.position = new_position
                    except ValueError:
                        logger.error("Invalid position data")

                # Process rewards
                elif message.startswith("reward:"):
                    try:
                        reward = float(message[7:])
                        self.last_reward = reward

                        # Track reward for plotting
                        self.rewards_history.append(reward)

                        # Update Q-table if learning is active
                        if (
                            self.learning_active
                            and self.current_state is not None
                            and self.last_action is not None
                        ):
                            next_state = self.get_discrete_state()
                            self.q_agent.update_q_table(
                                self.current_state, self.last_action, reward, next_state
                            )
                            self.current_state = next_state

                            # Optionally adjust action persistence based on rewards
                            if abs(reward) > 10:
                                self.action_persistence = max(
                                    0, self.action_persistence - 3
                                )
                    except ValueError:
                        logger.error("Invalid reward value")

                elif message == "start_learning":
                    self.mode = self.Mode.LEARN
                    self.current_state = None  # Force state recalculation
                    logger.info("Entering learning mode")

                elif message == "send q_table":
                    self.send_q_table()

                elif message == "plot_learning":
                    self.plot_rewards()

                elif message == "load_q_table":
                    try:
                        self.q_agent.load_q_table(SimulationConfig.Q_TABLE_PATH)
                        logger.info("Q-table loaded from file")
                    except Exception as e:
                        logger.error(f"Error loading Q-table: {e}")

                else:
                    # Only print command messages under specific conditions
                    if message.startswith(RLConfig.ACTION_COMMAND_PREFIX):
                        # Extract action number from command
                        try:
                            current_action = int(message.split(":")[1])
                            action_name = get_action_name(current_action)
                        except (ValueError, IndexError):
                            logger.info(f"Received command: {message}")
                    elif (
                        not message.startswith("reward:")
                        and not message.startswith("seek goal:")
                        and not message.startswith("position:")
                        and not message.startswith("exploration:")
                        and not message.startswith("persistence:")
                    ):
                        logger.info(
                            f"Received command: {AnsiCodes.RED_FOREGROUND}{message}{AnsiCodes.RESET}"
                        )

                    # Continue processing other message types
                    if message.startswith("learn:"):
                        coords = message[6:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.learning_active = True
                                self.mode = self.Mode.LEARN
                                logger.info(f"Learning to reach target at ({x}, {y})")
                            except ValueError:
                                logger.error("Invalid coordinates for learning target")

                    # Process goal seeking
                    elif message.startswith("seek goal:"):
                        coords = message[10:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.mode = self.Mode.SEEK_GOAL
                                self.learning_active = False  # Exploitation

                                # Reset action persistence to allow quicker reactions
                                self.action_persistence = 0
                                self.current_persistent_action = None

                                # Reset target reached flag
                                self.target_reached_reported = False

                                logger.info(f"Seeking goal at ({x}, {y})")
                            except ValueError:
                                logger.error("Invalid coordinates for goal")

                    # Update exploration rate
                    elif message.startswith("exploration:"):
                        try:
                            new_rate = float(message[12:])
                            self.q_agent.exploration_rate = new_rate
                        except ValueError:
                            logger.error("Invalid exploration rate")

                    # Process action persistence adjustments
                    elif message.startswith("persistence:"):
                        try:
                            new_persistence = int(message[12:])
                            self.action_persistence_duration = new_persistence
                            # Reset current persistence counter to try new value immediately
                            self.action_persistence = 0
                            logger.debug(
                                f"Action persistence updated to {new_persistence} steps"
                            )
                        except ValueError:
                            logger.error("Invalid persistence value")

                    # Other mode switches
                    elif message == "stop learn":
                        self.learning_active = False
                        self.mode = self.Mode.AVOID_OBSTACLES
                        logger.info("Learning mode stopped")
                    elif message == "learn":
                        self.learning_active = True
                        self.mode = self.Mode.LEARN
                        if (
                            not hasattr(self, "learn_mode_reported")
                            or not self.learn_mode_reported
                        ):
                            logger.info("Learning mode activated")
                            self.learn_mode_reported = True
                    elif message == "avoid obstacles":
                        self.mode = self.Mode.AVOID_OBSTACLES
                    elif message == "move forward":
                        self.mode = self.Mode.MOVE_FORWARD
                    elif message == "stop":
                        self.mode = self.Mode.STOP
                        self.handle_reset()  # Add proper reset handling
                    elif message == "turn":
                        self.mode = self.Mode.TURN

            # Get sensor data
            delta = (
                self.distanceSensors[0].getValue() - self.distanceSensors[1].getValue()
            )
            speeds = [0.0, 0.0]

            # Simplify mode handling - reduce heuristic overheads
            # Basic obstacle avoidance with less complexity
            if self.mode == self.Mode.AVOID_OBSTACLES:
                speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)

                # Basic obstacle safety with fewer special cases
                left_sensor = self.distanceSensors[0].getValue()
                right_sensor = self.distanceSensors[1].getValue()

                if (
                    left_sensor > 800 and right_sensor > 800
                ):  # Both sensors detect close obstacles
                    speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]  # Back up
                elif left_sensor > 800:  # Left obstacle
                    speeds = [self.maxSpeed / 2, -self.maxSpeed / 3]
                elif right_sensor > 800:  # Right obstacle
                    speeds = [-self.maxSpeed / 3, self.maxSpeed / 2]

            elif self.mode == self.Mode.MOVE_FORWARD:
                speeds[0] = self.maxSpeed
                speeds[1] = self.maxSpeed

                # Basic safety for MOVE_FORWARD - still need to avoid obstacles
                left_sensor = self.distanceSensors[0].getValue()
                right_sensor = self.distanceSensors[1].getValue()
                if (
                    left_sensor > 800 or right_sensor > 800
                ):  # Only for very close obstacles
                    speeds = [0.0, 0.0]  # Stop instead of crashing

            elif self.mode == self.Mode.TURN:
                speeds[0] = self.maxSpeed / 2
                speeds[1] = -self.maxSpeed / 2

            elif self.mode == self.Mode.STOP:
                speeds = [0.0, 0.0]

            elif self.mode == self.Mode.SEEK_GOAL or self.mode == self.Mode.LEARN:
                # Get current position
                position = None
                if self.gps:
                    try:
                        position = self.gps.getValues()
                        if position and len(position) >= 2:
                            position = position[:2]  # Just x, y coordinates
                    except Exception:
                        position = None

                # Fallback to position from supervisor
                if position is None:
                    position = self.position

                # Update our current position
                if position:
                    self.position = position

                # Learning mode - use reinforcement learning agent
                if self.mode == self.Mode.LEARN and self.learning_active:
                    # Get current state if not already set
                    if self.current_state is None:
                        self.current_state = self.get_discrete_state()

                    # Calculate distance to target for STOP filtering
                    current_distance = None
                    if self.target_position and self.position:
                        try:
                            current_distance = calculate_distance(
                                self.position, self.target_position
                            )
                        except Exception:
                            current_distance = None

                    # Get action using Q-learning agent with simplified persistence logic
                    if self.action_persistence == 0:
                        action = self.q_agent.choose_action(
                            self.current_state, current_distance
                        )
                        self.current_persistent_action = action
                        # Set action persistence with less complexity
                        self.action_persistence = self.action_persistence_duration
                    else:
                        action = self.current_persistent_action
                        self.action_persistence -= 1

                    # Execute action
                    speeds = self.q_agent.execute_action(action)
                    self.last_action = action

                # Goal seeking mode - use learned policy without exploration
                elif self.mode == self.Mode.SEEK_GOAL and self.target_position:
                    # Get current state
                    state = self.get_discrete_state()

                    # Calculate distance to target
                    current_distance = calculate_distance(
                        self.position, self.target_position
                    )

                    # Check if we've reached the target
                    if current_distance < RLConfig.TARGET_THRESHOLD:
                        # Target reached logic
                        if not self.target_reached_reported:
                            logger.info("🎯 Target reached in SEEK_GOAL mode!")
                            self.target_reached_reported = True

                        speeds = [0.0, 0.0]  # Stop the robot
                        # Force motor velocities to zero for more reliable stopping
                        self.motors[0].setVelocity(0.0)
                        self.motors[1].setVelocity(0.0)
                    else:
                        # Reset the flag if we move away from the target
                        if (
                            self.target_reached_reported
                            and current_distance > RLConfig.TARGET_THRESHOLD * 1.5
                        ):
                            self.target_reached_reported = False

                        # Simplified goal seeking behavior
                        # Use Q-learning policy with minimal adjustments
                        action = self.q_agent.choose_best_action(
                            state, current_distance
                        )

                        # Log action and state periodically
                        if random.random() < 0.01:  # ~1% of steps
                            action_name = get_action_name(action)
                            logger.debug(
                                f"Goal seeking: state={state}, action={action_name}, "
                                f"distance={current_distance:.2f}"
                            )

                        speeds = self.q_agent.execute_action(action)

                        # Basic safety override for imminent collisions
                        left_sensor = self.distanceSensors[0].getValue()
                        right_sensor = self.distanceSensors[1].getValue()

                        if left_sensor > 800 and right_sensor > 800:
                            speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]
                        elif left_sensor > 800:
                            speeds = [self.maxSpeed / 2, -self.maxSpeed / 3]
                        elif right_sensor > 800:
                            speeds = [-self.maxSpeed / 3, self.maxSpeed / 2]

                # No target or not in learning mode
                else:
                    # Use basic obstacle avoidance
                    speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                    speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)

                    # Apply obstacle safety logic for non-learning modes
                    left_sensor = self.distanceSensors[0].getValue()
                    right_sensor = self.distanceSensors[1].getValue()

                    # Only override if obstacles are very close
                    if left_sensor > 800 and right_sensor > 800:  # Higher threshold
                        speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]  # Back up
                    elif left_sensor > 800:  # Left obstacle very close
                        speeds[0] = self.maxSpeed / 2
                        speeds[1] = -self.maxSpeed / 3
                    elif right_sensor > 800:  # Right obstacle very close
                        speeds[0] = -self.maxSpeed / 3
                        speeds[1] = self.maxSpeed / 2

            # Set the motor speed
            self.motors[0].setVelocity(speeds[0])
            self.motors[1].setVelocity(speeds[1])

            # Perform a simulation step, quit the loop when Webots is about to quit.
            if self.step(self.timeStep) == -1:
                # Before exiting, save some data and plot final results
                if self.learning_active and len(self.rewards_history) > 0:
                    self.save_learning_progress()
                    self.plot_rewards()

                # Save the Q-table
                self.q_agent.save_q_table(SimulationConfig.Q_TABLE_PATH)
                logger.info("Robot controller exiting")
                break

    def get_discrete_state(self):
        """Generate a discrete state representation for Q-learning using the centralized rl_utils function."""
        if not self.position or not self.target_position:
            return None

        # Get wheel velocities
        left_wheel_velocity = self.motors[0].getVelocity()
        right_wheel_velocity = self.motors[1].getVelocity()
        wheel_velocities = [left_wheel_velocity, right_wheel_velocity]

        # Get sensor readings
        left_sensor_value = self.distanceSensors[0].getValue()
        right_sensor_value = self.distanceSensors[1].getValue()

        # Use the QLearningAgent's method, which delegates to the centralized function
        return self.q_agent.get_discrete_state(
            self.position,
            self.target_position,
            self.orientation,
            left_sensor_value,
            right_sensor_value,
            wheel_velocities,
        )

    def save_learning_progress(self):
        """Store current learning progress in the robot's custom data."""
        if hasattr(self, "q_agent") and self.q_agent.q_table:
            data = f"learning_active:{self.learning_active},exploration:{self.q_agent.exploration_rate}"
            try:
                self.setCustomData(data)
                logger.info(f"Learning progress saved to robot: {data}")
            except Exception as e:
                logger.error(f"Could not save data: {e}")

    def send_q_table(self):
        """Output Q-table statistics for supervisor review."""
        if not hasattr(self, "emitter"):
            return
        try:
            q_table_size = len(self.q_agent.q_table)
            logger.info(f"Q-table information: {q_table_size} states")
        except Exception as e:
            logger.error(f"Error with Q-table: {e}")

    def plot_rewards(self):
        """Plot the rewards history to visualize the learning progress."""
        if not self.rewards_history:
            logger.warning("No rewards to plot")
            return

        # Plot raw rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards_history, label="Rewards", color="lightblue", alpha=0.3)

        # Apply smoothing for better visualization
        if len(self.rewards_history) > 10:
            window = min(len(self.rewards_history) // 10, 50)  # Adaptive window size
            window = max(window, 2)  # Ensure window is at least 2
            rewards_smoothed = np.convolve(
                self.rewards_history, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(len(rewards_smoothed)),
                rewards_smoothed,
                label="Smoothed Rewards",
                color="blue",
                linewidth=2,
            )

        # Add title and labels
        plt.title(f"Learning Progress - {self.robot_name}")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.legend()

        # Save the plot
        try:
            plot_dir = SimulationConfig.PLOT_DIR
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{self.robot_name}_learning.png")
            plt.savefig(plot_path)
            logger.info(f"Learning plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        plt.close()

    def handle_reset(self):
        """Handle a reset command by ensuring robot is properly stopped."""
        # Stop motors
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)

        # Reset physics-related state
        self.orientation = 0.0
        self.action_persistence = 0
        self.current_persistent_action = None

        # Give time for physics to settle
        for _ in range(3):
            if self.step(self.timeStep) == -1:
                break


# Initialize and run the controller
if __name__ == "__main__":
    controller = Slave()
    controller.run()
