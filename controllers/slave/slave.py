"""
Enhanced slave controller with additional Webots robot API functionality
"""

from controller import AnsiCodes, Robot
from common import normalize_angle, calculate_distance
import math
import random


class Enumerate(object):
    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)


class Slave(Robot):
    Mode = Enumerate("STOP MOVE_FORWARD AVOID_OBSTACLES TURN SEEK_GOAL LEARN")
    timeStep = 32
    maxSpeed = 10.0
    mode = Mode.AVOID_OBSTACLES
    motors = []
    distanceSensors = []

    def boundSpeed(self, speed):
        return max(-self.maxSpeed, min(self.maxSpeed, speed))

    def __init__(self):
        super(Slave, self).__init__()

        # Get robot information
        self.robot_name = self.getName()

        # Get basic time step from the world
        self.world_time_step = int(self.getBasicTimeStep())
        self.timeStep = self.world_time_step

        # Try to load custom data from previous runs (Q-learning data)
        try:
            custom_data = self.getCustomData()
            if custom_data and custom_data.strip():
                print(f"Found custom data: {custom_data}")
                # Could parse this to retrieve learning parameters
        except Exception:
            # Removed redundant print about no custom data
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
            # Try to get the GPS device directly
            gps_device = self.getDevice("gps")
            self.gps = gps_device
            self.gps.enable(self.timeStep)
        except Exception:
            print("INFO: Using supervisor position updates")
            self.gps = None

        # For positioning without GPS
        self.position = [0, 0]  # Default position when unknown

        # Learning parameters
        self.learning_active = False
        self.exploration_rate = 0.3
        self.target_position = None
        self.last_reward = 0
        self.q_table = {}

        # Control verbose output
        self.verbose_rewards = False  # Set to True to see all reward messages
        self.reward_count = 0
        self.reward_report_freq = 20  # Only report every Nth reward

        # Exploration parameters
        self.random_behavior_counter = 0
        self.random_behavior_duration = 20  # Steps to perform random behavior
        self.random_direction = 1  # 1 for right, -1 for left
        self.is_random_walking = False

        # Add simulation time tracking
        self.start_time = self.getTime()

    def run(self):
        while True:
            # Read the supervisor order.
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getString()
                self.receiver.nextPacket()

                # Process position updates - reduce verbosity
                if message.startswith("position:"):
                    try:
                        coords = message[9:].split(",")
                        if len(coords) == 2:
                            new_position = [float(coords[0]), float(coords[1])]

                            # Only print significant updates when in SEEK_GOAL mode
                            if (
                                self.mode == self.Mode.SEEK_GOAL
                                and self.target_position
                                and (
                                    not self.position
                                    or calculate_distance(self.position, new_position)
                                    > 0.2
                                )
                            ):
                                # Calculate and display distance to target
                                dist_to_target = calculate_distance(
                                    new_position, self.target_position
                                )
                                print(
                                    f"Position: ({new_position[0]:.2f}, {new_position[1]:.2f}) - "
                                    f"Distance to target: {dist_to_target:.2f}"
                                )

                            # Update stored position
                            self.position = new_position
                    except ValueError:
                        print("Invalid position data")
                # Process messages
                elif message.startswith("reward:"):
                    # Process reward from the supervisor
                    try:
                        self.last_reward = float(message[7:])
                        self.reward_count += 1

                        # Only report rewards occasionally to reduce spam
                        if (
                            self.verbose_rewards
                            or self.reward_count % self.reward_report_freq == 0
                        ):
                            print(f"Reward update: {self.last_reward:.2f}")
                    except ValueError:
                        print("Invalid reward value")

                # Special commands
                elif message == "randomize":
                    self.random_behavior_counter = self.random_behavior_duration
                    self.random_direction = 1 if random.random() > 0.5 else -1
                    self.is_random_walking = True
                    # Removed print about random behavior
                else:
                    # Only print for certain message types, not all
                    if not message.startswith("reward:"):
                        print(
                            "I should "
                            + AnsiCodes.RED_FOREGROUND
                            + message
                            + AnsiCodes.RESET
                            + "!"
                        )

                    # Continue processing other message types
                    if message.startswith("learn:"):
                        # Extract target position for learning
                        coords = message[6:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.learning_active = True
                                self.mode = self.Mode.LEARN
                                print(f"Learning to reach target at ({x}, {y})")
                            except ValueError:
                                print("Invalid coordinates for learning target")

                    elif message.startswith("seek goal:"):
                        # Extract target position for seeking
                        coords = message[10:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.mode = self.Mode.SEEK_GOAL
                                print(f"Seeking goal at ({x}, {y})")
                            except ValueError:
                                print("Invalid coordinates for goal")

                    elif message.startswith("exploration:"):
                        # Update exploration rate
                        try:
                            self.exploration_rate = float(message[12:])
                            print(
                                f"Exploration rate updated to {self.exploration_rate:.2f}"
                            )
                        except ValueError:
                            print("Invalid exploration rate")

                    elif message == "send q_table":
                        # Would send Q-table to supervisor
                        print("Q-table requested (not implemented)")

                    elif message == "stop learn":
                        self.learning_active = False
                        self.mode = self.Mode.AVOID_OBSTACLES
                        print("Learning mode stopped")

                    elif message == "learn":
                        self.learning_active = True
                        self.mode = self.Mode.LEARN
                        print("Learning mode activated")

                    elif message == "avoid obstacles":
                        self.mode = self.Mode.AVOID_OBSTACLES

                    elif message == "move forward":
                        self.mode = self.Mode.MOVE_FORWARD

                    elif message == "stop":
                        self.mode = self.Mode.STOP

                    elif message == "turn":
                        self.mode = self.Mode.TURN

            # Get sensor data
            delta = (
                self.distanceSensors[0].getValue() - self.distanceSensors[1].getValue()
            )
            speeds = [0.0, 0.0]

            # Handle special random behavior to escape local minimums
            if self.is_random_walking and self.random_behavior_counter > 0:
                speeds[0] = self.random_direction * self.maxSpeed / 2
                speeds[1] = -self.random_direction * self.maxSpeed / 2
                self.random_behavior_counter -= 1
                if self.random_behavior_counter == 0:
                    self.is_random_walking = False
                    # Removed print about random behavior completion
            elif self.mode == self.Mode.AVOID_OBSTACLES:
                speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)

            elif self.mode == self.Mode.MOVE_FORWARD:
                speeds[0] = self.maxSpeed
                speeds[1] = self.maxSpeed

            elif self.mode == self.Mode.TURN:
                speeds[0] = self.maxSpeed / 2
                speeds[1] = -self.maxSpeed / 2

            elif self.mode == self.Mode.SEEK_GOAL or self.mode == self.Mode.LEARN:
                # Use position from supervisor instead of GPS
                position = None

                # First try GPS if available
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

                if self.target_position:
                    # Simple goal seeking behavior
                    # Use distance sensors for obstacle avoidance
                    left_obstacle = self.distanceSensors[0].getValue() < 500
                    right_obstacle = self.distanceSensors[1].getValue() < 500

                    # If we have position data, use it to guide the robot
                    if position:
                        try:
                            # Calculate angle to target
                            target_angle = math.atan2(
                                self.target_position[1] - position[1],
                                self.target_position[0] - position[0],
                            )

                            # Get robot's orientation estimate
                            robot_orientation = delta * 0.1

                            # Calculate the angle difference
                            angle_diff = normalize_angle(
                                target_angle - robot_orientation
                            )

                            # Choose action based on angle difference
                            if abs(angle_diff) < 0.2:  # Roughly aligned with target
                                speeds[0] = self.maxSpeed
                                speeds[1] = self.maxSpeed
                            elif angle_diff > 0:  # Target is to the left
                                speeds[0] = self.maxSpeed / 2
                                speeds[1] = self.maxSpeed
                            else:  # Target is to the right
                                speeds[0] = self.maxSpeed
                                speeds[1] = self.maxSpeed / 2
                        except Exception as e:
                            print(f"Error in goal navigation: {e}")
                            # Fallback to simple movement pattern
                            speeds[0] = self.maxSpeed / 2
                            speeds[1] = self.maxSpeed / 2
                    else:
                        # No position data, use basic search pattern
                        # Rotate slowly to search
                        speeds[0] = self.maxSpeed / 4
                        speeds[1] = -self.maxSpeed / 4

                    # Obstacle avoidance always takes priority
                    if left_obstacle:
                        speeds[0] = -self.maxSpeed / 2
                        speeds[1] = self.maxSpeed
                    elif right_obstacle:
                        speeds[0] = self.maxSpeed
                        speeds[1] = -self.maxSpeed / 2

                    # Occasionally introduce randomness to prevent getting stuck
                    if self.learning_active and random.random() < 0.02:  # 2% chance
                        random_turn = random.choice([-1, 1])
                        speeds[0] = random_turn * self.maxSpeed / 2
                        speeds[1] = -random_turn * self.maxSpeed / 2
                        # Removed print about random movement
                else:
                    # No target, just avoid obstacles
                    speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                    speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)

            # Periodically report simulation statistics
            if self.getTime() - self.start_time > 60:  # Every minute of simulation time
                self.report_statistics()
                self.start_time = self.getTime()

            self.motors[0].setVelocity(speeds[0])
            self.motors[1].setVelocity(speeds[1])

            # Perform a simulation step, quit the loop when Webots is about to quit.
            if self.step(self.timeStep) == -1:
                # Before exiting, save some data to the robot
                if self.learning_active:
                    self.save_learning_progress()
                break

    def get_mode_name(self, mode_value):
        """Convert a mode value to its string name"""
        for name, value in vars(self.Mode).items():
            if value == mode_value:
                return name
        return f"UNKNOWN_MODE({mode_value})"

    def report_statistics(self):
        """Report various statistics about the robot's performance"""
        elapsed = self.getTime()
        current_pos = self.position if self.position else [0, 0]

        stats = {
            "time": elapsed,
            "position": current_pos,
            "mode": self.get_mode_name(self.mode),
            "reward_count": self.reward_count,
        }

        if self.target_position:
            distance = calculate_distance(current_pos, self.target_position)
            stats["distance_to_target"] = distance

        print(f"--- Statistics at {elapsed:.1f}s ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def save_learning_progress(self):
        """Save learning progress to robot custom data"""
        # For now, just a placeholder showing the concept
        if hasattr(self, "q_table") and self.q_table:
            # In a real implementation, we'd serialize the q_table
            data = f"learning_active:{self.learning_active},exploration:{self.exploration_rate}"
            try:
                self.setCustomData(data)
                print(f"Learning progress saved to robot: {data}")
            except Exception as e:
                print(f"Could not save data: {e}")


controller = Slave()
controller.run()
