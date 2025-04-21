"""Q-Learning controller that manages the training process."""

from common.config import RLConfig, RobotConfig, SimulationConfig
from common.rl_utils import calculate_distance, calculate_reward, get_action_name
import random


class QLearningController:
    def __init__(self, driver, logger):
        self.driver = driver
        self.logger = logger

        # Training parameters
        self.episode_count = 0
        self.max_episodes = RLConfig.MAX_EPISODES
        self.training_active = False
        self.episode_step = 0
        self.max_steps = RLConfig.MAX_STEPS_PER_EPISODE

        # Learning parameters
        self.exploration_rate = RLConfig.EXPLORATION_RATE
        self.min_exploration_rate = RLConfig.MIN_EXPLORATION_RATE
        self.exploration_decay = RLConfig.EXPLORATION_DECAY

        # Target tracking
        self.target_positions = RobotConfig.TARGET_POSITIONS
        self.start_positions = RobotConfig.START_POSITIONS
        self.current_target_index = 0
        self.current_start_index = 0

        # Performance tracking
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.rewards_history = []
        self.episode_rewards = []

        # Action state tracking
        self.last_action = None
        self.action_counter = 0
        self.action_persistence = RLConfig.ACTION_PERSISTENCE_INITIAL
        self.previous_distance_to_target = None  # Track distance for reward calc

        # Goal seeking state
        self.goal_seeking_active = False
        self.goal_seeking_start_time = 0
        self.goal_reached = False

    def start_learning(self):
        """Initialize the learning process."""
        self.logger.info("Starting learning process")
        self.training_active = True
        self.episode_count = 0
        self.reset_statistics()
        self.exploration_rate = RLConfig.EXPLORATION_RATE

        self.load_q_table()  # Load any existing Q-table before starting

        # Reset action state
        self.last_action = None
        self.action_counter = 0

        # Tell robot to prepare for learning
        self.driver.emitter.send("stop".encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        # Set target position first so slave knows the goal
        target_position = self.target_positions[self.current_target_index]
        target_msg = f"learn:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(target_msg.encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        # Now start learning mode
        self.driver.emitter.send("start_learning".encode("utf-8"))
        self.start_new_episode()

    def reset_statistics(self):
        """Reset learning statistics for a new training session."""
        self.logger.info("Resetting learning statistics")
        self.episode_rewards = []
        self.rewards_history = []
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.episode_step = 0

    def calculate_reward(self, current_position):
        """
        Compute reward based on progress toward the target.
        Uses the centralized reward function from rl_utils.

        Args:
            current_position (list): Current [x, y, z] position of the robot.

        Returns:
            float: Reward value.
        """
        # Use the centralized reward function
        reward = calculate_reward(
            current_position[:2],
            self.driver.target_position,
            self.previous_distance_to_target,
            RLConfig.TARGET_THRESHOLD,
        )

        # Update previous distance for next calculation
        current_distance = calculate_distance(
            current_position[:2], self.driver.target_position
        )
        self.previous_distance_to_target = current_distance

        return reward

    def manage_training_step(self, position):
        """Process one training step."""
        if not self.training_active:
            return

        # Calculate current distance to target
        current_distance = calculate_distance(position[:2], self.driver.target_position)

        # Calculate and send reward using centralized reward function
        if self.previous_distance_to_target is not None:
            reward = self.calculate_reward(position)

            # Send reward to slave for Q-learning update
            self.driver.emitter.send(f"reward:{reward}".encode("utf-8"))
            self.episode_rewards.append(reward)

            if SimulationConfig.ENABLE_DETAILED_LOGGING and self.episode_step % 50 == 0:
                self.logger.info(
                    f"Training: Episode {self.episode_count}, Step {self.episode_step}, Distance {current_distance:.2f}, Reward:{reward:.2f}"
                )

        # Send action command if needed
        if self.action_counter <= 0 or current_distance < RLConfig.TARGET_THRESHOLD:
            # Time to choose a new action
            action = self.choose_action(current_distance)
            self.execute_action(action)
            self.last_action = action

            # Adaptive action persistence based on action type and distance
            if action in [1, 2]:  # TURN_LEFT or TURN_RIGHT
                # Shorter persistence for turning actions
                self.action_counter = max(1, self.action_persistence // 2)
            elif current_distance < 0.3:
                # Shorter persistence when close to target
                self.action_counter = max(1, self.action_persistence // 2)
            else:
                # Standard persistence for forward/backward
                self.action_counter = self.action_persistence
        else:
            # Continue with current action
            self.action_counter -= 1

        # Store current distance for next step
        self.previous_distance_to_target = current_distance

        # Check episode completion
        if self.check_episode_complete(current_distance):
            self.complete_episode()

    def choose_action(self, current_distance):
        """
        Choose an action for the robot using a simpler epsilon-greedy approach.

        This is a simplified version just for the driver controller.
        The actual Q-learning happens in the slave robot.
        """
        # Use epsilon-greedy approach
        allow_stop = current_distance < RLConfig.TARGET_THRESHOLD
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)
        if random.random() < self.exploration_rate:
            # Simple random exploration without special cases
            return random.choice(action_indices)
        else:
            # Simple exploitation logic
            if allow_stop:
                return 4  # STOP
            else:
                return 0  # FORWARD

    def execute_action(self, action):
        """Send an action command to the slave robot."""
        # Send the action command
        action_msg = f"{RLConfig.ACTION_COMMAND_PREFIX}{action}"
        self.driver.emitter.send(action_msg.encode("utf-8"))

        # Log action if needed
        if SimulationConfig.ENABLE_DETAILED_LOGGING and self.episode_step % 10 == 0:
            self.logger.debug(f"Executing action: {get_action_name(action)}")

    def start_new_episode(self):
        """Start a new training episode with a new target and position."""
        self.episode_count += 1
        self.episode_step = 0
        self.episode_rewards = []

        # Select target and start positions
        self.current_target_index = (self.current_target_index + 1) % len(
            self.target_positions
        )
        self.current_start_index = (self.current_start_index + 1) % len(
            self.start_positions
        )

        target_position = self.target_positions[self.current_target_index]
        start_position = self.start_positions[self.current_start_index]

        # Set the target position for the driver
        self.driver.set_target_position(target_position)

        # Move robot to start position
        self.driver.reset_robot_position(start_position)

        # Reset the previous distance to target
        self.previous_distance_to_target = calculate_distance(
            start_position[:2], target_position
        )

        # Reset action state
        self.last_action = None
        self.action_counter = 0

        self.logger.info(f"Starting episode {self.episode_count}/{self.max_episodes}")

        # Send updated target position to slave
        target_msg = f"learn:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(target_msg.encode("utf-8"))

        # Adjust action persistence based on learning progress
        self.action_persistence = max(
            RLConfig.ACTION_PERSISTENCE_MIN,
            int(
                RLConfig.ACTION_PERSISTENCE_INITIAL
                * (RLConfig.ACTION_PERSISTENCE_DECAY**self.episode_count)
            ),
        )

    def check_episode_complete(self, current_distance):
        """Check if the current episode should be terminated."""
        # Increment step counter
        self.episode_step += 1

        # Episode is complete if:
        # 1. Robot reached target
        if current_distance < RLConfig.TARGET_THRESHOLD:
            self.logger.info(f"Target reached! Distance: {current_distance:.2f}")
            self.successful_reaches += 1
            return True

        # 2. Maximum steps reached
        if self.episode_step >= self.max_steps:
            self.logger.info(f"Episode timed out after {self.episode_step} steps")
            return True

        return False

    def complete_episode(self):
        """Finalize the current episode and prepare for the next one."""
        self.total_episodes_completed += 1

        # Calculate episode statistics
        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / max(1, len(self.episode_rewards))
        self.rewards_history.append(total_reward)

        self.logger.info(f"Episode {self.episode_count} completed")
        self.logger.info(
            f"Total reward: {total_reward:.2f}, Average reward: {avg_reward:.2f}"
        )

        # Calculate and report success rate
        success_rate = (self.successful_reaches / self.total_episodes_completed) * 100
        self.logger.info(
            f"Success rate: {success_rate:.1f}% ({self.successful_reaches}/{self.total_episodes_completed})"
        )

        # Only send exploration rate update when it actually changes
        new_exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
        if new_exploration_rate != self.exploration_rate:
            self.exploration_rate = new_exploration_rate
            self.logger.info(f"Exploration rate decayed to {self.exploration_rate:.3f}")
            self.driver.emitter.send(
                f"exploration:{self.exploration_rate}".encode("utf-8")
            )

        # Check if training is complete
        if self.episode_count >= self.max_episodes:
            self.end_training()
            return

        # Start next episode
        self.start_new_episode()

    def end_training(self):
        """End the training process and save results."""
        self.training_active = False
        self.logger.info("Training complete")

        # Calculate and report final success rate
        success_rate = (self.successful_reaches / self.total_episodes_completed) * 100
        self.logger.info(
            f"Final success rate: {success_rate:.1f}% ({self.successful_reaches}/{self.total_episodes_completed})"
        )

        # Stop the robot
        self.driver.emitter.send("stop".encode("utf-8"))

        # Wait a bit for the robot to stop
        for _ in range(3):
            self.driver.step(RobotConfig.TIME_STEP)

        # Notify slave to save Q-table
        self.save_q_table()

        # Plot results
        self.driver.plot_training_results(self.rewards_history)

        # Start goal seeking behavior after training
        self.logger.info("Starting post-training goal seeking...")
        self.start_goal_seeking()

    def start_goal_seeking(self):
        """Start goal-seeking behavior using the learned Q-table."""
        # Select a target position from the available targets
        target_position = self.target_positions[self.current_target_index]

        # Reset robot to a suitable starting position
        start_position = self.start_positions[self.current_start_index]
        self.driver.reset_robot_position(start_position)

        # Set the target for the driver to track progress
        self.driver.set_target_position(target_position)

        # Make sure the robot is fully stopped before switching modes
        self.driver.emitter.send("stop".encode("utf-8"))
        for _ in range(5):  # Wait longer to ensure stop completes
            self.driver.step(RobotConfig.TIME_STEP)

        # Clear any pending commands and ensure clean state
        self.driver.clear_pending_commands()

        # Send message to the slave to enter goal-seeking mode with the target
        seek_message = f"seek goal:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(seek_message.encode("utf-8"))

        # Delay to ensure the command is processed
        for _ in range(3):
            self.driver.step(RobotConfig.TIME_STEP)

        self.logger.info(f"Goal seeking started. Target: {target_position}")

        # Set a flag that the driver can check to know we're in goal-seeking mode
        self.goal_seeking_active = True
        self.goal_seeking_start_time = self.driver.getTime()
        self.goal_reached = False

    def save_q_table(self):
        """Save the current Q-table."""
        try:
            # Tell the slave to save its Q-table
            self.driver.emitter.send("save_q_table".encode("utf-8"))
            self.logger.info("Requested slave to save Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table save: {e}")

    def load_q_table(self):
        """Load the Q-table if it exists."""
        try:
            # Tell the slave to load its Q-table
            self.driver.emitter.send("load_q_table".encode("utf-8"))
            self.logger.info("Requested slave to load Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table load: {e}")
