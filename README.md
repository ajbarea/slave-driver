# Self Adaptive Systems with Reinforcement Learning

A robotics simulation project demonstrating self-adaptive behaviors through reinforcement learning techniques. This project uses Webots robotics simulator to implement Q-learning for autonomous robot navigation and obstacle avoidance.

## Project Overview

This project demonstrates how reinforcement learning can be used to create self-adaptive robot behaviors. The system consists of:

- **Driver Controller**: A supervisor that manages the simulation, sends commands to robots, and handles the reinforcement learning process
- **Slave Robot**: A robot that learns to navigate towards target positions while avoiding obstacles
- **Shared Utilities**: Common functions and algorithms used by both controllers

## Key Features

- **Q-Learning Implementation**: Robot learns optimal navigation strategies through trial and error
- **Adaptive Behavior**: Robot improves performance over time as it learns from experience
- **Goal-Seeking**: Robot can be directed to reach specific target positions
- **Obstacle Avoidance**: Integrated sensors allow detection and avoidance of obstacles
- **Performance Tracking**: Visual charts and statistics show learning progress
- **Multiple Training Scenarios**: Training with different starting positions and target locations

## Project Structure

```plaintext
slave_driver/
├── common/
│   ├── common.py        # Shared utility functions
│   ├── config.py        # Configuration parameters
│   └── logger.py        # Logging utilities
│
├── controllers/
│   ├── driver/              # Supervisor controller
│   │   ├── driver.py        # Main supervisor code
│   │   ├── q_learning_controller.py  # RL controller
│   │   └── data/            # Saved Q-tables and performance data
│   │       
│   └── slave/               # Robot controller
│       ├── slave.py         # Robot control code
│       ├── q_learning_agent.py  # Agent implementation
│       └── data/           # Performance visualization
│
└── worlds/                  # Webots world files
```

## How to Run

1. Install Webots robotics simulator
2. Open the project world file in Webots
3. The simulation will start with the robot in obstacle avoidance mode
4. Use keyboard commands (see below) to control the simulation

## Keyboard Controls

- **I** - Display help message
- **A** - Switch to obstacle avoidance mode
- **F** - Move forward
- **S** - Stop
- **T** - Turn
- **R** - Reset robot position
- **G** - Get current robot position
- **L** - Start reinforcement learning

## Reinforcement Learning Implementation

The project uses Q-learning, a model-free reinforcement learning algorithm:

- **State Space**: Robot's position relative to target, sensor readings, and wheel velocities
- **Action Space**: FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD, STOP
- **Reward Function**: Sophisticated reward calculation based on distance improvements, orientation, and proximity to target
- **Learning Parameters**: Adaptive exploration rate, learning rate, and discount factor

## Implementation Details

### State Representation

The robot's state is discretized into:

- Distance bins (adaptive granularity based on proximity to target)
- Angle bins (adaptive number based on distance)
- Obstacle detection (left and right sensors with graduated sensitivity)
- Velocity bins (tracking robot's current movement speed)

### Adaptive Behaviors

- **Exploration vs. Exploitation**: Controlled by an adaptive exploration rate that decreases over time
- **Action Persistence**: Dynamic action duration for smoother movement
- **Learning Rate Adaptation**: Learning rate adjusts based on training progress
- **Discount Factor Adjustment**: Adapts based on proximity to goal and reward significance
- **Random Movements**: Occasional random behaviors to escape local minimums
- **Success Tracking**: Success rates tracked across episodes to measure improvement

### Visualization

- Learning progress is visualized through matplotlib charts
- Performance metrics include:
  - Rewards over time
  - Success rate (percentage of successful target reaches)
  - Learning curve visualizations
  - TD error tracking

## Dependencies

- Python 3.x
- Webots robotics simulator
- NumPy
- Matplotlib

## Running Training

1. Press **L** to start the reinforcement learning process
2. The robot will undergo multiple training episodes
3. Learning statistics will be displayed in the console
4. After training completes, the robot automatically switches to goal-seeking mode
5. Performance plots are saved to the data directory

## Performance Analysis

The project includes tools to visualize and analyze the robot's learning performance:

- Q-table state coverage analysis
- Reward trends over time
- Success rate statistics
- Learning rate and discount factor adaptation statistics
- TD error analysis for learning quality assessment

## Extending the Project

To extend functionality or adapt to new scenarios:

1. Modify reward functions in common.py
2. Adjust state/action spaces in q_learning_agent.py
3. Configure learning parameters in config.py
4. Create new training scenarios in driver.py
