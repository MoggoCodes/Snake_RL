# Snake Reinforcement Learning

A deep reinforcement learning project where an AI agent learns to play the classic Snake game using Deep Q-Learning (DQN).

## Project Overview

This project implements a Snake game environment and trains a Deep Q-Network (DQN) to play it. The agent learns to control the snake, collect food, and avoid collisions through trial and error.

### Key Features

- Custom Gym environment for Snake game
- Deep Q-Network (DQN) implementation with experience replay
- Apple Metal GPU acceleration support
- Real-time visualization of training progress
- Configurable hyperparameters and reward structure

## Components

- `snake_env.py`: Custom Gym environment implementing the Snake game
  - Handles game logic, state representation, and reward calculation
  - Provides a 5-channel state observation:
    1. Snake body positions
    2. Snake head position
    3. Food position
    4. Horizontal distance to food
    5. Vertical distance to food

- `dqn_agent.py`: Deep Q-Network implementation
  - Experience replay buffer for improved learning
  - Separate neural network paths for spatial and distance information
  - Target network for stable training
  - Gradient clipping and dropout for regularization

- `train.py`: Training script with visualization
  - Real-time display of game state
  - Episode statistics tracking
  - Model checkpointing

- `play.py`: Script to watch trained agent play

## Requirements

This project uses `pixi` for dependency management. Make sure you have `pixi` installed.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd snake_rl
```

2. Install dependencies using pixi:
```bash
pixi install
```

## Usage

### Training

To train the agent:
```bash
pixi run train
```

The training script will display:
- Live visualization of the game
- Episode number
- Current score
- Average score
- Current exploration rate (epsilon)

Training can be interrupted at any time with Ctrl+C.

### Playing

To watch a trained agent play:
```bash
pixi run play
```

## Model Architecture

The DQN uses a dual-path architecture:
1. Spatial path: Processes snake and food positions
   - Convolutional layers for spatial feature extraction
2. Distance path: Processes distance information
   - 1x1 convolutions for numerical distance processing
3. Combined processing
   - Merges spatial and distance features
   - Fully connected layers for action selection

## Reward Structure

The agent receives rewards based on:
- +25.0 for eating food
- -10.0 for collisions
- +0.2 * distance_delta for moving closer to food
- -0.1 for changing direction
- +0.1 for staying alive

## Performance

The agent typically learns to:
1. Avoid collisions
2. Navigate towards food
3. Achieve scores of 1-2 within 100-200 episodes
4. Show improving performance with longer training

## Contributing

Feel free to open issues or submit pull requests with improvements.
