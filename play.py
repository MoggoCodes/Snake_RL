import torch
import pygame
from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import argparse

def play_game(model_path, n_games=5):
    # Initialize environment
    env = SnakeEnv()
    
    # Device selection prioritizing Apple Metal
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Initialize agent
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, device)
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0  # No exploration, only exploitation
    
    for game in range(n_games):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            # Get action from agent
            action = agent.select_action(state)
            
            # Take action
            state, reward, done, truncated, info = env.step(action)
            score = info['score']
            
            # Render game
            env.snake.draw(env.screen)
            env.food.draw(env.screen)
            pygame.display.flip()
            pygame.time.wait(100)  # Add delay to make it visible
            
            if done or truncated:
                print(f"Game {game + 1} Score: {score}")
                pygame.time.wait(1000)  # Wait a second before next game
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/final_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--games', type=int, default=5,
                      help='Number of games to play')
    args = parser.parse_args()
    
    # Initialize Pygame
    pygame.init()
    
    try:
        play_game(args.model, args.games)
    finally:
        pygame.quit() 