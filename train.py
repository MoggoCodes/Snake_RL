import torch
import numpy as np
from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import os
import pygame
import sys

def plot_scores(scores, avg_scores, filename='learning_curve.png'):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(scores, label='Score')
        plt.plot(avg_scores, label='Average Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Error plotting scores: {e}")

def validate_state(state):
    """Validate state tensor for training"""
    if state is None:
        return False
    if not isinstance(state, np.ndarray):
        return False
    if len(state.shape) != 3:
        return False
    if np.isnan(state).any() or np.isinf(state).any():
        return False
    return True

def train(n_episodes=1000, max_steps=1000):
    env = None
    try:
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
        
        # Training metrics
        scores = []
        avg_scores = []
        best_score = 0
        score_window = deque(maxlen=100)
        
        for episode in range(n_episodes):
            try:
                # Update episode counter in environment
                env.set_episode(episode + 1)
                
                state, _ = env.reset()
                if not validate_state(state):
                    print(f"Invalid state after reset in episode {episode + 1}")
                    continue
                
                episode_score = 0
                
                for step in range(max_steps):
                    # Handle Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining interrupted by user")
                            return agent, scores, avg_scores
                    
                    # Select action
                    action = agent.select_action(state)
                    
                    # Take action
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # Validate next_state
                    if not validate_state(next_state):
                        print(f"Invalid next_state in episode {episode}, step {step}")
                        break
                    
                    # Store transition in memory
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    # Train agent
                    agent.train_step()
                    
                    episode_score += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                # Update metrics
                score_window.append(info['score'])
                avg_score = np.mean(list(score_window))
                scores.append(info['score'])
                avg_scores.append(avg_score)
                
                # Save best model
                if avg_score > best_score:
                    best_score = avg_score
                    try:
                        agent.save('best_model.pth')
                    except Exception as e:
                        print(f"Error saving best model: {e}")
                
                # Print progress
                if (episode + 1) % 10 == 0:
                    print(f"Episode: {episode + 1}, Score: {info['score']}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
                    
                # Plot learning curve
                if (episode + 1) % 100 == 0:
                    plot_scores(scores, avg_scores)
                    
                # Save checkpoint every 100 episodes
                if (episode + 1) % 100 == 0:
                    try:
                        agent.save(f'models/checkpoint_{episode+1}.pth')
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
            
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        return agent, scores, avg_scores
    
    except Exception as e:
        print(f"Critical error during training: {e}")
        return None, [], []
    
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    try:
        # Train the agent
        agent, scores, avg_scores = train()
        
        # Save final model and plot
        if agent is not None:
            try:
                agent.save('models/final_model.pth')
                plot_scores(scores, avg_scores, 'models/final_learning_curve.png')
                print("Training completed! Final model saved as 'models/final_model.pth'")
            except Exception as e:
                print(f"Error saving final results: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()
        sys.exit() 