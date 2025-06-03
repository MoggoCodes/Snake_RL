import gymnasium as gym
import numpy as np
from pygame.math import Vector2
from snake_game import Snake, Food, GRID_SIZE, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, BACKGROUND_COLOR
import pygame

class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Initialize Pygame and create screen
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake RL')
        
        # Initialize font for episode counter
        self.font = pygame.font.Font(None, 24)  # None uses default system font
        self.current_episode = 0
        
        # Define action space (0: up, 1: right, 2: down, 3: left)
        self.action_space = gym.spaces.Discrete(4)
        
        # Define observation space (grid_size x grid_size x 5)
        # Channel 1: Snake body
        # Channel 2: Snake head
        # Channel 3: Food location
        # Channel 4: Horizontal distance to food (negative = left, positive = right)
        # Channel 5: Vertical distance to food (negative = up, positive = down)
        self.observation_space = gym.spaces.Box(
            low=-GRID_SIZE, high=GRID_SIZE,
            shape=(GRID_SIZE, GRID_SIZE, 5),
            dtype=np.float32
        )
        
        self.snake = None
        self.food = None
        self.score = 0
        self.steps = 0
        self.max_steps = GRID_SIZE * GRID_SIZE * 2
        self.clock = pygame.time.Clock()
    
    def set_episode(self, episode):
        """Set the current episode number"""
        self.current_episode = episode
    
    def _draw_episode_counter(self):
        """Draw semi-transparent episode counter in the corner"""
        try:
            text = f"Episode: {self.current_episode}"
            text_surface = self.font.render(text, True, (255, 255, 255))
            
            # Create a semi-transparent surface for the text background
            text_bg = pygame.Surface(text_surface.get_size())
            text_bg.fill(BACKGROUND_COLOR)
            text_bg.set_alpha(128)  # 128 is semi-transparent (0 is invisible, 255 is solid)
            
            # Position in top-right corner with small padding
            padding = 10
            pos_x = SCREEN_WIDTH - text_surface.get_width() - padding
            pos_y = padding
            
            # Draw background and text
            self.screen.blit(text_bg, (pos_x, pos_y))
            self.screen.blit(text_surface, (pos_x, pos_y))
        except Exception as e:
            print(f"Error drawing episode counter: {e}")

    def _ensure_valid_position(self, pos):
        """Ensure position is within grid bounds"""
        x = max(0, min(int(pos.x), GRID_SIZE - 1))
        y = max(0, min(int(pos.y), GRID_SIZE - 1))
        return Vector2(x, y)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        try:
            self.snake = Snake()
            self.food = Food(self.snake)
            self.score = 0
            self.steps = 0
            
            # Clear the screen
            self.screen.fill(BACKGROUND_COLOR)
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            self._draw_episode_counter()
            pygame.display.flip()
            
            return self._get_observation(), {}
        except Exception as e:
            print(f"Error in reset: {e}")
            self.close()
            raise
    
    def _get_food_distance(self, head_pos, food_pos):
        """Get actual distance to food in both dimensions"""
        dx = food_pos.x - head_pos.x
        dy = food_pos.y - head_pos.y
        return dx, dy
    
    def _get_observation(self):
        try:
            obs = np.zeros((GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
            
            # Set snake body
            for block in self.snake.body[1:]:
                pos = self._ensure_valid_position(block)
                obs[int(pos.y)][int(pos.x)][0] = 1.0
            
            # Set snake head
            head = self._ensure_valid_position(self.snake.body[0])
            obs[int(head.y)][int(head.x)][1] = 1.0
            
            # Set food
            food_pos = self._ensure_valid_position(self.food.position)
            obs[int(food_pos.y)][int(food_pos.x)][2] = 1.0
            
            # Set distance to food
            dx, dy = self._get_food_distance(head, food_pos)
            
            # Fill distance channels with the actual distance values
            obs[:, :, 3] = dx  # Horizontal distance channel
            obs[:, :, 4] = dy  # Vertical distance channel
            
            return obs.astype(np.float32)
        except Exception as e:
            print(f"Error in get_observation: {e}")
            return np.zeros((GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
    
    def _get_reward(self, collision, ate_food, changed_direction, distance_delta):
        reward = 0.0
        
        # Major rewards/penalties
        if collision:
            reward -= 10.0  # Heavy penalty for dying
        if ate_food:
            reward += 25.0  # Major reward for eating food
        
        # Distance-based reward
        reward += distance_delta * 0.2  # Increased weight for distance-based reward
        
        # Minor rewards/penalties
        if changed_direction:
            reward -= 0.1  # Reduced penalty for direction changes
        
        # Living reward
        reward += 0.1  # Small reward for staying alive
        
        return reward
    
    def _get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def step(self, action):
        try:
            self.steps += 1
            
            # Handle Pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None, 0, True, False, {"score": self.score}
            
            # Get current direction and position
            old_direction = self.snake.direction
            old_distance = self._get_manhattan_distance(self.snake.body[0], self.food.position)
            
            # Convert action to direction
            if action == 0:  # Up
                new_direction = Vector2(0, -1)
            elif action == 1:  # Right
                new_direction = Vector2(1, 0)
            elif action == 2:  # Down
                new_direction = Vector2(0, 1)
            else:  # Left
                new_direction = Vector2(-1, 0)
            
            # Check if direction changed
            changed_direction = new_direction != old_direction
            
            # Don't allow 180-degree turns
            if new_direction != -old_direction:
                self.snake.direction = new_direction
            
            # Move snake
            self.snake.move()
            
            # Calculate new distance and distance change
            new_distance = self._get_manhattan_distance(self.snake.body[0], self.food.position)
            distance_delta = old_distance - new_distance  # Positive if moved closer
            
            # Check collision
            collision = self.snake.check_collision()
            
            # Check if food was eaten
            ate_food = False
            if self.snake.body[0] == self.food.position:
                self.food = Food(self.snake)
                self.snake.grow()
                self.score += 1
                ate_food = True
            
            # Update display
            self.screen.fill(BACKGROUND_COLOR)
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            self._draw_episode_counter()
            pygame.display.flip()
            
            # Get reward
            reward = self._get_reward(collision, ate_food, changed_direction, distance_delta)
            
            # Check if game is done
            done = collision or self.steps >= self.max_steps
            
            # Control game speed
            self.clock.tick(30)
            
            return self._get_observation(), reward, done, False, {"score": self.score}
        
        except Exception as e:
            print(f"Error in step: {e}")
            self.close()
            return self._get_observation(), -1.0, True, False, {"score": self.score}
    
    def close(self):
        try:
            pygame.quit()
        except:
            pass
        super().close() 