import pygame
import sys
import random
import numpy as np
from pygame.math import Vector2

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 40
GRID_SIZE = 20
SCREEN_WIDTH = CELL_SIZE * GRID_SIZE
SCREEN_HEIGHT = CELL_SIZE * GRID_SIZE

# Colors
BACKGROUND_COLOR = (40, 40, 40)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
SCORE_COLOR = (255, 255, 255)

class Snake:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(1, 0)
        self.new_block = False
        
    def draw(self, screen):
        for block in self.body:
            block_rect = pygame.Rect(block.x * CELL_SIZE, block.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, SNAKE_COLOR, block_rect)
    
    def move(self):
        if self.new_block:
            body_copy = self.body[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy
    
    def grow(self):
        self.new_block = True
    
    def check_collision(self):
        head = self.body[0]
        # Check wall collision
        if not (0 <= head.x < GRID_SIZE and 0 <= head.y < GRID_SIZE):
            return True
        # Check self collision
        if head in self.body[1:]:
            return True
        return False

class Food:
    def __init__(self, snake):
        self.position = self.generate_random_pos(snake)
        
    def draw(self, screen):
        food_rect = pygame.Rect(self.position.x * CELL_SIZE, self.position.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, FOOD_COLOR, food_rect)
    
    def generate_random_pos(self, snake):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            pos = Vector2(x, y)
            if pos not in snake.body:
                return pos

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food(self.snake)
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        
    def update(self):
        self.snake.move()
        self.check_collision()
        self.check_food()
    
    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.snake.draw(self.screen)
        self.food.draw(self.screen)
        self.draw_score()
        pygame.display.flip()
    
    def check_collision(self):
        if self.snake.check_collision():
            self.game_over()
    
    def check_food(self):
        if self.snake.body[0] == self.food.position:
            self.food = Food(self.snake)
            self.snake.grow()
            self.score += 1
    
    def game_over(self):
        pygame.quit()
        sys.exit()
    
    def draw_score(self):
        score_text = self.font.render(f'Score: {self.score}', True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))

def main():
    game = Game()
    SCREEN_UPDATE = pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE, 150)  # Update every 150ms
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == SCREEN_UPDATE:
                game.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and game.snake.direction.y != 1:
                    game.snake.direction = Vector2(0, -1)
                if event.key == pygame.K_DOWN and game.snake.direction.y != -1:
                    game.snake.direction = Vector2(0, 1)
                if event.key == pygame.K_LEFT and game.snake.direction.x != 1:
                    game.snake.direction = Vector2(-1, 0)
                if event.key == pygame.K_RIGHT and game.snake.direction.x != -1:
                    game.snake.direction = Vector2(1, 0)
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        
        game.draw()
        game.clock.tick(60)

if __name__ == '__main__':
    main() 