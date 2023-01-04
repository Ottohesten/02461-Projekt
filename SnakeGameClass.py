import pygame
import time
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('times new roman', 25)

CLOCK_SPEED = 200
BLOCK_SIZE = 20

WIDTH = 640
HEIGHT = 480


BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', ["x", "y"])


class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.slow = True
        self.pause = False
        self.runai = True
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.player_direction = None
        self.change_to = self.direction
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        # print("Started new game")

    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()
            
    # def step(self, action):
    #     self.frame_iteration += 1
    #     # 1. collect user input
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
        
    #     # 2. move
    #     self.move(action) # update the head
    #     self.snake.insert(0, self.head)
        
    #     # 3. check if game over
    #     reward = 0
    #     game_over = False
    #     if self.is_collision() or self.frame_iteration > 100*len(self.snake):
    #         game_over = True
    #         reward = -10
    #         return reward, game_over, self.score

    #     # 4. place new food or just move
    #     if self.head == self.food:
    #         self.score += 1
    #         reward = 10
    #         self.place_food()
    #     else:
    #         self.snake.pop()
        
    #     # 5. update ui and clock
    #     self.update_ui()
    #     self.clock.tick(CLOCK_SPEED)
    #     # 6. return game over and score
    #     return reward, game_over, self.score
        
    def step(self, action=None, player_direction=None):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.change_to = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.change_to = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.change_to = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.change_to = Direction.DOWN
                elif event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_s:
                    self.slow = not self.slow
                elif event.key == pygame.K_a:
                    self.runai = not self.runai
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        if self.pause:
            pass
        if self.slow:
            self.clock.tick(10)
        
        
        # if self.change_to == Direction.UP and self.direction != Direction.DOWN:
        #     self.direction = Direction.UP
        # if self.change_to == Direction.DOWN and self.direction != Direction.UP:
        #     self.direction = Direction.DOWN
        # if self.change_to == Direction.LEFT and self.direction != Direction.RIGHT:
        #     self.direction = Direction.LEFT
        # if self.change_to == Direction.RIGHT and self.direction != Direction.LEFT:
        #     self.direction = Direction.RIGHT
        

        # 2. Move
        if action:
            self.move(action)
        else:
            print("no action")
            self.move(None,direction=self.direction) # update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            # print("Collision")
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            # print("ate food")
            self.score += 10
            reward = 10
            self.place_food()
        else:
            # We remove last element in list because the snake didn't get longer
            self.snake.pop()
        


        # 5. Update ui
        self.update_ui()
        self.clock.tick(CLOCK_SPEED)
    
        return reward, game_over, self.score

    
    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Score
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False



    def move(self, action):
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir
        
            
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)







if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        reward, game_over, score = game.step()
        if game_over:
            # game.reset()
            pass
        
        
    # print('Final Score', score)
        
        
    # pygame.quit()

