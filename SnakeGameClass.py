import pygame
import time
import random
import numpy as np
from HelperClasses import Direction, Point, Board, Snake

pygame.init()
font = pygame.font.SysFont('times new roman', 25)

CLOCK_SPEED = 1000
BLOCK_SIZE = 20

WIDTH = 10
HEIGHT = 10


BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)


class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.render_size_modifier = 1000//self.w
        self.display = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.slow = True
        self.pause = False
        self.runai = True
        self.render = True
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.snake = Snake((self.h, self.w))
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.state = np.zeros((self.h, self.w))
        self.place_snake()
        self.place_food()
        return self.state

    def place_snake(self):
        for point in self.snake.body:
            self.state[point.y][point.x] = 0.5
        # self.state[self.snake.head.y][self.snake.head.x] = 0.75

    def place_food(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.food = Point(x, y)
        if self.food in self.snake.body:
            self.place_food()
        self.state[y][x] = 1.0
            
    
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake.head
        # hits boundary
        if pt.x > self.w -1 or pt.x < 0 or pt.y > self.h -1 or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake.body[1:]:
            return True

        return False
    
    def update_state(self):
        new_state = np.zeros((self.h, self.w))
        for point in self.snake.body:
            new_state[point.y][point.x] = 0.5
        
        new_state[self.food.y][self.food.x] = 1.0
        self.state = new_state


    def move(self, action:int):
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # 0 = Straight,    1 = Left,   2 = Right
        if action == 0:
            new_dir = clock_wise[idx]
            self.snake.move(new_dir, self.food)
        elif action == 1:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            self.snake.move(new_dir, self.food)
        elif action == 2:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
            self.snake.move(new_dir, self.food)
        
        self.direction = new_dir

        
    def step(self, action):
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
                elif event.key == pygame.K_x:
                    self.render = not self.render
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        if self.pause:
            pass
        if self.slow:
            self.clock.tick(5)
        
        # Set the previous head to be used in calculating the manhattan distance
        self.previous_head = self.snake.head

        # Move snake
        self.move(action)



        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake.body):
            print("Collision")
            game_over = True
            reward = -10
            return self.state, reward, game_over, self.score

        # 4. Place new food or just move
        if self.snake.head == self.food:
            # print("ate food")
            self.score += 1
            reward = 10
            if not self.score == (self.w * self.h -3):
                self.place_food()
        else:
            # We remove last element in list because the snake didn't get longer
            pass
        
        self.update_state()

        # 5. Update ui
        if self.render:
            self.render_ui()
        self.clock.tick(CLOCK_SPEED)

        if self.previous_head is not None:
            prev_x , prev_y = self.previous_head
            x, y = self.snake.head
            
            # manhattan distance
            distance_before = abs(self.food.x - prev_x) + abs(self.food.y - prev_y)
            distance_after = abs(self.food.x - x) + abs(self.food.y - y)
            
            manhattan = distance_after - distance_before
            if manhattan < 0:
                reward =+ 1
            else:
                reward =- 1
                pass
            
            # print(manhattan)
            abcd = 10
    
        return self.state, reward, game_over, self.score

    def render_ui(self):
        modifier = self.render_size_modifier
        self.display.fill(BLACK)
        for idx, pt in enumerate(self.snake.body):
            if idx == 0:
                pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x*modifier, pt.y*modifier, modifier-2, modifier-2))
            else:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x*modifier, pt.y*modifier, modifier-2, modifier-2))
            # pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x*modifier, self.food.y*modifier, modifier, modifier))

        # Score
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()












if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        reward, game_over, score = game.step([1,0,0])
        if game_over:
            # game.reset()
            pass
        
        
    # print('Final Score', score)
        
        
    # pygame.quit()

