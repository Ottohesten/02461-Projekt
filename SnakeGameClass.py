import pygame
import time
import random
from enum import Enum
from collections import namedtuple


SNAKE_SPPED = 15
CLOCK_SPEED = 30
BLOCK_SIZE = 20

WIDTH = 720
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


pygame.init()

class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.score_font = pygame.font.SysFont("times new roman", 20)


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Score
        score_text = self.score_font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()

    def is_collision(self):
        return False



    def move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)



    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction is not Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction is not Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction is not Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction is not Direction.UP:
                    self.direction = Direction.DOWN

        # 2. Move
        self.move(self.direction) # update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            # We remove last element in list because the snake didn't get longer
            self.snake.pop()
        


        # update ui
        self.update_ui()
        self.clock.tick(CLOCK_SPEED)




if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game.step(Direction.DOWN)
        
        
    print('Final Score', score)
        
        
    pygame.quit()


print(Direction.UP)