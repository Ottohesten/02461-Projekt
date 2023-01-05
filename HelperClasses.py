from SnakeGameClass import SnakeGame
import numpy as np
import pandas as pd
from SnakeGameClass import Direction, Point
from collections import deque
game = SnakeGame()
directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

class Board:
    def __init__(self, width:int, height:int, block_size:int, snake=None, snake_head=None):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.twidth = width // block_size
        self.theight = height // block_size
        self.original_snake = snake
        self.snake = self.transform_snake()
        self.tsnake = self.transform_snake()
        self.snake_head = self.snake[0]
        self.board = self.generate_board()
        
    @property
    def shape(self):
        if self.width % self.block_size != 0:
            raise ValueError("Width must be a multiple of block size")
        elif self.height % self.block_size != 0:
            raise ValueError("Height must be a multiple of block size")
        return (self.height // self.block_size, self.width // self.block_size)
    
    
    def empty_board(self):
        return np.zeros(self.shape, dtype=int)
    
    def transform_snake(self):
        return [Point(x=x//self.block_size, y=y//self.block_size) for x,y in self.original_snake]
    
    def generate_board(self):
        board = self.empty_board()
        for point in self.snake:
            # x, y = int(point.x)//self.block_size , int(point.y)//self.block_size
            x, y = int(point.x), int(point.y)
            board[y, x] = 1
        return board
    
    def valid_cell(self, point):
        if int(point.x) < 0 or int(point.x) >= self.twidth or int(point.y) < 0 or int(point.y) >= self.theight:
            return False
        elif point in self.snake:
            return False
        elif self.board[int(point.y), int(point.x)] == 1:
            return False
        else:
            return True
        
    def get_neighbours(self, point):
        neighbours = []
        for direction in directions:
            if direction == Direction.UP:
                new_point = Point(x=point.x, y=point.y-1)
                if self.valid_cell(new_point):
                    neighbours.append(new_point)
            elif direction == Direction.DOWN:
                new_point = Point(x=point.x, y=point.y+1)
                if self.valid_cell(new_point):
                    neighbours.append(new_point)
            elif direction == Direction.LEFT:
                new_point = Point(x=point.x-1, y=point.y)
                if self.valid_cell(new_point):
                    neighbours.append(new_point)
            elif direction == Direction.RIGHT:
                new_point = Point(x=point.x+1, y=point.y)
                if self.valid_cell(new_point):
                    neighbours.append(new_point)
        return neighbours
        
        
        
    def get_available_space_direction(self, direction): #TODO Return the available space for a given direction
        visited = set()
        Q = deque()
        x,y = self.snake_head
        if direction == Direction.UP:
            Q.append(Point(x=x,y=y-1))
        elif direction == Direction.DOWN:
            Q.append(Point(x=x,y=y+1))
        elif direction == Direction.LEFT:
            Q.append(Point(x=x-1,y=y))
        elif direction == Direction.RIGHT:
            Q.append(Point(x=x+1,y=y))
        # print(f"{Q=}\n{visited=}")
        while Q:
            point = Q.popleft()
            if not self.valid_cell(point):
                # print("Initial point not valid")
                break
            neighbours = self.get_neighbours(point)
            # print(point)
            # print(neighbours)
            for neighbour in neighbours:
                if neighbour not in visited:
                    Q.append(neighbour)
                    visited.add(neighbour)
        return len(visited)
            
    def get_available_space(self):
        available_space = {direction.name: self.get_available_space_direction(direction) for direction in directions}
        return available_space
        
    
    def __repr__(self):
        return str(pd.DataFrame(self.board, dtype=int))
    

        
            

if __name__ == "__main__":
    snake = [Point(i,10) for i in range(0, 20)]
    board = Board(20, 20, 1, snake=snake)
    print(board)
    visited = {Point(3,4), Point(4,4)}
    if Point(3,4) in visited:
        print("yes")
    
    # visited = {1,2}
    


# print(game.snake)



# TODO implement snake class to be played with

