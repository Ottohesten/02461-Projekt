import torch
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from enum import Enum
import matplotlib.pyplot as plt

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
Point = namedtuple('Point', ["x", "y"])


class Snake:
    def __init__(self, board_shape):
        self.board_w = board_shape[1]
        self.board_h = board_shape[0]
        self.body:list[Point] = self.generate_snake_body()

    
    def generate_snake_body(self):
        x,y = self.board_w//2, self.board_h//2 
        return [
            Point(x=x, y=y),
            Point(x=x-1, y=y),
            Point(x=x-2, y=y),
        ]
    @property
    def head(self):
        return self.body[0]
    
    def move(self, direction, food=None):
        if direction == Direction.RIGHT:
            new_head = Point(x=self.head.x+1, y=self.head.y)
            self.body.insert(0, new_head)
        elif direction == Direction.LEFT:
            new_head = Point(x=self.head.x-1, y=self.head.y)
            self.body.insert(0, new_head)
        elif direction == Direction.UP:
            new_head = Point(x=self.head.x, y=self.head.y-1)
            self.body.insert(0, new_head)
        elif direction == Direction.DOWN:
            new_head = Point(x=self.head.x, y=self.head.y+1)
            self.body.insert(0, new_head)
        if food:
            if food == new_head:
                # print("ate food in snake move function")
                pass
            else:
                self.body.pop()


    
    def remove_last(self):
        self.body.pop()
    
    

    def __repr__(self):
        return str(self.body)


class Board:
    def __init__(self, width:int, height:int, snake:Snake=None, food:Point=None):
        self.width = width
        self.height = height
        self.food = food
        self.snake = snake
        self.snake_head = self.snake.head
        self.board = self.generate_board()
        
    @property
    def shape(self):
        return (self.height, self.width)
    
    
    def empty_board(self):
        return np.zeros(self.shape)

    def generate_board(self):
        board = self.empty_board()
        for point in self.snake.body:
            x, y = int(point.x), int(point.y)
            # print(f'x = {x}, y = {y}')
            if x >= self.width or y >= self.height:
                pass
            else:
                # The cells where the snake is will be denoted by 0.5
                board[y, x] = 0.5
        
        if self.food is not None and self.food not in self.snake.body:
            x, y = int(self.food.x), int(self.food.y)
            board[y, x] = 1.0
        
        return board
    
    def valid_cell(self, point):
        if int(point.x) < 0 or int(point.x) >= self.width or int(point.y) < 0 or int(point.y) >= self.height:
            return False
        elif point in self.snake.body:
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
        
        
        
    def get_available_space_direction(self, direction): # Return the available space for a given direction using breadth first search
        visited = set()
        Q = deque()
        x,y = self.snake.head
        if direction == Direction.UP:
            Q.append(Point(x=x,y=y-1))
        elif direction == Direction.DOWN:
            Q.append(Point(x=x,y=y+1))
        elif direction == Direction.LEFT:
            Q.append(Point(x=x-1,y=y))
        elif direction == Direction.RIGHT:
            Q.append(Point(x=x+1,y=y))

        while Q:
            point = Q.popleft()
            if not self.valid_cell(point):
                # print("Initial point not valid")
                break
            neighbours = self.get_neighbours(point)
            for neighbour in neighbours:
                if neighbour not in visited:
                    Q.append(neighbour)
                    visited.add(neighbour)
        return len(visited)
            
    def get_available_space(self): # Returns a dictionary with the available space for each direction, should return 0 if the next move in that way will result in dying.
        available_space = {direction.name: self.get_available_space_direction(direction) for direction in directions}
        return available_space
        
    def get_available_space_3directions(self, moving_direction):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(moving_direction)

        straight = clock_wise[idx]
        right_turn = clock_wise[(idx+1)%4]
        left_turn = clock_wise[(idx-1)%4]
        available_space = {straight: self.get_available_space_direction(straight), left_turn: self.get_available_space_direction(left_turn), right_turn: self.get_available_space_direction(right_turn) }
        return available_space

    def manhattan_distance(self, direction, food):
        x, y = self.snake.head
        if direction == Direction.UP:
            point = Point(x=x,y=y-1)
        elif direction == Direction.DOWN:
            point = Point(x=x,y=y+1)
        elif direction == Direction.LEFT:
            point = Point(x=x-1,y=y)
        elif direction == Direction.RIGHT:
            point = Point(x=x+1,y=y)
        
        dist = abs(point.x - food.x) + abs(point.y - food.y)
        return dist

    
    
    def manhattan_distance_3directions(self, moving_direction, food):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(moving_direction)

        straight = clock_wise[idx]
        right_turn = clock_wise[(idx+1)%4]
        left_turn = clock_wise[(idx-1)%4]

        distances = {straight: self.manhattan_distance(straight, food), left_turn: self.manhattan_distance(left_turn, food), right_turn: self.manhattan_distance(right_turn, food)}
        return distances

    def to_tensor(self, channels=3):
        if channels == 1:
            return torch.tensor(self.board).to(torch.float32)
        board_tensor = np.zeros((3, self.height, self.width), dtype=int)
        self.board_tensor = board_tensor
        
        
        for point in self.snake.body:
            if point.x >= self.width or point.y >= self.height:
                pass
            else:
                board_tensor[1, point.y, point.x] = 1
        
        board_tensor[2, self.food.y, self.food.x] = 1
        
        temp = np.array(board_tensor[1] + board_tensor[2],dtype=int)
        board_tensor[0] = temp^(temp&1==temp)
        
        return torch.tensor(board_tensor).to(torch.float32)
    
    def __repr__(self):
        return str(pd.DataFrame(self.board))
    

        
            

# if __name__ == "__main__":
#     snake = [Point(i,10) for i in range(0, 20)]
#     board = Board(20, 20, 1, snake=snake)
#     print(board)
#     visited = {Point(3,4), Point(4,4)}
#     if Point(3,4) in visited:
#         print("yes")
    
#     # visited = {1,2}
    


# print(game.snake)



# TODO implement snake class to be played with




def stack_frames(f1, f2):
    return torch.stack((f1,f2))


def drawnow(data):
    plt.ion()
    plt.plot(data, ".")
    plt.show()



    

'''

if __name__ == "__main__":
    snake = Snake((10,10))
    print(snake.body)
    
'''


