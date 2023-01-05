from SnakeGameClass import SnakeGame
import numpy as np
import pandas as pd
game = SnakeGame()


class Board:
    def __init__(self, width:int, height:int, block_size:int, snake=None):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.snake = snake
        self.board = self.generate_board()
        
    @property
    def shape(self):
        if self.width % self.block_size != 0:
            raise ValueError("Width must be a multiple of block size")
        elif self.height % self.block_size != 0:
            raise ValueError("Height must be a multiple of block size")
        return (self.width // self.block_size, self.height // self.block_size)
    
    
    def empty_board(self):
        return np.zeros(self.shape)
    
    def generate_board(self):
        board = self.empty_board()
        for point in self.snake:
            x, y = int(point.x)//20 , int(point.y)//20
            board[x, y] = 1
        return board
        
    def get_available_space(self): #TODO Return the available space for all 3 actions
        pass
        
    
    def __repr__(self):
        return str(pd.DataFrame(self.board))
    

        
            


        
# snake = game.snake
# print(snake)

# board = Board(640, 480, 20, snake)

# print(board)


# print(game.snake)



# TODO implement snake class to be played with

