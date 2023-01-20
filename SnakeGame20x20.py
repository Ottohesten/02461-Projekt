import numpy as np

class SnakeGame():
    def __init__(self):
        """Initializes gamestate"""
        
        self.reward = 0
        self.done = False
        #self.board = np.array([[3., 3., 3., 3., 3., 3., 3.],
        #                       [3., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 1., 1., 4., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 3.],
        #                       [3., 3., 3., 3., 3., 3., 3.]], dtype=np.float32)
        
        #self.board = np.array([[3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 1., 1., 4., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        #                       [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]], dtype=np.float32)
        
        self.board = np.array([[3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
                               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]], dtype=np.float32)
        self.randomize_fruit()
        self.snake_pos = np.array([[10, 10],
                                   [10, 11], 
                                   [10, 12]])
    def randomize_fruit(self):
        """Randomizes the position of the fruit to an availible cell"""
        
        self.fruit_x = np.random.randint(1, 21)
        self.fruit_y = np.random.randint(1, 21)
        
        while self.board[self.fruit_x, self.fruit_y] != 0.:
            self.fruit_x = np.random.randint(1, 21)
            self.fruit_y = np.random.randint(1, 21)
        
        self.board[self.fruit_x, self.fruit_y] = 2.
    
    
    def step(self, action):
        """Steps the game, moving the snake, giving rewards and updating game"""
        
        self.reward = 0
        self.done = False

        temporary_move = self.snake_pos[-1].copy()
        if action == 0:
            temporary_move[1] += 1
        elif action == 1:
            temporary_move[1] -= 1
        elif action == 2:
            temporary_move[0] += 1
        elif action == 3:
            temporary_move[0] -= 1
        self.snake_pos = np.vstack((self.snake_pos, temporary_move))
        if self.board[temporary_move[0], temporary_move[1]] == 2.:
            self.reward = 4
            
            self.randomize_fruit()
            
        elif self.board[temporary_move[0], temporary_move[1]] == 1. or self.board[temporary_move[0], temporary_move[1]] == 3.:
            self.reward = -2
            self.done = True
            
        else:
            # Reward is -0.03 for 5x5, -0.15 for 10x10, -0.1 for 20x20
            self.reward = -0.1
            self.board[self.snake_pos[0, 0], self.snake_pos[0, 1]] = 0 
            self.snake_pos = self.snake_pos[1:]
            
        for body_block in self.snake_pos[:-1]:
            self.board[body_block[0], body_block[1]] = 1
           
        self.board[self.snake_pos[-1, 0], self.snake_pos[-1, 1]] = 4

        return self.board.flatten(), self.reward, self.done
