import torch
import random
import numpy as np
from collections import deque
from HelperClasses import Direction, Point, Board
from SnakeGameClass import SnakeGame
# from Solution import SnakeGame
from model import Linear_QNet, QTrainer
import time

MAX_MEMORY = 100_00
BATCH_SIZE = 32
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(10, 32, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake.head

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        board = Board(game.w, game.h, game.snake)
        available_space = board.get_available_space_3directions(game.direction)
        distances = board.manhattan_distance_3directions(game.direction, game.food)

        state = [
            # # Danger straight
            # (dir_r and game.is_collision(point_r)) or 
            # (dir_l and game.is_collision(point_l)) or 
            # (dir_u and game.is_collision(point_u)) or 
            # (dir_d and game.is_collision(point_d)),

            # # Danger right
            # (dir_u and game.is_collision(point_r)) or 
            # (dir_d and game.is_collision(point_l)) or 
            # (dir_l and game.is_collision(point_u)) or 
            # (dir_r and game.is_collision(point_d)),

            # # Danger left
            # (dir_d and game.is_collision(point_r)) or 
            # (dir_u and game.is_collision(point_l)) or 
            # (dir_r and game.is_collision(point_u)) or 
            # (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_r,
            dir_l,
            dir_u,
            dir_d,
            
            # Food location 
            # game.food.x > game.snake.head.x,  # food right
            # game.food.x < game.snake.head.x,  # food left
            # game.food.y < game.snake.head.y,  # food up
            # game.food.y > game.snake.head.y  # food down
            ] + list(available_space.values()) + list(distances.values())

        # Returns a 11 long array with False or True values
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 500 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 501) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            abcd = 10
            # print("random move")
            # print(final_move)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            abcd = 10
            # print("decided move")
            # print(prediction)
            # print(final_move)

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    # game = SnakeGameAI()'
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.step(action=final_move)
        # reward, done, score = game.play_step(action=final_move)
        # print(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if agent.n_games > 1000:
            abcd = 10
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)
            print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
    pass