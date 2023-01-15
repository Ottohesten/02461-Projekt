import torch
import random
import numpy as np
from collections import deque
from HelperClasses import Direction, Point, Board
from SnakeGameClass import SnakeGame
# from Solution import SnakeGame
from model import Linear_QNet, QTrainer, Conv_QNet, ConvQtrainer
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.epsilon_step = 0.0005
        self.epsilon_min = 0.01
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet(2, 16, (5,5))
        self.trainer = ConvQtrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        board = Board(game.w, game.h, 1, game.snake.body, food=game.food)
        
        # Returns a 11 long array with False or True values
        return board.to_tensor(channels=1)
    
    def get_states(self, game):
        pass


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        test = 10
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state_stacked):
        # stack the frames
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1 - self.epsilon_step*self.n_games
        final_move = [0,0,0]
        if random.uniform(0, 1) < max(self.epsilon,0.01):
            move = random.randint(0, 2)
            final_move[move] = 1
            abcd = 10
            # print("random move")
            # print(final_move)
        else:
            state0 = state_stacked.unsqueeze(0)
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
    first_iteration = True
    while True:
        if first_iteration:
            state_old = agent.get_state(game)
            reward, done, score = game.step(action=[1,0,0])

            

        # get old state
        state_current = agent.get_state(game)

        # Stack the two states to CWH dimension with c = 2, w,h = width,heigh
        stack_1 = torch.stack((state_old, state_current))

        # get move from 2 states so we can tell direction
        final_move = agent.get_action(stack_1)

        # perform move and get new state
        reward, done, score = game.step(action=final_move)
        # reward, done, score = game.play_step(action=final_move)
        # print(final_move)

        # Get new state after doing action
        state_new = agent.get_state(game)

        # Stack the new state with the current so we get state 0-1, stacked and state 1-2 stacked
        stack_2 = torch.stack((state_current, state_new))

        # train short memory
        agent.train_short_memory(stack_1, final_move, reward, stack_2, done)

        # remember
        agent.remember(stack_1, final_move, reward, stack_2, done)

        # Make state current the stack old, so when the loop goes again it starts with getting current state which will be the new state
        state_old = state_current


        if agent.n_games > 3000:
            abcd = 10
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

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