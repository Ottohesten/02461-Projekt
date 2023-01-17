import torch
import random
import numpy as np
import pandas as pd
from collections import deque
from SnakeGameClass_4Directions import SnakeGame, Direction, Point
# from Solution import SnakeGameAI, Direction, Point
from model import Conv_QNet
from HelperClasses import Board
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MAX_MEMORY = 50_000
BATCH_SIZE = 32

LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1 # randomness
        self.epsilon_step = 0.0001
        self.nframes = 4
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet()
        # self.model.load_state_dict(torch.load("saved_models/highest_trained.pth"))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss = torch.nn.MSELoss()
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        board = Board(game.w, game.h, game.snake, food=game.food)
        
        # Returns a 11 long array with False or True values
        return board.to_tensorflow_tensor().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def get_random_action(self):
        return random.randint(0,3)
    
    def get_predicted_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state0)
        # print(prediction)
        move = torch.argmax(prediction).item()
        return move
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.epsilon_start - self.epsilon_step*self.n_games
        if random.uniform(0, 1) < max(self.epsilon,0.005):
            move = self.get_random_action()
        else:
            move = self.get_predicted_action(state)
            # print(move)
        return move
    
    def get_init_states(self, env):
        current_state = np.zeros((self.nframes, env.h, env.w))

        for i in range(self.nframes):
            current_state[i,:,:] = env.state
        
        return current_state, current_state



    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        
        # Make two vectors with the predictions and the target values with dimension (Batch_size, 4) where 4 are the 4 directions
        preds = self.model(states)
        targets = preds.clone()
        next_preds = self.model(next_states)
        
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]: # if the game is not done
                Q_next = next_preds[idx]
                Q_new = rewards[idx] + self.gamma * torch.max(Q_next).item()
        
            targets[idx][actions[idx]] = Q_new
            abcd = 10
        
        self.optimizer.zero_grad()
        loss = self.loss(preds, targets)
        # print(loss.item())
        loss.backward()
        self.optimizer.step()
        # print("done fitting")
        
        

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = SnakeGame()
    model_filepath = "saved_models/"
    data_filepath = "data/conv_score_data.csv"

    while True:
        env.reset()
        current_state, next_state = agent.get_init_states(env)
        done = False
        agent.n_games += 1



        while not done:
            # Get an action that is either random or prediced based on epsilon
            action = agent.get_action(current_state)

            # Get the new state
            state, reward, done, score = env.step(action=action)

            # reshape shape to be of dimension (1, height, width, 1)
            state = np.reshape(state, (1, env.h, env.w))
            next_state = np.append(next_state, state, axis=0)
            next_state = np.delete(next_state, 0, axis=0)

            if score >= 1:
                abcd = 10
            

            # Remember the parameters for trainging
            
            agent.remember(current_state, action, reward, next_state, done)

            # This is where we train the network
            agent.experience_replay()

            # We set the current state to the next state so that in the next iteration we start with this state
            current_state = next_state

            if score > record:
                record = score
                torch.save(agent.model.state_dict(), f"{model_filepath}highest_performer.pth")  # Save the model which has gotten the highest score during the training session
            
            if done:
                with open(data_filepath, "a") as data_file:
                    data_file.write("\n")
                    data_file.write(str(score))


                print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}\t epsilon: {agent.epsilon}")
                if agent.n_games % 100 == 0:
                    torch.save(agent.model.state_dict(), f"{model_filepath}highest_trained.pth") # Save the model every 100 games
                    # data.to_csv("data/Data_no_illegal_action3.csv")
                    pass
                if agent.n_games == 1000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}conv_1000.pth")
                if agent.n_games == 10000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}conv_10000.pth")
                if agent.n_games == 25000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}conv_25000.pth")
                if agent.n_games == 50000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}conv_50000.pth")

            




# TEST THE MODEL BY LOADING IT IN FROM FILE AND NOT TRAINING IT WHILE RUNNING
def test():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = SnakeGame()


    while True:
        env.reset()
        current_state, next_state = agent.get_init_states(env)
        done = False
        
        agent.n_games += 1

        while not done:
            # Get an action that is either random or prediced based on epsilon
            action = agent.get_predicted_action(current_state)

            # Get the new state
            state, reward, done, score = env.step(action=action)

            # reshape shape to be of dimension (1, height, width, 1)
            state = np.reshape(state, (1, env.h, env.w))
            next_state = np.append(next_state, state, axis=0)
            next_state = np.delete(next_state, 0, axis=0)


            # We set the current state to the next state so that in the next iteration we start with this state
            current_state = next_state

            if score > record:
                record = score

            if done:
                print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}")
        
        # if agent.n_games > 1000:
        #     abcd = 10
        # if done:
        #     # train long memory, plot result
        #     game.reset()
        #     state = agent.get_first_state(game)
        #     agent.n_games += 1
        #     # agent.train_long_memory()

        #     if score > record:
        #         record = score
        #         agent.model.save("tf_model")

        #     # print('Game', agent.n_games, 'Score', score, 'Record:', record)
        #     print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}")

        #     plot_scores.append(score)
        #     total_score += score
        #     mean_score = total_score / agent.n_games
        #     plot_mean_scores.append(mean_score)
        #     # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
    # test()
    pass