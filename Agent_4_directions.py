import torch
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
from SnakeGameClass_4Directions import SnakeGame, Direction, Point
# from Solution import SnakeGameAI, Direction, Point
# from model import Linear_QNet, QTrainer
from tf_models import create_model
from HelperClasses import Board
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MAX_MEMORY = 100_00
BATCH_SIZE = 32

LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.epsilon_step = 0.001
        self.nframes = 4
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # self.model = create_model()
        self.model = tf.keras.models.load_model("tf_model_most_trained")
        # self.model = tf.keras.models.load_model("tf_model_highest_performer")
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        board = Board(game.w, game.h, 1, game.snake.body, food=game.food)
        
        # Returns a 11 long array with False or True values
        return board.to_tensorflow_tensor().numpy()

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


    def get_random_action(self):
        return random.randint(0,3)
    
    def get_predicted_action(self, state):
        state0 = tf.expand_dims(state, axis=0)
        prediction = self.model(state0)[0]
        # print(prediction)
        move = np.argmax(prediction)
        return move

    
    def get_move_right(self):
        return 1
    
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1.0 - self.epsilon_step*self.n_games
        if random.uniform(0, 1) < max(self.epsilon,0.01):
            move = self.get_random_action()
        else:
            move = self.get_predicted_action(state)
            # print(move)
        return move
    
        
    def get_first_state(self, game, predict=False):
        # Get the first 4 frames and turn them into a state
        first_nframe_states = []
        while len(first_nframe_states) < self.nframes:
            state = self.get_state(game)
            # action = self.get_random_action()
            action = self.get_move_right() # only move right to make sure we don't kill outselves
            reward, done, score = game.step(action=action)
            state_new = self.get_state(game)
            
            if not done:
                first_nframe_states.append(state_new)
            else:
                print("done")
        
        stacked = np.array(first_nframe_states).transpose()
        state = stacked
        
        return state
    def get_init_states(self, env):
        current_state = np.zeros((env.h, env.w, self.nframes))

        for i in range(self.nframes):
            current_state[:,:,i] = env.state
        
        return current_state, current_state



    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Make two vectors with the predictions and the target values with dimension (Batch_size, 4) where 4 are the 4 directions
        targets = self.model.predict(states, verbose=0)
        next_preds = self.model.predict(next_states, verbose=0)
        
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]: # if the game is not done
                Q_next = next_preds[idx]
                Q_new = rewards[idx] + self.gamma * np.max(Q_next)
        
            targets[idx][actions[idx]] = Q_new
            abcd = 10
        
        self.model.fit(states, targets, epochs=1, verbose=1)
        # print("done fitting")
        
        
        
    


def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = SnakeGame()

    while True:
        env.reset()
        current_state, next_state = agent.get_init_states(env)
        done = False
        agent.n_games += 1
        data = pd.DataFrame()


        while not done:
            # Get an action that is either random or prediced based on epsilon
            action = agent.get_action(current_state)

            # Get the new state
            state, reward, done, score = env.step(action=action)

            # reshape shape to be of dimension (1, height, width, 1)
            state = np.reshape(state, (env.h, env.w, 1))
            next_state = np.append(next_state, state, axis=-1)
            next_state = np.delete(next_state, 0, axis=-1)

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
                agent.model.save("tf_model_highest_performer", save_format="h5") # Save the model which has gotten the highest score during the training session
            
            if done:
                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
                data["scores"] = scores


                print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}\t epsilon: {agent.epsilon}")
                if agent.n_games % 100 == 0:
                    agent.model.save("tf_model_most_trained", save_format="h5") # Save the model every 100 games
                    data.to_csv("Data_no_illegal_action2.csv")
            





        # # Get move
        # action = agent.get_action(state)

        # # perform move and get new state
        # state, reward, done, score = env.step(action=action)
        # # print(f"Reward: {reward}\t Done: {done}\t Score: {score}")
        # # print(final_move)
        # state_new_1frame = agent.get_state(game)
        
        # # add the new frame to the end of the frame states
        # state_new = np.append(state, np.expand_dims(state_new_1frame, axis=-1), axis=-1)
        
        # # remove the first element from the state so we still have nframes
        # state_new = np.delete(state_new, 0, axis=-1)


        # # remember
        # agent.remember(state, action, reward, state_new, done)
        
        # experience replay
        # agent.experience_replay()
        
        # state = state_new
        
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
            # plot(plot_scores, plot_mean_scores)












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
            state = np.reshape(state, (env.h, env.w, 1))
            next_state = np.append(next_state, state, axis=-1)
            next_state = np.delete(next_state, 0, axis=-1)


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