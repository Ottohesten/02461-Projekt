import torch
import random
import numpy as np
import tensorflow as tf
from collections import deque
from SnakeGameClass_4Directions import SnakeGame, Direction, Point
# from Solution import SnakeGameAI, Direction, Point
# from model import Linear_QNet, QTrainer
from tf_models import create_model
from HelperClasses import Board


MAX_MEMORY = 100_000
BATCH_SIZE = 200
INIT_MEMORY = 1000

LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.epsilon_step = 0.001
        self.nframes = 2
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = create_model()
        self.target_model = create_model()
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
    
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1 - self.epsilon_step*self.n_games
        if random.uniform(0, 1) < max(self.epsilon,0.01):
            move = self.get_random_action()
        else:
            state0 = tf.expand_dims(state, axis=0)
            prediction = self.model(state0)[0]
            # print(prediction)
            move = np.argmax(prediction)
            print(move)
        return move
        
    def get_first_state(self, game, predict=False):
        # Get the first 4 frames and turn them into a state
        first_nframe_states = []
        while len(first_nframe_states) < self.nframes:
            state = self.get_state(game)
            action = self.get_random_action()
            reward, done, score = game.step(action=action)
            state_new = self.get_state(game)
            
            if not done:
                first_nframe_states.append(state_new)
            else:
                print("done")
        
        stacked = np.array(first_nframe_states).transpose()
        state = stacked
        
        return state
    
    def experience_replay(self):
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
        

        
        # print("running in experience replay")
        # for state, action, reward, next_state, done in minibatch:
            
        #     target = reward
            
        #     if not done:
        #         Q_next = self.model.predict(np.expand_dims(next_state, axis=0))
        #         best_future_action = np.argmax(Q_next)
        #         pred = self.target_model.predict(np.expand_dims(next_state, axis=0))[0][best_future_action]
        #         target = reward + self.gamma * pred
            
        #     # get a 4 long array with the target value for each action, set the target value for the action that was taken to the target value
        #     target_vector = self.model.predict(np.expand_dims(state, axis=0))[0]
        #     target_vector[action] = target
            
            
        #     targets.append(target_vector)
            
        #     states.append(state)
            
        self.model.fit(states, targets, epochs=1, verbose=1)
        # print("done fitting")
        
        
        
    


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    # game = SnakeGameAI()
    
    state = agent.get_first_state(game)
    # fill the memory with initial data with random actions
    while len(agent.memory) < INIT_MEMORY:
        action = agent.get_random_action()
        reward, done, score = game.step(action=action)
        
        # The new state is only one frame, we want to append this to the state with 4 frames, and remove the first frame.
        state_new_1frame: np.ndarray = agent.get_state(game)
        # state_new = tf.stack((state[:,:,1], state[:,:,2], state[:,:,3],state_new_1frame), axis=-1)
        
        # add the new frame to the end of the frame states
        state_new = np.append(state, np.expand_dims(state_new_1frame, axis=-1), axis=-1)
        
        # remove the first element from the state so we still have nframes
        state_new = np.delete(state_new, 0, axis=-1)
        
        # This line tests if last element of the state equals the second last element of the new state. Should return True
        # print(state_new[:,:,-2] == state[:,:,1])
        
        
        agent.remember(state, action, reward, state_new, done)
        # print(f"Memory: {len(agent.memory)}")
        # print(f"Memory: {agent.memory}")
        state = state_new
        
        if done:
            game.reset()
            state = agent.get_first_state(game)
            
    # Main game loop to play the game after we have created the initial 4 frames, and filled the memory to the INIT_MEMORY
    while True:
        # Just for clarification, we get the state from the prefilled memory
        state = state

        # Get move
        action = agent.get_action(state)

        # perform move and get new state
        reward, done, score = game.step(action=action)
        # print(f"Reward: {reward}\t Done: {done}\t Score: {score}")
        # print(final_move)
        state_new_1frame = agent.get_state(game)
        
        # add the new frame to the end of the frame states
        state_new = np.append(state, np.expand_dims(state_new_1frame, axis=-1), axis=-1)
        
        # remove the first element from the state so we still have nframes
        state_new = np.delete(state_new, 0, axis=-1)


        # remember
        agent.remember(state, action, reward, state_new, done)
        
        # experience replay
        agent.experience_replay()
        
        state = state_new
        
        if agent.n_games > 1000:
            abcd = 10
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("tf_model")

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