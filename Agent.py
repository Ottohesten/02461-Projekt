import torch
import random
import numpy as np
from collections import deque
from HelperClasses import Direction, Point, Board
from SnakeGameClass import SnakeGame
# from Solution import SnakeGame
from model import Linear_QNet
import time

MAX_MEMORY = 50_000
BATCH_SIZE = 32
LR = 0.001

class Agent:

    def __init__(self, model=None):
        self.n_games = 0
        self.epsilon_start = 1 # randomness
        self.epsilon_step = 0.0001
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        if model is not None:
            self.model = model
            print("loaded in model")
        else:
            self.model = Linear_QNet(11, 32, 3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss = torch.nn.MSELoss()


    def process_state(self, game):
        head = game.snake.head

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        board = Board(game.w, game.h, game.snake)
        available_space = board.get_available_space_3directions(game.direction)
        distances = board.manhattan_distance_3directions(game.direction, game.food)
        snake_length = int(len(game.snake.body))

        state = [
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            snake_length,

            ] + list(available_space.values()) + list(distances.values())

        # Returns a 11 long array with False or True values
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def get_random_action(self):
        return random.randint(0,2)

    def get_predicted_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state0)
        # print(prediction)
        move = torch.argmax(prediction).item()
        return move

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.epsilon_start - self.epsilon_step*self.n_games
        if random.uniform(0, 1) < max(self.epsilon,0.01):
            move = self.get_random_action()
        else:
            move = self.get_predicted_action(state)
        return move

    
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
    record = 0
    agent = Agent()
    env = SnakeGame()
    model_filepath = "saved_models/"
    data_filepath = "data/pre_score_data_10x10.csv"
    
    while True:
        # Get initial state
        env.reset()
        done = False
        agent.n_games += 1

        # Process the state into the values we cant to pass to the network
        processed_state = agent.process_state(env)
        while not done:

            # Get action
            action = agent.get_action(processed_state)

            # Get the new state
            next_state, reward, done, score = env.step(action=action)

            processed_next_state = agent.process_state(env)

            # Remember
            agent.remember(processed_state, action, reward, processed_next_state, done)

            # This is where we train the network
            agent.experience_replay()

            processed_state = processed_next_state

            if score > record:
                record = score


            if done:
                with open(data_filepath, "a") as data_file:
                        data_file.write("\n")
                        data_file.write(str(score))
                
                print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}\t epsilon: {agent.epsilon}")
                if agent.n_games == 1000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_1000.pth")
                if agent.n_games == 5000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_5000.pth")
                if agent.n_games == 10000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_10000.pth")
                if agent.n_games == 12500:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_12500.pth")
                if agent.n_games == 15000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_15000.pth")
                if agent.n_games == 25000:
                    torch.save(agent.model.state_dict(), f"{model_filepath}pre_10x10_25000.pth")



def test():
    record = 0
    grid_size = 20
    
    model_type = "pre"
    model = Linear_QNet(11, 32, 3)
    trained_for = 10000
    
    
    model.load_state_dict(torch.load(f"saved_models/{str(model_type)}_{str(grid_size)}x{str(grid_size)}_{str(trained_for)}.pth"))
    agent = Agent(model)
    env = SnakeGame(grid_size,grid_size)
    data_filepath = f"data/tests/{str(model_type)}_{str(grid_size)}x{str(grid_size)}_{str(trained_for)}_data.csv"
    
    with open(data_filepath, "a+") as data_file:
                    data_file.write(f"{str(model_type)}_{str(grid_size)}x{str(grid_size)}_{str(trained_for)}")


    while True:
        # Get initial state
        env.reset()
        done = False
        agent.n_games += 1

        # Process the state into the values we cant to pass to the network
        processed_state = agent.process_state(env)
        while not done:

            # Get action
            action = agent.get_predicted_action(processed_state)

            # Get the new state
            next_state, reward, done, score = env.step(action=action)

            processed_next_state = agent.process_state(env)

            processed_state = processed_next_state

            if score > record:
                record = score


            if done:
                with open(data_filepath, "a") as data_file:
                        data_file.write("\n")
                        data_file.write(str(score))
                        pass
                if agent.n_games == 1000:
                    quit()
                print(f"Game: {agent.n_games}\t Score: {score}\t Record: {record}")
    




if __name__ == '__main__':
    # train()
    test()
    pass