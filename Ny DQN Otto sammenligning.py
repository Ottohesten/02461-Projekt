#%%
import SnakeGame20x20 as SnakeGameMatrixAI
import torch
import numpy as np
#import matplotlib.pyplot as plt
#import gym
import time

n_games = 50000
game_amount = 0
learning_rate = 0.0005
gamma_discount = 0.95
epsilon = 1
epsilon_step = 0.00005
epsilon_min = 0.001
batch_size = 32
buffer_size = 20000

q_net = torch.nn.Sequential(
    # Was 288 for 20x20
    torch.nn.Linear(484, 72),
    torch.nn.ReLU(),
    torch.nn.Linear(72, 72),
    torch.nn.ReLU(),
    #torch.nn.Linear(72, 72),
    #torch.nn.ReLU(),
    torch.nn.Linear(72, 4))

optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

actions = np.array([0., 1., 2., 3.], dtype=np.float32)

obs_buffer = np.zeros((buffer_size, 484))
obs_next_buffer = np.zeros((buffer_size, 484))
action_buffer = np.zeros(buffer_size)
reward_buffer = np.zeros(buffer_size)
done_buffer = np.zeros(buffer_size)


step_count = 0
for i in range(n_games):
    score = 0

    # Reset game
    snakegame = SnakeGameMatrixAI.SnakeGame()
    
    observation = snakegame.board.flatten()
    
    done = False
    
    if game_amount == 999:
        torch.save(q_net, "20x20x1000.pth")
    elif game_amount == 9999:
        torch.save(q_net, "20x20x10000.pth")
    elif game_amount == 24999:
        torch.save(q_net, "20x20x25000.pth")
    elif game_amount == 49999:
        torch.save(q_net, "20x20x50000.pth")
    

    game_amount += 1
        
    while not done:
        step_count += 1

        # Reduce epsilon
        epsilon = np.maximum(epsilon-epsilon_step, epsilon_min)

        # Choose random or net based on epsilon greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(q_net(torch.tensor(observation)).detach().numpy())
        
        # Take action
        observation_next, reward, done = snakegame.step(action)
        
        # Update score if fruit
        if reward == 4:
            score += 1
        
        # Store data in buffer - modulus to wrap around and forget old replays
        buf_index = step_count % buffer_size
        obs_buffer[buf_index] = observation
        obs_next_buffer[buf_index] = observation_next
        action_buffer[buf_index] = action
        reward_buffer[buf_index] = reward
        done_buffer[buf_index] = done
        
        # Update observation to be new state
        observation = observation_next
        
        # Update neural network starting at step_count 32
        if step_count > batch_size:
            # Choose a minibatch from buffer
            batch_index = np.random.choice(buffer_size, size=batch_size)
            
            # Compute q-val for batch and new state
            out = q_net(torch.tensor(obs_buffer[batch_index]).float())
            val = out[np.arange(batch_size), action_buffer[batch_index]]
            
            out_next = q_net(torch.tensor(obs_next_buffer[batch_index]).float())
            
            # Don't update target, only q(s, a). If done, don't receive more future rewards
            with torch.no_grad():
                target = torch.tensor(reward_buffer[batch_index]) + gamma_discount * torch.max(out_next, dim = 1).values * (1-done_buffer[batch_index])
            
            # Quadratic loss between Q and Bellman
            l = loss(val, target.float())
            
            # Adam gradient descent
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    
    # Append score to document for analysis
    with open('20x20.txt', 'a') as f:
        f.write("\n" + str(score))
    
    print(f'Score = {score}')
    #print(snakegame.board)
    action_count = 0
    


#%%
import pygame
import SnakeGameMatrixAI

pygame.init()

cell_size = 60
cell_number = 12
screen = pygame.display.set_mode((cell_size * cell_number, cell_size * cell_number))
clock = pygame.time.Clock()

for i in range(25):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            
    snakegame = SnakeGameMatrixAI.SnakeGame()   
    observation = snakegame.board.flatten()
    done = False
    score = 0
    rounds = 0
    while not done:
        clock.tick(5)
        screen.fill(pygame.color.Color("BLACK"))
        action = np.argmax(q_net(torch.tensor(observation)).detach().numpy())
        observation, reward, done = snakegame.step(action)
        print(snakegame.board)
        #print(snakegame.snake_pos)
        score += reward
        rounds += 1
        #print(action)
        #env.render()
        for e, row in enumerate(snakegame.board):
            for i, item in enumerate(row):
                if item == 0:
                    rect = pygame.Rect(i * cell_size, e * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, pygame.color.Color("BLACK"), rect)
                    
                elif item == 1 or item == 4:
                    rect = pygame.Rect(i * cell_size, e * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, pygame.color.Color("GOLD"), rect)
                    
                elif item == 3:
                    rect = pygame.Rect(i * cell_size, e * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, pygame.color.Color("WHITE"), rect)
                    
                elif item == 2:
                    rect = pygame.Rect(i * cell_size, e * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, pygame.color.Color("RED"), rect)
                    
        
        pygame.display.update()
        
pygame.quit()
print(f"Score = {score}")