# TEST OF DQN: 1000 SAMPLES
import SnakeGame20x20 as SnakeGameMatrixAI
import numpy as np
import torch
q_net = torch.load('20x20x1000.pth')
print("Loaded!")
for i in range(1000):
    snakegame = SnakeGameMatrixAI.SnakeGame()   
    observation = snakegame.board.flatten()
    done = False
    score = 0
    step_count = 0
    while not done:
        # Use q_net to choose action
        action = np.argmax(q_net(torch.tensor(observation)).detach().numpy())
        observation, reward, done = snakegame.step(action)
        
        # Stop if agent takes too long
        if step_count > 100 * snakegame.snake_pos.size:
            done = True
        
        # If apple, score += 1
        if reward == 4:
            score += 1
            step_count = 0
            
        # Save to file
        step_count += 1
    with open('20x20x1000xmodelscores.txt', 'a') as f:
        f.write(str(score) + "\n")
        
    print(f"Score = {score}")