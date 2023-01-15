import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from HelperClasses import drawnow

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                ns = next_state[idx]
                Q_next = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(Q_next)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        # loss = self.criterion(pred, target)
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


# Convolutional section

Conv_QNet = lambda channels_in, channels_out, kernel_size, : nn.Sequential(
    torch.nn.Conv2d(channels_in, channels_out, kernel_size),
    torch.nn.ReLU(),
    # torch.nn.AvgPool2d((2, 2)),
    torch.nn.Conv2d(channels_out, channels_out, (5,5)),
    torch.nn.ReLU(),
    # torch.nn.AvgPool2d((2, 2)),
    torch.nn.Flatten(),
    torch.nn.Linear(channels_out*2*2,3), # for 10x10 grid
    # torch.nn.Linear(channels_out*3*3,3), # for 20x20 grid
)


class ConvQtrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_list = []
        # self.criterion = nn.L1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            # print("unsqueezed")

        # 1: predicted Q values with current state
        # print(state.shape)
        # print(next_state.shape)
        pred = self.model(state)
        

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_next = self.model(next_state[idx])
                Q_next = self.model(torch.unsqueeze(next_state[idx],0))
                Q_new = reward[idx] + self.gamma * torch.max(Q_next)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # loss = self.criterion(pred, target)
        loss = self.criterion(target, pred)
        self.loss_list.append(loss.item())

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()