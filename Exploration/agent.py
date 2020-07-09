import random
import numpy as np
import os
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Exploration:
    def __init__(self, state_dim, action_dim, n_hidden_target, n_hidden_prediction, n_output, optimizer, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.target = NN(state_dim, n_hidden_target, n_output)
        self.prediction = NN(state_dim, n_hidden_prediction, n_output)
        self.optimizer = optimizer(self.prediction.parameters(), lr=lr)

    def get_exploration_reward(self, state, action, next_state):
        next_state = torch.tensor(next_state.data).reshape(-1, self.state_dim)
        y = self.target(next_state).detach()
        y_hat = self.prediction(next_state)
        reward = torch.sum((y_hat - y) ** 2, dim=1)
        return reward

    def update(self, transition):
        state, action, next_state, reward, done = transition
        L = self.get_exploration_reward(state, action, next_state)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim # dimensionalite of state space
        self.action_dim = action_dim # count of available actions
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.batch_size = 150
        self.beta = 50
        self.eps = 0.05
        self.steps = 0

        self.exploration = Exploration(**{'state_dim': state_dim,
                  'action_dim': action_dim,
                  'n_hidden_target': 256,
                  'n_hidden_prediction': 5,
                  'n_output': 8,
                  'optimizer': optim.Adam,
                  'lr': 0.001})

        self.Q_target = NN(state_dim, 300, action_dim)
        self.Q_policy = copy.deepcopy(self.Q_target)
        self.optimizer = optim.Adam(self.Q_policy.parameters(), lr=0.001)
        
    def act(self, state):
        if random.random() < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state.data)
            with torch.no_grad():
                action = np.array(torch.argmax(self.Q_policy(state)))
        return action

    def update(self, transition):
        self.steps += 1
        state, action, next_state, reward, done = transition
        action = torch.tensor(action)
        transition = state, action, next_state, reward, done
        self.memory.append(transition)
        self.exploration.update(transition)
        if len(self.memory) < self.batch_size:
            return
        state, action, next_state, reward, done = zip(*random.sample(self.memory, self.batch_size))
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.int8).reshape(-1, 1)
        expl_reward = self.exploration.get_exploration_reward(state, action, next_state)
        expl_reward = (expl_reward - expl_reward.mean()) / expl_reward.std()
        reward = (torch.tensor(reward, dtype=torch.float) + self.beta * expl_reward).unsqueeze(1)

        Q_policy = self.Q_policy(state).gather(1, action)
        Q_target = reward + (1 - done) * self.gamma * self.Q_target(next_state).max(1)[0].detach().unsqueeze(1)

        L = F.smooth_l1_loss(Q_policy, Q_target)
        self.optimizer.zero_grad()
        L.backward()
        for param in self.Q_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps < 10000 and self.steps % 20 == 0:
            self.update_target_network()
        elif self.steps % 100 == 0:
            self.update_target_network()

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_policy.state_dict())

    def reset(self):
        pass
