import numpy as np
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
from gym import make
import pybullet_envs

env = make("HalfCheetahBulletEnv-v0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transform(x):
    return torch.tensor(x)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(26, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_linear = nn.Linear(256, 6)
        self.sigma_linear = nn.Linear(256, 6)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_linear(x))
        sigma = F.softplus(self.sigma_linear(x)) + 1e-5
        return mu, sigma


class Agent:
    def __init__(self):
        self.model = Actor()
        state_dict = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.load_state_dict(state_dict)
        self.model.to(device)

    def act(self, state):
        state = transform(state).to(device)
        mu, sigma = self.model(state)
        multivatiate_normal = MultivariateNormal(mu, scale_tril=torch.diag(sigma))
        action = multivatiate_normal.sample()
        action.clamp_(env.action_space.low[0], env.action_space.high[0])
        return action.cpu().numpy()

    def reset(self):
        pass
