import numpy as np
import os
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from .train import transform
from gym import make
env = make("Pendulum-v0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.to(device)

    def act(self, state):
        state = transform(state).to(device)
        mu, sigma = self.model(state)[0]
        mu = 2 * torch.tanh(mu)
        sigma = F.softplus(sigma) + 1e-05
        normal_distribution = Normal(mu, sigma)
        action = normal_distribution.sample()
        action.clamp_(env.action_space.low[0], env.action_space.high[0])
        return np.array([action.item()])

    def reset(self):
        pass
