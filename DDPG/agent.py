import random
import numpy as np
import os
import torch
from gym import make
import pybullet_envs
from .train import transform, PolicyNetwork

env = make("AntBulletEnv-v0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        n_features = 256
        self.model = PolicyNetwork(state_dim, n_features, action_dim)
        state_dict = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.load_state_dict(state_dict)
        self.model.to(device)

    def act(self, state):
        state = transform(state).to(device)
        action = self.model(state)
        return action.cpu().detach().numpy()[0]

    def reset(self):
        pass
