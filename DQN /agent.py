import random
import os
import torch
from .train import transform_state


random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + '/agent.pkl')
        self.model.to(device)

    def act(self, state):
        state = transform_state(state).to(device)
        action = self.model(state).max(1)[1]
        return action.item()

    def reset(self):
        pass

