import random
import numpy as np
import os
from .train import transform_state
random.seed(1)


class Agent:
    def __init__(self):
        with np.load(__file__[:-8] + "/agent.npz") as data:
            self.weight = data['arr_0']
            self.bias = data['arr_1']

    def act(self, state):
        return np.argmax(transform_state(state).T.dot(self.weight) + self.bias.T)

    def reset(self):
        pass
