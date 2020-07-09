from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random

N_STEP = 3
GAMMA = 0.9


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state.reshape(state.shape[0], 1))
    return np.array(result)


class AQL:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.weight, self.bias = self.init_weights(action_dim, state_dim)

    def update(self, transition, lr=0.001):
        state, action, next_state, reward, done = transition
        Q_cur = self.act(state, target=True)[0, action]
        Q_next = np.max(self.act(next_state, target=True))
        target = reward + GAMMA * Q_next
        diff = Q_cur - target
        self.weight[:, action] -= lr * diff * state.reshape((state.shape[0],))
        self.bias[action] -= lr * diff

    def act(self, state, target=False):
        Q = state.T @ self.weight + self.bias.T
        if target:
            return Q
        return np.argmax(Q)

    def save(self):
        weight = np.array(self.weight)
        bias = np.array(self.bias)
        np.savez("agent.npz", weight, bias)

    def init_weights(self, n_action, n_features):
        w = np.random.randn(n_features, n_action) * 0.01
        b = np.random.randn(n_action, 1) * 0.01
        return w, b


if __name__ == "__main__":
    env = make("MountainCar-v0")
    aql = AQL(state_dim=2, action_dim=3)
    eps = 1
    episodes = 2000
    env.seed(1)
    np.random.seed(1)
    random.seed(1)
    rewards_20 = deque(maxlen=20)
    best = -200

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            modified_reward = reward + 10 * abs(next_state[1])
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(modified_reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            # env.render()
        print(total_reward)
        rewards_20.append(total_reward)
        eps *= 0.95
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        average_20 = sum(rewards_20) / len(rewards_20)
        if average_20 > best:
            best = average_20
            aql.save()
    env.close()
