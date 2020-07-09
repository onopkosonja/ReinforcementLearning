import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import make
from collections import deque, namedtuple
from torch.distributions.normal import Normal


def transform(x):
    return torch.Tensor([x])


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.transition = namedtuple('Transition', ('state', 'reward', 'next_state', 'done', 'log_prob'))

    def add(self, *args):
        new_transition = self.transition(*args)
        self.buffer.append(new_transition)

    def sample(self):
        batch = self.transition(*zip(*self.buffer))
        self.__reset()
        return batch

    def __len__(self):
        return len(self.buffer)

    def __reset(self):
        self.buffer = []


class A2C:
    def __init__(self, gamma, batch_size, optimizer, lr_value, lr_policy, network_params_policy, network_params_value):
        self.gamma = gamma
        self.batch_size = batch_size

        self.value = self.build_network(**network_params_value)
        self.policy = self.build_network(**network_params_policy)
        self.buffer = ReplayBuffer(max_size=batch_size)
        self.batch_size = batch_size

        self.value_optimizer = optimizer(self.value.parameters(), lr_value)
        self.policy_optimizer = optimizer(self.policy.parameters(), lr_policy)

    def build_network(self, state_dim, n_output):
        return nn.Sequential(nn.Linear(state_dim, 64),
                      nn.ReLU(),
                      nn.Linear(64, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, n_output))


    def act(self, state):
        mu, sigma = self.policy(state)[0]
        mu = 2 * torch.tanh(mu)
        sigma = F.softplus(sigma) + 1e-05
        normal_distribution = Normal(mu, sigma)
        action = transform(normal_distribution.sample())
        action.clamp_(env.action_space.low[0], env.action_space.high[0])
        log_prob = normal_distribution.log_prob(action)
        return action, log_prob

    def update(self, Q, Q_target, error, log_probs):
        L_value = F.mse_loss(Q, Q_target)
        self.value_optimizer.zero_grad()
        L_value.backward()
        for param in self.value.parameters():
            param.grad.data.clamp_(-10, 10)
        self.value_optimizer.step()

        L_policy = -(log_probs * error.detach()).mean()
        self.policy_optimizer.zero_grad()
        L_policy.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-10, 10)
        self.policy_optimizer.step()

    def save(self):
        torch.save(self.policy, "agent.pkl")


if __name__ == "__main__":
    env = make("Pendulum-v0")

    network_params_policy = {'state_dim': env.observation_space.shape[0],
                             'n_output': 2}

    network_params_value = {'state_dim': env.observation_space.shape[0],
                             'n_output': 1}

    parameters = {'optimizer': optim.Adam,
                  'lr_value': 0.001,
                  'lr_policy': 0.0001,
                  'batch_size': 50,
                  'gamma': 0.93,
                  'network_params_value': network_params_value,
                  'network_params_policy': network_params_policy}

    algo = A2C(**parameters)
    episodes = 10000
    rewards = deque(maxlen=50)
    best = -2000

    for i in range(episodes):
        state = transform(env.reset())
        total_reward = 0
        done = False

        while not done:
            action, log_prob = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform(next_state)
            total_reward += reward
            modified_reward = transform((reward + 8) / 10)
            algo.buffer.add(state, modified_reward, next_state, transform(done), log_prob)

            if len(algo.buffer) == algo.batch_size:
                batch = algo.buffer.sample()
                state_batch = torch.cat(batch.state)
                reward_batch = torch.cat(batch.reward).unsqueeze(1)
                next_state_batch = torch.cat(batch.next_state)
                done_batch = torch.cat(batch.done).unsqueeze(1)
                log_prob_batch = torch.cat(batch.log_prob).unsqueeze(1)

                Q_cur = algo.value(state_batch)
                with torch.no_grad():
                    Q_next = algo.value(next_state_batch)
                target = reward_batch + (1 - done_batch) * algo.gamma * Q_next
                error = target - Q_cur

                algo.update(Q_cur, target, error, log_prob_batch)

            state = next_state

        rewards.append(total_reward)

        if len(rewards) == 50 and np.mean(rewards) > best:
            best = np.mean(rewards)
            print(f'NEW BEST {best}')
            algo.save()

        print(f'episode # {i} reward: {total_reward}')

