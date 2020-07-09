import pybullet_envs
import torch
import random
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import make
from collections import deque, namedtuple


def transform(x):
    return torch.Tensor([x])


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, *args):
        new_transition = self.transition(*args)
        self.buffer.append(new_transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = self.transition(*zip(*batch))
        states = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward).unsqueeze(1)
        actions = torch.cat(batch.action)
        done = torch.cat(batch.done).unsqueeze(1)
        return states, actions, next_states, rewards, done

    def __len__(self):
        return len(self.buffer)


class Noise:
    def __init__(self, action_dim, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.state = np.ones(action_dim) * self.mu

    def get_noise(self):
        state = self.state
        d = self.theta * (self.mu - state) + self.sigma * np.random.randn(self.action_dim)
        self.state = state + d
        return transform(self.state)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, n_features, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, n_features)
        self.fc2 = nn.Linear(n_features + action_dim, n_features)
        self.fc3 = nn.Linear(n_features, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_features, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.fc3 = nn.Linear(n_features, action_dim)

    def forward(self, action):
        x = F.relu(self.fc1(action))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class DDPG:
    def __init__(self, gamma, action_space, buffer_size, batch_size, target_update_rate, eps, eps_decay,
                 optimizer, lr_value, lr_policy, network_params, noise_params):
        self.action_space = action_space
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]

        self.value = ValueNetwork(**network_params)
        self.policy = PolicyNetwork(**network_params)
        self.value_target = copy.deepcopy(self.value)
        self.policy_target = copy.deepcopy(self.policy)

        self.noise = Noise(**noise_params)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.value_optimizer = optimizer(self.value.parameters(), lr_value)
        self.policy_optimizer = optimizer(self.policy.parameters(), lr_policy)

    def act(self, state, eps=0):
        with torch.no_grad():
            action = self.policy(state)
        action = (action + eps * self.noise.get_noise()).float().numpy()[0]
        return np.clip(action, self.min_action, self.max_action)

    def update_target(self, nn, nn_target):
        for param, param_target in zip(nn.parameters(), nn_target.parameters()):
            param_target.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * param_target.data)

    def update(self, Q, Q_target, state):
        L_value = F.smooth_l1_loss(Q, Q_target)
        self.value_optimizer.zero_grad()
        L_value.backward()
        for param in self.value.parameters():
            param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()

        action = self.policy(state)
        L_policy = -self.value(state, action).mean()
        self.policy_optimizer.zero_grad()
        L_policy.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        self.update_target(self.value, self.value_target)
        self.update_target(self.policy, self.policy_target)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


def exploit(agent, env, episodes=10):
    total_reward = 0
    for _ in range(episodes):
        state = transform(env.reset())
        done = False
        while not done:
            action = agent.act(state, 0)
            next_state, reward, done, _ = env.step(action)
            next_state = transform(next_state)
            total_reward += reward
            state = next_state
    return total_reward / episodes


if __name__ == "__main__":
    env = make("AntBulletEnv-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    networks_params = {'state_dim': state_dim,
                       'action_dim': action_dim,
                       'n_features': 256}

    noise_params = {'mu': 0,
                    'theta': 0.15,
                    'sigma': 0.3,
                    'action_dim': action_dim}

    parameters = {'optimizer': optim.Adam,
                  'action_space': env.action_space,
                  'lr_value': 1e-3,
                  'lr_policy': 1e-4,
                  'target_update_rate': 0.05,
                  'buffer_size': 50000,
                  'batch_size': 128,
                  'gamma': 0.99,
                  'eps': 1,
                  'eps_decay': 0.97,
                  'noise_params': noise_params,
                  'network_params': networks_params}

    algo = DDPG(**parameters)
    best_reward = -1e6
    best_10 = -1e6
    best_50 = -1e6
    episodes = 1000
    rewards_50 = deque(maxlen=50)
    for i in range(episodes):
        state = transform(env.reset())
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = algo.act(state, eps=algo.eps)
            next_state, reward, done, _ = env.step(action)
            next_state = transform(next_state)
            total_reward += reward
            steps += 1

            algo.buffer.add(state, transform(action), transform(reward), next_state, transform(done))

            if len(algo.buffer) > algo.batch_size:
                state_batch, action_batch, next_state_batch, reward_batch, done_batch = algo.buffer.sample(algo.batch_size)

                Q_cur = algo.value(state_batch, action_batch)
                with torch.no_grad():
                    next_action_batch = algo.policy_target(next_state_batch)
                Q_next = algo.value_target(next_state_batch, next_action_batch).detach()
                Q_target = reward_batch + (1 - done_batch) * algo.gamma * Q_next

                algo.update(Q_cur, Q_target, state_batch)

            state = next_state
        algo.eps = max(algo.eps * algo.eps_decay, 0.01)

        print(f'#{i}, reward={total_reward}')

        rewards_50.append(total_reward)

        r_50 = sum(rewards_50) / 50

        if len(rewards_50) == 50 and r_50 > best_50:
            best_50 = r_50
            print(f'NEW 50 BEST: {best_50}')
            algo.save('best_50.pkl')

        if i % 20 == 0:
            exploit_reward = exploit(algo, env)
            if best_reward < exploit_reward:
                best_reward = exploit_reward
                print(f'NEW EXPL BEST {best_reward}')
                algo.save('agent_1.pkl')
