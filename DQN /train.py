from gym import make
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple


def transform_state(x):
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
        return batch

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, optimizer, buffer_size, gamma, lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()
        self.optimizer = optimizer(self.policy_network.parameters(), lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma

    def build_network(self):
        return nn.Sequential(nn.Linear(self.state_dim, 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Linear(256, self.action_dim))

    def update(self, batch):
        not_final_index = [i for i, d in enumerate(batch.done) if d is False]
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)[not_final_index]

        Q_policy = self.policy_network(state_batch).gather(1, action_batch)
        Q_target_expected = reward_batch
        Q_target_expected[not_final_index] += self.gamma * self.target_network(next_state_batch).max(1)[0]

        L = F.mse_loss(Q_policy, Q_target_expected.unsqueeze(1))
        self.optimizer.zero_grad()
        L.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state):
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    def save(self):
        torch.save(self.policy_network, "agent.pkl")


if __name__ == "__main__":
    env = make("LunarLander-v2")
    env.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    action_dim = 4
    state_dim = 8
    buffer_size = 100000
    gamma = 0.99
    optimizer = optim.Adam
    learning_rate = 0.0001
    dqn = DQN(state_dim, action_dim, optimizer, buffer_size, gamma, learning_rate)
    episodes = 1000
    rewards_50 = deque(maxlen=50)
    best = -200
    batch_size = 64
    eps = 1
    eps_decay = 0.995
    eps_min = 0.05

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            if random.random() < eps:
                action = torch.tensor([[random.choice(range(action_dim))]])
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = transform_state(next_state)
            reward = transform_state(reward)
            total_reward += reward
            dqn.buffer.add(state, action, reward, next_state, done)

            if len(dqn.buffer) >= batch_size:
                sample = dqn.buffer.sample(batch_size)
                dqn.update(sample)

            state = next_state
            steps += 1

        dqn.update_target_network()

        rewards_50.append(total_reward)
        eps = max(eps_min, eps * eps_decay)

        if len(rewards_50) >= 50:
            average_50 = sum(rewards_50) / len(rewards_50)
            if average_50 > best:
                print(average_50)
                best = average_50
                dqn.save()
