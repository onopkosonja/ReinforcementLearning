from gym import make
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import pybullet_envs
from gym import make

def transform(x):
    return torch.Tensor([x])


def transform_state(x):
    return torch.tensor(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.transition = namedtuple('Transition', ('state', 'next_state', 'reward', 'action', 'done'))

    def add(self, *args):
        new_transition = self.transition(*args)
        self.buffer.append(new_transition)

    def sample(self, batch_size):
        ind = np.random.choice(np.arange(self.max_size), batch_size, replace=False)
        buff = self.transition(*zip(*self.buffer))
        states = torch.cat(buff.state)[ind]
        next_states = torch.cat(buff.next_state)[ind]
        rewards = torch.cat(buff.reward).unsqueeze(1)[ind]
        actions = torch.cat(buff.action).unsqueeze(1)[ind]
        done = torch.cat(buff.done).unsqueeze(1)[ind]
        return states, actions, next_states, rewards, done

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_linear = nn.Linear(256, action_dim)
        self.sigma_linear = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu_linear(x))
        sigma = F.softplus(self.sigma_linear(x))
        return mu, sigma.squeeze() + 1e-5


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(self, state_dim, action_dim, optimizer, gamma, clip, trajectory_size, epochs, batch_size, lr_actor,
                 lr_critic):
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.trajectory_size = trajectory_size
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=trajectory_size)

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.actor_old = Actor(state_dim=state_dim, action_dim=action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim=state_dim)

        self.actor_optimizer = optimizer(self.actor.parameters(),  lr_actor)
        self.critic_optimizer = optimizer(self.critic.parameters(),  lr_critic)

    def act(self, state):
        with torch.no_grad():
            mu, sigma = self.actor(state)
        multivatiate_normal = MultivariateNormal(mu, scale_tril=torch.diag(sigma))
        action = multivatiate_normal.sample()
        action.clamp_(env.action_space.low[0], env.action_space.high[0])
        return action.numpy()[0]

    def get_value(self, state, action):
        mu, sigma = self.actor(state)
        multivatiate_normal_new = MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))
        log_prob = multivatiate_normal_new.log_prob(action)

        with torch.no_grad():
            mu_old, sigma_old = self.actor_old(state)
        multivatiate_normal_old = MultivariateNormal(mu_old, scale_tril=torch.diag_embed(sigma_old))
        old_log_prob = multivatiate_normal_old.log_prob(action)

        return log_prob.unsqueeze(1), old_log_prob.unsqueeze(1),

    def update(self):

        for _ in range(self.epochs):
            states, actions, next_states, rewards, done = self.buffer.sample(self.batch_size)

            V_cur = algo.critic(states)
            with torch.no_grad():
                V_next = algo.critic(next_states)
            V_target = rewards + (1 - done) * self.gamma * V_next
            advantage = (V_target - V_cur).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std())

            new_log_prob, old_log_prob = self.get_value(states, actions)
            ratio = torch.exp(new_log_prob - old_log_prob)
            s1 = ratio * advantage
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
            L_actor = (-torch.min(s1, s2)).mean()

            self.actor_optimizer.zero_grad()
            L_actor.backward()
            for param in self.actor.parameters():
                param.grad.data.clamp_(-10, 10)
            self.actor_optimizer.step()

            L_critic = F.mse_loss(V_target, V_cur)
            self.critic_optimizer.zero_grad()
            L_critic.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-10, 10)
            self.critic_optimizer.step()
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.reset()

    def save(self):
        torch.save(self.actor.state_dict(), "agent1.pkl")


if __name__ == "__main__":
    env = make("HalfCheetahBulletEnv-v0")
    parameters = {'state_dim': 26,
                  'action_dim': 6,
                  'optimizer': optim.Adam,
                  'lr_actor': 1e-5,
                  'lr_critic': 1e-4,
                  'gamma': 0.99,
                  'clip': 0.1,
                  'trajectory_size': 2000,
                  'epochs': 128,
                  'batch_size': 64}

    algo = PPO(**parameters)
    episodes = 10000
    best = -1500
    rewards = deque(maxlen=50)

    for i in range(episodes):
        state = transform(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform(next_state)
            total_reward += reward
            steps += 1
            algo.buffer.add(state, next_state, transform(reward), transform(action), transform(done))

            if len(algo.buffer) == algo.trajectory_size:
                algo.update()

            state = next_state

        rewards.append(total_reward)

        if len(rewards) == 50 and np.mean(rewards) > best:
            best = np.mean(rewards)
            print(f'NEW BEST {best}')
            algo.save()

        print(f'episode # {i} reward: {total_reward}')




