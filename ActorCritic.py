import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Categorical
from DQN_v14 import DQN

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('PyTorch is using CPU.')

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = self._init_layers(obs_dim, act_dim, hidden_size)
        self.critic = self._init_layers(obs_dim, 1, hidden_size)

    def _init_layers(self, in_dim, out_dim, hidden_size):
        layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

        # 使用标准正态分布进行随机初始化
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)  # 使用正态分布初始化权重
                init.constant_(m.bias, 0)  # 使用常数初始化偏差

        if out_dim == 1:
            # 对于critic的输出层，我们通常使用标量，所以没有softmax
            pass
        else:
            # 对于actor的输出层，我们通常使用softmax进行归一化
            layers.add_module('softmax', nn.Softmax(dim=-1))

        return layers

    def actor_forward(self, obs):
        prob = self.actor(obs)
        return prob

    def critic_forward(self, obs):
        value = self.critic(obs)
        return value

class A2CAgent:
    def __init__(self, hidden_size=256, critic_lr=0.001, actor_lr=0.0001, gamma=0.99, tau=0.001):
        self.obs_dim = 22
        self.act_dim = 57 * 5 + 1
        self.gamma = gamma
        self.training_steps = []
        self.entropy_alpha = 0.01
        self.tau = tau

        self.network = ActorCritic(self.obs_dim, self.act_dim, hidden_size).to(device)
        self.target_network = ActorCritic(self.obs_dim, self.act_dim, hidden_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=critic_lr)

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        prob = self.network.actor_forward(obs)
        dist = Categorical(prob)
        action = dist.sample()
        value = self.network.critic_forward(obs)
        return [action.item(), value]

    def update(self, trajectory, alpha):
        self.entropy_alpha = alpha
        states, actions, rewards, delays, next_states = trajectory

        obs = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        delays = torch.tensor(delays, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_states, dtype=torch.float32).to(device)

        values = self.network.critic_forward(obs)

        with torch.no_grad():
            next_values = torch.zeros_like(values)  # Initialize next_values as zero
            if len(next_obs) > 0:
                next_values = self.target_network.critic_forward(next_obs)

        deltas = rewards + self.gamma ** delays * next_values - values

        # Update Critic
        critic_loss = deltas.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, train_param in zip(self.target_network.critic.parameters(), self.network.critic.parameters()):
            target_param.data.copy_(self.tau * train_param.data + (1.0 - self.tau) * target_param.data)

        probs = self.network.actor_forward(obs)
        dists = Categorical(probs)
        values = self.network.critic_forward(obs)

        with torch.no_grad():
            next_values = torch.zeros_like(values)
            if len(next_obs) > 0:
                next_values = self.network.critic_forward(next_obs)

        deltas = rewards + self.gamma ** delays * next_values - values

        actor_loss = -(dists.log_prob(actions) * deltas.detach()).mean()
        entropy_loss = -self.entropy_alpha * dists.entropy().mean()  # 添加熵项
        loss = actor_loss + entropy_loss
        self.training_steps.append(float(critic_loss))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))

class PPOAgent(A2CAgent):
    def __init__(self, *args, clip_epsilon=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
        self.num_training_steps = 0

    def update(self, trajectory, alpha):
        self.entropy_alpha = alpha
        self.num_training_steps += 1

        states, actions, rewards, delays, next_states = trajectory

        obs = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        delays = torch.tensor(delays, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_states, dtype=torch.float32).to(device)

        # Compute values and next_values
        values = self.network.critic_forward(obs)
        with torch.no_grad():
            next_values = torch.zeros_like(values)
            if len(next_obs) > 0:
                next_values = self.target_network.critic_forward(next_obs)

        # Compute delta for critic update
        deltas = rewards + self.gamma ** delays * next_values - values

        # Update Critic as usual
        critic_loss = deltas.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, train_param in zip(self.target_network.critic.parameters(), self.network.critic.parameters()):
            target_param.data.copy_(self.tau * train_param.data + (1.0 - self.tau) * target_param.data)

        # After training critic, compute new delta
        values = self.network.critic_forward(obs)

        with torch.no_grad():
            next_values = torch.zeros_like(values)
            if len(next_obs) > 0:
                next_values = self.network.critic_forward(next_obs)

        deltas = rewards + self.gamma ** delays * next_values - values

        # Compute new probabilities
        probs = self.network.actor_forward(obs)
        dists = Categorical(probs)
        log_probs = dists.log_prob(actions)

        # Compute entropy loss
        entropy_loss = -self.entropy_alpha * dists.entropy().mean()

        # Compute old log probabilities
        with torch.no_grad():
            old_probs = self.target_network.actor_forward(obs)
            old_dists = Categorical(old_probs)
            old_log_probs = old_dists.log_prob(actions)

        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)  # remember that log(a/b) = log(a) - log(b)

        # Compute surrogate loss
        surrogate_loss = ratio * deltas
        clipped_surrogate_loss = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * deltas

        # Final loss is minimum of surrogate loss and clipped surrogate loss, minus entropy loss
        loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean() + entropy_loss

        # update old policy
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.training_steps.append(float(critic_loss.item()))


class BSACAgent(A2CAgent):
    def __init__(self, *args, clip_epsilon=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
        self.num_training_steps = 0
        self.actor_eps = []
        self.entropy_eps = []
        self.ratio = []

    def critic_update(self, M):
        batch_size = 512

        indices = np.random.randint(0, len(M), size=batch_size)
        batch = [M[i] for i in indices]

        states, actions, rewards, delay, next_states = zip(*batch)

        obs = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        delays = torch.tensor(delay, dtype=torch.float32).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(device)
        non_final_next_obs = torch.tensor([s for s in next_states if s is not None], dtype=torch.float32).to(device)

        # Compute values and next_values
        values = self.network.critic_forward(obs)
        next_obs_values = torch.zeros(batch_size, device=device)
        next_obs_values[non_final_mask] = self.target_network.critic_forward(non_final_next_obs).squeeze()

        # Compute delta for critic update
        deltas = rewards + self.gamma ** delays * next_obs_values - values

        # Update Critic as usual
        critic_loss = deltas.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Do soft update for target critic network
        for target_param, train_param in zip(self.target_network.critic.parameters(), self.network.critic.parameters()):
            target_param.data.copy_(self.tau * train_param.data + (1.0 - self.tau) * target_param.data)

        self.training_steps.append(float(critic_loss.item()))

    def actor_update(self, M, alpha):
        self.entropy_alpha = alpha
        self.num_training_steps += 1

        batch_size = len(M)

        # indices = np.random.randint(0, len(M), size=batch_size)
        batch = [M[i] for i in range(len(M))]
        states, actions, rewards, delay, next_states = zip(*batch)

        obs = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        delays = torch.tensor(delay, dtype=torch.float32).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(device)
        non_final_next_obs = torch.tensor([s for s in next_states if s is not None], dtype=torch.float32).to(device)

        # After training critic, compute new delta (Advantage Function)
        values = self.network.critic_forward(obs).squeeze(1).detach()
        next_obs_values = torch.zeros(batch_size, device=device)
        next_obs_values[non_final_mask] = self.network.critic_forward(non_final_next_obs).squeeze()
        deltas = rewards + self.gamma ** delays * next_obs_values - values

        # value normalization
        mean_delta = torch.mean(deltas)
        std_delta = torch.std(deltas)
        deltas = (deltas - mean_delta) / (std_delta + 1e-8)

        # Compute new policy probabilities
        probs = self.network.actor_forward(obs)
        dists = Categorical(probs)
        log_probs = dists.log_prob(actions)

        # Compute entropy loss
        entropy_loss = -self.entropy_alpha * dists.entropy().mean()

        # Compute old policy log probabilities
        with torch.no_grad():
            old_probs = self.target_network.actor_forward(obs)
            old_dists = Categorical(old_probs)
            old_log_probs = old_dists.log_prob(actions)

        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)  # remember that log(a/b) = log(a) - log(b)

        # Compute surrogate loss
        surrogate_loss = ratio * deltas
        clipped_surrogate_loss = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * deltas

        # Final loss is minimum of surrogate loss and clipped surrogate loss, minus entropy loss
        loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean() + entropy_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # keep old policy using target actor network
        if self.num_training_steps % 2 == 0:
            self.target_network.actor.load_state_dict(self.network.actor.state_dict())

        self.actor_eps.append(- float(loss.item()))
        self.entropy_eps.append(- float(entropy_loss.item()))
        self.ratio.append(float(ratio.mean().item()))

    def update(self, M, actorm, alpha):
        for i in range(5):
            self.critic_update(M)

        self.actor_update(actorm, alpha)






