import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time

if torch.cuda.is_available():
    device = torch.device("cuda")          # Use GPU

else:
    device = torch.device("cpu")           # Use CPU
    print('PyTorch is using CPU.')



class DQN(nn.Module):
    def __init__(self, dinput, doutput):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dinput, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, doutput)
        )

    def forward(self, x):
        if type(x) is list:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            output = self.fc(x_tensor)
        else:
            output = self.fc(x)
        return output

discount_rate = 0.99

def calc_loss(M, DQN_target, DQN_training, criterion):
    batch_size = 256

    indices = np.random.randint(0, len(M), size=batch_size)
    batch = [M[i] for i in indices]

    states, actions, rewards, delay, next_states  = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    delay = torch.tensor(delay, dtype=torch.float32).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(device)
    non_final_next_states = torch.tensor([s for s in next_states if s is not None], dtype=torch.float32).to(device)
    state_action_values = DQN_training(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    next_state_values = torch.zeros(batch_size, device=device)
    # Use DQN_training to select the action for DQN_target
    best_actions = DQN_training(non_final_next_states).argmax(1).unsqueeze(1)
    next_state_values[non_final_mask] = DQN_target(non_final_next_states).gather(1, best_actions).squeeze(1).detach()
    expected_state_action_values = rewards + (discount_rate**delay * next_state_values)

    loss = criterion(state_action_values, expected_state_action_values)

    return loss

def train(M,DQN_target,DQN_training,optimizer,criterion):
    # zero the gradient
    optimizer.zero_grad()

    # sample one batch and then calculate the loss
    loss = calc_loss(M,DQN_target,DQN_training,criterion)

    # apply gradient clipping
    # max_gradient_norm = 0.5  # 设置梯度裁剪的阈值
    # torch.nn.utils.clip_grad_norm_(DQN_training.parameters(), max_gradient_norm)

    # back backpropagation and gradient descent
    loss.backward()
    optimizer.step()

