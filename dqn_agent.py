import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.001, gamma=0.9, memory_size=100_000, batch_size=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.model = LinearQNet(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0  # exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if random.random() < self.epsilon:
            move = np.zeros(self.action_size)
            move[random.randint(0, self.action_size - 1)] = 1
            return move
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0.unsqueeze(0))
        move = np.zeros(self.action_size)
        move[torch.argmax(prediction).item()] = 1
        return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            mini_sample = random.sample(self.memory, len(self.memory))
        else:
            mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step([state], [action], [reward], [next_state], [done])

    def _train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone().detach()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx])).item()
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def save(self, filename="model.pth", best_score=0):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'best_score': best_score
        }, filename)

    def load(self, filename="model.pth"):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.epsilon = checkpoint.get('epsilon', 1.0)
        return checkpoint.get('best_score', 0) 