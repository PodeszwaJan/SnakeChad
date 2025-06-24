import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque, OrderedDict

# The network architecture is the same
from dqn_agent import LinearQNet

class ADQNAgent:
    """Averaged-Deep Q-Network Agent."""
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.001, gamma=0.9, 
                 memory_size=100_000, batch_size=1000, update_freq=100, k_avg=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.k_avg = k_avg

        # Main network for action selection and learning
        self.model = LinearQNet(state_size, hidden_size, action_size)
        
        # Target network for stable Q-value estimation
        self.target_model = LinearQNet(state_size, hidden_size, action_size)
        
        # History of the last K model weights for averaging
        self.model_history = deque(maxlen=self.k_avg)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initial synchronization
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_history.append(self.model.state_dict())

    def get_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            move = np.zeros(self.action_size)
            move[random.randint(0, self.action_size - 1)] = 1
            return move
        
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state_tensor)
        move = np.zeros(self.action_size)
        move[torch.argmax(prediction).item()] = 1
        return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self._train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step([state], [action], [reward], [next_state], [done])

    def _update_target_network(self):
        """
        The core of Averaged-DQN. This function averages the weights from
        the model history and loads them into the target network.
        """
        if not self.model_history:
            return

        # Start with a dictionary of tensors initialized to zero
        avg_state_dict = OrderedDict()
        for key in self.model_history[0]:
            avg_state_dict[key] = torch.zeros_like(self.model_history[0][key])

        # Sum up all the weights from the history
        for state_dict in self.model_history:
            for key in state_dict:
                avg_state_dict[key] += state_dict[key]
        
        # Divide by the number of models in history to get the average
        num_models = len(self.model_history)
        for key in avg_state_dict:
            avg_state_dict[key] /= num_models

        # Load the averaged weights into the target network
        self.target_model.load_state_dict(avg_state_dict)

    def _train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone().detach()

        # Get Q-values for next states from the (averaged) target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * next_q_values[idx].item()
            
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        
        # Periodically update the history and the target network
        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            self.model_history.append(self.model.state_dict())
            self._update_target_network()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def save(self, filename="adqn_model.pth", best_score=0):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_history': self.model_history, # Save the history
            'epsilon': self.epsilon,
            'best_score': best_score,
            'update_counter': self.update_counter
        }, filename)

    def load(self, filename="adqn_model.pth"):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.model_history = checkpoint.get('model_history', deque([self.model.state_dict()], maxlen=self.k_avg))
        self.epsilon = checkpoint.get('epsilon', 1.0)
        self.update_counter = checkpoint.get('update_counter', 0)
        
        # Crucially, rebuild the target network from the loaded history
        self._update_target_network()
        self.target_model.eval()

        return checkpoint.get('best_score', 0)