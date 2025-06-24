import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    This network has a shared body and two separate 'heads':
    1. The Actor head: Outputs a probability distribution over actions (policy).
    2. The Critic head: Outputs a single value estimating how good the current state is.
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        # Shared layers that process the initial state
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor head: determines which action to take
        self.actor = nn.Linear(hidden_size, action_size)
        # Critic head: evaluates the state
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        # Pass the state through the shared layers
        x = self.shared(state)
        # Get action probabilities from the actor head
        action_probs = F.softmax(self.actor(x), dim=-1)
        # Get the state value from the critic head
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    """Proximal Policy Optimization Agent."""
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        # --- Hyperparameters ---
        self.gamma = gamma                # Discount factor for future rewards
        self.eps_clip = eps_clip          # Clipping parameter for the policy update
        self.K_epochs = K_epochs          # Number of times to train on the same batch of data

        # The policy network (actor-critic)
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # A temporary buffer to store experiences for a few episodes before updating
        self.memory = []

    def get_action(self, state):
        """
        Selects an action by sampling from the policy distribution.
        This allows for natural exploration.
        """
        with torch.no_grad(): # We don't need gradients for action selection
            state_tensor = torch.tensor(np.array(state), dtype=torch.float)
            action_probs, state_value = self.policy(state_tensor)
            
            # Create a categorical distribution to sample from
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        # Return the chosen action index, its log probability, and the critic's state value
        return action.item(), action_log_prob.item(), state_value.item()

    def remember(self, state, action, log_prob, reward, done, value):
        """Store a single transition from the current trajectory."""
        self.memory.append((state, action, log_prob, reward, done, value))

    def update(self):
        """
        Updates the policy for K epochs using the collected batch of experiences.
        This is the core of the PPO learning step.
        """
        # 1. Calculate Rewards-to-Go (what was the actual cumulative reward from each step?)
        rewards = []
        discounted_reward = 0
        # Iterate backwards through the collected trajectory
        for _, _, _, reward, done, _ in reversed(self.memory):
            if done:
                discounted_reward = 0 # Reset reward if the episode ended
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards for more stable training
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. Convert lists to tensors for batch processing
        old_states = torch.tensor(np.array([s[0] for s in self.memory]), dtype=torch.float)
        old_actions = torch.tensor(np.array([s[1] for s in self.memory]), dtype=torch.long)
        old_log_probs = torch.tensor([s[2] for s in self.memory], dtype=torch.float)

        # 3. Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Re-evaluate the old states with the current policy
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            state_values = torch.squeeze(state_values)

            # Calculate the advantage (how much better was the action than the critic's estimate?)
            advantages = rewards - state_values.detach()

            # Calculate the policy ratio (how likely is the new policy to take the old action?)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            
            # The PPO clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss is a combination of policy loss, value loss, and entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. Clear memory after update, as PPO is on-policy
        self.memory = []

    def save(self, filename="ppo_model.pth", best_score=0):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'best_score': best_score
        }, filename)

    def load(self, filename="ppo_model.pth"):
        """Load the model state."""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval() # Set model to evaluation mode
        return checkpoint.get('best_score', 0)