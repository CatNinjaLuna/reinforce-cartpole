import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

# Compute discounted returns
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Main training loop
def train_reinforce(env_name="CartPole-v1", max_episodes=1000, render=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    episode_rewards = []
    reward_window = deque(maxlen=100)

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns = compute_returns(rewards)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        reward_window.append(total_reward)

        avg_reward = np.mean(reward_window)

        print(f"Episode {episode}: Total Reward = {total_reward:.2f} | Avg (last 100) = {avg_reward:.2f}")

        # Early stopping if solved
        if avg_reward >= 195.0 and episode >= 100:
            print(f"Solved after {episode} episodes!")
            break

    env.close()

    # Plot reward curve and save to file
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE on CartPole-v1')
    plt.grid()
    
    # Save the figure to the root directory before showing it
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("Training curve saved as 'training_curve.png'")
    
    # Optionally still show the plot
    plt.show()

if __name__ == "__main__":
    train_reinforce()