import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os

# Policy Network for REINFORCE
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

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.policy_head(shared_out)
        state_value = self.value_head(shared_out)
        return action_probs, state_value

# Compute discounted returns (used in REINFORCE)
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Main training loop for REINFORCE (Policy Gradient)
def train_reinforce(env_name="CartPole-v1", max_episodes=1000, render=True):
    print("\n---------- Starting REINFORCE (Policy Gradient) Training ----------\n")
    
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

    # Plot reward curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE (Policy Gradient) on CartPole-v1')
    plt.grid()
    
    # Save the figure to the root directory before showing it
    plt.savefig('policy_gradient_training_curve.png', dpi=300, bbox_inches='tight')
    print("Training curve saved as 'policy_gradient_training_curve.png'")
    
    plt.show()
    
    return episode_rewards

# Main training loop for Actor-Critic
def train_actor_critic(env_name="CartPole-v1", max_episodes=1000, gamma=0.99, render=True):
    print("\n---------- Starting Actor-Critic Training ----------\n")
    
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    episode_rewards = []
    reward_window = deque(maxlen=100)
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs, value = model(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        
        # Compute advantages and returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        advantages = returns - values.detach()
        
        # Loss: policy + value
        log_probs = torch.stack(log_probs)
        actor_loss = -(log_probs * advantages).sum()
        critic_loss = nn.MSELoss()(values, returns)
        loss = actor_loss + critic_loss
        
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
    
    # Plot reward curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Actor-Critic on CartPole-v1')
    plt.grid()
    
    plt.savefig("actor_critic_training_curve.png", dpi=300, bbox_inches='tight')
    print("Training curve saved as 'actor_critic_training_curve.png'")
    
    plt.show()
    
    return episode_rewards

# Run both methods and compare
def compare_methods(render=False):
    print("\n========== Comparing REINFORCE and Actor-Critic Methods ==========\n")
    
    # Train using both methods
    pg_rewards = train_reinforce(render=render)
    ac_rewards = train_actor_critic(render=render)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(pg_rewards, label='REINFORCE (Policy Gradient)')
    plt.plot(ac_rewards, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparison: REINFORCE vs Actor-Critic on CartPole-v1')
    plt.legend()
    plt.grid()
    
    plt.savefig("rl_methods_comparison.png", dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'rl_methods_comparison.png'")
    
    plt.show()

if __name__ == "__main__":
    # Choose which method(s) to run
    method = input("Select method (1: REINFORCE, 2: Actor-Critic, 3: Both & Compare): ")
    render = input("Enable rendering? (y/n): ").lower() == 'y'
    
    if method == '1':
        train_reinforce(render=render)
    elif method == '2':
        train_actor_critic(render=render)
    elif method == '3':
        compare_methods(render=render)
    else:
        print("Invalid selection. Please run again and select 1, 2, or 3.")