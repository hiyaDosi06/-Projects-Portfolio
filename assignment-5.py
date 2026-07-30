import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 1. Define Deep Q-Network (DQN) Architecture
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# 2. Replay Buffer for Experience Replay
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


# 3. Environment & Hyperparameters Setup
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 0.001  # Learning rate
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
NUM_EPISODES = 200

# Networks & Optimizer
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# 4. Training Loop
epsilon = EPSILON_START
episode_rewards = []

print("Training Cart-Pole Agent...")
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_t)
                action = q_values.argmax().item()

        # Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Save Transition in Buffer
        replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        # Train Step via Replay Buffer
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = (
                replay_buffer.sample(BATCH_SIZE)
            )
            states, actions, rewards = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
            )
            next_states, dones = next_states.to(device), dones.to(device)

            # Compute current Q(s, a)
            q_values = (
                policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )

            # Compute Target Q = r + gamma * max_a Q_target(s', a)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + (GAMMA * max_next_q * (1 - dones))

            # Loss computation & Backpropagation
            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay Epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    # Sync Target Network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 20 == 0:
        avg_reward = np.mean(episode_rewards[-20:])
        print(
            f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward (Last 20): {avg_reward:.1f} | Epsilon: {epsilon:.2f}"
        )

env.close()

# 5. Plot Training Curve
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Reward per Episode")
plt.axhline(
    y=500, color="r", linestyle="--", label="Max Score Threshold (v1)"
)
plt.title("Cart-Pole Deep Q-Learning Performance")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()