import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99  # Discount factor
LR = 5e-4  # Learning rate
BUFFER_SIZE = 100_000  # Replay memory capacity
BATCH_SIZE = 64  # Mini-batch size for gradient descent
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay factor per episode
TARGET_UPDATE_FREQ = 4  # How often to update target Q-network
MAX_EPISODES = 500  # Total training episodes
MAX_STEPS = 1000  # Max steps per episode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==========================================
# 2. Deep Q-Network Architecture
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 3. Replay Buffer for Experience Replay
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 4. DQN Agent
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = EPSILON_START

        # Online Network & Target Network
        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            BATCH_SIZE
        )

        # Compute Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(
            1
        )

        # Compute V(s') using Target Net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = rewards + (
                GAMMA * next_q_values * (1 - dones)
            )

        # Loss calculation (Smooth L1 Loss / Huber Loss)
        loss = nn.MSELoss()(
            state_action_values, expected_state_action_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ==========================================
# 5. Training Loop Entry Point
# ==========================================
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    scores_window = deque(maxlen=100)

    print(f"\n--- Training DQN on {ENV_NAME} ---")

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        score = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            score += reward

            if done:
                break

        agent.update_epsilon()

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        scores_window.append(score)
        avg_score = np.mean(scores_window)

        if episode % 20 == 0 or avg_score >= 200.0:
            print(
                f"Episode {episode}\tAverage Score (Last 100): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}"
            )

        if avg_score >= 200.0:
            print(f"\n🎉 Environment solved in {episode} episodes!")
            torch.save(
                agent.policy_net.state_dict(), "lunar_lander_dqn.pth"
            )
            print("Model checkpoint saved to 'lunar_lander_dqn.pth'.")
            break

    env.close()


if __name__ == "__main__":
    train()