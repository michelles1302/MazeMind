
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Neural network predicting the value of each action."""

    def __init__(self, state_size=64, action_size=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    def __init__(
        self,
        state_size=64,
        action_size=4,
        learning_rate=0.001,
        gamma=0.95,
        batch_size=32,
        memory_size=5000,
        target_update=100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cpu")

        self.policy_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.memory = deque(maxlen=memory_size)
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def remember(
        self, state, action, reward, next_state, done
    ):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory, self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.as_tensor(
            np.array(states), dtype=torch.float32
        )
        actions = torch.as_tensor(
            actions, dtype=torch.long
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32
        )
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32
        )

        current_q = self.policy_net(states).gather(
            1, actions
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(
                next_states
            ).max(dim=1).values

            target_q = rewards + (
                self.gamma * next_q * (1 - dones)
            )

        loss = nn.functional.smooth_l1_loss(
            current_q, target_q
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 10.0
        )

        self.optimizer.step()
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())
