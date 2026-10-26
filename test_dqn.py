
import numpy as np
import torch

from dqn_agent import DQNAgent

agent = DQNAgent()

state = np.zeros(64, dtype=np.float32)

action = agent.select_action(state, epsilon=0.0)

assert 0 <= action < 4

print("DQN agent initialized successfully!")
print("Selected action:", action)
print("Neural network:", agent.policy_net)
print("PyTorch version:", torch.__version__)

# Check that learning works after collecting enough experiences.
for _ in range(40):
    agent.remember(
        state, 0, -0.05, state, False
    )

loss = agent.learn()
assert loss is not None

print("Training step successful!")
print("Loss:", round(loss, 4))
print("MazeMind DQN is ready.")
