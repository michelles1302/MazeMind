
import numpy as np

from dqn_agent import DQNAgent


# 0 = free cell, 1 = wall
MAZE = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
], dtype=int)

ROWS, COLS = MAZE.shape
START = (0, 0)
GOAL = (7, 7)

# Up, right, down, left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def encode_state(position):
    """Represent the robot's position as a 64-element vector."""
    state = np.zeros(ROWS * COLS, dtype=np.float32)
    index = position[0] * COLS + position[1]
    state[index] = 1.0
    return state


def move(position, action):
    """Apply an action and return next position, reward, and done."""
    dr, dc = ACTIONS[action]
    nr = position[0] + dr
    nc = position[1] + dc

    # Invalid moves receive a penalty.
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return position, -1.0, False

    if MAZE[nr, nc] == 1:
        return position, -1.0, False

    next_position = (nr, nc)

    if next_position == GOAL:
        return next_position, 10.0, True

    return next_position, -0.05, False


def train_dqn(
    episodes=500,
    learning_rate=0.001,
    gamma=0.95,
    epsilon=1.0,
    progress_callback=None,
):
    """Train a DQN agent and return its model and training metrics."""

    agent = DQNAgent(
        state_size=ROWS * COLS,
        action_size=len(ACTIONS),
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=32,
        memory_size=5000,
        target_update=100,
    )

    rewards = []
    successes = []
    losses = []

    for episode in range(episodes):
        position = START
        total_reward = 0.0
        success = False
        episode_losses = []

        # Gradually reduce exploration.
        explore = max(
            0.02,
            epsilon * (1 - episode / episodes),
        )

        for step in range(100):
            state = encode_state(position)

            action = agent.select_action(state, explore)

            next_position, reward, reached_goal = move(
                position, action
            )

            next_state = encode_state(next_position)

            # End the episode on success or at the step limit.
            done = reached_goal or step == 99

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            position = next_position
            total_reward += reward

            if reached_goal:
                success = True
                break

        rewards.append(total_reward)
        successes.append(int(success))

        losses.append(
            float(np.mean(episode_losses))
            if episode_losses else 0.0
        )

        if progress_callback is not None:
            progress_callback(
                (episode + 1) / episodes,
                episode + 1,
                episodes,
            )

    return agent, rewards, successes, losses


def get_dqn_path(agent, max_steps=100):
    """Run the trained agent greedily to obtain its route."""
    position = START
    path = [position]
    visited = {position}

    for _ in range(max_steps):
        state = encode_state(position)
        action = agent.select_action(state, epsilon=0.0)

        next_position, _, done = move(position, action)

        if next_position == position:
            break

        position = next_position
        path.append(position)

        if done:
            break

        if position in visited:
            break

        visited.add(position)

    return path
