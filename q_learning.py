import numpy as np

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

def state_id(position):
    return position[0] * COLS + position[1]


def move(position, action):
    dr, dc = ACTIONS[action]
    nr = position[0] + dr
    nc = position[1] + dc

    # Stay in place when hitting a boundary or wall
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return position, -1.0, False

    if MAZE[nr, nc] == 1:
        return position, -1.0, False

    next_position = (nr, nc)

    if next_position == GOAL:
        return next_position, 10.0, True

    return next_position, -0.05, False


def train_agent(episodes, alpha, gamma, epsilon):
    q_table = np.zeros((ROWS * COLS, len(ACTIONS)))
    rewards = []
    successes = []

    for episode in range(episodes):
        position = START
        total_reward = 0.0
        success = False

        explore = max(0.02, epsilon * (1 - episode / episodes))

        for _ in range(150):
            s = state_id(position)

            if np.random.random() < explore:
                action = np.random.randint(len(ACTIONS))
            else:
                action = int(np.argmax(q_table[s]))

            next_position, reward, done = move(position, action)
            ns = state_id(next_position)

            target = reward
            if not done:
                target += gamma * np.max(q_table[ns])

            q_table[s, action] += alpha * (
                target - q_table[s, action]
            )

            position = next_position
            total_reward += reward

            if done:
                success = True
                break

        rewards.append(total_reward)
        successes.append(int(success))

    return q_table, rewards, successes


def get_path(q_table, max_steps=100):
    position = START
    path = [position]
    visited = {position}

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state_id(position)]))
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
