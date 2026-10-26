
import time

from q_learning import train_agent, get_path, GOAL
from dqn_trainer import train_dqn, get_dqn_path


def compare_algorithms(
    episodes=500,
    alpha=0.15,
    gamma=0.95,
    epsilon=1.0,
):
    results = {}

    # Q-Learning
    start_time = time.perf_counter()

    q_table, q_rewards, q_successes = train_agent(
        episodes, alpha, gamma, epsilon
    )

    q_time = time.perf_counter() - start_time
    q_path = get_path(q_table)

    results["Q-Learning"] = {
        "rewards": q_rewards,
        "successes": q_successes,
        "training_time": q_time,
        "path": q_path,
        "goal_reached": q_path[-1] == GOAL,
    }

    # DQN
    start_time = time.perf_counter()

    agent, dqn_rewards, dqn_successes, dqn_losses = train_dqn(
        episodes=episodes,
        learning_rate=0.001,
        gamma=gamma,
        epsilon=epsilon,
    )

    dqn_time = time.perf_counter() - start_time
    dqn_path = get_dqn_path(agent)

    results["DQN"] = {
        "rewards": dqn_rewards,
        "successes": dqn_successes,
        "training_time": dqn_time,
        "path": dqn_path,
        "goal_reached": dqn_path[-1] == GOAL,
        "losses": dqn_losses,
    }

    return results
