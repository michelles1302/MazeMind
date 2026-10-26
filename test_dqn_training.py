
from dqn_trainer import train_dqn, get_dqn_path, GOAL

print("Starting DQN maze training...")

agent, rewards, successes, losses = train_dqn(
    episodes=500,
    learning_rate=0.001,
    gamma=0.95,
    epsilon=1.0,
    progress_callback=lambda fraction, episode, total: (
        print(f"Episode {episode}/{total}")
        if episode % 50 == 0 or episode == total
        else None
    ),
)

path = get_dqn_path(agent)

print("\nTraining finished!")
print(f"Success rate: {100 * sum(successes[-100:]) / len(successes[-100:]):.1f}%")
print(f"Average reward (last 100): {sum(rewards[-100:]) / len(rewards[-100:]):.2f}")
print(f"Learned route: {path}")
print(f"Goal reached: {path[-1] == GOAL}")
print(f"Final training loss: {losses[-1]:.4f}")
