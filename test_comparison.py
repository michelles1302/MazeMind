
from comparison import compare_algorithms

print("Starting Q-Learning vs DQN comparison...")
print("Training both algorithms for 500 episodes.\n")

results = compare_algorithms(episodes=500)

for name, result in results.items():
    success_rate = 100 * sum(result["successes"][-100:]) / len(
        result["successes"][-100:]
    )
    avg_reward = sum(result["rewards"][-100:]) / len(
        result["rewards"][-100:]
    )

    print(f"\n--- {name} ---")
    print(f"Success rate (last 100 episodes): {success_rate:.1f}%")
    print(f"Average reward (last 100 episodes): {avg_reward:.2f}")
    print(f"Training time: {result['training_time']:.2f} seconds")
    print(f"Learned route length: {len(result['path']) - 1} steps")
    print(f"Goal reached by learned route: {result['goal_reached']}")

print("\nComparison finished!")
