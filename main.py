
import matplotlib.pyplot as plt
from utils import parse_vrp_file, euclidean_distance
from env import CVRPEnv
from q_learning import QLearningAgent

# Load data
coords, demands, depot_id, capacity = parse_vrp_file("A-n32-k5.vrp")
env = CVRPEnv(coords, demands, depot_id, capacity)
agent = QLearningAgent(actions=[i for i in coords if i != depot_id])

# Train agent
episodes = 500
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        possible_actions = list(env.remaining_customers)
        if not possible_actions:
            break
        agent.actions = possible_actions
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    if episode % 50 == 0:
        print(f"Episode {episode}, Total reward: {total_reward:.2f}")

# Get learned route
learned_route, total_distance = env.render_route()
print(f"Learned route: {learned_route}")
print(f"Total distance: {total_distance:.2f}")

# Plot learned route
def plot_route(coords, route, title, ax, color='blue'):
    x = [coords[node][0] for node in route]
    y = [coords[node][1] for node in route]
    ax.plot(x, y, marker='o', color=color)
    for i, node in enumerate(route):
        ax.text(coords[node][0], coords[node][1], str(node), fontsize=9)
    ax.set_title(title)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_route(coords, learned_route, "Learned Route", axs[0])

# Dummy optimal route (placeholder)
optimal_route = list(coords.keys())  # Replace with known optimal if available
plot_route(coords, optimal_route, "Dummy Optimal Route", axs[1], color='green')

plt.tight_layout()
plt.savefig("cvrp_rl_comparison.png")
plt.show()
