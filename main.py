import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
from rl.env import CVRPEnv
from src.rl.q_learning import QLearningAgent
from src.utils import parse_vrp_file, parse_solution_file, plot_route
import time
from scipy.spatial import distance
import json
from termcolor import fg, attr

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Load configuration from the rl folder
config_path = Path(__file__).resolve().parent / "rl" / "config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Use config parameters
n_episodes = config["n_episodes"]
learning_rate = config["learning_rate"]
discount_factor = config["discount_factor"]

print(f"{fg('green')}Starting CVRP solver...{attr('reset')}")

# Get the project root directory (parent of src)
project_root = Path(__file__).resolve().parent.parent
print(f"Project root: {project_root}")

# Load and parse data
instance_name = "A-n32-k5"
vrp_file = project_root / "data" / f"{instance_name}.vrp"
sol_file = project_root / "data" / f"{instance_name}.sol"

print(f"\nLoading problem instance: {instance_name}")
print(f"VRP file path: {vrp_file}")
print(f"Solution file path: {sol_file}")

if not vrp_file.exists():
    print(f"Error: VRP file not found at {vrp_file}")
    sys.exit(1)

if not sol_file.exists():
    print(f"Error: Solution file not found at {sol_file}")
    sys.exit(1)

coords, demands, capacity, depot_id = parse_vrp_file(str(vrp_file))
optimal_routes, optimal_distance = parse_solution_file(str(sol_file))

print(f"Problem size: {len(coords)} nodes")
print(f"Vehicle capacity: {capacity}")
print(f"Optimal distance: {optimal_distance}")

# Adjust node indexing to be zero-based
coords = {i: coords[node_id] for i, node_id in enumerate(coords.keys())}
demands = {i: demands[node_id] for i, node_id in enumerate(demands.keys())}

depot_id = 0  # Assuming the depot is the first node

# Calculate distances between all nodes
num_nodes = len(coords)
distances = [[0] * num_nodes for _ in range(num_nodes)]
node_keys = list(coords.keys())  # Get the keys of the nodes

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            node_i = node_keys[i]
            node_j = node_keys[j]
            distances[i][j] = distance.euclidean(coords[node_i], coords[node_j])

# Update instance_data with calculated distances
instance_data = {
    "nodes": coords,
    "demands": demands,
    "depot_id": depot_id,
    "capacity": capacity,
    "distances": distances,  # Include calculated distances
    "optimal_routes": optimal_routes,
    "optimal_distance": optimal_distance,
}

env = CVRPEnv(instance_data)
agent = QLearningAgent()

# Training loop
best_distance = float("inf")
best_route = None
print(f"\nStarting training for {n_episodes} episodes...")

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    moves_this_episode = 0
    episode_start_time = time.time()

    # Print episode start status
    print(f"Episode {episode + 1}/{n_episodes}, Moves: 0", end="\r", flush=True)

    while not done:
        # Get valid moves and choose action
        valid_moves = env.get_valid_moves()
        action = agent.choose_action(state, valid_moves)

        # Take action
        next_state, reward, done = env.step(action)

        # Get valid moves for next state
        next_valid_moves = env.get_valid_moves() if not done else []

        # Update Q-values
        agent.update(state, action, reward, next_state, next_valid_moves)

        total_reward += reward
        state = next_state
        moves_this_episode += 1

        # Update progress on the same line
        print(
            f"Episode {episode + 1}/{n_episodes}, Moves: {moves_this_episode}, Last action: {action}",
            end="\r",
            flush=True,
        )

    # Get final route and distance
    route, distance = env.render_route()
    episode_time = time.time() - episode_start_time

    # Update best solution
    if distance < best_distance:
        best_distance = distance
        best_route = route.copy()
        print(f"New best solution! Distance: {distance:.2f}", end=" ")
        if optimal_distance:
            gap = ((distance - optimal_distance) / optimal_distance) * 100
            print(f"(Gap: {gap:.2f}%)")
        else:
            print()

    # Decay exploration rate
    agent.decay_epsilon()

    # Print detailed progress every 10 episodes
    if (episode + 1) % 100 == 0:
        stats = agent.get_statistics()
        print(f"\nProgress after {episode + 1} episodes:")
        print(
            f"epsilon: {stats['epsilon']:.3f} | Actions: {stats['total_actions']} | Q-table: {stats['q_table_size']}"
        )
        print(f"Current: {distance:.2f} | Best: {best_distance:.2f}", end=" ")
        if optimal_distance:
            gap = ((best_distance - optimal_distance) / optimal_distance) * 100
            print(f"| Gap: {gap:.2f}%")
        print("-" * 50)

print("\nTraining completed!")
print(f"Best distance found: {best_distance:.2f}")
if optimal_distance:
    final_gap = ((best_distance - optimal_distance) / optimal_distance) * 100
    print(f"Final gap to optimal: {final_gap:.2f}%")

# Get final solution
print("\nGenerating final solution...")
learned_route, total_distance = env.render_route()

# Print final routes
print("\nFinal Routes found by RL agent:")
current_route = []
route_num = 1
total_demand = 0

for i, node in enumerate(learned_route):
    if node == depot_id:
        if current_route:  # If we have a route to print
            route_str = f"Route #{route_num}: {' '.join(map(str, current_route[1:]))}"  # Skip the starting depot
            route_demand = sum(demands[n] for n in current_route)
            print(f"{route_str} (Demand: {route_demand})")
            total_demand += route_demand
            route_num += 1
        current_route = [node]  # Start new route
    else:
        current_route.append(node)

# Print the last route if it exists and doesn't end with depot
if current_route and current_route[-1] != depot_id:
    current_route.append(depot_id)
    route_str = f"Route #{route_num}: {' '.join(map(str, current_route[1:-1]))}"  # Skip both depots
    route_demand = sum(demands[n] for n in current_route)
    print(f"{route_str} (Demand: {route_demand})")
    total_demand += route_demand

print(f"\nTotal Demand: {total_demand}")
print(f"Total Distance: {total_distance:.2f}")

if optimal_distance:
    print(f"Optimal Distance: {optimal_distance:.2f}")
    gap = ((total_distance - optimal_distance) / optimal_distance) * 100
    print(f"Gap to optimal: {gap:.2f}%")

print("\nGenerating plots...")
# Plot final routes
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot learned route
plot_route(
    coords,
    learned_route,
    "Learned Route",
    axs[0],
    total_distance=total_distance,
    optimal_distance=optimal_distance,
    depot_id=depot_id,
)

# Plot optimal route if available
if optimal_routes:
    print("\nOptimal Routes:")
    # Process optimal routes correctly
    optimal_route = []
    for i, route in enumerate(optimal_routes, 1):
        # Each route should start and end with depot
        route_with_depot = [depot_id] + route + [depot_id]
        optimal_route.extend(
            route_with_depot[:-1]
        )  # Don't add depot twice between routes
        print(f"Route {i}: {' -> '.join(map(str, route))}")
    optimal_route.append(depot_id)  # Add final depot visit

    plot_route(
        coords,
        optimal_route,
        "Optimal Route",
        axs[1],
        total_distance=optimal_distance,
        depot_id=depot_id,
    )
else:
    print("\nNo optimal routes found in solution file.")
    axs[1].set_title("Optimal Route (Not Available)")

plt.tight_layout()
print("\nSaving plot to 'cvrp_rl_comparison.png'...")
plt.savefig("cvrp_rl_comparison.png")
plt.show()

print("\nDone!")
