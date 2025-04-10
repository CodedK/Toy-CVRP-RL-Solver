from src.utils import euclidean_distance
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load configuration
config_path = Path(__file__).resolve().parent / "config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Use config parameters
instance_name = config["instance_name"]
n_episodes = config["n_episodes"]
capacity = config["capacity"]
depot_id = config["depot_id"]

# Example of colored print
print(f"{fg('green')}Starting CVRP solver...{attr('reset')}")
print(f"{fg('blue')}Project root: {project_root}{attr('reset')}")
print(f"{fg('yellow')}Loading problem instance: {instance_name}{attr('reset')}")

best_distances = []


class CVRPEnv:
    """CVRP environment class."""

    def __init__(self, instance_data):
        """Initialize the CVRP environment."""
        # Extract data from instance
        self.nodes = instance_data["nodes"]
        self.demands = instance_data["demands"]
        self.depot_id = instance_data["depot_id"]
        self.capacity = instance_data["capacity"]
        self.distances = instance_data["distances"]
        self.optimal_routes = instance_data.get("optimal_routes", None)
        self.optimal_distance = instance_data.get("optimal_distance", float("inf"))

        # State variables
        self.current_node = self.depot_id
        self.unvisited = set(self.nodes.keys()) - {self.depot_id}
        self.current_load = 0
        self.current_route = [self.depot_id]
        self.all_routes = []
        self.total_distance = 0
        self.last_depot_visit_idx = 0  # Index of last depot visit in current route
        self.best_solution = {"routes": [], "distance": float("inf")}

    def get_state(self):
        """Convert current environment state to a hashable representation."""
        unvisited_tuple = tuple(sorted(self.unvisited))
        current_route = self.current_route[self.last_depot_visit_idx :]
        current_route_tuple = tuple(current_route)
        at_depot = self.current_node == self.depot_id
        return (
            self.current_node,
            unvisited_tuple,
            self.current_load,
            current_route_tuple,
            at_depot,
        )

    def get_valid_moves(self):
        """Get list of valid next moves from current state."""
        valid_moves = []

        # Check if we can visit any unvisited nodes
        for node in self.unvisited:
            if self.current_load + self.demands[node] <= self.capacity:
                valid_moves.append(node)

        # Determine if returning to depot is valid
        nodes_visited_since_depot = (
            len(self.current_route) - self.last_depot_visit_idx - 1
        )
        can_return_to_depot = (
            self.current_node != self.depot_id  # Not already at depot
            and nodes_visited_since_depot > 0  # Visited some nodes since last depot
            and (
                not valid_moves or not self.unvisited
            )  # Can't add more nodes or all nodes visited
        )

        if can_return_to_depot:
            valid_moves.append(self.depot_id)

        return valid_moves

    def step(self, action):
        """Execute action and return new state, reward, done."""
        if action not in self.get_valid_moves():
            return self.get_state(), -1000, True  # Invalid action penalty

        # Commenting out debug prints
        # print(f"Current node: {self.current_node}, Action: {action}")
        # print(f"Number of nodes: {len(self.nodes)}")

        # Calculate distance for this step
        distance = self.distances[self.current_node][action]
        self.total_distance += distance

        # Update current position and load
        self.current_node = action
        if action != self.depot_id:
            self.current_load += self.demands[action]
            self.unvisited.remove(action)
        else:
            self.current_load = 0
            self.last_depot_visit_idx = len(self.current_route)

        # Update route
        self.current_route.append(action)

        # Check if solution is complete
        done = len(self.unvisited) == 0 and self.current_node == self.depot_id

        if done:
            # Complete solution found
            self.all_routes.append(self.current_route)
            reward = self.validate_solution()  # Validate and reward/penalize
            if reward > 0 and self.total_distance < self.best_solution["distance"]:
                self.best_solution = {
                    "routes": self.all_routes.copy(),
                    "distance": self.total_distance,
                }
            if distance < best_distance:
                best_distance = distance
                best_route = self.current_route.copy()
                best_distances.append(best_distance)
            else:
                best_distances.append(best_distance)
        else:
            # Intermediate reward based on local improvement
            reward = -distance  # Negative distance as immediate cost

        return self.get_state(), reward, done

    def reset(self):
        """Reset environment to initial state."""
        self.current_node = self.depot_id
        self.unvisited = set(self.nodes.keys()) - {self.depot_id}
        self.current_load = 0
        self.current_route = [self.depot_id]
        self.all_routes = []
        self.total_distance = 0
        self.last_depot_visit_idx = 0
        return self.get_state()

    def get_solution(self):
        """Return current solution."""
        return {
            "routes": self.all_routes if self.all_routes else [self.current_route],
            "distance": self.total_distance,
        }

    def render_route(self):
        """Return the complete route and its total distance."""
        complete_route = []
        for route in self.all_routes:
            if route:  # Only add non-empty routes
                complete_route.extend(route[:-1])  # Add route without final depot

        # Add final depot visit
        if complete_route:
            complete_route.append(self.depot_id)
        else:
            complete_route = [self.depot_id]  # Handle empty solution case

        total_distance = sum(
            self._calculate_distance(complete_route[i], complete_route[i + 1])
            for i in range(len(complete_route) - 1)
        )

        return complete_route, total_distance

    def _calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        return euclidean_distance(self.nodes[node1], self.nodes[node2])

    def get_total_distance(self):
        """Calculate total distance of all routes."""
        total_dist = 0
        for route in self.all_routes:
            total_dist += self._calculate_route_distance(route)
        return total_dist

    def _calculate_route_distance(self, route):
        """Calculate total distance of a route."""
        if len(route) < 2:
            return 0
        return sum(
            self._calculate_distance(route[i], route[i + 1])
            for i in range(len(route) - 1)
        )

    def validate_solution(self):
        """Validate the current solution and penalize if invalid."""
        visited_clients = set()
        valid = True
        total_demand = 0

        for route in self.all_routes:
            # Check if route starts and ends with the depot
            if route[0] != self.depot_id or route[-1] != self.depot_id:
                valid = False
                break

            # Check if route includes at least one customer
            if len(route) <= 2:  # Only depot and no customers
                valid = False
                break

            # Check capacity constraints
            route_demand = sum(
                self.demands[node] for node in route if node != self.depot_id
            )
            if route_demand > self.capacity:
                valid = False
                break

            # Check if each client is visited only once
            for node in route:
                if node != self.depot_id:
                    if node in visited_clients:
                        valid = False
                        break
                    visited_clients.add(node)

            total_demand += route_demand

        # Check if all clients are visited
        if visited_clients != set(self.demands.keys()):
            valid = False

        if not valid:
            # Penalize invalid solution
            return -1000

        return 1000 / self.total_distance  # Reward valid solution


# Plot minimization progress
plt.figure(figsize=(10, 5))
plt.plot(best_distances, label="Best Distance")
plt.xlabel("Episode")
plt.ylabel("Distance")
plt.title("Minimization Progress")
plt.legend()
plt.grid(True)
plt.show()
