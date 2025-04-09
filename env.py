
import numpy as np
from utils import euclidean_distance

class CVRPEnv:
    def __init__(self, coords, demands, depot, capacity):
        self.coords = coords
        self.demands = demands
        self.depot = depot
        self.capacity = capacity
        self.customers = [i for i in coords if i != depot]
        self.reset()

    def reset(self):
        self.remaining_customers = set(self.customers)
        self.current_node = self.depot
        self.remaining_capacity = self.capacity
        self.total_distance = 0
        self.visited_route = [self.depot]
        return self._get_state()

    def _get_state(self):
        return (self.current_node, self.remaining_capacity, tuple(sorted(self.remaining_customers)))

    def step(self, action):
        if action not in self.remaining_customers:
            return self._get_state(), -100, True  # Invalid move
        demand = self.demands[action]
        if demand > self.remaining_capacity:
            return self._get_state(), -100, True  # Overloaded

        distance = euclidean_distance(self.coords[self.current_node], self.coords[action])
        self.remaining_capacity -= demand
        self.current_node = action
        self.remaining_customers.remove(action)
        self.total_distance += distance
        self.visited_route.append(action)

        done = len(self.remaining_customers) == 0
        if done:
            self.total_distance += euclidean_distance(self.coords[self.current_node], self.coords[self.depot])
            self.visited_route.append(self.depot)

        return self._get_state(), -distance, done

    def render_route(self):
        return self.visited_route, self.total_distance
