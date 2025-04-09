
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(float)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table[(state, action)]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [(a, self.get_q(state, a)) for a in self.actions]
        q_values = sorted(q_values, key=lambda x: x[1], reverse=True)
        return q_values[0][0]

    def update(self, state, action, reward, next_state):
        max_q_next = max([self.get_q(next_state, a) for a in self.actions])
        current_q = self.q_table[(state, action)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        self.q_table[(state, action)] = new_q
