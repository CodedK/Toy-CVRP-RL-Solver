import random
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.total_actions = 0
        self.last_action = None
        self.consecutive_same_actions = 0
        self.max_consecutive_actions = 3  # Maximum allowed consecutive same actions
        self.env = None  # Reference to environment

    def set_environment(self, env):
        """Set reference to environment for accessing valid moves."""
        self.env = env

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair, initialize if not exists."""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]

    def choose_action(self, state, valid_moves):
        """Choose action using epsilon-greedy policy."""
        self.total_actions += 1

        if not valid_moves:
            return None

        # Exploration
        if np.random.random() < self.epsilon:
            action = random.choice(valid_moves)
            self.consecutive_same_actions = (
                0 if action != self.last_action else self.consecutive_same_actions + 1
            )
            self.last_action = action
            return action

        # Exploitation
        q_values = {action: self.get_q_value(state, action) for action in valid_moves}

        # Penalize actions that have been repeated too many times
        if (
            self.last_action in q_values
            and self.consecutive_same_actions >= self.max_consecutive_actions
        ):
            q_values[self.last_action] -= 1000  # Large penalty for too many repeats

        best_action = max(q_values.items(), key=lambda x: x[1])[0]

        # Update consecutive action counter
        self.consecutive_same_actions = (
            0 if best_action != self.last_action else self.consecutive_same_actions + 1
        )
        self.last_action = best_action

        return best_action

    def update(self, state, action, reward, next_state, next_valid_moves):
        """Update Q-values using Q-learning update rule."""
        if not next_valid_moves:  # Terminal state
            next_max_q = 0
        else:
            next_q_values = [
                self.get_q_value(next_state, next_action)
                for next_action in next_valid_moves
            ]
            next_max_q = max(next_q_values) if next_q_values else 0

        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)

        # Update Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_statistics(self):
        """Return agent statistics."""
        return {
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "q_table_size": len(self.q_table),
            "consecutive_actions": self.consecutive_same_actions,
        }
