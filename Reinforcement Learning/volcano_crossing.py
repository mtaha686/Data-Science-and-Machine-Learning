import numpy as np


class VolcanoCrossing:
    def __init__(self, grid_size=5, slip_probability=0.0):
        self.grid_size = grid_size
        self.slip_probability = slip_probability
        self.states = [(i, j) for i in range(grid_size)
                       for j in range(grid_size)]
        self.actions = ['up', 'down', 'left', 'right']
        self.start_state = (0, 0)
        self.goal_state = (grid_size - 1, grid_size - 1)
        # Define rewards for end states and obstacles
        self.rewards = {(self.goal_state): 1, (2, 2): -1}

    def get_next_state(self, state, action):
        if action == 'up':
            next_state = (max(0, state[0] - 1), state[1])
        elif action == 'down':
            next_state = (min(self.grid_size - 1, state[0] + 1), state[1])
        elif action == 'left':
            next_state = (state[0], max(0, state[1] - 1))
        elif action == 'right':
            next_state = (state[0], min(self.grid_size - 1, state[1] + 1))

        # Introduce slip probability
        if np.random.rand() < self.slip_probability:
            return self.get_random_state()
        return next_state

    def get_random_state(self):
        return tuple(np.random.choice(self.grid_size, size=2))
