import numpy as np
from volcano_crossing import VolcanoCrossing


def run_q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    Q_values = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    for episode in range(num_episodes):
        state = env.start_state

        while state != env.goal_state:
            action = epsilon_greedy_policy(Q_values, state, epsilon, env)
            next_state = env.get_next_state(state, action)
            reward = env.rewards.get(next_state, 0)

            # Update Q-values using Q-Learning
            Q_values[state[0], state[1], env.actions.index(action)] += alpha * (
                reward + gamma * np.max(Q_values[next_state[0], next_state[1]]) -
                Q_values[state[0], state[1], env.actions.index(action)]
            )

            state = next_state

    average_utility = np.mean(Q_values)
    return Q_values, average_utility


def epsilon_greedy_policy(Q_values, state, epsilon, env):
    if np.random.rand() < epsilon:
        return np.random.choice(env.actions)
    else:
        return env.actions[np.argmax(Q_values[state[0], state[1]])]
