import math

import numpy as np
import pandas as pd
import random

np.set_printoptions(threshold=np.inf)

# Importing the results of the GBM simulations
excel_file = r"C:\Users\aufie\OneDrive\Desktop\state_space.xlsx"
df = pd.read_excel(excel_file, index_col=0)
general_state_space = df.to_numpy()
vbe_state_space = general_state_space[0:11]
be_state_space = general_state_space[12:31]
la_state_space = general_state_space[32:49]
bu_state_space = general_state_space[50:72]
vbu_state_space = general_state_space[73:98]

state_space = ['VERY_BEARISH', 'BEARISH', 'LATERALIZING', 'BULLISH', 'VERY_BULLISH']
num_returns = 99

# Define possible action combinations with weights
action_space = [(0, 1), (0.1, 0.9), (0.2, 0.8),
                (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
                (0.6, 0.4), (0.7, 0.3), (0.8, 0.2),
                (0.9, 0.1), (1, 0)]

num_states = len(state_space)
num_actions = len(action_space)
Q = 0.0001*np.random.rand(num_states, num_actions)

# Define hyperparameters
gamma = 0.9
num_episodes = 10000000
weight_ABS = 0.5
weight_FIC = 0.5


def select_action(g_state, epsilon_value):
    if random.uniform(0, 1) < epsilon_value:
        g_action_idx = np.random.randint(num_actions)  # Exploration
    else:
        # Choose the action with the highest Q-value from either Q1 or Q2
        g_action_idx = np.argmin(Q[g_state, :])  # Exploitation
    return g_action_idx


def select_state(g_situation):
    if general_state_space[g_situation, 4] < -0.15:
        g_state = 0
    elif -0.15 < general_state_space[g_situation, 4] < -0.05:
        g_state = 1
    elif -0.05 < general_state_space[g_situation, 4] < 0.05:
        g_state = 2
    elif 0.05 < general_state_space[g_situation, 4] < 0.15:
        g_state = 3
    else:
        g_state = 4

    return g_state


def select_state_space(g_state):
    if g_state == 0:
        g_state_space = vbe_state_space
    elif g_state == 1:
        g_state_space = be_state_space
    elif g_state == 2:
        g_state_space = la_state_space
    elif g_state == 3:
        g_state_space = bu_state_space
    else:
        g_state_space = vbu_state_space

    return g_state_space


# Q-learning algorithm
for trial in range(10):
    for episode in range(num_episodes):

        situation = np.random.randint(num_returns)

        state = select_state(situation)

        action_idx = select_action(state, 0.05)

        chosen_action = action_space[action_idx]

        cost = 0

        current_state_space = select_state_space(state)

        for instance in range(10):
            occurrence = np.random.randint(len(current_state_space))

            etf_value = chosen_action[0] * current_state_space[occurrence, 1] + chosen_action[1] * \
                current_state_space[occurrence, 2]

            index_value = current_state_space[occurrence, 4]

            cost += math.sqrt(((etf_value - index_value) ** 2))

        next_situation = np.random.randint(num_returns)

        next_state = select_state(next_situation)

        Q[state, action_idx] = (
                Q[state, action_idx] +
                ((1 / (episode + 1)) ** 0.51) * (
                        cost + gamma * Q[next_state, np.argmin(Q[next_state, :])] - Q[state, action_idx])
        )

    policy = [np.argmin(Q[0]), np.argmin(Q[1]), np.argmin(Q[2]),
              np.argmin(Q[3]), np.argmin(Q[4])]
    print(Q)
    print(policy)

    total_costs = 0
    excel_file = r"C:\Users\aufie\OneDrive\Desktop\general_state_space.xlsx"
    df1 = pd.read_excel(excel_file, index_col=0)
    actual_state_space = df1.to_numpy()

    for instance in range(num_returns):
        state = select_state(instance)

        # Choose the best action according to the learned Q-values (exploitation only)
        action = np.argmin(Q[state])
        chosen_action = action_space[action]

        # Simulate the action in the environment (assuming your environment has a step function)
        # next_state, reward = env.step(state, action)

        # Calculate the reward (modify based on your environment)
        etf_value = chosen_action[0] * actual_state_space[instance, 1] + chosen_action[1] * \
            actual_state_space[instance, 2]
        index_value = actual_state_space[instance, 4]
        cost0 = ((etf_value - index_value) ** 2)

        total_costs += cost0  # Accumulate rewards

    average_test_cost = total_costs / num_returns
    print("Average test cost:", average_test_cost)


df1 = pd.DataFrame(Q)
df1.to_excel(excel_writer=r"C:\Users\aufie\OneDrive\Desktop\q_table.xlsx")