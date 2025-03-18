import gym
import random

env = gym.make('Taxi-v3', render_mode='rgb_array')

# Training

from IPython.display import clear_output
import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table.shape)

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(1000000):
    obs, info = env.reset()
    penalties, reward = 0, 0
    done = False
    
    while not done:
        #Exploration
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs])
        
        next_state, reward, done, truncated, info = env.step(action)
        q_old = q_table[obs, action]
        next_max = np.max(q_table[next_state])
        q_new = (1 - alpha) * q_old + alpha * (reward + gamma * next_max)
        q_table[obs, action] = q_new

        if reward == -10:
            penalties += 1

        obs = next_state
    
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode:', i)
        
print('Training Finished')

import pickle
from pathlib import Path

q_table_path = Path(__file__).parent.parent / 'q_table.pkl'

with open(q_table_path, 'wb') as file_:
    pickle.dump(q_table, file_)