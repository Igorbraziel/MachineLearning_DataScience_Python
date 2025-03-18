import pickle
from pathlib import Path
from time import sleep
from IPython.display import clear_output
import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Taxi-v3', render_mode='ansi')

q_table_path = Path(__file__).parent.parent / 'q_table.pkl'

with open(q_table_path, 'rb') as file_:
    q_table = pickle.load(file_)
    
total_penalties = 0
episodes = 50
frames = []

for _ in range(50):
    obs, info = env.reset()
    penalties, reward = 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[obs])
        obs, reward, done, truncated, info = env.step(action)
        
        if reward == -10:
            penalties += 1
            
        frames.append(
            {
                'frame': env.render(),
                'obs': obs,
                'action': action,
                'reward': reward,
            }
        )
    total_penalties += penalties
    
for frame in frames:
    clear_output(wait=True)
    print(frame['frame'])
    sleep(0.3)
    
print('Total penalties:', total_penalties)