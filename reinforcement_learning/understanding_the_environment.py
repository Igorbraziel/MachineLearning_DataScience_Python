import gym
import random

env = gym.make('Taxi-v3', render_mode='rgb_array')

# Reset the environment
obs, info = env.reset()

import matplotlib.pyplot as plt
# Render and dispaly the image
img = env.render()
plt.imshow(img)
plt.show()

# Decode the state (posição do táxi, localização do passageiro e destino)
taxi_row, taxi_col, passenger_loc, destination = env.unwrapped.decode(obs)

# Maping of destination for colors
colors = ["Red (R)", "Green (G)", "Yellow (Y)", "Blue (B)"]

print(f"The passenger must be taken for the position of color: {colors[destination]}")

# Training

from IPython.display import clear_output
import numpy as np
print(env.observation_space.shape)

