import numpy

print(numpy.__file__)


import matplotlib
import matplotlib.pyplot as plt
import gymtorax
import gymnasium as gym
from IPython import get_ipython
from IPython.display import display, clear_output
import time

# Create environment using the default configuration
# An environment consists of the state, actions, and the reward of the physics-based simulator
env = gym.make(
    "gymtorax/IterHybrid-v0", render_mode="rgb_array", plot_config="default"
)  # Built-in TORAX plot configuration

# Reset environment to its starting conditions
# observation: the initial state, info: extra diagnostic info about the environment i.e. metadata
observation, info = env.reset()

# Run episode
terminated = False
while not terminated:
    # Sample a random action (replace with your RL agent later)
    action = env.action_space.sample()

    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode is over, reset the environment and exit loop
    if terminated or truncated:
        observation, info = env.reset()
        break  # Without break, another episode will automatically start

env.close()
print(observation.keys())
print("")
print(observation["profiles"].keys())
print("")
print(observation["scalars"].keys())

env = gym.make("gymtorax/IterHybrid-v0", render_mode="rgb_array", plot_config="default")
observation, info = env.reset()

print(env.action_space.sample())
print("")
print(env.action_space)
print("")
print(observation)

# img = env.render()
# get_ipython().run_line_magic('matplotlib', 'inline')  # Necessary to plot images with Gymnasium
# plt.imshow(img)
# plt.show()

env.close()
