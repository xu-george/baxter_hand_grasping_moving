import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
import time
from hand_env.dy_grasping import DyGrasping

env = DyGrasping(renders=True, max_episode_steps=100, reward_type="dense", control_model="p_o", traj="line", predict=True)

for i in range(100000): 

    if (i +1) % 100 == 0:
        env.reset()
    action = np.array([0.0, 0.0, 0.0, 0.0, 0])
    next_state, reward, terminated, truncated, _ = env.step(action)
    # time.sleep(0.1)