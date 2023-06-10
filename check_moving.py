import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
import time
from hand_env.dy_grasping import DyGrasping

env = DyGrasping(renders=True, max_episode_steps=100, reward_type="dense", control_model="p_o", traj="line")

for i in range(10000): 
    action = np.array([0.0, 0.0, 0.0, 0.0, 0])
    next_state, reward, terminated, truncated, _ = env.step(action)
    # time.sleep(0.1)