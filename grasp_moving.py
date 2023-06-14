import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

from hand_env.dy_grasping import DyGrasping
import time
from utils import FrameStack

# get current path
import os
cur_path = os.path.abspath(os.path.dirname(__file__))


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

    parser.add_argument('--env-name', default='BaxterPaddleGrasp_auto_position')
    parser.add_argument('--epoch_step', default=100)

    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.985, metavar='G',
                        help='discount factor for reward (default: 0.985)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--alpha_lr', type=float, default=1e-4, metavar='G',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=3000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    
    parser.add_argument('--traj', default="circle")
    parser.add_argument('--speed', default=1, type=float)
    parser.add_argument('--predict', default="True")    

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 10000)

    if args.predict == "True":
        predict = True
    else:
        predict = False
    
    env = FrameStack(DyGrasping(renders=True, max_episode_steps=args.epoch_step, reward_type="dense", control_model="p_o", traj=args.traj, predict=predict, speed=args.speed), 3)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    #args.hidden_size = 16
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_checkpoint(cur_path + "/checkpoints/BaxterPaddleGrasp_auto_position")
    # agent.load_checkpoint("trained_model/position")

    episodes = 20
    
    path = cur_path + "/results/"
    file_name = args.traj + "_" + str(args.speed) + "_" + str(args.predict) + ".txt"

    success_time = np.zeros((episodes, args.epoch_step))

    for eposide in range(episodes):

        state = env.reset()               
        episode_reward = 0
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):         

            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action) # Step

            # to observe
            # time.sleep(0.01)
            episode_reward += reward
            state = next_state  

            success_time[eposide, steps] = info["success"]
            steps += 1        

    # save success time
    np.savetxt(path + file_name, success_time, fmt="%d")
    env.close()