import os, sys
import argparse
import shutil
import yaml
from copy import deepcopy

import numpy as np
import deepmind_lab
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue, Queue
from gym.spaces import Box
from gym.spaces import Discrete
from models.policy import Policy
from agents.learner import Learner
from agents.actor import Actor
from misc.q_manager import QManager
from misc.storage import RolloutStorage
from levels import LEVELS



CONFIG = {'width': '96', 'height': '72'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=96,
                      help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=72,
                      help='Vertical size of the observations')
    parser.add_argument('--level_script', type=str,
                      default='tests/empty_room_test',
                      help='The environment level script to load')
    parser.add_argument('--experiment_id', type=int, required=True,
                      help='Experiment ID')
    parser.add_argument('--lr', type=float, default=0.00001,
                      help='Learning rate')
    parser.add_argument('--num_actors', type=int, default=1,
                      help='Number of Actors')
    parser.add_argument('--num_steps', type=int, default=200,
                      help='Number of Steps to learn')
    parser.add_argument('--total_num_steps', type=int, default=4096,
                      help='Number of Steps to learn')
    parser.add_argument('--seed', type=int, default=2019,
                      help='Random seed')
    parser.add_argument('--coef_hat', type=float, default=1.0)
    parser.add_argument('--rho_hat', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount rate')
    parser.add_argument('--entropy_coef', type=float, default=0.0033)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=40)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--is_instruction', type=str2bool, default=True)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    processes = []
    q_trace = Queue(maxsize=300)
    q_batch = Queue(maxsize=3)
    q_manager = QManager(args, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()
    processes.append(p)

    envs = []
    actors = []

    args.result_dir = os.path.join('results',str(args.experiment_id))
    args.model_dir = os.path.join(args.result_dir, 'models')

    try:
        os.makedirs(args.model_dir)
    except:
        shutil.rmtree(args.model_dir)
        os.makedirs(args.model_dir)

    env = deepmind_lab.Lab('tests/empty_room_test', ['RGB_INTERLEAVED'], config=CONFIG)
    env.reset()

    obs_shape = env.observations()['RGB_INTERLEAVED'].shape
    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    print('Observation Space: ', obs_shape)
    action_space = Discrete(9)
    env.close()

    actor_critic = Policy(
        obs_shape,
        action_space)
    actor_critic.to(args.device)


    learner = Learner(args, q_batch, actor_critic)


    for i in range(args.num_actors):
        print('Build Actor {:d}'.format(i))
        rollouts = RolloutStorage(args.num_steps,
                                  1,
                                  obs_shape,
                                  action_space,
                                  actor_critic.recurrent_hidden_state_size)
        actor_critic = Policy(
            obs_shape,
            action_space)
        actor_critic.to(args.device)

        actor_name = 'actor_' + str(i)
        actor = Actor(args, q_trace, learner, actor_critic, rollouts, LEVELS[i], actor_name)
        actors.append(actor)

    print('Run processes')

    for rank, a in enumerate(actors):
        p = mp.Process(target=a.performing, args=(rank,))
        p.start()
        processes.append(p)

    learner.learning()

    for p in processes:
        p.join()
