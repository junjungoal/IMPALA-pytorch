import os, sys
import time
import numpy as np
import six

import torch
import deepmind_lab
import torch.multiprocessing as mp
from collections import deque
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
  'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
  'look_right': _action(20, 0, 0, 0, 0, 0, 0),
  'forward_look_left': _action(-20, 0, 0, 1, 0, 0, 0),
  'forward_look_right': _action(20, 0, 0, 1, 0, 0, 0),
  'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
  'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
  'forward': _action(0, 0, 0, 1, 0, 0, 0),
  'backward': _action(0, 0, 0, -1, 0, 0, 0),
  'fire': _action(0, 0, 0, 0, 1, 0, 0),
}
CONFIG = {'height': '72', 'width': '96', 'logLevel': 'WARN'}

ACTION_LIST = list(six.viewvalues(ACTIONS))

class Actor(object):
    """
    Args:
    """
    def __init__(self, args, q_trace, learner, actor_critic, rollouts, level, actor_name=None):
        self.args = args
        self.q_trace = q_trace
        self.learner = learner
        self.actor_critic = actor_critic
        self.rollouts = rollouts
        self.actor_name = actor_name
        self.level = level

    def performing(self, rank):
        """
        """
        print('Build Environment for {}'.format(self.actor_name))
        self.env = deepmind_lab.Lab(self.level, ['RGB_INTERLEAVED', 'INSTR'], config=CONFIG) # INSTR: instruction
        torch.manual_seed(self.args.seed)
        writer = SummaryWriter(log_dir=self.args.result_dir)

        self.env.reset()
        state = self.env.observations()
        obs = state['RGB_INTERLEAVED'].transpose((2, 0, 1))
        instr = state['INSTR']
        done = True
        total_reward = 0.
        total_episode_length = 0
        num_episodes = 0

        iterations = 0
        timesteps = 0



        while True:
            self.actor_critic.load_state_dict(self.learner.actor_critic.state_dict())
            if done:
                recurrent_hidden_states = torch.zeros((1, 512))
            else:
                recurrent_hidden_states = recurrent_hidden_states.detach()


            self.rollouts.init()
            self.rollouts.obs[0].copy_(torch.from_numpy(obs))
            self.rollouts.to(device)


            for step in range(self.args.num_steps):
                value, action, action_log_prob, recurrent_hidden_states, logits, _ = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])
                reward = self.env.step(ACTION_LIST[int(action.item())], num_steps=4)
                state = self.env.observations()
                obs = torch.from_numpy(state['RGB_INTERLEAVED'].transpose((2, 0, 1)))
                instr = state['INSTR']

                total_reward += reward

                if self.args.reward_clipping == 'abs_one':
                    reward = np.clip(reward, -1, 1)
                else:
                    squeezed = np.tanh(reward/5.0)
                    if reward<0:
                        reward = squeezed*0.3
                    else:
                        reward = squeezed * 5.0

                masks = torch.FloatTensor([[0.0] if done else [1.0]])
                action_onehot = torch.zeros(self.actor_critic.n_actions)
                action_onehot[action] = 1.

                self.rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, torch.from_numpy(np.array([[reward]])), masks, logits, action_onehot)

                timesteps += 1
                if done:
                    num_episodes += 1
                    total_episode_length += 1
            self.q_trace.put((self.rollouts.obs[:, 0].detach().to("cpu"), self.rollouts.actions[:, 0].detach().to("cpu"), self.rollouts.rewards[:, 0].detach().to("cpu"),\
                    self.rollouts.action_log_probs[:, 0].detach().to("cpu"), self.rollouts.masks[:, 0].detach().to('cpu'), self.rollouts.logits[:, 0].detach().to('cpu'), self.rollouts.action_onehot[:, 0].detach().to('cpu')))
            if done:
                self.env.reset()
                obs = self.env.observations()['RGB_INTERLEAVED'].transpose((2, 0, 1))
                if timesteps >= self.args.total_num_steps:
                    writer.add_scalar(self.actor_name + '_total_reward', total_reward/num_episodes, iterations)
                    iterations += 1
                    total_reward = 0
                    num_episodes = 0
                    timesteps = 0

