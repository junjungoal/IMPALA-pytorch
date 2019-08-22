import os, sys


import torch
import time
import numpy as np
import torch.multiprocessing as mp
from torch.optim import RMSprop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, args, q_batch, actor_critic):
        self.args = args
        self.q_batch = q_batch
        self.actor_critic = actor_critic
        self.argsimizer = RMSprop(self.actor_critic.parameters(), lr=args.lr)
        self.actor_critic.share_memory()

    def learning(self):
        torch.manual_seed(self.args.seed)
        coef_hat = torch.Tensor([[self.args.coef_hat]]).to(device)
        rho_hat = torch.Tensor([[self.args.rho_hat]]).to(device)

        i = 0

        while True:
            values, coef, rho, entropies, log_prob = [], [], [], [], []
            obs, actions, rewards, log_probs, masks, mu_logits, action_onehot = self.q_batch.get(block=True)
            #print('Get batch: obs: {}, action: {}, reward: {}, prob: {}'.format(obs.shape, actions.shape, rewards.shape, probs.shape))
            obs_shape = obs.shape[3:]


            recurrent_hidden_states = torch.zeros((self.args.batch_size, self.actor_critic.recurrent_hidden_state_size), device=device)
            for step in range(obs.size(1)):
                if step >= actions.size(1):  # noted that s[, n_step+1, ...] but a[, n_step,...]
                    value  =  self.actor_critic.get_value(obs[:, step], recurrent_hidden_states, masks[:, step])
                    values.append(value)
                    break

                value, action_log_prob, dist_entropy, recurrent_hidden_states =  self.actor_critic.evaluate_actions(obs[:, step], recurrent_hidden_states, masks[:, step], actions[:, step])
                values.append(value)

                #logit_a = action_onehot[:, step] * logits.detach() + (1-action_onehot[:, step]) * (1-logits.detach())
                #logit_a = logit_a.detach()
                #prob_a = action_onehot[:, step] * mu_logits[:, step] + (1-action_onehot[:, step]) * (1-mu_logits[:, step])

                #print(torch.exp(action_log_prob.detach()-log_probs[:, step]))

                #is_rate = torch.cumprod(logit_a/(prob_a+1e-6), dim=1)[:, -1]
                #is_rate = torch.sum(torch.exp(logit_a - prob_a), dim=1)
                #print(torch.exp(-action_log_prob.detach()+log_probs[:, step]))
                is_rate = torch.exp(action_log_prob.detach() - log_probs[:, step])
                coef.append(torch.min(coef_hat, is_rate))
                rho.append(torch.min(rho_hat, is_rate))
                entropies.append(dist_entropy)
                log_prob.append(action_log_prob)

            policy_loss = 0
            value_loss = 0
            for rev_step in reversed(range(obs.size(1)-1)):
                # r + args * v(s+1) - V(s)
                fix_vp = rewards[:, rev_step] + self.args.gamma * (values[rev_step+1]+value_loss) - values[rev_step]

                delta_v = rho[rev_step] * (rewards[:, rev_step] + self.args.gamma * values[rev_step+1] - values[rev_step])
                # value_loss = v_{s} - V(x_{s})
                value_loss = self.args.gamma * coef[rev_step] * value_loss + delta_v

                policy_loss = policy_loss \
                                - rho[rev_step]*log_prob[rev_step]*fix_vp.detach() \
                                - self.args.entropy_coef * entropies[rev_step]

            self.argsimizer.zero_grad()
            policy_loss = policy_loss.sum()
            value_loss = value_loss.sum()
            loss = policy_loss + self.args.value_loss_coef * value_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
            #print("v_loss {:.3f} p_loss {:.3f}".format(value_loss.item(), policy_loss.item()))
            self.argsimizer.step()


            if (i % self.args.save_interval == 0):
                torch.save(self.actor_critic, os.path.join(self.args.model_dir, "impala.pt"))
            i+= 1

