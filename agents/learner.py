import os, sys


import torch
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, args, q_batch, actor_critic):
        self.args = args
        self.q_batch = q_batch
        self.actor_critic = actor_critic
        self.optimizer = RMSprop(self.actor_critic.parameters(), lr=args.lr)
        self.actor_critic.share_memory()

    def learning(self):
        writer = SummaryWriter(log_dir=self.args.result_dir)
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

                value, action_log_prob, logits, recurrent_hidden_states =  self.actor_critic.evaluate_actions(obs[:, step], recurrent_hidden_states, masks[:, step], actions[:, step])
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
                policy = F.softmax(logits, dim=1)
                log_policy = F.log_softmax(logits, dim=1)
                entropy = torch.sum(-policy*log_policy)
                entropies.append(entropy)

                log_prob.append(action_log_prob)

            policy_loss = 0
            baseline_loss = 0
            entropy_loss = 0
            vs = torch.zeros((obs.size(1)-1, obs.size(0))).to(device)

            """
            vs: v-trace target
            """
            for rev_step in reversed(range(obs.size(1)-1)):
                # r + args * v(s+1) - V(s)
                #fix_vp = rewards[:, rev_step] + self.args.gamma * (values[rev_step+1]+value_loss) - values[rev_step]
                delta_s = rho[rev_step] * (rewards[:, rev_step] + self.args.gamma * values[rev_step+1]-values[rev_step])
                # value_loss = v_{s} - V(x_{s})
                advantages = rho[rev_step] * (rewards[:, rev_step] + self.args.gamma * vs[rev_step] - values[rev_step])

                vs[rev_step-1] = values[rev_step] + delta_s + self.args.gamma * coef[rev_step] * (vs[rev_step]-values[rev_step+1])

                policy_loss += log_prob[rev_step]*advantages.detach()

            baseline_loss = torch.sum(0.5*(vs.detach() - torch.stack(values))**2)
            entropy_loss = self.args.entropy_coef*torch.sum(torch.stack(entropies))
            policy_loss = policy_loss.sum()
            loss = policy_loss + self.args.value_loss_coef * baseline_loss - entropy_loss

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
            print("v_loss {:.3f} p_loss {:.3f} entropy_loss {:.5f} loss {:.3f}".format(baseline_loss.item(), policy_loss.item(), entropy_loss.item(), loss.item()))
            self.optimizer.step()
            writer.add_scalar('total_loss', float(loss.item()), i)


            if (i % self.args.save_interval == 0):
                torch.save(self.actor_critic, os.path.join(self.args.model_dir, "impala.pt"))
            i+= 1

