import torch
import time
import numpy as np
import torch.multiprocessing as mp
from torch.optim import RMSprop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Learner(object):
    def __init__(self, opt, q_batch, actor_critic):
        self.opt = opt
        self.q_batch = q_batch
        self.actor_critic = actor_critic
        self.optimizer = RMSprop(self.actor_critic.parameters(), lr=opt.lr)
        self.actor_critic.share_memory()

    def learning(self):
        torch.manual_seed(self.opt.seed)
        coef_hat = torch.Tensor([[self.opt.coef_hat]]).to(device)
        rho_hat = torch.Tensor([[self.opt.rho_hat]]).to(device)

        while True:
            values, coef, rho, entropies, log_prob = [], [], [], [], []
            obs, actions, rewards, log_probs, masks = self.q_batch.get(block=True)
            #print('Get batch: obs: {}, action: {}, reward: {}, prob: {}'.format(obs.shape, actions.shape, rewards.shape, probs.shape))
            obs_shape = obs.shape[3:]


            recurrent_hidden_states = torch.zeros((self.opt.batch_size, self.actor_critic.recurrent_hidden_state_size), device=device)
            for step in range(obs.size(1)):
                if step >= actions.size(1):  # noted that s[, n_step+1, ...] but a[, n_step,...]
                    value  =  self.actor_critic.get_value(obs[:, step], recurrent_hidden_states, masks[:, step])
                    values.append(value)
                    break

                value, action_log_prob, dist_entropy, recurrent_hidden_states =  self.actor_critic.evaluate_actions(obs[:, step], recurrent_hidden_states, masks[:, step], actions[:, step])
                values.append(value)

                #action_onehot = torch.zeros((self.opt.batch_size, self.actor_critic.n_actions), device=device)
                #action_onehot.scatter_(1, actions[:, step].long(), 1)
                #logit_a = action_onehot * pi_logits + (1-action_onehot) * (1-pi_logits)
                #logit_a = logit_a.detach()
                #prob_a = action_onehot * logits[:, step] + (1-action_onehot) * (1-logits[:, step])

                #is_rate = torch.cumprod(logit_a/(prob_a+1e-6), dim=1)[:, -1]
                is_rate = torch.exp(action_log_prob - log_probs[:, step])
                coef.append(torch.min(coef_hat, is_rate))
                rho.append(torch.min(rho_hat, is_rate))
                entropies.append(dist_entropy)
                log_prob.append(action_log_prob)

            policy_loss = 0
            value_loss = 0
            for rev_step in reversed(range(obs.size(1)-1)):
                # r + gamma * v(s+1) - V(s)
                fix_vp = rewards[:, rev_step] + self.opt.gamma * (values[rev_step+1]+value_loss) - values[rev_step]

                delta_v = rho[rev_step] * (rewards[:, rev_step] + self.opt.gamma * values[rev_step+1] - values[rev_step])
                # value_loss = v_{s} - V(x_{s})
                value_loss = self.opt.gamma * coef[rev_step] * value_loss + delta_v

                policy_loss = policy_loss \
                                - rho[rev_step]*log_prob[rev_step]*fix_vp.detach() \
                                - self.opt.entropy_coef * entropies[rev_step]

            self.optimizer.zero_grad()
            policy_loss = policy_loss.sum()
            value_loss = value_loss.sum()
            loss = policy_loss + self.opt.value_loss_coef * value_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.opt.max_grad_norm)
            #print("v_loss {:.3f} p_loss {:.3f}".format(value_loss.item(), policy_loss.item()))
            self.optimizer.step()

