import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.recurrent_hidden_state_size = recurrent_hidden_state_size

        self.init()

    def init(self):
        """
        Initialise the class and this method is being used from when we test the agent
        so that we've decided to make it available outward
        """
        self.obs = torch.zeros(self.num_steps + 1, 1, *self.obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            self.num_steps + 1, 1, self.recurrent_hidden_state_size)
        self.rewards = torch.zeros(self.num_steps, 1, 1)
        self.value_preds = torch.zeros(self.num_steps + 1, 1, 1)
        self.returns = torch.zeros(self.num_steps + 1, 1, 1)
        self.action_log_probs = torch.zeros(self.num_steps, 1, 1)
        if self.action_space.__class__.__name__ == 'Discrete':
            num_actions = self.action_space.n
            self.action_shape = 1
        else:
            self.action_shape = self.action_space.shape[0]
        self.actions = torch.zeros(self.num_steps, 1, self.action_shape)
        if self.action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.zeros(self.num_steps + 1, 1, 1)
        self.masks[0] = 1.

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.logits = torch.zeros(self.num_steps, 1, num_actions)
        self.action_onehot = torch.zeros(self.num_steps, 1, num_actions)

        self.step = 0

    def to(self, device):
        """
        After init being executed, we need to format the empty matrices, so call this API!
        ```python
        # instantiate the class
        rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
        obs = envs.reset() # reset the env
        rollouts.obs[0].copy_(obs) # set the initial state
        rollouts.to(device) # turns the matrices to GPU/CPU mode accordingly
        ```
        """
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.logits = self.logits.to(device)
        self.action_onehot = self.action_onehot.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, logits=None, action_onehot=None):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if logits is not None:
            self.logits[self.step].copy_(logits)
        if action_onehot is not None:
            self.action_onehot[self.step].copy_(action_onehot)

        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.logits[0].copy_(self.logits[-1])

    def reload(self, device):
        obs = self.obs[-1]
        recurrent_hidden_states = self.recurrent_hidden_states[-1]
        masks = self.masks[-1]
        logits = self.logits[-1]
        action_onehot = self.action_onehot[-1]
        self.init()
        self.to(device)
        self.obs[0].copy_(obs)
        self.recurrent_hidden_states[0].copy_(recurrent_hidden_states)
        self.masks[0].copy_(masks)
        self.logits[0].copy_(logits)
        self.action_onehot[0].copy_(action_onehot)

