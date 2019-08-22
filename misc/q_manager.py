import torch
import time
import numpy as np
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QManager(object):
    """
    single-machine implementation
    """

    def __init__(self, args, q_trace, q_batch):
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.traces_p = []
        self.traces_masks = []
        self.traces_logits = []
        self.traces_action_onehot = []

        self.lock = mp.Lock()

        self.q_trace = q_trace
        self.q_batch = q_batch
        self.args = args

    def listening(self):
        while True:
            trace = self.q_trace.get(block=True)
            # in
            self.traces_s.append(trace[0])
            self.traces_a.append(trace[1])
            self.traces_r.append(trace[2])
            self.traces_p.append(trace[3])
            self.traces_masks.append(trace[4])
            self.traces_logits.append(trace[5])
            self.traces_action_onehot.append(trace[6])

            # produce_batch
            if len(self.traces_s) > self.args.batch_size:
                self.produce_batch()

    def produce_batch(self):
        batch_size = self.args.batch_size
        res_s, res_a, res_r, res_p, res_masks, res_logits, res_action_onehot = self.traces_s[:batch_size], self.traces_a[:batch_size], \
                self.traces_r[:batch_size], self.traces_p[:batch_size], self.traces_masks[:batch_size], self.traces_logits[:batch_size], self.traces_action_onehot[:batch_size]

        del self.traces_s[:batch_size]
        del self.traces_a[:batch_size]
        del self.traces_r[:batch_size]
        del self.traces_p[:batch_size]
        del self.traces_masks[:batch_size]
        del self.traces_logits[:batch_size]
        del self.traces_action_onehot[:batch_size]


        # stack batch and put
        self.q_batch.put((torch.stack(res_s, dim=0).to(device), torch.stack(res_a, dim=0).to(device),
            torch.stack(res_r, dim=0).to(device), torch.stack(res_p, dim=0).to(device), torch.stack(res_masks, dim=0).to(device), torch.stack(res_logits, dim=0).to(device), torch.stack(res_action_onehot, dim=0).to(device)))
