import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util

from collections import defaultdict


class OutputAggregator(object):
    def __init__(self, agg_method):


        self.agg_method = agg_method
        if self.agg_method == 'max':
            self._reduce = np.max
        elif self.agg_method == 'mean':
            self._reduce = np.mean
        


    def aggregate(self, keys, probs, data_loader, device):
        key2probs = defaultdict(list)
        for key, output in zip(keys, probs):
            key2probs[key].append(output)
        return {key: self._reduce(probs) for key, probs in key2probs.items()}
