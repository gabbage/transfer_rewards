import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity

class CosineCoherenceRanker(nn.Module):
    def __init__(self, seed):
        super(CosineCoherenceRanker, self).__init__()
        self.seed = seed
        random.seed(seed)
        self.cos = CosineSimilarity(dim=0)

    def forward(self, x_sents, x_acts):
        """ general Ranker implementation norm: if y_sents is empty, return the original
            score, else return which one is 'better' """
        x_cosines = []
        for i in range(len(x_sents)-1):
            x_cosines.append(self.cos(x_sents[i].mean(0), x_sents[i+1].mean(0)).item())
        return np.array(x_cosines).mean()

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        coh1 = self.forward(x_sents, x_acts)
        coh2 = self.forward(y_sents, y_acts)
        if coh1 < coh2:
            return 1.0
        else:
            return 0.0
