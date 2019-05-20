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

    def forward(self, x_sents, x_acts, len_dialog):
        """ general Ranker implementation norm: if y_sents is empty, return the original
            score, else return which one is 'better' """
        x = x_sents.mean(-2).view(int(x_sents.size(0)/len_dialog), len_dialog, x_sents.size(-1))
        scores = []
        for i in range(x.size(0)):
            cosines = []
            for j in range(x.size(1)-1):
                cosines.append(self.cos(x[i][j], x[i][j+1]).unsqueeze(0))
            scores.append(torch.cat(cosines, 0).mean(0).unsqueeze(0))

        return torch.cat(scores, 0), None


    def compare(self, x_sents, x_acts, y_sents, y_acts):
        coh1 = self.forward(x_sents, x_acts)
        coh2 = self.forward(y_sents, y_acts)
        if coh1 < coh2:
            return 1.0
        else:
            return 0.0
