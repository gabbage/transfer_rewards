import random
import torch.nn as nn

class RandomCoherenceRanker(nn.Module):
    def __init__(self, seed):
        super(RandomCoherenceRanker, self).__init__()
        self.seed = seed
        random.seed(seed)

    def forward(self, x_sents, x_acts):
        return float(random.randint(0,1))

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        return float(random.randint(0,1))

