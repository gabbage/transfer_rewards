import random
import torch
import torch.nn as nn

class RandomCoherenceRanker(nn.Module):
    def __init__(self, seed):
        super(RandomCoherenceRanker, self).__init__()
        self.seed = seed
        random.seed(seed)

    def forward(self, x_sents, x_acts):
        result = [random.uniform(0,1) for _ in range(x_acts.size(0))]
        return torch.tensor(result, dtype=torch.float), None

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        return float(random.randint(0,1))

    def __str__(self):
        return "RandomCoherenceRanker"

