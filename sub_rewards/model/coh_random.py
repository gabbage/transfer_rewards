import random
import torch
import torch.nn as nn

class RandomCoherenceRanker(nn.Module):
    def __init__(self, seed):
        super(RandomCoherenceRanker, self).__init__()
        self.seed = seed
        random.seed(seed)

    def forward(self, x_sents, x_acts, len_dialog):
        result = [random.uniform(0,1) for _ in range(int(x_acts.size(0)/len_dialog))]
        return torch.tensor(result, dtype=torch.float), None

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        return float(random.randint(0,1))

    def __str__(self):
        return "RandomCoherenceRanker"

