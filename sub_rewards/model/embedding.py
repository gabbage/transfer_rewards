import random
import os
import math
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torchtext as tt

class GloveEmbedding(nn.Module):
    def __init__(self, args):
        super(GloveEmbedding, self).__init__()
        self.args = args
        self.embed_dim = 300
        self.vocab = self._load_vocab()
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim)
        self.embed.weight.data.copy_(self.vocab.vectors)

    def _load_vocab(self):
        f = open(os.path.join(self.args.datadir, "itos.txt"), "r")
        cnt = Counter()
        for i, word in enumerate(f):
            cnt[word[:-1].lower()] = i

        return tt.vocab.Vocab(cnt)

    def forward(self, x):
        # to ids
        x = [[[self.vocab.stoi[w] for w in sent] for sent in dialog] for dialog in x]

        # to tensors
        x = torch.tensor(x, dtype=torch.long)

        # to embedding
        return self.embed(x)

