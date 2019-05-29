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

    def forward(self, batch):
        X = [s[0] for s in batch]
        X_left = [l for (l,r) in X]
        X_right = [r for (l,r) in X]
        print(X_right)

        # to ids
        X_left = [[[self.vocab.stoi[w] for w in sent] for sent in dialog] for dialog in X_left]
        X_right = [[[self.vocab.stoi[w] for w in sent] for sent in dialog] for dialog in X_right]

        # to tensors
        X_left = torch.tensor(X_left, dtype=torch.long)
        X_right = torch.tensor(X_right, dtype=torch.long)

        # to embedding
        X_left = self.embed(X_left)
        X_right = self.embed(X_right)

        return (X_left, X_right)

