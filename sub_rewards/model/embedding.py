import random
import os
import math
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torchtext as tt
from data_preparation import load_vocab

class GloveEmbedding(nn.Module):
    def __init__(self, args):
        super(GloveEmbedding, self).__init__()
        self.args = args
        self.embed_dim = 300
        self.vocab = load_vocab(args)
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim)
        self.embed.weight.data.copy_(self.vocab.vectors)

    def forward(self, x):
        return self.embed(x)

