import random
import os
import math
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torchtext as tt
from allennlp.modules.elmo import Elmo, batch_to_ids
from data_preparation import load_vocab

#elmo_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#elmo_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo_options_file = "/ukp-storage-1/buecker/transfer_rewards/sub_rewards/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
elmo_weight_file = "/ukp-storage-1/buecker/transfer_rewards/sub_rewards/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" 
class GloveEmbedding(nn.Module):
    def __init__(self, args):
        super(GloveEmbedding, self).__init__()
        self.args = args
        self.embed_dim = 300
        self.vocab = load_vocab(args)
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim, padding_idx=0)
        self.embed.weight.data.copy_(self.vocab.vectors)

    def forward(self, x):
        return self.embed(x)

class ElmoEmbedding(nn.Module):
    def __init__(self, args, device):
        super(ElmoEmbedding, self).__init__()
        self.elmo = Elmo(elmo_options_file, elmo_weight_file, 2, dropout=0, requires_grad=True).to(device)
        self.device = device
        #otimization TODO: vocab_to_cache argument

    def forward(self, x):
        return self.elmo(x.contiguous())['elmo_representations'][1]

