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
from sentence_transformers import SentenceTransformer

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
        self.vocab = list(load_vocab(args).stoi.keys())
        self.elmo = Elmo(elmo_options_file, elmo_weight_file, 1, dropout=0, requires_grad=False, vocab_to_cache=self.vocab).to(device)
        self.device = device
        #otimization TODO: vocab_to_cache argument

    def forward(self, x):
        return self.elmo(x.contiguous())['elmo_representations'][0]


class BertEmbedding(nn.Module):
    def __init__(self, args, device):
        super(BertEmbedding, self).__init__()
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    def forward(self, batch):
        # batch = List[List[str]]
        batch_encodings = []
        for dialog in batch:
            dialog_encodings = []
            enc = torch.tensor(self.model.encode(dialog), dtype=torch.float)
            batch_encodings.append(enc)

        batch = torch.stack(batch_encodings)
        return batch

