import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.attention import Attention
from model.embedding import GloveEmbedding
from data_preparation import get_stopword_ids

class CosineCoherence(nn.Module):
    def __init__(self, args, device):
        super(CosineCoherence, self).__init__()
        self.seed = args.seed
        self.cos = CosineSimilarity(dim=-1)
        self.emb = GloveEmbedding(args)
        self.device = device

    def forward(self, x_dialogues, x_acts, lengths):
        x_lengths = lengths[0]
        x = self.emb(x_dialogues)
        # x = x.mean(-2) #TODO: use lengths to get the mean, due to padding we'd otherwise get wrong values
        x = torch.sum(x, dim=-2)
        x = torch.div(x, x_lengths.view(x_lengths.size(0), x_lengths.size(1), 1).type(torch.FloatTensor))

        y = torch.narrow(x, dim=1, start=1, length=x.size(1)-1)
        x = torch.narrow(x, dim=1, start=0, length=x.size(1)-1)
        scores = self.cos(x,y).mean(-1)
        return scores, None

    def __str__(self):
        return "cosine"


class MTL_Model3(nn.Module):
    def __init__(self, args, device, collect_da_predictions=True):
        super(MTL_Model3, self).__init__()
        self.input_size = args.embedding_dim
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_layers
        self.num_dialogacts = args.num_classes
        self.device = device
        self.emb = GloveEmbedding(args)

        self.bilstm_u = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_u.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        self.bilstm_d = nn.LSTM(2*self.hidden_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_d.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.attn_u = Attention(2*self.hidden_size)
        self.attn_d = Attention(2*self.hidden_size)

        self.ff_u = nn.Linear(2*self.hidden_size, self.num_dialogacts)
        self.ff_d = nn.Linear(2*self.hidden_size, 1)
        nn.init.normal_(self.ff_d.weight, mean=0, std=1)
        nn.init.normal_(self.ff_u.weight, mean=0, std=1)

        self.dropout_u = nn.Dropout(args.dropout_prob)

        self.collect_da_predictions = collect_da_predictions
        self.da_predictions = []

        #add weights to the loss function to account for the distribution of dialog acts in daily dialog
        #nll_class_weights = torch.tensor([0.0, 2.1861911569232313, 3.4904300472491396, 6.120629125122877, 10.787031308006435]).to(device)
        nll_class_weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]).to(device)
        self.nll = nn.NLLLoss(weight=nll_class_weights, reduction='none')

    def forward(self, x_dialogues, x_acts, lengths):
        s_lengths = lengths[0]
        d_lengths = lengths[1]

        x = self.emb(x_dialogues)
        old_size = (x.size(0), x.size(1), x.size(2), x.size(3))
        ten_sents = x.view(old_size[0]*old_size[1], old_size[2], old_size[3]) 
        ten_acts = x_acts.view(old_size[0]*old_size[1]) 

        loss_da = torch.zeros(ten_acts.size(0)).to(self.device)
        h0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)
        ten_sents = pack_padded_sequence(ten_sents, s_lengths.view(s_lengths.size(0)*s_lengths.size(1)), batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm_u(ten_sents, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        H = self.attn_u(out)

        # view_size1 = int(H.size(0)/old_size[1])
        H1 = H.view(old_size[0], old_size[1], H.size(1))
        H_u = self.dropout_u(H1)
        m = self.ff_u(H_u)
        pda = F.log_softmax(m, dim=2)

        _, da_pred = torch.max(pda.view(old_size[0]*old_size[1], pda.size(2)).data, 1)

        loss_da = self.nll(pda.view(old_size[0] * old_size[1], pda.size(2)), ten_acts)
        loss2 = torch.sum(loss_da.view(old_size[0], old_size[1]), dim=1)

        # H = H.unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)
        H1 = pack_padded_sequence(H1, d_lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm_d(H1, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(1)
        return (s_coh, (da_pred, loss2))

    def __str__(self):
        return "model-3"

class MTL_Model4(nn.Module):
    def __init__(self, args, device, collect_da_predictions=True):
        super(MTL_Model4, self).__init__()
        self.input_size = args.embedding_dim
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_layers
        self.num_dialogacts = args.num_classes
        self.device = device
        self.emb = GloveEmbedding(args)

        self.bilstm_u = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_u.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        self.bilstm_d = nn.LSTM(2*self.hidden_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_d.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.attn_u = Attention(2*self.hidden_size)
        self.attn_d = Attention(2*self.hidden_size)

        self.ff_u = nn.Linear(2*self.hidden_size, self.num_dialogacts)
        self.ff_d = nn.Linear(2*self.hidden_size, 1)
        nn.init.normal_(self.ff_d.weight, mean=0, std=1)
        nn.init.normal_(self.ff_u.weight, mean=0, std=1)

        self.collect_da_predictions = collect_da_predictions
        self.da_predictions = []

        self.nll = nn.NLLLoss(reduction='none')

    def forward(self, x_dialogues, x_acts, lengths):
        s_lengths = lengths[0]
        d_lengths = lengths[1]

        x = self.emb(x_dialogues)
        old_size = (x.size(0), x.size(1), x.size(2), x.size(3))
        ten_sents = x.view(old_size[0]*old_size[1], old_size[2], old_size[3]) 
        ten_acts = x_acts.view(old_size[0]*old_size[1]) 

        loss_da = torch.zeros(ten_acts.size(0)).to(self.device)
        h0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)
        ten_sents = pack_padded_sequence(ten_sents, s_lengths.view(s_lengths.size(0)*s_lengths.size(1)), batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm_u(ten_sents, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        H = self.attn_u(out)

        H1 = H.view(old_size[0], old_size[1], H.size(1))

        h0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)
        H1 = pack_padded_sequence(H1, d_lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm_d(H1, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)

        m = self.ff_u(out)
        pda = F.log_softmax(m, dim=2)

        _, da_pred = torch.max(pda.view(old_size[0]*old_size[1], pda.size(2)).data, 1)

        loss_da = self.nll(pda.view(old_size[0] * old_size[1], pda.size(2)), ten_acts)
        loss2 = torch.sum(loss_da.view(old_size[0], old_size[1]), dim=1)

        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(1)
        return (s_coh, (da_pred, loss2))

    def __str__(self):
        return "model-4"

