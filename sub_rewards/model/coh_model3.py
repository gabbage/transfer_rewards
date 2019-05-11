import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention

class MTL_Model3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dialogacts, device):
        super(MTL_Model3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dialogacts = num_dialogacts
        self.device = device

        self.bilstm_u = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True).to(device)
        self.bilstm_d = nn.LSTM(2*hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True).to(device)

        self.attn_u = Attention(2*hidden_size).to(device)
        self.attn_d = Attention(2*hidden_size).to(device)

        self.ff_u = nn.Linear(2*hidden_size, num_dialogacts).to(device)
        self.ff_d = nn.Linear(2*hidden_size, 2).to(device)

        self.nll = nn.NLLLoss().to(device)

    def forward(self, x_sents, x_acts):
        """ general Ranker implementation norm: if y_sents is empty, return the original
            score, else return which one is 'better' """

        H = []
        loss_da = torch.tensor(0.0, requires_grad=True).to(self.device)

        for (sent, da) in zip(x_sents, x_acts):
            sent = sent.unsqueeze(0) # To account for the batch_size (=1)
            h0 = torch.zeros(self.num_layers*2, sent.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
            c0 = torch.zeros(self.num_layers*2, sent.size(0), self.hidden_size).to(self.device)
            out, _ = self.bilstm_u(sent, (h0, c0))
            # out = out.view(sent.size(0), 1, 2, hidden_size) # extract both directions
            hu = self.attn_u(out)
            H.append(hu)
            pda = F.softmax(self.ff_u(hu))
            # da = da.type(torch.float)
            loss_da = loss_da +self.nll(pda, da)

        H_ten = torch.cat([h.unsqueeze(0) for h in H], 0)
        h0 = torch.zeros(self.num_layers*2, H_ten.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H_ten.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_d(H_ten, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd)
        return (s_coh, loss_da)


