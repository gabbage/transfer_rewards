import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention

class MTL_Model3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dialogacts):
        super(MTL_Model3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dialogacts = num_dialogacts

        self.bilstm_u = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.bilstm_d = nn.LSTM(2*hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

        self.attn_u = Attention(2*hidden_size)
        self.attn_d = Attention(2*hidden_size)

        self.ff_u = nn.Linear(2*hidden_size, num_dialogacts)
        self.ff_d = nn.Linear(2*hidden_size, 1)

        self.nll = nn.NLLLoss()

    def forward(self, x_sents, x_acts):
        H = []
        loss_da = torch.tensor(0.0)

        for (sent, da) in zip(x_sents, x_acts):
            sent = sent.unsqueeze(0) # To account for the batch_size (=1)
            h0 = torch.zeros(self.num_layers*2, sent.size(0), self.hidden_size)# 2 for bidirection 
            c0 = torch.zeros(self.num_layers*2, sent.size(0), self.hidden_size)
            out, _ = self.bilstm_u(sent, (h0, c0))
            # out = out.view(sent.size(0), 1, 2, hidden_size) # extract both directions
            hu = self.attn_u(out)
            H.append(hu)
            m = self.ff_u(hu)
            pda = F.log_softmax(m, dim=1)
            loss_da = loss_da +self.nll(pda, da)

        H_ten = torch.cat([h for h in H], 0).unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H_ten.size(0), self.hidden_size)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H_ten.size(0), self.hidden_size)
        out, _ = self.bilstm_d(H_ten, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(0).squeeze(0)
        return (s_coh, loss_da)

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        coh1, _ = self.forward(x_sents, x_acts)
        coh2, _ = self.forward(y_sents, y_acts)
        if coh1.item() < coh2.item():
            return 1.0
        else:
            return 0.0


