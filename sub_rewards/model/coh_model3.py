import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention

# dont use bias in attention
class MTL_Model3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dialogacts, device):
        super(MTL_Model3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dialogacts = num_dialogacts
        self.device = device

        self.bilstm_u = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.bilstm_d = nn.LSTM(2*hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

        self.attn_u = Attention(2*hidden_size)
        self.attn_d = Attention(2*hidden_size)

        self.ff_u = nn.Linear(2*hidden_size, num_dialogacts)
        self.ff_d = nn.Linear(2*hidden_size, 1)
        nn.init.normal_(self.ff_d.weight, mean=0, std=1)
        nn.init.normal_(self.ff_u.weight, mean=0, std=1)

        self.nll = nn.NLLLoss(reduction='sum')

    def forward(self, x_sents, x_acts):
        ten_sents = torch.cat([x.unsqueeze(0) for x in x_sents], 0)
        ten_acts = torch.cat(x_acts, 0)
        loss_da = torch.zeros(ten_acts.size(0)).to(self.device)
        h0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_u(ten_sents, (h0, c0))
        H = self.attn_u(out)
        m = self.ff_u(H)
        pda = F.log_softmax(m, dim=1)
        loss_da = self.nll(pda, ten_acts)

        H = H.unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_d(H, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(0).squeeze(0)
        return (s_coh, loss_da)

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        coh1, _ = self.forward(x_sents, x_acts)
        coh2, _ = self.forward(y_sents, y_acts)
        if coh1.cpu().item() < coh2.cpu().item():
            return 1.0
        else:
            return 0.0


