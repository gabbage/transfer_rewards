import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.attention import Attention

# dont use bias in attention
class MTL_Model3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dialogacts, device, collect_da_predictions=True):
        super(MTL_Model3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dialogacts = num_dialogacts
        self.device = device

        self.bilstm_u = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_u.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        self.bilstm_d = nn.LSTM(2*hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        for param in self.bilstm_d.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.attn_u = Attention(2*hidden_size)
        self.attn_d = Attention(2*hidden_size)

        self.ff_u = nn.Linear(2*hidden_size, num_dialogacts)
        self.ff_d = nn.Linear(2*hidden_size, 1)
        nn.init.normal_(self.ff_d.weight, mean=0, std=1)
        nn.init.normal_(self.ff_u.weight, mean=0, std=1)

        self.collect_da_predictions = collect_da_predictions
        self.da_predictions = []

        self.nll = nn.NLLLoss(reduction='none')

    def forward(self, x_sents, x_acts, len_dialog):
        ten_sents = x_sents # torch.cat([x.unsqueeze(0) for x in x_sents], 0)
        ten_acts = x_acts #.view(int(x_acts.size(0)/len_dialog), len_dialog) #torch.cat(x_acts, 0)

        loss_da = torch.zeros(ten_acts.size(0)).to(self.device)
        h0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_u(ten_sents, (h0, c0))
        H = self.attn_u(out)

        view_size1 = int(H.size(0)/len_dialog)
        H1 = H.view(view_size1, len_dialog, H.size(1))
        m = self.ff_u(H1)
        pda = F.log_softmax(m, dim=2)

        if self.collect_da_predictions:
            _, pred = torch.max(pda.view(view_size1*len_dialog, pda.size(2)).data, 1)
            self.da_predictions = self.da_predictions + list(zip(pred.detach().cpu().numpy().tolist(), ten_acts.detach().cpu().numpy().tolist()))

        loss_da = self.nll(pda.view(view_size1*len_dialog, pda.size(2)), ten_acts)
        loss2 = torch.sum(loss_da.view(view_size1, len_dialog), dim=1)

        # H = H.unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_d(H1, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(1)
        return (s_coh, loss2)

    def compare(self, x_sents, x_acts, y_sents, y_acts):
        coh1, _ = self.forward(x_sents, x_acts)
        coh2, _ = self.forward(y_sents, y_acts)
        if coh1.cpu().item() < coh2.cpu().item():
            return 1.0
        else:
            return 0.0


    def __str__(self):
        return "Model-3"
