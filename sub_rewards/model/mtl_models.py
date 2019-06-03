import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity

from model.attention import Attention
from model.embedding import GloveEmbedding

class CosineCoherence(nn.Module):
    def __init__(self, args, device):
        super(CosineCoherence, self).__init__()
        self.seed = args.seed
        self.cos = CosineSimilarity(dim=-1)
        self.emb = GloveEmbedding(args)
        self.device = device

    def forward(self, x_dialogues, x_acts):
        x = self.emb(x_dialogues)
        x = x.mean(-2)
        y = torch.narrow(x, dim=1, start=1, length=x.size(1)-1)
        y = torch.cat([y, torch.ones(y.size(0), 1, y.size(2)).to(self.device)], dim=1)
        scores = self.cos(x,y).mean(dim=-1)
        return scores, None

        # for i in range(x.size(0)):
            # cosines = []
            # for j in range(x.size(1)-1):
                # cosines.append(self.cos(x[i][j], x[i][j+1]).unsqueeze(0))
            # scores.append(torch.cat(cosines, 0).mean(0).unsqueeze(0))

        # return torch.cat(scores, 0), None

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

        self.bilstm_u = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True, bias=False)
        for param in self.bilstm_u.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        self.bilstm_d = nn.LSTM(2*self.hidden_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True, bias=False)
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

    def forward(self, x_dialogues, x_acts):
        x = self.emb(x_dialogues)
        old_size = (x.size(0), x.size(1), x.size(2), x.size(3))
        ten_sents = x.view(old_size[0]*old_size[1], old_size[2], old_size[3]) 
        ten_acts = x_acts.view(old_size[0]*old_size[1]) 

        loss_da = torch.zeros(ten_acts.size(0)).to(self.device)
        h0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, ten_sents.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_u(ten_sents, (h0, c0))
        H = self.attn_u(out)

        # view_size1 = int(H.size(0)/old_size[1])
        H1 = H.view(old_size[0], old_size[1], H.size(1))
        m = self.ff_u(H1)
        pda = F.log_softmax(m, dim=2)

        _, da_pred = torch.max(pda.view(old_size[0]*old_size[1], pda.size(2)).data, 1)

        loss_da = self.nll(pda.view(old_size[0] * old_size[1], pda.size(2)), ten_acts)
        loss2 = torch.sum(loss_da.view(old_size[0], old_size[1]), dim=1)

        # H = H.unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_d(H1, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(1)
        return (s_coh, (da_pred, loss2))

    def __str__(self):
        return "model-3"


class MTL_Model4(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dialogacts, device):
        super(MTL_Model4, self).__init__()
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

        # H = H.unsqueeze(0)
        h0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)# 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, H1.size(0), self.hidden_size).to(self.device)
        out, _ = self.bilstm_d(H1, (h0, c0))
        hd = self.attn_d(out)
        s_coh = self.ff_d(hd).squeeze(1)

        H2 = out.contiguous().view(out.size(0)* out.size(1), out.size(2))

        m = self.ff_u(H2)
        pda = F.log_softmax(m, dim=1)
        loss_da = self.nll(pda, ten_acts)
        loss2 = torch.sum(loss_da.view(view_size1, len_dialog), dim=1)

        return (s_coh, loss2)

    def __str__(self):
        return "model-4"

