import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


#copied from https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html

def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.normal_(out)
    return out

class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score, dim=0).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)

        return condensed_x

# attn = Attention(100)
# x = Variable(torch.randn(16,30,100))
# print(x.size(-2))
# t = torch.ones(size=(16,100), dtype=torch.long)
# # print(attn(x).size() == (16,100))
# nll = nn.NLLLoss(reduction='none')
# # print(nll(x, t).size())

# x = torch.randn(2,10,4)
# t = torch.ones(2,10, dtype=torch.long)
# a = nll(x,t)
# print(torch.sum(a.view(2,5), dim=1))

