import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.attention import Attention
from torch.nn.modules.distance import CosineSimilarity

class GloveEmbedding(nn.Module):
    def __init__(self):
        super(GloveEmbedding, self).__init__()

    def forward(self, x_left, x_right):
        pass
