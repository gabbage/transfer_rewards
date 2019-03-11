import torch


class PenalizedTanh(torch.nn.Module):
    def forward(self, x):
        return torch.max(torch.tanh(x), 0.25 * torch.tanh(x))
