import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, scores):
        return self.net(scores)
