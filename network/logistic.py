import torch
from torch import nn

class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6766, 50)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xb):
        x = self.lin(xb)
        return x