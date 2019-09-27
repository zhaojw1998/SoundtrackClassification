import torch
from torch import nn
from config import params
class NeuralNetwork(nn.Module):
    def __init__(self, dropout = params['dropout']):
        super().__init__()
        self.lin1 = nn.Linear(64*96, 4096)
        #self.lin2 = nn.Linear(4096, 2048)
        self.lin3 = nn.Linear(4096, 1024)
        #self.lin4 = nn.Linear(1024, 512)
        #self.lin5 = nn.Linear(512, 256)
        #self.lin6 = nn.Linear(256, 128)
        #self.lin7 = nn.Linear(128, 64)
        self.lin8 = nn.Linear(1024, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.normalize = nn.LayerNorm(64*96)

    def forward(self, x):
        #print(x.shape)
        x= x.view(x.shape[0], -1)
        x= self.normalize(x)
        x = self.relu(self.lin1(x))
        #x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        #x = self.relu(self.lin4(x))
        #x = self.relu(self.lin5(x))
        #x = self.relu(self.lin6(x))
        #x = self.relu(self.lin7(x))
        x = self.dropout(x)
        x = self.lin8(x)
        return x