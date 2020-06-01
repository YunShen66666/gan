import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,500)
        self.fc3 = nn.Linear(500,28*28)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        in_size = x.size(0)

        model = nn.Sequential(self.fc1,self.relu,self.fc2,self.relu,self.fc3,self.tanh)
        x = model(x)

        x = x.view(in_size,1,28,28)
        return x