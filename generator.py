import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,3,padding=1) #输入（batch_dize,1,28,28），卷积核尺寸（batch_size,3,3,3）
        self.conv2 = nn.Conv2d(10,1,3,padding=1)

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        model = nn.Sequential(self.conv1,self.relu,self.conv2,self.tanh)
        x = model(x)
        return x