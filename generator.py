import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10,10*28*28) #细节 这参数输入元素数，不用加batch batch输入的时候加

        self.conv1 = nn.Conv2d(10,5,3,padding=1) #输入（batch_dize,1,28,28），卷积核尺寸（batch_size,3,3,3）
        self.conv2 = nn.Conv2d(5,1,3,padding=1)

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self,x):
        in_size = x.size(0)

        model = nn.Sequential(self.conv1,self.relu,self.conv2,self.tanh)

        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(in_size,10,28,28)
        x = model(x)

        return x