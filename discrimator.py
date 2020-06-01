import torch
import torch.nn as nn

class Discrimiator(nn.Module):
    def __init__(self):
        super().__init__()#细节 这没有冒号


        self.fc1 = nn.Linear(28*28,10) #细节 这参数输入元素数，不用加batch batch输入的时候加
        self.fc2 = nn.Linear(10,1)
        self.relu = nn.LeakyReLU(0.2)

        self.avg_pool = nn.AvgPool2d(2,2)

        self.sigmod = nn.Sigmoid()




    def forward(self,x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmod(x)

        return x