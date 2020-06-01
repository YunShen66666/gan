import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(1,10,3,padding=1) #输入（batch_dize,1,28,28），卷积核尺寸（batch_size,3,3,3）
        self.conv2 = nn.Conv2d(10,20,3,padding=1)
        self.conv3 = nn.Conv2d(20,10,3,padding=1)
        self.conv4 = nn.Conv2d(10,1,3,padding=1)
=======

        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,500)
        self.fc3 = nn.Linear(500,28*28)

>>>>>>> faded556aa05478dfef4c76a99cd2661ef543e4a
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        in_size = x.size(0)

        model = nn.Sequential(self.fc1,self.relu,self.fc2,self.relu,self.fc3,self.tanh)
        x = model(x)

        x = x.view(in_size,1,28,28)
        return x