import torch
import torch.nn as nn

class Discrimiator(nn.Module):
    def __init__(self):
        super().__init__()#细节 这没有冒号
        self.conv1 = nn.Conv2d(1,10,3,padding=1) #输入单通道图片(1,32,32) 1batch_size  (1,1,32,32)
        self.conv2 = nn.Conv2d(10,20,3,padding=1)

        self.fc1 = nn.Linear(20*28*28,500) #细节 这参数输入元素数，不用加batch batch输入的时候加
        self.fc2 = nn.Linear(500,1)
        self.relu = nn.LeakyReLU()

        self.sigmod = nn.Sigmoid()



    def forward(self,x):
        in_size = x.size(0)
        model = nn.Sequential(self.conv1,self.relu,self.max_pool,self.conv2,self.relu)
        x = model(x)
        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmod(x)
        return x