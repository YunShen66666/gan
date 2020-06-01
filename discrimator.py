import torch
import torch.nn as nn

class Discrimiator(nn.Module):
    def __init__(self):
        super().__init__()#细节 这没有冒号
        self.conv1 = nn.Conv2d(1,10,3,padding=1) #输入单通道图片(1,32,32) 1batch_size  (1,1,32,32)
        self.conv2 = nn.Conv2d(10,20,3,padding=1)

        self.fc1 = nn.Linear(20*14*14,50) #细节 这参数输入元素数，不用加batch batch输入的时候加
        self.fc2 = nn.Linear(50,1)
        self.relu = nn.LeakyReLU(0.2)

        self.avg_pool = nn.AvgPool2d(2,2)
        self.sigmod = nn.Sigmoid()
        self.batch_normalization = nn.BatchNorm1d(500)


    def forward(self,x):
        in_size = x.size(0)
        model = nn.Sequential(self.conv1,self.relu,self.max_pool,self.conv2,self.relu)
        x = model(x)
        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = self.batch_normalization(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmod(x)
        return x