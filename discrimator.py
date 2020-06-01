import torch
import torch.nn as nn

class Discrimiator(nn.Module):
    def __init__(self):
        super().__init__()#细节 这没有冒号

        self.fc1 = nn.Linear(28*28,500) #细节 这参数输入元素数，不用加batch batch输入的时候加
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,10)
        self.fc4 = nn.Linear(10,1)
        self.relu = nn.LeakyReLU()

        self.sigmod = nn.Sigmoid()




    def forward(self,x):
        in_size = x.size(0)
        x = x.view(in_size,-1)
        model = nn.Sequential(self.fc1,self.relu,self.fc2,self.relu,self.fc3,self.relu,self.fc4,self.sigmod)
        x = model(x)
        return x