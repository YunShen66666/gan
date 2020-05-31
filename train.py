import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import transforms,datasets

import generator
import discrimator

BATCH_SIZE = 512
Learning_rate = 0.0001
EPOCH = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform_train = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
# ])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',download=False,train=True,transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True
)

G = generator.Generator()
D = discrimator.Discrimiator()

generator1 = G.to(DEVICE)
discrimator1 = D.to(DEVICE)

optim_G = optimizer.Adam(generator1.parameters(),lr=Learning_rate)
optim_D = optimizer.Adam(discrimator1.parameters(),lr=Learning_rate)

citizerion = nn.BCELoss()

def train():
    for epoch in range(EPOCH):
        discrimator1.train()
        generator1.train()
        for i,(img,target) in enumerate(train_loader):
            in_size = img.size(0)
            fake_img = torch.rand(in_size,1,28,28).to(DEVICE)
            img = img.to(DEVICE)
            fake_img = generator1(fake_img)

            loss_d = citizerion(discrimator1(img),torch.ones(in_size,1).to(DEVICE))+citizerion(discrimator1(fake_img),torch.zeros(in_size,1).to(DEVICE))
            optim_D.zero_grad()
            loss_d.backward()
            optim_D.step()

            loss_g = citizerion(discrimator1(generator1(torch.rand(in_size,1,28,28).to(DEVICE))),torch.ones(in_size,1).to(DEVICE))
            optim_G.zero_grad()
            loss_g.backward()
            optim_G.step()

            if i%50==0:
                print("epich:{}/{} , batch:{}/{} , loss_d={} , loss_g={}".format(
                    epoch,EPOCH,i,len(train_loader),loss_d.item()*1000,loss_g.item()*1000))
        if epoch%7==0:
            state = {'generator':generator1.state_dict(),'discrimator':discrimator1.state_dict()}
            torch.save(state,"epoch{}".format(epoch))
train()