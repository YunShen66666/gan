import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import transforms,datasets

import generator
import discrimator

BATCH_SIZE = 512
Learning_rate = 0.001
EPOCH = 10
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

optim_G = optimizer.SGD(generator1.parameters(),lr=Learning_rate)
optim_D = optimizer.SGD(discrimator1.parameters(),lr=Learning_rate)

citizerion = nn.BCELoss()

def train():
    for epoch in range(EPOCH):
        discrimator1.train()
        generator1.train()
        for i,(img,target) in enumerate(train_loader):
            fake_img = torch.rand(BATCH_SIZE,1,28,28).to(DEVICE)
            img = img.to(DEVICE)
            fake_img = generator1(fake_img)

            loss = citizerion(discrimator1(img),torch.ones(BATCH_SIZE,1))+citizerion(discrimator1(fake_img),torch.zeros(BATCH_SIZE,1))
            loss.backward()
            optim_D.step()
            optim_G.step()
            if i%30==0:
                print("epich:{}/{} , batch:{}/{} , loss={}".format(
                    epoch,EPOCH,i,len(train_loader),loss.item()))
        if epoch%2==0:
            state = {'generator':generator1.state_dict(),'discrimator':discrimator1.state_dict()}
            torch.save(state,"epoch{}".format(epoch))
train()