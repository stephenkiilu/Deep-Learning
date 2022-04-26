
from importlib.metadata import metadata
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from data import dataloader
from utils import transform, plot
from config import args
from models import resnet, CNN
import argparse
import train

metadata=args.metadata
path=args.path

data= dataloader.CustomDataSet(metadata, path, transform= transform)

batch_size= args.bs
train_size= args.train_size
train_size= int(train_size*len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


## model
parser=argparse.ArgumentParser()
parser.add_argument('-m','--model_name',help='This is the name of model',required=True,type=str)
parser.add_argument('-n','--num_epochs',help='This is the num of epochs',required=True,type=int)

mains_arg=vars(parser.parse_args())



num_epochs=mains_arg['num_epochs']
if mains_arg['model_name'].lower()=='resnet':
    model=resnet()

elif mains_arg['model_name'].lower()=='cnn':
    model=CNN()


criterion= nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_trained, percent, val_loss, val_acc, train_loss, train_acc= train.train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)
plot(train_loss,val_loss)