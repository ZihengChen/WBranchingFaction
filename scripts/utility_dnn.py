import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data import Dataset, DataLoader



    
class MyDataset(Dataset):
    def __init__(self, datatable,n, transform=None):
        self.data  = np.reshape(datatable[:,0:-1],(-1,n)).astype('float32')
        self.label = datatable[:,-1].astype('int')
        self.transform = transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = {'feature': self.data[idx,:], 'label': self.label[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Net(nn.Module):
    def __init__(self,n,m1,m2,m3,c):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n, m1)
        self.fc2 = nn.Linear(m1, m2)
        self.fc3 = nn.Linear(m2, m3)
        self.fc4 = nn.Linear(m3, c)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Net_dropout(nn.Module):
    def __init__(self,n,m1,m2,m3,c):
        super(Net_dropout, self).__init__()
        self.fc1 = nn.Linear(n, m1)
        self.fc2 = nn.Linear(m1, m2)
        self.fc3 = nn.Linear(m2, m3)
        self.fc4 = nn.Linear(m3, c)
        #self.drop= nn.Dropout(p=0.25)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
 
def CutScoreEff(mc,a,b,step):
    bkg   = mc[-2]
    sig   = mc[-1] - bkg
    bkg   = np.array([np.sum(bkg[i:]) for i in range(bkg.size)])
    sig   = np.array([np.sum(sig[i:]) for i in range(sig.size)])
    bkgeff= bkg/bkg[1]
    sigeff= sig/sig[1]
    cut   = np.arange(a,b,step)

    effx = np.array([sigeff[2:4].mean(),sigeff[5],sigeff[7:9].mean(),sigeff[10]])
    effy = np.array([bkgeff[2:4].mean(),bkgeff[5],bkgeff[7:9].mean(),bkgeff[10]])

    fig, axes = plt.subplots(1, 2, figsize=(8,3))
    ax = axes[0]
    for x in [0.05,0.1,0.15,0.2]:
        ax.axvline(x,color="k",linestyle='--')
    ax.plot(cut, sig, lw=2, label='signal')
    ax.plot(cut, bkg, lw=2, label='background')
    ax.set_xlabel("Cut on Score")
    ax.legend()
    ax.grid()

    ax = axes[1]
    ax.plot(sigeff, bkgeff,c="r",lw=2)
    ax.plot(effx,effy,'ko')
    ax.set_xlabel("Sig_eff")
    ax.set_ylabel("bkg_eff")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()