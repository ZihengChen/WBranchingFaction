import matplotlib.pyplot as plt
from utility_dfplotter import *
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


class trainingDataLoader():
    def __init__(self,selection,nbjet):
        self.selection = selection
        self.nbjet = nbjet
        
        self.var_list = ['dijet_eta', 'dijet_mass', 'dijet_phi', 'dijet_pt', 'dijet_pt_over_m',
                         'dilepton_eta', 'dilepton_mass', 'dilepton_phi', 'dilepton_pt','dilepton_pt_over_m',
                         'jet1_energy', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_tag',
                         'jet2_energy', 'jet2_eta', 'jet2_phi', 'jet2_pt', 'jet2_tag',
                         'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r', 
                         'lepton1_energy', 'lepton1_eta','lepton1_reliso', 'lepton1_phi','lepton1_pt',
                         'lepton2_energy', 'lepton2_eta','lepton2_reliso', 'lepton2_phi','lepton2_pt',
                         'lepton_delta_eta', 'lepton_delta_phi','lepton_delta_r'
                        ]
        self.nvar = len(self.var_list)
        
    def loadData(self):          
        MCzz = DFCutter(self.selection, self.nbjet, "mcdiboson").getDataFrame()
        MCdy = DFCutter(self.selection, self.nbjet, "mcdy").getDataFrame()
        MCt  = DFCutter(self.selection, self.nbjet, "mct").getDataFrame()
        MCtt = DFCutter(self.selection, self.nbjet, "mctt").getDataFrame()
        
        MCsg  = pd.concat([MCt,MCtt],ignore_index=True)
        if self.selection == 'mumu':
            MCsg0 = MCsg.query('genCategory != 14')
            MCsg1 = MCsg.query('genCategory == 14')
        if self.selection == 'ee':
            MCsg0 = MCsg.query('genCategory != 10')
            MCsg1 = MCsg.query('genCategory == 10')
        
        
        MClist = [MCzz,MCdy,MCsg0,MCsg1]
        
        self.df_list, self.N_list, self.NRaw_list = [],[],[]
        for i in range(len(MClist)):
            #MClist[i] = MClist[i].reset_index(drop=True)
            
            n    = int(np.sum(MClist[i].eventWeight))
            nRaw = int(np.sum(MClist[i].eventWeight/MClist[i].eventWeightSF))
            df   = MClist[i].sample(int(n),replace=False)
            
            self.N_list.append(n)
            self.NRaw_list.append(nRaw)
            self.df_list.append(df)
        
        
        MCbkg   = pd.concat(self.df_list[0:3],ignore_index=True)
        MCsig   = self.df_list[3]
        
        drop_list = [v for v in MCsig.columns if not v in self.var_list]
        MCbkg.drop( drop_list, axis=1, inplace=True)
        MCsig.drop( drop_list, axis=1, inplace=True)
        
        MCbkg['label'] = 0
        MCsig['label'] = 1
        
        df_train = pd.concat([MCbkg,MCsig],ignore_index=True)
        df_train = df_train.reset_index(drop=True)
        self.norm = {}
        for v in df_train.columns:
            if v != 'label':
                mu = df_train[v].mean()
                sigma = df_train[v].std()
                df_train[v] = (df_train[v] - mu)/sigma
                self.norm[v] = (mu,sigma)
        self.df_train = df_train
        