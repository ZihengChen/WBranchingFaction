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


import utility_common as common



class TrainingDataLoader():
    def __init__(self,selection,nbjet):
        self.selection = selection
        self.nbjet = nbjet
        
        self.var_list = common.featureList()

        self.nvar = len(self.var_list)

        if nbjet == '==1':
            self.bname = '1b'
        if nbjet == '>1':
            self.bname = '2b'

        
    def loadData(self):          
        MCzz = DFCutter(self.selection, self.nbjet, "mcdiboson").getDataFrame()
        MCdy = DFCutter(self.selection, self.nbjet, "mcdy").getDataFrame()
        MCt  = DFCutter(self.selection, self.nbjet, "mct").getDataFrame()
        MCtt = DFCutter(self.selection, self.nbjet, "mctt").getDataFrame()
        
        MCsg  = pd.concat([MCt,MCtt],ignore_index=True)
        if self.selection == 'mumu':
            MCsg0 = MCsg.query('(genCategory != 14) & (genCategory != 5)')
            MCsg1 = MCsg.query('(genCategory == 14) | (genCategory == 5)')
        if self.selection == 'ee':
            MCsg0 = MCsg.query('(genCategory != 10) & (genCategory != 4)')
            MCsg1 = MCsg.query('(genCategory == 10) | (genCategory == 4)')
        
        
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
        np.save( common.getBaseDirectory()+"data/networks/{}{}_norm.npy".format(self.selection,self.bname), self.norm)
        self.df_train = df_train


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


class DNNGrader():
    def __init__(self, selection, nbjet):
        self.selection = selection
        self.nbjet = nbjet
        
        self.var_list = common.featureList()
        self.nvar = len(self.var_list)

        if nbjet == '==1':
            self.bname = '1b'
        if nbjet == '>1':
            self.bname = '2b'

        if self.selection[-1] == 'p':
            self.selection = self.selection[0:-1]

        self.norm = np.load( common.getBaseDirectory()+"data/networks/{}{}_norm.npy".format(self.selection,self.bname) ).item()
        self.net  = torch.load(common.getBaseDirectory()+"data/networks/{}{}.pt".format(self.selection,self.bname))

    def gradeDF(self, df, querySoftmax=None ):
        
        # remove other column if not in feature list
        drop_list = [ v for v in df.columns if not v in self.var_list ]
        temp = df.drop(drop_list, axis=1)

        # normalize features
        for v in temp.columns:
            if v != 'label':
                mu,sigma = self.norm[v]
                temp[v] = (temp[v]-mu)/sigma
        
        # calculate MVA
        temp['softmax'] = np.ones(len(temp)).astype(int)


        tempset = MyDataset(temp.as_matrix(),self.nvar)
        
        temploader = DataLoader(tempset, batch_size=tempset.__len__(), shuffle=False, num_workers=1)
        tempiter = iter(temploader).next()
        tempoutputs  = self.net(Variable(tempiter["feature"]))
        
        tempsoftmax  = F.softmax(tempoutputs,dim=1).data.numpy()
        tempSignalP  = tempsoftmax[:,1]
        #temppredicts = isSignal.astype(int)
        #torch.max(tempoutputs.data, 1)[1] # return likelihood, predict
        df = df.reset_index(drop=True)
        df["softmax"] = tempSignalP

        if (not querySoftmax is None) :
            df = df.query("softmax>{}".format(querySoftmax))
            df.reset_index(drop=True, inplace=True)
        
        return df


    
    def gradeDFList(self,dfList, querySoftmax=None):
        return [self.gradeDF(idf,querySoftmax) for idf in dfList]




    
