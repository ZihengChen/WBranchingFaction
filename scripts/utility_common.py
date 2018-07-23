import glob
import os, sys

import pandas as pd
from pylab import *

def makeDirectory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def dataDirectory(isLocal=True):
    if isLocal:
        # on my local macbook
        dataDir = "/Users/zihengchen/Documents/Analysis/workplace/data/"
    else: 
        # on bahumut
        dataDir = "/mnt/data/zchen/Analysis/"
    return dataDir

def muonFakeSF():
    return 0.05

def electronFakeSF():
    return 0.05

def matrixToArray(mtx):
    indrow = np.array([0,1,0,2,3,2,2,3,4,0,0,0,1,1,1,0,1,2,3,4,5])
    indcol = np.array([0,1,1,2,3,3,4,4,4,2,3,4,2,3,4,5,5,5,5,5,5])
    return mtx[indrow,indcol]
    
def WWBranchNames():
    branches = (r'$ee$',r'$\mu\mu$',r'$e\mu$',
                r'$\tau_e\tau_e$',r'$\tau_\mu\tau_\mu$',r'$\tau_e\tau_\mu$',
                r'$\tau_e\tau_h$',r'$\tau_\mu\tau_h$',r'$\tau_h\tau_h$',
                r'$e\tau_e$',r'$e\tau_\mu$',r'$e\tau_h$',
                r'$\mu\tau_e$',r'$\mu\tau_\mu$',r'$\mu\tau_h$',
                r'$eh$',r'$\mu h$',r'$\tau_eh$',r'$\tau_\mu h$',r'$\tau_hh$',r'$hh$')
    return branches