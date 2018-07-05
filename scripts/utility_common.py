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
        dataDir = "/home/zchen/Documents/Analysis/workplace/data/"
    return dataDir

def fakeRate():
    return 0.05
