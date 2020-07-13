import glob
import os, sys, errno
from pathlib import Path



import pandas as pd
from pylab import *
from scipy.stats import beta


def makeDirectory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def getBaseDirectory():
    home = str(Path.home())
    baseDir = home + "/Documents/Analysis/wbranch/"
    return baseDir

def getFakeSF(obj):
    sf = 0
    if obj == "e":
        sf = 1.0 #0.12
    elif obj == 'mu':
        sf = 1.0 #0.09
    elif obj == 'tau':
        sf = 1.0
    elif obj == 'etau':
        sf = 1.0 #1.157
    elif obj == 'mutau':
        sf = 1.0 #1.059

    return sf

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

def getEfficiency(k,n,alpha=0.317):
    isZero = np.logical_or(k==0, n==0)

    
    center = k/n
    lower  = beta.ppf(  alpha/2,k+1,n-k+1)
    higher = beta.ppf(1-alpha/2,k+1,n-k+1)

    err = abs(lower-higher)/2
    var = err**2

    center[isZero] = 0
    var[isZero] = 0
    
    var[center>=1] = 0

    return center,var

def getRatio(k,n,kvar,nvar):
    f = k/n
    isZero = np.logical_or(k==0, n==0)
    f[isZero] = 0

    fvar = f**2 * (kvar/k**2 + nvar/n**2)
    return f,fvar

def featureList():
    ls=['dijet_eta', 'dijet_mass', 'dijet_phi', 'dijet_pt', 'dijet_pt_over_m',
        'dilepton_eta', 'dilepton_mass', 'dilepton_phi', 'dilepton_pt','dilepton_pt_over_m',
        'jet1_energy', 'jet1_eta', 'jet1_phi', 'jet1_pt', 'jet1_tag',
        'jet2_energy', 'jet2_eta', 'jet2_phi', 'jet2_pt', 'jet2_tag',
        'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r', 
        'lepton1_energy', 'lepton1_eta','lepton1_reliso', 'lepton1_phi','lepton1_pt',
        'lepton2_energy', 'lepton2_eta','lepton2_reliso', 'lepton2_phi','lepton2_pt',
        'lepton_delta_eta', 'lepton_delta_phi','lepton_delta_r'
        ]
    return ls

def channelLsit():
    ls = [  r'$\mu e$', r'$\mu \mu$',r'$\mu \tau$',r'$\mu h$',
            r'$e e$',   r'$e\mu$',   r'$e\tau$',   r'$e h$' ]

    return ls


def symlink( src, tar):
    if os.path.exists(tar):
        os.remove(tar)
    os.symlink(src, tar)

def get_cross_section(name):
    crossections = {
        'ww'              : 12178,
        'wz_2l2q'         : 5595,
        'wz_3lnu'         : 4430,
        'zz_2l2nu'        : 564,
        'zz_2l2q'         : 3220,
        'zz_4l'           : 1210,
        # Z
        'zjets_m-10to50_amcatnlo'  : 18610000,
        'zjets_m-50_amcatnlo'      :  5765400,
        'z0jets_m-50_amcatnlo':4757000,
        'z1jets_m-50_amcatnlo':884400,
        'z2jets_m-50_amcatnlo':338900,
        # W
        'w1jets'          :  9493000,
        'w2jets'          :  3120000,
        'w3jets'          :  942300,
        'w4jets'          :  524100,
        # top
        't_tw'            :  35850,
        'tbar_tw'         :  35850,
        'ttbar_inclusive' :  832000,
        't_t'             :  136020,
        'tbar_t'          :   80950,
        # gjets
        'gjets_ht40to100': 17410000,
        'gjets_ht100to200': 5363000,
        'gjets_ht200to400': 1178000,
        'gjets_ht400to600':  131800,
        'gjets_ht600toinf':  44270,
        
    }
    xs = crossections.get(name, 0)
    return xs
    