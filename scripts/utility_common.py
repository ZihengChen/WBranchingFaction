import glob
import os, sys
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

    baseDir = home + "/Documents/Analysis/workplace/"

    return baseDir

def getFakeSF(obj):
    sf = 0
    if obj == "e":
        sf = 0.12
    elif obj == 'mu':
        sf = 0.09
    elif obj == 'tau':
        sf = 1.0
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

    center = k/n
    lower  = beta.ppf(  alpha/2,k+1,n-k+1)
    higher = beta.ppf(1-alpha/2,k+1,n-k+1)

    err = abs(lower-higher)/2
    var = err**2

    return center,var

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



    



###################################
# Plotting
###################################
def showCovar(covar,sameCNorm=False):
    lablesName = ['stat',r"$b^\tau_\mu$",r"$b^\tau_e$",
                r"$\sigma_{VV}$",r"$\sigma_{Z}$",r"$\sigma_{W}$",
                r"QCD in $\mu4j$",r"QCD in $e4j$",r"QCD in $l\tau$","L",r"$\sigma_{tt}$",r"$\sigma_{tW}$",
                r"$\epsilon_e$",r"$\epsilon_\mu$",r"$\epsilon_\tau$",r'$j \to \tau$ MisID',
                "e energy",r"$\mu$ energy",r"$\tau$ energy",
                "JES","JER","bTag","Mistag"]
    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']
    
    n0,n1,n2,n3 = 1E-8, 5E-8, 1E-6, 8E-6
    if sameCNorm:
        n0,n1,n2,n3 = 1E-6,1E-6,1E-6,1E-6

    normList = [n2,n0,n0,n0,
                n1,n1,n2,n2,
                n2,n1,n0,n0,
                n2,n2,n3,n3,
                n1,n1,n2,n3,
                n2,n3,n1,n3]


    NCOL = 6
    N = covar.shape[0]
    
    NROW = int(N/NCOL)+1
    
    for i in range(NROW):
        for j in range(NCOL):
            index = i*NCOL + j
            if index<N:
                matrix = covar[index]

                norm = normList[index]
                plt.subplot(NROW,NCOL,index+1)
                
                plt.imshow(matrix,cmap='RdBu_r',vmin=-norm,vmax=norm)

                plt.xticks([2.5,5.5,8.5],['','',''])
                plt.yticks([2.5,5.5,8.5],['','',''])
                plt.grid('True',lw=1,linestyle='--')
                plt.title(lablesName[index]+ ' [{:1.0E}]'.format(norm),fontsize=12)


    norm = 8E-6
    plt.subplot(NROW,NCOL,NROW*NCOL)
    plt.imshow(np.sum(covar,axis=0),cmap='RdBu_r',vmin=-norm,vmax=norm,)
    # cbar = plt.colorbar( ticks=[-norm,  0,  norm],shrink=0.8)
    # cbar.ax.set_yticklabels(['-{:1.0E}'.format(norm), '0', '{:1.0E}'.format(norm) ],fontsize=8)
    
    plt.xticks([])
    plt.xticks([2.5,5.5,8.5],['','',''])
    plt.yticks([2.5,5.5,8.5],['','',''])
    plt.grid('True',lw=1,linestyle='--')
    plt.title('Total'+ ' [{:1.0E}]'.format(norm),fontsize=12)



def showSingleCovar(covar, norm=1e-6, titleName=''):

    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']

    matrix = covar

    plt.imshow(matrix,cmap='RdBu_r',vmin=-norm,vmax=norm)
    plt.xticks(ticks_pos,ticks_name)
    plt.yticks(ticks_pos,ticks_name)
    plt.title(titleName,fontsize=14)
    plt.grid('False')

    plt.axvline(2.5,c='k',lw=1,linestyle='--')
    plt.axvline(5.5,c='k',lw=1,linestyle='--')
    plt.axvline(8.5,c='k',lw=1,linestyle='--')
    plt.axhline(2.5,c='k',lw=1,linestyle='--')
    plt.axhline(5.5,c='k',lw=1,linestyle='--')
    plt.axhline(8.5,c='k',lw=1,linestyle='--')

    #plt.colorbar()


def showParameterCov(corr):
    ticks = [r'$\bar{\beta}_e$',r'$\bar{\beta}_\mu$',r'$\bar{\beta}_\tau$']
    plt.figure(figsize=(6,4),facecolor='w')
    plt.imshow(corr,cmap='PRGn_r',vmax=1,vmin=-1)
    plt.xticks([0,1,2],ticks, fontsize=14)
    plt.yticks([0,1,2],ticks, fontsize=14)
    for i in range(3):
        for j in range(3):
            v = corr[i,j]
            if abs(v)>0.5:
                fontc = 'w'
            else:
                fontc = 'k'
            plt.text(i-0.2,j+0.1,'{:.2f}'.format(v), color=fontc, fontsize=14)
    plt.colorbar( ticks=[-1, 0, 1],shrink=1)