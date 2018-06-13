import glob
import pandas as pd
import os, sys
#import ray.dataframe as pd
from pylab import *

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')


def LoadDataframe(pikledir):
    path = pikledir
    pickle_list = glob.glob(path + "/*.pkl")

    df = pd.DataFrame()
    temp_list = []
    for temp_file in pickle_list:
        temp_df = pd.read_pickle(temp_file)
        temp_list.append(temp_df)
    
    df = pd.concat(temp_list,ignore_index=True)
    return df

def GetPlotDir(selection, nbjetcut):
    plotoutdir = '/home/zchen/Documents/Analysis/workplace/plot/{}/'.format(selection)
    if nbjetcut == '>=1':
        plotoutdir = '/home/zchen/Documents/Analysis/workplace/plot/{}/combined/'.format(selection)
    if nbjetcut == '==1':
        plotoutdir = '/home/zchen/Documents/Analysis/workplace/plot/{}/binned_nBJets/1b/'.format(selection)
    if nbjetcut == '>1':
        plotoutdir = '/home/zchen/Documents/Analysis/workplace/plot/{}/binned_nBJets/2b/'.format(selection)
    return plotoutdir

def GetSelectionCut(slt):
    zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
    lmveto = " & (dilepton_mass>12) "
    ssveto = " & (lepton1_q != lepton2_q) "
    sltcut = {
            "mumu"  : " (lepton1_pt > 25) & (lepton1_reliso < 0.15) & (lepton2_pt > 10) & (lepton1_reliso < 0.15) " + lmveto + ssveto + zveto,
            "mutau" : " (lepton1_pt > 30) & (lepton1_reliso < 0.15) " + lmveto +  ssveto,
            "mu4j"  : " (lepton1_pt > 30) & (lepton1_reliso < 0.15) ",
            "emu"   : " (lepton1_pt > 25) " + lmveto +  ssveto, 
            "emu2"  : " (lepton2_pt > 30) & (lepton1_pt < 24)" + lmveto +  ssveto, 
            "ee"    : " (lepton1_pt > 30) " + lmveto + ssveto + zveto,
            "etau"  : " (lepton1_pt > 30) " + lmveto + ssveto,
            "e4j"   : "(lepton1_pt > 30) " 
            }
    return sltcut[slt]

    # cuts = GetSelectionCut(selection) + "& (nBJets{})".format(nbjetcut)