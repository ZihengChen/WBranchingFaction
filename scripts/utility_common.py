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
    plotoutdir = '../../plot/{}/'.format(selection)
    if nbjetcut == '>=1':
        plotoutdir = '../../plot/{}/combined/'.format(selection)
    if nbjetcut == '==1':
        plotoutdir = '../../plot/{}/binned_nBJets/1b/'.format(selection)
    if nbjetcut == '>1':
        plotoutdir = '../../plot/{}/binned_nBJets/2b/'.format(selection)
    return plotoutdir

def GetSelectionCut(slt, nbjetcut, shiftEnergyScale=None, shiftJet=None):
    zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
    lmveto = " & (dilepton_mass>12) "
    ssveto = " & (lepton1_q != lepton2_q) "
    osveto = " & (lepton1_q == lepton2_q) "

    if shiftEnergyScale is None:
        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + ssveto + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + ssveto + zveto,
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + ssveto,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + ssveto,
                "mu4j"  : " (lepton1_pt > 30) ",
                "e4j"   : " (lepton1_pt > 30) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15) " + lmveto +  ssveto, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30) " + lmveto +  ssveto, 
 
                }
    elif shiftEnergyScale == 'e':
        # shift e threshold up by 0.5% to 30.15
        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + ssveto + zveto,
                "ee"    : " (lepton1_pt > 30*1.005) & (lepton2_pt > 15*1.005) " + lmveto + ssveto + zveto,
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + ssveto,
                "etau"  : " (lepton1_pt > 30*1.005) & (lepton2_pt > 20) " + lmveto + ssveto,
                "mu4j"  : " (lepton1_pt > 30) ",
                "e4j"   : " (lepton1_pt > 30*1.005) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt*1.005>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15*1.005)" + lmveto +  ssveto, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt*1.005<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30*1.005)" + lmveto +  ssveto, 

                }
    elif shiftEnergyScale == 'mu':
        # shift e threshold up by 0.5% to 30.15
        sltcut = {
                "mumu"  : " (lepton1_pt > 25*1.005) & (lepton2_pt > 10*1.005) " + lmveto + ssveto + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + ssveto + zveto,
                "mutau" : " (lepton1_pt > 30*1.005) & (lepton2_pt > 20) " + lmveto + ssveto,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + ssveto,
                "mu4j"  : " (lepton1_pt > 30*1.005) ",
                "e4j"   : " (lepton1_pt > 30) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt*1.005)) & (lepton1_pt > 25*1.005) & (lepton2_pt > 15)" + lmveto +  ssveto, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt*1.005)) & (lepton1_pt > 10*1.005) & (lepton2_pt > 30)" + lmveto +  ssveto, 

                }   
    elif shiftEnergyScale == 'tau':
        # shift e threshold up by 0.5% to 30.15
        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + ssveto + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + ssveto + zveto,
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20.2) " + lmveto + ssveto,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20.2) " + lmveto + ssveto,
                "mu4j"  : " (lepton1_pt > 30) ",
                "e4j"   : " (lepton1_pt > 30) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15)" + lmveto +  ssveto, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30)" + lmveto +  ssveto, 

                }


    nJetsName  = "nJets"
    nBJetsName = "nBJets"

    if shiftJet in ["JESUp","JESDown","JERUp","JERDown"]:
        nJetsName  = nJetsName  + shiftJet
        nBJetsName = nBJetsName + shiftJet
    
    if shiftJet in ["BTagUp","BTagDown","MistagUp","MistagDown"]:
        nBJetsName = nBJetsName + shiftJet

    nbveto = " & ({}{})".format(nBJetsName,nbjetcut)
    
    if "4j" in slt:
        njveto = " & ({}>=4)".format(nJetsName)
    else:
        njveto = " & ({}>=2)".format(nJetsName)

    cut = sltcut[slt] + njveto + nbveto
    
    return cut



    # cuts = GetSelectionCut(selection) + "& (nBJets{})".format(nbjetcut)