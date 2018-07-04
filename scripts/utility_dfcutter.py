import utility_common as common
import pandas as pd
from pylab import *

import glob
import os, sys


class DFCutter:
    def __init__(self,selection,nbjet,name):
        '''
        initialize a DFCutter with selection, nbjet, name
        e.g. DFCutter("mumu","==1","mctt")
        '''
        self.selection = selection
        self.nbjet = nbjet
        self.name = name
        self.cut = self._cut()

        if self.selection == "emu2":
            self.pickleDirectry = common.dataDirectory() + "pickle/emu/{}/".format( self.name )
        else:
            self.pickleDirectry = common.dataDirectory() + "pickle/{}/{}/".format(self.selection, self.name )

    def getDataFrame(self,variation=""):
        '''
        para:   variation==""
        return: dataframe, given selection, nbjet, name
        '''

        # MARK -- read pickles given selection and name
        # for tt read dedicated pickle
        if self.name == "mctt":
            # for tt theoretical variation
            if variation in [ 'fsrup','fsrdown','isrup','isrdown','up','down','hdampdown','hdampup']:
                pickName = self.pickleDirectry + "ntuple_ttbar_inclusive_{}.pkl".formate(variation)
            # use nominal tt sample
            else:
                pickName = self.pickleDirectry + "ntuple_ttbar_inclusive.pkl"
            dataFrame = pd.read_pickle(pickName)
        # for not tt, read all pickles in a directory
        else:
            pickles = glob.glob( self.pickleDirectry + "/*.pkl")
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True)
        
        # MARK -- variate the dataframe for MC
        if not "data2016" in self.name:
            dataFrame = self._variateDataFrame(dataFrame,variation)

        # MARK -- cut the dataframe
        dataFrame = dataFrame.query(self.cut)

        # MARK -- post processing
        # drop if data of emu,mue
        if (self.selection in ["emu","emu2"]) and ("data2016" in self.name):
            dataFrame = dataFrame.drop_duplicates(subset=['runNumber', 'evtNumber'])
        # reindex the dataframe
        dataFrame = dataFrame.reset_index(drop=True)
        return dataFrame

    def _cut(self):
        zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
        lmveto = " & (dilepton_mass>12) "
        ssveto = " & (lepton1_q != lepton2_q) "
        osveto = " & (lepton1_q == lepton2_q) "
        
        nbveto = " & (nBJets{})".format(self.nbjet)

        njveto = " & (nJets >= 2)"
        if self.selection in ["mu4j","e4j"]:
            njveto = " & (nJets >= 4)"

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

        return sltcut[self.selection] + njveto + nbveto

    def _variateDataFrame(self, df, variation):
        return df



