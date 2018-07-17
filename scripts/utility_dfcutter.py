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

        if self.selection == "emu2":
            self.pickleDirectry = common.dataDirectory() + "pickles/emu/{}/".format( self.name )
        else:
            self.pickleDirectry = common.dataDirectory() + "pickles/{}/{}/".format(self.selection, self.name )

    def getDataFrame(self,variation=""):
        '''
        para:   variation==""
        return: dataframe, given selection, nbjet, name
        '''

        # MARK -- read pickles given selection and name
        # for tt read dedicated pickle
        if self.name == "mctt":
            dataFrame = pd.read_pickle(self.pickleDirectry + "ntuple_ttbar_inclusive.pkl")
        # for not tt, read all pickles in a directory
        else:
            pickles = glob.glob( self.pickleDirectry + "/*.pkl")
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True)
        
        # MARK -- variate the dataframe for MC
        if not "data2016" in self.name:
            dataFrame = self._variateDataFrame(dataFrame,variation)

        # MARK -- cut the dataframe
        dataFrame = dataFrame.query(self._cut(variation))

        # MARK -- post processing
        # drop if data of emu,mue
        if (self.selection in ["emu","emu2"]) and ("data2016" in self.name):
            dataFrame = dataFrame.drop_duplicates(subset=['runNumber', 'evtNumber'])
        # reindex the dataframe
        dataFrame = dataFrame.reset_index(drop=True)
        return dataFrame

    def _cut(self,variation):
        zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
        lmveto = " & (dilepton_mass>12) "

        leptonSign = " & (lepton1_q != lepton2_q) "
        if variation == 'ss':
            leptonSign = " & (lepton1_q == lepton2_q) "

        
        nbveto = " & (nBJets{})".format(self.nbjet)

        njveto = " & (nJets >= 2)"
        if '4j' in self.selection:
            njveto = " & (nJets >= 4)"

        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + leptonSign + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + leptonSign + zveto,
                "mutau" : " (lepton1_pt > 25) & (lepton2_pt > 20) " + lmveto + leptonSign,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + leptonSign,
                "mu4j"  : " (lepton1_pt > 30) ",
                "e4j"   : " (lepton1_pt > 30) ",

                "mu4j_fakes"  : " (lepton1_pt > 30) ",
                "e4j_fakes"   : " (lepton1_pt > 30) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15) " + lmveto +  leptonSign, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30) " + lmveto +  leptonSign, 
                }

        return sltcut[self.selection] + njveto + nbveto

    def _variateDataFrame(self, df, variation):
        # variate e,m,t energy correction
        if variation == 'EPtDown':
            if self.selection in ['ee','etau','e4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection == ['ee','emu']:
                df.lepton2_pt = df.lepton2_pt * 0.995
        
        if variation == 'MuPtDown':
            if self.selection in ['mumu','emu','mutau','mu4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection == ['mumu']:
                df.lepton2_pt = df.lepton2_pt * 0.995
            
        if variation == 'TauPtDown':
            if self.selection in ['mutau','etau']:
                df.lepton2_pt = df.lepton2_pt * 0.99

        # variate Jet Energy corrections
        if variation in ["JESUp","JESDown","JERUp","JERDown"]:
            df.nJets  = df["nJets" +variation]
            df.nBJets = df["nBJets"+variation]
        
        # variate bTagging 
        if variation in ["BTagUp","BTagDown","MistagUp","MistagDown"]:
            df.nBJets = df["nBJets"+variation]

        # variate tt theoretical LHE weights
        if (self.name== "mctt"):
            if (variation in ["RenormUp","RenormDown","FactorUp","FactorDown","PDFUp","PDFDown"]):

                variableNames = {
                    "RenormUp"  : "qcd_weight_up_nom",
                    "RenormDown": "qcd_weight_down_nom",
                    "FactorUp"  : "qcd_weight_nom_up",
                    "FactorDown": "qcd_weight_nom_down",
                    "PDFUp"     : "pdf_weight_up",
                    "PDFDown"   : "pdf_weight_down"
                    }
                variableName = variableNames[variation]
                df.eventWeight = df.eventWeight * df[variableName]
            
            # for tt theoretical variation
            if variation in [ 'FSRUp','FSRDown','ISRUp','ISRDown','UEUp','UEDown','MEPSUp','MEPSDown']:
                pickName = self.pickleDirectry + "ntuple_ttbar_inclusive_{}.pkl".format(variation)
                df = pd.read_pickle(pickName)
        
        return df



