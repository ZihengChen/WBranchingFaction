import utility_common as common
import pandas as pd
from pylab import *
from utility_dnn import *

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

        self.baseDir = common.getBaseDirectory() 

        if self.selection == "emu2":
            if 'mctt' in self.name:
                self.pickleDirectry = self.baseDir + "data/pickles/emu/mctt/"
            else:
                self.pickleDirectry = self.baseDir + "data/pickles/emu/{}/".format( self.name )
        else:
            if 'mctt' in self.name:
                self.pickleDirectry = self.baseDir + "data/pickles/{}/mctt/".format(self.selection)
            else:
                self.pickleDirectry = self.baseDir + "data/pickles/{}/{}/".format(self.selection, self.name )

    def getDataFrame(self,variation="", querySoftmax=None):
        '''
        para:   variation==""
        return: dataframe, given selection, nbjet, name
        '''

        # MARK -- read pickles given selection and name
        # for tt read dedicated pickle
        if self.name == "mctt":
            dataFrame = pd.read_pickle(self.pickleDirectry + "ntuple_ttbar_inclusive.pkl")
        elif self.name == "mctt_2l2nu":
            dataFrame = pd.read_pickle(self.pickleDirectry + "ntuple_ttbar_2l2nu.pkl")
        elif self.name == "mctt_semilepton":
            dataFrame = pd.read_pickle(self.pickleDirectry + "ntuple_ttbar_semilepton.pkl")
        # for not tt, read all pickles in a directory
        else:
            pickles = glob.glob( self.pickleDirectry + "/*.pkl")
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True)
        
        # MARK -- variate the dataframe for MC
        if not "data2016" in self.name or variation == 'ss':
            dataFrame = self._variateDataFrame(dataFrame,variation)

        # MARK -- cut the dataframe
        dataFrame = dataFrame.query(self._cut(variation))

        # MARK -- post processing
        # drop if data of emu,mue
        if (self.selection in ["emu","emu2"]) and ("data2016" in self.name):
            dataFrame = dataFrame.drop_duplicates(subset=['runNumber', 'evtNumber'])

        # query sortmax if needed
        if not querySoftmax is None:
            dataFrame = DNNGrader(self.selection,self.nbjet).gradeDF(dataFrame,querySoftmax=0.05)

        # reindex the dataframe
        dataFrame.reset_index(drop=True, inplace=True)

        return dataFrame

    def _cut(self, variation):
        zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
        lmveto = " & (dilepton_mass>12) "

        leptonSign = " & (lepton1_q != lepton2_q) "

        nbveto = " & (nBJets{})".format(self.nbjet)

        njveto = " & (nJets >= 2)"
        if '4j' in self.selection:
            njveto = " & (nJets >= 4)"

        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + leptonSign + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + leptonSign + zveto,
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + leptonSign,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + leptonSign,
                "mu4j"  : " (lepton1_pt > 30) ",
                "e4j"   : " (lepton1_pt > 30) ",

                "mu4j_fakes"  : " (lepton1_pt > 30) ",
                "e4j_fakes"   : " (lepton1_pt > 30) ",

                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15) " + lmveto +  leptonSign, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30) " + lmveto +  leptonSign, 
                }
        
        totalcut = sltcut[self.selection] + njveto + nbveto

        threshold = ''
        if self.selection in ["mutau","etau"]:

            if 'pta' in variation:
                threshold = " & (lepton1_pt > {}) ".format( variation[-2:] )
            if 'ptb' in variation:
                threshold = " & (lepton2_pt > {}) ".format( variation[-2:] )
            
        return totalcut + threshold


    def _variateDataFrame(self, df, variation):
        # variate the sign of lepton2 to same sign for tau fakes
        if variation == 'ss':
            df.lepton2_q = - df.lepton2_q

        # variate e,m,t energy correction
        if variation == 'EPtDown':
            if self.selection in ['ee','etau','e4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection in ['ee','emu','emu2']:
                df.lepton2_pt = df.lepton2_pt * 0.995
        
        if variation == 'MuPtDown':
            if self.selection in ['mumu','emu','emu2','mutau','mu4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection in ['mumu']:
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

        # variate tt theoretical weight and LHE weights
        if (self.name== "mctt"):
            # for lhe weight variation
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



