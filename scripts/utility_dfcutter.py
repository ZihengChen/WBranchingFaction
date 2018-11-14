import utility_common as common
import pandas as pd
from pylab import *

import glob
import os, sys


class DFCutter:
    def __init__(self,selection, nbjet, name,  njet=None):
        '''
        initialize a DFCutter with selection and name
        For Example, cutter = DFCutter("mumu","mctt")
        '''


        self.selection = selection
        self.name      = name
        self.baseDir   = common.getBaseDirectory() 

        folderOfSelection = self.selection

        if self.selection == "emu2":
            folderOfSelection = 'emu'
        if self.selection == "mutau_fakes":
            folderOfSelection = 'mutau'
        if self.selection == "etau_fakes":
            folderOfSelection = 'etau'
        if self.selection == 'mumuc':
            folderOfSelection = 'mumu'
        if self.selection == 'eec':
            folderOfSelection = 'ee'  
        
        self.pickleDirectry = self.baseDir + "data/pickles/{}/".format(folderOfSelection)

        # default jet requirments
        self.nb = " & (nBJets{})".format(nbjet)
        if njet is None:
            if '4j' in self.selection:
                self.nj = " & (nJets >= 4)"
            else:
                self.nj = " & (nJets >= 2)"
        else:
            self.nj = " & (nJets {})".format(njet)


    def getDataFrame(self,variation=""):

        ###############################################
        # MARK -- read pickles given selection and name
        ###############################################
        # for tt
        if self.name == "mctt":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_inclusive.pkl")
        elif self.name == "mctt_2l2nu":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_2l2nu.pkl")
        elif self.name == "mctt_semilepton":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_semilepton.pkl")
        # for dy
        elif self.name == "mcdy":
            z = glob.glob( self.pickleDirectry + "mcz/*.pkl")
            w = glob.glob( self.pickleDirectry + "mcw/*.pkl")
            pickles = z + w
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles],ignore_index=True,sort=False)
        # for Z,W,VV
        else:
            pickles = glob.glob( self.pickleDirectry + "{}/*.pkl".format(self.name) )
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles],ignore_index=True,sort=False)

        ###############################################
        # MARK -- variate the dataframe for MC
        ###############################################
        if not "data" in self.name:
            dataFrame = self._variateDataFrame(dataFrame,variation)

        ###############################################
        # MARK -- cut the dataframe
        ###############################################
        dataFrame.query(self._cut(), inplace = True)


        ###############################################
        # MARK -- post processing for data and MC
        ###############################################
        if ("data" in self.name) :
            # for data
            # drop if data of emu,mue
            if (self.selection in ["emu","emu2",'emu_tau']):
                dataFrame = dataFrame.drop_duplicates(subset=['runNumber', 'evtNumber'])
        else:
            # for MC
            # 0.95 is the default normalization in BLT, change it for 0.92 for Vtight working points  
            if (self.selection in ["mutau","etau"]):
                dataFrame.eventWeight = dataFrame.eventWeight*(0.92/0.95)
                #dataFrame = self._modifyTauIDCorrection(dataFrame) 


        # reindex the dataframe
        dataFrame.reset_index(drop=True, inplace=True)
        return dataFrame

    def _cut(self):
        oppoSign= " & (lepton1_q != lepton2_q) "
        sameSign= " & (lepton1_q == lepton2_q) "
        zveto   = " & (dilepton_mass<80 | dilepton_mass>102) "
        zmass   = " & (dilepton_mass>80 & dilepton_mass<102) "
        wmass   = " & (dijet_m<95 & dijet_m>65) "
        topmass = " & (trijet_mass1<200 & trijet_mass1>140) "
        lmveto  = " & (dilepton_mass>12) "

        sltcut  = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + oppoSign + zveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + oppoSign + zveto,
                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15) " + lmveto + oppoSign, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30) " + lmveto + oppoSign, 
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + oppoSign,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + oppoSign,
                "mu4j"  : " (lepton1_pt > 30) " ,
                "e4j"   : " (lepton1_pt > 30) " ,
                
                "mutau_fakes" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + sameSign,
                "etau_fakes"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + sameSign,
                "mu4j_fakes"  : " (lepton1_pt > 30) ",
                "e4j_fakes"   : " (lepton1_pt > 30) ",


                "emu_tau"   : " ((triggerLepton == 1 | triggerLepton == 3) & (lepton1_pt > 25) & (lepton2_pt > 15))  | ((triggerLepton == 2 | triggerLepton == 3 ) & (lepton1_pt > 10) & (lepton2_pt > 30)) " + lmveto + oppoSign, 

                "mumuc"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + oppoSign + zmass,
                "eec"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + oppoSign + zmass,
                }

        totalcut = sltcut[self.selection] 
        totalcut = totalcut + self.nj + self.nb
            
        return totalcut


    def _variateDataFrame(self, df, variation):

        # variate e,m,t energy correction
        if variation == 'EPtDown':
            if self.selection in ['ee','etau','e4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection in ['ee','emu','emu2','emu_tau']:
                df.lepton2_pt = df.lepton2_pt * 0.995
        
        if variation == 'MuPtDown':
            if self.selection in ['mumu','emu','emu2','mutau','mu4j','emu_tau']:
                df.lepton1_pt = df.lepton1_pt * 0.998

            if self.selection in ['mumu']:
                df.lepton2_pt = df.lepton2_pt * 0.998
            
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
                pickName = self.pickleDirectry + "mctt/ntuple_ttbar_inclusive_{}.pkl".format(variation)
                df = pd.read_pickle(pickName)
        
        return df

    # def _modifyTauIDCorrection(self, df):

    #     df.eventWeight = df.eventWeight/0.95

    #     if ("mctt" in self.name) or ("mct" in self.name):
    #         slt = (df.genCategory==12)
    #         df[slt].eventWeight = df[slt].eventWeight*0.92

    #         slt = (df.genCategory==15)
    #         df[slt].eventWeight = df[slt].eventWeight*0.92
    #     return df


