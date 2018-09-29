import utility_common as common
import pandas as pd
from pylab import *
from torch_dnn import *

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

        folderOfSelection = self.selection
        if self.selection == "emu2":
            folderOfSelection = 'emu'
        
        if self.selection == "mutau_fakes":
            folderOfSelection = 'mutau'
            
        if self.selection == "etau_fakes":
            folderOfSelection = 'etau'
        
        if self.selection == "eep":
            folderOfSelection = 'ee'

        if self.selection == "mumup":
            folderOfSelection = 'mumu'


        # if self.selection == "mumu0_fakes":
        #     folderOfSelection = 'mumu0'
        # if self.selection == "mutau0_fakes":
        #     folderOfSelection = 'mutau0'
        # if self.selection == "ee0_fakes":
        #     folderOfSelection = 'ee0'
        # if self.selection == "etau0_fakes":
        #     folderOfSelection = 'etau0'

        self.pickleDirectry = self.baseDir + "data/pickles/{}/".format(folderOfSelection)
        #print(self.pickleDirectry)


    def getDataFrame(self,variation="", querySoftmax=None):
        '''
        para:   variation==""
        return: dataframe, given selection, nbjet, name
        '''

        # MARK -- read pickles given selection and name
        # for tt read dedicated pickle
        if self.name == "mctt":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_inclusive.pkl")
        elif self.name == "mctt_2l2nu":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_2l2nu.pkl")
        elif self.name == "mctt_semilepton":
            dataFrame = pd.read_pickle(self.pickleDirectry + "mctt/ntuple_ttbar_semilepton.pkl")
        
        # for dy
        elif self.name == "mcdy":
            highmass = glob.glob( self.pickleDirectry + "mcz/*.pkl")
            highmass = [i for i in highmass if ('m-50' in i)]# and ('raw' in i)]
            
            lowmass = glob.glob( self.pickleDirectry + "mcz/*.pkl")
            lowmass = [i for i in lowmass if ('m-10to50' in i) ]

            pickles = lowmass+highmass + glob.glob( self.pickleDirectry + "mcw/*.pkl")

            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True, sort=False)


        elif self.name == "mcz":

            highmass = glob.glob( self.pickleDirectry + "mcz/*.pkl")
            highmass = [i for i in highmass if ('m-50' in i)]#and ('raw' in i)]

            lowmass = glob.glob( self.pickleDirectry + "mcz/*.pkl")
            lowmass = [i for i in lowmass if ('m-10to50' in i) ]

            pickles = lowmass+highmass

            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True, sort=False)


        # for not tt or DY, read all pickles in a directory
        else:
            pickles = glob.glob( self.pickleDirectry + "{}/*.pkl".format(self.name) )
            dataFrame = pd.concat([ pd.read_pickle(pickle) for pickle in pickles], ignore_index=True, sort=False)
        
        # MARK -- variate the dataframe for MC
        if not "data2016" in self.name:
            dataFrame = self._variateDataFrame(dataFrame,variation)

        # # MARK -- MVA grade mumu and ee
        # if self.selection in ["mumup","eep"]:
        #     dataFrame = DNNGrader(self.selection,self.nbjet).gradeDF(dataFrame)

        # MARK -- cut the dataframe
        dataFrame.query(self._cut(), inplace = True)

        # MARK -- post processing for MC
        if ("data2016" in self.name) :
            # drop if data of emu,mue
            if (self.selection in ["emu","emu2"]):
                dataFrame = dataFrame.drop_duplicates(subset=['runNumber', 'evtNumber'])
        else:
            # 0.95 is the default normalization in BLT, change it for 0.92 for Vtight working points  
            if (self.selection in ["mutau","etau"]):
                dataFrame.eventWeight = dataFrame.eventWeight*(0.92/0.95)
                #dataFrame = self._modifyTauIDCorrection(dataFrame) 


        # reindex the dataframe
        dataFrame.reset_index(drop=True, inplace=True)
        return dataFrame

    def _cut(self):
        zveto  = " & (dilepton_mass<80 | dilepton_mass>102) "
        zmass  = " & (dilepton_mass>85 & dilepton_mass<97) "
        wmass  = " & (dijet_m<95 & dijet_m>65) "
        topmass = " & (trijet_mass1<200 & trijet_mass1>140) "
        lmveto = " & (dilepton_mass>12) "

        leptonSign = " & (lepton1_q != lepton2_q) "
        sameSign = " & (lepton1_q == lepton2_q) "

        nbveto = " & (nBJets{})".format(self.nbjet)

        njveto = " & (nJets >= 2)"
        if '4j' in self.selection:
            njveto = " & (nJets >= 4)"

        # prime = ''
        # if self.selection == "mumup":
        #     if self.nbjet == '==1':
        #         prime = '& (softmax>0.25)'
        #     else:
        #         prime = '& (softmax>0.0002)'
        #     #prime = '& (lepton2_pt < 25)'

        # if self.selection == "eep":
        #     if self.nbjet == '==1':
        #         prime = '& (softmax>0.0005)'
        #     else:
        #         prime = '& (softmax>0.0001)'
        #     #prime = '& (lepton2_pt < 30)'


        sltcut = {
                "mumu"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + leptonSign + zveto + njveto + nbveto,
                "ee"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + leptonSign + zveto + njveto + nbveto,
                "emu"   : " ((triggerLepton == 1) | (triggerLepton == 3 & lepton1_pt>lepton2_pt)) & (lepton1_pt > 25) & (lepton2_pt > 15) " + lmveto +  leptonSign + njveto + nbveto, 
                "emu2"  : " ((triggerLepton == 2) | (triggerLepton == 3 & lepton1_pt<lepton2_pt)) & (lepton1_pt > 10) & (lepton2_pt > 30) " + lmveto +  leptonSign + njveto + nbveto, 
  
                "mutau" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + leptonSign + njveto + nbveto,
                "etau"  : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + leptonSign + njveto + nbveto,
                "mutau_fakes": " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + sameSign + njveto + nbveto,
                "etau_fakes" : " (lepton1_pt > 30) & (lepton2_pt > 20) " + lmveto + sameSign + njveto + nbveto,

                "mu4j"  : " (lepton1_pt > 30) " + njveto + nbveto,
                "e4j"   : " (lepton1_pt > 30) " + njveto + nbveto,
                "mu4j_fakes"  : " (lepton1_pt > 30) " + njveto + nbveto,
                "e4j_fakes"   : " (lepton1_pt > 30) " + njveto + nbveto,

                # "mumup"  : " (lepton1_pt > 25) & (lepton2_pt > 10) " + lmveto + leptonSign + zveto + prime + njveto + nbveto,
                # "eep"    : " (lepton1_pt > 30) & (lepton2_pt > 15) " + lmveto + leptonSign + zveto + prime + njveto + nbveto,

                "mumu0"  : " (lepton1_pt > 25) & (lepton2_pt > 10) & (nJets==0)" + lmveto + leptonSign + zmass ,
                "ee0"    : " (lepton1_pt > 30) & (lepton2_pt > 15) & (nJets==0)" + lmveto + leptonSign + zmass ,
                "mutau0" : " (lepton1_pt > 30) & (lepton2_pt > 20) & (nJets==0)" + lmveto + leptonSign + zmass ,
                "etau0"  : " (lepton1_pt > 30) & (lepton2_pt > 20) & (nJets==0)" + lmveto + leptonSign + zmass ,
                "mumu0_fakes"  : " (lepton1_pt > 25) & (lepton2_pt > 10) & (nJets==0)" + lmveto + sameSign + zmass ,
                "ee0_fakes"    : " (lepton1_pt > 30) & (lepton2_pt > 15) & (nJets==0)" + lmveto + sameSign + zmass ,
                "mutau0_fakes" : " (lepton1_pt > 30) & (lepton2_pt > 20) & (nJets==0)" + lmveto + sameSign + zmass ,
                "etau0_fakes"  : " (lepton1_pt > 30) & (lepton2_pt > 20) & (nJets==0)" + lmveto + sameSign + zmass ,
                }
        
        totalcut = sltcut[self.selection] 



        # threshold = ''
        # if self.selection in ["mutau","etau"]:

        #     if 'pta' in variation:
        #         threshold = " & (lepton1_pt > {}) ".format( variation[-2:] )
        #     if 'ptb' in variation:
        #         threshold = " & (lepton2_pt > {}) ".format( variation[-2:] )
            
        return totalcut #+ threshold


    def _variateDataFrame(self, df, variation):
        # # variate the sign of lepton2 to same sign for tau fakes
        # if variation == 'ss':
        #     df.lepton2_q = - df.lepton2_q

        # variate e,m,t energy correction
        if variation == 'EPtDown':
            if self.selection in ['ee','etau','e4j']:
                df.lepton1_pt = df.lepton1_pt * 0.995

            if self.selection in ['ee','emu','emu2']:
                df.lepton2_pt = df.lepton2_pt * 0.995
        
        if variation == 'MuPtDown':
            if self.selection in ['mumu','emu','emu2','mutau','mu4j']:
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

    def _modifyTauIDCorrection(self, df):

        df.eventWeight = df.eventWeight/0.95

        if ("mctt" in self.name) or ("mct" in self.name):
            slt = (df.genCategory==12)
            df[slt].eventWeight = df[slt].eventWeight*0.92

            slt = (df.genCategory==15)
            df[slt].eventWeight = df[slt].eventWeight*0.92


        return df


