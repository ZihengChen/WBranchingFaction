import os, sys, tqdm

import numpy as np
import pandas as pd
import ROOT as root

import utility_common as common
from utility_bltreaderDict import *


class BLTReader:

    def __init__(self, inputRootFileName, selection, includeTTTheory=False):

        self.baseDir = common.getBaseDirectory(isLocal=False) 

        self.inputRootFile = root.TFile(self.baseDir+"data/root/"+inputRootFileName)

        self.selection = selection
        self.lumin = 35.864
        
        self.includeTTTheory = includeTTTheory

        self._getCrossection()
        self._getNameList()
        
    #############################
    ## Save number of Gen 
    #############################
        
    def outputNGen(self):
        names = ["t","tt","tt_2l","tt_semilepton"] 
        nGens = [self.getNGen("t_tw")+self.getNGen("tbar_tw"),
                 self.getNGen("ttbar_inclusive"),
                 self.getNGen("ttbar_2l2nu"),
                 self.getNGen("ttbar_semilepton")
                 ] 
        
        if self.includeTTTheory:
            names = names + [self.mcttTheorylistFileNames[name] for name in self.mcttTheorylist]
            nGens = nGens + [self.getNGen(name) for name in self.mcttTheorylist]

        df = pd.DataFrame({"name":names, "ngen":nGens })
        df.to_pickle(self.baseDir+"data/pickles/ngen.pkl")
        
    def getNGen(self,name):
        histogram = self.inputRootFile.Get('GenCategory_'+name)
        nGen = []
        for i in range(1,22,1):
            nGen.append(histogram.GetBinContent(i))
        nGen = np.array(nGen)
        return nGen    
    
    #############################
    ## Read BLT to pick 
    #############################
    
    def readBLT(self):
        # loop over all names
        for name in self.mclist + self.datalist:
            self.makePickle(name)
        print(self.selection + " finished!")

    # MARK-1 -- ntuple to pickle
    def makePickle(self,name):
        scaleFactor = self._getScaleFactor(name)
        outputPath = self._getOutputPath(name)

        common.makeDirectory(outputPath, clear=False)

        tree = self.inputRootFile.Get('{}/bltTree_{}'.format(self.selection,name))
        print(name)

        if tree.GetEntriesFast() > 0:
            ntuple = self.fillNtuple(tree, name, scaleFactor)
            dataframe = pd.DataFrame(ntuple)

            outfileName = name
            if name in self.mcttTheorylist:
                outfileName = self.mcttTheorylistFileNames[name]

            dataframe.to_pickle( outputPath+'ntuple_{}.pkl'.format(outfileName))
            
    # MARK-2 -- tree to ntuple
    def fillNtuple(self, tree, name, scaleFactor):
        n = int(tree.GetEntriesFast())
        # loop over all events
        for i in range(n):

            tree.GetEntry(i)
            entry = {}

            # do not double count DY1234Jet in inclusive DY samples
            avoidDY1234Jet = (name in ['zjets_m-50','zjets_m-10to50']) & (0 < tree.nPartons < 5)
            if ( avoidDY1234Jet ):
                n -= 1
                continue

            # and this event to the ntuple
            entry.update(self._getAllVariables(tree, self.selection, name, scaleFactor))
            n -= 1
            yield entry


    #############################
    ## private helper functions
    #############################
    def _getAllVariables(self, tree, selection, name, scaleFactor):
        if selection in ["ee","mumu","emu","mutau","etau","mu4j","e4j","mu4j_fakes","e4j_fakes"]:
            dictionary = getAllVariables_multileptonSelection(tree, selection, name, scaleFactor)
        else: 
            #selection in ["ee_e","mumu_e","mumu_mu","ee_mu"]:
            dictionary = getAllVariables_fakeSelection(tree, selection, name, scaleFactor)
        return dictionary
        
    def _getScaleFactor(self,name):
        if name in self.datalist:
            scaleFactor = 1
        if name in self.mclist:
            # get crosssection for the name
            xs = self.xsTable[name]
            # get nGenTotal for the name
            histogram = self.inputRootFile.Get("TotalEvents_"+name)
            nGenTotal = histogram.GetBinContent(1)
            # calculate SF to lumin
            scaleFactor = self.lumin * xs/nGenTotal
        return scaleFactor
        
    
    def _getCrossection(self):
        self.xsTable = { 

                    'ww'              : 12178,
                    'wz_2l2q'         : 5595,
                    'wz_3lnu'         : 4430,
                    'zz_2l2nu'        : 564,
                    'zz_2l2q'         : 3220,
                    'zz_4l'           : 1210,
                
                    'zjets_m-10to50'  : 18610000,
                    'z1jets_m-10to50' : 1.18*730300,
                    'z2jets_m-10to50' : 1.18*387400,
                    'z3jets_m-10to50' : 1.18*95020,
                    'z4jets_m-10to50' : 1.18*36710,

                    'zjets_m-50'      :  5765400,
                    'z1jets_m-50'     :  1.18*1012000,
                    'z2jets_m-50'     :  1.18*334700,
                    'z3jets_m-50'     :  1.18*102300,
                    'z4jets_m-50'     :  1.18*54520,

                    'w1jets'          :  9493000,
                    'w2jets'          :  3120000,
                    'w3jets'          :  942300,
                    'w4jets'          :  524100,
                
                    't_tw'            :  35850,
                    'tbar_tw'         :  35850,
                    'ttbar_inclusive' :  832000,
                    'ttbar_2l2nu'     :  87340,
                    'ttbar_semilepton':  364456,
                    
                    'ttbar_inclusive_fsrdown'   :  832000,
                    'ttbar_inclusive_fsrup'     :  832000,
                    'ttbar_inclusive_isrdown'   :  832000,
                    'ttbar_inclusive_isrup'     :  832000,
                    'ttbar_inclusive_hdampdown' :  832000,
                    'ttbar_inclusive_hdampup'   :  832000,
                    'ttbar_inclusive_down'      :  832000,
                    'ttbar_inclusive_up'        :  832000,
                
                    'TTZToLLNuNu'     : 252.9,
                    'TTZToQQ'         : 529.7,
                    'TTWJetsToLNu'    : 204.3,
                    'TTWJetsToQQ'     : 406.2,
                    'ttHJetTobb'      : 295.0,

                    'qcd_ht100to200'  : 27990000000,
                    'qcd_ht200to300'  : 1712000000,
                    'qcd_ht300to500'  : 347700000,
                    'qcd_ht500to700'  : 32100000,
                    'qcd_ht700to1000' : 6831000,
                    'qcd_ht1000to1500': 1207000,
                    'qcd_ht1500to2000': 119900,
                    'qcd_ht2000'      : 25240,
                }

    def _getOutputPath(self,name):

        outputPath  = self.baseDir+"data/pickles/"+self.selection+"/"

        if name in self.datalist:
            outputPath += "data2016/"
        elif name in self.mcdibosonlist:
            outputPath += "mcdiboson/"
        elif name in self.mcdylist:
            outputPath += "mcdy/"
        elif name in self.mctlist:
            outputPath += "mct/"
        elif name in self.mcttlist:
            outputPath += "mctt/" 
        elif name in self.mcttbosonlist:
            outputPath += "mcttboson/"
        elif name in self.mcttTheorylist:
            outputPath += "mctt/"
        return outputPath

    def _getNameList(self):
        ## 1. define the datalist
        if self.selection in ["mumu","mutau","mu4j","mu4j_fakes","mumu_mu","mumu_e","emu_tau"]:
            self.datalist = [
                'muon_2016B', 'muon_2016C','muon_2016D','muon_2016E',
                'muon_2016F','muon_2016G','muon_2016H'
                ]

        elif self.selection in ["ee","etau","e4j","e4j_fakes","ee_mu","ee_e"]:
            self.datalist = [
                'electron_2016B', 'electron_2016C','electron_2016D','electron_2016E',
                'electron_2016F','electron_2016G','electron_2016H'
                ]

        elif self.selection in ["emu"]:
            self.datalist = [
                'muon_2016B', 'muon_2016C','muon_2016D','muon_2016E',
                'muon_2016F','muon_2016G','muon_2016H',
                'electron_2016B', 'electron_2016C', 'electron_2016D','electron_2016E',
                'electron_2016F','electron_2016G','electron_2016H'
                ]


        ## 2. define the MC list
        self.mcqcdlist = ['qcd_ht100to200','qcd_ht200to300','qcd_ht300to500','qcd_ht500to700',
                          'qcd_ht700to1000','qcd_ht1000to1500','qcd_ht1500to2000','qcd_ht2000']

        self.mcttbosonlist  = [ 'TTZToLLNuNu']

        self.mcdibosonlist  = [ 'ww','wz_2l2q','wz_3lnu','zz_2l2nu','zz_2l2q','zz_4l']

        self.mcdylist       = [ 'zjets_m-10to50','zjets_m-50',
                                'z1jets_m-10to50','z1jets_m-50',
                                'z2jets_m-10to50','z2jets_m-50',
                                'z3jets_m-10to50','z3jets_m-50',
                                'z4jets_m-10to50','z4jets_m-50',
                                'w1jets','w2jets','w3jets','w4jets']

        self.mctlist        = [ 't_tw','tbar_tw']
        
        if self.selection in ["ee_mu","ee_e","mumu_mu","mumu_e","emu_tau"]:
            self.mcttlist       = [ 'ttbar_inclusive']
        else:
            self.mcttlist       = [ 'ttbar_inclusive','ttbar_2l2nu','ttbar_semilepton']
        #self.mcttlist       = [ 'ttbar_semilepton']

        self.mcttTheorylist = [ 'ttbar_inclusive_fsrdown','ttbar_inclusive_fsrup',
                                'ttbar_inclusive_isrdown','ttbar_inclusive_isrup',
                                'ttbar_inclusive_hdampdown','ttbar_inclusive_hdampup',
                                'ttbar_inclusive_down','ttbar_inclusive_up']

        self.mclist = self.mcttbosonlist + self.mcdibosonlist + self.mcdylist + self.mctlist + self.mcttlist 
        #self.mclist = self.mcttlist
        if self.includeTTTheory:
            self.mclist = self.mclist + self.mcttTheorylist

        self.mcttTheorylistFileNames = {
            'ttbar_inclusive_fsrdown' : 'ttbar_inclusive_FSRDown',
            'ttbar_inclusive_fsrup'   : 'ttbar_inclusive_FSRUp',
            'ttbar_inclusive_isrdown' : 'ttbar_inclusive_ISRDown',
            'ttbar_inclusive_isrup'   : 'ttbar_inclusive_ISRUp',
            'ttbar_inclusive_hdampdown':'ttbar_inclusive_MEPSDown',
            'ttbar_inclusive_hdampup' : 'ttbar_inclusive_MEPSUp' ,
            'ttbar_inclusive_down'    : 'ttbar_inclusive_UEDown',
            'ttbar_inclusive_up'      : 'ttbar_inclusive_UEUp'
        }
    






