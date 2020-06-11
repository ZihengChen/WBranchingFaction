import os, sys, tqdm, datetime

import numpy as np
import pandas as pd
import ROOT as root

import utility_common as common
from utility_bltreaderDict_multilep import *
from multiprocessing import Pool



class BLTReader:

    def __init__(self, inputRootFileName, selection="ee", inputRootType="", outputFolder="pickles_2016"):

        self.baseDir = common.getBaseDirectory() 

        self.inputRootFile = root.TFile(self.baseDir+'data/root/'+inputRootFileName)
        self.inputRootType = inputRootType
        self.outputFolder = outputFolder

        

        self.selection = selection
        self.lumin = 35.864

        self._getCrossection()
        self._getNameList()
    
    #############################
    ## Save number of Gen 
    #############################
        
    def outputNGen(self):
        f = self.inputRootFile

        if self.inputRootType == "mcttsys":
          names = [ 'tt_tauReweight',
                    'tt_fsrUp','tt_fsrDown',
                    'tt_isrUp','tt_isrDown',
                    'tt_ueUp','tt_ueDown',
                    'tt_mepsUp','tt_mepsDown' ]

          nGens = [ self.getNGen('ttbar_inclusive_tauReweight'),
                    self.getNGen('ttbar_inclusive_fsrUp'), self.getNGen('ttbar_inclusive_fsrDown'),
                    self.getNGen('ttbar_inclusive_isrUp'),self.getNGen('ttbar_inclusive_isrDown'),
                    self.getNGen('ttbar_inclusive_ueUp'),  self.getNGen('ttbar_inclusive_ueDown'),
                    self.getNGen('ttbar_inclusive_mepsUp'),self.getNGen('ttbar_inclusive_mepsDown') ]
        else:
          names = [ 't','tt','tt_2l2nu','tt_semilepton']
          nGens = [ self.getNGen('t_tw')+self.getNGen('tbar_tw'), 
                    self.getNGen('ttbar_inclusive'),
                    self.getNGen('ttbar_2l2nu'),
                    self.getNGen('ttbar_semilepton')
                    ] 

        df = pd.DataFrame({'name':names, 'ngen':nGens })
        df.to_json(self.baseDir+'data/'+ self.outputFolder+'/ngen_{}.json'.format(self.inputRootType))
        
    def getNGen(self, name):
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
        # for name in self.datalist:
            self.makePickle(name)
        print(self.selection + ' finished!')

        # pool = Pool(16)
        # pool.map(self.makePickle, self.mclist + self.datalist)

        

    # MARK-1 -- ntuple to pickle
    def makePickle(self,name):
        scaleFactor = self._getScaleFactor(name)
        outputPath = self._getOutputPath(name)

        common.makeDirectory(outputPath, clear=False)
        tree = self.inputRootFile.Get('{}/bltTree_{}'.format(self.selection,name))

        # # correct a typo in root file
        # if name == "ttbar_inclusive_isrUp": 
        #     tree = self.inputRootFile.Get('{}/bltTree_ttbar_inclusive_ifsrUp'.format(self.selection,name))

        if tree.GetEntriesFast() > 0:
            ntuple = self.fillNtuple(tree, name, scaleFactor)
            dataframe = pd.DataFrame(ntuple)

            outfileName = name
            dataframe.to_pickle( outputPath+'ntuple_{}.pkl'.format(outfileName))
            
    # MARK-2 -- tree to ntuple
    def fillNtuple(self, tree, name, scaleFactor):
        n = int(tree.GetEntriesFast())

        # loop over all events
        for i in range(n):
            
            tree.GetEntry(i)
            entry = {}

            # add this event to the ntuple
            entry.update(self._getAllVariables(tree, self.selection, name, scaleFactor))
            n -= 1
            yield entry


    #############################
    ## private helper functions
    #############################
    def _getAllVariables(self, tree, selection, name, scaleFactor):
        if selection in [ 'ee','mumu','emu','mutau','etau','mutau_fakes','etau_fakes',
                          'mu4j','e4j','mu4j_fakes','e4j_fakes',
                          'mumutau','eetau','emutau','mumue','eemu']:
            dictionary = getAllVariables_multileptonSelection(tree, selection, name, scaleFactor)
        return dictionary
        
    def _getScaleFactor(self,name):
        if name in self.datalist:
            scaleFactor = 1
        if name in self.mclist:
            # get crosssection for the name
            xs = self.xsTable[name]
            # get nGenTotal for the name
            histogram = self.inputRootFile.Get('TotalEvents_'+name)

            # correct a typo in root file
            # if name == "ttbar_inclusive_isrUp": 
            #   histogram = self.inputRootFile.Get('TotalEvents_ttbar_inclusive_ifsrUp')              
            print('TotalEvents_'+name)

            nGenTotal = histogram.GetBinContent(1) - 2*histogram.GetBinContent(10)
            # calculate SF to lumin
            scaleFactor = self.lumin * xs/nGenTotal
        return scaleFactor
        
    
    def _getCrossection(self):
        self.xsTable = { 
                    # diboson
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
                    'ttbar_2l2nu'     :  87340,
                    'ttbar_semilepton':  364456,
                    # for systematics
                    'ttbar_inclusive_tauReweight' :  832000,
                    'ttbar_inclusive_fsrUp'    :  832000,
                    'ttbar_inclusive_fsrDown'  :  832000,
                    'ttbar_inclusive_isrUp'    :  832000,
                    'ttbar_inclusive_isrDown'  :  832000,
                    'ttbar_inclusive_ueUp'     :  832000,
                    'ttbar_inclusive_ueDown'   :  832000,
                    'ttbar_inclusive_mepsUp'   :  832000,
                    'ttbar_inclusive_mepsDown' :  832000,
                }

    def _getOutputPath(self,name):

        outputPath  = self.baseDir+'data/'+self.outputFolder+'/'+self.selection+'/'

        if name in self.datalist:
            outputPath += 'data/'
        elif name in self.mcdibosonlist:
            outputPath += 'mcdiboson/'
        elif name in self.mczlist:
            outputPath += 'mcz/'
        elif name in self.mcwlist:
            outputPath += 'mcw/'
        elif name in self.mctlist:
            outputPath += 'mct/'
        elif name in self.mcttlist:
            outputPath += 'mctt/' 
        elif name in self.mcttsyslist:
            outputPath += 'mcttsys/' 
        return outputPath

    def _getNameList(self):
        ## 1. define the datalist
        if self.selection in ['mumu','mutau','mutau_fakes','mu4j','mu4j_fakes','mumutau','mumue']:
            self.datalist = [
                'muon_2016B', 'muon_2016C','muon_2016D','muon_2016E',
                'muon_2016F','muon_2016G','muon_2016H'
                ]

        elif self.selection in ['ee','etau','etau_fakes','e4j','e4j_fakes','eetau','eemu']:
            self.datalist = [
                'electron_2016B', 'electron_2016C','electron_2016D','electron_2016E',
                'electron_2016F','electron_2016G','electron_2016H'
                ]

        elif self.selection in ['emu','emutau']:
            self.datalist = [
                'muon_2016B', 'muon_2016C','muon_2016D','muon_2016E',
                'muon_2016F','muon_2016G','muon_2016H',
                'electron_2016B', 'electron_2016C', 'electron_2016D','electron_2016E',
                'electron_2016F','electron_2016G','electron_2016H'
                ]


        ## 2. define the MC list
        self.mcdibosonlist  = [ 'ww','wz_2l2q','wz_3lnu','zz_2l2nu','zz_2l2q','zz_4l']
        self.mczlist        = [ 'zjets_m-10to50_amcatnlo','zjets_m-50_amcatnlo'] + ['z0jets_m-50_amcatnlo','z1jets_m-50_amcatnlo','z2jets_m-50_amcatnlo']
        self.mcwlist        = [ 'w1jets','w2jets','w3jets','w4jets' ]
        self.mctlist        = [ 't_tw','tbar_tw']
        self.mcttlist       = [ 'ttbar_inclusive','ttbar_2l2nu','ttbar_semilepton']
        self.mcttsyslist    = [ 'ttbar_inclusive_tauReweight',
                                'ttbar_inclusive_fsrUp','ttbar_inclusive_fsrDown',
                                'ttbar_inclusive_isrUp','ttbar_inclusive_isrDown',
                                'ttbar_inclusive_ueUp','ttbar_inclusive_ueDown',
                                'ttbar_inclusive_mepsUp','ttbar_inclusive_mepsDown']

        self.mclist = self.mcdibosonlist+self.mcwlist+self.mczlist+self.mctlist+self.mcttlist
        
        

        
        # overwrite if there is a inputRootType
        if self.inputRootType == "mcttsys":
          self.datalist = []
          self.mclist = self.mcttsyslist
        

