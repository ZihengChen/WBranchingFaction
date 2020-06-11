import os, sys, tqdm, datetime

import numpy as np
import pandas as pd
import ROOT as root

import utility_common as common
from utility_bltreaderDict_multilep import *
from multiprocessing import Pool



class BLTReader:

    def __init__(self, inputRootFileName, selection="ee", inputRootType="", outputFolder="pickles_2018"):

        self.baseDir = common.getBaseDirectory() 

        self.inputRootFile = root.TFile(self.baseDir+'data/root/'+inputRootFileName)
        self.inputRootType = inputRootType
        self.outputFolder = outputFolder

        self.selection = selection
        self.lumin = 59.688

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
                    self.getNGen('ttbar_inclusive_ifsrUp'),self.getNGen('ttbar_inclusive_isrDown'),
                    self.getNGen('ttbar_inclusive_ueUp'),  self.getNGen('ttbar_inclusive_ueDown'),
                    self.getNGen('ttbar_inclusive_mepsUp'),self.getNGen('ttbar_inclusive_mepsDown') ]
        else:
          names = [ 't','tt_2l2nu','tt_semilepton','tt_hadronic']
          nGens = [ self.getNGen('t_tw')+self.getNGen('tbar_tw'), 
                    self.getNGen('ttbar_2l2nu'),
                    self.getNGen('ttbar_semilepton'),
                    self.getNGen('ttbar_hadronic')
                    ] 

        df = pd.DataFrame({'name':names, 'ngen':nGens })
        df.to_json(self.baseDir+'data/' + self.outputFolder + '/ngen_{}.json'.format(self.inputRootType))
        
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

    # MARK-1 -- ntuple to pickle
    def makePickle(self,name):
        scaleFactor = self._getScaleFactor(name)
        outputPath = self._getOutputPath(name)

        common.makeDirectory(outputPath, clear=False)
        tree = self.inputRootFile.Get('{}/bltTree_{}'.format(self.selection,name))

        # correct a typo in root file
        if name == "ttbar_inclusive_isrUp": 
            tree = self.inputRootFile.Get('{}/bltTree_ttbar_inclusive_ifsrUp'.format(self.selection,name))

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
                          'mu4j','e4j','mu4j_fakes','e4j_fakes','mumutau','eetau','emutau']:
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
            if name == "ttbar_inclusive_isrUp": 
              histogram = self.inputRootFile.Get('TotalEvents_ttbar_inclusive_ifsrUp')              
            print('TotalEvents_'+name)
            nGenTotal = histogram.GetBinContent(1) - 2*histogram.GetBinContent(10)
            # calculate SF to lumin
            scaleFactor = self.lumin * xs/nGenTotal
        return scaleFactor
        
    
    def _getCrossection(self):
        self.xsTable = { 
                    # diboson
                    'ww'              : 113898,
                    'wz'              : 23767,
                    'zz'              : 15878,
                    'wz_2l2q'         : 5595,
                    'wz_3lnu'         : 4430,
                    'zz_2l2nu'        : 564,
                    'zz_2l2q'         : 3220,
                    'zz_4l'           : 1210,
                    
                    # Z
                    'zjets_m-10to50_amcatnlo'  : 18610000,
                    'zjets_m-50_amcatnlo'      :  5765400,
                    'DYJetsToLL_m-50' : 5765400,
                    'DY0JetsToLL'     : 4757000,
                    'DY1JetsToLL'     :  884400,
                    'DY2JetsToLL'     :  338900,

                    # W
                    'w1jets'          :  9493000,
                    'w2jets'          :  3120000,
                    'w3jets'          :  942300,
                    'w4jets'          :  524100,

                    'WJetsToLNu_HT_100To200' : 1345000,
                    'WJetsToLNu_HT_200To400' : 359700,
                    'WJetsToLNu_HT_400To600' : 48910,
                    'WJetsToLNu_HT_600To800' : 12050,
                    'WJetsToLNu_HT_800To1200' : 5501,

                    # top
                    't_tw'            :  35850,
                    'tbar_tw'         :  35850,
                    'ttbar_inclusive' :  832000,
                    'ttbar_2l2nu'     :  87340,
                    'ttbar_semilepton':  364456,
                    'ttbar_hadronic'    :  380204,

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
        if self.selection in ['mumu','mutau','mutau_fakes','mu4j','mu4j_fakes','mumutau']:
            self.datalist = ['muon_2018A','muon_2018B', 'muon_2018C','muon_2018D']

        elif self.selection in ['ee','etau','etau_fakes','e4j','e4j_fakes','eetau']:
            self.datalist = ['electron_2018A','electron_2018B', 'electron_2018C','electron_2018D']

        elif self.selection in ['emu','emutau']:
            self.datalist = ['muon_2018A','muon_2018B', 'muon_2018C','muon_2018D']
            self.datalist+= ['electron_2018A','electron_2018B', 'electron_2018C','electron_2018D']


        ## 2. define the MC list
        self.mcdibosonlist  = [ 'ww','wz','zz']
        self.mczlist        = [ 'DYJetsToLL_m-50']
        self.mcwlist        = [ 'WJetsToLNu_HT_100To200','WJetsToLNu_HT_200To400','WJetsToLNu_HT_400To600','WJetsToLNu_HT_600To800','WJetsToLNu_HT_800To1200' ]
        self.mctlist        = [ 't_tw','tbar_tw']       
        self.mcttlist       = [ 'ttbar_2l2nu','ttbar_semilepton','ttbar_hadronic']

        self.mclist = self.mcdibosonlist+self.mcwlist+self.mczlist+self.mctlist+self.mcttlist

