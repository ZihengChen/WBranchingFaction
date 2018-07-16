import os, sys, tqdm

import numpy as np
import pandas as pd
import ROOT as root

import utility_common as common


class BLTReader:

    def __init__(self, inputRootFileName, selection, includeTTTheory=False):

        self.dataDirectory = common.dataDirectory(isLocal=False) 

        self.inputRootFileName = inputRootFileName
        self.inputRootFile = root.TFile(self.dataDirectory+"root/"+inputRootFileName)

        self.selection = selection
        self.lumin = 35.864
        
        self.includeTTTheory = includeTTTheory
        self._setCrossection()
        self._setNameList()
        
    #############################
    ## Save number of Gen 
    #############################
        
    def outputNGen(self):
        names = ["t","tt"] 
        nGens = [self.getNGen("t_tw")+self.getNGen("tbar_tw"),self.getNGen("ttbar_inclusive")] 
        
        if self.includeTTTheory:
            names = names + self.mcttTheorylist
            nGens = nGens + [self.getNGen(name) for name in self.mcttTheorylist]

        df = pd.DataFrame({"name":names, "ngen":nGens })
        df.to_pickle(self.dataDirectory+"pickle/ngen.pkl")
        
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
        for name in self.datalist + self.mclist:
            self.makePickle(name)
        print(self.selection + " finished!")

    # MARK-1 -- ntuple to pickle
    def makePickle(self,name):
        scaleFactor = self._getScaleFactor(name)
        outputPath = self._getOutputPath(name)

        makeDirectory(outputPath, clear=False)

        tree = self.inputRootFile.Get('{}/bltTree_{}'.format(self.selection,name))

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
            entry.update(self._fillAllVariables(tree, name, scaleFactor))
            n -= 1
            yield entry


    #############################
    ## private helper functions
    #############################

    def _fillAllVariables(self, tree, name, scaleFactor):

        out_dict = {}
        
        # 0. Filling Event Info
        if self.selection == 'emu':
            out_dict['runNumber']    =  tree.runNumber
            out_dict['evtNumber']    =  tree.evtNumber

        out_dict['nMuons']       =  tree.nMuons
        out_dict['nElectrons']   =  tree.nElectrons
        out_dict['nJets']        =  tree.nJets
        out_dict['nBJets']       =  tree.nBJets
        out_dict['nPV']          =  tree.nPV
        out_dict['triggerLepton']=  tree.triggerLeptonStatus

        out_dict['eventWeight']  =  scaleFactor * tree.eventWeight
        out_dict['eventWeightSF']=  scaleFactor

        out_dict['met']          =  tree.met
        out_dict['metPhi']       =  tree.metPhi
        out_dict['genCategory']  =  tree.genCategory


        if name == 'ttbar_inclusive':
            out_dict['qcd_weight_nominal']   = tree.qcdWeights[0]
            out_dict['qcd_weight_nom_up']    = tree.qcdWeights[1]
            out_dict['qcd_weight_nom_down']  = tree.qcdWeights[2]
            out_dict['qcd_weight_up_nom']    = tree.qcdWeights[3]
            out_dict['qcd_weight_up_up']     = tree.qcdWeights[4]
            out_dict['qcd_weight_up_down']   = tree.qcdWeights[5]
            out_dict['qcd_weight_down_nom']  = tree.qcdWeights[6]
            out_dict['qcd_weight_down_up']   = tree.qcdWeights[7]
            out_dict['qcd_weight_down_down'] = tree.qcdWeights[8]

            pdf_var = tree.pdfWeight
            out_dict['pdf_var']              = pdf_var #+ (tree.qcdWeights[0] - tree.qcdWeights[9])**2
            out_dict['pdf_weight_up']        = 1+np.sqrt(pdf_var/99)
            out_dict['pdf_weight_down']      = 1-np.sqrt(pdf_var/99)
            out_dict['alpha_s_err']          = tree.alphaS
        
        if "2016" in name:
            out_dict["nJetsJESUp"]   = tree.nJets
            out_dict["nJetsJESDown"] = tree.nJets
            out_dict["nJetsJERUp"]   = tree.nJets
            out_dict["nJetsJERDown"] = tree.nJets
            out_dict["nBJetsJESUp"]   = tree.nBJets
            out_dict["nBJetsJESDown"] = tree.nBJets
            out_dict["nBJetsJERUp"]   = tree.nBJets
            out_dict["nBJetsJERDown"] = tree.nBJets

            out_dict["nBJetsBTagUp"]   = tree.nBJets
            out_dict["nBJetsBTagDown"] = tree.nBJets
            out_dict["nBJetsMistagUp"]   = tree.nBJets
            out_dict["nBJetsMistagDown"] = tree.nBJets
            
        else:
            out_dict["nJetsJESUp"]   = tree.nJetsJESUp
            out_dict["nJetsJESDown"] = tree.nJetsJESDown
            out_dict["nJetsJERUp"]   = tree.nJetsJERUp
            out_dict["nJetsJERDown"] = tree.nJetsJERDown
            out_dict["nBJetsJESUp"]   = tree.nBJetsJESUp
            out_dict["nBJetsJESDown"] = tree.nBJetsJESDown
            out_dict["nBJetsJERUp"]   = tree.nBJetsJERUp
            out_dict["nBJetsJERDown"] = tree.nBJetsJERDown

            out_dict["nBJetsBTagUp"]   = tree.nBJetsBTagUp
            out_dict["nBJetsBTagDown"] = tree.nBJetsBTagDown
            out_dict["nBJetsMistagUp"]   = tree.nBJetsMistagUp
            out_dict["nBJetsMistagDown"] = tree.nBJetsMistagDown


        
        # 1. Filling leptons
        lep1 = tree.leptonOneP4
        out_dict['lepton1_flavor']  = tree.leptonOneFlavor
        out_dict['lepton1_q']       = np.sign(tree.leptonOneFlavor)
        out_dict['lepton1_iso']     = tree.leptonOneIso
        out_dict['lepton1_reliso']  = tree.leptonOneIso/lep1.Pt()
        out_dict['lepton1_mother']  = tree.leptonOneMother
        out_dict['lepton1_d0']      = abs(tree.leptonOneD0)
        out_dict['lepton1_dz']      = abs(tree.leptonOneDZ)
        out_dict['lepton1_pt']      = lep1.Pt()
        out_dict['lepton1_eta']     = lep1.Eta()
        out_dict['lepton1_phi']     = lep1.Phi()
        out_dict['lepton1_mt']      = (2*lep1.Pt()*tree.met*(1-np.cos(lep1.Phi()-tree.metPhi )))**0.5
        out_dict['lepton1_energy']  = lep1.Energy()
            
        if self.selection in ['e4j','mu4j','e4j_fakes','mu4j_fakes']:
            jet1, jet2, jet3, jet4 = tree.jetOneP4, tree.jetTwoP4, tree.jetThreeP4, tree.jetFourP4
            tag1, tag2, tag3 ,tag4 = tree.jetOneTag, tree.jetTwoTag, tree.jetThreeTag, tree.jetFourTag   
            
            lightjets = np.argsort([tag1,tag2,tag3,tag4])[:2]
            if (0 in lightjets) and (1 in lightjets):
                jet_b1= jet3
                jet_b2= jet4
                dijet = jet1+jet2
            if (0 in lightjets) and (2 in lightjets):
                jet_b1= jet2
                jet_b2= jet4
                dijet = jet1+jet3
            if (0 in lightjets) and (3 in lightjets):
                jet_b1= jet2
                jet_b2= jet3
                dijet = jet1+jet4
            if (1 in lightjets) and (2 in lightjets):
                jet_b1= jet1
                jet_b2= jet4
                dijet = jet2+jet3
            if (1 in lightjets) and (3 in lightjets):
                jet_b1= jet1
                jet_b2= jet3
                dijet = jet2+jet4
            if (2 in lightjets) and (3 in lightjets):
                jet_b1= jet1
                jet_b2= jet2
                dijet = jet3+jet4

            out_dict['dijet_m']      = dijet.M()
            deltaphi_jet_b1 = abs(dijet.DeltaPhi(jet_b1))
            deltaphi_jet_b2 = abs(dijet.DeltaPhi(jet_b2))
            deltar_jet_b1 = abs(dijet.DeltaR(jet_b1))
            deltar_jet_b2 = abs(dijet.DeltaR(jet_b2))
            trijet_mass1 = (dijet+jet_b1).M()
            trijet_mass2 = (dijet+jet_b2).M()

            if deltar_jet_b1>deltar_jet_b2:
                deltar_jet_b1,deltar_jet_b2     = deltar_jet_b2,  deltar_jet_b1
                deltaphi_jet_b1,deltaphi_jet_b2 = deltaphi_jet_b2,deltaphi_jet_b1
                trijet_mass1,   trijet_mass2    = trijet_mass2,   trijet_mass1

            out_dict['deltaphi_jet_b1'] = deltaphi_jet_b1
            out_dict['deltaphi_jet_b2'] = deltaphi_jet_b2
            out_dict['deltar_jet_b1'] = deltar_jet_b1
            out_dict['deltar_jet_b2'] = deltar_jet_b2
            out_dict['trijet_mass1'] = trijet_mass1
            out_dict['trijet_mass2'] = trijet_mass2


            
            # 2. Filling Jets
            out_dict['jet1_pt']      = jet1.Pt()
            out_dict['jet1_eta']     = jet1.Eta()
            
            out_dict['jet2_pt']      = jet2.Pt()
            out_dict['jet2_eta']     = jet2.Eta()

            out_dict['jet3_pt']      = jet3.Pt()
            out_dict['jet3_eta']     = jet3.Eta()
            
            out_dict['jet4_pt']      = jet4.Pt()
            out_dict['jet4_eta']     = jet4.Eta()


        else:
            # lepton2
            lep2 = tree.leptonTwoP4

            out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
            out_dict['lepton2_q']       = np.sign(tree.leptonTwoFlavor)
            out_dict['lepton2_iso']     = tree.leptonTwoIso
            out_dict['lepton2_reliso']  = tree.leptonTwoIso/lep2.Pt()
            out_dict['lepton2_mother']  = tree.leptonTwoMother
            out_dict['lepton2_d0']      = abs(tree.leptonTwoD0)
            out_dict['lepton2_dz']      = abs(tree.leptonTwoDZ)
            
            out_dict['lepton2_pt']      = lep2.Pt()
            out_dict['lepton2_eta']     = lep2.Eta()
            out_dict['lepton2_phi']     = lep2.Phi()
            out_dict['lepton2_mt']      = (2*lep2.Pt()*tree.met*(1-np.cos(lep2.Phi()-tree.metPhi )))**0.5
            out_dict['lepton2_energy']  = lep2.Energy() 

            out_dict['lepton_delta_eta']= abs(lep1.Eta() - lep2.Eta())
            out_dict['lepton_delta_phi']= abs(lep1.DeltaPhi(lep2))
            out_dict['lepton_delta_r']  = lep1.DeltaR(lep2)
            
            # dilepton
            dilepton = lep1 + lep2

            out_dict['dilepton_mass']      = dilepton.M()
            out_dict['dilepton_pt']        = dilepton.Pt()
            out_dict['dilepton_eta']       = dilepton.Eta()
            out_dict['dilepton_phi']       = dilepton.Phi()
            out_dict['dilepton_pt_over_m'] = dilepton.Pt()/ ( dilepton.M() + 1e-6)
            
            
            # 2. Filling Jets
            jet1, jet2 = tree.jetOneP4,    tree.jetTwoP4

            out_dict['jet1_pt']      = jet1.Pt()
            out_dict['jet1_eta']     = jet1.Eta()
            out_dict['jet1_phi']     = jet1.Phi()
            out_dict['jet1_energy']  = jet1.Energy()
            out_dict['jet1_tag']     = tree.jetOneTag

            out_dict['jet2_pt']      = jet2.Pt()
            out_dict['jet2_eta']     = jet2.Eta()
            out_dict['jet2_phi']     = jet2.Phi()
            out_dict['jet2_energy']  = jet2.Energy()
            out_dict['jet2_tag']    = tree.jetTwoTag
            
            out_dict['jet_delta_eta']   = abs(jet1.Eta() - jet2.Eta())
            out_dict['jet_delta_phi']   = abs(jet1.DeltaPhi(jet2))
            out_dict['jet_delta_r']     = jet1.DeltaR(jet2)
            # dijet
            dijet      = jet1 + jet2
            out_dict['dijet_mass']      = dijet.M()
            out_dict['dijet_pt']        = dijet.Pt()
            out_dict['dijet_eta']       = dijet.Eta()
            out_dict['dijet_phi']       = dijet.Phi()
            out_dict['dijet_pt_over_m'] = dijet.Pt()/(dijet.M()+ 1e-6)

        return out_dict


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
        
    
    def _setCrossection(self):
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
        outputPath  = self.dataDirectory+"pickle/"+self.selection+"/"
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

    def _setNameList(self):
        ## 1. define the datalist
        if self.selection in ["mumu","mutau","mu4j","mu4j_fakes"]:
            self.datalist = [
                'muon_2016B', 'muon_2016C','muon_2016D','muon_2016E',
                'muon_2016F','muon_2016G','muon_2016H'
                ]

        elif self.selection in ["ee","etau","e4j","e4j_fakes"]:
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

        self.mcttlist       = [ 'ttbar_inclusive']

        self.mcttTheorylist = [ 'ttbar_inclusive_fsrdown','ttbar_inclusive_fsrup',
                                'ttbar_inclusive_isrdown','ttbar_inclusive_isrup',
                                'ttbar_inclusive_hdampdown','ttbar_inclusive_hdampup',
                                'ttbar_inclusive_down','ttbar_inclusive_up']

        self.mclist = self.mcttbosonlist + self.mcdibosonlist + self.mcdylist + self.mctlist + self.mcttlist 
        
        if self.includeTTTheory:
            self.mclist + self.mcttTheorylist

        self.mcttTheorylistFileNames = {
            'ttbar_inclusive_fsrdown' : 'ttbar_inclusive_FSRDown',
            'ttbar_inclusive_fsrup'   : 'ttbar_inclusive_FSRUp',
            'ttbar_inclusive_isrdown' : 'ttbar_inclusive_ISRDown',
            'ttbar_inclusive_isrup'   : 'ttbar_inclusive_ISRUp',,
            'ttbar_inclusive_hdampdown':'ttbar_inclusive_MEPSDown',
            'ttbar_inclusive_hdampup' : 'ttbar_inclusive_MEPSUp' ,
            'ttbar_inclusive_down'    : 'ttbar_inclusive_UEDown',
            'ttbar_inclusive_up'      : 'ttbar_inclusive_UEUp'
        }
    






