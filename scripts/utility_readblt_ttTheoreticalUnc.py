import os, sys
import numpy as np
import pandas as pd
import ROOT as r
from ROOT import TTree, TObjArray, TCanvas, TPad, TFile, TPaveText
import json
from collections import OrderedDict
from multiprocessing import Pool
from memory_profiler import profile
from root_pandas import read_root
import rootpy
from rootpy.io import root_open
from tqdm import tqdm, trange
from collections import defaultdict

selection = sys.argv[1]
lumin = 35.864
XS_table ={ 
            'ttbar_inclusive_fsrdown':  832000,
            'ttbar_inclusive_fsrup':  832000,
            'ttbar_inclusive_isrdown':  832000,
            'ttbar_inclusive_isrup':  832000,
            'ttbar_inclusive_hdampdown':  832000,
            'ttbar_inclusive_hdampup':  832000,
            'ttbar_inclusive_down':  832000,
            'ttbar_inclusive_up':  832000,
            'ttbar_inclusive'  :  832000,
          }
        

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')


def fill_lepton_vars(tree, name, SF):

    out_dict = {}
    # o. Filling Event Info
    if selection == 'emu':
        out_dict['runNumber']    =  tree.runNumber
        out_dict['evtNumber']    =  tree.evtNumber

    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV
    out_dict['triggerLepton'] = tree.triggerLeptonStatus

    out_dict['eventWeight']  =  SF * tree.eventWeight
    out_dict['eventWeightSF']=  SF

    out_dict['met']          =  tree.met
    out_dict['metPhi']       =  tree.metPhi
    out_dict['genCategory']  =  tree.genCategory
    
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
        
    if selection in ['e4j','mu4j']:
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
        dilepton   = lep1 + lep2

        out_dict['dilepton_mass']      = dilepton.M()
        out_dict['dilepton_pt']        = dilepton.Pt()
        out_dict['dilepton_eta']       = dilepton.Eta()
        out_dict['dilepton_phi']       = dilepton.Phi()
        out_dict['dilepton_pt_over_m'] = dilepton.Pt()/dilepton.M()
        
        
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
        dijet = jet1 + jet2
        out_dict['dijet_mass']      = dijet.M()
        out_dict['dijet_pt']        = dijet.Pt()
        out_dict['dijet_eta']       = dijet.Eta()
        out_dict['dijet_phi']       = dijet.Phi()
        out_dict['dijet_pt_over_m'] = dijet.Pt()/dijet.M()
    return out_dict



def fill_ntuple(tree, name, SF):
    n = int(tree.GetEntriesFast())

    for i in trange(n,
                    desc       = selection+' selection *** '+name,
                    leave      = True,
                    unit_scale = True,
                    ncols      = 100,
                    total      = n
                    ):
        tree.GetEntry(i)
        entry = {}
        #if (name in ['zjets_m-50','zjets_m-10to50']) & (0 < tree.nPartons < 5):
        #    n -= 1
        #    continue
        entry.update(fill_lepton_vars(tree, name, SF))
        n -= 1
        yield entry

def pickle_ntuple(ntuple_data):
    # unpack input data
    name        = ntuple_data[0]
    SF          = ntuple_data[1]
    input_file  = ntuple_data[2]
    output_path = ntuple_data[3]
    if name in mcttTheorylist:
        output_path +="mcttTheory/"  
    make_directory(output_path, clear=False)
    
    # get the tree, convert to dataframe, and save df to pickle
    tree   = input_file.Get('{}/bltTree_{}'.format(selection,name))

    if tree.GetEntriesFast() >0:

        ntuple = fill_ntuple(tree, name, SF)
        df     = pd.DataFrame(ntuple)
        df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, name))


    
    
###########################################    
# My Main function
input_root_file_name  = "/mnt/data/zchen/Analysis/root/2016DataMC_ttTheoreticUnc.root"
input_root_file = TFile(input_root_file_name)
output_directory = "/home/zchen/Documents/Analysis/workplace/data/pickle/{}/".format(selection)

## 1. define the datalist

    
## 2. define the MC list
mcttTheorylist = ['ttbar_inclusive_fsrdown','ttbar_inclusive_fsrup',
                'ttbar_inclusive_isrdown','ttbar_inclusive_isrup',
                'ttbar_inclusive_hdampdown','ttbar_inclusive_hdampup',
                'ttbar_inclusive_down','ttbar_inclusive_up'
               ]
mclist =  mcttTheorylist 

## 3. Calculate SF for each element in datalist and mclist
SF_table = {}
SF_table = defaultdict(lambda: 1.0, SF_table)
for mc in mclist:
    h = input_root_file.Get("TotalEvents_"+mc)
    nGenTotal = h.GetBinContent(1)
    crossection = XS_table[mc]
    SF_table[mc] = crossection*lumin/nGenTotal

## 4. pickle ntuples
dataset_list = mclist
for dataset in dataset_list:
    SF = SF_table[dataset]
    root2df_config = [dataset,SF,input_root_file,output_directory]
    pickle_ntuple(root2df_config)


#out_dict['lepton1_q']       = tree.leptonOneQ
#out_dict['lepton1_iso']     = tree.leptonOneIso
#out_dict['lepton1_flavor']  = tree.leptonOneFlavor
#out_dict['lepton1_trigger'] = tree.leptonOneTrigger
#out_dict['lepton2_q']       = tree.leptonTwoQ
#out_dict['lepton2_iso']     = tree.leptonTwoIso
#out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
#out_dict['lepton2_trigger'] = tree.leptonTwoTrigger
