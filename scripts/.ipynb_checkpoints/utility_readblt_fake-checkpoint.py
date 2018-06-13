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

XS_table ={ 'ww'              :  12178,
            'wz_2l2q'         :  5595,
            'wz_3lnu'         :  4430,
            'zz_2l2nu'        :  564,
            'zz_2l2q'         :  3220,
            'zz_4l'           :  1210,
           
           'zjets_m-10to50'   : 18610000,
           'z1jets_m-10to50'  : 1.18*730300,
           'z2jets_m-10to50'  : 1.18*387400,
           'z3jets_m-10to50'  : 1.18*95020,
           'z4jets_m-10to50'  : 1.18*36710,

           'zjets_m-50'       :  5765400,
           'z1jets_m-50'      :  1.18*1012000,
           'z2jets_m-50'      :  1.18*334700,
           'z3jets_m-50'      :  1.18*102300,
           'z4jets_m-50'      :  1.18*54520,

           'w1jets'           :  9493000,
           'w2jets'           :  3120000,
           'w3jets'           :  942300,
           'w4jets'           :  524100,
           
           't_tw'             :  35850,
           'tbar_tw'          :  35850,
           'ttbar_inclusive'  :  832000,
           
           'TTZToLLNuNu'      : 252.9,
           'TTZToQQ'          : 529.7,
           'TTWJetsToLNu'     : 204.3,
           'TTWJetsToQQ'      : 406.2,
           'ttHJetTobb'       : 295.0,

           'qcd_ht100to200'   :27990000000,
           'qcd_ht200to300'   :1712000000,
           'qcd_ht300to500'   :347700000,
           'qcd_ht500to700'   :32100000,
           'qcd_ht700to1000'  :6831000,
           'qcd_ht1000to1500' :1207000,
           'qcd_ht1500to2000' :119900,
           'qcd_ht2000'       :25240,
          }


def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')


def fill_lepton_vars(tree, name, SF):

    out_dict = {}
    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV
    out_dict['nPU']          =  tree.nPU
    
    correction = tree.eventWeightPu
    correction *= tree.eventWeightTrigger
    correction *= tree.eventWeightId
    correction *= tree.eventWeightIso
    
    weight = SF * correction
    if name in ['zjets_m-50','zjets_m-10to50']:
        out_dict['nPartons']  =  tree.nPartons
    #     if 0 < tree.nPartons < 5:
    #         weight  = 0.0
    out_dict['eventWeight']  =  weight
    out_dict['met']          =  tree.met
    out_dict['metPhi']       =  tree.metPhi

    
    # 1. Filling leptons
    lep1 = tree.leptonOneP4
    lep2 = tree.leptonTwoP4
    dilepton = lep1+lep2

    out_dict['lepton1_pt']  = lep1.Pt()
    out_dict['lepton1_eta'] = lep1.Eta()
    out_dict['lepton1_phi'] = lep1.Phi()

    out_dict['lepton2_pt']  = lep2.Pt()
    out_dict['lepton2_eta'] = lep2.Eta()
    out_dict['lepton2_phi'] = lep2.Phi()

    out_dict['dilepton_mass'] = dilepton.M()
    
    lep3 = tree.leptonThreeP4
    out_dict['lepton3_iso']     = tree.leptonThreeISO
    
    out_dict['lepton3_deltaPhi']= lep3.DeltaPhi(dilepton) #tree.leptonThreeDeltaPhi
    out_dict['lepton3_pt']      = lep3.Pt()
    out_dict['lepton3_eta']     = lep3.Eta()
    out_dict['lepton3_phi']     = lep3.Phi()
    
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
        entry.update(fill_lepton_vars(tree, name, SF))
        n -= 1
        yield entry

def pickle_ntuple(ntuple_data):
    # unpack input data
    name        = ntuple_data[0]
    SF          = ntuple_data[1]
    input_file  = ntuple_data[2]
    output_path = ntuple_data[3]
    if name in datalist:
        output_path +="data2016/"
    elif name in mcttbosonlist:
        output_path +="mcttboson/"
    elif name in mcqcdlist:
        output_path +="mcqcd/"
    elif name in mcdibosonlist:
        output_path +="mcdiboson/"
    elif name in mcdylist:
        output_path +="mcdy/"
    elif name in mctlist:
        output_path +="mct/"
    elif name in mcttlist:
        output_path +="mctt/"
    make_directory(output_path, clear=False)
    
    # get the tree, convert to dataframe, and save df to pickle
    #froot  = TFile(input_file)
    #tree   = froot.Get('tree_{}'.format(name))
    tree   = input_file.Get('tree_{}'.format(name))

    if tree.GetEntriesFast() >0:
    
        ntuple = fill_ntuple(tree, name, SF)
        df     = pd.DataFrame(ntuple)
        df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, name))


###########################################      
###########################################    
#               Main 
###########################################   
###########################################   

input_root_file_name  = "/home/zchen/Documents/Analysis/workplace/data/root/{}.root".format(selection)
input_root_file = TFile(input_root_file_name)
output_directory = "/home/zchen/Documents/Analysis/workplace/data/fake/{}/".format(selection)

## 1. define datalist
datalist      = ['muon_2016B', 'muon_2016C', 'muon_2016D','muon_2016E',
                 'muon_2016F','muon_2016G','muon_2016H']
## 2. define mclist
mcttbosonlist = ['TTZToLLNuNu','TTZToQQ','TTWJetsToLNu','TTWJetsToQQ','ttHJetTobb']

mcqcdlist     = ['qcd_ht100to200','qcd_ht200to300','qcd_ht300to500',
                 'qcd_ht500to700','qcd_ht700to1000','qcd_ht1000to1500',
                 'qcd_ht1000to1500','qcd_ht1500to2000','qcd_ht2000']
mcdibosonlist = ['ww','wz_2l2q','wz_3lnu','zz_2l2nu','zz_2l2q','zz_4l' ]
mcdylist      = ['zjets_m-10to50','zjets_m-50', 
                 #'z1jets_m-10to50','z2jets_m-10to50','z3jets_m-10to50','z4jets_m-10to50',
                 #'z1jets_m-50','z2jets_m-50','z3jets_m-50','z4jets_m-50',
                 'w1jets','w2jets','w3jets','w4jets']
mctlist       = ['t_tw','tbar_tw']
mcttlist      = ['ttbar_inclusive']
mclist =  mcttbosonlist + mcdibosonlist + mcdylist + mctlist + mcttlist

## 3. Calculate SF for each element in datalist and mclist
SF_table = {}
SF_table = defaultdict(lambda: 1.0, SF_table)
for mc in mclist:
    h = input_root_file.Get("TotalEvents_"+mc)
    nGenTotal = h.GetBinContent(1)
    crossection = XS_table[mc]
    SF_table[mc] = crossection * 35.864 / nGenTotal
for data in datalist:
    SF_table[data] = 1.0

## 4. pickle ntuples
dataset_list = datalist + mclist
for dataset in dataset_list:
    SF = SF_table[dataset]
    root2df_config = [dataset,SF,input_root_file,output_directory]
    pickle_ntuple(root2df_config)
