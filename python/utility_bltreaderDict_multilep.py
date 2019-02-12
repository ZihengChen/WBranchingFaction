import numpy as np


def getAllVariables_multileptonSelection( tree, selection, name, scaleFactor):
    isData = ('2016' in name)
    isDilepton = not ('4j' in selection)
    
    out_dict = {}
    
    # 0. Filling Event Info
    if selection in ['emu','emu_tau']:
        out_dict['runNumber']    =  tree.runNumber
        out_dict['evtNumber']    =  tree.evtNumber
    

    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nTaus']        =  tree.nTaus
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV
    out_dict['trTest']       =  tree.triggerLeptonStatus
    out_dict['genCategory']  =  tree.genCategory
    
    if isData:
        out_dict['eventWeight']  =  scaleFactor * tree.eventWeight 
    else:
        out_dict['eventWeight']  =  scaleFactor * tree.eventWeight * tree.genWeight
        
    out_dict['eventWeightSF'] =  scaleFactor
    out_dict['met']           =  tree.met
    out_dict['metPhi']        =  tree.metPhi



    # 1. Filling leptons
    lep1 = tree.leptonOneP4
    out_dict['lepton1_flavor']  = tree.leptonOneFlavor
    out_dict['lepton1_q']       = np.sign(tree.leptonOneFlavor)
    out_dict['lepton1_reliso']  = tree.leptonOneIso/lep1.Pt()
    out_dict['lepton1_pt']      = lep1.Pt()
    out_dict['lepton1_eta']     = lep1.Eta()
    out_dict['lepton1_phi']     = lep1.Phi()
    out_dict['lepton1_mt']      = (2*lep1.Pt()*tree.met*(1-np.cos(lep1.Phi()-tree.metPhi )))**0.5
    
    if isDilepton:
        lep2 = tree.leptonTwoP4
        out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
        out_dict['lepton2_q']       = np.sign(tree.leptonTwoFlavor)
        out_dict['lepton2_reliso']  = tree.leptonTwoIso/lep2.Pt()
        out_dict['lepton2_pt']      = lep2.Pt()
        out_dict['lepton2_eta']     = lep2.Eta()
        out_dict['lepton2_phi']     = lep2.Phi()
        out_dict['lepton2_mt']      = (2*lep2.Pt()*tree.met*(1-np.cos(lep2.Phi()-tree.metPhi )))**0.5

        out_dict['lepton_delta_eta']= abs(lep1.Eta() - lep2.Eta())
        out_dict['lepton_delta_phi']= abs(lep1.DeltaPhi(lep2))
        out_dict['lepton_delta_r']  = lep1.DeltaR(lep2)
            
        # dilepton
        dilepton = lep1 + lep2
        out_dict['dilepton_mass']      = dilepton.M()
        out_dict['dilepton_pt']        = dilepton.Pt()
    
    
    # 2. Filling tau info
    if selection in ['etau','mutau','ee_tau','mumu_tau','emu_tau']:
        out_dict['tauMVA']            = tree.tauMVA
        out_dict['tauDecayMode']      = tree.tauDecayMode
        out_dict['tauVetoedJetPt']    = tree.tauVetoedJetPt
        out_dict['tauVetoedJetPtUnc'] = tree.tauVetoedJetPtUnc
         

        if isData:
            out_dict['tauGenFlavor']      = 26 # default value
            out_dict['tauGenFlavorHad']   = 26 # default value
        else:
            out_dict['tauGenFlavor']      = tree.tauGenFlavor
            out_dict['tauGenFlavorHad']   = tree.tauGenFlavorHad

        if selection in ['ee_tau','mumu_tau','emu_tau']:
            leptonTauP4 = tree.leptonTauP4
            out_dict['tauPt']  = leptonTauP4.Pt()
            out_dict['tauEta'] = leptonTauP4.Eta()
            out_dict['tauPhi'] = leptonTauP4.Phi()
            out_dict['tauMass']= leptonTauP4.M()
        else:
            out_dict['tauMass']= lep2.M()

    return out_dict