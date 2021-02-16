import numpy as np


def getAllVariables_multileptonSelection( tree, selection, name, scaleFactor):
    isData = ('201' in name) 
    isDilepton = not ('4j' in selection)
    isMCZ = 'zjets' in name
    isTauReweight = 'tauReweight' in name
    isSingleElectronTrigger = selection in ["ee","emu","etau","e4j","etau_fakes","e4j_fakes"]
    isDYx = selection in ["eemu","mumue"]
    
    out_dict = {}
    
    # 0. Filling Event Info
    if selection in ['emu','emutau']:
        out_dict['runNumber']    =  tree.runNumber
        out_dict['evtNumber']    =  tree.evtNumber
    

    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nTaus']        =  tree.nTaus
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV
    out_dict['nPU']          =  tree.nPU
    out_dict['trTest']       =  tree.triggerLeptonStatus
    out_dict['genCategory']  =  tree.genCategory
    out_dict['nPartons']     =  tree.nPartons

    if isTauReweight:
      out_dict['genTauOneDaughters']  =  tree.genTauOneDaughters
      out_dict['genTauTwoDaughters']  =  tree.genTauTwoDaughters

    
    if isData:
        out_dict['eventWeight']  =  scaleFactor * tree.eventWeight
    else:
        out_dict['eventWeight']  =  scaleFactor * tree.eventWeight * tree.genWeight
    if isSingleElectronTrigger:
        out_dict['eleTriggerVarTagSyst']    =  tree.eleTriggerVarTagSyst
        out_dict['eleTriggerVarProbeSyst']  =  tree.eleTriggerVarProbeSyst
        out_dict['prefiringWeight'] = tree.prefiringWeight
        out_dict['prefiringVar'] = tree.prefiringVar
        
    

    out_dict['topPtWeight']   =  tree.topPtWeight
    out_dict['topPtVar']      =  tree.topPtVar
    out_dict['eventWeightSF'] =  scaleFactor
    out_dict['met']           =  tree.met
    out_dict['metPhi']        =  tree.metPhi
    out_dict['ht']            =  tree.ht
    out_dict['htSum']         =  tree.htSum

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
    
    # lepton impact parameters
    if not selection in ['eetau','mumutau','emutau']:
        out_dict['lepton1_d0']      = tree.leptonOneD0
        out_dict['lepton1_dZ']      = tree.leptonOneDz
        out_dict['lepton1_sip3d']   = tree.leptonOneSip3d
        if isDilepton:
            out_dict['lepton2_d0']      = tree.leptonTwoD0
            out_dict['lepton2_dZ']      = tree.leptonTwoDz
            out_dict['lepton2_sip3d']   = tree.leptonTwoSip3d




    # 2. Filling tau info
    if selection in ['etau','mutau','eetau','mumutau','emutau']:
        if selection in ['eetau','mumutau','emutau']:
            out_dict['tauMVA']        = tree.leptonThreeIso 
        else:
            out_dict['tauMVA']        = tree.tauMVA
        out_dict['tauDecayMode']      = tree.tauDecayMode
        out_dict['tauVetoedJetPt']    = tree.tauVetoedJetPt
        out_dict['tauVetoedJetPtUnc'] = tree.tauVetoedJetPtUnc
         

        if isData:
            out_dict['tauGenFlavor']      = 26 # default value
            out_dict['tauGenFlavorHad']   = 26 # default value
        else:
            out_dict['tauGenFlavor']      = tree.tauGenFlavor
            out_dict['tauGenFlavorHad']   = tree.tauGenFlavorHad

        if selection in ['eetau','mumutau','emutau']:
            leptonThreeP4 = tree.leptonThreeP4
            out_dict['tauPt']  = leptonThreeP4.Pt()
            out_dict['tauEta'] = leptonThreeP4.Eta()
            out_dict['tauPhi'] = leptonThreeP4.Phi()
            out_dict['tauMass']= leptonThreeP4.M()
        else:
            out_dict['tauMass']= lep2.M()

    if isDYx:
      leptonThreeP4 = tree.leptonThreeP4
      out_dict['lepton3_pt']  = leptonThreeP4.Pt()
      out_dict['lepton3_eta']  = leptonThreeP4.Eta()
      out_dict['lepton3_phi']  = leptonThreeP4.Phi()
      out_dict['lepton3_mt']  = (2*leptonThreeP4.Pt()*tree.met*(1-np.cos(leptonThreeP4.Phi()-tree.metPhi)))**0.5
      out_dict['lepton3_reliso'] = tree.leptonThreeIso
      out_dict['lepton3_passiso'] = tree.leptonThreeIsoPassTight
      out_dict['nProbs'] = tree.nProbs
      out_dict['lepton3_dilepton_delta_r']  = leptonThreeP4.DeltaR(dilepton)
      out_dict['lepton3_dilepton_delta_phi']  = abs(leptonThreeP4.DeltaPhi(dilepton))



    # 3. jet systematics
    out_dict['nJetsJESUp']        =  tree.nJetsJESUp
    out_dict['nJetsJESDown']      =  tree.nJetsJESDown
    out_dict['nJetsJERUp']        =  tree.nJetsJERUp
    out_dict['nJetsJERDown']      =  tree.nJetsJERDown


    out_dict['nBJetsJESUp']       =  tree.nBJetsJESUp
    out_dict['nBJetsJESDown']     =  tree.nBJetsJESDown
    out_dict['nBJetsJERUp']       =  tree.nBJetsJERUp
    out_dict['nBJetsJERDown']     =  tree.nBJetsJERDown
    out_dict['nBJetsBTagUp']      =  tree.nBJetsBTagUp
    out_dict['nBJetsBTagDown']    =  tree.nBJetsBTagDown
    out_dict['nBJetsMistagUp']    =  tree.nBJetsMistagUp
    out_dict['nBJetsMistagDown']  =  tree.nBJetsMistagDown


    # 4. extra
    ############################
    # lepton1 reco/id uncertainties
    tempstd = 0.0
    if tree.leptonOneRecoWeight>0.8:
        tempstd = np.sqrt(tree.leptonOneRecoVar)/tree.leptonOneRecoWeight
    out_dict['lepton1_recostd'] = tempstd
    tempstd = 0.0
    if tree.leptonOneIDWeight>0.8:
        tempstd = np.sqrt(tree.leptonOneIDVar)/tree.leptonOneIDWeight
    out_dict['lepton1_idstd'] = tempstd


    # lepton2 reco/id uncertainties
    if isDilepton:
        tempstd = 0.0
        if tree.leptonTwoRecoWeight>0.8:
            tempstd = np.sqrt(tree.leptonTwoRecoVar)/tree.leptonTwoRecoWeight
        out_dict['lepton2_recostd'] = tempstd
        tempstd = 0.0
        if tree.leptonTwoIDWeight>0.8:
            tempstd = np.sqrt(tree.leptonTwoIDVar)/tree.leptonTwoIDWeight
        out_dict['lepton2_idstd'] = tempstd
    ############################


    return out_dict