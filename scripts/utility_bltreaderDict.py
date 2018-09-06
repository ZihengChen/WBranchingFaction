import numpy as np

def getAllVariables_multileptonSelection( tree, selection, name, scaleFactor):

    out_dict = {}
    
    # 0. Filling Event Info
    if selection == 'emu':
        out_dict['runNumber']    =  tree.runNumber
        out_dict['evtNumber']    =  tree.evtNumber

     # 0. Filling Event Info
    if selection in ['etau','mutau']:
        out_dict['tauMVA']    =  tree.tauMVA
        out_dict['tauDecayMode'] =  tree.tauDecayMode


    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV
    out_dict['triggerLepton']=  tree.triggerLeptonStatus

    out_dict['eventWeight']  =  scaleFactor * tree.eventWeight * tree.genWeight
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
        out_dict['pdf_var']              = pdf_var
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
        
    if selection in ['e4j','mu4j','e4j_fakes','mu4j_fakes']:
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
        out_dict['jet2_tag']     = tree.jetTwoTag
        
        out_dict['jet_delta_eta']   = abs(jet1.Eta() - jet2.Eta())
        out_dict['jet_delta_phi']   = abs(jet1.DeltaPhi(jet2))
        out_dict['jet_delta_r']     = jet1.DeltaR(jet2)
        # dijet
        dijet = jet1 + jet2
        out_dict['dijet_mass']      = dijet.M()
        out_dict['dijet_pt']        = dijet.Pt()
        out_dict['dijet_eta']       = dijet.Eta()
        out_dict['dijet_phi']       = dijet.Phi()
        out_dict['dijet_pt_over_m'] = dijet.Pt()/(dijet.M()+ 1e-6)

    return out_dict

def getAllVariables_fakeSelection( tree, selection, name, scaleFactor):

    out_dict = {}

    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV

    out_dict['eventWeight']  =  scaleFactor * tree.eventWeight
    out_dict['eventWeightSF']=  scaleFactor
    out_dict['met']          =  tree.met
    out_dict['metPhi']       =  tree.metPhi

    
    # 1. Filling leptons
    lep1 = tree.leptonOneP4
    lep2 = tree.leptonTwoP4
    dilepton = lep1+lep2

    out_dict['lepton1_pt']  = lep1.Pt()
    out_dict['lepton1_eta'] = lep1.Eta()
    out_dict['lepton1_phi'] = lep1.Phi()
    out_dict['lepton1_iso'] = tree.leptonOneIso
    

    out_dict['lepton2_pt']  = lep2.Pt()
    out_dict['lepton2_eta'] = lep2.Eta()
    out_dict['lepton2_phi'] = lep2.Phi()
    out_dict['lepton2_iso'] = tree.leptonTwoIso

    out_dict['dilepton_mass'] = dilepton.M()
    
    lep3 = tree.leptonThreeP4
    out_dict['lepton3_iso']     = tree.leptonThreeIso
    if selection in ['ee_e','mumu_e']:
        out_dict['lepton3_isopass'] = tree.leptonThreeIsoPass
    if selection in ['emu_tau']:
        out_dict['tauDecayMode'] = tree.tauDecayMode
        out_dict['tauMVA']       = tree.tauMVA
        #out_dict['tauMVAOld']    = tree.tauMVAOld
        out_dict['tauChHadIso']  = tree.taupuppiChHadIso
        out_dict['tauNeuNeuIso'] = tree.taupuppiNeuHadIso
        out_dict['tauGammaISO']  = tree.taupuppiGammaIso
        
    out_dict['lepton3_deltaPhi']= lep3.DeltaPhi(dilepton)
    out_dict['lepton3_pt']      = lep3.Pt()
    out_dict['lepton3_eta']     = lep3.Eta()
    out_dict['lepton3_phi']     = lep3.Phi()

    trilepton = lep1+lep2+lep3
    out_dict['trilepton_pt']      = trilepton.Pt()
    out_dict['trilepton_eta']     = trilepton.Pt()
    out_dict['trilepton_phi']     = trilepton.Pt()
    out_dict['trilepton_mass']    = trilepton.M()
    
    return out_dict