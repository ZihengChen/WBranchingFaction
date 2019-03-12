#!/usr/bin/env python

from utility_dfcounter import *
from utility_dfcutter import *
import multiprocessing as mp

import os 


def countDataFrames(variation=''):
    labels  = ['trigger','usetag','acc','accVar','nfake','nfakeVar','nmcbg','nmcbgVar','nmcsg','nmcsgVar','ndata','ndataVar']
    records = []
    
    for trigger in ['mu','e']:
        for usetag in ['1b','2b']:
            
            print( variation + '-- counting '+trigger+usetag + ' ...')

            counter = DFCounter(trigger,usetag)
            counter.setVariation(variation)

            acc,accVar = counter.returnAcc(enhanceStat=False)
            nfake,nfakeVar = counter.returnNFake()
            nmcbg,nmcbgVar = counter.returnNMCbg()
            nmcsg,nmcsgVar = counter.returnNMCsg()
            ndata,ndataVar = counter.returnNData()
            records.append( (trigger,usetag,acc,accVar,nfake,nfakeVar,nmcbg,nmcbgVar,nmcsg,nmcsgVar,ndata,ndataVar) )

    df = pd.DataFrame.from_records(records, columns=labels)
    df.to_pickle( common.getBaseDirectory() + 'data/counts/count_{}.pkl'.format(variation))
    print( 'counting finished with variation {}'.format(variation))


def runCountDataFrames(vlist):
    processes = []
    for v in vlist:
        processes.append(mp.Process(target=countDataFrames,args=(v,)))
    for ps in processes:
        ps.start()
    for ps in processes:
        ps.join()

if __name__ == '__main__':
    nThread = 4

    variations = [
        '',
        'EIDEffDown','ERecoEffDown','MuIDEffDown','MuRecoEffDown',
        'TauIDEffDown','JetToTauIDEffDown',
        'EPtDown','MuPtDown','Tau0PtDown','Tau1PtDown','Tau10PtDown',
        'JESUp','JESDown','JERUp','JERDown','BTagUp','BTagDown','MistagUp','MistagDown',
        'PileupUp','PileupDown'
        ]

    nVar = len(variations)
    for i in range(0, nVar, nThread):
        if i+nThread <= nVar:
            runCountDataFrames(variations[i:i+nThread])
        else:
            runCountDataFrames(variations[i:nVar])


    ############################
    # get signal composition
    ############################
    
    print("get signal composition")
    sigcomp = []
    for selection in ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:
        for nbjet in ['==1','>1']:
            
            mctt = DFCutter(selection,nbjet,"mctt").getDataFrame()
            mct  = DFCutter(selection,nbjet,"mct" ).getDataFrame()
            mcsg = pd.concat([mct,mctt],ignore_index=True)
            
            nSig =  mcsg.eventWeight.sum()
            temp = []
            
            for i in range(21):
                nSig_i = mcsg[mcsg.genCategory==i+1].eventWeight.sum()
                temp.append(nSig_i/nSig)
            sigcomp.append(temp)

    sigcomp = np.array(sigcomp).T
    np.save(common.getBaseDirectory()+'data/counts/sigcomp_',sigcomp)
    print("signal composition saved")


    ############################
    # generate latex tabel
    ############################
    os.system('python {}'.format(common.getBaseDirectory()+'scripts/note_sigcomp.py'))
    os.system('python {}'.format(common.getBaseDirectory()+'scripts/note_sigacc.py'))
    os.system('python {}'.format(common.getBaseDirectory()+'scripts/note_yields.py'))
    





