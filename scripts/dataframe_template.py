#!/usr/bin/env python
import multiprocessing as mp
from utility_dftemplater import *


def templateDataFrames(variation=''):
    baseDir = common.getBaseDirectory()

    tp = DFTemplater(variation)
    xShp,xCnt,yShp,yCnt = tp.makeTemplatesAndTargets()
    
    # save the template and target
    np.save(baseDir + "data/templates/shaping_signalRegion/X_{}".format(variation),xShp)
    np.save(baseDir + "data/templates/counting_signalRegion/X_{}".format(variation),xCnt)
    if variation == '':
        np.save(baseDir + "data/templates/shaping_signalRegion/Y_",yShp)
        np.save(baseDir + "data/templates/counting_signalRegion/Y_",yCnt)
    
def runTemplateDataFrames(vlist):
    processes = []
    for v in vlist:
        processes.append(mp.Process(target=templateDataFrames,args=(v,)))
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
            runTemplateDataFrames(variations[i:i+nThread])
        else:
            runTemplateDataFrames(variations[i:nVar])