#!/usr/bin/env python
from multiprocessing import Pool
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

if __name__ == '__main__':
    variations = [
        '',
        'EIDEffDown','ERecoEffDown','MuIDEffDown','MuRecoEffDown',
        'TauIDEffDown','JetToTauIDEffDown',
        'EPtDown','MuPtDown','Tau0PtDown','Tau1PtDown','Tau10PtDown',
        'JESUp','JESDown','JERUp','JERDown','BTagUp','BTagDown','MistagUp','MistagDown',
        'PileupUp','PileupDown','TopPtReweightDown',
        'tauReweight1000Down','tauReweight11000Down','tauReweight21000Down','tauReweight3000Down','tauReweight13000Down',
        'tauReweight1000Up','tauReweight11000Up','tauReweight21000Up','tauReweight3000Up','tauReweight13000Up',
        'fsrUp','fsrDown','isrUp','isrDown',
        'ueUp','ueDown',
        'mepsUp','mepsDown'
    ]
    
    pool = Pool(16)
    pool.map(templateDataFrames, variations)