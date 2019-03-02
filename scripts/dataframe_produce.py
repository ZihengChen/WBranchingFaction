#!/usr/bin/env python

from utility_bltreader import *
import multiprocessing as mp

def runMP(filename, selections):
    
    processes = []
    
    for slt in selections:
        rd = BLTReader(filename, slt)
        processes.append(mp.Process(target=rd.readBLT))
    
    for ps in processes:
        ps.start()
    for ps in processes:
        ps.join()

if __name__ == '__main__':
    filename = "arc.root"

    ee = BLTReader(filename,"ee")
    ee.outputNGen()

    runMP(filename,["ee","mumu","emu","mutau","etau"])
    runMP(filename,["mu4j","mu4j_fakes","e4j","e4j_fakes"])


