#!/usr/bin/env python

from utility_bltreader import *
from multiprocessing import Pool

def run(configs):
    filename, slt, fileType = configs
    rd = BLTReader(filename, slt, fileType)
    rd.readBLT()

if __name__ == '__main__':
    filename = "approval_20191010.root"
    filename_mcttsys = "approval_ttsys.root"

    selections = ["ee","mumu","emu",
                "mutau","etau","mutau_fakes","etau_fakes",
                "mu4j","mu4j_fakes","e4j","e4j_fakes"]

    

    ee = BLTReader(filename, selection="ee", inputRootType="")
    ee.outputNGen()
    ee = BLTReader(filename_mcttsys, selection="ee",inputRootType="mcttsys")
    ee.outputNGen()


    configs = [(filename, slt, "") for slt in selections]
    configs += [(filename_mcttsys, slt, "mcttsys") for slt in selections]
    pool = Pool(16)
    pool.map(run, configs)


    
 

