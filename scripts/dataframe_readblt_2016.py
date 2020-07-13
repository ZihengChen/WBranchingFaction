#!/usr/bin/env python

from utility_bltreader_2016 import *
from multiprocessing import Pool

outputFolder = "pickles_2016"

def run(configs):
    filename, slt, fileType = configs
    rd = BLTReader(filename, slt, fileType,  outputFolder)
    rd.readBLT()

if __name__ == '__main__':
    
    selections = ["ee","mumu","emu",
              "mutau","etau","mutau_fakes","etau_fakes",
              "mu4j","mu4j_fakes","e4j","e4j_fakes"]

    filename = "Run2016_20200712.root" #golden
    ee = BLTReader(filename, selection="ee", inputRootType="",outputFolder=outputFolder)
    ee.outputNGen()
    configs = [(filename, slt, "") for slt in selections]
    

    # fiilename_mcttsys = "Run2016_20200525.root"
    # ee = BLTReader(filename_mcttsys, selection="ee",inputRootType="mcttsys",outputFolder=outputFolder)
    # ee.outputNGen()
    # configs += [(filename_mcttsys, slt, "mcttsys") for slt in selections]

    pool = Pool(16)
    pool.map(run, configs)
