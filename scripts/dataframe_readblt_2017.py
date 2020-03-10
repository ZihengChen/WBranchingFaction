#!/usr/bin/env python

from utility_bltreader_2017 import *
from multiprocessing import Pool

outputFolder = 'pickles_2017'
def run(configs):
    filename, slt, fileType = configs
    rd = BLTReader(filename, slt, fileType, outputFolder)
    rd.readBLT()

if __name__ == '__main__':
    filename = "2017_20200305.root" #Run2017_20200107.root

    selections = ["ee","mumu","emu",
                "mutau","etau","mutau_fakes","etau_fakes",
                "mu4j","mu4j_fakes","e4j","e4j_fakes"]

    

    ee = BLTReader(filename, selection="ee", inputRootType="", outputFolder=outputFolder)
    ee.outputNGen()


    configs = [(filename, slt, "") for slt in selections]
    pool = Pool(16)
    pool.map(run, configs)