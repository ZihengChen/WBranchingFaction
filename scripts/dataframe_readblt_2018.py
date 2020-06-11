#!/usr/bin/env python

from utility_bltreader_2018 import *
from multiprocessing import Pool

outputFolder = 'pickles_2018'
def run(configs):
    filename, slt, fileType = configs
    rd = BLTReader(filename, slt, fileType, outputFolder)
    rd.readBLT()

if __name__ == '__main__':
    filename = "Run2018_20200521.root"# "Run2018_20200112.root"

    selections = ["ee","mumu","emu",
                "mutau","etau","mutau_fakes","etau_fakes",
                "mu4j","mu4j_fakes","e4j","e4j_fakes"]

    

    ee = BLTReader(filename, selection="ee", inputRootType="", outputFolder=outputFolder)
    ee.outputNGen()


    configs = [(filename, slt, "") for slt in selections]
    pool = Pool(16)
    pool.map(run, configs)
