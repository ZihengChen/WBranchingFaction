#!/usr/bin/env python

from utility_bltreader import *
from multiprocessing import Pool

def run(configs):
    filename, slt = configs
    rd = BLTReader(filename, slt)
    rd.readBLT()

if __name__ == '__main__':
    
    filename = "20190805.root"
    ee = BLTReader(filename,"ee")
    ee.outputNGen()


    selections = ["ee","mumu","emu","mutau","etau","mutau_fakes","etau_fakes","mu4j","mu4j_fakes","e4j","e4j_fakes"]
    pool = Pool(4)
    pool.map(run, [(filename,slt) for slt in selections])
 

