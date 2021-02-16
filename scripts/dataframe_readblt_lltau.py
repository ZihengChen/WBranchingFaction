#!/usr/bin/env python

from utility_bltreader import *
from multiprocessing import Pool

def run(configs):
    filename, slt, fileType, folder = configs
    rd = BLTReader(filename, slt, fileType, folder)
    rd.readBLT()

if __name__ == '__main__':
    selections = ["eetau","mumutau","emutau"]


    filename = "Run2016_lltauTight_20200908.root"
    folder = "pickles_lltauTight"
    configs = [(filename, slt, "", folder) for slt in selections]


    pool = Pool(3)
    pool.map(run, configs)
