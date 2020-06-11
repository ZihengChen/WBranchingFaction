#!/usr/bin/env python

from utility_bltreader_2016 import *
from multiprocessing import Pool

def run(configs):
    filename, slt, fileType, folder = configs
    rd = BLTReader(filename, slt, fileType, folder)
    rd.readBLT()

if __name__ == '__main__':
    selections = ["eemu","mumue","eetau","mumutau","emutau"]


    filename = "Run2016_llx_20200414.root"
    folder = "pickles_2016_llx_VTtau"
    configs = [(filename, slt, "", folder) for slt in selections]


    pool = Pool(5)
    pool.map(run, configs)
