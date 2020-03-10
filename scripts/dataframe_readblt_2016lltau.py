#!/usr/bin/env python

from utility_bltreader import *
from multiprocessing import Pool

def run(configs):
    filename, slt, fileType, folder = configs
    rd = BLTReader(filename, slt, fileType, folder)
    rd.readBLT()

if __name__ == '__main__':
    selections = ["eetau","mumutau","emutau"]


    filename_Tight = "study_TightTauFakeScaleFactor.root"
    folder_Tight = "pickles_TightTauMisid"
    filename_VTight = "study_VTightTauFakeScaleFactor.root"
    folder_VTight = "pickles_VTightTauMisid"

    configs = [(filename_Tight, slt, "", folder_Tight) for slt in selections]
    configs += [(filename_VTight, slt, "", folder_VTight) for slt in selections]


    pool = Pool(6)
    pool.map(run, configs)

