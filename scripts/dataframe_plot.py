#!/usr/bin/env python

from utility_dfplotter import *
from multiprocessing import Pool

def plotDataFrame(slt):
    plotter = DFPlotter(slt,"==1")
    plotter.plotKinematics()
    plotter = DFPlotter(slt,">1")
    plotter.plotKinematics()


if __name__ == '__main__':
    selections = ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]

    pool = Pool(8)
    pool.map(plotDataFrame, selections)
    # for slt in selections:
    #     plotDataFrame(slt)