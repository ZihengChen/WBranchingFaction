#!/usr/bin/env python

from utility_dfplotter import *
from multiprocessing import Pool

folderOfPickles='pickles_2016'
# folderOfPickles='pickles_2017'
# folderOfPickles='pickles_2018'
def plotDataFrame(slt):
    plotter = DFPlotter(slt,"==1",njet=None, folderOfPickles=folderOfPickles)
    plotter.plotKinematics()
    plotter = DFPlotter(slt,">1", njet=None, folderOfPickles=folderOfPickles)
    plotter.plotKinematics()


if __name__ == '__main__':
    selections = ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]
    pool = Pool(8)
    pool.map(plotDataFrame, selections)
