#!/usr/bin/env python

from utility_dfplotter import *

def plotDataFrames(selections):
    
    for slt in selections:
        for nbjet in ["==1",">1"]:
            plotter = DFPlotter(slt,nbjet)
            plotter.plotKinematics()


if __name__ == '__main__':
    selections = ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]
    plotDataFrames(selections)


