#!/usr/bin/env python


import glob
import os, sys
import utility_common as common
from shutil import copyfile


if __name__ == '__main__':
    baseDir = common.getBaseDirectory() 

    for tr in ['e','mu']:
        for v in ['lepton1_eta','lepton1_pt','lepton2_eta','lepton2_pt']:
            files = glob.glob(baseDir + 'plots/kinematics/{}*/*/*{}*'.format(tr,v))
            dstDir = baseDir + 'plots/note_kinematics/'
            common.makeDirectory(dstDir,clear=False)

            for f in files:
                fdir, fname = os.path.split(f)
                copyfile(f, dstDir+fname)