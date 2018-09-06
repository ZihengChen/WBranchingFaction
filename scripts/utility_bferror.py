import utility_common as common
from pylab import *
from utility_bfsolver import *
from tqdm import tqdm, trange

class BFSovler3D_Error:
    def __init__(self, statCombined=False):
        self.tb = BFSolver_Toolbox()
        self.baseDir = common.getBaseDirectory() 

        counts = pd.read_pickle( self.baseDir + "data/counts/count_.pkl")
        
        self.a, self.aVar = counts.acc, counts.accVar
        self.ndata, self.ndataVar = counts.ndata, counts.ndataVar
        self.nmcbg, self.nmcbgVar = counts.nmcbg, counts.nmcbgVar
        self.nfake, self.nfakeVar = counts.nfake, counts.nfakeVar

        self.statCombined = statCombined
        if self.statCombined:
            self.setStatCombinedWeights()

    def setStatCombinedWeights (self):
        statVar  = self.errStat('data')**2
        statVar += self.errStat('mcbg')**2
        statVar += self.errStat('mcsg')**2

        w = 1/statVar

        self.cWeight = w/np.sum(w,axis=0)
        self.statErr = 1/np.sum(w,axis=0)**0.5

        BW = 0
        for icata in range(4):
            a  = self.a[icata]
            ndata = self.ndata[icata]
            nmcbg = self.nmcbg[icata]
            nfake = self.nfake[icata]
            w  = self.cWeight[icata]

            slv = BFSolver3D(a)
            BW  += slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake)) * w
        self.BW = BW

        print("{:6.4f}+/-{:6.4f}, {:6.4f}+/-{:6.4f}, {:6.4f}+/-{:6.4f}".format( self.BW[0], self.statErr[0],
                                                                                self.BW[1], self.statErr[1], 
                                                                                self.BW[2], self.statErr[2]
                                                                                ))

        
        
    def errStat(self, errSource):
    
        errs = []
        for icata in range(4):
            a,aVar  = self.a[icata], self.aVar[icata]
            ndata,ndataVar = self.ndata[icata],self.ndataVar[icata]
            nmcbg,nmcbgVar = self.nmcbg[icata],self.nmcbgVar[icata]
            nfake,nfakeVar = self.nfake[icata],self.nfakeVar[icata]

            slv = BFSolver3D(a)
            BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

            ## data: by err propagation
            if errSource == "data":

                dBW = []
                for c in range(4):
                    # variate ndata to ndata1
                    ndata1 = ndata.copy()
                    ndata1[c] = ndata[c] + ndataVar[c]**0.5
                    # get BW1 corresponding to ndata1
                    BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata1, nMcbg=nmcbg+nfake))
                    # push deriveratives
                    dBW.append( BW1-BW )
                dBW = np.array(dBW)
                # propagating error
                errFromSource = np.sum(dBW**2,axis=0)**0.5
            
            ## mcbg: by std of toys which variate mcbg
            elif errSource == "mcbg":
            
                dBW = []
                for c in range(4):
                    # variate nmcbg to nmcbg1
                    nmcbg1 = nmcbg.copy()
                    nmcbg1[c] = nmcbg[c]+nmcbgVar[c]**0.5
                    # get BW1 corresponding to nmcbg1
                    BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg1+nfake))
                    # push deriveratives
                    dBW.append( BW1-BW )
                dBW = np.array(dBW)
                # propagating error
                errFromSource = np.sum(dBW**2,axis=0)**0.5

            ## mcbg: by std of toys which variate mcbg
            elif errSource == "fake":
                dBW = []
                for c in range(4):
                    # variate nfake to nfake1
                    nfake1 = nfake.copy()
                    nfake1[c] = nfake[c]+nfakeVar[c]**0.5
                    # get BW1 corresponding to nfake1
                    BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake1))
                    # push deriveratives
                    dBW.append( BW1-BW )
                dBW = np.array(dBW)
                # propagating error
                errFromSource = np.sum(dBW**2,axis=0)**0.5

            elif errSource == "mcsg":
                dBW = []
                for c in range(4):
                    for i in range(6):
                        for j in range(6):
                            # variate a to a1
                            if i == j and a[c,i,j]>0: 
                                # variate a to a1
                                a1 = a.copy()
                                a1[c,i,j] = a[c,i,j] + aVar[c,i,j]**0.5
                                # get BW1 corresponding to a1
                                slv1 = BFSolver3D(a1)
                                BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                                dBW.append( BW1-BW )
                
                            if i < j and a[c,i,j]>0:
                                # variate a to a1
                                a1 = a.copy()
                                a1[c,i,j] = a[c,i,j] + aVar[c,i,j]**0.5
                                a1[c,j,i] = a[c,j,i] + aVar[c,j,i]**0.5
                                # get BW1 corresponding to a1
                                slv1 = BFSolver3D(a1)
                                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                                dBW.append( BW1-BW )
                dBW = np.array(dBW)
                # propagating error
                errFromSource = np.sum(dBW**2,axis=0)**0.5

            else:
                print("invalid stat err source")
                
            errs.append(errFromSource)

        errs = np.array(errs)
        return errs

    def errConstent(self, errSource):

        errs = []
        for icata in range(4):
            a,aVar  = self.a[icata], self.aVar[icata]
            ndata,ndataVar = self.ndata[icata],self.ndataVar[icata]
            nmcbg,nmcbgVar = self.nmcbg[icata],self.nmcbgVar[icata]
            nfake,nfakeVar = self.nfake[icata],self.nfakeVar[icata]

            slv = BFSolver3D(a)
            BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

            if errSource =='BtmUp':
                slv1 = BFSolver3D(a,btm=0.1736+0.0005)
            if errSource =='BteUp':
                slv1 = BFSolver3D(a,bte=0.1785+0.0004)


            BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

            errs.append(BW1-BW)
                 
        errs = np.array(errs) # 4x3 2D-array
        if self.statCombined:
            errs = np.sum(errs * self.cWeight, axis=0 ) # 3 1D-array
        return errs


    
    def errSystem_crossSection(self, errSource):
        errs = []
        for icata in range(4):

            a     = self.a[icata]
            ndata = self.ndata[icata]
            nmcbg = self.nmcbg[icata]
            nfake = self.nfake[icata]

            slv = BFSolver3D(a)
            if errSource == "mcbg":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=1.05*nmcbg+nfake))
            elif errSource == "fakemu":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                variateScale = np.array([1.,1.,1.,1.])
                if icata in [0,1]:
                    variateScale = np.array([1,1,1,1.30])
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake*variateScale))
            
            elif errSource == "fakee":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                variateScale = np.array([1.,1.,1.,1.])
                if icata in [2,3]:
                    variateScale = np.array([1,1,1,1.30])
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake*variateScale))

            elif errSource == "faketau":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                variateScale = np.array([1.,1.,1.30,1.])
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake*variateScale))

            elif errSource == "lumin":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg*1.025+nfake))

            elif errSource == "mctt":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                
                counts1 = pd.read_pickle(self.baseDir + "data/counts/count_{}.pkl".format("TTXSUp"))
                slv1 = BFSolver3D(counts1.acc[icata])
                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
            elif errSource == "mctw":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                
                counts1 = pd.read_pickle(self.baseDir + "data/counts/count_{}.pkl".format("TWXSUp"))
                slv1 = BFSolver3D(counts1.acc[icata])
                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

            else:
                print("invalid stat err source")
            errs.append(BW1-BW)  
                         
        errs = np.array(errs) # 4x3 2D-array
        if self.statCombined:
            errs = np.sum(errs * self.cWeight, axis=0 ) # 3 1D-array
        return errs



    def errSystem_objectEff(self,errSource):
        # measured based on 20% per MisID from jet per 100GeV
        # df = DFCutter('etau','>1',"mctt").getDataFrame()
        # np.sum( df.eventWeight*(1+0.002*df.lepton2_pt) )/ np.sum(df.eventWeight)

        jetMisTauIDErrList = [1.07958,1.07994,1.07973, 1.07965]
        errs = []
        for icata in range(4):
            

            a     = self.a[icata]
            ndata = self.ndata[icata]
            nmcbg = self.nmcbg[icata]
            nfake = self.nfake[icata]

            slv = BFSolver3D(a)
            BW = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

            
                
            if errSource == "e":

                effUp = np.array([0.01,0,0,0]) + 1
                ## tuning up a
                slv1 = BFSolver3D( effUp[:,None,None]*a )
                ## tuning up nmcbg
                BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                
                errs.append(BW1-BW)  

            elif errSource == "mu":

                effUp = np.array([0,0.01,0,0]) + 1
                ## tuning up a
                slv1 = BFSolver3D( effUp[:,None,None]*a )
                ## tuning up nmcbg
                BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                
                errs.append(BW1-BW)


            elif errSource == "tauID":
                a1 = a.copy()

                if icata in [0,1]:
                    trigger = 1
                if icata in [2,3]:
                    trigger = 0

                a1[2,trigger,4] = a[2,trigger,4]*1.05
                a1[2,4,trigger] = a[2,4,trigger]*1.05

                slv1 = BFSolver3D(a1)
                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                errs.append(BW1-BW)


            elif errSource == "jetMisTauID":
                jetMisTauIDErr = jetMisTauIDErrList[icata]
                if icata in [0,1]:
                    trigger = 1
                if icata in [2,3]:
                    trigger = 0

                a1 = a.copy()
                a1[2,trigger,5] = a[2,trigger,5]*jetMisTauIDErr
                a1[2,5,trigger] = a[2,5,trigger]*jetMisTauIDErr

                slv1 = BFSolver3D(a1)
                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                errs.append(BW1-BW)


            else:
                print("invalid stat err source")

            
        errs = np.array(errs) # 4x3 2D-array
        if self.statCombined:
            errs = np.sum(errs * self.cWeight, axis=0 ) # 3 1D-array
        return errs


    def errSystem_energyScale(self,errSource="E"):
        errs = []

        counts1 = pd.read_pickle(self.baseDir + "data/counts/count_{}.pkl".format(errSource+"PtDown"))

        for icata in range(4):
            
            # nominal tuning
            slv  = BFSolver3D(self.a[icata])
            BW   = slv.solveQuadEqn(slv.setMeasuredX(nData=self.ndata[icata], nMcbg=self.nmcbg[icata]+self.nfake[icata]))
            # down tuning
            slv1 = BFSolver3D( counts1.acc[icata] )
            BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=counts1.ndata[icata], nMcbg=counts1.nmcbg[icata]+counts1.nfake[icata]))
            # difference between down and nominal
            errs.append(BW1-BW) 

        errs = np.array(errs) # 4x3 2D-array
        if self.statCombined:
            errs = np.sum(errs * self.cWeight, axis=0 ) # 3 1D-array
        return errs


    def errSystem_upDownVariation(self,errSource="JES"):
        '''
        "ISR","FSR","UE","MEPS","JES","JER","BTag","Mistag","Renorm","Factor","PDF"
        '''

        counts1 = pd.read_pickle(self.baseDir + "data/counts/count_{}.pkl".format(errSource+"Up"))
        counts2 = pd.read_pickle(self.baseDir + "data/counts/count_{}.pkl".format(errSource+"Down"))


        errs = []
        for icata in range(4):
                
            # up tuning
            slv1 = BFSolver3D( counts1.acc[icata] )
            BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=counts1.ndata[icata], nMcbg=counts1.nmcbg[icata]+counts1.nfake[icata]))
            # down tuning
            slv2 = BFSolver3D( counts2.acc[icata] )
            BW2  = slv2.solveQuadEqn(slv2.setMeasuredX(nData=counts2.ndata[icata], nMcbg=counts2.nmcbg[icata]+counts2.nfake[icata]))
            # differentce between up and down tuning
            errs.append((BW1-BW2)/2)
                
        errs = np.array(errs) # 4x3 2D-array
        if self.statCombined:
            errs = np.sum(errs * self.cWeight, axis=0 ) # 3 1D-array
        return errs


    def io_printErrorForExcelFormat(self,error):

        

        error = np.abs(error/0.1086 * 100)


        if self.statCombined:
            # print [source,br] matrix
            for i in range(error.shape[0]):
                print("{:5.3f},{:5.3f},{:5.3f}".format(error[i,0],error[i,1],error[i,2]))
        else:
            # print [cata,source,br] matrix
            for i in range(error.shape[1]):
                print("{:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}" \
                    .format(error[0,i,0],error[0,i,1],error[0,i,2],
                            error[1,i,0],error[1,i,1],error[1,i,2],
                            error[2,i,0],error[2,i,1],error[2,i,2],
                            error[3,i,0],error[3,i,1],error[3,i,2]
                        ))

    def smearAcc(self,a,aVar):
        smear = np.zeros_like(a)
        # loop over channels
        for slt in range(4): 
            # loop over hight
            for i in range(smear.shape[0]):
                # loop over width
                for j in range(smear.shape[1]):
                    # action on half element and above thrd = 0.001
                    if (i<=j) & (a[slt,i,j]>0.001):
                        smear[slt,i,j] = np.random.normal( 0, aVar[slt,i,j]**0.5)
                        if i != j :
                            smear[slt,j,i] = smear[slt,i,j]
        return smear
