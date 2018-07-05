import utility_common as common
from pylab import *
from utility_bfsolver import *
from tqdm import tqdm, trange

class BFSovler3D_Error:
    def __init__(self):
        self.tb = BFSolver_Toolbox()
        self.dataDir = common.dataDirectory() 

        counts = pd.read_pickle( self.dataDir + "count/count_.pkl")
        
        self.a, self.aVar = counts.acc, counts.accVar
        self.ndata, self.ndataVar = counts.ndata, counts.ndataVar
        self.nmcbg, self.nmcbgVar = counts.nmcbg, counts.nmcbgVar
        self.nfake, self.nfakeVar = counts.nfake, counts.nfakeVar
        
    
    def errStat(self, errSource, npoints=100):
    
        errs = []
        for icata in range(4):

            a,aVar  = self.a[icata], self.aVar[icata]
            ndata,ndataVar = self.ndata[icata],self.ndataVar[icata]
            nmcbg,nmcbgVar = self.nmcbg[icata],self.nmcbgVar[icata]
            nfake,nfakeVar = self.nfake[icata],self.nfakeVar[icata]

            slv = BFSolver3D(a)

            ## data: by err propagation
            if errSource == "data":
                BW = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                # taking derivertive dBW/dnData
                deltaNData = np.identity(4)
                dBW_over_dnData = []
                for i in range(4):
                    BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata+deltaNData[i], nMcbg=nmcbg+nfake))
                    dBW_over_dnData.append(BW1-BW)
                dBW_over_dnData = np.array(dBW_over_dnData)
                # propagating error
                errFromSource = np.matmul(ndataVar, dBW_over_dnData**2)**0.5
            
            ## mcbg: by std of toys which variate mcbg
            elif errSource == "mcbg":
                temp = []
                for i in trange( npoints, desc = errSource+'-'+str(icata)):
                    smear = np.random.normal(0, nmcbgVar**0.5)
                    BW1   = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                    temp.append(BW1)
                errFromSource = np.array(temp).std(axis=0)

            ## mcbg: by std of toys which variate mcbg
            elif errSource == "fake":
                temp = []
                for i in trange( npoints, desc=errSource+'-'+str(icata)):
                    smear = np.random.normal(0, nfakeVar**0.5)
                    BW1   = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                    temp.append(BW1)
                errFromSource = np.array(temp).std(axis=0)

            elif errSource == "mcsg":
                temp = []
                for i in trange( npoints, desc=errSource+'-'+str(icata)):
                    smear = self.smearAcc(a,aVar)
                    slv1  = BFSolver3D( a + smear)
                    BW1   = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))    
                    temp.append(BW1)  
                errFromSource = np.array(temp).std(axis=0)

            else:
                print("invalid stat err source")
                
            errs.append(errFromSource)

        errs = np.array(errs)
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
            elif errSource == "fake":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                BW1 = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake*1.15))
            elif errSource == "mcsg":
                BW  = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                slv1 = BFSolver3D(1.05*a)
                BW1 = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
            else:
                print("invalid stat err source")
            errs.append(BW1-BW)                
        errs = np.array(errs)
        return errs

    def errSystem_objectEff(self,errSource):
        errs = []
        for icata in range(4):

            a     = self.a[icata]
            ndata = self.ndata[icata]
            nmcbg = self.nmcbg[icata]
            nfake = self.nfake[icata]

            slv = BFSolver3D(a)
                
            if errSource == "e":
                effUp = np.array([0.01,0,0,0]) + 1
            elif errSource == "mu":
                effUp = np.array([0,0.01,0,0]) + 1
            elif errSource == "tau":
                effUp = np.array([0,0,0.05,0]) + 1
            else:
                print("invalid stat err source")

            BW = slv.solveQuadEqn(slv.setMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
            ## tuning up a
            slv1 = BFSolver3D( effUp[:,None,None]*a )
            ## tuning up nmcbg
            BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
            errs.append(BW1-BW)                
        
        errs = np.array(errs)
        return errs


    def errSystem_energyScale(self,errSource="e"):
        errs = []

        counts1 = pd.read_pickle(self.dataDir + "count/count_{}.pkl".format(errSource+"PtDown"))

        for icata in range(4):
            
            # nominal tuning
            slv  = BFSolver3D(self.a[icata])
            BW   = slv.solveQuadEqn(slv.setMeasuredX(nData=self.ndata[icata], nMcbg=self.nmcbg[icata]+self.nfake[icata]))
            # down tuning
            slv1 = BFSolver3D( counts1.acc[icata] )
            BW1  = slv1.solveQuadEqn(slv1.setMeasuredX(nData=counts1.ndata[icata], nMcbg=counts1.nmcbg[icata]+counts1.nfake[icata]))
            # difference between down and nominal
            errs.append(BW1-BW) 

        errs = np.array(errs)
        return errs


    def errSystem_upDownVariation(self,errSource="JES"):
        '''
        "ISR","FSR","UE","MEPS","JES","JER","BTag","Mistag","Renorm","Factor","PDF"
        '''


        counts1 = pd.read_pickle(self.dataDir + "count/count_{}.pkl".format(errSource+"Up"))
        counts2 = pd.read_pickle(self.dataDir + "count/count_{}.pkl".format(errSource+"Down"))


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
                
        errs = np.array(errs)
        return errs


    def io_printErrorForExcelFormat(self,error):
        error = np.abs(error/0.1086 * 100)

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
