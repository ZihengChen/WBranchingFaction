from pylab import *
from utility_bfsolver3D import *
from tqdm import tqdm, trange

class BFCalc3D_Error:
    def __init__(self):
        self.tb = BFCalc3D_Toolbox()
    
    def errStat(self,npoints=100,errSource="data"):
    
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:

                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                bf = BFCalc3D_ThreeSelectorRatios(a)
                
                temp = []

                for i in trange(npoints,desc=trigger+'-'+tag):
                    if errSource == "data":
                        smear = np.random.normal(0,ndataVar**0.5)
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata+smear, nMcbg=nmcbg+nfake))
                    elif errSource == "mcbg":
                        smear = np.random.normal(0,nmcbgVar**0.5)
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                    elif errSource == "fake":
                        smear = np.random.normal(0,nfakeVar**0.5)
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                       
                    elif errSource == "mcsg":
                        smear = self.tb.SmearAcc(a,aVar)
                        bftemp = BFCalc3D_ThreeSelectorRatios(a+smear)
                        BW = bf.SolveQuadEqn(bftemp.PredictX())                        
                    else:
                        print("invalid stat err source")
                    temp.append(BW)

                errs.append(np.array(temp).std(axis=0))

        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5
    
    def errSystem_CrossSection(self,errSource="mcbg"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                bf0 = BFCalc3D_ThreeSelectorRatios(a)
                if errSource == "mcbg":
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    BW = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=1.05*nmcbg+nfake))
                elif errSource == "fake":
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    BW = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake*1.15))
                elif errSource == "mcsg":
                    BW0= bf0.SolveQuadEqn(bf0.PredictX())
                    bftemp = BFCalc3D_ThreeSelectorRatios(1.05*a)
                    BW = bf0.SolveQuadEqn(bftemp.PredictX())
                else:
                    print("invalid stat err source")
                errs.append(BW-BW0)                
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5

    def errSystem_ObjectEff(self,errSource="e"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                bf0 = BFCalc3D_ThreeSelectorRatios(a)
                if errSource == "e":
                    effUp = np.array([0.02,0,0,0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                elif errSource == "mu":
                    effUp = np.array([0.0,0.02,0,0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                elif errSource == "tau":
                    effUp = np.array([0,0,0.05,0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                else:
                    print("invalid stat err source")
                errs.append(BW-BW0)                
        
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5


    def errSystem_TTTheory(self,errSource="isrup"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                

                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                bf0 = BFCalc3D_ThreeSelectorRatios(a)
                BW0 = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                atemp,atempVar = self.tb.GetAcc(trigger,tag,errSource)
                bftemp = BFCalc3D_ThreeSelectorRatios( atemp )
                BW = bftemp.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                errs.append(BW-BW0)
                
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5

    # def temp(self):
    #     errs = []
    #     for trigger in ["mu","e"]:
    #         for tag in ["1b","2b"]:
    #             a,aVar  = self.tb.GetAcc(trigger,tag)
    #             ndata,ndataVar = self.tb.GetNData(trigger,tag)
    #             nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
    #             nfake,nfakeVar = self.tb.GetNFake(trigger,tag)   
        
    #     errs = np.array(errs)
    #     return errs, 1/np.sum(1/errs**2,axis=0)**0.5