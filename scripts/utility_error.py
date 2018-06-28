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
                ## data: by err propagation
                if errSource == "data":
                    BW0= bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    # taking derivertive dBW/dnData
                    deltaNData = np.identity(4)
                    dBW_over_dnData = []
                    for i in range(4):
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata+deltaNData[i], nMcbg=nmcbg+nfake))
                        dBW_over_dnData.append(BW-BW0)
                    dBW_over_dnData = np.array(dBW_over_dnData)
                    errFromSource = np.matmul(ndataVar,dBW_over_dnData**2, )**0.5
                
                ## mcbg: by std of toys which variate mcbg
                elif errSource == "mcbg":
                    temp = []
                    for i in trange(npoints,desc=errSource+'-'+trigger+'-'+tag):
                        smear = np.random.normal(0,nmcbgVar**0.5)
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                        temp.append(BW)
                    errFromSource = np.array(temp).std(axis=0)

                ## mcbg: by std of toys which variate mcbg
                elif errSource == "fake":
                    temp = []
                    for i in trange(npoints,desc=errSource+'-'+trigger+'-'+tag):
                        smear = np.random.normal(0,nfakeVar**0.5)
                        BW = bf.SolveQuadEqn(bf.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake+smear))
                        temp.append(BW)
                    errFromSource = np.array(temp).std(axis=0)

                elif errSource == "mcsg":
                    temp = []
                    for i in trange(npoints,desc=errSource+'-'+trigger+'-'+tag):
                        smear = self.tb.SmearAcc(a,aVar)
                        bftemp = BFCalc3D_ThreeSelectorRatios(a+smear)
                        BW = bf.SolveQuadEqn(bftemp.PredictX())    
                        temp.append(BW)  
                    errFromSource = np.array(temp).std(axis=0)

                else:
                    print("invalid stat err source")
                    
                
                errs.append(errFromSource)

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
                    effUp = np.array([0.01,0,0,0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                elif errSource == "mu":
                    effUp = np.array([0.0,0.01,0,0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                elif errSource == "tau":
                    effUp = np.array([0, 0, 0.05, 0]) + 1
                    BW0= bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))
                    #BW = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=effUp*ndata, nMcbg=nmcbg+nfake))
                    ## tuning up a
                    bftemp = BFCalc3D_ThreeSelectorRatios( effUp[:,None,None]*a )
                    ## tuning up nmcbg
                    BW = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=effUp*nmcbg+nfake))
                else:
                    print("invalid stat err source")
                errs.append(BW-BW0)                
        
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5


    def errSystem_TTTheory(self,errSource="isr"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                
                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                # bf0 = BFCalc3D_ThreeSelectorRatios(a)
                # BW0 = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                atemp,atempVar = self.tb.GetAcc(trigger,tag,errSource+"up")
                bftemp = BFCalc3D_ThreeSelectorRatios( atemp )
                BW1 = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))


                atemp,atempVar = self.tb.GetAcc(trigger,tag,errSource+"down")
                bftemp = BFCalc3D_ThreeSelectorRatios( atemp )
                BW2 = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                errs.append(np.abs(BW1-BW2)/2)
                
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5

    def errSystem_EnergyScale(self,errSource="e"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                
                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                bf0 = BFCalc3D_ThreeSelectorRatios(a)
                BW0 = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                ndataTemp,ndataVarTemp = self.tb.GetNData(trigger,tag,shiftEnergyScale=errSource)
                BW = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndataTemp, nMcbg=nmcbg+nfake))

                errs.append(BW-BW0) 

        errs = np.array(errs)
        return errs


    def errSystem_Jet(self,errSource="JES"):
        errs = []
        for trigger in ["mu","e"]:
            for tag in ["1b","2b"]:
                
                a,aVar  = self.tb.GetAcc(trigger,tag)
                ndata,ndataVar = self.tb.GetNData(trigger,tag)
                nmcbg,nmcbgVar = self.tb.GetNMcbg(trigger,tag)
                nfake,nfakeVar = self.tb.GetNFake(trigger,tag)

                # bf0 = BFCalc3D_ThreeSelectorRatios(a)
                # BW0 = bf0.SolveQuadEqn(bf0.SetMeasuredX(nData=ndata, nMcbg=nmcbg+nfake))

                atemp,atempVar = self.tb.GetAcc(trigger,tag,errSource+"Up")
                nmcbgtemp,nmcbgtempVar = self.tb.GetNMcbg(trigger,tag,shiftJet=errSource+"Up")
                bftemp = BFCalc3D_ThreeSelectorRatios( atemp )
                BW1 = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=nmcbgtemp+nfake))


                atemp,atempVar = self.tb.GetAcc(trigger,tag,errSource+"Down")
                nmcbgtemp,nmcbgtempVar = self.tb.GetNMcbg(trigger,tag,shiftJet=errSource+"Down")
                bftemp = BFCalc3D_ThreeSelectorRatios( atemp )
                BW2 = bftemp.SolveQuadEqn(bftemp.SetMeasuredX(nData=ndata, nMcbg=nmcbgtemp+nfake))

                errs.append(np.abs(BW1-BW2)/2)
                
        errs = np.array(errs)
        return errs#, 1/np.sum(1/errs**2,axis=0)**0.5

    def io_printErrorForExcelFormat(self,error):
        error = np.abs(err/0.1086 * 100)

        for i in range(error.shape[1]):
            print("{:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}, {:5.3f},{:5.3f},{:5.3f}" \
                .format(error[0,i,0],error[0,i,1],error[0,i,2],
                        error[1,i,0],error[1,i,1],error[1,i,2],
                        error[2,i,0],error[2,i,1],error[2,i,2],
                        error[3,i,0],error[3,i,1],error[3,i,2]
                    ))


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