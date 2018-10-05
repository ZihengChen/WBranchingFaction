import utility_common as common
from utility_dfcutter import *
from utility_bfsolver import Yield

class DFCounter():
    def __init__(self,trigger,usetag):
        self.variation = ""

        self.trigger = trigger
        self.usetag  = usetag

        self._setConfiguration(trigger,usetag)


    def setVariation(self,variation):
        self.variation = variation
        self._setConfiguration(self.trigger,self.usetag)

    #############################
    ## Given trigger,usetag
    #############################

    def returnNData(self):
        n,nVar = [],[]
        for slt in self.selections:
            temp,tempVar = self.getNData(slt,self.nbjet)
            n.append(temp)
            nVar.append(tempVar)
        n = np.array(n)
        nVar = np.array(nVar)
        return n,nVar # 4x1,4x1

    def returnNFake(self):
        n,nVar = [],[]
        for slt in self.selections:
            temp,tempVar = self.getNFake(slt,self.nbjet)
            n.append(temp)
            nVar.append(tempVar)
        n = np.array(n)
        nVar = np.array(nVar)
        return n,nVar # 4x1,4x1
    
    def returnNMCbg(self):
        n,nVar = [],[]
        for slt in self.selections:
            temp,tempVar = self.getNMCbg(slt,self.nbjet)
            n.append(temp)
            nVar.append(tempVar)
        n = np.array(n)
        nVar = np.array(nVar)
        return n,nVar # 4x1,4x1

    def returnAcc(self, enhanceStat=True):
        acc,accVar = [],[] 
        for slt in self.selections:
            temp,tempVar = self.getAcc(slt,self.nbjet, enhanceStat)
            acc.append(temp)
            accVar.append(tempVar)
        acc = np.array(acc)
        accVar = np.array(accVar)
        return acc,accVar# 4x6x6,4x6x6


    def returnNMCsg(self):
        acc,accVar = [],[] 
        for slt in self.selections:
            temp,tempVar = self.getNMCsg(slt,self.nbjet)
            acc.append(temp)
            accVar.append(tempVar)
        acc = np.array(acc)
        accVar = np.array(accVar)
        return acc,accVar # 4x1,4x1


    #############################
    ## Given selection, nbjet
    #############################

    def getNData(self,selection,nbjet):
        df = DFCutter(selection,nbjet,"data2016").getDataFrame()
        n = np.sum(df.eventWeight)
        nVar = n
        return n, nVar
    
    def getNFake(self,selection,nbjet):


        if selection == "mu4j":
            fakeSF = common.getFakeSF('mu')

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame(self.variation)
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame(self.variation)
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2
        
        elif selection == "e4j":
            fakeSF = common.getFakeSF('e')

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame(self.variation)
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame(self.variation)
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2

        elif selection == "mutau":
            fakeSF = common.getFakeSF('tau')

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame(self.variation)
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame(self.variation)
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2

        elif selection == "etau":
            fakeSF = common.getFakeSF('tau')

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame(self.variation)
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame(self.variation)
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2


        else:
            n,nVar = 0,0
        return n, nVar

    def getNMCbg(self,selection,nbjet):
        n, nVar = [], []
        for name in ["mcdiboson","mcz","mcw"]:
            df = DFCutter(selection,nbjet,name).getDataFrame(self.variation) # get MC dataframe with variation
            n.append( np.sum(df.eventWeight) )
            nVar.append( np.sum(df.eventWeight**2) )

        return np.array(n), np.array(nVar)

    def getAcc(self,selection, nbjet, enhanceStat=True):

        # tW
        nGenMCt = self.dfNGen.query("name=='t'" ).ngen.values[0]
        df = DFCutter(selection,nbjet,"mct").getDataFrame(self.variation)
        nMCt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

        accMCt, accMCtVar = common.getEfficiency(nMCt, nGenMCt)


        # tt
        # variated tt
        if self.variation in ['FSRUp','FSRDown','ISRUp','ISRDown','UEUp','UEDown','MEPSUp','MEPSDown']:
            nGenMCtt = self.dfNGen[self.dfNGen.name=='ttbar_inclusive_'+self.variation].ngen.values[0]
            df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation)
            nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

            num = nMCtt 
            den = nGenMCtt

        # nominal tt
        else:
            # inclusive tt
            if enhanceStat:
                nGenMCtt = self.dfNGen[self.dfNGen.name=='tt'].ngen.values[0]
                df = DFCutter(selection,nbjet,'mctt').getDataFrame(self.variation)
                nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

                nGenMCtt_2l2nu = self.dfNGen[self.dfNGen.name=='tt_2l2nu'].ngen.values[0]
                df = DFCutter(selection,nbjet,'mctt_2l2nu').getDataFrame(self.variation)
                nMCtt_2l2nu = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

                nGenMCtt_semilepton = self.dfNGen[self.dfNGen.name=='tt_semilepton'].ngen.values[0]
                df = DFCutter(selection,nbjet,'mctt_semilepton').getDataFrame(self.variation)
                nMCtt_semilepton = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

                num = nMCtt + nMCtt_2l2nu + nMCtt_semilepton
                den = nGenMCtt + nGenMCtt_2l2nu + nGenMCtt_semilepton

            else:
                nGenMCtt = self.dfNGen[self.dfNGen.name=='tt'].ngen.values[0]
                df = DFCutter(selection,nbjet,'mctt').getDataFrame(self.variation)
                nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

                num = nMCtt 
                den = nGenMCtt


        accMCtt, accMCttVar = common.getEfficiency(num, den)


        # combine tt and tW
        acc = self.c_ttxs * accMCtt + self.c_txs * accMCt
        accVar = self.c_ttxs**2 * accMCttVar + self.c_txs**2 * accMCtVar

        # acc,accVar = accMCtt,accMCttVar

        acc = acc[self.arrayToMatrix]
        accVar = accVar[self.arrayToMatrix]

        return acc,accVar


    def getNMCsg(self,selection,nbjet):

        # tW
        df = DFCutter(selection,nbjet,"mct").getDataFrame(self.variation)
        nMCt, nMCtVar = np.sum(df.eventWeight), np.sum(df.eventWeight**2)
    
        # tt

        df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation)
        nMCtt,nMCttVar = np.sum(df.eventWeight), np.sum(df.eventWeight**2)


        return nMCt+nMCtt, nMCtVar+nMCttVar
        # return nMCtt,nMCttVar


    
    #############################
    ## private helper functions
    #############################

    def _countDataFrameByTauDecay(self, df, normToLumin=True, withWeights=True):

        yields = []
        for i in range(1,22,1):
            temp = df[ df.genCategory == i ]
            if withWeights:
                # withWeights=True, normToLumin=True
                countingWeight = temp.eventWeight
                if not normToLumin:
                    # withWeights=True, normToLumin=False
                    countingWeight = countingWeight/temp.eventWeightSF
                n = np.sum(countingWeight)
                #nVar = np.sum(countingWeight**2)

            else:
                # withWeights=False, normToLumin=False
                n = len(temp)
                #nVar = n
            
            if n is nan:
                n = 0
                #nVar = 0

            yields.append(n)
            #yieldsVar.append(nVar)

        # if withNVar:
        #     return np.array(yields),np.array(yieldsVar)
        # else:
        return np.array(yields)
    
    def _setConfiguration(self,trigger,usetag):
        self.baseDir = common.getBaseDirectory() 

        # config nbjet and selections
        if trigger == "mu":
            self.selections = ["emu","mumu","mutau","mu4j"]
        elif trigger == "e":
            self.selections = ["ee" ,"emu2","etau" , "e4j"]
        
        if usetag == "1b":
            self.nbjet = "==1"
        if usetag == "2b":
            self.nbjet = ">1"

        # read nGen from file
        dfNGen = pd.read_pickle(self.baseDir + "data/pickles/ngen.pkl")
        self.dfNGen   = dfNGen


        self.ttxs,self.txs = 832,35.85*2
        if self.variation == 'TTXSUp':
            self.ttxs = self.ttxs * 1.05
        if self.variation == 'TWXSUp':
            self.txs = self.txs * 1.05


        self.c_ttxs = self.ttxs/(self.ttxs+self.txs)
        self.c_txs  = self.txs /(self.ttxs+self.txs)

        self.arrayToMatrix = np.array([ 
                                [ 0, 2, 9,10,11,15],
                                [ 2, 1,12,13,14,16],
                                [ 9,12, 3, 5, 6,17],
                                [10,13, 5, 4, 7,18],
                                [11,14, 6, 7, 8,19],
                                [15,16,17,18,19,20]
                                ])




class DFCounter_selection():
    def __init__(self,selection,nbjet):
        self.variation = ""

        self.selection = selection
        self.nbjet  = nbjet

        self._setConfiguration()


    def setVariation(self,variation):
        self.variation = variation
        self._setConfiguration()

    #############################
    ## Given trigger,usetag
    #############################

    def returnNData(self):
        n,nVar = [],[]
    
        for slt in [None, self.workingPoint]:
            temp,tempVar = self.getNData(self.selection, self.nbjet, querySoftmax=slt)

            n.append(temp)
            nVar.append(tempVar)
        n = np.array(n)
        nVar = np.array(nVar)
        return n,nVar # 2x1,2x1
    
    def returnNMCbg(self):
        n,nVar = [],[]
        for slt in [None, self.workingPoint]:
            temp,tempVar = self.getNMCbg(self.selection, self.nbjet, querySoftmax=slt)
            n.append(temp)
            nVar.append(tempVar)
        n = np.array(n)
        nVar = np.array(nVar)
        return n,nVar # 2x1,2x1

    def returnAcc(self):
        acc,accVar = [],[] 
        for slt in [None, self.workingPoint]:
            temp,tempVar = self.getAcc(self.selection, self.nbjet, querySoftmax=slt)
            acc.append(temp)
            accVar.append(tempVar)
        acc = np.array(acc)
        accVar = np.array(accVar)
        return acc,accVar# 2x6x6,2x6x6


    #############################
    ## Given selection, nbjet
    #############################

    def getNData(self,selection,nbjet, querySoftmax=None):
        df = DFCutter(selection,nbjet,"data2016").getDataFrame(querySoftmax=querySoftmax)
        n = np.sum(df.eventWeight)
        nVar = n
        return n, nVar

    def getNMCbg(self,selection,nbjet, querySoftmax=None):
        n, nVar = 0.0, 0.0
        for name in ["mcdiboson","mcdy"]:
            df = DFCutter(selection,nbjet,name).getDataFrame(self.variation,querySoftmax=querySoftmax) # get MC dataframe with variation
            n    = n + np.sum(df.eventWeight)
            nVar = nVar + np.sum(df.eventWeight**2)
        return n, nVar

    def getAcc(self,selection,nbjet, querySoftmax=None):

        # tW
        nGenMCt = self.dfNGen.query("name=='t'" ).ngen.values[0]
        df = DFCutter(selection,nbjet,"mct").getDataFrame(self.variation,querySoftmax=querySoftmax)
        nMCt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

        accMCt, accMCtVar = common.getEfficiency(nMCt, nGenMCt)


        # tt
        # variated tt
        if self.variation in ['FSRUp','FSRDown','ISRUp','ISRDown','UEUp','UEDown','MEPSUp','MEPSDown']:
            nGenMCtt = self.dfNGen[self.dfNGen.name=='ttbar_inclusive_'+self.variation].ngen.values[0]
            df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation,querySoftmax=querySoftmax)
            nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

            num = nMCtt 
            den = nGenMCtt
        # nominal tt

        else: #self.variation == '' or ('pta' in self.variation) or ('pta' in self.variation) : 
            # inclusive tt
            nGenMCtt = self.dfNGen[self.dfNGen.name=='tt'].ngen.values[0]
            df = DFCutter(selection,nbjet,'mctt').getDataFrame(self.variation,querySoftmax=querySoftmax)
            nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

            nGenMCtt_2l2nu = self.dfNGen[self.dfNGen.name=='tt_2l2nu'].ngen.values[0]
            df = DFCutter(selection,nbjet,'mctt_2l2nu').getDataFrame(self.variation,querySoftmax=querySoftmax)
            nMCtt_2l2nu = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

            nGenMCtt_semilepton = self.dfNGen[self.dfNGen.name=='tt_semilepton'].ngen.values[0]
            df = DFCutter(selection,nbjet,'mctt_semilepton').getDataFrame(self.variation,querySoftmax=querySoftmax)
            nMCtt_semilepton = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

            num = nMCtt + nMCtt_2l2nu + nMCtt_semilepton
            den = nGenMCtt + nGenMCtt_2l2nu + nGenMCtt_semilepton

        accMCtt, accMCttVar = common.getEfficiency(num, den)


        # combine tt and tW
        acc = self.c_ttxs * accMCtt + self.c_txs * accMCt
        accVar = self.c_ttxs**2 * accMCttVar + self.c_txs**2 * accMCtVar

        acc = acc[self.arrayToMatrix]
        accVar = accVar[self.arrayToMatrix]

        return acc,accVar

    
    #############################
    ## private helper functions
    #############################

    def _countDataFrameByTauDecay(self, df, normToLumin=True, withWeights=True):

        yields = []
        for i in range(1,22,1):
            temp = df[ df.genCategory == i ]
            if withWeights:
                # withWeights=True, normToLumin=True
                countingWeight = temp.eventWeight
                if not normToLumin:
                    # withWeights=True, normToLumin=False
                    countingWeight = countingWeight/temp.eventWeightSF
                n = np.sum(countingWeight)

            else:
                # withWeights=False, normToLumin=False
                n = len(temp)
            
            if n is nan:
                n = 0
            yields.append(n)
            
        return np.array(yields)
    
    def _setConfiguration(self):
        self.baseDir = common.getBaseDirectory() 

        if self.selection =="mumu":
            if self.nbjet == '==1': 
                self.workingPoint = 0.07
            if self.nbjet == '>1': 
                self.workingPoint = 0.07
        
        if self.selection =="ee":
            if self.nbjet == '==1': 
                self.workingPoint = 0.05
            if self.nbjet == '>1': 
                self.workingPoint = 0.2


        # read nGen from file
        dfNGen = pd.read_pickle(self.baseDir + "data/pickles/ngen.pkl")
        self.dfNGen   = dfNGen


        self.ttxs,self.txs = 832,35.85*2
        if self.variation == 'TTXSUp':
            self.ttxs = self.ttxs * 1.05
        if self.variation == 'TWXSUp':
            self.txs = self.txs * 1.05


        self.c_ttxs = self.ttxs/(self.ttxs+self.txs)
        self.c_txs  = self.txs /(self.ttxs+self.txs)

        self.arrayToMatrix = np.array([ 
                                [ 0, 2, 9,10,11,15],
                                [ 2, 1,12,13,14,16],
                                [ 9,12, 3, 5, 6,17],
                                [10,13, 5, 4, 7,18],
                                [11,14, 6, 7, 8,19],
                                [15,16,17,18,19,20]
                                ])



def summaryCounts():
    counts = {}

    cols = ['acc', 'accVar', 'nmcbg', 'nmcbgVar', 'nfake','nfakeVar', 'ndata', 'ndataVar']


    # nominal count
    df0 = pd.read_pickle( common.getBaseDirectory()  + "data/counts/count_.pkl")
    count0 = []
    for c in cols:
        count0.append(np.r_[ df0[c][0],df0[c][1],df0[c][2],df0[c][3]])
    counts['nominal'] = count0
            
    # variate up 
    for variation in ['TTXS','TWXS']:
        
        dfup = pd.read_pickle( common.getBaseDirectory()  + "data/counts/count_{}Up.pkl".format(variation))
        count = []
        for c in cols:
            tempup = np.r_[ dfup[c][0], dfup[c][1], dfup[c][2], dfup[c][3]]
            temp0  = np.r_[ df0[c][0],  df0[c][1],  df0[c][2],  df0[c][3]]
            count.append(tempup - temp0)
        counts[variation] = count

    # variate down
    for variation in ['EPt','MuPt','TauPt']:
        
        dfdw = pd.read_pickle( common.getBaseDirectory()  + "data/counts/count_{}Down.pkl".format(variation))
        count = []
        for c in cols:
            tempdw = np.r_[ dfdw[c][0], dfdw[c][1], dfdw[c][2], dfdw[c][3]]
            temp0  = np.r_[ df0[c][0],  df0[c][1],  df0[c][2],  df0[c][3]]
            count.append(temp0 - tempdw)
        counts[variation] = count
        
    # variate up-down
    for variation in ['JES','JER','BTag',"Mistag"]:
        
        dfdw = pd.read_pickle( common.getBaseDirectory()  + "data/counts/count_{}Down.pkl".format(variation))
        dfup = pd.read_pickle( common.getBaseDirectory()  + "data/counts/count_{}Up.pkl".format(variation))
        count = []
        for c in cols:
            tempdw = np.r_[ dfdw[c][0], dfdw[c][1], dfdw[c][2], dfdw[c][3]]
            tempup = np.r_[ dfup[c][0], dfup[c][1], dfup[c][2], dfup[c][3]]
            count.append( (tempup-tempdw)/2 )
        counts[variation] = count

    
    # lepton eff
    for variation in ['EffE','EffMu','EffTau']:
        if variation == 'EffE':
            effup = 1.01
            effup = np.array([effup,1,1,1]*2+[effup**2,effup,effup,effup]*2)
        if variation == 'EffMu':
            effup = 1.01
            effup = np.array([effup,effup**2,effup,effup]*2+[1,effup,1,1]*2)
        if variation == 'EffTau':
            effup = 1.05
            effup = np.array([1,1,effup,1]*2+[1,1,effup,1]*2)

        count = []
        for c in cols:
            temp0  = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
            if c=='acc':
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]] * effup.reshape(16,1,1)
            elif c=='nmcbg':
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]] * effup.reshape(16,1)
            else:
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
                
            count.append( tempup-temp0 )
        counts[variation] = count

    # background xs
    for variation in ['VVXS','ZXS','WXS']:
        if variation == 'VVXS':
            effup = np.array([[1.1, 1, 1]])
        if variation == 'EffMu':
            effup = np.array([[1, 1.05, 1]])
        if variation == 'EffTau':
            effup = np.array([[1, 1, 1.05]])

        count = []
        for c in cols:
            temp0  = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
            if c == 'nmcbg':
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]] * effup
            else:
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
            count.append( tempup-temp0 )
            
        counts[variation] = count
    

    # fake rate
    for variation in ['fakee','fakemu','faketau']:
        if variation == 'fakee':
            effup = 1.25
            effup = np.array([1,1,1,1]*2+ [1,1,1,effup]*2)
        if variation == 'fakemu':
            effup = 1.25
            effup = np.array([1,1,1,effup]*2+[1,1,1,1]*2)
        if variation == 'faketau':
            effup = 1.25
            effup = np.array([1,1,effup,1]*4)

        count = []
        for c in cols:
            temp0  = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
            if c == 'nfake':
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]] * effup
            else:
                tempup = np.r_[df0[c][0],df0[c][1],df0[c][2],df0[c][3]]
            count.append( tempup-temp0 )
        counts[variation] = count



    counts = pd.DataFrame.from_dict(counts,orient='index',columns=cols)

    return counts



def plotCountsSummary(count0):
    v = np.array([0,0,1,1,1,0])

    mask2 = np.outer(v,v)
    mask1 = np.outer(v+1,v+1) - np.ones([6,6]) - 3*mask2
    mask0 = np.ones([6,6]) - mask1 - mask2



    allYield = []
    for i in range(16):
        #count0 = fitter.count0

        ndata = count0.ndata[i]
        nmcvv = count0.nmcbg[i,0]
        nmcz  = count0.nmcbg[i,1]
        nmcw  = count0.nmcbg[i,2]
        nfake = count0.nfake[i]
        
        a = count0.acc[i]
        
        nmcsg0 = Yield(a*mask0,0,0,xs=832+35.85*2,lumin=35847,bte=.1785,btm=.1736).predict(BW=np.array([0.1080]*3))
        nmcsg1 = Yield(a*mask1,0,0,xs=832+35.85*2,lumin=35847,bte=.1785,btm=.1736).predict(BW=np.array([0.1080]*3))
        nmcsg2 = Yield(a*mask2,0,0,xs=832+35.85*2,lumin=35847,bte=.1785,btm=.1736).predict(BW=np.array([0.1080]*3))
        
        allYield.append([nfake,nmcvv,nmcz,nmcw,nmcsg0,nmcsg1,nmcsg2,ndata])
    allYield = np.array(allYield)




    labelList = ['QCD','VV','Z','W',r'tt/tW-0$\tau$',r'tt/tW-1$\tau$',r'tt/tW-2$\tau$']
    colorList = ["grey","#a32020", "#e0301e", "#eb8c00", "#49feec", "deepskyblue", "mediumpurple","k"]
    xlabelList = [r"$\mu e $",r"$\mu \mu $",r"$\mu \tau $",r"$\mu j $",
                r"$\mu e $",r"$\mu \mu $",r"$\mu \tau $",r"$\mu j $",
                r"$e e $",r"$e \mu $",r"$e \tau $",r"$e j $",
                r"$e e $",r"$e \mu $",r"$e \tau $",r"$e j $"
                ]
    plt.figure(figsize=(15,5),facecolor='w')
    centers = np.arange(4)
    centers = np.r_[centers,centers+5,centers+10,centers+15]




    ndata = allYield[:,-1]
    plt.errorbar(centers,ndata,yerr=ndata**0.5,color="k", fmt='.',markersize=10)

    handles = []

    for p in range(0,7):
        pbottom = 0
        if p > 0:
            pbottom = np.sum(allYield[:,:p],axis=1)
        h = plt.bar(centers,allYield[:,p],
                    width =0.8,
                    bottom=pbottom,
                    label=labelList[p],
                    color=colorList[p])
        handles.append(h)
    for s in [4,9,14,19]:
        plt.axvline(s,linestyle='--',color='grey')
        

    plt.yscale('log')
    plt.xticks(centers,xlabelList)
    plt.xlim(-1,21.5)
    plt.ylim(1,5e6)
    plt.text(0.5, 1e6, r'$\mu - 1b$',fontsize=15)
    plt.text(5.5, 1e6, r'$\mu - 2b$',fontsize=15)
    plt.text(10.5, 1e6, r'$e - 1b$',fontsize=15)
    plt.text(15.5, 1e6, r'$e - 2b$',fontsize=15)
    plt.grid(axis='y',linestyle='--',color='grey',alpha=0.5)
    plt.title("L=35.9/fb (13TeV)",loc="right")


    plt.legend(loc='upper right', handles=handles[::-1])
    plt.savefig('../plots/yields.png',dpi=300)