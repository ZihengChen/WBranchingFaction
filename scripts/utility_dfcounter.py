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



