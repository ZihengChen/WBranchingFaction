import utility_common as common
from utility_dfcutter import *


def countDataFrames(variation=""):
    labels  = ["trigger","usetag","acc","accVar","nmcbg","nmcbgVar","nfake","nfakeVar","ndata","ndataVar"]
    records = []
    for trigger in ["mu","e"]:
        for usetag in ["1b","2b"]:
            #print( "counting "+trigger+usetag + " ...")

            counter = DFCounter(trigger,usetag)
            counter.setVariation(variation)

            acc,accVar     = counter.returnAcc()
            nmcbg,nmcbgVar = counter.returnNMCbg()
            nfake,nfakeVar = counter.returnNFake()
            ndata,ndataVar = counter.returnNData()

            records.append( (trigger,usetag,acc,accVar,nmcbg,nmcbgVar,nfake,nfakeVar,ndata,ndataVar) )

    df = pd.DataFrame.from_records(records, columns=labels)
    df.to_pickle( common.dataDirectory() + "count/count_{}.pkl".format(variation))
    print( "counting finished!")
    
class DFCounter():
    def __init__(self,trigger,usetag):

        self.trigger = trigger
        self.usetag  = usetag
        self._setConfiguration(trigger,usetag)

    def setVariation(self,variation):
        self.variation = variation

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

    def returnAcc(self):
        acc,accVar = [],[]
        for slt in self.selections:
            temp,tempVar = self.getAcc(slt,self.nbjet)
            acc.append(temp)
            accVar.append(tempVar)
        acc = np.array(acc)
        accVar = np.array(accVar)
        return acc,accVar # 4x6x6,4x6x6


    #############################
    ## Given selection, nbjet
    #############################

    def getNData(self,selection,nbjet):
        df = DFCutter(selection,nbjet,"data2016").getDataFrame()
        n = np.sum(df.eventWeight)
        nVar = n
        return n, nVar
    
    def getNFake(self,selection,nbjet):

        if selection in ["mu4j"]:
            df = DFCutter(selection,nbjet,"data2016_inverseISO").getDataFrame()
            n = np.sum(df.eventWeight) * common.fakeRate()
            nVar = np.sum(df.eventWeight**2) * common.fakeRate()**2
        else:
            n,nVar = 0,0
        return n, nVar

    def getNMCbg(self,selection,nbjet):
        n, nVar = 0.0, 0.0
        for name in ["mcdiboson","mcdy"]:
            df = DFCutter(selection,nbjet,name).getDataFrame(self.variation) # get MC dataframe with variation
            n    = n + np.sum(df.eventWeight)
            nVar = nVar + np.sum(df.eventWeight**2)
        return n, nVar

    def getAcc(self,selection,nbjet):
        df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation)
        nMCtt = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

        df = DFCutter(selection,nbjet,"mct").getDataFrame(self.variation)
        nMCt  = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)

        accMCtt = nMCtt/self.nGenMCtt
        accMCt  = nMCt /self.nGenMCt

        accMCttVar = accMCtt * (1-accMCtt) / self.nGenMCtt
        accMCtVar  = accMCt  * (1-accMCt ) / self.nGenMCt 

        acc = self.c_ttxs * accMCtt + self.c_txs * accMCt
        accVar = self.c_ttxs**2 * accMCttVar + self.c_txs**2 * accMCtVar

        #if matrixFormat:
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
    
    def _setConfiguration(self,trigger,usetag):
        self.dataDirectry = common.dataDirectory()
        self.variation = ""

        # config nbjet and selections
        if trigger == "mu":
            self.selections = ["emu","mumu","mutau","mu4j"]
        elif trigger == "e":
            self.selections = ["ee","emu2","etau","e4j"]
        
        if usetag == "1b":
            self.nbjet = "==1"
        if usetag == "2b":
            self.nbjet = ">1"

        # read nGen from file
        dfNGen = pd.read_pickle(self.dataDirectry + "pickle/ngen.pkl")
        self.nGenMCt  = dfNGen.query("name=='t'" ).ngen.values[0]
        self.nGenMCtt = dfNGen.query("name=='tt'").ngen.values[0]

        self.ttxs,self.txs = 832,35.85*2
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

