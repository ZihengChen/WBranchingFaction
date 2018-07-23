import utility_common as common
from utility_dfcutter import *


def countDataFrames(variation=""):
    labels  = ["trigger","usetag","acc","accVar","accs","accsVar","nmcbg","nmcbgVar","nfake","nfakeVar","ndata","ndataVar"]
    records = []
    for trigger in ["mu","e"]:
        for usetag in ["1b","2b"]:
            print( "counting "+trigger+usetag + " ...")

            counter = DFCounter(trigger,usetag)
            counter.setVariation(variation)

            acc,accVar,accs,accsVar = counter.returnAcc()
            nmcbg,nmcbgVar = counter.returnNMCbg()
            nfake,nfakeVar = counter.returnNFake()
            ndata,ndataVar = counter.returnNData()
 
            records.append( (trigger,usetag,acc,accVar,accs,accsVar,nmcbg,nmcbgVar,nfake,nfakeVar,ndata,ndataVar) )

    df = pd.DataFrame.from_records(records, columns=labels)
    df.to_pickle( common.dataDirectory() + "counts/count_{}.pkl".format(variation))
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
        accs,accsVar = [],[] # breakdown acc to tt,tw, tt_includsive, tt_2l2nu
        for slt in self.selections:
            temp,tempVar,temps,tempsVar = self.getAcc(slt,self.nbjet)
            acc.append(temp)
            accVar.append(tempVar)

            accs.append(temps)
            accsVar.append(tempsVar)
        acc = np.array(acc)
        accVar = np.array(accVar)
        accs = np.array(accs)
        accsVar = np.array(accsVar)
        return acc,accVar,accs,accsVar # 4x6x6,4x6x6


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
            fakeSF = common.muonFakeSF()

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame()
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame()
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2
        
        elif selection == "e4j":
            fakeSF = common.electronFakeSF()

            temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame()
            n    = np.sum(temp.eventWeight)
            nVar = np.sum(temp.eventWeight**2)
            for name in ['mcdiboson','mcdy','mct','mctt']:
                temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame()
                n    -= np.sum(temp.eventWeight)
                nVar += np.sum(temp.eventWeight**2)

            n *= fakeSF
            nVar *= fakeSF**2

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


        # tW
        df = DFCutter(selection,nbjet,"mct").getDataFrame(self.variation)
        nMCt  = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)
        accMCt  = nMCt /self.nGenMCt
        accMCtVar = accMCt  * (1-accMCt ) / self.nGenMCt 

        if self.variation in ['FSRUp','FSRDown','ISRUp','ISRDown','UEUp','UEDown','MEPSUp','MEPSDown']:
            nGenMCtt = self.dfNGen[self.dfNGen.name=='ttbar_inclusive_'+self.variation].ngen.values[0]

            df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation)
            nMCtt  = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)
            accMCtt = nMCtt/self.nGenMCtt
            accMCttVar = accMCtt * (1-accMCtt) / nGenMCtt

            accs = np.array([   accMCtt[self.arrayToMatrix], 
                                accMCt[self.arrayToMatrix]])

            accsVar = np.array([    accMCttVar[self.arrayToMatrix], 
                                    accMCtVar[self.arrayToMatrix]])
            
        else:

            # tt_inclusive
            df = DFCutter(selection,nbjet,"mctt").getDataFrame(self.variation)
            _nMCtt  = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)
            _accMCtt = _nMCtt/self.nGenMCtt
            _accMCttVar = _accMCtt * (1-_accMCtt) / self.nGenMCtt

            # tt_2l2nu
            df = DFCutter(selection,nbjet,"mctt_2l2nu").getDataFrame(self.variation)
            _nMCtt_2l2nu = self._countDataFrameByTauDecay(df, normToLumin=False, withWeights=True)
            _accMCtt_2l2nu = _nMCtt_2l2nu/self.nGenMCtt_2l2nu
            _accMCttVar_2l2nu = _accMCtt_2l2nu * (1-_accMCtt_2l2nu) / self.nGenMCtt_2l2nu

            # tt
            accMCtt = (_nMCtt + _nMCtt_2l2nu)/ ( self.nGenMCtt + self.nGenMCtt_2l2nu )
            accMCttVar = accMCtt * (1-accMCtt) / ( self.nGenMCtt + self.nGenMCtt_2l2nu )


            accs = np.array([   accMCtt[self.arrayToMatrix], 
                                accMCt[self.arrayToMatrix], 
                                _accMCtt[self.arrayToMatrix], 
                                _accMCtt_2l2nu[self.arrayToMatrix]])

            accsVar = np.array([    accMCttVar[self.arrayToMatrix], 
                                    accMCtVar[self.arrayToMatrix],
                                     _accMCttVar[self.arrayToMatrix], 
                                    _accMCttVar_2l2nu[self.arrayToMatrix]])
            

            

        acc = self.c_ttxs * accMCtt + self.c_txs * accMCt
        accVar = self.c_ttxs**2 * accMCttVar + self.c_txs**2 * accMCtVar

        #if matrixFormat:
        acc = acc[self.arrayToMatrix]
        accVar = accVar[self.arrayToMatrix]



        return acc,accVar,accs,accsVar


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
        dfNGen = pd.read_pickle(self.dataDirectry + "pickles/ngen.pkl")
        self.dfNGen = dfNGen
        self.nGenMCt  = dfNGen.query("name=='t'" ).ngen.values[0]
        self.nGenMCtt = dfNGen.query("name=='tt'").ngen.values[0]
        self.nGenMCtt_2l2nu = dfNGen.query("name=='tt_2l2nu'").ngen.values[0] + 1e-9


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

