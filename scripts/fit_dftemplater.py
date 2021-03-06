import utility_common as common
from utility_dfcutter import *


class DFTemplater():
    def __init__(self, variation='',
                 signalRegion = True
                 ):
                 
        self.varation = variation
        self.controlRegion = not signalRegion
        self.signalRegion  = signalRegion


    def makeTemplatesAndTargets(self):
        
        templatesShp, templatesCnt = [],[]
        targetsShp,   targetsCnt   = [],[]
        
        #####################
        # signal region
        #####################
        if self.signalRegion:
            for nbjet in ['==1','>1']:
                for selection in  ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:
                    # config
                    njet = None
                    v,bins = self._channelConfig(selection, nbjet, njet)

                    # templates
                    chTemplatesShp,chTemplatesCnt = self._makeChTemp(selection, nbjet, njet, v, bins)
                    templatesShp.append(chTemplatesShp)
                    templatesCnt.append(chTemplatesCnt)

                    if self.varation == '':
                        # targets
                        df = DFCutter(selection,nbjet,'data2016',njet).getDataFrame()
                        tempShp, tempCnt = self._binDataFrame(df,v,bins)
                        targetsShp.append(tempShp)
                        targetsCnt.append(tempCnt)

        #####################
        # control region
        #####################
        if self.controlRegion:
            for selection in ["mumuc","eec","mutau","etau"]: #
                # config 
                nbjet,njet = '<1', None
                v,bins = self._channelConfig(selection, nbjet, njet)

                # templates
                chTemplatesShp,chTemplatesCnt = self._makeChTemp(selection, nbjet, njet, v, bins)
                templatesShp.append(chTemplatesShp)
                templatesCnt.append(chTemplatesCnt)

                if self.varation == '':
                    # targets
                    df = DFCutter(selection,nbjet,'data2016',njet).getDataFrame()
                    tempShp, tempCnt = self._binDataFrame(df,v,bins)
                    targetsShp.append(tempShp)
                    targetsCnt.append(tempCnt)


        #####################
        # return the result
        #####################
        templatesShp = np.array(templatesShp)
        templatesCnt = np.array(templatesCnt)
        if self.varation == '':
            targetsShp = np.concatenate(targetsShp)
            targetsCnt = np.concatenate(targetsCnt)

        return  templatesShp, templatesCnt, targetsShp, targetsCnt

        

    def _makeChTemp(self,selection,nbjet,njet,v,bins):

        chTemplatesShp,chTemplatesCnt = [],[]
                
        # singals
        df = DFCutter(selection,nbjet,'mctt',njet).getDataFrame(self.varation)
        for i in range(1,22,1):
            dfi = df[df.genCategory==i]
            tempShp, tempCnt = self._binDataFrame(dfi,v,bins)
            chTemplatesShp.append(tempShp)
            chTemplatesCnt.append(tempCnt)
            
        df = DFCutter(selection,nbjet,'mct',njet).getDataFrame(self.varation)
        for i in range(1,22,1):
            dfi = df[df.genCategory==i]
            tempShp, tempCnt = self._binDataFrame(dfi,v,bins)
            chTemplatesShp.append(tempShp)
            chTemplatesCnt.append(tempCnt)
        # backgrounds
        df = DFCutter(selection,nbjet,'mcw',njet).getDataFrame(self.varation)
        tempShp, tempCnt =  self._binDataFrame(df,v,bins)
        chTemplatesShp.append(tempShp)
        chTemplatesCnt.append(tempCnt)
        
        df = DFCutter(selection,nbjet,'mcz',njet).getDataFrame(self.varation)
        tempShp, tempCnt =  self._binDataFrame(df,v,bins)
        chTemplatesShp.append(tempShp)
        chTemplatesCnt.append(tempCnt)

        df = DFCutter(selection,nbjet,'mcdiboson',njet).getDataFrame(self.varation)
        tempShp, tempCnt =  self._binDataFrame(df,v,bins)
        chTemplatesShp.append(tempShp)
        chTemplatesCnt.append(tempCnt)
        
        # QCD
        tempShp, tempCnt = self._getFake(selection,nbjet,njet,v,bins)
        chTemplatesShp.append(tempShp)
        chTemplatesCnt.append(tempCnt)


        # for arr in chTemplatesShp:
        #     print(arr.shape)

        # return
        chTemplatesShp = np.concatenate(chTemplatesShp)
        chTemplatesCnt = np.concatenate(chTemplatesCnt)
        return chTemplatesShp, chTemplatesCnt

        
    def _getFake(self,selection,nbjet,njet,v,bins):

        returnZeroShp = np.zeros([1, bins.size-1])
        returnZeroCnt = np.zeros([1,1])
        
        if selection == "mu4j":
            fakeSF = common.getFakeSF('mu')
        elif selection == "e4j":
            fakeSF = common.getFakeSF('e')
        elif selection in ["etau",'mutau']:
            fakeSF = common.getFakeSF('tau')
        else:
            return returnZeroShp, returnZeroCnt
        
        df   = DFCutter(selection+'_fakes', nbjet, "data2016", njet).getDataFrame()
        tempShp,tempCnt = self._binDataFrame(df,v,bins)

        for name in ['mcdiboson','mcdy','mct','mctt']:
            df    = DFCutter(selection+'_fakes', nbjet, name, njet).getDataFrame()
            temp1,temp2 = self._binDataFrame(df,v,bins)
            tempShp -= temp1
            tempCnt -= temp2
        tempShp *= fakeSF
        tempCnt *= fakeSF

        return tempShp,tempCnt


    def _binDataFrame(self, df, v, bins ):
        tempShp = np.histogram(df[v], bins, weights=df.eventWeight)[0]
        tempCnt = np.sum(df.eventWeight)

        tempShp = np.reshape(tempShp,(1,-1))
        tempCnt = np.reshape(tempCnt,(1,-1))
        return tempShp, tempCnt

    def _channelConfig(self, selection, nbjet, njet):

        if selection == 'emu':
            v = 'lepton2_pt'
            bins = np.arange(15,165,5)
        
        if selection == 'mumu':
            v = 'lepton2_pt'
            bins = np.arange(10,160,5)
        
        if selection == 'mutau':
            v = 'lepton2_pt'
            bins = np.arange(20,170,5)
            if nbjet == '<1':
                v = 'lepton1_mt'
                bins = np.arange(0,150,5)

        if selection == 'mu4j':
            v = 'lepton1_pt'
            bins = np.arange(30,180,5)

        if selection == 'ee':
            v = 'lepton2_pt'
            bins = np.arange(15,165,5)
        
        if selection == 'emu2':
            v = 'lepton1_pt'
            bins = np.arange(10,160,5)
        
        if selection == 'etau':
            v = 'lepton2_pt'
            bins = np.arange(20,170,5)
            if nbjet == '<1':
                v = 'lepton1_mt'
                bins = np.arange(0,150,5)

        if selection == 'e4j':
            v = 'lepton1_pt'
            bins = np.arange(30,180,5)

        if selection == 'mumuc':
            v = 'lepton2_pt'
            bins = np.arange(10,70,2)

        if selection == 'eec':
            v = 'lepton2_pt'
            bins = np.arange(10,70,2)

        
        return v, bins



# baseDir = common.getBaseDirectory()
# for variation in ['','EPtDown','MuPtDown','TauPtDown',"JESUp","JESDown","JERUp","JERDown","BTagUp","BTagDown","MistagUp","MistagDown"]:
#     temp = np.load(baseDir + "data/templatesCounting/templatesX_{}.npy".format(variation))
#     temp = temp[0:16]
#     np.save(baseDir+ "data/templates/counting_signalRegion/X_{}.npy".format(variation),temp)
#     if variation == '':
#         temp = np.load(baseDir + "data/templatesCounting/templatesY_{}.npy".format(variation))
#         temp = temp[0:16]
#         np.save(baseDir+ "data/templates/counting_signalRegion/Y_{}.npy".format(variation),temp)