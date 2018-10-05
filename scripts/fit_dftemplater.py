import utility_common as common
from utility_dfcutter import *


class DFTemplater():
    def __init__(self, variation='', tempType='Counting'):
        self.varation = variation
        self.tempType = tempType


    def makeTargets(self):

        '''
        return target shape as (c x b)
        '''

        targets = []
        
        for nbjet in ['==1','>1']:
            for selection in  ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:
                njet = None
                v,bins = self._channelConfig(selection, nbjet, njet)
                df = DFCutter(selection,nbjet,'data2016',None).getDataFrame()
                temp = self._binDataFrame(df,v,bins)
                targets.append(temp)

        targets = np.concatenate(targets)
        self.targets = targets
        return targets

    
    def makeTemplates(self):
        
        '''
        return target shape as (c x t x b)
        '''

        templates = []
        
        # channels in signal regions
        for nbjet in ['==1','>1']:
            for selection in  ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:
                njet = None
                channelTemplates = self._makeChTemp(selection, nbjet, njet)
                #print(channelTemplates.shape)
                templates.append(channelTemplates)
                
        templates = np.array(templates)
        self.templates = templates
        return templates
        

    def _makeChTemp(self,selection, nbjet, njet):

        v,bins = self._channelConfig(selection, nbjet, njet)


        channelTemplates = []
                
        # singals
        df = DFCutter(selection,nbjet,'mctt',njet).getDataFrame(self.varation)
        for i in range(1,22,1):
            dfi = df[df.genCategory==i]
            temp = self._binDataFrame(dfi,v,bins)
            channelTemplates.append(temp)
            
        df = DFCutter(selection,nbjet,'mct',njet).getDataFrame(self.varation)
        for i in range(1,22,1):
            dfi = df[df.genCategory==i]
            temp = self._binDataFrame(dfi,v,bins)
            channelTemplates.append(temp)
        # backgrounds
        df = DFCutter(selection,nbjet,'mcw',njet).getDataFrame(self.varation)
        temp = self._binDataFrame(df,v,bins)
        channelTemplates.append(temp)
        
        df = DFCutter(selection,nbjet,'mcz',njet).getDataFrame(self.varation)
        temp = self._binDataFrame(df,v,bins)
        channelTemplates.append(temp)

        df = DFCutter(selection,nbjet,'mcdiboson',njet).getDataFrame(self.varation)
        temp = self._binDataFrame(df,v,bins)
        channelTemplates.append(temp)
        
        # QCD
        temp = self._getFake(selection, nbjet, njet , v, bins)
        channelTemplates.append(temp)

        # for arr in channelTemplates:
        #     print(arr.shape)

        # return
        channelTemplates = np.concatenate(channelTemplates)
        return channelTemplates

        
    def _getFake(self,selection,nbjet,njet, v, bins):
        if self.tempType is 'Shape':
            returnZero = np.zeros([1, bins.size-1])
        else:
            returnZero = np.zeros([1,1])
        
        if selection == "mu4j":
            fakeSF = common.getFakeSF('mu')
        elif selection == "e4j":
            fakeSF = common.getFakeSF('e')
        elif selection in ["etau",'mutau']:
            fakeSF = common.getFakeSF('tau')
        else:
            return returnZero
        
        df   = DFCutter(selection+'_fakes', nbjet, "data2016", njet).getDataFrame()
        temp = self._binDataFrame(df,v,bins)

        for name in ['mcdiboson','mcdy','mct','mctt']:
            df    = DFCutter(selection+'_fakes', nbjet, name, njet).getDataFrame()
            temp -= self._binDataFrame(df,v,bins)
        temp *= fakeSF
        return temp


    def _binDataFrame(self, df, v, bins ):
        if self.tempType is 'Shape':
            temp = np.histogram(df[v], bins, weights=df.eventWeight)[0]
        else:
            temp = np.sum(df.eventWeight)
        return np.reshape(temp,(1,-1))

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

        if selection == 'e4j':
            v = 'lepton1_pt'
            bins = np.arange(30,180,5)
        
        return v, bins