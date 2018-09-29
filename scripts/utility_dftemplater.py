import utility_common as common
from utility_dfcutter import *


class DFTemplater():
    def __init__(self, variation=''):
        self.varation = variation
    
    def makeTemplates(self):
        
        '''
        return target shape as (c x t x b)
        '''

        templates = []
        
        # channels in signal regions
        for nbjet in ['==1','>1']:
            for selection in  ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:

                channelTemplates = []
                
                # singals
                df = DFCutter(selection,nbjet,'mctt').getDataFrame(self.varation)
                for i in range(1,22,1):
                    dfi = df[df.genCategory==i]
                    temp = np.sum( dfi.eventWeight )
                    temp = np.reshape(temp,(1,-1))
                    channelTemplates.append(temp)
                    
                df = DFCutter(selection,nbjet,'mct').getDataFrame(self.varation)
                for i in range(1,22,1):
                    dfi = df[df.genCategory==i]
                    temp = np.sum( dfi.eventWeight )
                    temp = np.reshape(temp,(1,-1))
                    channelTemplates.append(temp)
                # backgrounds
                df = DFCutter(selection,nbjet,'mcw').getDataFrame(self.varation)
                temp = np.sum( df.eventWeight )
                temp = np.reshape(temp,(1,-1))
                channelTemplates.append(temp)
                
                df = DFCutter(selection,nbjet,'mcz').getDataFrame(self.varation)
                temp = np.sum( df.eventWeight )
                temp = np.reshape(temp,(1,-1))
                channelTemplates.append(temp)

                df = DFCutter(selection,nbjet,'mcdiboson').getDataFrame(self.varation)
                temp = np.sum( df.eventWeight )
                temp = np.reshape(temp,(1,-1))
                channelTemplates.append(temp)
                
                # QCD
                temp = self._getNFake(selection, nbjet)
                temp = np.reshape(temp,(1,-1))
                channelTemplates.append(temp)

                # return
                channelTemplates = np.concatenate(channelTemplates)
                templates.append(channelTemplates)
                
                
        templates = np.array(templates)
        self.templates = templates
        return templates
        
    def makeTargets(self):

        '''
        return target shape as (c x b)
        '''

        targets = []
        
        for nbjet in ['==1','>1']:
            for selection in  ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:

                df = DFCutter(selection,nbjet,'data2016').getDataFrame()
                temp = np.sum( df.eventWeight )
                temp = np.reshape(temp,(1,-1))
                targets.append(temp)

        targets = np.concatenate(targets)
        self.targets = targets
        return targets
        
    def _getNFake(self,selection,nbjet):
        
        if selection == "mu4j":
            fakeSF = common.getFakeSF('mu')
        elif selection == "e4j":
            fakeSF = common.getFakeSF('e')
            
        elif selection in ["etau",'mutau']:
            fakeSF = common.getFakeSF('tau')
        else:
            return 0
        
        temp = DFCutter(selection+'_fakes',nbjet,"data2016").getDataFrame()
        n    = np.sum(temp.eventWeight)
        for name in ['mcdiboson','mcdy','mct','mctt']:
            temp  = DFCutter(selection+'_fakes',nbjet,name).getDataFrame()
            n    -= np.sum(temp.eventWeight)  
        n *= fakeSF
        return n