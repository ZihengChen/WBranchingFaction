import utility_common as common
from utility_dfcutter import *


class DFTemplater():
    def __init__(self, variation='',):
                 
        self.varation = variation


    def makeTemplatesAndTargets(self):
        
        templatesShp, templatesCnt = [],[]
        targetsShp,   targetsCnt   = [],[]
        
        nbjet = '>=0'

        for selection in  ["emu_tau"]:
            njet = '>=0'
            # config
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



        for selection in  ["mumu_tau","ee_tau"]:
            for njet in ['==0','>0']:
                # config
                
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

        breakdownList = [   'tauGenFlavor<7 and tauGenFlavor>0',    # quark  -> tauID
                            'tauGenFlavor==21',                     # gluon  -> tauID
                            'tauGenFlavor==26 or tauGenFlavor==0',  # lepton -> tauID
                            'tauGenFlavor==15',                     # tau_h  -> tauID
                        ] 


        for name in ['mctt','mct','mcw','mcz','mcdiboson']:
            df = DFCutter(selection,nbjet,name,njet).getDataFrame(self.varation)
            for q in breakdownList:
                dfi = df.query(q)
                tempShp, tempCnt = self._binDataFrame(dfi,v,bins)
                chTemplatesShp.append(tempShp)
                chTemplatesCnt.append(tempCnt)

        # return
        chTemplatesShp = np.concatenate(chTemplatesShp)
        chTemplatesCnt = np.concatenate(chTemplatesCnt)
        return chTemplatesShp, chTemplatesCnt

        

    def _binDataFrame(self, df, v, bins ):
        tempShp = np.histogram(df[v], bins, weights=df.eventWeight)[0]
        tempCnt = np.sum(df.eventWeight)

        tempShp = np.reshape(tempShp,(1,-1))
        tempCnt = np.reshape(tempCnt,(1,-1))
        return tempShp, tempCnt

    def _channelConfig(self, selection, nbjet, njet):

        if selection == 'emu_tau':
            v = 'tauPt'
            bins = np.arange(20,81,5)
        
        if selection == 'mumu_tau':
            v = 'tauPt'
            bins = np.arange(20,81,5)

        if selection == 'ee_tau':
            v = 'tauPt'
            bins = np.arange(20,81,5)
        
        return v, bins