import utility_common as common
from utility_dfcutter import *


class DFTemplater():
    def __init__(self, variation='',folderOfPickles='pickles_TightTauMisid'):
                 
        self.varation = variation
        self.folderOfPickles = folderOfPickles


    def makeTemplatesAndTargets(self):
        
        templatesShp, templatesCnt = [],[]
        templatesShpVar, templatesCntVar = [],[]
        targetsShp,   targetsCnt   = [],[]
        
        nbjet = '>=0'


        for selection in  ["emutau","mumutau","eetau"]:
            for njet in ['==0','>0']:
                # config
                v,bins = self._channelConfig(selection, nbjet, njet)

                # templates
                chTemplatesShp,chTemplatesShpVar,chTemplatesCnt,chTemplatesCntVar = self._makeChTemp(selection, nbjet, njet, v, bins)
                templatesShp.append(chTemplatesShp)
                templatesShpVar.append(chTemplatesShpVar)

                templatesCnt.append(chTemplatesCnt)
                templatesCntVar.append(chTemplatesCntVar)

                if self.varation == '':
                    # targets
                    df = DFCutter(selection,nbjet,'data',njet,self.folderOfPickles).getDataFrame()
                    tempShp,tempShpVar, tempCnt,tempCntVar = self._binDataFrame(df,v,bins)
                    targetsShp.append(tempShp)
                    targetsCnt.append(tempCnt)

        #####################
        # return the result
        #####################
        templatesShp = np.array(templatesShp)
        templatesCnt = np.array(templatesCnt)
        templatesShpVar = np.concatenate(templatesShpVar)
        templatesCntVar = np.concatenate(templatesCntVar)
        if self.varation == '':
            targetsShp = np.concatenate(targetsShp)
            targetsCnt = np.concatenate(targetsCnt)

        return  templatesShp,templatesShpVar, templatesCnt,templatesCntVar, targetsShp, targetsCnt

        

    def _makeChTemp(self,selection,nbjet,njet,v,bins):

        chTemplatesShp,chTemplatesCnt = [],[]
        chTemplatesShpVar,chTemplatesCntVar = [],[]

        breakdownList = [   'tauGenFlavor<=5 and tauGenFlavor>4',  # heavy quark  -> tauID
                            'tauGenFlavor<=4 and tauGenFlavor>=1',  # light quark  -> tauID
                            'tauGenFlavor==21',                     # gluon  -> tauID
                            'tauGenFlavor==26 or tauGenFlavor==0',  # lepton -> tauID
                            'tauGenFlavor==15'                      # tau_h  -> tauID
                        ] 



        for name in ['mctt','mct','mcz','mcdiboson']:
            print(name)
            df = DFCutter(selection,nbjet,name,njet, self.folderOfPickles).getDataFrame(self.varation)
            for q in breakdownList:
                dfi = df.query(q)
                tempShp, tempShpVar, tempCnt, tempCntVar = self._binDataFrame(dfi,v,bins)
                chTemplatesShp.append(tempShp)
                chTemplatesCnt.append(tempCnt)
                chTemplatesShpVar.append(tempShpVar)
                chTemplatesCntVar.append(tempCntVar)

        # return
        chTemplatesShp = np.concatenate(chTemplatesShp)
        chTemplatesCnt = np.concatenate(chTemplatesCnt)

        chTemplatesShpVar = np.concatenate(chTemplatesShpVar)
        chTemplatesCntVar = np.concatenate(chTemplatesCntVar)
        chTemplatesShpVar = np.sum(chTemplatesShpVar,axis=0).reshape(1,-1)
        chTemplatesCntVar = np.sum(chTemplatesCntVar,axis=0).reshape(1,-1)
        
        return chTemplatesShp,chTemplatesShpVar, chTemplatesCnt,chTemplatesCntVar

        

    def _binDataFrame(self, df, v, bins ):
        tempShp = np.histogram(df[v], bins, weights=df.eventWeight)[0]
        tempCnt = np.sum(df.eventWeight)
        tempShp = np.reshape(tempShp,(1,-1))
        tempCnt = np.reshape(tempCnt,(1,-1))


        tempShpVar = np.histogram(df[v], bins, weights=df.eventWeight**2)[0]
        tempCntVar = np.sum(df.eventWeight**2)
        tempShpVar = np.reshape(tempShpVar,(1,-1))
        tempCntVar = np.reshape(tempCntVar,(1,-1))

        return tempShp,tempShpVar, tempCnt,tempCntVar

    def _channelConfig(self, selection, nbjet, njet):

        if selection == 'emutau':
            v = 'tauPt'
            bins = np.arange(20,81,5)
        
        if selection == 'mumutau':
            v = 'tauPt'
            bins = np.arange(20,81,5)

        if selection == 'eetau':
            v = 'tauPt'
            bins = np.arange(20,81,5)
        
        return v, bins