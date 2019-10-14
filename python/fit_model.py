from pylab import *
import utility_common as common
import utility_commonPlot as commonp



class PredictiveModel():
    def __init__(self, x):
        self.x = x  
        self.c = x.shape[0]
        self.t = x.shape[1]
        self.b = x.shape[2]

        

    def pertLayer_beta(self,x, params):

        y = x.copy()

        ######################
        # defining paramters
        bwe = params[0]/0.1080
        bwm = params[1]/0.1080
        bwt = params[2]/0.1080
        bwh = (1 - np.sum(params))/0.6760
        ######################
        
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * bwe*bwe 
            y[:,idx+1,:]  = x[:,idx+1,:]  * bwm*bwm 
            y[:,idx+2,:]  = x[:,idx+2,:]  * bwe*bwm
            y[:,idx+3,:]  = x[:,idx+3,:]  * bwt**2
            y[:,idx+4,:]  = x[:,idx+4,:]  * bwt**2 
            y[:,idx+5,:]  = x[:,idx+5,:]  * bwt**2   
            y[:,idx+6,:]  = x[:,idx+6,:]  * bwt**2   
            y[:,idx+7,:]  = x[:,idx+7,:]  * bwt**2   
            y[:,idx+8,:]  = x[:,idx+8,:]  * bwt**2   
            y[:,idx+9,:]  = x[:,idx+9,:]  * bwe*bwt  
            y[:,idx+10,:] = x[:,idx+10,:] * bwe*bwt  
            y[:,idx+11,:] = x[:,idx+11,:] * bwe*bwt  
            y[:,idx+12,:] = x[:,idx+12,:] * bwm*bwt  
            y[:,idx+13,:] = x[:,idx+13,:] * bwm*bwt  
            y[:,idx+14,:] = x[:,idx+14,:] * bwm*bwt  
            y[:,idx+15,:] = x[:,idx+15,:] * bwe*bwh  
            y[:,idx+16,:] = x[:,idx+16,:] * bwm*bwh  
            y[:,idx+17,:] = x[:,idx+17,:] * bwt*bwh  
            y[:,idx+18,:] = x[:,idx+18,:] * bwt*bwh  
            y[:,idx+19,:] = x[:,idx+19,:] * bwt*bwh  
            y[:,idx+20,:] = x[:,idx+20,:] * bwh*bwh  

        return y

    def pertLayer_btl(self,x, params):

        y = x.copy()

        ######################
        # defining perturbation
        bte = params[0]*0.002 + 1.0
        btm = params[1]*0.002 + 1.0
        bth = (1 - bte*0.1785-btm*0.1736) / 0.6479
        ######################
        
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * 1
            y[:,idx+1,:]  = x[:,idx+1,:]  * 1
            y[:,idx+2,:]  = x[:,idx+2,:]  * 1
            y[:,idx+3,:]  = x[:,idx+3,:]  * bte*bte
            y[:,idx+4,:]  = x[:,idx+4,:]  * btm*btm 
            y[:,idx+5,:]  = x[:,idx+5,:]  * bte*btm
            y[:,idx+6,:]  = x[:,idx+6,:]  * bte*bth
            y[:,idx+7,:]  = x[:,idx+7,:]  * btm*bth
            y[:,idx+8,:]  = x[:,idx+8,:]  * bth*bth
            y[:,idx+9,:]  = x[:,idx+9,:]  * bte
            y[:,idx+10,:] = x[:,idx+10,:] * btm
            y[:,idx+11,:] = x[:,idx+11,:] * bth
            y[:,idx+12,:] = x[:,idx+12,:] * bte
            y[:,idx+13,:] = x[:,idx+13,:] * btm
            y[:,idx+14,:] = x[:,idx+14,:] * bth
            y[:,idx+15,:] = x[:,idx+15,:] * 1
            y[:,idx+16,:] = x[:,idx+16,:] * 1
            y[:,idx+17,:] = x[:,idx+17,:] * bte
            y[:,idx+18,:] = x[:,idx+18,:] * btm
            y[:,idx+19,:] = x[:,idx+19,:] * bth
            y[:,idx+20,:] = x[:,idx+20,:] * 1
        
        return y




    def pertLayer_itp(self, x, params):
        
        y = x.copy()

        ######################
        # defining perturbation
        energye,energym,energyt = params[0],params[1],params[2]
        jes,jer,btag,mistag = params[3],params[4],params[5],params[6]
        ######################

        perturbation = np.zeros_like(x)
        perturbation += self.dx_energye * energye
        perturbation += self.dx_energym * energym
        perturbation += self.dx_energyt * energyt

        perturbation += self.dx_jes * jes
        perturbation += self.dx_jer * jer
        perturbation += self.dx_btag * btag
        perturbation += self.dx_mistag * mistag

        y += perturbation
        
        return y

        
    def configTemplateVariation(self,region,shaping=False):
        baseDir = common.getBaseDirectory()

        if shaping:
            folderType = 'shaping'
        else:
            folderType = 'counting'

        # nominal - dwon
        dx_list = []
        for variation in ['EPt','MuPt','TauPt']:
            x0 = np.load(baseDir + "data/templates/{}_{}Region/X_{}.npy".format(folderType,region,''))
            x1 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Down.npy".format(folderType,region,variation))
            dx = (x0-x1)
            dx_list.append(dx)

        self.dx_energye = dx_list[0]
        self.dx_energym = dx_list[1]
        self.dx_energyt = dx_list[2]

        # (up-down)/2
        dx_list = []
        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Down.npy".format(folderType,region,variation))
            x2 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Up.npy"  .format(folderType,region,variation))
            dx = (x2-x1)/2
            dx_list.append(dx)
        self.dx_jes     = dx_list[0]
        self.dx_jer     = dx_list[1]
        self.dx_btag    = dx_list[2]
        self.dx_mistag  = dx_list[3]
    

        






