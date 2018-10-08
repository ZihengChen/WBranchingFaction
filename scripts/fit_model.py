from pylab import *
import utility_common as common
import utility_commonPlot as commonp



class PredictiveModel_np():
    def __init__(self, x,
                 controlTauID  = False
                 ):
        self.x = x
        self.controlTauID = controlTauID
        
        # define constant
        self._getShapeVariation()
        
    def predict(self, params):

        x = self.x
        
        # splite parameters
        params_beta  = params[0:3]
        params_btl   = params[3:5]
        params_xs    = params[5:14]
        params_eff   = params[14:18]
        params_shape = params[18:25]
        
        # variating templates
        h1 = self.pertLayer_beta (x , params_beta)
        h2 = self.pertLayer_btl  (h1, params_btl)
        h3 = self.pertLayer_xs   (h2, params_xs)
        h4 = self.pertLayer_eff  (h3, params_eff)
        h5 = self.pertLayer_shape(h4, params_shape)
        
        # prediction and regulization
        y    = np.sum(h5,axis=1)
        regu = np.sum(0.5*params[3:]**2)
        return y, regu
    
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
            y[:16,idx+0,:]  = x[:16,idx+0,:]  * bwe*bwe 
            y[:16,idx+1,:]  = x[:16,idx+1,:]  * bwm*bwm 
            y[:16,idx+2,:]  = x[:16,idx+2,:]  * bwe*bwm
            y[:16,idx+3,:]  = x[:16,idx+3,:]  * bwt**2
            y[:16,idx+4,:]  = x[:16,idx+4,:]  * bwt**2 
            y[:16,idx+5,:]  = x[:16,idx+5,:]  * bwt**2   
            y[:16,idx+6,:]  = x[:16,idx+6,:]  * bwt**2   
            y[:16,idx+7,:]  = x[:16,idx+7,:]  * bwt**2   
            y[:16,idx+8,:]  = x[:16,idx+8,:]  * bwt**2   
            y[:16,idx+9,:]  = x[:16,idx+9,:]  * bwe*bwt  
            y[:16,idx+10,:] = x[:16,idx+10,:] * bwe*bwt  
            y[:16,idx+11,:] = x[:16,idx+11,:] * bwe*bwt  
            y[:16,idx+12,:] = x[:16,idx+12,:] * bwm*bwt  
            y[:16,idx+13,:] = x[:16,idx+13,:] * bwm*bwt  
            y[:16,idx+14,:] = x[:16,idx+14,:] * bwm*bwt  
            y[:16,idx+15,:] = x[:16,idx+15,:] * bwe*bwh  
            y[:16,idx+16,:] = x[:16,idx+16,:] * bwm*bwh  
            y[:16,idx+17,:] = x[:16,idx+17,:] * bwt*bwh  
            y[:16,idx+18,:] = x[:16,idx+18,:] * bwt*bwh  
            y[:16,idx+19,:] = x[:16,idx+19,:] * bwt*bwh  
            y[:16,idx+20,:] = x[:16,idx+20,:] * bwh*bwh  

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
            y[:16,idx+0,:]  = x[:16,idx+0,:]  * 1
            y[:16,idx+1,:]  = x[:16,idx+1,:]  * 1
            y[:16,idx+2,:]  = x[:16,idx+2,:]  * 1
            y[:16,idx+3,:]  = x[:16,idx+3,:]  * bte*bte
            y[:16,idx+4,:]  = x[:16,idx+4,:]  * btm*btm 
            y[:16,idx+5,:]  = x[:16,idx+5,:]  * bte*btm
            y[:16,idx+6,:]  = x[:16,idx+6,:]  * bte*bth
            y[:16,idx+7,:]  = x[:16,idx+7,:]  * btm*bth
            y[:16,idx+8,:]  = x[:16,idx+8,:]  * bth*bth
            y[:16,idx+9,:]  = x[:16,idx+9,:]  * bte
            y[:16,idx+10,:] = x[:16,idx+10,:] * btm
            y[:16,idx+11,:] = x[:16,idx+11,:] * bth
            y[:16,idx+12,:] = x[:16,idx+12,:] * bte
            y[:16,idx+13,:] = x[:16,idx+13,:] * btm
            y[:16,idx+14,:] = x[:16,idx+14,:] * bth
            y[:16,idx+15,:] = x[:16,idx+15,:] * 1
            y[:16,idx+16,:] = x[:16,idx+16,:] * 1
            y[:16,idx+17,:] = x[:16,idx+17,:] * bte
            y[:16,idx+18,:] = x[:16,idx+18,:] * btm
            y[:16,idx+19,:] = x[:16,idx+19,:] * bth
            y[:16,idx+20,:] = x[:16,idx+20,:] * 1
        
        return y


    def pertLayer_xs(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        ttxs = params[0]*0.05 + 1
        txs  = params[1]*0.05 + 1
        wxs  = params[2]*0.05 + 1
        zxs  = params[3]*0.05 + 1
        vvxs = params[4]*0.10 + 1
        eqcdxs = params[5]*0.25 + 1
        mqcdxs = params[6]*0.25 + 1
        tqcdxs = params[7]*0.25 + 1
        lumin  = params[8]*0.025 + 1
        ######################
        
        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs * lumin
        
        # QCD in relevaent channels
        y[ 7,45,:] = x[ 7,45,:] * eqcdxs 
        y[15,45,:] = x[15,45,:] * eqcdxs 
        y[ 3,45,:] = x[ 3,45,:] * mqcdxs 
        y[11,45,:] = x[11,45,:] * mqcdxs
        y[ 2,45,:] = x[ 2,45,:] * tqcdxs 
        y[10,45,:] = x[10,45,:] * tqcdxs 
        y[ 6,45,:] = x[ 6,45,:] * tqcdxs 
        y[14,45,:] = x[14,45,:] * tqcdxs

        #y[18,45,:] = x[18,45,:] * tqcdxs 
        #y[19,45,:] = x[19,45,:] * tqcdxs
        
        return y
    
    def pertLayer_eff(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        effe = params[0]*0.01 + 1
        effm = params[1]*0.01 + 1
        efft = params[2]*0.05 + 1
        efftmis = params[3]*0.05 + 1
        ######################
        qcd = -1
        
        for i in range(2):
            idx = i*8
            y[0+idx,:qcd,:] = x[0+idx,:qcd,:] * effm * effe
            y[1+idx,:qcd,:] = x[1+idx,:qcd,:] * effm * effm
            y[3+idx,:qcd,:] = x[3+idx,:qcd,:] * effm 
            y[4+idx,:qcd,:] = x[4+idx,:qcd,:] * effe * effe
            y[5+idx,:qcd,:] = x[5+idx,:qcd,:] * effe * effm
            y[7+idx,:qcd,:] = x[7+idx,:qcd,:] * effe 

            for j in range(2):
                idx2 = j*21
                y[2+idx,14+idx2,:] = x[2+idx,14+idx2,:] * effm * efft
                y[2+idx,16+idx2,:] = x[2+idx,16+idx2,:] * effm * efftmis
                y[6+idx,11+idx2,:] = x[6+idx,11+idx2,:] * effe * efft
                y[6+idx,15+idx2,:] = x[6+idx,15+idx2,:] * effe * efftmis
        
        if self.controlTauID:
            y[16,:qcd,:] = x[16,:qcd,:] * effm * effm # ZToMuMu
            y[17,:qcd,:] = x[17,:qcd,:] * effe * effe # ZToEE
            

            for j in range(2):
                idx2 = j*21
                y[18,14+idx2,:] = x[18,14+idx2,:] * effm * efft
                y[18,16+idx2,:] = x[18,16+idx2,:] * effm * efftmis
                y[19,11+idx2,:] = x[19,11+idx2,:] * effe * efft
                y[19,15+idx2,:] = x[19,15+idx2,:] * effe * efftmis
            y[18,42,:] = x[18,42,:] * effm * efftmis
            y[18,43,:] = x[18,43,:] * effm * efft
            y[19,42,:] = x[19,42,:] * effe * efftmis
            y[19,43,:] = x[19,43,:] * effe * efft
            # y[18,,:] = x[18,:qcd,:] * effm * efft # ZToTauTauToMuH
            # y[19,:qcd,:] = x[19,:qcd,:] * effe * efft # ZToTauTauToEH
        
        return y
    
            
    def pertLayer_shape(self, x, params):
        
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

        
        
    def _getShapeVariation(self):
        
        baseDir = common.getBaseDirectory()

        # lepton energy
        dx_list = []
        for variation in ['EPtDown','MuPtDown','TauPtDown']:
            x0 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(''))
            x1 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(variation))
            
            if not self.controlTauID:
                x0 = x0[0:16]
                x1 = x1[0:16]

            dx = (x0-x1)
            dx_list.append(dx)

        self.dx_energye = dx_list[0]
        self.dx_energym = dx_list[1]
        self.dx_energyt = dx_list[2]

        # Jet energy and btag
        dx_list = []
        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/templatesX_{}Down.npy".format(variation))
            x2 = np.load(baseDir + "data/templates/templatesX_{}Up.npy".format(variation))
            
            if not self.controlTauID:
                x1 = x1[0:16]
                x2 = x2[0:16]
                
            dx = (x0-x1)/2
            dx_list.append(dx)
        self.dx_jes     = dx_list[0]
        self.dx_jer     = dx_list[1]
        self.dx_btag    = dx_list[2]
        self.dx_mistag  = dx_list[3]