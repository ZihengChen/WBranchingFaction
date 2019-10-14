
from fit_model import *

class PredictiveModel_Signal(PredictiveModel):
    def __init__(self, x, shaping=False):
        super().__init__(x)
        super().configTemplateVariation('signal',shaping)

    
    def predict(self, params):
        
        x = self.x.copy()

        # splite parameters
        params_beta  = params[0:3]
        params_btl   = params[3:5]
        params_xs    = params[5:14]
        params_eff   = params[14:18]
        params_shape = params[18:25]
        
        # variating templates
        h1 = self.pertLayer_beta ( x, params_beta)
        h2 = self.pertLayer_btl  (h1, params_btl)
        h3 = self.pertLayer_xs   (h2, params_xs)
        h4 = self.pertLayer_eff  (h3, params_eff)
        h5 = self.pertLayer_itp  (h4, params_shape)
        
        # prediction and regulization
        y = np.sum(h5,axis=1)
        
        return y


    def pertLayer_xs(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        ttxs   = params[0]*0.05 + 1
        txs    = params[1]*0.05 + 1
        wxs    = params[2]*0.05 + 1
        zxs    = params[3]*0.10 + 1
        vvxs   = params[4]*0.10 + 1
        eqcdxs = params[5]*0.25 + 1
        mqcdxs = params[6]*0.25 + 1
        tqcdxs = params[7]*0.25 + 1
        lumin  = params[8]*.025 + 1
        ######################
        
        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs  * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs * lumin
        
        # QCD in relevaent channels
        for i in range(2):
            idx = i*8
            y[ 7+idx,45,:] = x[ 7+idx,45,:] * eqcdxs 
            y[ 3+idx,45,:] = x[ 3+idx,45,:] * mqcdxs 
            y[ 2+idx,45,:] = x[ 2+idx,45,:] * tqcdxs 
            y[ 6+idx,45,:] = x[ 6+idx,45,:] * tqcdxs 

        return y
    
    def pertLayer_eff(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        effe = params[0]*0.01 + 1
        effm = params[1]*0.01 + 1
        efft = params[2]*0.05 + 1
        efftmis = params[3]*0.08 + 1
        ######################
        
        for i in range(2):
            idx = i*8
            y[0+idx,:45,:] = x[0+idx,:45,:] * effm * effe
            y[1+idx,:45,:] = x[1+idx,:45,:] * effm * effm
            y[3+idx,:45,:] = x[3+idx,:45,:] * effm 
            y[4+idx,:45,:] = x[4+idx,:45,:] * effe * effe
            y[5+idx,:45,:] = x[5+idx,:45,:] * effe * effm
            y[7+idx,:45,:] = x[7+idx,:45,:] * effe 

            for j in range(2):
                idx2 = j*21
                y[2+idx,14+idx2,:] = x[2+idx,14+idx2,:] * effm * efft
                y[2+idx,16+idx2,:] = x[2+idx,16+idx2,:] * effm * efftmis
                y[6+idx,11+idx2,:] = x[6+idx,11+idx2,:] * effe * efft
                y[6+idx,15+idx2,:] = x[6+idx,15+idx2,:] * effe * efftmis

        return y
    