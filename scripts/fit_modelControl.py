from fit_model import *

class PredictiveModel_Control(PredictiveModel):
    def __init__(self, x, shaping=False):
        super().__init__(x)
        super().configTemplateVariation('control',shaping)

    def predict(self, params):
        
        x = self.x.copy()

        # splite parameters
        params_beta  = params[0:3]
        params_btl   = params[3:5]
        params_xs    = params[5:14]
        params_eff   = params[14:18]
        params_shape = params[18:25]
        
        # variating templates
        #h1 = self.pertLayer_beta ( x, params_beta)
        #h2 = self.pertLayer_btl  (h1, params_btl)
        h3 = self.pertLayer_xs   ( x, params_xs)
        h4 = self.pertLayer_eff  (h3, params_eff)
        h5 = self.pertLayer_shape(h4, params_shape)
        
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
        zxs    = params[3]*0.05 + 1
        vvxs   = params[4]*0.10 + 1
        eqcdxs = params[5]*0.25 + 1
        mqcdxs = params[6]*0.25 + 1
        tqcdxs = params[7]*0.25 + 1
        lumin  = params[8]*.025 + 1
        ######################


        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs * lumin

        # QCD in relevaent channels
        y[2,45,:] = x[2,45,:] * tqcdxs 
        y[3,45,:] = x[3,45,:] * tqcdxs
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

        for j in range(2):
            idx = j*21
            y[2,14+idx,:] = x[2,14+idx,:] * effm * efft
            y[2,16+idx,:] = x[2,16+idx,:] * effm * efftmis
            y[3,11+idx,:] = x[3,11+idx,:] * effe * efft
            y[3,15+idx,:] = x[3,15+idx,:] * effe * efftmis
        y[2,42,:] = x[2,42,:] * effm * efftmis
        y[2,43,:] = x[2,43,:] * effm * efft
        y[3,42,:] = x[3,42,:] * effe * efftmis
        y[3,43,:] = x[3,43,:] * effe * efft

        return y