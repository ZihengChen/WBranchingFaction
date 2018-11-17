from pylab import *
import utility_common as common
import utility_commonPlot as commonp



class PredictiveModel():
    def __init__(self,x):
        self.x = x  
        self.c = x.shape[0]
        self.t = x.shape[1]
        self.b = x.shape[2]

    def predict(self, params, returnTemplates=False):
        
        x = self.x.copy()

        # splite parameters
        params_sf    = params[0:1]
        params_xs    = params[1:9]
        params_eff   = params[9:11]
        
        # variating templates
        h1 = self.pertLayer_sf  ( x, params_sf)
        h2 = self.pertLayer_xs  (h1, params_xs)
        h3 = self.pertLayer_eff (h2, params_eff)
        
        if returnTemplates:
            return h3
        else:
            # prediction and regulization
            y = np.sum(h3,axis=1)
            return y

    def pertLayer_sf(self,x, params):

        y = x.copy()

        ######################
        # defining paramters
        sf = params[0]
        ######################

        for i in range(5):
            y[:,i*4+0,:] = y[:,i*4+0,:] * sf
            y[:,i*4+1,:] = y[:,i*4+1,:] * sf 

        return y


    def pertLayer_xs(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        ttxs   = params[0]*0.05 + 1
        txs    = params[1]*0.05 + 1
        wxs    = params[2]*0.05 + 1
        zxs    = params[3]*0.05 + 1
        zxs0   = params[4]*0.05 + 1
        zxs1   = params[5]*0.05 + 1
        vvxs   = params[6]*0.10 + 1
        lumin  = params[7]*.025 + 1
        ######################
        y[:, 0:4, :] = x[:, 0:4,:]  * ttxs * lumin
        y[:, 4:8, :] = x[:, 4:8,:]  * txs  * lumin
        y[:, 8:12,:] = x[:, 8:12,:] * wxs  * lumin
        y[0,12:16,:] = x[0,12:16,:] * zxs  * lumin     
        y[1,12:16,:] = x[1,12:16,:] * zxs0 * lumin 
        y[2,12:16,:] = x[2,12:16,:] * zxs1 * lumin 
        y[3,12:16,:] = x[3,12:16,:] * zxs0 * lumin 
        y[4,12:16,:] = x[4,12:16,:] * zxs1 * lumin 
        y[:,16:20,:] = x[:,16:20,:] * vvxs * lumin  

        return y
    

    def pertLayer_eff(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        effe = params[0]*0.01 + 1
        effm = params[1]*0.01 + 1
        ######################

        y[0,:,:] = x[0,:,:] * effe * effm
        y[1,:,:] = x[1,:,:] * effm * effm
        y[2,:,:] = x[2,:,:] * effm * effm
        y[3,:,:] = x[3,:,:] * effe * effe
        y[4,:,:] = x[4,:,:] * effe * effe

        return y
        






