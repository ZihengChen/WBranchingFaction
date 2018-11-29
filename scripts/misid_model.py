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

        nsf,nxs,neff = 12,7,2
        # splite parameters
        end = 0
        params_sf  = params[end:end+nsf]
        end += nsf
        params_xs = params[end:end+nxs]
        end += nxs
        params_eff   = params[end:end+neff]
        end += neff
        
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
        # sfq = params[0] # sf for light quark -> tau
        # sfb = params[1] # sf for heavy quark -> tau
        # sfg = params[2] # sf for glon -> tau
        # sfl = params[3] # sf for lepton -> tau
        ######################

        for i in range(5):
            # y[:,i*5+0:i*5+4,:] = y[:,i*5+0:i*5+4,:] * params[0] # pt = 20-25 GeV

            # y[:,i*5+0:i*5+3,0:1] = y[:,i*5+0:i*5+3,0:1] * params[0] # pt = 20-25 GeV
            # y[:,i*5+0:i*5+3,1:2] = y[:,i*5+0:i*5+3,1:2] * params[1] # pt = 25-30 GeV
            # y[:,i*5+0:i*5+3,2:4] = y[:,i*5+0:i*5+3,2:4] * params[2] # pt = 30-40 GeV
            # y[:,i*5+0:i*5+3,4:6] = y[:,i*5+0:i*5+3,4:6] * params[3] # pt = 40-50 GeV
            # y[:,i*5+0:i*5+3,6:9] = y[:,i*5+0:i*5+3,6:9] * params[4] # pt = 50-60 GeV
            # y[:,i*5+0:i*5+3,9: ] = y[:,i*5+0:i*5+3,9: ] * params[5] # pt = 60-   GeV


            ########################
            y[:,i*5+0:i*5+1,0:1] = y[:,i*5+0:i*5+1,0:1] * params[0] # pt = 20-25 GeV
            y[:,i*5+0:i*5+1,1:2] = y[:,i*5+0:i*5+1,1:2] * params[1] # pt = 25-30 GeV
            y[:,i*5+0:i*5+1,2:5] = y[:,i*5+0:i*5+1,2:5] * params[2] # pt = 30-40 GeV
            y[:,i*5+0:i*5+1,4:6] = y[:,i*5+0:i*5+1,4:6] * params[3] # pt = 40-50 GeV
            y[:,i*5+0:i*5+1,6:9] = y[:,i*5+0:i*5+1,6:9] * params[4] # pt = 50-60 GeV
            y[:,i*5+0:i*5+1,9: ] = y[:,i*5+0:i*5+1,9: ] * params[5] # pt = 60-   GeV

            y[:,i*5+1:i*5+2,0:1] = y[:,i*5+1:i*5+2,0:1] * params[6] # pt = 20-25 GeV
            y[:,i*5+1:i*5+2,1:2] = y[:,i*5+1:i*5+2,1:2] * params[7] # pt = 25-30 GeV
            y[:,i*5+1:i*5+2,2:4] = y[:,i*5+1:i*5+2,2:4] * params[8] # pt = 30-40 GeV
            y[:,i*5+1:i*5+2,4:6] = y[:,i*5+1:i*5+2,4:6] * params[9] # pt = 40-50 GeV
            y[:,i*5+1:i*5+2,6:9] = y[:,i*5+1:i*5+2,6:9] * params[10] # pt = 50-60 GeV
            y[:,i*5+1:i*5+2,9: ] = y[:,i*5+1:i*5+2,9: ] * params[11] # pt = 60-   GeV
        return y


    def pertLayer_xs(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        ttxs   = params[0]*0.05 + 1
        txs    = params[1]*0.05 + 1
        wxs    = params[2]*0.05 + 1
        zxs0   = params[3]*0.05 + 1  - 0.05
        zxs1   = params[4]*0.05 + 1  #- 0.05
        vvxs   = params[5]*0.10 + 1
        lumin  = params[6]*.025 + 1
        ######################
        y[:, 0: 5, :]= x[:, 0: 5,:] * ttxs * lumin  # tt
        y[:, 5:10, :]= x[:, 5:10,:] * txs  * lumin  # tW
        y[:,10:15,:] = x[:,10:15,:] * wxs  * lumin  # W+jets
        y[0,15:20,:] = x[0,15:20,:] * zxs0 * lumin  # Z+0jets
        y[2,15:20,:] = x[2,15:20,:] * zxs0 * lumin  # Z+0jets
        y[4,15:20,:] = x[4,15:20,:] * zxs0 * lumin  # Z+0jets
        y[1,15:20,:] = x[1,15:20,:] * zxs1 * lumin  # Z+1jets
        y[3,15:20,:] = x[3,15:20,:] * zxs1 * lumin  # Z+1jets
        y[5,15:20,:] = x[5,15:20,:] * zxs1 * lumin  # Z+1jets
        y[:,20:25,:] = x[:,20:25,:] * vvxs * lumin  # VV

        return y
    

    def pertLayer_eff(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        effe = params[0]*0.01 + 1
        effm = params[1]*0.01 + 1
        ######################

        y[0,:,:] = x[0,:,:] * effe * effm
        y[1,:,:] = x[1,:,:] * effe * effm
        y[2,:,:] = x[2,:,:] * effm * effm
        y[3,:,:] = x[3,:,:] * effm * effm
        y[4,:,:] = x[4,:,:] * effe * effe
        y[5,:,:] = x[5,:,:] * effe * effe

        return y
        






