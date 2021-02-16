from pylab import *
import utility_common as common
import utility_commonPlot as commonp



class PredictiveModel():
    def __init__(self,x, splitJetFlavor=False):
        self.x = x  
        self.c = x.shape[0]
        self.t = x.shape[1]
        self.b = x.shape[2]

        self.splitJetFlavor = splitJetFlavor

    def predict(self, params, returnTemplates=False):
        
        x = self.x.copy()

        nxs,neff = 6,2
        if self.splitJetFlavor:
          nsf = 10
        else:
          nsf = 5
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
            # y[:,o1:i*5+4,:] = y[:,o1:i*5+4,:] * params[0] # pt = 20-25 GeV
            
            if self.splitJetFlavor:
                o1,o2 = i*5+0, i*5+1 # b->tau
                p1,p2 = 0,15 # tt, tw, Z
                y[p1:p2,o1:o2,0:1] = y[p1:p2,o1:o2,0:1] * params[0] # pt = 20-25 GeV
                y[p1:p2,o1:o2,1:2] = y[p1:p2,o1:o2,1:2] * params[1] # pt = 25-30 GeV
                y[p1:p2,o1:o2,2:5] = y[p1:p2,o1:o2,2:5] * params[2] # pt = 30-40 GeV
                y[p1:p2,o1:o2,4:6] = y[p1:p2,o1:o2,4:6] * params[3] # pt = 40-50 GeV
                y[p1:p2,o1:o2,6: ] = y[p1:p2,o1:o2,6: ] * params[4] # pt = 50-60 GeV
                # y[p1:p2,o1:o2,9: ] = y[p1:p2,o1:o2,9: ] * params[5] # pt = 60-   GeV
                
                o1,o2 = i*5+1, i*5+3 # light->tau and g->tau
                y[p1:p2,o1:o2,0:1] = y[p1:p2,o1:o2,0:1] * params[5] # pt = 20-25 GeV
                y[p1:p2,o1:o2,1:2] = y[p1:p2,o1:o2,1:2] * params[6] # pt = 25-30 GeV
                y[p1:p2,o1:o2,2:4] = y[p1:p2,o1:o2,2:4] * params[7] # pt = 30-40 GeV
                y[p1:p2,o1:o2,4:6] = y[p1:p2,o1:o2,4:6] * params[8] # pt = 40-50 GeV
                y[p1:p2,o1:o2,6: ] = y[p1:p2,o1:o2,6: ] * params[9] # pt = 50-60 GeV
                # y[p1:p2,o1:o2,9: ] = y[p1:p2,o1:o2,9: ] * params[11] # pt = 60-   GeV

            else:

                o1,o2 = i*5+0, i*5+3 # jet->tau
                p1,p2 = 0,15 # tt, tw, Z
                y[p1:p2,o1:o2,0:1] = y[p1:p2,o1:o2,0:1] * params[0] # pt = 20-25 GeV
                y[p1:p2,o1:o2,1:2] = y[p1:p2,o1:o2,1:2] * params[1] # pt = 25-30 GeV
                y[p1:p2,o1:o2,2:4] = y[p1:p2,o1:o2,2:4] * params[2] # pt = 30-40 GeV
                y[p1:p2,o1:o2,4:6] = y[p1:p2,o1:o2,4:6] * params[3] # pt = 40-50 GeV
                y[p1:p2,o1:o2,6: ] = y[p1:p2,o1:o2,6: ] * params[4] # pt = 50-60 GeV
                # y[p1:p2,o1:o2,9: ] = y[p1:p2,o1:o2,9: ] * params[5] # pt = 60-   GeV
        return y


    def pertLayer_xs(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        ttxs   = params[0]*0.05 + 1
        txs    = params[1]*0.05 + 1
        zxs0   = params[2]*0.05 + 1  #- 0.05 # Z0jet has an normalization correction
        zxs1   = params[3]*0.05 + 1  #+ 0.05
        vvxs   = params[4]*0.10 + 1
        lumin  = params[5]*.025 + 1
        ######################
        y[:, 0: 5, :]= x[:, 0: 5,:] * ttxs * lumin  # tt
        y[:, 5:10, :]= x[:, 5:10,:] * txs  * lumin  # tW
        y[0,10:15,:] = x[0,10:15,:] * zxs0 * lumin  # Z+0jets
        y[2,10:15,:] = x[2,10:15,:] * zxs0 * lumin  # Z+0jets
        y[4,10:15,:] = x[4,10:15,:] * zxs0 * lumin  # Z+0jets
        y[1,10:15,:] = x[1,10:15,:] * zxs1 * lumin  # Z+1jets
        y[3,10:15,:] = x[3,10:15,:] * zxs1 * lumin  # Z+1jets
        y[5,10:15,:] = x[5,10:15,:] * zxs1 * lumin  # Z+1jets
        y[:,15:20,:] = x[:,15:20,:] * vvxs * lumin  # VV

        return y
    

    def pertLayer_eff(self, x, params):

        y = x.copy()

        ######################
        # defining perturbation
        effe = params[0]*0.02 + 1
        effm = params[1]*0.02 + 1
        ######################

        y[0,:,:] = x[0,:,:] * effe * effm
        y[1,:,:] = x[1,:,:] * effe * effm
        y[2,:,:] = x[2,:,:] * effm * effm
        y[3,:,:] = x[3,:,:] * effm * effm
        y[4,:,:] = x[4,:,:] * effe * effe
        y[5,:,:] = x[5,:,:] * effe * effe

        return y
        






