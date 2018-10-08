from fit_torch_layer import *
from fit_torch_helper import *

class PredictiveModel(tc.nn.Module):
    def __init__(self, 
                 controlTauID  = False):
        super(PredictiveModel,self).__init__()

        self.controlTauID = controlTauID

        #################
        # define layers #
        #################


        # perturbative layer parameterized by 
        # bf of W decay
        self.layer_beta = PertLayer_beta() 

        # bf of tau decay
        self.layer_btl = PertLayer_btl() 

        # cross sections and Lumin
        self.layer_xs = PertLayer_xs(self.controlTauID)

        # lepton efficiency
        self.layer_eff = PertLayer_eff(self.controlTauID)

        # shape-modifying paramters
        # lepton Energy, Jet Energy, bTagging
        self.layer_shape = PertLayer_shape(self.controlTauID)
        
    def forward(self, x):
        '''
        Forward propogation through parametrized layers.

        input: x
            [x] is templets as a tensor of shape (c,t,b).
            c for number of channels, c=16.
            t for number of templetes per channel, t=21x2+4=46.
            b for number of bins per templetes, b=1 for now.

        output: y, regu
            [y] is prediction as a tensor of shape (c,b).
            [regu] is a scalar regulaization of naussance parameters.
            Chi2 of the gaussian error of nuisance parameters 
            is used as the regulization here.
        '''
        
        # forward propogation through all layers
        h1 = self.layer_beta  (x)
        h2 = self.layer_btl   (h1)
        h3 = self.layer_xs    (h2)
        h4 = self.layer_eff   (h3)
        h5 = self.layer_shape (h4)
        
        # prediction and regulization
        y = tc.sum(h5,1)
        regu  = tc.sum(0.5*self.layer_btl.params**2)
        regu += tc.sum(0.5*self.layer_xs.params**2)
        regu += tc.sum(0.5*self.layer_eff.params**2)
        regu += tc.sum(0.5*self.layer_shape.params**2)
        return y,regu







    