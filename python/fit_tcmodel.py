from fit_tchelper import *
from fit_tcmodelLayer import *

class TCPredictiveModel(tc.nn.Module):
    def __init__(self, shaping = False):
        super(TCPredictiveModel,self).__init__()
        # perturbative layer parameterized by 
        # bf of W decay
        self.layer_beta = PertLayer_beta() 
        # bf of tau decay
        self.layer_btl  = PertLayer_btl() 
        # cross sections and Lumin
        self.layer_xs   = PertLayer_xs()
        # lepton efficiency
        self.layer_eff  = PertLayer_eff()
        # template-modifying paramters
        self.layer_itp  = PertLayer_itp(shaping)

    def forward(self, x, xctl=None):

        # regulization
        regu = self.paramRegulization()
        
        # forward propogation through all layers
        h1 = self.layer_beta.forward(x)
        h2 = self.layer_btl.forward(h1)
        h3 = self.layer_xs. forwardSignal(h2)
        h4 = self.layer_eff.forwardSignal(h3)
        h5 = self.layer_itp.forwardSignal(h4)
        # prediction
        y = tc.sum(h5,1)

        if not xctl is None:
            h3clt = self.layer_xs. forwardControl(xctl)
            h4clt = self.layer_eff.forwardControl(h3clt)
            h5clt = self.layer_itp.forwardControl(h4clt)
            # prediction
            yctl = tc.sum(h5clt,1)
        else:
            yctl = None

        return y,regu,yctl

    def paramRegulization(self):
        regu = 0
        for name, param in self.named_parameters():
            graded = param.requires_grad
            nuisance = not ('layer_beta' in name)
            if graded and nuisance:
                regu += 0.5*tc.sum(param**2)
        return regu