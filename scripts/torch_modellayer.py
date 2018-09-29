import torch as tc
import torch.nn.functional as F
from   torch.autograd import Variable
from torch.nn import Parameter
import utility_common as common
from pylab import *
from torch.autograd import grad


class PertLayer_Beta(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Beta,self).__init__()
    
        # define parameters of interest
        self.bwe   = Parameter(tc.tensor(.109), requires_grad=True)
        self.bwm   = Parameter(tc.tensor(.109), requires_grad=True)
        self.bwt   = Parameter(tc.tensor(.109), requires_grad=True)

        # define nuisance paramters
        self.bte   = Parameter(tc.tensor(1.), requires_grad=True)
        self.btm   = Parameter(tc.tensor(1.), requires_grad=True)
        self.bth   = Parameter(tc.tensor(1.), requires_grad=True)
        
        # define constant
        self.ll, self.hh, self.lh = .1080**2, .6760**2, .1080*.6760
        

    def forward(self, x):
        
        bwh = 1 - self.bwe - self.bwm - self.bwt
        
        y = tc.ones_like(x) * x
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * self.bwe*self.bwe /self.ll
            y[:,idx+1,:]  = x[:,idx+1,:]  * self.bwm*self.bwm /self.ll
            y[:,idx+2,:]  = x[:,idx+2,:]  * self.bwe*self.bwm /self.ll
            y[:,idx+3,:]  = x[:,idx+3,:]  * self.bwt**2*self.bte*self.bte /self.ll
            y[:,idx+4,:]  = x[:,idx+4,:]  * self.bwt**2*self.btm*self.btm /self.ll
            y[:,idx+5,:]  = x[:,idx+5,:]  * self.bwt**2*self.bte*self.btm /self.ll
            y[:,idx+6,:]  = x[:,idx+6,:]  * self.bwt**2*self.bte*self.bth /self.ll
            y[:,idx+7,:]  = x[:,idx+7,:]  * self.bwt**2*self.btm*self.bth /self.ll
            y[:,idx+8,:]  = x[:,idx+8,:]  * self.bwt**2*self.bth*self.bth /self.ll
            y[:,idx+9,:]  = x[:,idx+9,:]  * self.bwe*self.bwt*self.bte /self.ll
            y[:,idx+10,:] = x[:,idx+10,:] * self.bwe*self.bwt*self.btm /self.ll
            y[:,idx+11,:] = x[:,idx+11,:] * self.bwe*self.bwt*self.bth /self.ll
            y[:,idx+12,:] = x[:,idx+12,:] * self.bwm*self.bwt*self.bte /self.ll
            y[:,idx+13,:] = x[:,idx+13,:] * self.bwm*self.bwt*self.btm /self.ll
            y[:,idx+14,:] = x[:,idx+14,:] * self.bwm*self.bwt*self.bth /self.ll
            y[:,idx+15,:] = x[:,idx+15,:] * self.bwe* bwh /self.lh
            y[:,idx+16,:] = x[:,idx+16,:] * self.bwm* bwh /self.lh
            y[:,idx+17,:] = x[:,idx+17,:] * self.bwt*self.bte* bwh /self.lh
            y[:,idx+18,:] = x[:,idx+18,:] * self.bwt*self.btm* bwh /self.lh
            y[:,idx+19,:] = x[:,idx+19,:] * self.bwt*self.bth* bwh /self.lh
            y[:,idx+20,:] = x[:,idx+20,:] * bwh * bwh /self.hh

        # calculate regulations for nuisance parameters
        regu = 0
        regu += (self.bte-1)**2/(2*0.004)
        regu += (self.btm-1)**2/(2*0.004)
        regu += (self.bth-1)**2/(2*0.002)
        
        return y, regu
    
class PertLayer_XS(tc.nn.Module):
    def __init__(self):
        super(PertLayer_XS,self).__init__()
    
        # define nuisance paramters
        self.ttxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.txs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.wxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.zxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.vvxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.qcdxs = Parameter(tc.tensor(1.), requires_grad=True)
        self.lumin = Parameter(tc.tensor(1.), requires_grad=True) 
            

    def forward(self, x):
        
        y = tc.ones_like(x) * x
        y[:, 0:21,:] = x[:, 0:21,:] * self.ttxs * self.lumin
        y[:,21:42,:] = x[:,21:42,:] * self.txs * self.lumin
        y[:,42,:] = x[:,42,:] * self.wxs * self.lumin
        y[:,43,:] = x[:,43,:] * self.zxs * self.lumin
        y[:,44,:] = x[:,44,:] * self.vvxs * self.lumin
        y[:,45,:] = x[:,45,:] * self.qcdxs * self.lumin
        
        
        regu = 0
        regu += (self.ttxs-1)**2/(2*0.05)
        regu += (self.txs-1)**2 /(2*0.05)
        regu += (self.wxs-1)**2 /(2*0.05)
        regu += (self.zxs-1)**2 /(2*0.05)
        regu += (self.vvxs-1)**2/(2*0.10)
        regu += (self.qcdxs-1)**2/(2*0.25)
        regu += (self.lumin-1)**2/(2*0.025)
        
        return y, regu


class PertLayer_XS(tc.nn.Module):
    def __init__(self):
        super(PertLayer_XS,self).__init__()
    
        # define nuisance paramters
        self.ttxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.txs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.wxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.zxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.vvxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.qcdxs = Parameter(tc.tensor(1.), requires_grad=True)
        self.lumin = Parameter(tc.tensor(1.), requires_grad=True) 
            

    def forward(self, x):
        
        y = tc.ones_like(x) * x
        y[:, 0:21,:] = x[:, 0:21,:] * self.ttxs * self.lumin
        y[:,21:42,:] = x[:,21:42,:] * self.txs * self.lumin
        y[:,42,:] = x[:,42,:] * self.wxs * self.lumin
        y[:,43,:] = x[:,43,:] * self.zxs * self.lumin
        y[:,44,:] = x[:,44,:] * self.vvxs * self.lumin
        y[:,45,:] = x[:,45,:] * self.qcdxs * self.lumin
        
        # calculate regulations for nuisance parameters
        regu = 0
        regu += (self.ttxs-1)**2/(2*0.05)
        regu += (self.txs-1)**2 /(2*0.05)
        regu += (self.wxs-1)**2 /(2*0.05)
        regu += (self.zxs-1)**2 /(2*0.05)
        regu += (self.vvxs-1)**2/(2*0.10)
        regu += (self.qcdxs-1)**2/(2*0.25)
        regu += (self.lumin-1)**2/(2*0.025)
        
        return y, regu


class PertLayer_Eff(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Eff,self).__init__()
    
        # define nuisance paramters
        self.effe  = Parameter(tc.tensor(1.), requires_grad=True)
        self.effm  = Parameter(tc.tensor(1.), requires_grad=True)
        self.efft  = Parameter(tc.tensor(1.), requires_grad=True)
            

    def forward(self, x):
        
        y = tc.ones_like(x)
        for i in range(2):
            idx = i*8
            y[0+idx,:,:] = x[0+idx,:,:] * self.effm * self.effe
            y[1+idx,:,:] = x[1+idx,:,:] * self.effm * self.effm
            y[2+idx,:,:] = x[2+idx,:,:] * self.effm * self.efft
            y[3+idx,:,:] = x[3+idx,:,:] * self.effm 
            y[4+idx,:,:] = x[4+idx,:,:] * self.effe * self.effe
            y[5+idx,:,:] = x[5+idx,:,:] * self.effe * self.effm
            y[6+idx,:,:] = x[6+idx,:,:] * self.effe * self.efft
            y[7+idx,:,:] = x[7+idx,:,:] * self.effe 

        # calculate regulations for nuisance parameters
        regu = 0
        regu += (self.effe-1)**2 /(2*0.02)
        regu += (self.effm-1)**2 /(2*0.012)
        regu += (self.efft-1)**2 /(2*0.05)
        
        return y, regu


class PertLayer_Shape(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Shape,self).__init__()
        self._getDx()

        self.energye = Parameter(tc.tensor(1.), requires_grad=True)
        self.energym = Parameter(tc.tensor(1.), requires_grad=True)
        self.energyt = Parameter(tc.tensor(1.), requires_grad=True)

        self.jes     = Parameter(tc.tensor(1.), requires_grad=True)
        self.jer     = Parameter(tc.tensor(1.), requires_grad=True)
        self.btag    = Parameter(tc.tensor(1.), requires_grad=True)
        self.mistag  = Parameter(tc.tensor(1.), requires_grad=True)



    def forward(self,x):

        perturbation = tc.zeros_like(x)

        perturbation += self.dx_energye * self.energye
        perturbation += self.dx_energym * self.energym
        perturbation += self.dx_energyt * self.energyt

        perturbation += self.dx_jes * self.jes
        perturbation += self.dx_jer * self.jer
        perturbation += self.dx_btag * self.btag
        perturbation += self.dx_mistag * self.mistag

        y = x + perturbation


        # calculate regulations for nuisance parameters
        regu = 0
        regu += (self.energye)**2 /(2*1)
        regu += (self.energym)**2 /(2*1)
        regu += (self.energyt)**2 /(2*1)

        regu += (self.jes)**2 /(2*1)
        regu += (self.jer)**2 /(2*1)
        regu += (self.btag)**2 /(2*1)
        regu += (self.mistag)**2/(2*1)
        
        return y, regu

    def _getDx(self):

        dx_list = []
        baseDir = common.getBaseDirectory()

        for variation in ['EPtDown','MuPtDown','TauPtDown']:
            x0 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(''))
            x1 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(variation))
            dx = tc.from_numpy((x0-x1)).type(tc.FloatTensor)
            dx_list.append(dx)

        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/templatesX_{}Down.npy".format(variation)) 
            x2 = np.load(baseDir + "data/templates/templatesX_{}Up.npy".format(variation))
            dx = tc.from_numpy((x0-x1)/2).type(tc.FloatTensor)
            dx_list.append(dx)

        self.dx_energye = dx_list[0]
        self.dx_energym = dx_list[1]
        self.dx_energyt = dx_list[2]

        self.dx_jes     = dx_list[3]
        self.dx_jer     = dx_list[4]
        self.dx_btag    = dx_list[5]
        self.dx_mistag  = dx_list[6]

