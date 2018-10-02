from torch_helper import *

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

class PertLayer_Beta(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Beta,self).__init__()
    
        # define parameters of interest
        self.bwe   = Parameter(tc.tensor(.109), requires_grad=True)
        self.bwm   = Parameter(tc.tensor(.108), requires_grad=True)
        self.bwt   = Parameter(tc.tensor(.107), requires_grad=True)
        
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
            y[:,idx+3,:]  = x[:,idx+3,:]  * self.bwt**2 /self.ll
            y[:,idx+4,:]  = x[:,idx+4,:]  * self.bwt**2 /self.ll
            y[:,idx+5,:]  = x[:,idx+5,:]  * self.bwt**2 /self.ll
            y[:,idx+6,:]  = x[:,idx+6,:]  * self.bwt**2 /self.ll
            y[:,idx+7,:]  = x[:,idx+7,:]  * self.bwt**2 /self.ll
            y[:,idx+8,:]  = x[:,idx+8,:]  * self.bwt**2 /self.ll
            y[:,idx+9,:]  = x[:,idx+9,:]  * self.bwe*self.bwt /self.ll
            y[:,idx+10,:] = x[:,idx+10,:] * self.bwe*self.bwt /self.ll
            y[:,idx+11,:] = x[:,idx+11,:] * self.bwe*self.bwt /self.ll
            y[:,idx+12,:] = x[:,idx+12,:] * self.bwm*self.bwt /self.ll
            y[:,idx+13,:] = x[:,idx+13,:] * self.bwm*self.bwt /self.ll
            y[:,idx+14,:] = x[:,idx+14,:] * self.bwm*self.bwt /self.ll
            y[:,idx+15,:] = x[:,idx+15,:] * self.bwe * bwh /self.lh
            y[:,idx+16,:] = x[:,idx+16,:] * self.bwm * bwh /self.lh
            y[:,idx+17,:] = x[:,idx+17,:] * self.bwt * bwh /self.lh
            y[:,idx+18,:] = x[:,idx+18,:] * self.bwt * bwh /self.lh
            y[:,idx+19,:] = x[:,idx+19,:] * self.bwt * bwh /self.lh
            y[:,idx+20,:] = x[:,idx+20,:] * bwh * bwh /self.hh

        # calculate regulations for nuisance parameters
        regu = 0
        return y, regu


class PertLayer_Btl(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Btl,self).__init__()

        # define nuisance paramters
        self.bte   = Parameter(tc.tensor(1.0), requires_grad=True)
        self.btm   = Parameter(tc.tensor(1.1), requires_grad=True)
        

    def forward(self, x):
        
        bth = (1 - self.bte*0.1785 - self.btm*0.1736) / 0.6479
        
        y = tc.ones_like(x) * x
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * 1
            y[:,idx+1,:]  = x[:,idx+1,:]  * 1
            y[:,idx+2,:]  = x[:,idx+2,:]  * 1
            y[:,idx+3,:]  = x[:,idx+3,:]  * self.bte*self.bte
            y[:,idx+4,:]  = x[:,idx+4,:]  * self.btm*self.btm 
            y[:,idx+5,:]  = x[:,idx+5,:]  * self.bte*self.btm 
            y[:,idx+6,:]  = x[:,idx+6,:]  * self.bte*bth
            y[:,idx+7,:]  = x[:,idx+7,:]  * self.btm*bth 
            y[:,idx+8,:]  = x[:,idx+8,:]  * bth * bth 
            y[:,idx+9,:]  = x[:,idx+9,:]  * self.bte
            y[:,idx+10,:] = x[:,idx+10,:] * self.btm
            y[:,idx+11,:] = x[:,idx+11,:] * bth
            y[:,idx+12,:] = x[:,idx+12,:] * self.bte
            y[:,idx+13,:] = x[:,idx+13,:] * self.btm
            y[:,idx+14,:] = x[:,idx+14,:] * bth
            y[:,idx+15,:] = x[:,idx+15,:] * 1
            y[:,idx+16,:] = x[:,idx+16,:] * 1
            y[:,idx+17,:] = x[:,idx+17,:] * self.bte
            y[:,idx+18,:] = x[:,idx+18,:] * self.btm
            y[:,idx+19,:] = x[:,idx+19,:] * bth
            y[:,idx+20,:] = x[:,idx+20,:] * 1

        # calculate regulations for nuisance parameters
        regu = 0
        regu += 0.5*((self.bte-1)/0.002)**2 
        regu += 0.5*((self.btm-1)/0.002)**2         
        return y, regu

    
class PertLayer_Xs(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Xs,self).__init__()
    
        # define nuisance paramters
        self.ttxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.txs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.wxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.zxs   = Parameter(tc.tensor(1.), requires_grad=True)
        self.vvxs  = Parameter(tc.tensor(1.), requires_grad=True)
        self.eqcdxs= Parameter(tc.tensor(1.), requires_grad=True)
        self.mqcdxs= Parameter(tc.tensor(1.), requires_grad=True)
        self.tqcdxs= Parameter(tc.tensor(1.), requires_grad=True)
        self.lumin = Parameter(tc.tensor(1.), requires_grad=True)
            

    def forward(self, x):
        
        y = tc.ones_like(x) * x
        y[:, 0:21,:] = x[:, 0:21,:] * self.ttxs * self.lumin
        y[:,21:42,:] = x[:,21:42,:] * self.txs  * self.lumin
        y[:,42,:] = x[:,42,:] * self.wxs  * self.lumin
        y[:,43,:] = x[:,43,:] * self.zxs  * self.lumin
        y[:,44,:] = x[:,44,:] * self.vvxs * self.lumin
        
        y[ 7,45,:] = x[ 7,45,:] * self.eqcdxs 
        y[15,45,:] = x[15,45,:] * self.eqcdxs 
        y[ 3,45,:] = x[ 3,45,:] * self.mqcdxs 
        y[11,45,:] = x[11,45,:] * self.mqcdxs 
        y[ 2,45,:] = x[ 2,45,:] * self.tqcdxs 
        y[10,45,:] = x[10,45,:] * self.tqcdxs 
        y[ 6,45,:] = x[ 6,45,:] * self.tqcdxs 
        y[14,45,:] = x[14,45,:] * self.tqcdxs 
        
        
        regu = 0
        regu += 0.5*((self.ttxs-1)/0.05)**2 
        regu += 0.5*((self.txs -1)/0.05)**2 
        regu += 0.5*((self.wxs -1)/0.05)**2 
        regu += 0.5*((self.zxs -1)/0.05)**2 
        regu += 0.5*((self.vvxs-1)/0.10)**2 
        
        regu += 0.5*((self.eqcdxs-1)/0.25)**2 
        regu += 0.5*((self.mqcdxs-1)/0.25)**2 
        regu += 0.5*((self.tqcdxs-1)/0.25)**2 
        regu += 0.5*((self.lumin -1)/0.025)**2 
        
        return y, regu

class PertLayer_Eff(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Eff,self).__init__()
    
        # define nuisance paramters
        self.effe  = Parameter(tc.tensor(1.), requires_grad=True)
        self.effm  = Parameter(tc.tensor(1.), requires_grad=True)
        self.efft  = Parameter(tc.tensor(1.), requires_grad=True)
            

    def forward(self, x):
        qcd = -1
        
        y = tc.ones_like(x) * x
        for i in range(2):
            idx = i*8
            y[0+idx,:qcd,:] = x[0+idx,:qcd,:] * self.effm * self.effe
            y[1+idx,:qcd,:] = x[1+idx,:qcd,:] * self.effm * self.effm
            y[2+idx,:qcd,:] = x[2+idx,:qcd,:] * self.effm * self.efft
            y[3+idx,:qcd,:] = x[3+idx,:qcd,:] * self.effm 
            y[4+idx,:qcd,:] = x[4+idx,:qcd,:] * self.effe * self.effe
            y[5+idx,:qcd,:] = x[5+idx,:qcd,:] * self.effe * self.effm
            y[6+idx,:qcd,:] = x[6+idx,:qcd,:] * self.effe * self.efft
            y[7+idx,:qcd,:] = x[7+idx,:qcd,:] * self.effe 

        # calculate regulations for nuisance parameters
        regu = 0
        regu += 0.5*((self.effe-1)/0.01)**2 
        regu += 0.5*((self.effm-1)/0.01)**2 
        regu += 0.5*((self.efft-1)/0.05)**2 
        
        return y, regu


class PertLayer_Shape(tc.nn.Module):
    def __init__(self):
        super(PertLayer_Shape,self).__init__()
        self._getShapeVariation()

        self.energye = Parameter(tc.tensor(0.), requires_grad=True)
        self.energym = Parameter(tc.tensor(0.), requires_grad=True)
        self.energyt = Parameter(tc.tensor(0.), requires_grad=True)

        self.jes     = Parameter(tc.tensor(0.), requires_grad=True)
        self.jer     = Parameter(tc.tensor(0.), requires_grad=True)
        self.btag    = Parameter(tc.tensor(0.), requires_grad=True)
        self.mistag  = Parameter(tc.tensor(0.), requires_grad=True)



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
        regu += 0.5*(self.energye)**2
        regu += 0.5*(self.energym)**2
        regu += 0.5*(self.energyt)**2

        regu += 0.5*(self.jes)**2
        regu += 0.5*(self.jer)**2
        regu += 0.5*(self.btag)**2
        regu += 0.5*(self.mistag)**2
        
        return y, regu

    def _getShapeVariation(self):
        
        baseDir = common.getBaseDirectory()

        # lepton energy
        dx_list = []
        for variation in ['EPtDown','MuPtDown','TauPtDown']:
            x0 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(''))
            x1 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(variation))
            dx = tc.from_numpy((x0-x1)).type(tc.FloatTensor).to(device)
            dx_list.append(dx)

        self.dx_energye = dx_list[0]
        self.dx_energym = dx_list[1]
        self.dx_energyt = dx_list[2]

        # Jet energy and btag
        dx_list = []
        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/templatesX_{}Down.npy".format(variation)) 
            x2 = np.load(baseDir + "data/templates/templatesX_{}Up.npy".format(variation))
            dx = tc.from_numpy((x0-x1)/2).type(tc.FloatTensor).to(device)
            dx_list.append(dx)
        self.dx_jes     = dx_list[0]
        self.dx_jer     = dx_list[1]
        self.dx_btag    = dx_list[2]
        self.dx_mistag  = dx_list[3]














# epoch = 0
# while True:
#     y,regu = model.forward(X)
#     loss = tc.sum( (y-Y)**2/(2*Y) ) + regu
#     optimizer.zero_grad()
#     loss.backward(retain_graph=True)
#     optimizer.step()
    
#     # calculate the update
#     params = dict(model.named_parameters())
#     bwe = params['layer_beta.bwe'].data
#     bwm = params['layer_beta.bwm'].data
#     bwt = params['layer_beta.bwt'].data
#     if epoch > 5000:
#         if (tc.abs(bwe-bwe0)<1e-9) and (tc.abs(bwm-bwm0)<1e-9) and (tc.abs(bwt-bwt0)<1e-9):
#             break
#     if epoch % 100 == 0:
#         print(bwe,bwm,bwt)
#     bwe0,bwm0,bwt0 = bwe,bwm,bwt
#     epoch += 1
    
    
# print(bwe,bwm,bwt)