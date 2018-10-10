from fit_torch_helper import *

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")


class PertLayer_beta(tc.nn.Module):
    def __init__(self):
        super(PertLayer_beta,self).__init__()
    
        # define parameters of interest
        self.params = Parameter(tc.tensor([.107,.108,.109]), requires_grad=True)

    def forward(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        bwe = self.params[0]/0.1080
        bwm = self.params[1]/0.1080
        bwt = self.params[2]/0.1080
        bwh = (1 - tc.sum(self.params) )/0.6760
        ######################
        
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * bwe*bwe 
            y[:,idx+1,:]  = x[:,idx+1,:]  * bwm*bwm 
            y[:,idx+2,:]  = x[:,idx+2,:]  * bwe*bwm
            y[:,idx+3,:]  = x[:,idx+3,:]  * bwt**2
            y[:,idx+4,:]  = x[:,idx+4,:]  * bwt**2 
            y[:,idx+5,:]  = x[:,idx+5,:]  * bwt**2   
            y[:,idx+6,:]  = x[:,idx+6,:]  * bwt**2   
            y[:,idx+7,:]  = x[:,idx+7,:]  * bwt**2   
            y[:,idx+8,:]  = x[:,idx+8,:]  * bwt**2   
            y[:,idx+9,:]  = x[:,idx+9,:]  * bwe*bwt  
            y[:,idx+10,:] = x[:,idx+10,:] * bwe*bwt  
            y[:,idx+11,:] = x[:,idx+11,:] * bwe*bwt  
            y[:,idx+12,:] = x[:,idx+12,:] * bwm*bwt  
            y[:,idx+13,:] = x[:,idx+13,:] * bwm*bwt  
            y[:,idx+14,:] = x[:,idx+14,:] * bwm*bwt  
            y[:,idx+15,:] = x[:,idx+15,:] * bwe*bwh  
            y[:,idx+16,:] = x[:,idx+16,:] * bwm*bwh  
            y[:,idx+17,:] = x[:,idx+17,:] * bwt*bwh  
            y[:,idx+18,:] = x[:,idx+18,:] * bwt*bwh  
            y[:,idx+19,:] = x[:,idx+19,:] * bwt*bwh  
            y[:,idx+20,:] = x[:,idx+20,:] * bwh*bwh  

        return y


class PertLayer_btl(tc.nn.Module):
    def __init__(self):
        super(PertLayer_btl,self).__init__()

        # define nuisance paramters
        self.params = Parameter(tc.tensor([0,0.01]), requires_grad=True)
        

    def forward(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        bte = self.params[0]*0.002 + 1.0
        btm = self.params[1]*0.002 + 1.0
        bth = (1 - bte*0.1785-btm*0.1736) / 0.6479
        ######################

        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * 1
            y[:,idx+1,:]  = x[:,idx+1,:]  * 1
            y[:,idx+2,:]  = x[:,idx+2,:]  * 1
            y[:,idx+3,:]  = x[:,idx+3,:]  * bte*bte
            y[:,idx+4,:]  = x[:,idx+4,:]  * btm*btm 
            y[:,idx+5,:]  = x[:,idx+5,:]  * bte*btm
            y[:,idx+6,:]  = x[:,idx+6,:]  * bte*bth
            y[:,idx+7,:]  = x[:,idx+7,:]  * btm*bth
            y[:,idx+8,:]  = x[:,idx+8,:]  * bth*bth
            y[:,idx+9,:]  = x[:,idx+9,:]  * bte
            y[:,idx+10,:] = x[:,idx+10,:] * btm
            y[:,idx+11,:] = x[:,idx+11,:] * bth
            y[:,idx+12,:] = x[:,idx+12,:] * bte
            y[:,idx+13,:] = x[:,idx+13,:] * btm
            y[:,idx+14,:] = x[:,idx+14,:] * bth
            y[:,idx+15,:] = x[:,idx+15,:] * 1
            y[:,idx+16,:] = x[:,idx+16,:] * 1
            y[:,idx+17,:] = x[:,idx+17,:] * bte
            y[:,idx+18,:] = x[:,idx+18,:] * btm
            y[:,idx+19,:] = x[:,idx+19,:] * bth
            y[:,idx+20,:] = x[:,idx+20,:] * 1
        
        return y

    
class PertLayer_xs(tc.nn.Module):
    def __init__(self, controlTauID):
        super(PertLayer_xs,self).__init__()
        self.controlTauID = controlTauID
    
        # define nuisance paramters
        self.params = Parameter(1e-3*tc.ones(9), requires_grad=True)
            

    def forward(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        ttxs = self.params[0]*0.05 + 1
        txs  = self.params[1]*0.05 + 1
        wxs  = self.params[2]*0.05 + 1
        zxs  = self.params[3]*0.05 + 1
        vvxs = self.params[4]*0.10 + 1
        eqcdxs = self.params[5]*0.25 + 1
        mqcdxs = self.params[6]*0.25 + 1
        tqcdxs = self.params[7]*0.25 + 1
        lumin  = self.params[8]*0.025 + 1
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
        
        return y

class PertLayer_eff(tc.nn.Module):
    def __init__(self, controlTauID):
        super(PertLayer_eff,self).__init__()
        self.controlTauID = controlTauID

        # define nuisance paramters
        self.params = Parameter(1e-3*tc.ones(3), requires_grad=True)
            

    def forward(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        effe = self.params[0]*0.01 + 1
        effm = self.params[1]*0.01 + 1
        efft = self.params[2]*0.05 + 1
        ######################
        qcd = -1
        
        for i in range(2):
            idx = i*8
            y[0+idx,:qcd,:] = x[0+idx,:qcd,:] * effm * effe
            y[1+idx,:qcd,:] = x[1+idx,:qcd,:] * effm * effm
            y[2+idx,:qcd,:] = x[2+idx,:qcd,:] * effm * efft
            y[3+idx,:qcd,:] = x[3+idx,:qcd,:] * effm 
            y[4+idx,:qcd,:] = x[4+idx,:qcd,:] * effe * effe
            y[5+idx,:qcd,:] = x[5+idx,:qcd,:] * effe * effm
            y[6+idx,:qcd,:] = x[6+idx,:qcd,:] * effe * efft
            y[7+idx,:qcd,:] = x[7+idx,:qcd,:] * effe 
        
        if self.controlTauID:
            y[16,:qcd,:] = x[16,:qcd,:] * effm * effm # ZToMuMu
            y[17,:qcd,:] = x[17,:qcd,:] * effe * effe # ZToEE
            y[18,:qcd,:] = x[18,:qcd,:] * effm * efft # ZToTauTauToMuH
            y[19,:qcd,:] = x[19,:qcd,:] * effe * efft # ZToTauTauToEH
        
        return y


class PertLayer_itp(tc.nn.Module):
    def __init__(self, controlTauID):
        super(PertLayer_interpolation,self).__init__()
        self.controlTauID = controlTauID
        self._getShapeVariation()
        

        # define nuisance paramters
        self.params = Parameter(1e-3*tc.ones(7), requires_grad=True)



    def forward(self,x):

        ######################
        # defining perturbation
        energye,energym,energyt = self.params[0],self.params[1],self.params[2]
        jes,jer,btag,mistag = self.params[3],self.params[4],self.params[5],self.params[6]
        ######################

        perturbation = tc.zeros_like(x)
        perturbation += self.dx_energye * energye
        perturbation += self.dx_energym * energym
        perturbation += self.dx_energyt * energyt

        perturbation += self.dx_jes * jes
        perturbation += self.dx_jer * jer
        perturbation += self.dx_btag * btag
        perturbation += self.dx_mistag * mistag

        y = x + perturbation
        
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

            if not self.controlTauID:
                x0 = x0[0:16]
                x1 = x1[0:16]

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