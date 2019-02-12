from fit_tchelper import *



class PertLayer_beta(tc.nn.Module):
    def __init__(self):
        super(PertLayer_beta,self).__init__()
    
        # define parameters of interest
        self.params = Parameter(tc.tensor([.108,.1082,.1084]), requires_grad=True)

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
        p = init.normal_(tc.empty(2),mean=0,std=0.1)
        self.params = Parameter(p, requires_grad=True)
        

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
    def __init__(self):
        super(PertLayer_xs,self).__init__()
        
        # define nuisance paramters
        p = init.normal_(tc.empty(9),mean=0,std=0.1)
        self.params = Parameter(p, requires_grad=True)

    def forwardSignal(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        ttxs   = self.params[0]*0.05 + 1
        txs    = self.params[1]*0.05 + 1
        wxs    = self.params[2]*0.05 + 1
        zxs    = self.params[3]*0.05 + 1
        vvxs   = self.params[4]*0.10 + 1
        eqcdxs = self.params[5]*0.25 + 1
        mqcdxs = self.params[6]*0.25 + 1
        tqcdxs = self.params[7]*0.25 + 1
        lumin  = self.params[8]*.025 + 1
        ######################
        
        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs  * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs* lumin
        
        # QCD in relevaent channels
        for i in range(2):
            idx = i*8
            y[ 7+idx,45,:] = x[ 7+idx,45,:] * eqcdxs 
            y[ 3+idx,45,:] = x[ 3+idx,45,:] * mqcdxs 
            y[ 2+idx,45,:] = x[ 2+idx,45,:] * tqcdxs 
            y[ 6+idx,45,:] = x[ 6+idx,45,:] * tqcdxs 
        
        return y

    def forwardControl(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        ttxs   = self.params[0]*0.05 + 1
        txs    = self.params[1]*0.05 + 1
        wxs    = self.params[2]*0.05 + 1
        zxs    = self.params[3]*0.05 + 1
        vvxs   = self.params[4]*0.10 + 1
        eqcdxs = self.params[5]*0.25 + 1
        mqcdxs = self.params[6]*0.25 + 1
        tqcdxs = self.params[7]*0.25 + 1
        lumin  = self.params[8]*.025 + 1
        ######################

        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs  * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs* lumin

        # QCD in relevaent channels
        y[2,45,:] = x[2,45,:] * tqcdxs 
        y[3,45,:] = x[3,45,:] * tqcdxs
        return y
            

class PertLayer_eff(tc.nn.Module):
    def __init__(self):
        super(PertLayer_eff,self).__init__()

        # define nuisance paramters
        p = init.normal_(tc.empty(4),mean=0,std=0.1)
        self.params = Parameter(p, requires_grad=True)
    

    def forwardSignal(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        effe = self.params[0]*0.01 + 1
        effm = self.params[1]*0.01 + 1
        efft = self.params[2]*0.05 + 1
        efftmis = self.params[3]*0.08 + 1
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
                y[2+idx,14+idx2,:] = x[2+idx,14+idx2,:] * effm * efft     # tt/tw -> mutau
                y[2+idx,16+idx2,:] = x[2+idx,16+idx2,:] * effm * efftmis  # tt/tw -> muh
                y[6+idx,11+idx2,:] = x[6+idx,11+idx2,:] * effe * efft     # tt/tw -> etau
                y[6+idx,15+idx2,:] = x[6+idx,15+idx2,:] * effe * efftmis  # tt/tw -> eh
        
        return y

    def forwardControl(self, x):

        y = tc.ones_like(x) * x

        ######################
        # defining perturbation
        effe = self.params[0]*0.01 + 1
        effm = self.params[1]*0.01 + 1
        efft = self.params[2]*0.05 + 1
        efftmis = self.params[3]*0.08 + 1
        ######################

        # Z -> mumu
        y[0,:45,:] = x[0,:45,:] * effm * effm
        # Z -> ee
        y[1,:45,:] = x[1,:45,:] * effe * effe

        # Z -> tau tau
        for j in range(2):
            idx = j*21
            y[2,14+idx,:] = x[2,14+idx,:] * effm * efft    # tt/tw -> mutau
            y[2,16+idx,:] = x[2,16+idx,:] * effm * efftmis # tt/tw -> muh
            y[3,11+idx,:] = x[3,11+idx,:] * effe * efft    # tt/tw -> etau
            y[3,15+idx,:] = x[3,15+idx,:] * effe * efftmis # tt/tw -> eh
        y[2,42,:] = x[2,42,:] * effm * efftmis
        y[2,43,:] = x[2,43,:] * effm * efft
        y[3,42,:] = x[3,42,:] * effe * efftmis
        y[3,43,:] = x[3,43,:] * effe * efft

        return y


class PertLayer_itp(tc.nn.Module):
    def __init__(self, shaping):
        super(PertLayer_itp,self).__init__()
        self.configTemplateVariation(shaping)

        # define nuisance paramters
        p = init.normal_(tc.empty(self.n),mean=0,std=0.1)
        self.params = Parameter(p, requires_grad=True)


    def forwardSignal(self, x):
        perturbation = tc.zeros_like(x)
        for i in range(self.n):
            perturbation += self.dx_list_sig[i] * self.params[i]
        y = x + perturbation
        return y

    def forwardControl(self, x):
        perturbation = tc.zeros_like(x)
        for i in range(self.n):
            perturbation += self.dx_list_ctl[i] * self.params[i]
        y = x + perturbation
        return y

    def configTemplateVariation(self,shaping):
        self.dx_list_sig = templateVariation('signal', shaping)
        self.dx_list_ctl = templateVariation('control', shaping)
        self.n = len(self.dx_list_sig)


