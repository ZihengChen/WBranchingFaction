from pylab import *
import utility_common as common
import utility_commonPlot as commonp



class PredictiveModel_np():
    def __init__(self, x):
        self.x = x
        
        # define constant
        self.ll, self.hh, self.lh = .1080**2, .6760**2, .1080*.6760
        self._getShapeVariation()
        
    def predict(self, params):

        x = self.x
        
        # splite parameters
        params_beta  = params[0:3]
        params_btl   = params[3:5]
        params_xs    = params[5:14]
        params_eff   = params[14:17]
        params_shape = params[17:24]
        
        # variating templates
        h1, regu1 = self._pertTemplate_beta (x , params_beta)
        h2, regu2 = self._pertTemplate_btl  (h1, params_btl)
        h3, regu3 = self._pertTemplate_xs   (h2, params_xs)
        h4, regu4 = self._pertTemplate_eff  (h3, params_eff)
        h5, regu5 = self._pertTemplate_shape(h4, params_shape)
        
        # prediction and regulization
        y = np.sum(h5,axis=1)
        regu = regu1+regu2+regu3+regu4+regu5
        return y, regu
    
    def _pertTemplate_beta(self,x, params):

        y = x.copy()

        ######################
        # defining paramters
        bwe, bwm, bwt = params[0],params[1],params[2]
        bwh = 1 - bwe - bwm - bwt
        ######################
        
        for i in range(2):
            idx = i*21
            y[:,idx+0,:]  = x[:,idx+0,:]  * bwe*bwe /self.ll
            y[:,idx+1,:]  = x[:,idx+1,:]  * bwm*bwm /self.ll
            y[:,idx+2,:]  = x[:,idx+2,:]  * bwe*bwm /self.ll
            y[:,idx+3,:]  = x[:,idx+3,:]  * bwt**2  /self.ll
            y[:,idx+4,:]  = x[:,idx+4,:]  * bwt**2  /self.ll
            y[:,idx+5,:]  = x[:,idx+5,:]  * bwt**2  /self.ll
            y[:,idx+6,:]  = x[:,idx+6,:]  * bwt**2  /self.ll
            y[:,idx+7,:]  = x[:,idx+7,:]  * bwt**2  /self.ll
            y[:,idx+8,:]  = x[:,idx+8,:]  * bwt**2  /self.ll
            y[:,idx+9,:]  = x[:,idx+9,:]  * bwe*bwt /self.ll
            y[:,idx+10,:] = x[:,idx+10,:] * bwe*bwt /self.ll
            y[:,idx+11,:] = x[:,idx+11,:] * bwe*bwt /self.ll
            y[:,idx+12,:] = x[:,idx+12,:] * bwm*bwt /self.ll
            y[:,idx+13,:] = x[:,idx+13,:] * bwm*bwt /self.ll
            y[:,idx+14,:] = x[:,idx+14,:] * bwm*bwt /self.ll
            y[:,idx+15,:] = x[:,idx+15,:] * bwe*bwh /self.lh
            y[:,idx+16,:] = x[:,idx+16,:] * bwm*bwh /self.lh
            y[:,idx+17,:] = x[:,idx+17,:] * bwt*bwh /self.lh
            y[:,idx+18,:] = x[:,idx+18,:] * bwt*bwh /self.lh
            y[:,idx+19,:] = x[:,idx+19,:] * bwt*bwh /self.lh
            y[:,idx+20,:] = x[:,idx+20,:] * bwh*bwh /self.hh

        # calculate regulations for nuisance parameters
        regu = 0

        return y, regu
        

    def _pertTemplate_btl(self,x, params):

        y = x.copy()

        ######################
        # defining paramters
        bte, btm = params[0],params[1]
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

        # calculate regulations for nuisance parameters
        regu = 0
        regu += ((bte-1)/0.002)**2/2
        regu += ((btm-1)/0.002)**2/2
        
        return y, regu


    def _pertTemplate_xs(self, x, params):

        y = x.copy()

        ######################
        # defining paramters
        ttxs = params[0]
        txs  = params[1]
        wxs  = params[2]
        zxs  = params[3]
        vvxs = params[4]
        eqcdxs = params[5]
        mqcdxs = params[6]
        tqcdxs = params[7]
        lumin  = params[8]
        ######################
        
        y[:, 0:21,:] = x[:, 0:21,:] * ttxs * lumin
        y[:,21:42,:] = x[:,21:42,:] * txs * lumin
        y[:,42,:] = x[:,42,:] * wxs * lumin
        y[:,43,:] = x[:,43,:] * zxs * lumin
        y[:,44,:] = x[:,44,:] * vvxs * lumin
        
        y[ 7,45,:] = x[ 7,45,:] * eqcdxs 
        y[15,45,:] = x[15,45,:] * eqcdxs 
        y[ 3,45,:] = x[ 3,45,:] * mqcdxs 
        y[11,45,:] = x[11,45,:] * mqcdxs 
        y[ 2,45,:] = x[ 2,45,:] * tqcdxs 
        y[10,45,:] = x[10,45,:] * tqcdxs 
        y[ 6,45,:] = x[ 6,45,:] * tqcdxs 
        y[14,45,:] = x[14,45,:] * tqcdxs 
        
        # calculate regulations for nuisance parameters
        regu = 0
        regu += (ttxs-1)**2/(2*0.05**2)
        regu += (txs-1)**2 /(2*0.05**2)
        regu += (wxs-1)**2 /(2*0.05**2)
        regu += (zxs-1)**2 /(2*0.05**2)
        regu += (vvxs-1)**2/(2*0.10**2)
        regu += (eqcdxs-1)**2/(2*0.25**2)
        regu += (mqcdxs-1)**2/(2*0.25**2)
        regu += (tqcdxs-1)**2/(2*0.25**2)
        regu += (lumin-1)**2/(2*0.025**2)
        
        return y, regu
    
    def _pertTemplate_eff(self, x, params):

        y = x.copy()

        ######################
        # defining paramters
        effe,effm,efft = params[0],params[1],params[2]
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
        
        
        # calculate regulations for nuisance parameters
        regu = 0
        regu += (effe-1)**2 /(2*0.01**2)
        regu += (effm-1)**2 /(2*0.01**2)
        regu += (efft-1)**2 /(2*0.05**2)
        
        return y, regu
    
            
    def _pertTemplate_shape(self, x, params):
        
        y = x.copy()

        ######################
        # defining paramters
        energye,energym,energyt = params[0],params[1],params[2]
        jes,jer,btag,mistag = params[3],params[4],params[5],params[6]
        ######################

        perturbation = np.zeros_like(x)
        perturbation += self.dx_energye * energye
        perturbation += self.dx_energym * energym
        perturbation += self.dx_energyt * energyt

        perturbation += self.dx_jes * jes
        perturbation += self.dx_jer * jer
        perturbation += self.dx_btag * btag
        perturbation += self.dx_mistag * mistag

        y += perturbation
        
        
        # calculate regulations for nuisance parameters
        regu = 0
        regu += (energye)**2 /(2*1)
        regu += (energym)**2 /(2*1)
        regu += (energyt)**2 /(2*1)

        regu += (jes)**2 /(2*1)
        regu += (jer)**2 /(2*1)
        regu += (btag)**2 /(2*1)
        regu += (mistag)**2/(2*1)
        
        return y, regu

        
        
    def _getShapeVariation(self):
        
        baseDir = common.getBaseDirectory()

        # lepton energy
        dx_list = []
        for variation in ['EPtDown','MuPtDown','TauPtDown']:
            x0 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(''))
            x1 = np.load(baseDir + "data/templates/templatesX_{}.npy".format(variation))
            dx = (x0-x1)
            dx_list.append(dx)

        self.dx_energye = dx_list[0]
        self.dx_energym = dx_list[1]
        self.dx_energyt = dx_list[2]

        # Jet energy and btag
        dx_list = []
        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/templatesX_{}Down.npy".format(variation)) 
            x2 = np.load(baseDir + "data/templates/templatesX_{}Up.npy".format(variation))
            dx = (x0-x1)/2
            dx_list.append(dx)
        self.dx_jes     = dx_list[0]
        self.dx_jer     = dx_list[1]
        self.dx_btag    = dx_list[2]
        self.dx_mistag  = dx_list[3]