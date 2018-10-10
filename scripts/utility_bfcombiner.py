from scipy.optimize import minimize
import numdifftools as nd
from pylab import *
import utility_common as common
from fit_model import *


class BFCombiner():
    def __init__(self,var,beta0, paramTypy='beta'):
        self.var    = var
        self.invVar = np.linalg.pinv(var)
        self.beta0  = beta0
        self.paramTypy = paramTypy
        
        self.lsEstimator()
    

    def chisquared_beta(self,param):
        # three paramters
        # beta_e, beta_m, beta_tau
        beta = param
        beta = np.r_[beta,beta,beta,beta]
        delta = beta-self.beta0
        chiquared = delta.dot( self.invVar.dot(delta) )
        chiquared /= 2
        return chiquared

    def chisquared_r(self,param):
        # two paramters
        # r1,r2,l
        r1,r2,l = param[0],param[1],param[2]

        lep = 1./r1 + 1./r2 + 1.

        bwe = l/(lep * r1)
        bwm = l/(lep * r2)
        bwt = l/(lep *  1)

        beta = np.array([ bwe,bwm,bwt])

        beta = np.r_[beta,beta,beta,beta]
        delta = beta-self.beta0
        chiquared = delta.dot( self.invVar.dot(delta) ) 

        chiquared /= 2
        return chiquared



    def lsEstimator(self):

        if self.paramTypy=='r':
            result = minimize(
                fun = self.chisquared_r, 
                x0  = np.array([1.,1.,0.3240]),
                method = 'SLSQP',
                bounds = [(0.9,1.1),(0.9,1.1),(0,1)]
                )
        else:
            result = minimize(
                fun = self.chisquared_beta, 
                x0  = np.array([0.1080,0.1080,0.1080]),
                method = 'SLSQP',
                bounds = [(0,1),(0,1),(0,1)]
                )

        self.paramLS = result.x

    
    def paramSigma(self, invhess=True):
        if self.paramTypy =='r':
            hcalc = nd.Hessian(self.chisquared_r, step=1e-6, method='central')
        else:
            hcalc = nd.Hessian(self.chisquared_beta, step=1e-6, method='central')

        hess  = hcalc( self.paramLS )

        
        if np.linalg.det(hess) is not 0:
            hessinv = np.linalg.pinv(hess)
            sigmasq = hessinv.diagonal()

            if (sigmasq>=0).all():
                sigma   = np.sqrt(sigmasq)
                corvar  = hessinv/np.outer(sigma, sigma)
                return sigma, corvar
            else:
                print("Failed for boundaries, negetive sigma^2 exist in observed inv-hessian ")
                return np.zeros([3]), np.zeros([3,3])
        else:
            print("Failed for sigularity in Hessian matrix")
            return np.zeros([3]), np.zeros([3,3])



class BFCombiner_theta():
    def __init__(self,beta0, controlRegion=None):
        self.beta0 = beta0

        # read sigma of beta0
        baseDir = common.getBaseDirectory()
        sig     = np.load(baseDir + 'data/combine/sigma.npy')
        var     = np.load(baseDir + 'data/combine/covar.npy')
        self.sig_syst  = sig[1:]
        self.var_stat  = var[0]
        self.ivar_stat = np.linalg.pinv(self.var_stat)
        
        # some configuration
        self.n      = var.shape[0]-1
        self.param0 = np.r_[0.1081*np.ones(3), np.zeros(self.n)]

        # config control region
        self.controlRegion = controlRegion
        if not controlRegion is None:
            self.model1 = controlRegion[0]
            self.X1     = controlRegion[1]
            self.Y1     = controlRegion[2]

        self.lsEstimator()

        
    def loss (self, param):
        
        # perturbating beta0 with systematics
        param_syst  = param[3:]
        perturbation = np.zeros_like(self.beta0)
        for i in range(param_syst.size):
            perturbation += self.sig_syst[i] * param_syst[i]

        # calcuate chisquared
        beta  = np.r_[param[:3],param[:3],param[:3],param[:3]]
        delta = beta - (self.beta0 + perturbation)
        loss  = delta.dot( self.ivar_stat.dot(delta) )/2

        # add regulation of systematics
        loss += np.sum( 0.5*(param_syst)**2 )

        # add control
        if not self.controlRegion is None:
            y1    = self.model1.predict(param)
            loss += np.sum( (y1-self.Y1)**2/(2*self.Y1) )
            
        return loss
    
    def lsEstimator(self):
        paramBounds = [(0,1)]*3+[(-3,3)]*self.n
        result = minimize(fun=self.loss, x0=self.param0, method='SLSQP',
                          bounds=paramBounds)  
        self.result = result.x
    
    def paramSigma(self):
        hcalc = nd.Hessian(self.loss, step=1e-6, method='central')
        hess  = hcalc( self.result )
        ihess = np.linalg.inv(hess)
        sig   = np.sqrt(ihess.diagonal())
        cor   = ihess/np.outer(sig, sig)

        return sig, cor
        