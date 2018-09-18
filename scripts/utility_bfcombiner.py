from scipy.optimize import minimize
import numdifftools as nd
from pylab import *


class bfCombiner():
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

        #if invhess:
        # else:
        #     sigma = 1/hess.diagonal()**0.5
        #     return sigma






class bfCombiner_theta():
    def __init__(self,var,sigma,beta0):
        self.var = var
        self.sigma = sigma
        
        self.var_stat = var[0]
        self.var_syst = var[1:]
        self.sigma_stat = sigma[0]
        self.sigma_syst = sigma[1:]

        self.nTheta = var.shape[0] - 1
        self._getInv()
        
        self.beta0  = beta0
        self.param0 = np.r_[np.array([0.1081]*3),
                            0*np.ones([self.nTheta])
                           ]

        self.lsEstimator()

        
    def cost_nll (self, param):

        param_beta = param[0:3]
        param_syst = param[3:]
        
        beta = np.r_[param_beta,
                     param_beta,
                     param_beta,
                     param_beta]
        
        for i in range(self.nTheta):
            beta += self.sigma_syst[i] * param_syst[i]
        
        delta = beta-self.beta0
        chiquared = delta.dot( self.invVar_stat.dot(delta) )/2
        regulization = np.sum(param_syst**2/2)
        cost = chiquared + regulization

        return cost
    
    def lsEstimator(self):
        result = minimize(
            fun = self.cost_nll, 
            x0  = self.param0,
            method = 'SLSQP',
            bounds = [(0,1)]*3 + [(-3,3)]*self.nTheta
            )

        self.paramLS = result.x
    
    def paramSigma(self):
        hcalc = nd.Hessian(self.cost_nll, step=1e-6, method='central')
        hess  = hcalc( self.paramLS )

        if np.linalg.det(hess) is not 0:
            hessinv = np.linalg.inv(hess)
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
        
    
    def _getInv(self):
        self.invVar_stat = np.linalg.pinv(self.var_stat)
        
        invVar_syst = []
        for i in range(self.nTheta):
            mat = self.var_syst[i]
            if i == 5: # mu fakes in mu4j
                mat = mat[0:6,0:6]
                invmat = np.linalg.pinv(mat)
                invmat = np.pad(invmat,((0,6),(0,6)),'constant', constant_values=0)
            elif i == 6: # e fakes in e4j
                mat = mat[6:12,6:12]
                invmat = np.linalg.pinv(mat)
                invmat = np.pad(invmat,((6,0),(6,0)),'constant', constant_values=0)
            else:
                invmat = np.linalg.pinv(mat)
            invVar_syst.append(invmat)
        self.invVar_syst = np.array(invVar_syst)