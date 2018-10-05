

from pylab import *
from utility_bfsolver import *
from scipy.optimize import minimize
import numdifftools as nd

BWPDG = np.array([0.1071,0.1063,0.1138])
BWLPDG = 0.1080


class Fitter():
    def __init__(self):
        self.baseDir = common.getBaseDirectory() 

        self.counts = pd.read_pickle( self.baseDir  + "data/counts/count_summary.pkl")
        self.count0 = self.counts.loc['nominal']
        
    def nll(self,param):
        
        beta   = param[:3]
        thetaC = param[3:6]
        theta  = param[6:]
        
        
        a = self.count0.acc.copy()
        nmcbg = self.count0.nmcbg.copy()
        nfake = self.count0.nfake.copy()
        ndata = self.count0.ndata.copy()
        
        for i in range(theta.size):
            a += theta[i] * self.counts.acc[i+1]
            nmcbg += theta[i] * self.counts.nmcbg[i+1]
            nfake += theta[i] * self.counts.nfake[i+1]
            ndata += theta[i] * self.counts.ndata[i+1]

        _xs = 832*(1+theta[0])+35.85*2*(1+theta[1])
        _lumin = 35847 * (1+thetaC[0])
        _bte   = 0.1785 + 0.0005 * (1+thetaC[1])
        _btm   = 0.1736 + 0.0005 * (1+thetaC[2])
        
        nll = 0
        for ich in range(16):
            y = ndata[ich]
            f = Yield(a[ich],nmcbg[ich],nfake[ich],
                      xs=_xs,lumin=_lumin,bte=_bte,btm=_btm).predict(BW=beta)

            #nll += y*np.log(f) - f
            nll += (f-y)**2/y/2
        nll += np.sum(theta**2)/2 
        nll += np.sum(thetaC**2)/2  
        return nll

            
    def fit(self):
        result = minimize(
                fun = self.nll, 
                x0  = np.array([0.1080]*3 + [0]*21 ),
                method = 'SLSQP',
                bounds = [(0,1)]*3 + [(-3,3)]*21
                )

        self.paramML = result.x
        
        
    def paramSigma(self):
        hcalc = nd.Hessian(self.nll, step=1e-6, method='central')
        hess  = hcalc( self.paramML )

        if np.linalg.det(hess) is not 0:
            hessinv = np.linalg.inv(hess)
            sigmasq = hessinv.diagonal()

            if (sigmasq>=0).all():
                sigma   = np.sqrt(sigmasq)
                corvar  = hessinv/np.outer(sigma, sigma)
                return sigma, corvar
            else:
                print("Failed for boundaries, negetive sigma^2 exist in observed inv-hessian ")
        else:
            print("Failed for sigularity in Hessian matrix")     