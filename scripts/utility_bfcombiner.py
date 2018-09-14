from scipy.optimize import minimize
import numdifftools as nd
from pylab import *


class bfCombiner():
    def __init__(self,var,beta0):
        self.var    = var
        self.invVar = np.linalg.pinv(var)
        self.beta0  = beta0
        
        self.lsEstimator()
    
    def chisquared(self,betaP):
        betaP = np.r_[betaP,betaP,betaP,betaP]
        delta = betaP-self.beta0
        chiquared = delta.dot( self.invVar.dot(delta) )
        return chiquared
    
    def lsEstimator(self):
        result = minimize(fun = self.chisquared, 
                          x0  = np.array([0.1080,0.1080,0.1080]),
                          method = 'SLSQP',
                          bounds = [(0,1),(0,1),(0,1)]
                         )
        self.betaLS = result.x

    
    def bfvar(self, invhess=True):

        hcalc = nd.Hessian(self.chisquared, step=1e-6, method='central')
        hess  = hcalc( self.betaLS )

        if invhess:

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
        else:
            sigma = 1/hess.diagonal()**0.5
            return sigma


def showCovar(covar, showCorr=False):
    lablesName = ['stat',r"$b^\tau_\mu$",r"$b^\tau_e$",
                r"$\sigma_{VV}$",r"$\sigma_{Z}$",r"$\sigma_{W}$",
                r"QCD in $\mu4j$",r"QCD in $e4j$",r"QCD in $l\tau$","L",r"$\sigma_{tt}$",r"$\sigma_{tW}$",
                r"$\epsilon_e$",r"$\epsilon_\mu$",r"$\epsilon_\tau$",r'$j \to \tau$ MisID',
                "e energy",r"$\mu$ energy",r"$\tau$ energy",
                "JES","JER","bTag","Mistag"]
    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']


    NCOL = 6
    N = covar.shape[0]
    
    NROW = int(N/NCOL)+1
    
    for i in range(NROW):
        for j in range(NCOL):
            index = i*NCOL + j
            if index<N:
                matrix = covar[index]
                if showCorr:
                    sigma  = matrix.diagonal()**0.5
                    matrix = matrix/np.outer(sigma, sigma)
                    norm=1
                #norm = np.abs(matrix).max()
                else:
                    norm = 5e-6
                plt.subplot(NROW,NCOL,index+1)
                plt.imshow(matrix,cmap='RdBu_r',vmin=-norm,vmax=norm)
                plt.axvline(2.5,c='k',lw=0.5,linestyle='--')
                plt.axvline(5.5,c='k',lw=0.5,linestyle='--')
                plt.axvline(8.5,c='k',lw=0.5,linestyle='--')
                plt.axhline(2.5,c='k',lw=0.5,linestyle='--')
                plt.axhline(5.5,c='k',lw=0.5,linestyle='--')
                plt.axhline(8.5,c='k',lw=0.5,linestyle='--')
                plt.grid('True')
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(lablesName[index])
            else:
                plt.xticks([])
                plt.yticks([])

    plt.subplot(NROW,NCOL,NROW*NCOL)
    plt.imshow(np.sum(covar,axis=0),cmap='RdBu_r',vmin=-norm,vmax=norm)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Total')


def showSingleCovar(covar, showCorr=False, norm=1e-6, titleName=''):

    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']

    matrix = covar

    if showCorr:
        sigma  = matrix.diagonal()**0.5
        matrix = matrix/np.outer(sigma, sigma)
        norm=1

    plt.imshow(matrix,cmap='RdBu_r',vmin=-norm,vmax=norm)
    plt.xticks(ticks_pos,ticks_name)
    plt.yticks(ticks_pos,ticks_name)
    plt.title(titleName,fontsize=14)
    plt.grid('False')

    plt.axvline(2.5,c='k',lw=1,linestyle='--')
    plt.axvline(5.5,c='k',lw=1,linestyle='--')
    plt.axvline(8.5,c='k',lw=1,linestyle='--')
    plt.axhline(2.5,c='k',lw=1,linestyle='--')
    plt.axhline(5.5,c='k',lw=1,linestyle='--')
    plt.axhline(8.5,c='k',lw=1,linestyle='--')

    #plt.colorbar()




