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
        return chiquared

    def chisquared_r(self,param):
        # two paramters
        # r, l

        beta = np.array([   param[1]/(2+param[0]), 
                            param[1]/(2+param[0]),
                            param[1]/(2+param[0]) * param[0] ])

        beta = np.r_[beta,beta,beta,beta]
        delta = beta-self.beta0
        chiquared = delta.dot( self.invVar.dot(delta) )
        return chiquared


    def lsEstimator(self):

        if self.paramTypy=='r':
            result = minimize(
                fun = self.chisquared_r, 
                x0  = np.array([1,0.3240]),
                method = 'SLSQP',
                bounds = [(0,2),(0,1)]
                )
        else:
            result = minimize(
                fun = self.chisquared_beta, 
                x0  = np.array([0.1080,0.1080,0.1080]),
                method = 'SLSQP',
                bounds = [(0,1),(0,1),(0,1)]
                )

        self.paramLS = result.x

    
    def bfvar(self, invhess=True):
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




###################################
# Plotting
###################################
def showCovar(covar,sameCNorm=False):
    lablesName = ['stat',r"$b^\tau_\mu$",r"$b^\tau_e$",
                r"$\sigma_{VV}$",r"$\sigma_{Z}$",r"$\sigma_{W}$",
                r"QCD in $\mu4j$",r"QCD in $e4j$",r"QCD in $l\tau$","L",r"$\sigma_{tt}$",r"$\sigma_{tW}$",
                r"$\epsilon_e$",r"$\epsilon_\mu$",r"$\epsilon_\tau$",r'$j \to \tau$ MisID',
                "e energy",r"$\mu$ energy",r"$\tau$ energy",
                "JES","JER","bTag","Mistag"]
    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']
    
    n0,n1,n2,n3 = 1E-8, 5E-8, 1E-6, 8E-6
    if sameCNorm:
        n0,n1,n2,n3 = 1E-6,1E-6,1E-6,1E-6

    normList = [n2,n0,n0,n0,
                n1,n1,n2,n2,
                n2,n1,n0,n0,
                n2,n2,n3,n3,
                n1,n1,n2,n3,
                n2,n3,n1,n3]


    NCOL = 6
    N = covar.shape[0]
    
    NROW = int(N/NCOL)+1
    
    for i in range(NROW):
        for j in range(NCOL):
            index = i*NCOL + j
            if index<N:
                matrix = covar[index]

                norm = normList[index]
                plt.subplot(NROW,NCOL,index+1)
                
                plt.imshow(matrix,cmap='RdBu_r',vmin=-norm,vmax=norm)

                plt.xticks([2.5,5.5,8.5],['','',''])
                plt.yticks([2.5,5.5,8.5],['','',''])
                plt.grid('True',lw=1,linestyle='--')
                plt.title(lablesName[index]+ ' [{:1.0E}]'.format(norm),fontsize=12)


    norm = 8E-6
    plt.subplot(NROW,NCOL,NROW*NCOL)
    plt.imshow(np.sum(covar,axis=0),cmap='RdBu_r',vmin=-norm,vmax=norm,)
    # cbar = plt.colorbar( ticks=[-norm,  0,  norm],shrink=0.8)
    # cbar.ax.set_yticklabels(['-{:1.0E}'.format(norm), '0', '{:1.0E}'.format(norm) ],fontsize=8)
    
    plt.xticks([])
    plt.xticks([2.5,5.5,8.5],['','',''])
    plt.yticks([2.5,5.5,8.5],['','',''])
    plt.grid('True',lw=1,linestyle='--')
    plt.title('Total'+ ' [{:1.0E}]'.format(norm),fontsize=12)



def showSingleCovar(covar, norm=1e-6, titleName=''):

    ticks_pos = [1,4,7,10]
    ticks_name = [r'$\mu 1b$',r'$\mu 2b$',r'$e 1b$',r'$e 2b$']

    matrix = covar

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


def showParameterCov(corr):
    ticks = [r'$\bar{\beta}_e$',r'$\bar{\beta}_\mu$',r'$\bar{\beta}_\tau$']
    plt.figure(figsize=(6,4),facecolor='w')
    plt.imshow(corr,cmap='PRGn_r',vmax=1,vmin=-1)
    plt.xticks([0,1,2],ticks, fontsize=14)
    plt.yticks([0,1,2],ticks, fontsize=14)
    for i in range(3):
        for j in range(3):
            v = corr[i,j]
            if abs(v)>0.5:
                fontc = 'w'
            else:
                fontc = 'k'
            plt.text(i-0.2,j+0.1,'{:.2f}'.format(v), color=fontc, fontsize=14)
    plt.colorbar( ticks=[-1, 0, 1],shrink=1)


