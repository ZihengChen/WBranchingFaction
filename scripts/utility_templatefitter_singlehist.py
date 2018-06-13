from scipy.optimize import minimize
from pylab import *
import numdifftools as nd

class TemplateFitter:
    def __init__(self, 
                 template, templatevar2,
                 target,   targetvar2,
                 lock = None):

        self.ntemplate      = template.shape[0]
        
        self.template       = template
        self.templatevar2   = templatevar2
        self.target         = target
        self.targetvar2     = targetvar2

            
        if lock is None:
            self.lock = np.zeros(self.ntemplate)
        else:
            self.lock = lock            
        
        self.adjust_init     = np.ones(self.ntemplate)
        self.adjust          = self.fit()
        self.sigma,self.corr = self.fitvar()
        
    def chisquared(self, adjust):
        
        totalTemplate       = np.dot(adjust, self.template)
        totalTemplateVar2   = np.dot(adjust**2, self.templatevar2)

        diff_TemplateTarget = totalTemplate - self.target
        var2_TemplateTarget = totalTemplateVar2 + self.targetvar2
        
        nonzero = var2_TemplateTarget > 0
        diff_TemplateTarget = diff_TemplateTarget[nonzero]
        var2_TemplateTarget = var2_TemplateTarget[nonzero]
        chisquared          = diff_TemplateTarget**2/var2_TemplateTarget
        chisquared          = np.sum(chisquared)
        return chisquared
        
    def penalty(self, adjust):
        change_2 = (adjust - np.ones_like(adjust))**2
        change_var2 = np.sum(self.templatevar2,axis = 1)/np.sum(self.template,axis = 1)**2
        change_chi2 = change_2/change_var2
        temp   = np.dot(change_chi2, self.lock)
        return temp
    
    def cost(self, adjust):
        temp = self.chisquared(adjust) + self.penalty(adjust)
        return temp
    
    def fit(self):
        result = minimize(fun = self.cost, 
                          x0  = self.adjust_init,
                          method = 'SLSQP',
                          bounds = self.ntemplate * [(0,10),]
                         )
        return result.x

    def fitvar(self):
        n     = self.ntemplate
        hcalc = nd.Hessian(self.chisquared, step=1e-4, method='central')
        #hcalc = nd.Hessian(self.cost, step=1e-4, method='central')
        hess  = hcalc( self.adjust )

        if np.linalg.det(hess) is not 0:
            hessinv = np.linalg.inv(hess)
            sigmasq = hessinv.diagonal()

            if (sigmasq>=0).all():
                sigma   = np.sqrt(sigmasq)
                corvar  = hessinv/np.outer(sigma, sigma)
                return sigma, corvar
            else:
                print("Failed for boundaries, negetive sigma^2 exist in observed inv-hessian ")
                return np.zeros([n]), np.zeros([n,n])
        else:
            print("Failed for sigularity in Hessian matrix")
            return np.zeros([n]), np.zeros([n,n])

class TemplateFitter_Hist:
    def __init__(self, v,a,b,step,df_list,
                 lock = np.array([0,0,0,0,0,0])
                 ):
        self.v = v
        self.a = a
        self.b = b
        self.step   = step
        self.mybin  = np.arange(a,b,step)
        self.center = self.mybin[1:]-self.step/2

        self.variable_list = [mc[v] for mc in df_list[0:-1]]
        self.weight_list   = [mc.eventWeight for mc in df_list[0:-1]]
        self.Datav  = df_list[-1][v]
        self.Dataw  = df_list[-1].eventWeight
        
        self.lock    = lock

        self.getFittingResult()
        self.printFittingResult()

        # self.adjust, adjustweight_list in getFittingResult
        # self.mc,mctot, mc0,mctot0, hdata in plotFittingResult
    
    def settingPlot(self,xl,label_list,color_list,
    
                    logscale   = False,
                    isstacked  = True,
                    figuresize = (6,5.4)
                    ):
        self.xl = xl
        self.label_list = label_list
        self.color_list = color_list

        self.logscale   = logscale
        self.isstacked  = isstacked
        self.figuresize = figuresize


    def getMCDataHistAndErr(self):

        mchist, mcerr2 = [],[]
        for i in range(len(self.variable_list)):
            h,_    = np.histogram(self.variable_list[i], self.mybin, weights=self.weight_list[i])
            err2,_ = np.histogram(self.variable_list[i], self.mybin, weights=self.weight_list[i]**2)
            mchist.append(h)
            mcerr2.append(err2)
        mchist  = np.array(mchist)
        mcerr2  = np.array(mcerr2)


        datahist,_ = np.histogram(self.Datav, self.mybin)
        return mchist,mcerr2, datahist

    def getFittingResult(self):
        template,templatevar2,target = self.getMCDataHistAndErr()
        fitter = TemplateFitter(template,templatevar2,target,target,lock=self.lock)
        
        self.adjust = fitter.adjust
        self.adjustsigma = fitter.sigma
        self.totalcost = fitter.cost(fitter.adjust)
        self.adjustweight_list = [mcw*self.adjust[i] for i,mcw in enumerate(self.weight_list)]

    
    def printFittingResult(self):
        print('total cost is {:10.6f}'.format(self.totalcost))
        for i in range(self.adjust.size):
            print('adjust{:2} is {:6.3f} +/- {:6.3f}'.format(i, self.adjust[i], self.adjustsigma[i]) )
        
        




    def plotFittingResult(self,plotoutdir=None):
        #self.getFittingResult()

        fig, axes = plt.subplots(2, 1, sharex=True, 
                                 gridspec_kw={'height_ratios':[3,1]},
                                 figsize=self.figuresize)
        fig.subplots_adjust(hspace=0)
        ax = axes[0]

        ######################### 1. Main Plots #############################
        # 1.1. show initial MC
        mc0 = ax.hist(self.variable_list,
                    weights = self.weight_list,
                    label   = self.label_list[0:-1],
                    color   = self.color_list[0:-1],
                    bins    = self.mybin,
                    lw=0, alpha=0.5, 
                    histtype="stepfilled", 
                    stacked=self.isstacked
                    )
        self.mc0    = mc0[0] # keep only the stacked histogram, ignore the bin edges
        self.mctot0 = self.ConvertZeroInto(self.mc0[-1],into=1)
        
        # 1.2. show fitted MC
        mc = ax.hist(self.variable_list,
                    weights = self.adjustweight_list,
                    color   = self.color_list[0:-1],
                    bins    = self.mybin,
                    linestyle='-',
                    lw=3, alpha=1, 
                    histtype="step",
                     stacked=self.isstacked
                    )
        self.mc     = mc[0]
        self.mctot  = self.ConvertZeroInto(self.mc [-1],into=1)
        
        # 1.3. show data
        
        h,_ = np.histogram(self.Datav, self.mybin, weights=self.Dataw)
        ax.errorbar(self.center, h, yerr=h**0.5,
                    color=self.color_list[-1], 
                    label=self.label_list[-1],
                    fmt='.',markersize=10)
        self.hdata = h

        # 1.4. plot settings
        ax.grid()
        ax.legend(fontsize=10,loc="upper right")
        ax.text(0.04*self.b+0.96*self.a, 1.35*h.max(), r'CMS $preliminary$', style="italic",fontsize="15",fontweight='bold')          
        ax.set_xlim(self.a,self.b)
        ax.set_ylim(1,1.5*self.hdata.max())
        if self.logscale:
            ax.set_ylim(10,10*self.hdata.max())
            ax.set_yscale('log')
        ax.set_title("L=35.7/fb (13TeV)",loc="right")
        
        
        ######################### 2. Ratio Plots #############################
        ax = axes[1]
        ax.set_xlim(self.a,self.b)
        ax.set_ylim(0.5,1.5)
        ax.axhline(1,lw=1,color='k')
            
        ax.errorbar(self.center, self.hdata/self.mctot0, yerr=self.hdata**0.5/self.mctot0,
                    color = self.color_list[-1],
                    fmt='.',markersize=10)
        ax.errorbar(self.center, self.hdata/self.mctot, yerr=self.hdata**0.5/self.mctot,
                    color="r",
                    fmt='x',markersize=10,alpha=1)
        ax.grid()


        ############################ 3. End and Save ############################### 
        ax.set_xlabel(self.xl,fontsize=13)
        if plotoutdir is not None:
            fig.savefig(plotoutdir+"{}_fit.png".format(self.xl))
            
        
    def ConvertZeroInto(self,arr,into=1):
        for i in range(arr.size):
            if arr[i]==0:
                arr[i]=into
        return arr