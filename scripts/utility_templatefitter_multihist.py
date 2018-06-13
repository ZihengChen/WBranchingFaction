from scipy.optimize import minimize
from pylab import *
import numdifftools as nd
from utility_common import *


#label_list = ['Diboson','V+Jets', 'tW',"tt",'data']
#color_list = ["#a32020","#eb8c00","#49feec","deepskyblue","k"]

label_list = ['bg','signal','data']
color_list = ["#eb8c00","deepskyblue","k"]

class TemplateFitter:
    def __init__(self, 
                 template, templatevar2,
                 target,   targetvar2,
                 lock = None):
        self.nchannel       = len(template)
        self.ntemplate      = template[0].shape[0]
        
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
        chisquared = 0

        for i in range(self.nchannel):
            ch_template     = self.template[i]
            ch_templatevar2 = self.templatevar2[i]
            ch_target       = self.target[i]
            ch_targetvar2   = self.targetvar2[i]

            ch_totalTemplate       = np.dot(adjust,    ch_template)
            ch_totalTemplateVar2   = np.dot(adjust**2, ch_templatevar2)

            ch_diff_TemplateTarget = ch_totalTemplate     - ch_target
            ch_var2_TemplateTarget = ch_totalTemplateVar2 + ch_targetvar2

            # only consider non zero var
            nonzero = ch_var2_TemplateTarget > 0
            ch_diff_TemplateTarget = ch_diff_TemplateTarget[nonzero]
            ch_var2_TemplateTarget = ch_var2_TemplateTarget[nonzero]

            ch_chisquared = ch_diff_TemplateTarget**2 / ch_var2_TemplateTarget
            ch_chisquared = np.mean(ch_chisquared)
            chisquared   += ch_chisquared

        return chisquared
        
    def penalty(self, adjust):
        adjustpenalty = 0

        for i in range(self.nchannel):
            ch_template     = self.template[i]
            ch_templatevar2 = self.templatevar2[i]
            ch_target       = self.target[i]
            ch_targetvar2   = self.targetvar2[i]

            ch_change_2    = (adjust - np.ones_like(adjust))**2
            ch_change_var2 = np.sum(ch_template,axis = 1) / np.sum(ch_template,axis = 1)**2
            ch_change_chi2 = ch_change_2 / ch_change_var2
            ch_adjustpenalty = np.dot(ch_change_chi2, self.lock)
            
            adjustpenalty += ch_adjustpenalty

        return adjustpenalty
    
    def cost(self, adjust):
        return self.chisquared(adjust) + self.penalty(adjust)
    
    def fit(self):
        result = minimize(fun = self.cost, 
                          x0  = self.adjust_init,
                          method = 'SLSQP',
                          bounds = self.ntemplate * [(0,10),]
                         )
        return result.x

    def fitvar(self):
        n     = self.ntemplate
        #hcalc = nd.Hessian(self.chisquared, step=1e-4, method='central')
        hcalc = nd.Hessian(self.cost, step=1e-4, method='central')
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

class Template():
    def __init__(self, selection,nbjetcut):
    
        self.selection = selection
        self.nbjetcut = nbjetcut

        self.loaddflist()

    
    def loaddflist(self):
        pickledir  =  "../data/pickle/{}/".format(self.selection)

        cuts = GetSelectionCut(self.selection) + "& (nBJets{})".format(self.nbjetcut)

        Data = LoadDataframe(pickledir + "data2016").query(cuts)
        MCzz = LoadDataframe(pickledir + "mcdiboson").query(cuts)
        MCdy = LoadDataframe(pickledir + "mcdy").query(cuts)
        MCt  = LoadDataframe(pickledir + "mct").query(cuts)
        MCtt = LoadDataframe(pickledir + "mctt").query(cuts)
        if self.selection == "emu":
            Data = Data.drop_duplicates(subset=['runNumber', 'evtNumber'])

        MCsg = pd.concat([MCt,MCtt],ignore_index=True)
        MCbg = pd.concat([MCzz,MCdy],ignore_index=True)
        self.df_list = [MCbg,MCsg,Data]
        #self.df_list    = [MCzz,MCdy,MCt,MCtt,Data]


    def loadconfig(self, v,a,b,step):
        self.v = v
        self.mybin  = np.arange(a,b,step)
        self.center = self.mybin[1:]-step/2


    def loadvariable(self):        

        self.variable_list = [mc[self.v] for mc in self.df_list[0:-1]]
        self.weight_list   = [mc.eventWeight for mc in self.df_list[0:-1]]
        self.Datav  = self.df_list[-1][self.v]
        self.Dataw  = self.df_list[-1].eventWeight

    def maketemplate(self):
        self.loadvariable()

        mchist, mcerr2 = [],[]
        for i in range(len(self.variable_list)):
            h,_    = np.histogram(self.variable_list[i], self.mybin, weights=self.weight_list[i])
            err2,_ = np.histogram(self.variable_list[i], self.mybin, weights=self.weight_list[i]**2)
            mchist.append(h)
            mcerr2.append(err2)
        self.mchist  = np.array(mchist)
        self.mcerr2  = np.array(mcerr2)

        self.datahist,_ = np.histogram(self.Datav, self.mybin, weights=self.Dataw)
        self.dataerr2   = self.datahist
        
        return self.mchist,self.mcerr2, self.datahist, self.dataerr2



class FitPlotter():
    def __init__(self,v,a,b,step,df_list, adjust=None):
        self.v = v
        self.a = a
        self.b = b
        self.step   = step
        self.mybin  = np.arange(a,b,step)
        self.center = self.mybin[1:]-self.step/2

        self.n = len(df_list ) - 1

        self.variable_list  = [mc[v] for mc in df_list[0:-1]]
        self.weight_list    = [mc.eventWeight for i,mc in enumerate(df_list[0:-1])]
        self.Datav  = df_list[-1][v]
        self.Dataw  = df_list[-1].eventWeight


        if adjust is None:
            self.adjust = np.ones(self.n)
        else:
            self.adjust = adjust

        self.adjustweight_list  = [mc.eventWeight * self.adjust[i] for i,mc in enumerate(df_list[0:-1])]

    
    def settingPlot(self,
                    xl,
                    label_list,
                    color_list,
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
    
    def getHistogramError(self):
        variable = np.concatenate(self.variable_list)
        weight   = np.concatenate(self.weight_list)
        err,_    = np.histogram(variable, self.mybin, weights=weight**2)
        err      = err**0.5
        return err

    
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
                    lw=0, alpha=0.3, 
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
                    lw=2, alpha=1, 
                    histtype="step",
                     stacked=self.isstacked
                    )

        #mc = ax.hist(self.variable_list,
        #            weights = self.adjustweight_list,
        #            color   = len(self.variable_list)*['k'],
        #            bins    = self.mybin,
        #            linestyle='-',
        #            lw=0.5, alpha=1, 
        #            histtype="step",
        #            stacked=self.isstacked
        #            )

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
            fig.savefig(plotoutdir+"{}_fit.png".format(self.v))
    
    def ConvertZeroInto(self,arr,into=1):
        for i in range(arr.size):
            if arr[i]==0:
                arr[i]=into
        return arr



