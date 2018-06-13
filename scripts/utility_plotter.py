from utility_common import * 
from pylab import *

class SelectionPlotter:
    def __init__(self,v,a,b,step,df_list, adjust=None, hasFake=False):
        self.v = v
        self.a = a
        self.b = b
        self.step   = step
        self.mybin  = np.arange(a,b,step)
        self.center = self.mybin[1:]-self.step/2

        self.n = len(df_list ) - 1

        if adjust is None:
            self.adjust = np.ones(self.n)
        else:
            self.adjust = adjust
        
        self.hasFake = hasFake

        self.variable_list  = [mc[v].values for mc in df_list[0:-1]]
        self.weight_list    = [mc['eventWeight'].values * self.adjust[i] for i,mc in enumerate(df_list[0:-1])]
        self.Datav  = df_list[-1][v].values 
        self.Dataw  = df_list[-1]['eventWeight'].values

    
    def settingPlot(self,
                    xl,
                    label_list,
                    color_list,
                    logscale   = False,
                    isstacked  = True,
                    figuresize = (6,5.4),
                    plotWithoutXsErr = False
                    ):
        self.xl = xl
        self.label_list = label_list
        self.color_list = color_list
        self.logscale   = logscale
        self.isstacked  = isstacked
        self.figuresize = figuresize
        self.plotWithoutXsErr = plotWithoutXsErr
    
    def getHistogramError(self):
        variable = np.concatenate(self.variable_list)
        weight   = np.concatenate(self.weight_list)
        err,_    = np.histogram(variable, self.mybin, weights=weight**2)
        err      = err**0.5
        return err
    
    def getHistogramErrorDueToBgCrossSection(self):
        if self.hasFake:
            variable = np.concatenate(self.variable_list[1:3])
            weight   = np.concatenate(self.weight_list[1:3])
            yieldBg,_    = np.histogram(variable, self.mybin, weights=weight)
            errBg = 0.05 * yieldBg

            variable = self.variable_list[0]
            weight   = self.weight_list[0]
            yieldFake,_    = np.histogram(variable, self.mybin, weights=weight)
            errFake = 0.011/0.070 * yieldFake

            err = ( errBg**2 + errFake**2)**0.5
            return err
        else:
            variable = np.concatenate(self.variable_list[0:2])
            weight   = np.concatenate(self.weight_list[0:2])
            yieldBg,_    = np.histogram(variable, self.mybin, weights=weight)
            err = 0.05 * yieldBg
            return err
        


    def convertZeroInto(self,arr,into=1):
        for i in range(arr.size):
            if arr[i]==0:
                arr[i]=into
        return arr

    def makePlot(self, plotoutdir=None):
        plt.rc("figure",facecolor="w")
        fig, axes = plt.subplots(2, 1, sharex=True, 
                                 gridspec_kw={'height_ratios':[3,1]},
                                 figsize=self.figuresize)
        fig.subplots_adjust(hspace=0)
        ax = axes[0]

        ######################### 1. Main Plots #############################
        # 1.1. show MC
        mc =  ax.hist(self.variable_list,
                    weights = self.weight_list,
                    label   = self.label_list[0:-1],
                    color   = self.color_list[0:-1],
                    bins    = self.mybin,
                    lw=0, alpha=0.8, 
                    histtype="stepfilled", 
                    stacked=self.isstacked
                    )
        mc    = mc[0] # keep only the stacked histogram, ignore the bin edges
        self.mctot = self.convertZeroInto(mc[-1],into=1)
        if self.plotWithoutXsErr:
            self.mcerr = self.getHistogramError()
        else:
            self.mcerr = (self.getHistogramError()**2 + self.getHistogramErrorDueToBgCrossSection()**2)**0.5

        ax.errorbar(self.center, self.mctot, yerr=self.mcerr,
                    color="k", fmt='none', 
                    lw=200/self.mybin.size, 
                    mew=0, alpha=0.3
                    )

        # 1,2. show data
    
        h,_ = np.histogram(self.Datav, self.mybin, weights=self.Dataw)
        self.hdata = h
        ax.errorbar(self.center, self.hdata, yerr=self.hdata**0.5,
                    color=self.color_list[-1], 
                    label=self.label_list[-1],
                    fmt='.',markersize=10)

        # 1.3. plot settings
        if self.xl in ["lepton_delta_phi","bjet_delta_phi","lbjet_delta_phi","tauMVA"]:
            ax.legend(fontsize=10,loc="upper left")
        else:
            ax.legend(fontsize=10,loc="upper right")
            ax.text(0.04*self.b+0.96*self.a, 1.35*h.max(), 
                    r'CMS $preliminary$',
                    style="italic",fontsize="15",fontweight='bold')
            
        ax.grid()
        ax.set_xlim(self.a, self.b)
        ax.set_ylim(1,1.5*self.hdata.max())
        if self.logscale:
            ax.set_ylim(10,10*self.hdata.max())
            ax.set_yscale('log')
            
        ax.set_title("L=35.9/fb (13TeV)",loc="right")
        
        
        ######################### 2. Ratio Plots #############################
        ax = axes[1]
        ax.set_xlim(self.a,self.b)
        ax.set_ylim(0.5,1.5)
        ax.axhline(1,lw=1,color='k')

        ax.errorbar(self.center, np.ones_like(self.mctot), yerr=self.mcerr/self.mctot,
                    color="k", fmt='none', lw=200/self.mybin.size, mew=0, alpha=0.3)

        ax.errorbar(self.center, self.hdata/self.mctot, yerr=self.hdata**0.5/self.mctot,
                    color=self.color_list[-1],
                    label=self.label_list[-1],
                    fmt='.',markersize=10)
        ax.grid()
            
        ######################## 3. End and Save ############################### 
        ax.set_xlabel(self.xl,fontsize=13)
        if plotoutdir is not None:
            make_directory(plotoutdir,clear=False)
            fig.savefig(plotoutdir+"{}.png".format(self.v))




class SelectionCounter:
    def __init__(self):
        dummy = 0
        #print("Initialize SelectionCounter")

    def countGen_ByTauDecay(self,rootfile,dataset):
        hist = rootfile.Get('GenCategory_'+dataset)
        gens = []

        for i in range(1,22,1):
            gens.append(hist.GetBinContent(i))
        return gens
        
    def countSlt_ByTauDecay(self,measuredf, withweights=True):
        yields = []

        if withweights is True:
            for i in range(1,22,1):
                temp = measuredf[ measuredf.genCategory == i ]
                w    = np.sum(temp.eventWeight/temp.eventWeightSF)
                if w is nan:
                    w = 0
                yields.append(w)
       
        else:
            for i in range(1,22,1):
                temp = measuredf[ measuredf.genCategory == i ]
                w    = len(temp)
                if w is nan:
                    w = 0
                yields.append(w)

        return yields

    def countSlt_Scaled(self, df_list):
        Data = df_list[-1]
        weight_list    = [mc.eventWeight for mc in df_list[0:-1]]
        
        n_ScaledMC = []
        for mcweight in weight_list:
            n_ScaledMC.append(np.sum(mcweight)) 
        n_ScaledMC = np.array(n_ScaledMC)
        print("data:{}".format( np.sum(Data.eventWeight)) )
        print("TotalMC:{}".format( np.sum(n_ScaledMC) ) )
        print('------ MC break down ------')
        print(n_ScaledMC)    
        