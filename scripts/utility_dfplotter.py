import utility_common as common
from utility_dfcutter import *
from IPython.display import clear_output



class DFPlotter:
    def __init__(self,selection,nbjet):
        self.selection = selection
        self.nbjet = nbjet
        self._setConfiguration() 

    def getDataFrameList(self,variation=''):
        Data = DFCutter(self.selection, self.nbjet, "data2016").getDataFrame(variation)
        MCzz = DFCutter(self.selection, self.nbjet, "mcdiboson").getDataFrame(variation)
        MCdy = DFCutter(self.selection, self.nbjet, "mcdy").getDataFrame(variation)
        MCt  = DFCutter(self.selection, self.nbjet, "mct").getDataFrame(variation)
        MCtt = DFCutter(self.selection, self.nbjet, "mctt").getDataFrame(variation)

        # get signal MC dataframes
        MCsg = pd.concat([MCt,MCtt],ignore_index=True)
        MCsgList = [MCsg.query(q) for q in self.mcsgQueryList]

        # combine all dataframes as a list
        dfList = [MCzz,MCdy] + MCsgList + [Data]
        if self.hasFake:
            Fake = DFCutter(self.selection, self.nbjet, "data2016_inverseISO").getDataFrame(variation)
            dfList = [Fake] + dfList
        return dfList

    def plotKinematics(self):

        dfList = self.getDataFrameList()
        for index, row in self.pp.iterrows():

            v,a,b,step,xl = row["var"],row["lower"],row["upper"],row["step"],row["xlabel"]

            sk = ASingleKinematicPlot(v,a,b,step,dfList,adjust=self.adjust,hasFake=self.hasFake)
            sk.settingPlot(xl,self.labelList, self.colorList)
            sk.makePlot(self.outputPlotDir)

            print("making plots -- {} nbjet{}: {}/{}".format(self.selection, self.nbjet, index+1, len(self.pp)) )
            clear_output(wait=True)
            plt.close()


    def _setConfiguration(self):
        # MARK -- config output plot directory
        if self.nbjet == "==1":
            self.outputPlotDir = common.dataDirectory()+"../plot/{}/1b/".format(self.selection)
        elif self.nbjet == ">1":
            self.outputPlotDir = common.dataDirectory()+"../plot/{}/2b/".format(self.selection)
        
        common.makeDirectory(self.outputPlotDir)

        # MARK -- config plotting parameters for each selection
        # mumu
        if self.selection == "mumu":
            self.mcsgQueryList = [
                'genCategory>=16',
                'genCategory in [1,3,4,5,6,7,8,9,10,11,12]',
                'genCategory in [2]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow l + had$',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow \mu + \mu$',
                r'$tt/tW \rightarrow \mu+ \tau$',
                'data'
            ]
            self.colorList = ["#a32020", "#e0301e", "#eb8c00", "#49feec", "deepskyblue", "mediumpurple", "k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters.csv")
            self.adjust = [1,1,1,1,1,1]
            self.hasFake = False
        # ee
        elif self.selection == "ee":
            self.mcsgQueryList = [
                'genCategory>=16',
                'genCategory in [2,3,4,5,6,7,8,9,13,14,15]',
                'genCategory in [1]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow l + had$',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow e + e$',
                r'$tt/tW \rightarrow e \tau$',
                'data'
            ]
            self.colorList = ["#a32020", "#e0301e", "#eb8c00", "#49feec", "deepskyblue", "mediumpurple", "k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters.csv")
            self.adjust = [1,1,1,1,1,1]
            self.hasFake = False
        
        # mue and emu
        elif self.selection in ["emu","emu2"]:
            self.mcsgQueryList = [
                'genCategory>=16',
                'genCategory in [1,2,4,5,6,7,8,9]',
                'genCategory in [3]',
                'genCategory in [10,11,12]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow l + h$ (other)',
                r'$tt/tW \rightarrow l + l$ (other)',
                r'$tt/tW \rightarrow e + \mu$', 
                r'$tt/tW \rightarrow e + \tau$',
                r'$tt/tW \rightarrow \mu + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters.csv")
            self.adjust = [1,1,1,1,1,1,1]
            self.hasFake = False

        # mutau
        elif self.selection == "mutau":
            self.mcsgQueryList = [
                'genCategory in [16,21]',
                'genCategory in [1,2,3, 4,5,6,7,8,9, 10,11,12]',
                'genCategory in [17]',
                'genCategory in [18,19,20]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow l + h$ (other)',
                r'$tt/tW \rightarrow l + l$ (other) ',
                r'$tt/tW \rightarrow \mu + h$', 
                r'$tt/tW \rightarrow \tau + h$',
                r'$tt/tW \rightarrow \mu + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters.csv")
            self.adjust = [1,1,1,1,1,1,1]
            self.hasFake = False
        
        # etau
        elif self.selection == "etau":
            self.mcsgQueryList = [
                'genCategory in [17,21]',
                'genCategory in [1,2,3, 4,5,6,7,8,9, 13,14,15]',
                'genCategory in [16]',
                'genCategory in [18,19,20]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow l + h$ (other)',
                r'$tt/tW \rightarrow l + l$ (other) ',
                r'$tt/tW \rightarrow e + h$', 
                r'$tt/tW \rightarrow \tau + h$',
                r'$tt/tW \rightarrow e + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters.csv")
            self.adjust = [1,1,1,1,1,1,1]
            self.hasFake = False

        # mu4j
        elif self.selection == "mu4j":
            self.mcsgQueryList = [
                'genCategory in [16,18,19,20,21]',
                'genCategory in [1,2,3,4,5,6,7,8,9,10,11,12]',
                'genCategory in [17]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Fakes','Diboson','V+Jets',
                r'$tt/tW \rightarrow lh$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow \mu + h$',
                r'$tt/tW \rightarrow \mu+ \tau$',
                'data'
            ]
            self.colorList = ["gray","#a32020","#e0301e","#eb8c00","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters4j.csv")
            self.adjust = [common.fakeRate(),1,1,1,1,1,1]
            self.hasFake = True

        # e4j
        elif self.selection == "e4j":
            self.mcsgQueryList = [
                'genCategory in [17,18,19,20,21]',
                'genCategory in [1,2,3,4,5,6,7,8,9,13,14,15]',
                'genCategory in [16]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow lh$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow e + h$',
                r'$tt/tW \rightarrow e + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(common.dataDirectory()+"pp/plotparameters4j.csv")
            self.adjust = [1,1,1,1,1,1]
            self.hasFake = False


class ASingleKinematicPlot:
    def __init__(self, v, a,b,step, df_list, adjust=None, hasFake=False):
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
                    withXsErr = False
                    ):
        self.xl = xl
        self.label_list = label_list
        self.color_list = color_list
        self.logscale   = logscale
        self.isstacked  = isstacked
        self.figuresize = figuresize
        self.withXsErr = withXsErr
    
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
        
        if self.withXsErr:
            self.mcerr = (self.getHistogramError()**2 + self.getHistogramErrorDueToBgCrossSection()**2)**0.5
        else:   
            self.mcerr = self.getHistogramError()
        
            

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
            
        ax.grid(True,linestyle="--",alpha=0.5)
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
        ax.grid(True,linestyle="--",alpha=0.5)
            
        ######################## 3. End and Save ############################### 
        ax.set_xlabel(self.xl,fontsize=13)
        if plotoutdir is not None:
            fig.savefig(plotoutdir+"{}.png".format(self.v),dpi=300)
