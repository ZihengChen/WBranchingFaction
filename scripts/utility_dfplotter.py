import utility_common as common
from utility_dfcutter import *
from IPython.display import clear_output


class DFPlotter:
    def __init__(self,selection,nbjet, njet=None):
        self.selection = selection
        self.nbjet = nbjet
        self.njet  = njet
        self._setConfiguration() 

    def getDataFrameList(self, variation=''):
        Data = DFCutter(self.selection, self.nbjet, "data2016", self.njet).getDataFrame(variation)
        MCzz = DFCutter(self.selection, self.nbjet, "mcdiboson",self.njet).getDataFrame(variation)
        MCdy = DFCutter(self.selection, self.nbjet, "mcdy",     self.njet).getDataFrame(variation)
        MCt  = DFCutter(self.selection, self.nbjet, "mct",      self.njet).getDataFrame(variation)
        MCtt = DFCutter(self.selection, self.nbjet, "mctt",     self.njet).getDataFrame(variation)

        # get signal MC dataframes
        MCsg = pd.concat([MCt,MCtt],ignore_index=True, sort=False)
        MCsgList = [MCsg.query(q) for q in self.mcsgQueryList]

        # combine all dataframes as a list
        dfList = [MCzz,MCdy] + MCsgList + [Data]
        
        # add fakes if in mu4j and e4j
        if self.selection in ['mu4j','e4j']:

            names = ["data2016","mcdiboson","mcdy","mct","mctt"]

            Fake = pd.DataFrame()
            for name in names:
                temp =  DFCutter(self.selection+'_fakes',self.nbjet,name,self.njet).getDataFrame(variation)
                if not name == 'data2016':
                    temp.eventWeight = -1*temp.eventWeight
                Fake = Fake.append(temp,ignore_index=True, sort=False)

            dfList = [Fake] + dfList


        # add fakes if in mutau and etau
        if self.selection in ['mutau','etau']:

            names = ["data2016","mcdiboson","mcdy","mct","mctt"]

            Fake = pd.DataFrame()
            for name in names:
                temp =  DFCutter(self.selection+'_fakes',self.nbjet,name,self.njet).getDataFrame(variation)
                if not name == 'data2016':
                    temp.eventWeight = -1*temp.eventWeight
                Fake = Fake.append(temp,ignore_index=True, sort=False)

            dfList = [Fake] + dfList
            
        return dfList

    def plotKinematics(self):
        if self.outputPlotDir != None:
            common.makeDirectory(self.outputPlotDir)

        dfList = self.getDataFrameList()
        for index, row in self.pp.iterrows():

            v,a,b,step,xl = row["var"],row["lower"],row["upper"],row["step"],row["xlabel"]

            sk = ASingleKinematicPlot(v,a,b,step,dfList,adjust=self.adjust)
            sk.settingPlot(xl,self.labelList, self.colorList)
            sk.makePlot(self.outputPlotDir, self.selection)

            print("making plots -- {} nbjet{}: {}/{}".format(self.selection, self.nbjet, index+1, len(self.pp)) )
            clear_output(wait=True)
            plt.close()


    def _setConfiguration(self):
        self.outputPlotDir = None
        
        baseDirectory = common.getBaseDirectory()

        # MARK -- config output plot directory
        if self.nbjet == "==1":
            self.outputPlotDir = baseDirectory+"plots/kinematics/{}/1b/".format(self.selection)
            
        elif self.nbjet == ">1":
            self.outputPlotDir = baseDirectory+"plots/kinematics/{}/2b/".format(self.selection)

        elif self.nbjet == "<1":
            self.outputPlotDir = baseDirectory+"plots/kinematics/{}/0b/".format(self.selection)

    

        # MARK -- config plotting parameters for each selection
        # mumu
        if self.selection in ["mumu","mumuc"]:
            self.mcsgQueryList = [
                'genCategory>=16',
                'genCategory in [1,3,4,5,6,7,8,9,10,11,12]',
                'genCategory in [2]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow \mu + \mu$',
                r'$tt/tW \rightarrow \mu+ \tau$',
                'data'
            ]
            self.colorList = ["#a32020", "#e0301e", "#eb8c00", "#49feec", "deepskyblue", "mediumpurple", "k"]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_mumu.csv")
            self.adjust = [1,1,1,1,1,1]
            #self.hasFake = False
        # ee
        elif self.selection in ["ee","eec"]:
            self.mcsgQueryList = [
                'genCategory>=16',
                'genCategory in [2,3,4,5,6,7,8,9,13,14,15]',
                'genCategory in [1]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow e + e$',
                r'$tt/tW \rightarrow e \tau$',
                'data'
            ]
            self.colorList = ["#a32020", "#e0301e", "#eb8c00", "#49feec", "deepskyblue", "mediumpurple", "k"]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_ee.csv")
            self.adjust = [1,1,1,1,1,1]
            #self.hasFake = False
        
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
                r'$tt/tW \rightarrow$ (other)',
                r'$tt/tW \rightarrow l + l$ (other)',
                r'$tt/tW \rightarrow e + \mu$', 
                r'$tt/tW \rightarrow e + \tau$',
                r'$tt/tW \rightarrow \mu + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_emu.csv")
            self.adjust = [1,1,1,1,1,1,1]
            #self.hasFake = False

        # mutau
        elif "mutau" in self.selection:
            self.mcsgQueryList = [
                'genCategory in [16,21]',
                'genCategory in [1,2,3, 4,5,6,7,8,9, 10,11,12]',
                'genCategory in [17]',
                'genCategory in [18,19,20]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow l + l$ (other) ',
                r'$tt/tW \rightarrow \mu + h$', 
                r'$tt/tW \rightarrow \tau + h$',
                r'$tt/tW \rightarrow \mu + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_mutau.csv")
            self.adjust = [1,1,1,1,1,1,1]
            #self.adjust = [1/.95,1/.95,1/.95,1/.95,1/.95,1/.95,.89/.95]
            #self.hasFake = False
            if self.selection == 'mutau':
                self.fakeSF = common.getFakeSF('tau')
                self.colorList = ['grey'] + self.colorList
                self.adjust    = [self.fakeSF] + self.adjust
                self.labelList = ['Fakes']+self.labelList
            
        
        # etau
        elif "etau" in self.selection:
            self.mcsgQueryList = [
                'genCategory in [17,21]',
                'genCategory in [1,2,3, 4,5,6,7,8,9, 13,14,15]',
                'genCategory in [16]',
                'genCategory in [18,19,20]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow l + l$ (other) ',
                r'$tt/tW \rightarrow e + h$', 
                r'$tt/tW \rightarrow \tau + h$',
                r'$tt/tW \rightarrow e + \tau$',
                'data'
            ]
            self.colorList = ["#a32020","#e0301e","#eb8c00","gold","#49feec","deepskyblue","mediumpurple","k"]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_etau.csv")
            self.adjust = [1,1,1,1,1,1,1]
            #self.adjust = [1/.95,1/.95,1/.95,1/.95,1/.95,1/.95,.89/.95]
            #self.hasFake = False
            if self.selection == 'etau':
                self.fakeSF = common.getFakeSF('tau')
                self.colorList = ['grey'] + self.colorList
                self.adjust    = [self.fakeSF] + self.adjust
                self.labelList = ['Fakes']+self.labelList

        # mu4j
        elif "mu4j" in self.selection:
            self.mcsgQueryList = [
                'genCategory in [16,18,19,20,21]',
                'genCategory in [1,2,3,4,5,6,7,8,9,10,11,12]',
                'genCategory in [17]',
                'genCategory in [13,14,15]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow \mu + h$',
                r'$tt/tW \rightarrow \mu+ \tau$',
                'data'
            ]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_mu4j.csv")
            self.colorList = ["#a32020","#e0301e","#eb8c00","#49feec","deepskyblue","mediumpurple","k"]
            self.adjust = [1,1,1,1,1,1]

            if self.selection == 'mu4j':
                self.fakeSF = common.getFakeSF('mu')

                self.colorList = ['grey'] + self.colorList
                self.adjust    = [self.fakeSF] + self.adjust
                self.labelList = ['Fakes']+self.labelList

        # e4j
        elif "e4j" in self.selection:
            self.mcsgQueryList = [
                'genCategory in [17,18,19,20,21]',
                'genCategory in [1,2,3,4,5,6,7,8,9,13,14,15]',
                'genCategory in [16]',
                'genCategory in [10,11,12]'
            ]
            self.labelList = ['Diboson','V+Jets',
                r'$tt/tW \rightarrow$ other',
                r'$tt/tW \rightarrow ll$ other',
                r'$tt/tW \rightarrow e + h$',
                r'$tt/tW \rightarrow e + \tau$',
                'data'
            ]
            self.pp = pd.read_csv(baseDirectory+"scripts/plotterItemTables/itemTable_e4j.csv")
            self.colorList = ["#a32020","#e0301e","#eb8c00","#49feec","deepskyblue","mediumpurple","k"]
            self.adjust = [1,1,1,1,1,1]

            if self.selection == 'e4j':
                self.fakeSF = common.getFakeSF('e')

                self.colorList = ['grey'] + self.colorList
                self.adjust    = [self.fakeSF] + self.adjust
                self.labelList = ['Fakes']+self.labelList


class ASingleKinematicPlot:
    def __init__(self, v, a,b,step, df_list, adjust=None):
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
        
        #self.hasFake = hasFake

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
        # if self.hasFake:
        #     variable = np.concatenate(self.variable_list[1:3])
        #     weight   = np.concatenate(self.weight_list[1:3])
        #     yieldBg,_    = np.histogram(variable, self.mybin, weights=weight)
        #     errBg = 0.05 * yieldBg

        #     variable = self.variable_list[0]
        #     weight   = self.weight_list[0]
        #     yieldFake,_    = np.histogram(variable, self.mybin, weights=weight)
        #     errFake = 0.011/0.070 * yieldFake

        #     err = ( errBg**2 + errFake**2)**0.5
        #     return err
        # else:
        variable = np.concatenate(self.variable_list[0:2])
        weight   = np.concatenate(self.weight_list[0:2])
        yieldBg,_= np.histogram(variable, self.mybin, weights=weight)
        err = 0.05 * yieldBg
        return err
    

    def convertZeroInto(self,arr,into=1):
        for i in range(arr.size):
            if arr[i]==0:
                arr[i]=into
        return arr

    def makePlot(self, plotoutdir=None, selection=None):
        plt.rc("figure",facecolor="w")
        fig, axes = plt.subplots(2, 1, sharex=True, 
                                 gridspec_kw={'height_ratios':[3,1]},
                                 figsize=self.figuresize)
        fig.subplots_adjust(hspace=0)
        self.axes = axes
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
            if self.logscale:
                ax.text(0.04*self.b+0.96*self.a, 4*h.max(), 
                        r'CMS $preliminary$',
                        style="italic",fontsize="15",fontweight='bold')
            else:
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

            if selection is not None:
                if '0b' in plotoutdir:
                    fig.savefig(plotoutdir+"{}_0b_{}.pdf".format(selection,self.v))

                if '1b' in plotoutdir:
                    fig.savefig(plotoutdir+"{}_1b_{}.pdf".format(selection,self.v))
                if '2b' in plotoutdir:
                    fig.savefig(plotoutdir+"{}_2b_{}.pdf".format(selection,self.v))

            else:
                fig.savefig(plotoutdir+"{}.png".format(self.v),dpi=300)

            #







# class SystematicPlotter:
#     def __init__(self,selection,nbjet,source):
#         self.selection = selection
#         self.nbjet = nbjet
#         self.source = source
    
#         MCtt  = DFCutter(selection,nbjet,"mctt").getDataFrame('')
#         MCtt1 = DFCutter(selection,nbjet,"mctt").getDataFrame(self.source+'Up')
#         MCtt2 = DFCutter(selection,nbjet,"mctt").getDataFrame(self.source+'Down')

#         self._setConfiguration(selection)


#         self.MCTT = MCtt.query('genCategory=={}'.format(self.selectCata))
#         self.MCtt = MCtt.query('genCategory!={}'.format(self.selectCata))

#         self.MCTT1 = MCtt1.query('genCategory=={}'.format(self.selectCata))
#         self.MCtt1 = MCtt1.query('genCategory!={}'.format(self.selectCata))

#         self.MCTT2 = MCtt2.query('genCategory=={}'.format(self.selectCata))
#         self.MCtt2 = MCtt2.query('genCategory!={}'.format(self.selectCata))

#     def _setConfiguration(self,selection):
#         if selection == 'ee':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow e + e $']
#             self.selectCata = 1
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")
        
#         elif selection == 'emu2':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow e + \mu$']
#             self.selectCata = 3
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")

#         elif selection == 'etau':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow e + \tau$']
#             self.selectCata = 12
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")

#         elif selection == 'e4j':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow e + h $']
#             self.selectCata = 16
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_l4j.csv")

#         elif selection == 'emu':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow \mu + e$']
#             self.selectCata = 3
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")
#         elif selection == 'mumu':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow \mu + \mu$']
#             self.selectCata = 2
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")

#         elif selection == 'mutau':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow \mu + \tau$']
#             self.selectCata = 15
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_ll.csv")

#         elif selection == 'mu4j':
#             self.lables = [r'$tt \rightarrow other $',r'$tt \rightarrow \mu + h$']
#             self.selectCata = 17
#             self.pp = pd.read_csv(common.getBaseDirectory()+"scripts/plotterItemTables/itemTable_l4j.csv")


        

    
#     def plotKinematics(self):

#         for index, row in self.pp.iterrows():

#             v,a,b,step,xl = row["var"],row["lower"],row["upper"],row["step"],row["xlabel"]


#             bins    = np.arange(a,b,step)
#             centers = bins[:-1]+step/2

#             h  = np.histogram( self.MCtt [v], bins, weights = self.MCtt .eventWeight)[0]
#             h1 = np.histogram( self.MCtt1[v], bins, weights = self.MCtt1.eventWeight)[0]
#             h2 = np.histogram( self.MCtt2[v], bins, weights = self.MCtt2.eventWeight)[0]


#             H  = np.histogram( self.MCTT [v], bins, weights = self.MCTT .eventWeight)[0]
#             H1 = np.histogram( self.MCTT1[v], bins, weights = self.MCTT1.eventWeight)[0]
#             H2 = np.histogram( self.MCTT2[v], bins, weights = self.MCTT2.eventWeight)[0]


#             fig, axes = plt.subplots(2, 1, sharex=True, 
#                          facecolor='w',
#                          gridspec_kw={'height_ratios':[3,1]},
#                          figsize=(6,6))
#             fig.subplots_adjust(hspace=0)

#             # kinematic plots
#             ax = axes[0]
#             ax.hist([centers,centers], bins=bins, weights=[h,H],color=['C0','C1'],
#                     histtype="stepfilled",stacked=True,linewidth=0,alpha=0.4)

#             ax.hist([centers,centers], bins=bins, weights=[h,H],color=['C0','C1'],
#                     histtype="step",stacked=True,linewidth=2,
#                     label=self.lables
#                     )


#             ax.errorbar(centers,h,  yerr=[np.abs(h1-h),np.abs(h2-h)], fmt='.',color='C0',markersize=0,linewidth=1,alpha=1)
#             ax.errorbar(centers,h+H,yerr=[np.abs(H1-H),np.abs(H2-H)], fmt='.',color='C1',markersize=0,linewidth=1,alpha=1)
#             ax.legend()

#             ax.grid(True,linestyle='--',alpha=0.5)
#             ax.set_title('Error from '+self.source,fontsize="12",loc="left" )
#             ax.set_xlim(a, b)
#             ax.set_ylim(1,1.3*(H2+h).max())
#             ax.text(0.04*b+0.96*a, 1.2*(H2+h).max(), 
#                     r'CMS $Simulation$',
#                     style="italic",fontsize="15",fontweight='bold')
            


#             # ratio plots
#             ax = axes[1]

#             c, r1, r2 = centers[h>0], h1[h>0]/h[h>0], h2[h>0]/h[h>0]
#             ax.bar(c,r1-1, bottom=np.ones_like(c), width=step*0.8, align='center', color='C0',alpha=0.7)
#             ax.bar(c,r2-1, bottom=np.ones_like(c), width=step*0.8, align='center', color='C0',alpha=0.3)

#             C, R1, R2 = centers[H>0], H1[H>0]/H[H>0], H2[H>0]/H[H>0]
#             ax.bar(C,R1-1, bottom=np.ones_like(C), width=step*0.8, align='center', color='C1',alpha=0.7)
#             ax.bar(C,R2-1, bottom=np.ones_like(C), width=step*0.8, align='center', color='C1',alpha=0.3)

#             # ax.plot(centers[h>0],h1[h>0]/h[h>0],color='C0')
#             # ax.plot(centers[h>0],h2[h>0]/h[h>0],linestyle='--',color='C0')
#             # ax.fill_between(centers[h>0],h1[h>0]/h[h>0],h2[h>0]/h[h>0],alpha=0.2)

#             # ax.plot(centers[H>0],H1[H>0]/H[H>0],color='C1')
#             # ax.plot(centers[H>0],H2[H>0]/H[H>0],linestyle='--',color='C1')
#             # ax.fill_between(centers[H>0],H1[H>0]/H[H>0],H2[H>0]/H[H>0],alpha=0.2)

#             ax.axhline(1,linewidth=1,color='k')


#             ax.set_xlabel(v)
#             ax.grid(True,linestyle='--',alpha=0.5)
#             ax.set_xlim(a, b)
#             if self.source =='FSR':
#                 ax.set_ylim(0.4, 1.6)
#             else:
#                 ax.set_ylim(0.8, 1.2)


#             # save figure
#             plt.savefig(common.getBaseDirectory()+'plots/systematics/{}/{}/{}.png'.format(self.source ,self.selection,v),dpi=300)

#             print("making plots -- {} nbjet{}: {}/{}".format(self.selection, self.nbjet, index+1, len(self.pp)) )
#             clear_output(wait=True)
#             plt.close()




# for s in ["emu","mumu","mutau","mu4j","ee","emu2","etau","e4j"]:
#     sp = SystematicPlotter(s,">=1",'FSR')
#     sp.plotKinematics()