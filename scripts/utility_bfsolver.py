from pylab import *
import pandas as pd
from scipy import optimize
from utility_plotter import *

class BFCalc_SingleSelectorYield:
    def __init__(self,a, xs,lumin):
        self.a     = a        
        self.xs    = xs
        self.lumin = lumin

        al,bt,gm   = self.GetQuadPolyCoeff()
        self.alpha = al
        self.beta  = bt
        self.gamma = gm

    
    def BMatrix(self, r=1, smearBW=False, smearBt=False):
        # provide 5 PDG value
        BW_e  = 0.1075
        BW_m  = 0.1057 # 0.1057#0.1080
        Bt_e  = 0.1785
        Bt_m  = 0.1736
        
        if smearBW is True:
            #shift = np.random.normal(0, 0.0013)
            BW_e += np.random.normal(0, 0.0013)
            BW_m += np.random.normal(0, 0.0015)
        if smearBt is True:
            Bt_e += np.random.normal(0, 0.0005)
            Bt_m += np.random.normal(0, 0.0005)

        BW_t  = r * BW_m
        BW_h  = 1 - BW_e - BW_m - BW_t
        Bt_h  = 1 - Bt_e - Bt_m  

        BVector = np.array([BW_e,BW_m, BW_t*Bt_e, BW_t*Bt_m, BW_t*Bt_h, BW_h ])
        BMatrix = np.outer(BVector,BVector)
        return BMatrix


    def PredictYield(self, r=1):
        a       = self.a
        B       = self.BMatrix(r)
        NMatrix = (a*B)*self.xs*self.lumin
        N       = np.sum(NMatrix)
        return N

    def PredictYield_SmearOfConst(self, r=1, smearBW=False, smearBt=False):
        a       = self.a
        B       = self.BMatrix(r, smearBW, smearBt) # smear the Constents with PDG width
        NMatrix = (a*B)*self.xs*self.lumin
        N       = np.sum(NMatrix)
        return N

    def PredictYield_SmearOfSubN(self, r=1):
        a        = self.a
        B        = self.BMatrix(r)
        NMatrix  = (a*B)*self.xs*self.lumin
        NMatrix += np.random.normal( np.zeros_like(NMatrix), np.sqrt(NMatrix)) # smear the subN according sqrt
        N        = np.sum(NMatrix)
        return N



    # 1. analytical solvers

    def GetQuadPolyCoeff(self):
        N_r0  = self.PredictYield(r=0)
        N_rP  = self.PredictYield(r=1)
        N_rN  = self.PredictYield(r=-1)
        gamma = N_r0
        beta  = 0.5*( N_rP - N_rN )
        alpha = N_rP - beta  - gamma 
        return alpha, beta, gamma

    def SolveBF(self, n):
        a = self.alpha
        b = self.beta
        c = self.gamma - n

        delta = b**2 - 4* a*c
        if delta >= 0 :
            r1 = (-b + delta**0.5)/(2*a)
            r2 = (-b - delta**0.5)/(2*a)
            if (r1 >= 0) and (r1 < 10):
                return r1
            if (r2 >= 0) and (r2 < 10):
                return r2
            else:
                print( " no physical solution found for BF")
        else:
            print( " delta < 0")

    # 2. numerical solver 
    def SetYield(self, n):
        self.n = n
    def EqualYield(self, r):
        return self.PredictYield(r) - self.n
    def SolveBF_Numerical(self, n):
        self.SetYield(n)
        return optimize.brentq(self.EqualYield, 0.0, 2.0)
    

    # 3. test solver and get bf distribution
    #def GetBFDistribution(self, ntoy=1000, r=1):
    #    bfs = []
    #    for k in range(ntoy):
    #        # step 1 predict N with r value
    #        predictyield = self.PredictYield_SmearOfSubN(r)
    #        # step 2 solve f from N
    #        bf = self.SolveBF(predictyield)
    #        bfs.append(bf)
    #    bfs = np.array(bfs)
    #    return bfs

            
        
class BFCalc_MultiSelectorX:
    def __init__(self, 
                 a_mm, a_mt, a_em, a_mh,   
                 a_top, a_btm, 
                 xs, lumin,
                 IsBinomial = True):

        self.a_top = a_top
        self.a_btm = a_btm
        self.xs    = xs
        self.lumin = lumin
        self.IsBinomial = IsBinomial
        self.eps   = 1e-3
        
             
        self.a_mm  = a_mm
        self.a_mt  = a_mt
        self.a_em  = a_em
        self.a_mh  = a_mh
        self.BFCalc_mm = BFCalc_SingleSelectorYield(self.a_mm, self.xs, self.lumin)
        self.BFCalc_mt = BFCalc_SingleSelectorYield(self.a_mt, self.xs, self.lumin)
        self.BFCalc_em = BFCalc_SingleSelectorYield(self.a_em, self.xs, self.lumin)
        self.BFCalc_mh = BFCalc_SingleSelectorYield(self.a_mh, self.xs, self.lumin)

        self.BFCalc_top = BFCalc_SingleSelectorYield(self.a_top, self.xs, self.lumin)
        self.BFCalc_btm = BFCalc_SingleSelectorYield(self.a_btm, self.xs, self.lumin)
        self.BFCalc_btm_excludetop = np.zeros_like(a_btm)
        if self.IsBinomial is True:
            self.BFCalc_btm_excludetop = BFCalc_SingleSelectorYield(self.a_btm-self.a_top, self.xs, self.lumin)
        
    
    def PredictX(self, r = 1):
        ntop = self.BFCalc_top.PredictYield(r)
        nbtm = self.BFCalc_btm.PredictYield(r)
        X    = ntop/nbtm
        return  X
    
    def PredictX_SmearOfConst(self, r=1, smearBW=False, smearBt=False):
        ntop = self.BFCalc_top.PredictYield_SmearOfConst(r,smearBW,smearBt)
        if self.IsBinomial is True:
            nbtm = ntop + self.BFCalc_btm_excludetop.PredictYield_SmearOfConst(r,smearBW,smearBt)
        else:
            nbtm = self.BFCalc_btm.PredictYield_SmearOfConst(r,smearBW,smearBt)
        X = ntop/nbtm
        return X
    
    def PredictX_SmearOfSubN(self, r=1 ):
        ntop = self.BFCalc_top.PredictYield_SmearOfSubN(r)
        if self.IsBinomial is True:
            nbtm = ntop + self.BFCalc_btm_excludetop.PredictYield_SmearOfSubN(r)
        else:
            nbtm = self.BFCalc_btm.PredictYield_SmearOfSubN(r)
        X = ntop/nbtm
        return X

    def PredictError(self, r = 1 ):
        X  = self.PredictX(r)
        if self.IsBinomial is True:
            # binomial error
            sigmaX = ( X*(1-X)/ self.BFCalc_btm.PredictYield(r) )**0.5
        else:
            # uncertainty of A/B
            sigmaX = X * (1/self.BFCalc_top.PredictYield(r) + 1/self.BFCalc_btm.PredictYield(r) )**0.5
        
        dX_dr  = (self.PredictX(r+self.eps)- X)/self.eps
        dr_dX  = 1/dX_dr
        sigmar = dr_dX * sigmaX
        return sigmar,dr_dX,sigmaX
 

    # 1. analytical solver
    def SolveBF(self, X):
        a = X*self.BFCalc_btm.alpha - self.BFCalc_top.alpha
        b = X*self.BFCalc_btm.beta  - self.BFCalc_top.beta
        c = X*self.BFCalc_btm.gamma - self.BFCalc_top.gamma
        
        delta = b**2 - 4*a*c
        if delta >= 0 :
            r1 = (-b + delta**0.5)/(2*a)
            r2 = (-b - delta**0.5)/(2*a)
            if (r1 >= 0) and (r1 < 10):
                return r1
            if (r2 >= 0) and (r2 < 10):
                return r2
            else:
                print( " no physical solution found for BF")
        else:
            print( " delta < 0")

    # 2. numerical solver
    def SetX(self, X):
        self.X = X
    def EqualX(self, r):
        return self.PredictX(r) - self.X
    def SolveBF_Numerical(self,X):
        self.SetX(X)
        return optimize.brentq(self.EqualX, 0.0, 2.0)

    # 3. test solver and get bf distribution
    def GetBFFromToys_SmearOfConst(self, ntoy=1000, r=1, smearBW=False, smearBt=False):
        Xs = []
        for k in range(ntoy):
            predictx = self.PredictX_SmearOfConst(r,smearBW,smearBt)
            Xs.append(predictx)
        Xs = np.array(Xs)
        bfs= np.array([self.SolveBF(X) for X in Xs])
        return bfs,Xs

    
    def GetBFFromToys_SmearOfSubN(self, ntoy=1000, r=1):
        Xs = []
        for k in range(ntoy):
            predictx = self.PredictX_SmearOfSubN(r)
            Xs.append(predictx)
        Xs = np.array(Xs)
        bfs= np.array([self.SolveBF(X) for X in Xs])
        return bfs,Xs




class BFCalc_Toolbox:
    def __init__(self):
        print('initiate BFCalc_Toolbox')

    def IO_LoadAccTableIntoDf(self):
        dir = "../data/acceptance/"
        ttxs,twxs = 832,35.85*2
        # reshape 21 acc into a matrix
        accmatidx = np.array([[ 0, 2, 9,10,11,15],
                              [ 2, 1,12,13,14,16],
                              [ 9,12, 3, 5, 6,17],
                              [10,13, 5, 4, 7,18],
                              [11,14, 6, 7, 8,19],
                              [15,16,17,18,19,20]])

        selections = ['mumu','mutau','emu','emu2','mu4j','ee','etau','e4j']
        tags       = ['tt_1b','tt_2b','tw_1b','tw_2b','1b','2b']

        dfacc = []
        for selection in selections:
            filename = dir+'Acceptance - {}.csv'.format(selection)
            #df = pd.read_csv(filename,index_col=0)
            #df = df.loc[df.index == 'fraction accepted']
            #acc = np.array(df).astype(np.float)
            acc = np.genfromtxt(filename,delimiter=',')

            for i,tag in enumerate(tags):
                if i < 4:
                    temp = acc[i,:]
                elif tag is '1b':
                    temp = (acc[0,:]*ttxs + acc[2,:]*twxs)/(ttxs+twxs)
                elif tag is '2b':
                    temp = (acc[1,:]*ttxs + acc[3,:]*twxs)/(ttxs+twxs)
                
                dfacc.append((selection,tag,temp[accmatidx]))

        dfacc = pd.DataFrame.from_records(dfacc,columns=['sel','tag','acc'])
        return dfacc
        
    def IO_GetEffAsMatrix(self,cuts, sel,usetag):
        df = pd.read_pickle('../data/pickle_{}/mcsg_softmax/{}.pkl'.format(sel,usetag))

        sc = SelectionCounter()

        accmatidx = np.array([[ 0, 1, 3, 4, 5,15],
                              [ 1, 2, 6, 7, 8,16],
                              [ 3, 6, 9,10,12,17],
                              [ 4, 7,10,11,13,18],
                              [ 5, 8,12,13,14,19],
                              [15,16,17,18,19,20]])
        effs = []
        for cut in cuts:
            n   = np.array(sc.countSlt_ByTauDecay(df[df.softmax>cut]))
            n0  = np.array(sc.countSlt_ByTauDecay(df))
            nonzero = n0>0
            iszero  = n0==0

            eff = np.zeros_like(n)
            eff[nonzero] = n[nonzero]/n0[nonzero]
            eff = eff[accmatidx]
            effs.append(eff)
        return effs
    
    def Plot_Error_rTreuIsOne_GenerateToys_SmearOfConst(self, bf_list, bf_label_list, Xmin,Xmax, smearBW=False, smearBt=False):
        rr = np.arange(0.8,1.21,0.02)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5,9), gridspec_kw={ 'height_ratios':[3,1] })
        fig.subplots_adjust(hspace=0)
        

        for i,bf in enumerate(bf_list):
            usecolor = 'C{}'.format(i)
            uselabel = bf_label_list[i]

            # r and X distributions
            rdis,xdis  = bf.GetBFFromToys_SmearOfConst(ntoy=10000,r=1, smearBW=smearBW,smearBt=smearBt)
            xmean,xstd = xdis.mean(),xdis.std()
            rmean,rstd = rdis.mean(),rdis.std()

            # plot 1
            ax = axes[0]
            ax.axvline(1,0,1,linestyle="--",c='k')

            ax.fill([rmean-rstd,rmean-rstd,rmean+rstd,rmean+rstd],[0,1,1,0], facecolor = usecolor, lw=0,alpha=0.2)
            ax.fill_between(rr,xmean+xstd,xmean-xstd, facecolor = usecolor, lw=0,alpha=0.2)
            ax.plot(rr, np.array([bf.PredictX(r) for r in rr]), c=usecolor, label=uselabel)

            ax.set_ylabel(r"$X$")
            ax.set_ylim(Xmin,Xmax)
            ax.legend(fontsize=10,loc='upper left')
            ax.grid(True)

            # Plot2
            ax = axes[1]
            ax.hist(rdis,bins=np.arange(0.8,1.2,0.01),histtype="step",color= usecolor)
            ax.set_xlabel(r"$r_{calc}=B_\tau / B_\mu$",fontsize=12)
            ax.grid(True)

            # print Error
            print("------- {} ------".format(uselabel))
            print("From 10k Toys : x={:7.5f}+/-{:7.5f}, r={:5.4f}+/-{:5.4f}".format(xmean,xstd,rmean,rstd))
            print("-------------")

    def Plot_Error_rTreuIsOne_GenerateToys_SmearOfSubN(self, bf_list, bf_label_list, Xmin,Xmax):
        rr = np.arange(0.9,1.1,0.01)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5,9), gridspec_kw={ 'height_ratios':[3,1] })
        fig.subplots_adjust(hspace=0)
        

        for i,bf in enumerate(bf_list):
            usecolor = 'C{}'.format(i)
            uselabel = bf_label_list[i]

            # r and X distributions
            rdis,xdis  = bf.GetBFFromToys_SmearOfSubN(ntoy=10000,r=1)
            xmean,xstd = xdis.mean(),xdis.std()
            rmean,rstd = rdis.mean(),rdis.std()

            # plot 1
            ax = axes[0]
            ax.axvline(1,0,1,linestyle="--",c='k')

            ax.fill([rmean-rstd,rmean-rstd,rmean+rstd,rmean+rstd],[0,1,1,0], facecolor = usecolor, lw=0,alpha=0.2)
            ax.fill_between(rr,xmean+xstd,xmean-xstd, facecolor = usecolor, lw=0,alpha=0.2)

            ax.plot(rr, np.array([bf.PredictX(r) for r in rr]), c=usecolor, label=uselabel)

            ax.set_ylabel(r"$X$")
            ax.set_ylim(Xmin,Xmax)
            ax.legend(fontsize=10,loc='upper left')
            ax.grid(True)

            # Plot2
            ax = axes[1]
            ax.hist(rdis,bins=np.arange(0.9,1.1,0.005),histtype="step",color= usecolor)
            ax.set_xlabel(r"$r_{calc}=B_\tau / B_\mu$",fontsize=12)
            ax.grid(True)

            # print Error
            print("------- {} ------".format(uselabel))
            print("From 10k Toys : x={:7.5f}+/-{:7.5f}, r={:5.4f}+/-{:5.4f}".format(xmean,xstd,rmean,rstd))
            x_formula  = bf.PredictX(r=1)
            r_formula  = bf.SolveBF(x_formula)
            rstd_formula,_,xstd_formula = bf.PredictError(r=1)
            print("AnaCalculation: x={:7.5f}+/-{:7.5f}, r={:5.4f}+/-{:5.4f}".format(x_formula,rstd_formula,r_formula,rstd_formula))
            print("-------------")

    def Plot_Error_rTrueRange_GenerateToys_SmearOfSubN(self, bf_list, bf_label_list):
        rr = np.arange(0.9,1.1,0.02)
        plt.figure(figsize=(4,4))

        for i,bf in enumerate(bf_list):
            usecolor = 'C{}'.format(i)
            uselabel = bf_label_list[i]
            
            r_mean,r_std = [], []
            for i in rr:
                rdis,xdis  = bf.GetBFFromToys_SmearOfSubN(ntoy=1000,r=i)
                r_mean.append(rdis.mean())
                r_std.append(rdis.std())
            r_mean,r_std = np.array(r_mean),np.array(r_std)

            plt.fill_between(rr,r_mean+r_std,r_mean-r_std,facecolor=usecolor,edgecolor='None', alpha=0.2)
            plt.plot(rr,r_mean,color=usecolor,label=uselabel)

            plt.grid()
            plt.legend(fontsize=10,loc='upper left')

            plt.xlabel(r'$r_{true}$',fontsize=14)
            plt.ylabel(r'$r_{calc}$',fontsize=14)

    def Plot_Error_rTrueRange_ErrProp(self, bf_list, bf_label_list):
        rr = np.arange(0.9,1.1,0.02)
        plt.figure(figsize=(4,4))

        for i,bf in enumerate(bf_list):
            usecolor = 'C{}'.format(i)
            uselabel = bf_label_list[i]

            r_mean = np.array([bf.SolveBF(bf.PredictX(r)) for r in rr])
            r_std  = np.array([bf.PredictError(r) for r in rr])[:,0]

            plt.fill_between(rr,r_mean+r_std,r_mean-r_std,facecolor=usecolor,edgecolor='None', alpha=0.2)
            plt.plot(rr,r_mean,color=usecolor,label=uselabel)
            plt.legend(fontsize=10,loc='upper left')


            plt.grid()
            plt.xlabel(r'$r_{true}$',fontsize=14)
            plt.ylabel(r'$r_{calc}$',fontsize=14)

    def Plot_ImshowMatrix(self, mtx):
        plt.figure(figsize=(3,3))
        ticks = [r'$e$',r'$\mu$',r'$\tau_e$',r'$\tau_\mu$',r'$\tau_h$',r'$h$']
        plt.imshow(mtx,interpolation='None',cmap='viridis')
        plt.xticks(range(0,6),ticks)
        plt.yticks(range(0,6),ticks)


