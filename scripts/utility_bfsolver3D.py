from pylab import *
import pandas as pd
from scipy import optimize
from utility_plotter import *
from scipy.optimize import root
import sympy as sym

BWPDG = np.array([0.1071,0.1063,0.1138])

class NSignal:

    def __init__(self,a,xs,lumin, bte, btm):

        self.a     = a        
        self.xs    = xs
        self.lumin = lumin

        self.bte = bte
        self.btm = btm
        self.bth = 1-bte-btm

        #self.quadCoeff = self.QuadCoeffFromAnalytics()
        self.quadCoeff = self.__QuadCoeffFromBMatrix()
        self.measuredNSignal = self.PredictNSignal() # for initialize, further setup may needed
    
    # 1. Predict N signal
    def PredictNSignal(self, BW=BWPDG):
        terms = np.array([BW[0]**2,BW[1]**2,BW[2]**2, BW[0]*BW[1],BW[0]*BW[2],BW[1]*BW[2], BW[0],BW[1],BW[2],1])
        n = np.dot(self.quadCoeff,terms) * (self.xs*self.lumin)
        return n

    # 2. Calculate Measured N signal
    def SetMeasuredNSignal(self,nData,nMcbg):
        # if nData is not setup, simpluy use PredictNSignal+nmcbg
        if nData == 0:
            nData = self.PredictNSignal() + nMcbg
        self.measuredNSignal = nData - nMcbg
        return self.measuredNSignal

    ##########################################
    # Private helper functions about BMatrix #
    ##########################################
    def __QuadCoeffFromAnalytics(self):

        '''
        This result is identical to the other coeff methord __QuadCoeffFromBMatrix
        a-j are coeff for terms [xx,yy,zz,xy,xz,yz,x,y,z,1].
        and in this methord they come from Mathmatica symbolic calculator.

        In Mathmatica notebook
        ./notebooks/mathmatica/AnalyticalEquations.nb
        explicit forms of all coeffs are give analytically
        '''

        acc = self.a
        bte = self.bte
        btm = self.btm
        bth = self.bth

        j = acc[5,5]

        a = acc[0,0]-2*acc[0,5]+acc[5,5]
        b = acc[1,1]-2*acc[1,5]+acc[5,5]
        c = bte**2*acc[2,2] + 2*bte*btm*acc[2,3] + 2*bte*bth*acc[2,4] - 2*bte*acc[2,5] + \
            btm**2*acc[3,3]                      + 2*bth*btm*acc[3,4] - 2*btm*acc[3,5] + \
            bth**2*acc[4,4]                                           - 2*bth*acc[4,5] + \
            acc[5,5]

        d = 2*acc[0,1] - 2*acc[0,5] - 2*acc[1,5] + 2*acc[5,5]
        e = 2*bte*acc[0,2] + 2*btm*acc[0,3] + 2*bth*acc[0,4] - 2*acc[0,5] - 2*bte*acc[2,5] - 2*btm*acc[3,5] - 2*bth*acc[4,5] + 2*acc[5,5]
        f = 2*bte*acc[1,2] + 2*btm*acc[1,3] + 2*bth*acc[1,4] - 2*acc[1,5] - 2*bte*acc[2,5] - 2*btm*acc[3,5] - 2*bth*acc[4,5] + 2*acc[5,5]

        g = 2*acc[0,5] - 2*acc[5,5]
        h = 2*acc[1,5] - 2*acc[5,5]
        i = 2*bte*acc[2,5] + 2*btm*acc[3,5] + 2*bth*acc[4,5] - 2*acc[5,5]

        coeff = np.array([a,b,c,d,e,f,g,h,i,j]) 
        return  coeff # return 1x10 vector

    def __QuadCoeffFromBMatrix(self):

        '''
        This result is identical to the other coeff methord __QuadCoeffFromAnalytics
        a-j are coeff for terms [xx,yy,zz,xy,xz,yz,x,y,z,1]
        and in this methord they come directly from picking 
        relevant terms of AccBMatrix in python
        '''

        F = self.__AccBMatrix

        j =  F(np.array([0,0,0]))

        g = (F(np.array([1,0,0])) - F(np.array([-1,0,0])))/2
        a =  F(np.array([1,0,0])) - g - j

        h = (F(np.array([0,1,0])) - F(np.array([0,-1,0])))/2
        b =  F(np.array([0,1,0])) - h - j

        i = (F(np.array([0,0,1])) - F(np.array([0,0,-1])))/2
        c =  F(np.array([0,0,1])) - i - j

        d = F(np.array([1,1,0])) -a-b-g-h-j
        e = F(np.array([1,0,1])) -a-c-g-i-j
        f = F(np.array([0,1,1])) -b-c-h-i-j

        coeff = np.array([a,b,c,d,e,f,g,h,i,j]) *self.xs*self.lumin
        return  coeff # return 1x10 vector
    
    def __AccBMatrix(self, BW):

        '''
        A helper function of __QuadCoeffFromBMatrix to make it easier to
        collect relevant terms in python
        '''

        bVector = np.array([ BW[0],BW[1], BW[2]*self.bte,  BW[2]*self.btm, BW[2]*self.bth, 1-np.sum(BW) ])
        bMatrix = np.outer(bVector,bVector)
        return np.sum( self.a * bMatrix )


    
class BFCalc3D_ThreeSelectorRatios:

    def __init__(self, a, xs = 832+35.85*2, lumin=35847, bte=0.1785, btm=0.1736 ):
            
        self.xs = xs
        self.lumin = lumin

        self.nSignal = [ NSignal(a[i], xs,lumin,bte,btm) for i in range(4) ]

        self.measuredX = self.PredictX()

    # 1. Measured X
    def SetMeasuredX(self,nData = np.zeros(4), nMcbg = np.zeros(4)):
        
        for i in range(4):
            self.nSignal[i].SetMeasuredNSignal(nData[i],nMcbg[i])

        n = np.array( [ it.measuredNSignal for it in self.nSignal] )
        self.measuredX = n[0:3]/np.sum(n)
        return self.measuredX

    # 2. Predict X
    def PredictX(self, BW=BWPDG):
        n = np.array( [ it.PredictNSignal(BW) for it in self.nSignal] )
        return n[0:3]/np.sum(n)
    
    # 3. solve analytical quadratic equations

    def QuadEqn(self, obsX):

        quadEqn = []

        x,y,z = sym.symbols('x,y,z',real=True)
        terms = [x*x,y*y,z*z,x*y,x*z,y*z,x,y,z,1]

        # for each channel get quadCoeff
        coeff = np.array( [ it.quadCoeff for it in self.nSignal] )
        coeffNorm = np.sum(coeff,axis=0)

        for i in range(3):

            coeffQuadEqn = obsX[i]*coeffNorm - coeff[i]
            
            temp = 0
            for k,term in enumerate(terms):
                temp += coeffQuadEqn[k] * term
            quadEqn.append(temp)

        self.quadEqn = quadEqn

        return quadEqn

    def EvaluateLeftSideOfQuadEqn(self,paraBW):
        x,y,z = sym.symbols('x,y,z',real=True)
        leftSide = [ float(eq.evalf(subs={x: paraBW[0], y: paraBW[1], z: paraBW[2]})) for eq in self.quadEqn ]
        return np.array(leftSide) # return 1x3 vector

    def SolveQuadEqn(self, obsX):
        eqn = self.QuadEqn(obsX)
        paraBW0 = BWPDG
        solution  = root(self.EvaluateLeftSideOfQuadEqn, paraBW0).x
        return solution

    # 3.4 give ObsX and BWe and BWm, output BWt
    def PlaneFromQuadEqn(self,obsX):
        eqn = self.QuadEqn(obsX)
        x,y,z = sym.symbols('x,y,z',real=True)
        planes = [ sym.solve(it,[z])[1] for it in eqn ]
        self.planes = planes
        return planes


class BFCalc3D_Toolbox:
    def __init__(self):
        self.ttxs,self.twxs = 832,35.85*2
        self.c_ttxs = self.ttxs/(self.ttxs+self.twxs)
        self.c_twxs = self.twxs/(self.ttxs+self.twxs)

        self.accmatidx = np.array([ [ 0, 2, 9,10,11,15],
                                    [ 2, 1,12,13,14,16],
                                    [ 9,12, 3, 5, 6,17],
                                    [10,13, 5, 4, 7,18],
                                    [11,14, 6, 7, 8,19],
                                    [15,16,17,18,19,20]])
        
        self.ngen_tt = np.array([1811409.0, 1811532.0, 3620281.0, 57057.0, 54227.0, 111125.0, 418126.0, 407605.0, 763147.0, 642646.0, 626108.0, 2350887.0, 643368.0, 627277.0, 2353727.0, 22670017.0, 22653517.0, 4016590.0, 3923504.0, 14718933.0, 70930278.0])[self.accmatidx]
        self.ngen_tw = np.array([23105.0, 23040.0, 46342.0, 694.0, 717.0, 1420.0, 5260.0, 5158.0, 9727.0, 8199.0, 8029.0, 30054.0, 8163.0, 8006.0, 29970.0, 290101.0, 289467.0, 50864.0, 50050.0, 187515.0, 904067.0])[self.accmatidx]

    
    def IO_LoadAccTableIntoDf(self,name=None):
        dir = "/mnt/data/zchen/Analysis/acceptance/"
        
        selections = ['mumu','mutau','emu','emu2','mu4j','ee','etau','e4j']
        tags       = ['tt_1b','tt_2b','tw_1b','tw_2b','1b','2b']

        dfacc = []
        for selection in selections:
            if name is None:
                filename = dir+'Acceptance - {}.csv'.format(selection)
            else:
                filename = dir+'Acc_{}_{}.csv'.format(name,selection)

            acc = np.genfromtxt(filename,delimiter=',')

            for i,tag in enumerate(tags):
                if i < 4:
                    temp = acc[i,:]
                elif tag is '1b':
                    temp = acc[0,:]*self.c_ttxs + acc[2,:]*self.c_twxs
                elif tag is '2b':
                    temp = acc[1,:]*self.c_ttxs + acc[3,:]*self.c_twxs
                
                dfacc.append((selection,tag,temp[self.accmatidx]))

        dfacc = pd.DataFrame.from_records(dfacc,columns=['sel','tag','acc'])
        return dfacc
    

    def GetAcc(self,trigger,usetag,name=None):
        a, aVar = [],[]

        df_acc = self.IO_LoadAccTableIntoDf(name)

        if trigger == "mu":
            selectionList = ["emu","mumu","mutau","mu4j"]
        else:
            selectionList = ["ee","emu2","etau","e4j"]
        
            
        for selection in selectionList:
            a.append( df_acc.loc[(df_acc.sel==selection) & (df_acc.tag==usetag),'acc'].values[0] )

            a_tt = df_acc.loc[(df_acc.sel==selection) & (df_acc.tag=='tt_'+usetag),'acc'].values[0]
            a_tw = df_acc.loc[(df_acc.sel==selection) & (df_acc.tag=='tw_'+usetag),'acc'].values[0]

            aVar.append( self.c_ttxs**2 * a_tt*(1-a_tt)/self.ngen_tt + self.c_twxs**2 * a_tw*(1-a_tw)/self.ngen_tw )
        
        a = np.array(a)  
        aVar = np.array(aVar)
        return a, aVar


    def GetNData(self, trigger,usetag):

        n,nVar = [],[]

        if "1b" in usetag:
            nbjet = "==1"
        if "2b" in usetag:
            nbjet = ">1"

        if trigger == "mu":
            selectionList = ["emu","mumu","mutau","mu4j"]
        else:
            selectionList = ["ee","emu2","etau","e4j"]
        
            
        for selection in selectionList:
            if selection == "emu2":
                pickledir  =  "/mnt/data/zchen/Analysis/pickle/emu/"
            else:
                pickledir  =  "/mnt/data/zchen/Analysis/pickle/{}/".format(selection)
            cuts = GetSelectionCut(selection) + "& (nBJets{})".format(nbjet)
            
            Data = LoadDataframe(pickledir + "data2016").query(cuts)
            if selection in ["emu","emu2"]:
                Data = Data.drop_duplicates(subset=['runNumber', 'evtNumber'])
            
            n.append( np.sum(Data.eventWeight) )
            nVar.append( np.sum(Data.eventWeight) )
            
        n = np.array(n)  
        nVar = np.array(nVar)
        return n, nVar

    def GetNMcbg(self, trigger,usetag):
        n,nVar = [],[]

        if "1b" in usetag:
            nbjet = "==1"
        if "2b" in usetag:
            nbjet = ">1"

        if trigger == "mu":
            selectionList = ["emu","mumu","mutau","mu4j"]
        else:
            selectionList = ["ee","emu2","etau","e4j"]
            
        for selection in selectionList:
            if selection == "emu2":
                pickledir  =  "/mnt/data/zchen/Analysis/pickle/emu/"
            else:
                pickledir  =  "/mnt/data/zchen/Analysis/pickle/{}/".format(selection)
            cuts = GetSelectionCut(selection) + "& (nBJets{})".format(nbjet)
            MCzz = LoadDataframe(pickledir + "mcdiboson").query(cuts)
            MCdy = LoadDataframe(pickledir + "mcdy").query(cuts)
            MCbg = pd.concat([MCzz,MCdy],ignore_index=True)
            
            n.append(np.sum(MCbg.eventWeight))
            nVar.append( np.sum(MCbg.eventWeight**2) )
            
        n = np.array(n)  
        nVar = np.array(nVar)

        return n, nVar

    def GetNFake(self, trigger,usetag,fakeNorm={"mutau":0.03,"mu4j":0.05}):
        if trigger == "e":
            return np.zeros(4), np.zeros(4)
        

        if "1b" in usetag:
            nbjet = "==1"
        if "2b" in usetag:
            nbjet = ">1"

        n,nVar = [0,0],[0,0]
        
        for selection in ["mutau","mu4j"]:
            pickledir  =  "/mnt/data/zchen/Analysis/pickle/{}/".format(selection)
            Fake = LoadDataframe(pickledir + "data2016_inverseISO").query("nBJets{}&(lepton1_pt > 30)".format(nbjet))
            n.append( fakeNorm[selection]*np.sum(Fake.eventWeight) )
            nVar.append( fakeNorm[selection]*np.sum(Fake.eventWeight) )
        
        n = np.array(n)  
        nVar = np.array(nVar)
        return n, nVar

    def SmearAcc(self,a,aVar):
        smear = np.zeros_like(a)
        for slt in range(4):
            for i in range(smear.shape[0]):
                for j in range(smear.shape[1]):
                    if (i<=j) & (a[slt,i,j]>0.001):
                        smear[slt,i,j] = np.random.normal( 0, aVar[slt,i,j]**0.5)
                        if i != j :
                            smear[slt,j,i] = smear[slt,i,j]
        return smear
            
    def Plot_Imshow4Matrix(self, a, trigger):
        a_e,a_m,a_t,a_h = a[0],a[1],a[2],a[3]

        plt.figure(facecolor="w",figsize=(12,3))


        if trigger is "mu":
            titles=[r"$A_{\mu e} [\%]$", 
                    r"$A_{\mu \mu} [\%]$",
                    r"$A_{\mu \tau_h} [\%]$",
                    r"$A_{\mu h} [\%]$"]
        else:
            titles=[r"$A_{e e} [\%]$", 
                    r"$A_{e \mu} [\%]$",
                    r"$A_{e \tau_h} [\%]$",
                    r"$A_{e h} [\%]$"]
        
        for islt, mtx in enumerate([a_e,a_m,a_t,a_h]):

            plt.subplot(1,4,islt+1)
            ticks = [r'$e$',r'$\mu$',r'$\tau_e$',r'$\tau_\mu$',r'$\tau_h$',r'$h$']
            plt.imshow(mtx,interpolation='None',cmap='viridis')
            for i in range(0,6):
                for j in range(0,6):
                    h = mtx[i,j]
                    if h > 0.0005:
                        plt.text(i-0.4,j+0.2,"{:3.1f}".format(100*h),color="k")
            plt.xticks(range(0,6),ticks)
            plt.yticks(range(0,6),ticks)
            plt.title(titles[islt])



'''
class BFCalc3D_ErrorPropagater:
    def __init__(self,bf,x1,x2,x3,norm,usetag,trigger="mu"):
        self.usetag = usetag
        self.trigger = trigger
        self.bf = bf
        self.xs = bf.xs
        self.lumin = bf.lumin
        self.thrd = 0.001 # thrd for contributing acceptance
        self.eps  = 1e-6   # varational infinitesimal for gradiant
        
        # acceptance
        self.a_e  = bf.a_e
        self.a_m  = bf.a_m
        self.a_t  = bf.a_t
        self.a_h  = bf.a_h
        
        # a and var_a with vectorized form
        self.a         = np.copy(np.array([self.a_e,self.a_m,self.a_t,self.a_h]))
        self.a_vec     = self.a_vectorized() 
    
        self.var_a     = self.var_a()
        self.var_a_vec = self.var_a_vectorized()

        # normalized yields
        self.norm = norm
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        self.var_x = np.array([[ x1*(1-x1)/norm, -x1*x2/norm, -x1*x3/norm],
                               [ -x1*x2/norm, x2*(1-x2)/norm, -x2*x3/norm], 
                               [ -x1*x3/norm, -x2*x3/norm, x3*(1-x3)/norm]])

        self.bf1, self.bf2, self.bf3 = bf.SovleBF(x1,x2,x3)


    
    def grad2_g_x(self):
        eps = self.eps
        
        g1,g2,g3=self.bf.SovleBF(self.x1+eps,self.x2,self.x3)
        g1_xe = (g1-self.bf1)/eps
        g2_xe = (g2-self.bf2)/eps
        g3_xe = (g3-self.bf3)/eps
        
        g1,g2,g3 = self.bf.SovleBF(self.x1,self.x2+eps,self.x3)
        g1_xm = (g1-self.bf1)/eps
        g2_xm = (g2-self.bf2)/eps
        g3_xm = (g3-self.bf3)/eps

        g1,g2,g3 = self.bf.SovleBF(self.x1,self.x2,self.x3+eps)
        g1_xt = (g1-self.bf1)/eps
        g2_xt = (g2-self.bf2)/eps
        g3_xt = (g3-self.bf3)/eps

        g1_x = np.array([g1_xe,g1_xm,g1_xt])
        g2_x = np.array([g2_xe,g2_xm,g2_xt])
        g3_x = np.array([g3_xe,g3_xm,g3_xt])

        g1_x = np.outer(g1_x,g1_x) # grad squared  
        g2_x = np.outer(g2_x,g2_x) # grad squared  
        g3_x = np.outer(g3_x,g3_x) # grad squared  

        return np.array([g1_x,g2_x,g3_x])
    
    def grad2_g_a(self):
        eps = self.eps

        gradiant_g = []

        for slt in range(4):
            for i in range(6):
                for j in range(6):
                    if (i<=j) & (self.a[slt,i,j]>self.thrd):
                        # make a copy of self.a so that we can variate it
                        _a = np.copy(self.a)
                        # variate i,j and corresponding j,i
                        _a[slt,i,j] = _a[slt,i,j] + eps
                        if i != j:
                            _a[slt,j,i] = _a[slt,j,i] + eps
                        # get variational solver
                        _bf = BFCalc3D_ThreeSelectorRatios(_a[0], _a[1], _a[2], _a[3],
                                                           xs=self.xs, lumin=self.lumin)
                        g1,g2,g3 = _bf.SovleBF(self.x1,self.x2,self.x3)
                        gradiant_g.append([(g1-self.bf1)/eps, (g2-self.bf2)/eps, (g3-self.bf3)/eps])

        gradiant_g = np.array( gradiant_g )**2 # grad squared  
        gradiant_g = gradiant_g.T
        return gradiant_g
    
    def grad2_g_Bt(self):
        eps = self.eps
        gradiant_g = []

        _bf = BFCalc3D_ThreeSelectorRatios(self.a[0], self.a[1], self.a[2], self.a[3],
                                       xs=832+35.85*2,lumin=35847,
                                       Bt_e=0.1785+eps, Bt_m=0.1736)
        g1,g2,g3 = _bf.SovleBF(self.x1,self.x2,self.x3)
        gradiant_g.append([(g1-self.bf1)/eps, (g2-self.bf2)/eps, (g3-self.bf3)/eps])

        _bf = BFCalc3D_ThreeSelectorRatios(self.a[0], self.a[1], self.a[2], self.a[3],
                                       xs=832+35.85*2,lumin=35847,
                                       Bt_e=0.1785, Bt_m=0.1736+eps)
        g1,g2,g3 = _bf.SovleBF(self.x1,self.x2,self.x3)
        gradiant_g.append([(g1-self.bf1)/eps, (g2-self.bf2)/eps, (g3-self.bf3)/eps])

        gradiant_g = np.array( gradiant_g )**2 # grad squared  
        gradiant_g = gradiant_g.T
        return gradiant_g
        

    def a_vectorized(self):
        a_list = []
        for slt in range(4):
            for i in range(6):
                for j in range(6):
                    if (i<=j) & (self.a[slt,i,j]>self.thrd):
                        a_list.append(self.a[slt,i,j])
        return np.array(a_list)
    
    def var_a_vectorized(self):
        # vectorize total var_a
        var_a_list = []
        for slt in range(4):
            for i in range(6):
                for j in range(6):
                    if (i<=j) & (self.a[slt,i,j]>self.thrd):
                        var_a_list.append(self.var_a[slt,i,j])
        return np.array(var_a_list)

    def var_a(self):
        tb = BFCalc3D_Toolbox()
        df_acc = tb.IO_LoadAccTableIntoDf()

        if self.trigger == "mu":
            var_a_e = self.var_a_helper("emu",  df_acc)
            var_a_m = self.var_a_helper("mumu", df_acc)
            var_a_t = self.var_a_helper("mutau",df_acc)
            var_a_h = self.var_a_helper("mu4j", df_acc)
        else:
            var_a_e = self.var_a_helper("ee",   df_acc)
            var_a_m = self.var_a_helper("emu2", df_acc)
            var_a_t = self.var_a_helper("etau", df_acc)
            var_a_h = self.var_a_helper("e4j",  df_acc)

        # get total var_a
        var_a = np.copy(np.array([var_a_e,var_a_m, var_a_t, var_a_h]))
        for slt in range(4):
            for i in range(6):
                for j in range(6):
                    if (self.a[slt,i,j]<=self.thrd):
                        var_a[slt,i,j]=0.0
        return var_a


    def var_a_helper(self, selection,df_acc):

        ttxs,twxs = 832,35.85*2
        c_ttxs = ttxs/(ttxs+twxs)
        c_twxs = twxs/(ttxs+twxs)

        # reshape 21 acc into a matrix
        accmatidx = np.array([[ 0, 2, 9,10,11,15],
                              [ 2, 1,12,13,14,16],
                              [ 9,12, 3, 5, 6,17],
                              [10,13, 5, 4, 7,18],
                              [11,14, 6, 7, 8,19],
                              [15,16,17,18,19,20]])

        a_tt = df_acc.loc[(df_acc.sel==selection)  & (df_acc.tag=='tt_'+self.usetag),'acc'].values[0]
        a_tw = df_acc.loc[(df_acc.sel==selection)  & (df_acc.tag=='tw_'+self.usetag),'acc'].values[0]

        ngen_tt =  np.array([1811409.0, 1811532.0, 3620281.0, 57057.0, 54227.0, 111125.0, 418126.0, 407605.0, 763147.0, 642646.0, 626108.0, 2350887.0, 643368.0, 627277.0, 2353727.0, 22670017.0, 22653517.0, 4016590.0, 3923504.0, 14718933.0, 70930278.0])[accmatidx]
        ngen_tw =  np.array([23105.0, 23040.0, 46342.0, 694.0, 717.0, 1420.0, 5260.0, 5158.0, 9727.0, 8199.0, 8029.0, 30054.0, 8163.0, 8006.0, 29970.0, 290101.0, 289467.0, 50864.0, 50050.0, 187515.0, 904067.0])[accmatidx]*10

        var_a  = c_ttxs**2 * a_tt*(1-a_tt)/ngen_tt
        var_a += c_twxs**2 * a_tw*(1-a_tw)/ngen_tw

        return var_a
    
    def a_fluctuated(self):
        noise = np.zeros_like(self.a)
        for slt in range(4):
            for i in range(noise.shape[0]):
                for j in range(noise.shape[1]):
                    if (i<=j) & (self.a[slt,i,j]>self.thrd):
                        noise[slt,i,j] = noise[slt,i,j] + np.random.normal( 0, self.var_a[slt,i,j]**0.5)
                        if i != j :
                            noise[slt,j,i] = noise[slt,i,j]
        return self.a + noise




    def Plot_toy(self, bf1_list,bf2_list,bf3_list):

        mybin = np.arange(0.100, 0.120, 0.0002)
        plt.rc("figure",facecolor="w",figsize=(6,4))

        c0 = 'b'
        c1 = 'r'
        c2 = 'g'


        var = bf1_list
        s   = var.std()
        m   = var.mean()
        a   = (2*np.pi*s**2)**(-0.5)
        plt.hist(var,mybin,histtype='stepfilled',normed=True,alpha=0.2,color=c0,label="BW-e={:6.4f}+/-{:6.4f}".format(m,s))
        plt.plot(mybin+0.00005, a*np.exp(-(mybin - m)**2/(2*s**2)),'k-',lw=2)
        plt.errorbar(m, a, xerr=s, fmt='o',c='k')

        var = bf2_list
        s   = var.std()
        m   = var.mean()
        a   = (2*np.pi*s**2)**(-0.5)
        plt.hist(var,mybin,histtype='stepfilled',normed=True,alpha=0.2,color=c1,label="BW-m={:6.4f}+/-{:6.4f}".format(m,s))
        plt.plot(mybin+0.00005, a*np.exp(-(mybin - m)**2/(2*s**2)),'k-',lw=2)
        plt.errorbar(m, a, xerr=s, fmt='o',c='k')

        var = bf3_list
        s   = var.std()
        m   = var.mean()
        a   = (2*np.pi*s**2)**(-0.5)
        plt.hist(var,mybin,histtype='stepfilled',normed=True,alpha=0.2,color=c2,label="BW-t={:6.4f}+/-{:6.4f}".format(m,s))
        plt.plot(mybin+0.00005, a*np.exp(-(mybin - m)**2/(2*s**2)),'k-',lw=2)
        plt.errorbar(m, a, xerr=s, fmt='o',c='k')
        plt.xlabel('BW-l')

        # pdg
        #meam,std = 0.1086,0.0009
        #plt.axvline(meam,c='k',linestyle='--')
        #plt.fill([meam-std,meam-std,meam+std,meam+std], [0,1e5,1e5,0],'k',lw=0,alpha=0.1) 
        #plt.axvline(0.1080,c='k',linestyle='--',label='BW-l in MCtt')


        meam,std = 0.1071,0.0016
        plt.axvline(meam,c=c0,linestyle='--',lw=2)
        plt.fill([meam-std,meam-std,meam+std,meam+std], [0,1e5,1e5,0],c0,lw=0,alpha=0.1) 

        meam,std = 0.1063,0.0015
        plt.axvline(meam,c=c1,linestyle='--',lw=2)
        plt.fill([meam-std,meam-std,meam+std,meam+std], [0,1e5,1e5,0],c1,lw=0,alpha=0.1) 

        meam,std = 0.1138,0.0021
        plt.axvline(meam,c=c2,linestyle='--',lw=2)
        plt.fill([meam-std,meam-std,meam+std,meam+std], [0,1e5,1e5,0],c2,lw=0,alpha=0.1) 


        plt.grid()
        plt.legend(fontsize=10)
        plt.ylim(0,1000)



etrigger
# ee
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 2600.94,41.96,229.28,12.26
mc_data_1b,mc_data_2b = 28217.00,8318.00
#mc_data_1b,mc_data_2b = 30009.09,8821.14# this is MCtot
n_mm,sign_mm = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mm,sign_mm = mc_data_2b-mc_bkg_2b,(mc_data_2b+mc_bkg_2bu**2)


# et
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu =760.23,37.75,62.62,8.19
mc_data_1b,mc_data_2b = 17534.00,4532.00
#mc_data_1b,mc_data_2b = 17684.70,4736.11# this is MCtot
n_mt,sign_mt = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mt,sign_mt = mc_data_2b-mc_bkg_2b,(mc_data_2b+mc_bkg_2bu**2)


# em
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 175.50,12.64,10.57,2.83
mc_data_1b,mc_data_2b = 13151.00,4098.00
#mc_data_1b,mc_data_2b = 13643.08,4249.92# this is MCtot
n_em,sign_em = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_em,sign_em = mc_data_2b-mc_bkg_2b, mc_data_2b+mc_bkg_2bu**2


# eh
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 23570.06,201.79,2900.84,73.11
mc_data_1b,mc_data_2b = 400884.00,128975.00
#mc_data_1b,mc_data_2b = 408037.12,134347.36# this is MCtot
n_mh,sign_mh = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mh,sign_mh = mc_data_2b-mc_bkg_2b, mc_data_2b+mc_bkg_2bu**2


mutrigger
# mm
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 7243.09,72.84,538.85,19.54
#mc_data_1b,mc_data_2b = 70006.00,20074.00
mc_data_1b,mc_data_2b = 70162.81,20224.58 # this is MCtot
n_mm,sign_mm = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mm,sign_mm = mc_data_2b-mc_bkg_2b,(mc_data_2b+mc_bkg_2bu**2)


# mt
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu =1019.22,50.78,96.42,19.83
#mc_data_1b,mc_data_2b = 24785.00,6581.00
mc_data_1b,mc_data_2b = 24481.78,6676.30# this is MCtot
n_mt,sign_mt = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mt,sign_mt = mc_data_2b-mc_bkg_2b,(mc_data_2b+mc_bkg_2bu**2)


# me
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 474.09,20.45,35.88,5.53
#mc_data_1b,mc_data_2b = 78540.00,24911.00
mc_data_1b,mc_data_2b = 81636.23,25604.51# this is MCtot
n_em,sign_em = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_em,sign_em = mc_data_2b-mc_bkg_2b, mc_data_2b+mc_bkg_2bu**2


# mh
########################################################################
mc_bkg_1b,mc_bkg_1bu,mc_bkg_2b,mc_bkg_2bu = 29563.56,251.67,3244.14,78.43
#mc_data_1b,mc_data_2b = 633455.00,204747.00
mc_data_1b,mc_data_2b = 608002.69,202104.38# this is MCtot
n_mh,sign_mh = mc_data_1b-mc_bkg_1b, mc_data_1b+mc_bkg_1bu**2
#n_mh,sign_mh = mc_data_2b-mc_bkg_2b, mc_data_2b+mc_bkg_2bu**2


bf  = BFCalc3D_ThreeSelectorRatios(a_mm, a_mt, a_em, a_mh, xs=xs, lumin=35847,
                                   expyield_mm=n_mm,expyield_unc2_mm=sign_mm,
                                   expyield_mt=n_mt,expyield_unc2_mt=sign_mt,
                                   expyield_em=n_em,expyield_unc2_em=sign_em,
                                   expyield_mh=n_mh,expyield_unc2_mh=sign_mh,
                                   IsBinomial = True
                                  )

    x1_list.append(x1)
    x2_list.append(x2)
    x3_list.append(x3)
x1_list = np.array(x1_list)
x2_list = np.array(x2_list)
x3_list = np.array(x3_list)


a_ = a_mt
for i in range(6):
    for j in range(6):
        if (i<=j) and (a_[i,j]>0.001) :
            a_[i,j] = a_[i,j] + eps
            x1_,x2_,x3_ = BFCalc3D_ThreeSelectorRatios(a_mm, a_, a_em, a_mh, 
                                                       xs=xs, lumin=35847).PredictXs()
            g1_,g2_,g3_ = bf.SovleBF(x1_,x2_,x3_)
            gradiant_g.append([ (g1_-bf1)/eps, (g2_-bf2)/eps, (g3_-bf3)/eps ])
            
            
a_ = a_em
for i in range(6):
    for j in range(6):
        if (i<=j) and (a_[i,j]>0.001) :
            a_[i,j] = a_[i,j] + eps
            x1_,x2_,x3_ = BFCalc3D_ThreeSelectorRatios(a_mm, a_mt, a_, a_mh, 
                                                       xs=xs, lumin=35847).PredictXs()
            g1_,g2_,g3_ = bf.SovleBF(x1_,x2_,x3_)
            gradiant_g.append([ (g1_-bf1)/eps, (g2_-bf2)/eps, (g3_-bf3)/eps ])
a_ = a_mh
for i in range(6):
    for j in range(6):
        if (i<=j) and (a_[i,j]>0.001) :
            a_[i,j] = a_[i,j] + eps
            x1_,x2_,x3_ = BFCalc3D_ThreeSelectorRatios(a_mm, a_mt, a_em, a_, 
                                                       xs=xs, lumin=35847).PredictXs()
            g1_,g2_,g3_ = bf.SovleBF(x1_,x2_,x3_)
            gradiant_g.append([ (g1_-bf1)/eps, (g2_-bf2)/eps, (g3_-bf3)/eps ])

'''