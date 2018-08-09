
from utility_dfcutter import *
import sympy as sym
from scipy.optimize import root

BWPDG = np.array([0.1071,0.1063,0.1138])

class NSignal:

    def __init__(self,a,xs,lumin, bte, btm):

        self.a     = a        
        self.xs    = xs
        self.lumin = lumin

        self.bte = bte
        self.btm = btm
        self.bth = 1-bte-btm

        self.quadCoeff = self._quadCoeffFromAnalytics()
        self.quadCoeffFromBMatrix = self._quadCoeffFromBMatrix()
        self.measuredNSignal = self.predictNSignal() # for initialize, further setup may needed
    
    # 1. Predict N signal
    def predictNSignal(self, BW=BWPDG):
        terms = np.array([BW[0]**2,BW[1]**2,BW[2]**2, BW[0]*BW[1],BW[0]*BW[2],BW[1]*BW[2], BW[0],BW[1],BW[2],1])
        n = np.dot(self.quadCoeff,terms) * (self.xs*self.lumin)
        return n

    # 2. Calculate Measured N signal
    def setMeasuredNSignal(self,nData,nMcbg):
        # if nData is not setup, simpluy use predictNSignal+nmcbg
        if nData == 0:
            nData = self.predictNSignal() + nMcbg
        self.measuredNSignal = nData - nMcbg
        return self.measuredNSignal

    ##########################################
    # Private helper functions about BMatrix #
    ##########################################
    def _quadCoeffFromAnalytics(self):

        '''
        This result is identical to the other coeff methord _quadCoeffFromBMatrix
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

    def _quadCoeffFromBMatrix(self):

        '''
        This result is identical to the other coeff methord _quadCoeffFromAnalytics
        a-j are coeff for terms [xx,yy,zz,xy,xz,yz,x,y,z,1]
        and in this methord they come directly from picking 
        relevant terms of AccBMatrix in python
        '''

        F = self._accBMatrix

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

        coeff = np.array([a,b,c,d,e,f,g,h,i,j]) 
        return  coeff # return 1x10 vector
    
    def _accBMatrix(self, BW):

        '''
        A helper function of _quadCoeffFromBMatrix to make it easier to
        collect relevant terms in python
        '''

        bVector = np.array([ BW[0],BW[1], BW[2]*self.bte,  BW[2]*self.btm, BW[2]*self.bth, 1-np.sum(BW) ])
        bMatrix = np.outer(bVector,bVector)
        return np.sum( self.a * bMatrix )

class rSovler:
    def __init__(self, a, xs = 832+35.85*2, lumin=35847, bwl = 0.1080, bte=0.1785, btm=0.1736 ):
        
        self.xs = xs
        self.lumin = lumin
        self.bwl = bwl

        self.nSignal_den =  NSignal(a[0], xs,lumin,bte,btm)
        self.nSignal_num =  NSignal(a[1], xs,lumin,bte,btm)

        self.measuredX = self.predictX(self.bwl,)

    
    def predictX(self, r=1):
        BW = self.bwl * np.array([1,1,r])
        self.nSignal_den.predictNSignal()
        n = np.array( [ it.predictNSignal(BW) for it in self.nSignal] )
        return n[0:3]/np.sum(n)


    
    def getQuadEqn(self, obsX):

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


# super class
class BFSolver:

    def __init__(self, a, xs = 832+35.85*2, lumin=35847, bte=0.1785, btm=0.1736 ):
            
        self.xs = xs
        self.lumin = lumin

        self.nSignal = [ NSignal(a[i], xs,lumin,bte,btm) for i in range(4) ]

        self.measuredX = self.predictX()

    # 1. Measured X
    def setMeasuredX(self,nData = np.zeros(4), nMcbg = np.zeros(4)):
        
        for i in range(4):
            self.nSignal[i].setMeasuredNSignal(nData[i],nMcbg[i])

        n = np.array( [ it.measuredNSignal for it in self.nSignal] )
        self.measuredX = n[0:3]/np.sum(n)
        return self.measuredX

    # 2. Predict X
    def predictX(self, BW=BWPDG):
        n = np.array( [ it.predictNSignal(BW) for it in self.nSignal] )
        return n[0:3]/np.sum(n)
    
    # 3. solve analytical quadratic equations

    def getQuadEqn(self, obsX):

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

# 3D Inheritance of BFSolver
class BFSolver3D (BFSolver):

    def __init__(self, a, xs = 832+35.85*2, lumin=35847, bte=0.1785, btm=0.1736  ):
        super().__init__(a, xs, lumin, bte, btm)
    
    def evaluateLeftSideOfQuadEqn(self,paraBW):
        x,y,z = sym.symbols('x,y,z',real=True)
        leftSide = [ float(eq.evalf(subs={x: paraBW[0], y: paraBW[1], z: paraBW[2]})) for eq in self.quadEqn ]
        return np.array(leftSide) # return 1x3 vector

    def solveQuadEqn(self, obsX):
        eqn = self.getQuadEqn(obsX)
        paraBW0 = BWPDG
        solution  = root(self.evaluateLeftSideOfQuadEqn, paraBW0).x
        return solution

    # 3D planes
    def planeFromquadEqn(self,obsX):
        eqn = self.getQuadEqn(obsX)
        x,y,z = sym.symbols('x,y,z',real=True)
        planes = [ sym.solve(it,[z])[1] for it in eqn ]
        self.planes = planes
        return planes

# 1D Inheritance of BFSolver
class BFSolver1D (BFSolver):
    def __init__(self, a, xs = 832+35.85*2, lumin=35847, bte=0.1785, btm=0.1736, bWe=0.1071, bWm=0.1063 ):
        super().__init__(a, xs, lumin, bte, btm)
        self.bWe,self.bWm = bWe,bWm

    def evaluateLeftSideOfQuadEqn(self,paraBW):
        x,y,z = sym.symbols('x,y,z',real=True)
        leftSide = [ float(eq.evalf(subs={x:self.bWe, y:self.bWm, z:paraBW[i]})) for i,eq in enumerate(self.quadEqn) ]
        return np.array(leftSide) # return 1x3 vector

    def solveQuadEqn(self, obsX):
        eqn = self.getQuadEqn(obsX)
        paraBW0 = np.array([0.11,0.11,0.11])
        solution  = root(self.evaluateLeftSideOfQuadEqn, paraBW0).x
        return solution


class BFSolver_Toolbox:
    def __init__(self):
        nothing = 0

    def imshow4Matrix(self, a, trigger, showError=False):
        a_e,a_m,a_t,a_h = a[0],a[1],a[2],a[3]

        plt.figure(facecolor="w",figsize=(12,3))


        if trigger is "mu":
            titles=[r"$A_{\mu e} [10^{-2}]$", 
                    r"$A_{\mu \mu} [10^{-2}]$",
                    r"$A_{\mu \tau_h} [10^{-2}]$",
                    r"$A_{\mu h} [10^{-2}]$"]
        else:
            titles=[r"$A_{e e} [10^{-2}]$", 
                    r"$A_{e \mu} [10^{-2}]$",
                    r"$A_{e \tau_h} [10^{-2}]$",
                    r"$A_{e h} [10^{-2}]$"]

        if showError is True:
            if trigger is "mu":
                titles=[r"$\delta A_{\mu e} / A_{\mu e} [\%]$", 
                        r"$\delta A_{\mu \mu} / A_{\mu \mu} [\%]$",
                        r"$\delta A_{\mu \tau_h} / A_{\mu \tau_h} [\%]$",
                        r"$\delta A_{\mu h} / A_{\mu h} [\%]$"]
            else:
                titles=[r"$\delta A_{e e} /A_{e e} [\%]$", 
                        r"$\delta A_{e \mu}/A_{e \mu} [\%]$",
                        r"$\delta A_{e \tau_h}/A_{e \tau_h} [\%]$",
                        r"$\delta A_{e h}/A_{e h} [\%]$"]

        
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


