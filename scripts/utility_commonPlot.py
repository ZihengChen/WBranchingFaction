import pandas as pd
from pylab import *

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



def showParameterCovMat(cor,sig):
    fig, axes = plt.subplots(2, 1, sharex=True, facecolor='w',
                         gridspec_kw={'height_ratios':[6,1]},
                         figsize=(10,12))
    fig.subplots_adjust(hspace=0)


    # make plots
    lablesName = sysLabelsName()

    lablesPos = np.arange(0,len(lablesName),1)  

    
    ax = axes[0]
    ax.imshow(cor,cmap='PRGn_r',vmax=1,vmin=-1)
    ax.set_xticks(lablesPos)
    ax.set_xticklabels(lablesName)
    ax.set_yticks(lablesPos)
    ax.set_yticklabels(lablesName)
    for i in lablesPos:
        for j in lablesPos:
            value = cor[i,j]
            if abs(value)>0.5:
                fcolor = 'w'
            else:
                fcolor = 'k'
                
            ax.text(i-0.3,j+0.1,'{:4.2f}'.format(cor[i,j]),fontsize=6,color=fcolor )
            
    ax.axvline(2.5,color='grey',linewidth=1,linestyle='--')
    ax.axhline(2.5,color='grey',linewidth=1,linestyle='--')
    ax.set_ylim(24.5,-0.5)

    
    ax = axes[1]

    height = np.r_[np.zeros(3), 1/sig[3:]]
    ax.bar(lablesPos,height,color='grey')
    ax.axvline(2.5,color='grey',linewidth=1,linestyle='--')
    ax.axhline(1  ,color='k',linewidth=1,linestyle='-')
    #ax.set_xticks(lablesPos,lablesName)
    ax.set_xlim(-0.5,24.5)
    ax.set_ylim(0,10)
    ax.grid(axis='y',linestyle='--')
    ax.set_ylabel('constraint')

    # plt.title(r'$\beta_e   =10.80\times(1\pm${:4.2f}%),  '.format(sigma[0]/0.1080*100) + 
    #         r'$\beta_\mu =10.80\times(1\pm${:4.2f}%),  '.format(sigma[1]/0.1080*100) + 
    #         r'$\beta_\tau=10.80\times(1\pm${:4.2f}%)   '.format(sigma[2]/0.1080*100),
    #         fontsize=12
    #         )


def sysLabelsName():
    # make plots
    lablesName = [  r'$\beta_e$',r'$\beta_\mu$',r'$\beta_\tau$',
                    r"$b^\tau_\mu$",r"$b^\tau_e$",#r"$b^\tau_h$",
                    r"$\sigma_{tt}$",r"$\sigma_{tW}$",
                    r"$\sigma_{W}$",r"$\sigma_{Z}$",r"$\sigma_{VV}$",
                    r'$f_e$',r'$f_\mu$',r'$f_\tau$','L',
                    r"$\epsilon_e$",r"$\epsilon_\mu$",r"$\epsilon_\tau$",r"$j \to \tau$",
                    r"$E_e$",r"$E_\mu$",r"$E_\tau$",
                    "JES","JER","b","bMis"
                    ]
    return  lablesName



def showLossHistory(losses):
    plt.figure(facecolor='w',figsize=(6,4))
    plt.plot(losses, lw=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True,linestyle='--',alpha=0.6)


def showPerformance():
    trainingTime=np.array([ 0.52,  0.68, 1.58])
    hessianTime = np.array([1.45, 0.77, 1.32])

    c = np.arange(3)
    width = 0.3    

    displace = width/2+0.02

    plt.figure(facecolor='w',figsize=(6,4))
    plt.bar(c-displace,trainingTime,width, label='Training Process')
    plt.bar(c+displace,hessianTime,width, label='Hessian Calculation')
    #plt.yscale('log')
    plt.xticks(c,['Scipy','pyTorch (CPU)','pyTorch (GPU)'])
    plt.ylabel('running time [s]',fontsize=11)
    plt.legend(loc='upper left',fontsize=12)
    plt.ylim(0,2.5)
    plt.grid(axis='y',linestyle='--')
    for i in np.arange(3):
        h = trainingTime[i]
        plt.text(i-displace-0.1,0.1+h,str(h))
        h = hessianTime[i]
        plt.text(i+displace-0.1,0.1+h,str(h))
        
    #plt.savefig('../plots/fit/performance.png',dpi=300)