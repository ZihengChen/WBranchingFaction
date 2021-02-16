
#!/usr/bin/env python

import utility_common as common
from pylab import *
import pandas as pd


def getN(counts, i_trnb, i_channel):
    n, nVar = [],[]

    nsum, nsumVar = 0,0
    
    # QCD
    tempn = counts.nfake[i_trnb][i_channel]
    tempnVar = counts.nfakeVar[i_trnb][i_channel]
    # tempnVar += (0.25*tempn)**2 # add QCD rate
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar
    
    # VV
    tempn = counts.nmcbg[i_trnb][i_channel][0]
    tempnVar = counts.nmcbgVar[i_trnb][i_channel][0]
    # tempnVar += (0.10*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar
    
    # g
    tempn = counts.nmcbg[i_trnb][i_channel][1]
    tempnVar = counts.nmcbgVar[i_trnb][i_channel][1]
    # tempnVar += (0.10*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar
    
    # z
    tempn = counts.nmcbg[i_trnb][i_channel][2]
    tempnVar = counts.nmcbgVar[i_trnb][i_channel][2]
    # tempnVar += (0.05*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar

    # w
    tempn = counts.nmcbg[i_trnb][i_channel][3]
    tempnVar = counts.nmcbgVar[i_trnb][i_channel][3]
    # tempnVar += (0.05*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar

    # t other
    tempn = counts.nmcbg[i_trnb][i_channel][4]
    tempnVar = counts.nmcbgVar[i_trnb][i_channel][4]
    # tempnVar += (0.05*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar

    
    # tw
    tempn = counts.nmcsg[i_trnb][i_channel][0]
    tempnVar = counts.nmcsgVar[i_trnb][i_channel][0]
    # tempnVar += (0.05*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar
    
     # tt
    tempn = counts.nmcsg[i_trnb][i_channel][1]
    tempnVar = counts.nmcsgVar[i_trnb][i_channel][1]
    # tempnVar += (0.05*tempn)**2 + (0.025*tempn)**2 # add xs and lumin
    n.append( tempn ) 
    nVar.append( tempnVar )
    nsum += tempn
    nsumVar += tempnVar    
    
    # total expectation
    n.append( nsum ) 
    nVar.append( nsumVar )
    
    # data
    n.append( counts.ndata[i_trnb][i_channel] ) 
    nVar.append( counts.ndataVar[i_trnb][i_channel] ) 
    
    return np.array(n), np.array(nVar)

def getNForTable(counts):
    n,nVar = [],[]

    for i_trnb in [0,2]:
        for i_channel in range(4):
            # 1b
            temp_n, temp_nVar = getN(counts,i_trnb,i_channel)
            n.append(temp_n)
            nVar.append(temp_nVar)
            # 2b
            temp_n, temp_nVar = getN(counts,i_trnb+1,i_channel)
            n.append(temp_n)
            nVar.append(temp_nVar)

    n = np.array(n)
    nVar = np.array(nVar)
    
    return n, nVar


beginTable = r'''\begin{sidewaystable}[p]
    \centering
    \setlength{\tabcolsep}{0.4em}
    \renewcommand{\arraystretch}{2}
    \small
    \begin{tabular}{l|cccccc|cc}
    \hline
        & QCD & VV  & $\gamma$ & Z & W & t & tW & tt & total & data      \\
    \hline
    
'''

endTable = r'''
    \end{tabular}
    \caption{Estimates of the yields. The estimate of the expected yield is compared to
    the yield observed from data.  Uncertainties are statistical only.
    \label{tab:yields}}
\end{sidewaystable}
'''

colNames = [
    r'$\mu e$, $n_b=1$', 
    r'$\mu e$, $n_b\geq2$',
    r'$\mu\mu$, $n_b=1$',
    r'$\mu\mu$, $n_b\geq2$',
    r'$\mu\tau$, $n_b=1$', 
    r'$\mu\tau$, $n_b\geq2$',
    r'$\mu$+jets, $n_b=1$',
    r'$\mu$+jets, $n_b\geq2$',
    r'$e e$, $n_b=1$',
    r'$e e$, $n_b\geq2$',
    r'$e\mu$, $n_b=1$',
    r'$e\mu$, $n_b\geq2$',
    r'$e\tau$, $n_b=1$',
    r'$e\tau$, $n_b\geq2$',
    r'$e$+jets, $n_b=1$',
    r'$e$+jets, $n_b\geq2$'
]
    

if __name__ == "__main__":
    baseDir = common.getBaseDirectory() 
    counts = pd.read_pickle( baseDir + "data/counts/count_.pkl")
    n, nVar = getNForTable(counts)


    latex = open(baseDir+'tables/yields.tex','w') 
    latex.write(beginTable)

    ##########################
    # wright table body
    nrow, ncol = n.shape
    for i in range(nrow):
        rowString = '    {:34}'.format(colNames[i])
        for j in range(ncol):
            number = n[i,j]
            numberVar = nVar[i,j]**0.5
            if number > 0:
                rowString += ' & {:8.1f}$\pm${:7.1f}'.format(number,numberVar)
            else:
                rowString += ' & {:>8}$\pm${:>7}'.format('--', '--')
        rowString += r' \\ '
        rowString += '\n'
        latex.write(rowString)
        
        if i%2==1:
            latex.write(r'''    \hline
''')
    ##########################

    latex.write(endTable)
    latex.close()

    print("latex -- yields tables are update")