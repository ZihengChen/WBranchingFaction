#!/usr/bin/env python

import utility_common as common
from pylab import *
import pandas as pd


def getAcc(counts, i_trnb,i_channel):
    acc = counts.acc
    accVar = counts.accVar

    ttxs,txs = 832, 35.85*2
    c_ttxs = ttxs/(ttxs+txs)
    c_txs  = txs /(ttxs+txs)
    ca    = np.array([c_txs,c_ttxs]).reshape(1,2,1,1)
    caVar = np.array([c_txs**2,c_ttxs**2]).reshape(1,2,1,1)
    
    # add tt and tW together
    a    = np.sum(acc[i_trnb]*ca,axis=1)
    aVar = np.sum(accVar[i_trnb]*caVar,axis=1)
    
    # shape 6x6 matrix to 21x1 array
    a = common.matrixToArray(a[i_channel])
    aVar = common.matrixToArray(aVar[i_channel])
    
    return a, aVar

def getAccForTable(counts):
    a,aVar = [],[]

    for i_trnb in [0,2]:
        for i_channel in range(4):
            # 1b
            temp_a, temp_aVar = getAcc(counts,i_trnb,i_channel)
            a.append(temp_a)
            aVar.append(temp_aVar)
            # 2b
            temp_a, temp_aVar = getAcc(counts,i_trnb+1,i_channel)
            a.append(temp_a)
            aVar.append(temp_aVar)

    a = np.array(a).T
    aVar = np.array(aVar).T
    
    return a, aVar

beginTable = r'''\begin{sidewaystable}[p]
    \centering
    \setlength{\tabcolsep}{0.4em}
    \renewcommand{\arraystretch}{1.5}
    \small
    \begin{tabular}{|l|cc|cc|cc|cc|cc|cc|cc|cc|}
    
    
    \hline
    channel & \multicolumn{2}{|c|}{$\mu e$} & \multicolumn{2}{c|}{$\mu\mu$} & \multicolumn{2}{|c|}{$\mu \tau$} & \multicolumn{2}{|c|}{$\mu$+jets} & \multicolumn{2}{|c|}{$ee$} & \multicolumn{2}{|c|}{$e\mu$} & \multicolumn{2}{|c|}{$e \tau$} & \multicolumn{2}{|c|}{$e+jets$} \\
    \hline
    $\rm n_{b tag}$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ & $n_b=1$ & $n_b\geq2$ \\ 
    \hline
    
'''

endTable = r'''
    \hline
    \end{tabular}
    \caption{Efficiency of $t\bar{t}$+$tW$ events, breakdown by 21 WW decay.  Values are in percent.}
    \label{sigcomp}
    
\end{sidewaystable}
'''

colNames = [
    r'$tt/tW \to ee$',
    r'$tt/tW \to \mu\mu$',
    r'$tt/tW \to e\mu$',
    r'$tt/tW \to \tau_{e}\tau_{e}$',
    r'$tt/tW \to \tau_{\mu}\tau_{\mu}$',
    r'$tt/tW \to \tau_{e}\tau_{\mu}$',
    r'$tt/tW \to \tau_{e}\tau_{h}$',
    r'$tt/tW \to \tau_{\mu}\tau_{h}$',
    r'$tt/tW \to \tau_{h}\tau_{h}$',
    r'$tt/tW \to e\tau_{e}$',
    r'$tt/tW \to e\tau_{\mu}$',
    r'$tt/tW \to e\tau_{h}$',
    r'$tt/tW \to \mu\tau_{e}$',
    r'$tt/tW \to \mu\tau_{\mu}$',
    r'$tt/tW \to \mu\tau_{h}$',
    r'$tt/tW \to eh$',
    r'$tt/tW \to \mu h$',
    r'$tt/tW \to \tau_{e}h$',
    r'$tt/tW \to \tau_{\mu}h$',
    r'$tt/tW \to \tau_{h}h$',
    r'$tt/tW \to hh$',
]

baseDir = common.getBaseDirectory() 
counts = pd.read_pickle( baseDir + "data/counts/count_.pkl")
a, aVar = getAccForTable(counts)


if __name__=="__main__":

    latex = open(baseDir+'tables/sigacc.tex','w') 
    latex.write(beginTable)

    ##########################
    # wright table body
    nrow, ncol = a.shape
    for i in range(nrow):
        rowString = '    {:34}'.format(colNames[i])
        for j in range(ncol):
            number = a[i,j] * 100
            numberVar = aVar[i,j]**0.5 * 10000
            if number > 0.1:
                rowString += ' & {:5.2f}({:1.0f})'.format(number,numberVar)
            else:
                rowString += ' & {:^8}'.format('--')

        rowString += r' \\ '
        rowString += '\n'
        latex.write(rowString)
    ##########################
    
    latex.write(endTable)
    latex.close()

    print("latex -- sigacc tables are update")