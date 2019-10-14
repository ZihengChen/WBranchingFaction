#!/usr/bin/env python

import utility_common as common
from pylab import *

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
    \caption{Composition of accepted $t\bar{t}$+$tW$ events, breakdown by 21 WW decay.  Values are in percent.}
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
sigcomp = np.load(common.getBaseDirectory()+'data/counts/sigcomp_.npy')



if __name__=="__main__":

    latex = open(baseDir+'tables/sigcomp.tex','w') 
    latex.write(beginTable)

    ##########################
    # wright table body
    nrow, ncol = sigcomp.shape
    for i in range(nrow):
        rowString = '    {:34}'.format(colNames[i])
        for j in range(ncol):
            number = sigcomp[i,j] * 100
            if number > 0.5:
                rowString += ' & {:4.1f}'.format(number)
            else:
                rowString += ' & {:4}'.format('  --')

        rowString += r' \\ '
        rowString += '\n'
        latex.write(rowString)
    ##########################
    
    latex.write(endTable)
    latex.close()

    print("latex -- sigcomp tables are update")