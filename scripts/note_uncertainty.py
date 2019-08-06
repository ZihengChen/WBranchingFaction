#!/usr/bin/env python
import utility_common as common
from pylab import *

beginTable = r'''\begin{sidewaystable}[p]
  \small
  \renewcommand{\arraystretch}{1.2}
  \centering

  \begin{tabular}{|l|ccc|ccc|ccc|ccc|ccc|}
  \hline
  Error Source & \multicolumn{3}{c|}{$\mu$-1b} & \multicolumn{3}{c|}{$\mu$-2b} & \multicolumn{3}{c|}{$e$-1b} & \multicolumn{3}{c|}{$e$-2b} \\
  \hline
                & $B_e$ & $B_\mu$ & $B_\tau$ & $B_e$ & $B_\mu$ & $B_\tau$ & $B_e$ & $B_\mu$ & $B_\tau$ & $B_e$ & $B_\mu$ & $B_\tau$ \\
  \hline
'''

endTable = r'''  \end{tabular}
  \caption{ Statistical and systematic error of four categories. }
  \label{tab:syst_alt}
\end{sidewaystable}
'''


stat_colNames = [
    r'StatErr of Data',
    r'StatErr of bg MC',
    r'StatErr of sg MC',
]

syst_colNames = [
    r'PDG err of $Br^\tau_e$',
    r'PDG err of $Br^\tau_\mu$',
    r'2.5$\%$ err of luminosity',
    r'5$\%$ err of tt XS',
    r'5$\%$ err of tW XS',
    r'5$\%$ err of W+Jets XS',
    r'10$\%$ err of Z+Jets XS',
    r'10$\%$ err of VV XS',
    r'25$\%$ err of QCD in $e 4j$',
    r'25$\%$ err of QCD in $\mu 4j$',
    r'25$\%$ err of QCD in $e\tau$',
    r'25$\%$ err of QCD in $\mu\tau$',
    r'0.6$\%$ err of $\epsilon_e$ reco',
    r'1.4$\%$ err of $\epsilon_e$ id',
    r'0.1$\%$ err of $\epsilon_\mu$ reco',
    r'0.2$\%$ err of $\epsilon_\mu$ id',
    r'5$\%$ err of $\epsilon_\tau$',
    r'4.7$\%$ err of $\epsilon_{j\to\tau}$',
    r'0.5$\%$ err of $ES_{e}$',
    r'0.2$\%$ err of $ES_{\mu}$',
    r'1.2$\%$ err of $ES_{\tau\to\pi^\pm}$',
    r'1.2$\%$ err of $ES_{\tau\to\pi^\pm\pi^0}$',
    r'1.2$\%$ err of $ES_{\tau\to3\pi^\pm}$',

    r'0.5$\%$ err of $Br_{\tau\to\pi^\pm}$',
    r'0.5$\%$ err of $Br_{\tau\to\pi^\pm\pi^0}$',
    r'0.2$\%$ err of $Br_{\tau\to\pi^\pm2\pi^0}$',
    r'0.6$\%$ err of $Br_{\tau\to3\pi^\pm}$',
    r'0.6$\%$ err of $Br_{\tau\to3\pi^\pm\pi^0}$',
      
    r'Pileup',
    r'JES',
    r'JER',
    r'Btag',
    r'Mistag',
]

baseDir = common.getBaseDirectory()

stat_sigma = np.load(baseDir+'/data/combine/stat_sigma.npy')
syst_sigma = np.load(baseDir+'/data/combine/syst_sigma.npy')
stat_sigma = np.abs(stat_sigma)/0.1080*100
syst_sigma = np.abs(syst_sigma)/0.1080*100


if __name__ == "__main__":




    
    latex = open(baseDir+'tables/syst_alt.tex','w') 
    latex.write(beginTable)

    ###########################
    # statitiscal
    nrow, ncol = stat_sigma.shape
    for i in range(len(stat_colNames)):
        rowString = '  {:42}'.format(stat_colNames[i])
        for j in range(ncol):
            rowString += ' & {:5.3f}'.format(stat_sigma[i,j])
        rowString += r' \\ '
        rowString += '\n'
        latex.write(rowString)
    latex.write(r'  \hline'+'\n')
        
    # systematics
    nrow, ncol = syst_sigma.shape
    for i in range(len(syst_colNames)):
        rowString = '  {:42}'.format(syst_colNames[i])
        for j in range(ncol):
            rowString += ' & {:5.3f}'.format(syst_sigma[i,j])
        rowString += r' \\ '
        rowString += '\n'
        latex.write(rowString)
    latex.write(r'  \hline'+'\n')

    # total
    total_sigma  = np.sum(stat_sigma**2, axis=0)
    total_sigma += np.sum(syst_sigma**2, axis=0) 
    total_sigma  = total_sigma**0.5

    rowString = '  {:42}'.format('Total')
    for j in range(ncol):
        rowString += ' & {:5.3f}'.format(total_sigma[j])
    rowString += r' \\ '
    rowString += '\n'
    latex.write(rowString)
    latex.write(r'  \hline'+'\n')
    ###########################

        
    latex.write(endTable)
    latex.close()
