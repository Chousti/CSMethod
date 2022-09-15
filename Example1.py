#Example code to plot all \lambda_5, \kappa_5 modes
from CSMethod import *
import numpy as np
import matplotlib.pyplot as plt

#Plotting pre-amble
#plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False, "axes.unicode_minus": False})
font = {'family' : 'sans', 'weight' : 'normal', 'size'   : 10}
plt.rc('font', **font)
plt.rc('pgf', texsystem='pdflatex')
factor = 0.95
width = 5.78939 
plt.gcf().set_size_inches(width * factor, width * 9/16 * factor) 

#Main code
N = 8         #Number of Chebyshev components used
Wm = 0.315    #Present-day Matter-density Parameter
n = 5         #Perturbative order
[Lambda, Kappa, LambdaOld, KappaOld] = LCDMSolver(n,N,Wm)
xs = np.linspace(0,1,100)

plt.figure(1)

for i in range(0,len(Lambda)):
    tot = total(xs,Lambda[i])
    tot2 = total(xs,Kappa[i])
    if i == 0:
        plt.plot(xs,tot/tot[0],'r', label= r'$\lambda_n^{(l)}$')
        plt.plot(xs,tot2/tot2[0],'b',label=r'$\kappa_n^{(l)}$')
    else:
        plt.plot(xs,tot/tot[0],'r')
        plt.plot(xs,tot2/tot2[0],'b')

#Plotting
plt.ylabel(r'$\lambda_5^{(l)}/[\lambda_5^{(l)}]^{\mathrm{EdS}},\;\kappa_5^{(l)}/[\kappa_5^{(l)}]^{\mathrm{EdS}}$')
plt.xlabel(r'$a$')
plt.xlim([0,1])
plt.title(r'$\mathrm{Normalised\;solutions\;for\;time-dependent\;functions\;for\;}n = 5, \; \forall l$')
plt.legend()
plt.tight_layout()
plt.show()
