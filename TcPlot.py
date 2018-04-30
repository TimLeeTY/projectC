"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Plots all relevant graphs for M, E, C, chi and Tc
Uses data from previously .csv files from before
"""
from plot import NSize, NArr, kTArr
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def movingAvg(arr, n):
    """Calculate moving average of values in arr over length n"""
    csum = arr.cumsum()
    csum[n:] = csum[n:] - csum[:-n]
    return(csum[n - 1:] / n)


def fitTc(Tc, x, NArr):
    [Tc_inf, a, v] = x
    return(((Tc - Tc_inf - a * NArr**(-1/v))**2).sum())


def fitC(CFit, TArr):
    [Tc, a, b] = CFit
    Tlow = TArr[np.where(TArr < Tc)]
    Thigh = TArr[np.where(TArr >= Tc)]
    out = np.concatenate((a*(-np.log((Tc-Tlow)/Tc)), b*(-np.log((Thigh-Tc)/Tc))))
    return(out)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

Ncolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()
fig9, ax9 = plt.subplots()
fig10, ax10 = plt.subplots()

CMaxArr = np.zeros((2, NSize))
chiMaxArr = np.zeros((2, NSize))
kTArrFit = kTArr[np.where((kTArr < 2.5) & (kTArr > 2))]    # kT scaled relative to J

for i in range(NSize):
    N = NArr[i]
    CArr = np.loadtxt('C%i' % (N), unpack=True, delimiter=',').T
    chiArr = np.loadtxt('chiAbs%i' % (N), unpack=True, delimiter=',').T
    CMaxArr[0, i] = kTArr[np.argmax(movingAvg(CArr, 3))]
    chiMaxArr[0, i] = kTArr[np.argmax(movingAvg(chiArr, 3))]
for j in range(3):
    N = [15, 35, 55][j]
    E = np.loadtxt('Eng%i.csv' % (N), unpack=True, delimiter=',').T/N**2
    totMag = np.loadtxt('Mag%i.csv' % (N), unpack=True, delimiter=',').T
    EArr = np.array([E.mean(axis=0), E.std(axis=0)])
    MArr = np.array([np.abs(totMag).mean(axis=0) / N**2, (np.abs(totMag)/N**2).std(axis=0)])
    MagAvg = np.array([totMag.mean(axis=0), (totMag**2).mean(axis=0)]).T
    totMag = np.loadtxt('Mag%i.csv' % (N), unpack=True, delimiter=',').T
    CArr = np.loadtxt('C%i' % (N), unpack=True, delimiter=',').T
    chiArr = np.loadtxt('chi%i' % (N), unpack=True, delimiter=',').T/N**2
    chiAbsArr = np.loadtxt('chiAbs%i' % (N), unpack=True, delimiter=',').T/N**2
    CFit0 = [2.3, 0, 0]
    CFit = op.minimize(lambda x: ((fitC(x, kTArrFit)-CArr[0][np.where((kTArr < 2.5) & (kTArr > 2))])**2).sum(), CFit0)
    print(CFit.x)
    ax2.plot(kTArr, EArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax2.errorbar(kTArr[2*(j)::6], EArr[0][2*(j)::6], EArr[1][2*(j)::6], ls="None", color=Ncolor[j])
    # ax4.plot(kTArrFit, fitC(CFit.x, kTArrFit))
    ax4.plot(kTArr, CArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax4.errorbar(kTArr[2*(j)::6], CArr[0][2*(j)::6], CArr[1][2*(j)::6], ls="None", color=Ncolor[j])
    ax3.plot(kTArr, MArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax3.errorbar(kTArr[2*(j)::6], MArr[0][2*(j)::6], MArr[1][2*(j)::6], ls="None", color=Ncolor[j])
    ax6.plot(kTArr, chiArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax6.errorbar(kTArr[2*(j)::6], chiArr[0][2*(j)::6], chiArr[1][2*(j)::6], ls="None", color=Ncolor[j])
    ax8.plot(kTArr, chiAbsArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax8.errorbar(kTArr[2*(j)::6], chiAbsArr[0][2*(j)::6], chiAbsArr[1][2*(j)::6], ls="None", color=Ncolor[j])


ax2.legend()
ax3.legend()
ax4.legend()
ax6.legend()
ax8.legend()
ax3.axvspan(2.25, 2.35, color='red', alpha=0.3)
ax2.set_ylabel(r'Energy per site, $\langle \vert E \vert \rangle/N^2/J$')
ax2.set_xlabel(r'Temperature, $T$')
ax3.set_ylabel(r'Magnetisation per site, $\langle \vert M \vert \rangle/N^2$, /\mu')
ax3.set_xlabel(r'Temperature, $T$')
ax4.set_ylabel(r'Specific Heat per site, $C/N^2$')
ax4.set_xlabel(r'Temperature, $T$')
ax6.set_ylabel(r'Magnetic Susceptibility per site, $\chi/N^2$')
ax6.set_xlabel(r'Temperature, $T$')
ax8.set_ylabel(r'Modified Magnetic Susceptibility per site, $\chi^\prime/N^2$')
ax8.set_xlabel(r'Temperature, $T$')
fig2.savefig('2.pdf', format='pdf')
fig3.savefig('3.pdf', format='pdf')
fig4.savefig('4.pdf', format='pdf')
fig6.savefig('6.pdf', format='pdf')
fig8.savefig('8.pdf', format="pdf")

x0 = [2, 4, 1]
xOpt = op.minimize(lambda x: fitTc(CMaxArr[0], x, NArr), x0)
print(xOpt.x)
try:
    xCF, xCov = op.curve_fit((lambda NArr, a, v: 2/np.log(1+np.sqrt(2)) + a * NArr**(-1/v)),
                             NArr, CMaxArr[0], bounds=([0, 0], [100, 2]))
    x0 = xCF
    print("nu = %f +/- %f" % (x0[1], np.sqrt(np.diag(xCov))[1]))
except (RuntimeError):
    x0 = xOpt.x

TcFit0 = [2, -1]
TcFitx, TcFitCov = op.curve_fit((lambda NArr, Tc_inf, m: Tc_inf + m * (1/NArr)), NArr, CMaxArr[0], TcFit0)
TcFit0 = TcFitx
print(TcFit0)
print(np.sqrt(np.diag(TcFitCov)))

ax5.errorbar(NArr, CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None", marker='+')
ax5.plot(NArr, TcFit0[0] + TcFit0[1] * 1/NArr)
ax5.plot(NArr, 2/np.log(1 + np.sqrt(2))*np.ones(NSize))
# ax5.errorbar(NArr, CMaxArr[1])
ax7.errorbar(1/(NArr), CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None", marker='+')
ax7.plot(1/(NArr), TcFit0[0] + TcFit0[1]/NArr)

ax5.set_xlabel(r'$N$')
ax5.set_ylabel(r'$T_\mathrm{c}$')
ax7.set_xlabel(r'$1/N$')
ax7.set_ylabel(r'$T_\mathrm{c}$')
fig5.savefig('5.pdf', format="pdf")
fig7.savefig('7.pdf', format="pdf")

CMaxArr = chiMaxArr
x0 = [2, 4, 1]
xOpt = op.minimize(lambda x: fitTc(CMaxArr[0], x, NArr), x0)
print(xOpt.x)
try:
    xCF, xCov = op.curve_fit((lambda NArr, a, v: 2/np.log(1+np.sqrt(2)) + a * NArr**(-1/v)),
                             NArr, CMaxArr[0], bounds=([0, 0], [10, 2]))
    x0 = xCF
    print("nu = %f +/- %f" % (x0[1], np.sqrt(np.diag(xCov))[1]))
except (RuntimeError):
    x0 = xOpt.x

TcFit0 = [2, -1]
TcFitx, TcFitCov = op.curve_fit((lambda NArr, Tc_inf, m: Tc_inf + m * (1/NArr)), NArr, CMaxArr[0], TcFit0)
TcFit0 = TcFitx
print(TcFit0)
print(np.sqrt(np.diag(TcFitCov)))

ax9.errorbar(NArr, CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None", marker='+')
ax9.plot(NArr, TcFit0[0] + TcFit0[1] * 1/NArr)
ax9.plot(NArr, 2/np.log(1 + np.sqrt(2))*np.ones(NSize))
# ax5.errorbar(NArr, CMaxArr[1])
ax10.errorbar(1/(NArr), CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None", marker='+')
ax10.plot(1/(NArr), TcFit0[0] + TcFit0[1]/NArr)

ax9.set_xlabel(r'$N$')
ax9.set_ylabel(r'$T_\mathrm{c}$')
ax10.set_xlabel(r'$1/N$')
ax10.set_ylabel(r'$T_\mathrm{c}$')
fig9.savefig('9.pdf', format="pdf")
fig10.savefig('10.pdf', format="pdf")
plt.show()
