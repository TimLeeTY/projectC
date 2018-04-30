---
geometry: margin=2cm
output: pdf_document
header-includes:
    - \usepackage{fullpage}
---

# Appendix A: `ising.py`{-}

```
"""
"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Finds properties of a Ferromagnet using the Ising model, sampling spins at
random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np


def makeM(N, p, TSize):
    """Initialises the spins in the system"""
    ret = np.empty((N, N, TSize))
    if p == 1:
        return(np.ones((N, N, TSize), dtype=np.int8))
    else:
        M = np.concatenate((np.ones(int(N**2 * p)), -1 * np.ones(N**2 - int(N**2 * p))))
        for i in range(TSize):
            ret[:, :, i] = (np.random.permutation(M)).reshape((N, N))
            return(ret.astype(np.int8))


def energy(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=0) + np.roll(Mag, -1, axis=0))
    totEng = eng.sum(axis=(0, 1))
    return(totEng)


def MCStep(N, H, mu, J, kTArr, arrSize, TSize):
    """
    Performs each arrSize steps of the Metropolis algorithm, each sampling N^2=Ntot 
    points in the lattice, all temperatures are evolved simultaneously
    """
    E = np.zeros((arrSize, TSize))  # Holds energies of the system
    Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
    Mag[0] = np.ones((N, N, TSize))  # initialise homogeneous spin configuration
    E[0] = energy(Mag[0], H, mu, J)
    for arr in range(1, arrSize):
        if arr % 100 == 0:
            print(arr)
        """sample spins randomly, allowing for repeats within each step"""
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            """count number of neighbouring spins that are aligned"""
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+
                             Mag[arr, i, j-1] +Mag[arr, i, j+1]))/2+2)
            """map flipP onto sCount based on nSpins for all temperatures"""
            sCount = flipP[(nSpins, np.arange(TSize))]
            rSamp = np.random.rand(TSize)
            """compare random samples to the flipP and flip if sCount > rSamp"""
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
        E[arr] = energy(Mag[arr], H, mu, J)
    return(Mag[nRelax:].sum(axis=(1, 2)), E[nRelax:])


TSize = 100                     # Number of samples of temperature to be used
H, mu, J = 0, 1, 1
kTArr = np.linspace(1.6, 3, TSize) * J    # kT scaled relative to J
NArr = np.arange(10, 60, 5)     # Array that holds the values of N to be use
NSize = len(NArr)

if H == 0:
    flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) 
                      else 1 for kT in kTArr] for i in np.arange(5)])

nRelax = 5000
arrSize = 50000

if __name__ == '__main__':
    for j in range(NSize):
        N = NArr[j]
        print('N= %i' % (N))
        [totMag, E] = MCStep(N, H, mu, J, kTArr, arrSize, TSize)
        np.savetxt('Mag%i.csv' % (N), totMag, delimiter=',')
        np.savetxt('Eng%i.csv' % (N), E, delimiter=',')
```
\newpage

# Appendix B: `plot.py`{-}

```
"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Finds heat capacity and susceptibility using results from ising.py
Calculates associated errors using bootstrap method
"""

from ising import NSize, NArr, TSize, kTArr, nRelax, arrSize
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def bootstrap(EArr, totMag, tauC, kT):
    """ Finds error using the bootstrap resampling method"""
    nSamp = 10
    Cout = np.zeros((tauC, 2))
    ChiOut = np.zeros((tauC, 2))
    ChiAbsOut = np.zeros((tauC, 2))
    for j in range(tauC):
        EIndep = EArr[j::tauC]
        MagIndep = totMag[j::tauC]
        n = len(EIndep)
        C = np.zeros(n)
        chi = np.zeros(n)
        chiAbs = np.zeros(n)
        samples = np.random.choice(n, (n, nSamp))
        for i in samples:
            C[i] = np.divide(EIndep[samples[i]].var(), kT**2)/N**2
            chi[i] = np.divide(MagIndep[samples[i]].var(), kT)
            chiAbs[i] = np.divide(np.abs(MagIndep[samples[i]]).var(), kT)
        Cout[j] = [C.mean(axis=0), C.std(axis=0)]
        ChiOut[j] = [chi.mean(axis=0), chi.std(axis=0)]
        ChiAbsOut[j] = [chiAbs.mean(axis=0), chiAbs.std(axis=0)]
    return(Cout.mean(axis=0), ChiOut.mean(axis=0), ChiAbsOut.mean(axis=0))


def fitC(CFit, TArr):
    """ Fits an arbitrary curve to C as a function of T$ """
    [Tc, a, b] = CFit
    Tlow = TArr[np.where(TArr < Tc)]
    Thigh = TArr[np.where(TArr >= Tc)]
    out = np.concatenate((a*(-np.log((Tc-Tlow)/Tc)), b*(-np.log((Thigh-Tc)/Tc))))
    return(out)


def movingAvg(arr, n):
    """Calculate moving average of values in arr over length n"""
    csum = arr.cumsum()
    csum[n:] = csum[n:] - csum[:-n]
    return(csum[n - 1:] / n)


def autoCorr(totMag, MagAvg, tau):
    """Returns the correlation as a function of tau """
    MMag = totMag - MagAvg[0]
    return(np.mean(MMag[tau:] * MMag[:-tau]) / MagAvg[1])


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

kTArrFit = kTArr[np.where((kTArr < 2.5) & (kTArr > 2))  # kT scaled relative to J
CMaxArr = np.zeros((2, NSize))

if __name__ == '__main__':
    fig1, ax1 = plt.subplots()
    for j in range(NSize):
        N = NArr[j]
        """initialise arrays of observables as function of temperature"""
        tauC = np.zeros((2, TSize))
        CArr = np.zeros((2, TSize))
        chiArr = np.zeros((2, TSize))
        chiAbsArr = np.zeros((2, TSize))
        """read data from csv files"""
        E = np.loadtxt('Eng%i.csv' % (N), unpack=True, delimiter=',').T
        totMag = np.loadtxt('Mag%i.csv' % (N), unpack=True, delimiter=',').T
        tauArr = range(1, arrSize-nRelax-1)
        MagAvg = np.array([totMag.mean(axis=0), 
                          ((totMag-totMag.mean(axis=0))**2).mean(axis=0)]).T
        for i in range(TSize):
            for tau in tauArr:
                """loop through tau to find when A(t) < exp(-1)"""
                if np.abs(autoCorr(totMag[:, i], MagAvg[i], tau)) < np.exp(-1):
                    tauC[0, i] = tau - 0.5
                    [CArr[:, i], chiArr[:, i], chiAbsArr[:, i]] = 
                        bootstrap(E[:, i], totMag[:, i], tau, kTArr[i])
                    print('N=%f, i=%i, tau=%i' % (N, i, tau))
                    break
        """taking a moving average before averaging to avoid anomalous points"""
        CMaxArr[0, j] = kTArr[np.argmax(movingAvg(CArr, 3))+1]
        """fit arbitrary function to C as a rough guideline for T_c"""
        CFit0 = [2.3, 0, 0]
        CFit = op.minimize(
            lambda x: ((fitC(x, kTArrFit)-CArr[0][np.where((kTArr < 2.5) & 
                        (kTArr > 2))])**2).sum(), CFit0)
        print(CFit.x)
        CMaxArr[1, j] = CFit.x[0]
        """plot of tauC not included in report"""
        ax1.errorbar(kTArr, tauC[0], tauC[1], label=r'$N=%i$' % N)
        """save files to csv to be plotted in TcPlot.py"""
        np.savetxt('C%i' % (N), CArr, delimiter=',')
        np.savetxt('chi%i' % (N), chiArr, delimiter=',')
        np.savetxt('chiAbs%i' % (N), chiAbsArr, delimiter=',')
    print(CMaxArr)
    np.savetxt('CMax.csv', CMaxArr, delimiter=',')
    ax1.legend()
    fig1.savefig('1.pdf', format='pdf')
    plt.show()
``` 
\newpage

# Appendix C: `spins.py`
```
"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Plots spin configurations as 2D images
Still and animated versions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

kTArr = np.array([4])
TSize = len(kTArr)
H, mu, J = 0, 1, 1
N = 50
flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) 
                  if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])
arrSize = 22

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.ones((N, N, TSize))
for arr in range(1, arrSize):
    if arr in [2, 21]:
        for j in range(TSize):
            kT = kTArr[j]
            plt.figure()
            img = plt.imshow(Mag[arr-1][:, :, j], cmap='gray')
            plt.savefig("spin/T%.0f-arr%iones.pdf" % (kT*10, arr), format="pdf")
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+
                          Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1

kTArr = np.array([1, 2.7, 4])
TSize = 3
H, mu, J = 0, 1, 1
N = 50
flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) 
                  else 1 for kT in kTArr] for i in np.arange(5)])

arrSize = 22

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.random.choice((-1, 1), (N, N, TSize))
for arr in range(1, arrSize):
    if arr in [3, 21]:
        for j in range(TSize):
            kT = kTArr[j]
            plt.figure()
            img = plt.imshow(Mag[arr-1][:, :, j], cmap='gray')
            plt.savefig("spin/T%.0f-arr%i.pdf" % (kT*10, arr), format="pdf")
    """
    Performs each step of the MC technique, 
    each sampling N^2=Ntot points in the lattice
    """
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+
                         Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1


def animate(i):
    im.set_array(Mag[i, :, :, j])
    return im,


for j in range(TSize):
    fig = plt.figure()
    ims = []
    for i in range(arrSize):
        im = plt.imshow(Mag[i, :, :, j], cmap='gray',  animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, repeat_delay=1000)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='tyl35'), bitrate=1800)
    ani.save('T=%i.mp4' % j, writer=writer)

arrSize = 400
N = 50

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.random.choice((-1, 1), (N, N, TSize))
for arr in range(1, arrSize):
    """
    Performs each step of the MC technique, 
    each sampling N^2=Ntot points in the lattice
    """
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+
                         Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1

plt.figure()
for j in range(TSize):
    kT = kTArr[j]
    plt.plot((Mag.sum(axis=(1, 2))[:, j]), label="T=%.1f" % kT)
plt.legend()
plt.xlabel(r"MC Steps, t")
plt.ylabel(r"Total Magnetisation, $M$")
plt.savefig("Mag/Mag.pdf", format="pdf")
plt.show()
```
\newpage

# Appendix D: `TcPlot.py`
```
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

Ncolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
    MArr = np.array([np.abs(totMag).mean(axis=0) / N**2, 
                     (np.abs(totMag)/N**2).std(axis=0)])
    MagAvg = np.array([totMag.mean(axis=0), (totMag**2).mean(axis=0)]).T
    totMag = np.loadtxt('Mag%i.csv' % (N), unpack=True, delimiter=',').T
    CArr = np.loadtxt('C%i' % (N), unpack=True, delimiter=',').T
    chiArr = np.loadtxt('chi%i' % (N), unpack=True, delimiter=',').T/N**2
    chiAbsArr = np.loadtxt('chiAbs%i' % (N), unpack=True, delimiter=',').T/N**2
    CFit0 = [2.3, 0, 0]
    CFit = op.minimize(lambda x: ((fitC(x, kTArrFit)-CArr[0]
                       [np.where((kTArr < 2.5) & (kTArr > 2))])**2).sum(), CFit0)
    print(CFit.x)
    ax2.plot(kTArr, EArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax2.errorbar(kTArr[2*(j)::6], EArr[0][2*(j)::6], EArr[1][2*(j)::6], 
                 ls="None", color=Ncolor[j])
    # ax4.plot(kTArrFit, fitC(CFit.x, kTArrFit))
    ax4.plot(kTArr, CArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax4.errorbar(kTArr[2*(j)::6], CArr[0][2*(j)::6], CArr[1][2*(j)::6], 
                 ls="None", color=Ncolor[j])
    ax3.plot(kTArr, MArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax3.errorbar(kTArr[2*(j)::6], MArr[0][2*(j)::6], MArr[1][2*(j)::6], 
                 ls="None", color=Ncolor[j])
    ax6.plot(kTArr, chiArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax6.errorbar(kTArr[2*(j)::6], chiArr[0][2*(j)::6], chiArr[1][2*(j)::6],
                 ls="None", color=Ncolor[j])
    ax8.plot(kTArr, chiAbsArr[0], label=r'$N=%i$' % N, color=Ncolor[j])
    ax8.errorbar(kTArr[2*(j)::6], chiAbsArr[0][2*(j)::6], chiAbsArr[1][2*(j)::6],
                 ls="None", color=Ncolor[j])


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
    xCF, xCov = op.curve_fit((lambda NArr, a, v: 2/np.log(1+np.sqrt(2)) 
                + a * NArr**(-1/v)), NArr, CMaxArr[0], bounds=([0, 0], [100, 2]))
    x0 = xCF
    print("nu = %f +/- %f" % (x0[1], np.sqrt(np.diag(xCov))[1]))
except (RuntimeError):
    x0 = xOpt.x

TcFit0 = [2, -1]
TcFitx, TcFitCov = op.curve_fit((lambda NArr, Tc_inf, m: Tc_inf + m * (1/NArr)),
                                NArr, CMaxArr[0], TcFit0)
TcFit0 = TcFitx
print(TcFit0)
print(np.sqrt(np.diag(TcFitCov)))

ax5.errorbar(NArr, CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None",
             marker='+')
ax5.plot(NArr, TcFit0[0] + TcFit0[1] * 1/NArr)
ax5.plot(NArr, 2/np.log(1 + np.sqrt(2))*np.ones(NSize))
ax7.errorbar(1/(NArr), CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, 
             ls="None", marker='+')
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
    xCF, xCov = op.curve_fit((lambda NArr, a, v: 2/np.log(1+np.sqrt(2)) 
                + a * NArr**(-1/v)), NArr, CMaxArr[0], bounds=([0, 0], [10, 2]))
    x0 = xCF
    print("nu = %f +/- %f" % (x0[1], np.sqrt(np.diag(xCov))[1]))
except (RuntimeError):
    x0 = xOpt.x

TcFit0 = [2, -1]
TcFitx, TcFitCov = op.curve_fit((lambda NArr, Tc_inf, m: Tc_inf + m * (1/NArr)),
                                NArr, CMaxArr[0], TcFit0)
TcFit0 = TcFitx
print(TcFit0)
print(np.sqrt(np.diag(TcFitCov)))

ax9.errorbar(NArr, CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, ls="None",
             marker='+')
ax9.plot(NArr, TcFit0[0] + TcFit0[1] * 1/NArr)
ax9.plot(NArr, 2/np.log(1 + np.sqrt(2))*np.ones(NSize))
ax10.errorbar(1/(NArr), CMaxArr[0], np.ones(NSize)*(kTArr[1]-kTArr[0])/2, 
              ls="None", marker='+')
ax10.plot(1/(NArr), TcFit0[0] + TcFit0[1]/NArr)

ax9.set_xlabel(r'$N$')
ax9.set_ylabel(r'$T_\mathrm{c}$')
ax10.set_xlabel(r'$1/N$')
ax10.set_ylabel(r'$T_\mathrm{c}$')
fig9.savefig('9.pdf', format="pdf")
fig10.savefig('10.pdf', format="pdf")
plt.show()
```
\newpage

# Appendix E: `hyst.py`
```
"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Plots hysteresis loops for H varying between -1 to 1
plots corresponding energy graphs also
"""
import numpy as np
import matplotlib.pyplot as plt


def energy(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=0) + np.roll(Mag, -1, axis=0))
    totEng = eng.sum(axis=(0, 1)) - H * mu * Mag.sum(axis=(0, 1))
    return(totEng/N**2)


def triangle2(length, amplitude):
    section = length // 4
    x = np.linspace(0, amplitude, section+1)
    mx = -x
    return (np.r_[x, x[-2::-1], mx[1:], mx[-2:0:-1]])


def MCStep(N, H, mu, J, kT, arrSize, Mag0):
    E = np.zeros((arrSize))
    Mag = np.zeros((arrSize, N, N), dtype=np.int8)
    Mag[0] = Mag0
    E[0] = energy(Mag[0], H, mu, J)
    for arr in range(1, arrSize):
        if arr % 100 == 0:
            print(arr)
        """
        Performs each step of the MC technique, 
        each sampling N^2=Ntot points in the lattice
        """
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+
                             Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
            sCount = flipP[nSpins] * np.exp((-2*Mag[arr, i, j]*H*mu)/kT)
            rSamp = np.random.rand()
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
        E[arr] = energy(Mag[arr], H, mu, J)
    return(Mag[nRelax:].sum(axis=(1, 2)).mean(), E[nRelax:].mean(), Mag[-1])


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

H, mu, J = 0, 1, 1
HArr = triangle2(200, 1)
HSize = len(HArr)
N = 30
nRelax = 10
arrSize = 20

kT = 1
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) 
                  else 1 for i in np.arange(5)])

Mag0 = MCStep(N, H, mu, J, kT, arrSize, np.random.choice((-1, 1), (N, N)))
MagArr = np.zeros(HSize*2)
EArr = np.zeros(HSize*2)
MagArr[0] = Mag0[0]
EArr[0] = energy(Mag0[-1], H, mu, J)

for j in range(1, HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j] = Mag0[0]
    EArr[j] = Mag0[1]

for j in range(HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j+HSize] = Mag0[0]
    EArr[j+HSize] = Mag0[1]

ax1.plot(HArr, MagArr[HSize:], label="T=%.1f" % kT)
ax2.plot(HArr, EArr[HSize:], label="T=%.1f" % kT)

kT = 1.5
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) 
                  else 1 for i in np.arange(5)])

Mag0 = MCStep(N, H, mu, J, kT, arrSize, np.random.choice((-1, 1), (N, N)))
MagArr = np.zeros(HSize*2)
EArr = np.zeros(HSize*2)
MagArr[0] = Mag0[0]
EArr[0] = energy(Mag0[-1], H, mu, J)

for j in range(1, HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j] = Mag0[0]
    EArr[j] = Mag0[1]

for j in range(HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j+HSize] = Mag0[0]
    EArr[j+HSize] = Mag0[1]

ax1.plot(HArr, MagArr[HSize:], label="T=%.1f" % kT)
ax2.plot(HArr, EArr[HSize:], label="T=%.1f" % kT)

kT = 3.0
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) 
                  else 1 for i in np.arange(5)])

Mag0 = MCStep(N, H, mu, J, kT, arrSize, np.random.choice((-1, 1), (N, N)))
MagArr = np.zeros(HSize*2)
MagArr[0] = Mag0[0]

for j in range(1, HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j] = Mag0[0]

for j in range(HSize):
    H = HArr[j]
    Mag0 = MCStep(N, H, mu, J, kT, arrSize, Mag0[2])
    MagArr[j+HSize] = Mag0[0]

ax1.plot(HArr, MagArr[HSize:], label="T=%.1f" % kT)
ax1.legend()
ax2.legend()
ax1.set_xlabel(r'External Field, $H$')
ax1.set_ylabel(r'Magnetisation, $M$')
ax2.set_xlabel(r'External Field, $H$')
ax2.set_ylabel(r'Energy, $E/J$')
fig1.savefig('hyst.pdf', format="pdf")
fig2.savefig('hystEng.pdf', format="pdf")
plt.show()
```
Word count: 2849
