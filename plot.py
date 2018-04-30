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

kTArrFit = kTArr[np.where((kTArr < 2.5) & (kTArr > 2))]    # kT scaled relative to J
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
        MagAvg = np.array([totMag.mean(axis=0), ((totMag-totMag.mean(axis=0))**2).mean(axis=0)]).T
        for i in range(TSize):
            for tau in tauArr:
                """loop through tau to find when A(t) < exp(-1)"""
                if np.abs(autoCorr(totMag[:, i], MagAvg[i], tau)) < np.exp(-1):
                    tauC[0, i] = tau - 0.5
                    [CArr[:, i], chiArr[:, i], chiAbsArr[:, i]] = bootstrap(E[:, i], totMag[:, i], tau, kTArr[i])
                    print('N=%f, i=%i, tau=%i' % (N, i, tau))
                    break
        """taking a moving average before averaging to avoid anomalous points"""
        CMaxArr[0, j] = kTArr[np.argmax(movingAvg(CArr, 3))+1]
        """fit arbitrary function to C as a rough guideline for T_c"""
        CFit0 = [2.3, 0, 0]
        CFit = op.minimize(
            lambda x: ((fitC(x, kTArrFit)-CArr[0][np.where((kTArr < 2.5) & (kTArr > 2))])**2).sum(), CFit0
        )
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
