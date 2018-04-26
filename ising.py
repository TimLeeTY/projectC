"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Finds properties of a Ferromagnet using the Ising model, sampling spins at
random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def movingAvg(arr, n):
    """Calculate moving average of values in arr over length n"""
    csum = arr.cumsum()
    csum[n:] = csum[n:] - csum[:-n]
    return(csum[n - 1:] / n)


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


def autoCorr(totMag, tau):
    """Returns the correlation of the vector totMag a value of tau"""
    MMag = totMag - totMag.mean(axis=0)
    return(np.mean(MMag[tau:] * MMag[:-tau]) / np.mean(MMag**2))


def fitTc(Tc, x, NArr):
    [Tc_inf, a, v] = x
    return(((Tc - Tc_inf - a * NArr**(-1/v))**2).sum())


def fitC(CFit, TArr):
    [Tc, a, b] = CFit
    Tlow = TArr[np.where(TArr < Tc)]
    Thigh = TArr[np.where(TArr >= Tc)]
    out = np.concatenate((a*(-np.log((Tc-Tlow)/Tc)), b*(-np.log((Thigh-Tc)/Tc))))
    return(out)


def bootstrap(EArr, tauC, kT):
    nSamp = 10
    Cout = np.zeros((tauC, 2))
    for j in range(tauC):
        EIndep = EArr[j::tauC]
        n = len(EIndep)
        C = np.zeros(n)
        samples = np.random.choice(n, (n, nSamp))
        for i in samples:
            C[i] = np.divide(EIndep[samples[i]].var(), kT**2)/N**2
        Cout[j] = [C.mean(axis=0), C.std(axis=0)]
    return(Cout.mean(axis=0))


def energy(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=2) + np.roll(Mag, -1, axis=2))
    totEng = eng.sum(axis=(1, 2))
    return(totEng)


def energyAlt(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=0) + np.roll(Mag, -1, axis=0))
    totEng = eng.sum(axis=(0, 1))
    return(totEng)


def MCStepFastAlt(N, H, mu, J, kTArr, arrSize, TSize):
    E = np.zeros((arrSize, TSize))
    Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
    Mag[0] = (makeM(N, 1, TSize))
    E[0] = energyAlt(Mag[0], H, mu, J)
    for arr in range(1, arrSize):
        if arr % 100 == 0:
            print(arr)
        """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
            sCount = flipP[(nSpins, np.arange(TSize))]
            rSamp = np.random.rand(TSize)
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
        E[arr] = energyAlt(Mag[arr], H, mu, J)
    if N == 25:
        plt.plot(Mag[:, :, :, 50].sum(axis=(1, 2)))
    return(Mag[nRelax:].sum(axis=(1, 2)), E[nRelax:])


def MCStepFast(N, H, mu, J, kTArr, arrSize, TSize):
    Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
    Mag[0] = (makeM(N, 1, TSize))
    for arr in range(1, arrSize):
        if arr % 100 == 0:
            print(arr)
        """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
            sCount = flipP[(nSpins, np.arange(TSize))]
            rSamp = np.random.rand(TSize)
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
    EArr = energy(Mag, H, mu, J)
    return(Mag.sum(axis=(1, 2))[nRelax:], EArr[nRelax:])


TSize = 100                             # Number of samples of temperature to be used
H, mu, J = 0, 1, 1
kTArr = np.linspace(1.6, 3, TSize) * J    # kT scaled relative to J
kTArrFit = kTArr[np.where((kTArr < 2.5) & (kTArr > 2))]    # kT scaled relative to J
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()

NArr = np.arange(10, 60, 10)              # Array that holds the values of N to be use
NSize = len(NArr)
CMaxArr = np.zeros(NSize)

if H == 0:
    flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])

for j in range(NSize):
    N = NArr[j]
    tauC = np.zeros((2, TSize))
    CArr = np.zeros((2, TSize))
    chiArr = np.zeros((2, TSize))
    print('N= %i' % (N))
    nRelax = 5000
    arrSize = 5000*10
    [totMag, E] = MCStepFastAlt(N, H, mu, J, kTArr, arrSize, TSize)
    print(totMag.shape)
    EArr = np.array([E.mean(axis=0), E.std(axis=0)])
    MArr = np.array([np.abs(totMag).mean(axis=0) / N**2, np.abs(totMag).std(axis=0) / N**2])
    chiArr[0] = np.divide((totMag.var(axis=0)), kTArr)
    tauArr = range(1, arrSize-nRelax-1)
    for i in range(TSize):
        for tau in tauArr:
            if np.abs(autoCorr(totMag[:, i], tau)) < np.exp(-1):
                tauC[0, i] = tau - 0.5
                CArr[:, i] = bootstrap(E[:, i], tau, kTArr[i])
                print('N=%f, T=%f, tau=%i' % (N, kTArr[i], tau))
                break
    CMaxArr[j] = kTArr[np.argmax(movingAvg(CArr, 3))+1]
    CFit0 = [2.3, 0, 0]
    CFit = op.minimize(lambda x: ((fitC(x, kTArrFit)-CArr[0][np.where((kTArr < 2.5) & (kTArr > 2))])**2).sum(), CFit0)
    print(CFit.x)
    ax1.errorbar(kTArr, tauC[0], tauC[1], label=r'$N=%i$' % N)
    ax2.errorbar(kTArr, EArr[0], EArr[1], label=r'$N=%i$' % N)
    ax4.errorbar(kTArr, CArr[0], CArr[1], label=r'$N=%i$' % N)
    ax4.plot(kTArrFit, fitC(CFit.x, kTArrFit))
    ax3.errorbar(kTArr[::4], MArr[0, ::4], MArr[1, ::4], label=r'$N=%i$' % N)
    ax6.errorbar(kTArr, chiArr[0], chiArr[1], label=r'$N=%i$' % N)

x0 = [2, 4, 0.5]
xOpt = op.minimize(lambda x: fitTc(CMaxArr, x, NArr), x0)
print(xOpt.x)
try:
    xCF, xCov = op.curve_fit((lambda NArr, Tc_inf, a, v: Tc_inf + a * NArr**(-1/v)), NArr, CMaxArr, x0)
    x0 = xCF
except RuntimeError:
    x0 = xOpt.x

print(x0)

ax5.plot(NArr, x0[0] + x0[1] * NArr**(-1/x0[2]))
ax5.plot(NArr, 2/np.log(1 + np.sqrt(2))*np.ones(NSize))
ax5.errorbar(NArr, CMaxArr)

ax7.plot(1/(NArr), CMaxArr)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()
