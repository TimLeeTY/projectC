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
import time


def movingAvg(arr, n):
    """Calculate moving average of values in arr over length n"""
    csum = arr.cumsum()
    csum[n:] = csum[n:] - csum[:-n]
    return(csum[n - 1:] / n)


def delEnergy(M, H, mu, J, i, j, kT):
    """Calculate the change in energy change if the spin at i,j is flipped"""
    s = M[i, j]
    coup = 2 * s * (M[i + 1, j] + M[i - 1, j] + M[i, j - 1] +
                    M[i, j + 1])  # spin coupling term
    ext = 2 * mu * H * s  # field energy
    """Flip the spin if change in  E<0 or probabilistically based on Boltzmann distribution"""
    return((ext + coup < 0) or (np.random.rand(1) < np.exp(-1 * (ext + coup) / (kT))))


def MCStep(N, M, mu, J, kT):
    """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    """On average covers ~63 lattice sites (see readme for calculation)"""
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    for [i, j] in samples:
        # Calculate the change in energy change if the spin at i,j is flipped
        s = M[i, j]
        coup = 2 * s * (M[i + 1, j] + M[i - 1, j] + M[i, j - 1] + M[i, j + 1])  # spin coupling term
        ext = 2 * mu * H * s  # field energy
        if ((ext + coup < 0) or (np.random.rand(1) < np.exp(-1 * (ext + coup) / (kT)))):
            # Flip the spin if change in  E<0 or probabilistically based on Boltzmann distribution
            M[i, j] *= -1
    return(M)


def MCStepFast(n, M, mu, J, KT):
    """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    if H == 0:
        flipP = np.ones(5)
        for i in range(3,5):
            flipP[i] = np.exp(-2 * (2 * i - 4) / kT)   # cache the probability of flipping for each spin configuration
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    for [i, j] in samples:
        if np.random.rand() < flipP[int((M[i, j] * (M[i + 1, j] + M[i - 1, j] + M[i, j - 1] + M[i, j + 1]))/2+2)]:
            M[i, j] *= -1
    return(M)


def meanEnergy(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=2) + np.roll(Mag, -1, axis=2))
    totEng = eng.sum(axis=(1, 2))
    return([totEng.mean(), totEng.std()])


def makeM(N, p):
    """Initialises the spins in the system"""
    if p == 1:
        return(np.ones((N, N)))
    else:
        M = np.concatenate((np.ones(int(N**2 * p)), -1 * np.ones(N**2 - int(N**2 * p))))
        M = (np.random.permutation(M)).reshape((N, N))
        return(M)


def autoCorr(inMag, tauArr):
    """Returns the correlation of the vector inMag for each value of tau in tauArr"""
    MMag = inMag - inMag.mean()
    if isinstance(tauArr, int):
        tau = tauArr
        return(np.mean(MMag[tau:] * MMag[:-tau]) / np.mean(MMag**2))
    else:
        return(np.array([np.mean(MMag[tau:] * MMag[:-tau]) / np.mean(MMag**2)
                         for tau in tauArr]))


def findTauc(inMag, initTau):
    """Find critical value of tau using root finding method"""
    return(op.brentq((lambda tau: autoCorr(inMag, int(tau)) - np.exp(-1.)), 1, int(initTau)))


autoCorrV = np.vectorize(autoCorr, excluded=['inMag'])

NArr = np.arange(8, 15)                 # Array that holds the values of N to be used
NSize = len(NArr)
TSize = 20                              # Number of samples of temperature to be used
H, mu, J = 0, 1, 1
kTarr = np.linspace(1.5, 3, TSize) * J  # kT scaled relative to J
nSamp = 1                               # Number of samples to run
arrSize = 1000                           # Number of steps to take for each M-C simulation
eArr = np.zeros((nSamp, TSize, 2))
MArr = np.zeros((nSamp, TSize))
CMaxArr = np.zeros(NSize)
tauC = np.zeros((nSamp, TSize))
# plotTau = np.arange(1, 10)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

for l in range(NSize):
    N = NArr[l]
    for j in range(TSize):
        kT = kTarr[j]
        for i in range(nSamp):
            tnsamp0 = time.clock()
            Mag = np.zeros((arrSize, N, N))
            Mag[0] = (makeM(N, 1))
            t0 = time.clock()
            for k in range(1, arrSize):
                Mag[k] = MCStepFast(N, Mag[k - 1], mu, J, kT)
            tf = time.clock()
            inMag = Mag[50:].sum(axis=(1, 2))
            """
            # print autocorrelaion as a funcion of tau
            if np.abs(kT - 2.6) < 0.1:
                ax3.plot(np.log(np.abs(
                    autoCorr(inMag=inMag, tauArr=np.arange(1, 41)))))
            """
            eArr[i, j] = (meanEnergy(Mag[50:], H, mu, J))
            MArr[i, j] = np.abs(inMag.mean())
            for tau in range(1, 41):
                if np.abs(autoCorr(inMag, tau)) < np.exp(-1):
                    tauC[i, j] = tau - 0.5
                    break
            tnsampf = time.clock()
            print(tnsampf - tnsamp0, tf - t0)
    C = (eArr[:, :, 1])**2/(kT**2)
    if N % 2 == 0:
        ax1.errorbar(kTarr, tauC.mean(axis=0), tauC.std(axis=0), label=r'$N=%i$' % N)
        ax2.errorbar(kTarr, eArr[:, :, 0].mean(axis=0), eArr[:, :, 0].std(axis=0), label=r'$N=%i$' % N)
        ax3.errorbar(kTarr, MArr.mean(axis=0), MArr.std(axis=0), label=r'$N=%i$' % N)
        ax4.errorbar(kTarr, C.mean(axis=0), C.std(axis=0), label=r'$N=%i$' % N)
    CMaxArr[l] = kTarr[np.argmax(C.mean(axis=0))]

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

fig5, ax5 = plt.subplots()
ax5.errorbar(NArr, CMaxArr, kTarr[1]-kTarr[0])
plt.show()
