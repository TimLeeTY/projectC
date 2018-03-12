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
import multiprocessing as mp


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


def MCStepFast(N, H, mu, J, kTArr, arrSize, TSize):
    Mag = np.zeros((arrSize, N, N, TSize))
    Mag[0] = (makeM(N, 1, TSize))
    for arr in range(1, arrSize):
        """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
            sCount = flipP[(nSpins, np.arange(TSize))]
            rSamp = np.random.rand(TSize)
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
    return(Mag)


def energy(Mag, H, mu, J):
    """Returns the mean energy of a set of spins Mag"""
    eng = -1. * Mag * mu * H - J / 2 * Mag * (np.roll(Mag, 1, axis=1) + np.roll(
        Mag, -1, axis=1) + np.roll(Mag, 1, axis=2) + np.roll(Mag, -1, axis=2))
    totEng = eng.sum(axis=(1, 2))
    return(totEng)



def makeM(N, p, TSize):
    """Initialises the spins in the system"""
    ret = np.empty((N, N, TSize))
    if p == 1:
        return(np.ones((N, N, TSize)))
    else:
        M = np.concatenate((np.ones(int(N**2 * p)), -1 * np.ones(N**2 - int(N**2 * p))))
        for i in range(TSize):
            ret[:, :, i] = (np.random.permutation(M)).reshape((N, N))
        return(ret)


def autoCorr(inMag, tau):
    """Returns the correlation of the vector inMag for each value of tau in tauArr"""
    MMag = inMag - inMag.mean(axis=0)
    return(np.mean(MMag[tau:] * MMag[:-tau]) / np.mean(MMag**2))


def fitTc(Tc, x, NArr):
    [Tc_inf, a, v] = x
    return(((Tc - Tc_inf - a * NArr**(-1/v))**2).sum())


def bootstrap(Mag, tauC, kT):
    nSamp = 10
    MagIndep = Mag[::tauC]
    n = len(MagIndep)
    samples = np.random.choice(n, (n, nSamp))
    C = np.zeros(n)
    for i in samples:
        C[i] = np.divide(energy(MagIndep[samples[i]], H, mu, J).var(), kT**2)
    return([C.mean(axis=0), C.std(axis=0)])


def mainRun(N):
    tauC = np.zeros((2, TSize))
    CArr = np.zeros((2, TSize))
    chiArr = np.zeros((2, TSize))
    print('N= %i' % (N))
    nRelax = 20 * N
    arrSize = 100 * N
    Mag = MCStepFast(N, H, mu, J, kTArr, arrSize, TSize)[nRelax:]
    print(Mag.shape)
    inMag = Mag.sum(axis=(1, 2))
    EArr = np.array([energy(Mag, H, mu, J).mean(axis=0), energy(Mag, H, mu, J).std(axis=0)])
    MArr = np.array([np.abs(inMag).mean(axis=0) / N**2, np.abs(inMag).std(axis=0) / N**2])
    chiArr[0] = np.divide((inMag.var(axis=0)), kTArr)
    tauArr = range(1, arrSize-nRelax-1)
    for i in range(TSize):
        for tau in tauArr:
            if np.abs(autoCorr(inMag[:, i], tau)) < np.exp(-1):
                tauC[0, i] = tau - 0.5
                CArr[:, i] = bootstrap(Mag[:, :, :, i], tau, kTArr[i])
                break
    return(np.array([EArr, tauC, MArr, chiArr, CArr]))


nSamp = 1                                # Number of samples to run
TSize = 100                             # Number of samples of temperature to be used
H, mu, J = 0, 1, 1
kTArr = np.linspace(1.8, 3, TSize) * J    # kT scaled relative to J
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()

NArr = np.arange(10, 30, 4)              # Array that holds the values of N to be use
NSize = len(NArr)
eArr = np.zeros((nSamp, TSize))
sigEArr = np.zeros((nSamp, TSize))
CMaxArr = np.zeros((NSize))

flipP = np.ones((5, TSize))
if H == 0:
    flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])

p = mp.Pool(NSize)
out = np.array(p.map(mainRun, NArr))
out = np.transpose(out, (1, 0, 2, 3))
print(out.shape)
[EArr, tauCArr, MArr, chiArr, CArr] = out
for l in range(NSize):
    N = NArr[l]
    E = EArr[l]
    C = CArr[l]
    print(C.shape)
    M = MArr[l]
    Chi = chiArr[l]
    tauC = tauCArr[l]
    CMaxArr[l] = kTArr[np.argmax(C)]+(kTArr[0]-kTArr[1])/2
    if l % 1 == 0:
        ax1.errorbar(kTArr, tauC[0], tauC[1], label=r'$N=%i$' % N)
        ax2.errorbar(kTArr, E[0], E[1], label=r'$N=%i$' % N)
        ax4.errorbar(kTArr, C[0], C[1], label=r'$N=%i$' % N)
        ax3.errorbar(kTArr[::4], M[0, ::4], M[1, ::4], label=r'$N=%i$' % N)
        ax6.errorbar(kTArr, Chi[0], Chi[1], label=r'$N=%i$' % N)
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

ax7.plot(1/(NArr**2), CMaxArr)
"""
nSamp = 5
MArr = np.zeros((nSamp, TSize))
eArr = np.zeros((nSamp, TSize))
N = 10
for j in range(TSize):
    kT = kTArr[j]
    print('N= %i, kT= %f' % (N, kT))
    for i in range(nSamp):
        Mag = MCStepFast(N, H, mu, J, kT, arrSize)
        inMag = Mag[nRelax:].sum(axis=(1, 2))
        eArr[i, j] = (meanEnergy(Mag[nRelax:], H, mu, J))[0]
        MArr[i, j] = (inMag.mean())
print('N= {:d}, T_c= {:.2f}±{:.2f}'.format(N, CMaxArr.mean(), CMaxArr.std()))
ax3.errorbar(kTArr, MArr.mean(axis=0), MArr.std(axis=0), label=r'$N=%i$' % N)
ax2.errorbar(kTArr, eArr.mean(axis=0), eArr.std(axis=0), label=r'$N=%i$' % N)

for l in range(NSize):
    N = NArr[l]
    for j in range(TSize):
        kT = kTArr[j]
        print('N= %i, kT= %f' % (N, kT))
        for i in range(nSamp):
            #t0 = time.clock()
            Mag = MCStepFast(N, H, mu, J, kT, arrSize)
            #tf = time.clock()
            inMag = Mag[nRelax:].sum(axis=(1, 2))
            sigEArr[i, j] = (meanEnergy(Mag[nRelax:], H, mu, J))[1]/kT    # Standard deviation in total energy
            for tau in range(1, arrSize-nRelax-1):
                if np.abs(autoCorr(inMag, tau)) < np.exp(-1):
                    tauC[i, j] = tau - 0.5
                    break
            #tf2 = time.clock()
            #print(tf - t0, tf2 - t0)
    C = (sigEArr**2)                                           # Heat capacity from
    if N % 2 == 0:
        ax1.errorbar(kTArr, tauC.mean(axis=0), tauC.std(axis=0), label=r'$N=%i$' % N)
        ax4.errorbar(kTArr, C.mean(axis=0), C.std(axis=0), label=r'$N=%i$' % N)
    for i in range(nSamp):
        CMaxArr[i, l] = kTArr[np.argmax(C[i])]
"""

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()
