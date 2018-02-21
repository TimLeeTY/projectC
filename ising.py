"""
Project C: The Ising Model of a Ferromagnet
Finds properties of a Ferromagnet using the Ising model, sampling spins at random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pdb


def movingAvg(arr, n):
    csum = arr.cumsum()
    csum[n:] = csum[n:]-csum[:-n]
    return(csum[n-1:]/n)


def DelEnergy(M, H, mu, J, i, j, kT):
    s = M[i, j]
    coup = 2*s*(M[i+1, j]+M[i-1, j]+M[i, j-1]+M[i, j+1])  # spin coupling term
    ext = 2*mu*H*s  # field energy
    # print(ext+coup)
    return(np.random.rand(1) < np.exp(-1*max([ext+coup, 0])/(kT)))


def MCstep(N, M):
    """performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    """on average covers ~63 lattice sites (see readme for calculation)"""
    Ntot = N**2
    for k in range(Ntot):
        [i, j] = np.random.choice(np.arange(N)-1, [2, 1])
        # i=np.random.choice(range(1,N-1))
        if DelEnergy(M, H, mu, J, i, j, kT):
            M[i, j] *= -1
    return(M)


def meanEnergy(Mag, H, mu, J):
    eng = -1.*Mag*mu*H-J/2*Mag*(np.roll(Mag, 1, axis=1)+np.roll(
        Mag, -1, axis=1)+np.roll(Mag, 1, axis=2)+np.roll(Mag, -1, axis=2))
    return(eng.sum(axis=(1, 2)).mean())


def makeM(N, p):
    if p == 1:
        return(np.ones((N, N)))
    else:
        M = np.concatenate(
            (np.ones(int(N**2*p)), -1*np.ones(N**2-int(N**2*p))))
        M = (np.random.permutation(M)).reshape((N, N))
        return(M)


def autoCorr(inMag, tauArr):
    MMag = inMag-inMag.mean()
    if type(tauArr) == int:
        tau = tauArr
        return(np.mean(MMag[tau:]*MMag[:-tau])/np.mean(MMag**2))
    else:
        return(np.array([np.mean(MMag[tau:]*MMag[:-tau])/np.mean(MMag**2)for tau in tauArr]))


autoCorrV = np.vectorize(autoCorr)


def findTauc(inMag, initTau):
    return(op.brentq((lambda tau: autoCorr(inMag, int(tau)) - np.exp(-1.)), 1, int(initTau)))


N = 10
Tsize = 10
H, mu, J = 0, 1, 1
kTarr = np.linspace(2, 4, Tsize)*J
eArr = np.zeros(Tsize)
MArr = np.zeros(Tsize)
arrSize = 300
tauC = np.zeros((5, Tsize))
fig1, ax1 = plt.subplots()
plotTau = np.arange(1, 10)

for j in range(Tsize):
    kT = kTarr[j]
    for i in range(5):
        Mag = np.zeros((arrSize, N, N))
        Mag[0] = (makeM(N, 1))
        for k in range(1, arrSize):
            Mag[k] = MCstep(N, Mag[i-1])
        inMag = Mag[50:].sum(axis=(1, 2))
        #ax2.plot(plotTau, np.log(SArr.mean))
        for tau in range(1, 41):
            if autoCorr(inMag, tau) < np.exp(-1):
                tauC[i, j] = tau-0.5
                break
fig2, ax2 = plt.subplots()
ax2.errorbar(kTarr, tauC.mean(axis=0), tauC.std(axis=0))

for j in range(Tsize):
    kT = kTarr[j]
    M = makeM(N, 1)
    Mag = np.zeros((arrSize, N, N))
    for i in range(arrSize):
        Mag[i] = M
        M = MCstep(N, M)
    if j % int(Tsize/5) == 0:
        Mplot = movingAvg(Mag.sum(axis=(1, 2)), 10)
        ax1.plot(np.sign(Mplot[0])*Mplot)
    eArr[j] = meanEnergy(Mag, H, mu, J)
    MArr[j] = np.abs(Mag.sum(axis=(1, 2)).mean())
    #print(kT, autoCorr(Mag[50:].sum(axis=(1,2)),(arrSize-51))-np.exp(-1.))


fig, ax = plt.subplots()
ax.plot(kTarr, eArr, label='mean energy')
fig, ax = plt.subplots()
ax.plot(kTarr, MArr, label='mean magnetisation')
# fig,ax=plt.subplots()
#ax.plot(kTarr,tauC,label='mean magnetisation')
plt.show()
