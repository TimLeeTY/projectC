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
        """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
        samples = np.random.choice(np.arange(N)-1, (N**2, 2))
        Mag[arr] = Mag[arr - 1]
        for [i, j] in samples:
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
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
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for i in np.arange(5)])

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
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for i in np.arange(5)])

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
flipP = np.array([np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for i in np.arange(5)])

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
