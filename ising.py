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
    Performs each arrSize steps of the Metropolis algorithm, each sampling N^2=Ntot points in the
    lattice, all temperatures are evolved simultaneously
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
            nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
            """map flipP onto sCount based on nSpins for all temperatures"""
            sCount = flipP[(nSpins, np.arange(TSize))]
            rSamp = np.random.rand(TSize)
            """compare random samples to the flipP and flip if sCount > rSamp"""
            Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1
        E[arr] = energy(Mag[arr], H, mu, J)
    return(Mag[nRelax:].sum(axis=(1, 2)), E[nRelax:])


TSize = 100                             # Number of samples of temperature to be used
H, mu, J = 0, 1, 1
kTArr = np.linspace(1.6, 3, TSize) * J    # kT scaled relative to J
NArr = np.arange(10, 60, 5)              # Array that holds the values of N to be use
NSize = len(NArr)

if H == 0:
    flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])

nRelax = 5000
arrSize = 50000

if __name__ == '__main__':
    for j in range(NSize):
        N = NArr[j]
        print('N= %i' % (N))
        [totMag, E] = MCStep(N, H, mu, J, kTArr, arrSize, TSize)
        np.savetxt('Mag%i.csv' % (N), totMag, delimiter=',')
        np.savetxt('Eng%i.csv' % (N), E, delimiter=',')
