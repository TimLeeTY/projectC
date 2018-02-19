"""
Project C: The Ising Model of a Ferromagnet
Finds properties of a Ferromagnet using the Ising model, sampling spins at random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np
import matplotlib.pyplot as plt

def movingAvg(arr,n):
    csum=arr.cumsum()
    csum[n:]=csum[n:]-csum[:-n]
    return(csum[n-1:]/n)

def DelEnergy(M,H,mu,J,i,j,kT):
	s=M[i,j]
	coup=2*s*(M[i+1,j]+M[i-1,j]+M[i,j-1]+M[i,j+1]) #spin coupling term
	ext=2*mu*H*s #field energy
        #print(ext+coup)
	return(np.random.rand(1)<np.exp(-1*max([ext+coup,0])/(kT)))	

def MCstep(N,M):
    """performs each step of the MC technique, each sampling N^2=Ntot points in the lattice, on average covering ~63 lattice sites (see readme for calculation)"""
    Ntot=N**2
    for k in range(Ntot):
        [i,j]=np.random.choice(np.arange(N)-1,[2,1])
        #i=np.random.choice(range(1,N-1))
        if DelEnergy(M,H,mu,J,i,j,kT):
            M[i,j]*=-1
    return(M)

N=10
H,mu,J=0,1,1
kTarr=2**np.linspace(1,2,8)*J
arrSize=200
plt.figure()
for kT in kTarr:
    M=np.concatenate((np.ones(int(N**2/2)),-1*np.ones(int(N**2/2))))
    M=(np.random.permutation(M)).reshape((N,N))
    Mag=np.zeros(arrSize)
    for i in range(arrSize):
        Mag[i]=M.sum()
        M=MCstep(N,M)
    #plt.plot(Mag)
    plt.plot(np.abs(movingAvg(Mag,20)))
plt.show()
