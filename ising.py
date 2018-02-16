"""
Project C: The Ising Model of a Ferromagnet
Finds propeties of a Ferromagnet usin the Ising model, sampling spins at random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np
import matplotlib.pyplot as plt

def DelEnergy(M,H,mu,J,i,j,kT):
	s=M[i,j]
	coup=2*s*(M[i+1,j]+M[i-1,j]+M[i,j-1]+M[i,j+1]) #spin coupling term
	ext=2*mu*H*s #field energy
        #print(ext+coup)
	return(np.random.rand(1)<np.exp(-1*max([ext+coup,0])/(kT)))		
N=100
M=np.random.choice([-1,1],[N,N])

H,mu,J,kT=1,1,1,2**(-5)
Ntot=2**15
totspin=np.zeros(Ntot)
for k in range(Ntot):
    j=np.random.choice(range(1,N-1))
    i=np.random.choice(range(1,N-1))
    if DelEnergy(M,H,mu,J,i,j,kT):
        M[i,j]*=-1
    totspin[k]=(M.sum())

plt.figure()
plt.plot(totspin)
plt.show()
