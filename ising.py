"""
Project C: The Ising Model of a Ferromagnet
Finds propeties of a Ferromagnet usin the Ising model, sampling spins at random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np

def DelEnergy(M,H,mu,J,i,j):
	sisj=M[i,j]*(M[i+1,j]+M[i-1,j]+M[i,j-1]+M[i,j+1])
	return(2*J*sisj+mu*H*M[i,j])
def flip(M,N):
	sisj=M[i,j]*(M[i+1,j]+M[i-1,j]+M[i,j-1]+M[i,j+1])
	if 2*J*sisj+mu*H*M[i,j]<0:

N=100
M=np.random.choice([-1,1],[N,N])
print(M)

H,mu,J=1,1,1
print(DelEnergy(M, H, mu, J, 1,1))
