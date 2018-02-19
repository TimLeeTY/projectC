"""
Project C: The Ising Model of a Ferromagnet
Finds properties of a Ferromagnet using the Ising model, sampling spins at random and calculating the energy required to flip each spin

N x N lattice
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    """performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    """on average covers ~63 lattice sites (see readme for calculation)"""
    Ntot=N**2
    for k in range(Ntot):
        [i,j]=np.random.choice(np.arange(N)-1,[2,1])
        #i=np.random.choice(range(1,N-1))
        if DelEnergy(M,H,mu,J,i,j,kT):
            M[i,j]*=-1
    return(M)

def meanEnergy(Mag,H,mu,J):
    eng=-1.*Mag*mu*H-J/2*Mag*(np.roll(Mag,1, axis=1)+np.roll(Mag, -1,axis=1)+np.roll(Mag,1, axis=2)+np.roll(Mag,-1,axis=2))
    return(eng.sum(axis=(1,2)).mean())

def meanMag(Mag):
    return(np.abs(Mag.sum(axis=(1,2)).mean()))

def makeM(N,p):
    if p==1:
        return(np.ones((N,N)))
    else:
        M=np.concatenate((np.ones(int(N**2*p)),-1*np.ones(N**2-int(N**2*p))))
        M=(np.random.permutation(M)).reshape((N,N))
        return(M)

def autoCorr(MMag,tau):
    return(np.mean(MMag[tau:]*MMag[:-tau])/np.mean(MMag**2))

def findTauc(MMag,arrSize):
    return(scipy.optimize.brentq(lambda x: ( lambda MMag,tau: autoCorr(MMag,tau)-np.exp(-1)  )(MMag, x),0,arrSize))

N=16
Tsize=10
H,mu,J=0,1,1
kTarr=np.linspace(1,6,Tsize)*J
eArr=np.zeros(Tsize)
MArr=np.zeros(Tsize)
arrSize=200
fig1,ax1=plt.subplots()
fig2,ax2=plt.subplots()
for j in range(Tsize):
    kT=kTarr[j]
    M=makeM(N,1)
    Mag=np.zeros((arrSize,N,N))
    for i in range(arrSize):
        Mag[i]=M
        M=MCstep(N,M)
    #plt.plot(Mag)
    if j%2==0: 
        Mplot=movingAvg(Mag.sum(axis=(1,2)),10)
        ax1.plot(np.sign(Mplot[0])*Mplot)
    eArr[j]=meanEnergy(Mag,H,mu,J)
    MArr[j]=meanMag(Mag)
    tauArr=np.arange(19,100,10)
fig,ax=plt.subplots()
ax.plot(kTarr,eArr,label='mean energy')
fig,ax=plt.subplots()
ax.plot(kTarr,MArr,label='mean magnetisation')
plt.show()
