"""
Project C: The Ising Model of a Ferromagnet
-------------------------------------------
Plots spin configurations as 2D images
Still and animated versions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

kTArr = np.array([4])
TSize = len(kTArr)
H, mu, J = 0, 1, 1
N = 50
flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])
arrSize = 22

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.ones((N, N, TSize))
for arr in range(1, arrSize):
    if arr in [2, 21]:
        for j in range(TSize):
            kT = kTArr[j]
            plt.figure()
            img = plt.imshow(Mag[arr-1][:, :, j], cmap='gray')
            plt.savefig("spin/T%.0f-arr%iones.pdf" % (kT*10, arr), format="pdf")
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1

kTArr = np.array([1, 2.7, 4])
TSize = 3
H, mu, J = 0, 1, 1
N = 50
flipP = np.array([[np.exp(-2 * (2 * (i - 2)) / kT) if (i > 2) else 1 for kT in kTArr] for i in np.arange(5)])

arrSize = 22

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.random.choice((-1, 1), (N, N, TSize))
for arr in range(1, arrSize):
    if arr in [3, 21]:
        for j in range(TSize):
            kT = kTArr[j]
            plt.figure()
            img = plt.imshow(Mag[arr-1][:, :, j], cmap='gray')
            plt.savefig("spin/T%.0f-arr%i.pdf" % (kT*10, arr), format="pdf")
    """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1


def animate(i):
    im.set_array(Mag[i, :, :, j])
    return im,


for j in range(TSize):
    fig = plt.figure()
    ims = []
    for i in range(arrSize):
        im = plt.imshow(Mag[i, :, :, j], cmap='gray',  animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, repeat_delay=1000)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='tyl35'), bitrate=1800)
    ani.save('T=%i.mp4' % j, writer=writer)

arrSize = 400
N = 50

Mag = np.zeros((arrSize, N, N, TSize), dtype=np.int8)
Mag[0] = np.random.choice((-1, 1), (N, N, TSize))
for arr in range(1, arrSize):
    """Performs each step of the MC technique, each sampling N^2=Ntot points in the lattice"""
    samples = np.random.choice(np.arange(N)-1, (N**2, 2))
    Mag[arr] = Mag[arr - 1]
    for [i, j] in samples:
        nSpins = np.int_((Mag[arr, i, j]*(Mag[arr, i+1, j]+Mag[arr, i-1, j]+Mag[arr, i, j-1]+Mag[arr, i, j+1]))/2+2)
        sCount = flipP[(nSpins, np.arange(TSize))]
        rSamp = np.random.rand(TSize)
        Mag[arr, i, j] *= np.argmax(np.array([sCount, rSamp]), axis=0) * 2 - 1

plt.figure()
for j in range(TSize):
    kT = kTArr[j]
    plt.plot((Mag.sum(axis=(1, 2))[:, j]), label="T=%.1f" % kT)
plt.legend()
plt.xlabel(r"MC Steps, t")
plt.ylabel(r"Total Magnetisation, $M$")
plt.savefig("Mag/Mag.pdf", format="pdf")
plt.show()
