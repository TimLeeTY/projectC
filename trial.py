import numpy as np
import random
import matplotlib.pyplot as plt
import time


def init_spin_array(rows, cols):
    return np.ones((rows, cols))


def find_neighbors(spin_array, lattice, x, y):
    left = (x, y - 1)
    right = (x, (y + 1) % lattice)
    top = (x - 1, y)
    bottom = ((x + 1) % lattice, y)

    return [spin_array[left[0], left[1]],
            spin_array[right[0], right[1]],
            spin_array[top[0], top[1]],
            spin_array[bottom[0], bottom[1]]]


def energy(spin_array, lattice, x, y):
    return 2 * spin_array[x, y] * sum(find_neighbors(spin_array, lattice, x, y))


def main():
    RELAX_SWEEPS = 50
    lattice = 100
    sweeps = 100
    tempArr = np.arange(0.1, 5.0, 0.1)
    M = np.zeros(len(tempArr))
    for k in range(len(tempArr)):
        temperature = tempArr[k]
        spin_array = init_spin_array(lattice, lattice)
        # the Monte Carlo follows below
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        t0 = time.clock()
        for sweep in range(sweeps + RELAX_SWEEPS):
            for i in range(lattice):
                for j in range(lattice):
                    e = energy(spin_array, lattice, i, j)
                    if e <= 0:
                        spin_array[i, j] *= -1
                    elif np.exp((-1.0 * e)/temperature) > random.random():
                        spin_array[i, j] *= -1
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)
        tf = time.clock()
        print(tf - t0)
        M[k] = sum(mag[RELAX_SWEEPS:]) / sweeps
    plt.plot(tempArr, M)
    plt.show()


main()
