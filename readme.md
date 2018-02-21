# Computing Project C: Ising Model of Ferromagnets

This project is an investigation into the use of Monte-Carlo methods to tackle the Ising model of ferromagnetism which considers the interaction between spins of nearest neighbours. For simplicity, the 2-dimensional case is chosen meaning we only consider energy contributions from the four nearest neighbours of each spin. The model gives a way to evaluate the total energy of each microstate from which we can derive the probability of a certain change occurring(specifically the probability of a spin flipping). Hence, by generating random samples of the system at certain time steps, we can model the time evolution of the system given the temperature $T$.

## Background Theory

The energy of a set of spins in a regular lattice in the presence of some magnetic field can be given by:

$$ E=-J\sum_{<ij>} s_i s_j + H\mu \sum_{i=1}^{N^2}$$
