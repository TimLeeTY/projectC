---
title: "**Ising Model of Ferromagnets**"
author: Tim T. Y. Lee
geometry: margin=2cm
output: pdf_document
numbersections: true
abstract: "This project is an investigation into the use of Monte-Carlo methods to tackle the Ising model of ferromagnetism which considers the interaction between spins of nearest neighbours. For simplicity, the 2-dimensional case is chosen meaning we only consider energy contributions from the four nearest neighbours of each spin. The model gives a way to evaluate the total energy of each microstate from which we can derive the probability of a certain change occurring(specifically the probability of a spin flipping). Hence, by generating random samples of the system at certain time steps, we can model the time evolution of the system given the temperature $T$."
---

# Introduction

Ferromagnets are materials in which spin orientations are able to exhibit long range order at low enough temperatures, the occurrence of such behaviour is a well known case of spontaneous symmetry breaking exemplifying how a system chooses a preferred direction. The Ising model is an attempt at describing the behaviour of spins within a ferromagnet by considering the spin interaction of closest neighbours only, while this may be a substantial simplification, it can be a powerful starting point as it lets us write down a relatively simple closed form Hamiltonian.

A Monte-Carlo method is best understood as a random sampling of a function's parameter space, which when done sufficiently gives a reasonable estimate for the solution. The rate at which the results converges will depend on the size of the parameter space and how well behaved the function is, in particular, if there are divergence points within the parameter space, results may vary dramatically if not treated properly. Over the course of this investigation, the effects of said divergences will be analysed and discussed in the context of the Ising Model, for which a discontinuity occurs at some critical temperature $T_c$.

Our aim is to find reasonable solutions to the Ising model in 2 dimensions via a Monte-Carlo method from which we can extract information about the properties of a ferromagnet. 

# Background Theory

The energy of a set of spins in a regular lattice in the presence of some magnetic field can be given by:

\begin{equation}  E=-J\sum_{<ij>} s_i s_j - H\mu \sum_{i=1}^{N^2} \label{eq:Hamil} \end{equation}

# Results

The main results of these investigation are shown in the graphs below, with acoompanying errors
