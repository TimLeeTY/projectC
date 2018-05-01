---
title: |
    | NST-II Computational Physics Project
    | **Ising Model of Ferromagnets**
geometry: margin=2cm
output: pdf_document
numbersections: true
header-includes:
    - \usepackage{bm}
    - \usepackage{dsfont}
    - \usepackage{units}
    - \usepackage{fullpage}
    - \usepackage{amsmath}
    - \usepackage{mathtools}
    - \usepackage{mathrsfs}
    - \usepackage{float}
    - \usepackage{subfig}
    - \usepackage{xfrac}
    - \usepackage{caption}
    - \newcommand{\pdv}[3][]{ \frac{\partial^{#1}{#2}}{\partial {#3}^{#1}} }
    - \newcommand{\dv}[3][]{ \frac{\mathrm{d}^{#1}{#2}}{\mathrm{d} {#3}^{#1}} }
    - \newcommand{\abs}[2][ ]{\left\lvert {#2} \right\rvert^{#1}}
    - \newcommand{\avg}[1]{\left\langle {#1} \right\rangle}
    - \newcommand{\diff}[2][]{\mathop{}\!\mathrm{d}^{#1} {#2} \,}
    - \newcommand{\Tc}{ T_\mathrm{c}}
    - \newcommand{\kB}{ k_\mathrm{B}}
abstract: |
    This project is an investigation into the useage of Monte-Carlo methods to solve the Ising model of ferromagnetism. For simplicity, the 2-dimensional case is chosen and we consider only spin interactions between 4 closest neighbours. This model lets us evaluate the total energy of each microstate from which we can derive the probability of any particular spin flipping. The system can thus be evolved in time for some temperature, $T$, using the Metropolis algorithm. We are most interested in finding the critical temperature $T_\mathrm{c}$, below which the system exhibits spontaneous magnetisation. To this end, we identified the maxima in heat capacity and magnetic susceptibility, for each value of $N$ and extrapolated our results to get  $N\to\infty$, $T_{\mathrm{c},\infty} = 2.262 \pm 0.006$, agreeing with the analytic Onsager's result of $\sim 2.269$. The critical exponent, $\nu$, describing\: $T-T_\mathrm{c} \propto N^{-1/\nu}$ was found to be $0.89 \pm 0.06$, which is within $\sim 2 \sigma$ of the expeccted value of $1$.
---

# Introduction

Ferromagnets are materials in which spin orientations are able to exhibit long range order below some critical temperature, the occurrence of such behaviour is a well known case of spontaneous symmetry breaking [@sadler]. The Ising model attempts to describe the behaviour of spins within a ferromagnet by considering the spin interactions between closest neighbours only. While this may be a substantial simplification, it proves to be a powerful tool for explaining ferromagnetic characteristics and provides reasonably physical results.

A Monte-Carlo method is best understood as a random sampling of a function's parameter space, which if done sufficiently converges onto the solution [@monte]. The rate at which the result converges will depend on the size of the parameter space and how well behaved the function is; in particular, if points of divergence are not treated properly, the accuracy of results may vary dramatically. Over the course of this investigation, the effects of said divergences will be analysed and discussed in the context of the Ising Model near the transition tempearture $T_\mathrm{c}$.

Our aim is to find reasonable solutions to the Ising model in 2 dimensions via the Metropolis algorithm [@metropolis] (a specialised Monte Carlo method), allowing us to model the time evolution of spins within the magnet. Properties of the ferromagnet, including the mean magnetisation, $M$, specific heat, $C$, and magnetic susceptibility, $\chi$, can thus be found as functions of temperature. From analytical results generated from mean field theory, we know what all these quantities should display some form of discontinuity around the critical temperature [@stanley], which we will use to identify the location of $T_\mathrm{c}$. Due to finite scaling, we see that the critical temperature for large $N$ tends towards the analytic Onsager's result:  $T_\mathrm{c}(N\to\infty) = 1/(1+\ln 2) \approx 2.27$ [@onsager] (here and henceforth all temperature will be given in units of $J/\kB$, where $J$ is the spin interaction energy).

Finally, the behaviour of the magnet is also identified for nonzero external fields. In particular, we vary $H$ to form of a magnetic hysteresis loop for different temperatures, once again exhibiting ferromagnetic properties at subcritical temperature.

# Theoretical background

## Ising model

We begin by making the assumption that spins are only able to align along one axis; greatly simplifying calculations while still maintaining relatively physical. The system is described by a set of spins 1/2 particles, $\{s_i=\pm1\}$, arranged in a 2 dimensional lattice. In the presence of a magnetic field, $H$, the total energy can written in terms of spin interactions between closest neighbours and the spins' coupling to $H$:
\begin{equation}
    E=-J\sum_{\langle ij \rangle} s_i s_j - H\mu \sum_{i=1}^{N^2} s_i
    \label{eq:Hamil} 
\end{equation}

Where $J$ is the interaction energy, $\mu$ is the magnetic moment, and $\langle ij \rangle$ represents a sum between closest neighbours. For materials of interest to this investigation (ferromagnetic and paramagnetic), $J$ is positive meaning spin alignment is favoured. The change in energy when flipping a particular spin $s_i$ is then:

\begin{equation}
    \Delta E_i = \sum_\delta s_i s_{i+\delta} - 2H\mu s_i
    \label{eq:delE}
\end{equation}

Where $\delta$ represents the relative index of nearest neighbours.

We can determine the probability of the system taking on a spin configuration by considering the Boltzmann distribution:
\begin{equation}
    P(\mathbf{x}) = e^{-\beta E(\mathbf{x})} \quad \text{where} \quad \beta = \frac{1}{k_\mathrm{B}T}
    \label{eq:boltz}
\end{equation}

Where $\mathbf{x}$ represents a point within the phase space. And thus the thermal average of an observable, $F$, is defined in the canonical ensemble of states as per statistical mechanics:
\begin{equation}
    \langle F\rangle = \frac{1}{Z}\int F e^{-E\beta} \diff{\mathbf{x}} \quad \text{where} \quad Z=\int e^{-E\beta} \diff{\mathbf{x}}
    \label{eq:F}
\end{equation}

However, this integral is unwieldy even with the simplified Ising model. Instead, we use a Monte Carlo method with importance sampling based on the Boltzmann distribution known as the Metropolis algorithm [@metropolis], described in more detail in section \ref{sec:met}.

Finally, a note on the boundary conditions. To maximise the number of interacting spins, we have imposed periodic boundary conditions on the system such the 2D matrix is mapped onto a toroidal surface.

## Finite size scaling

It is important to note that in the limit of an infinite lattice, a true transition occurs at the critical temperature. However, for finite $N$, divergences in observables to $\infty$ are not possible, instead, we approximate the critical temperature as the location of extrema in the relevant observables.

From theory of finite-scaling relations, we know that the size-dependent ordering of temperature is given by [@landau]:
\begin{equation}
    T_\mathrm{c}(N) - T_\mathrm{c}(\infty) = a N^{-1/\nu} \quad N\to \infty
    \label{eq:TcInf}
\end{equation}

The critical exponent $\nu$ in this case is know to be 1.

# Method

## Metropolis algorithm \label{sec:met}

The Metropolis algorithm [@metropolis] can be summarised by the following flow chart:

\begin{figure}[H]
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.8in]{flow.pdf}}
\caption{Flow chart describing the Metropolis algorithm.}
\label{fig:flow}
\vspace{-10pt}
\end{figure}

1.  We choose some arbitrary starting configuration, and start with $t=0$.
2.  Choose a point within the lattice based on a uniform probability distribution. 
3.  Calculate the change in energy if the spin is flipped, $\Delta E$ as per equation \ref{eq:delE}. 
    - If $\Delta E<0$, the spin is flipped and the new configuration is recorded.
    - If $\Delta E>0$, we generate a random number, $\alpha$, in the uniform distribution $[0,1]$, if $\alpha < \exp{(-\beta\Delta E)}$, the spin is flipped and the new configuration is recorded.
5.  Repeat from step 2 $N^2$ times for 1 Monte Carlo step .
4.  Record observables, e.g.\ total magnetisation, $M_t$, and total energy $E_t$.
6.  Increment $t$ and repeat from step 2 until $I$ Monte Carlo steps are completed.
7.  Average $M$ and $E$, over $t$ and find their variance. Calculate $C$ and $\chi$ from averages and variances, as per equation \ref{eq:calc_obsv}
8.  Output data.

## Calculating observables

Using equation \ref{eq:F}, and the dissipation-fluctuation theory [@waldram] we can write down:

\begin{equation}
    \begin{aligned}
        C & = \pdv{E}{T} = \frac{\avg{E^2}-\avg{E}^2}{\kB T^2} \\
        \chi & = \pdv{M}{T} = \frac{\avg{M^2}-\avg{M}^2}{\kB T}
    \end{aligned}
    \label{eq:calc_obsv}
\end{equation}

Where $M$ is the total magnetisation, $E$ the total energy and the angle brackets in this instance represents a time average.

## Autocorrelation times

After the system reaches an equilibrium state, there is still no guarantee that samples from individual MC steps are uncorrelated. We therefore have to treat these results with care as they will introduce skew in the errors of observables. Introducing the autocorrelation function, $A(\tau)$:

\begin{equation}
    A(\tau) = \avg{M'(t)M'(t+\tau}/\avg{M'^2}\quad \text{where} \quad M'(t) = M(t) - \avg{M}
\end{equation}

For samples to be uncorrelated, we require that they must be separated by an interval $\Delta t>\tau_\mathrm{c}$ where $A(\tau_\mathrm{c})=e^{-1}$. This was used to determine the errors in $C$ and $\chi$.

## Bootstrap method (resampling) \label{sec:bootstrap}

We know that successive samples from the Metropolis algorithm will inevitably be correlated as multiple steps may be require to \`randomise' the spins. If calculations are made  based on these correlated values, the errors in observed quantities will display some bias. One way to circumvent this is to resample the data starting from a set of uncorrelated states. For some observable $A$, the error can be calculated thus:

Given $n$ uncorrelated states (by taking samples $\tau_\mathrm{c}$ steps apart), we resample the data $n$ times (allowing for overlaps) and compute $A_i$ for sample $i$. The overall error for $\avg{A}$ (angle brackets represents averaging over resampled states) will be:

\begin{equation}
    \sigma_A ^ 2 = \avg{A^2} - \avg{A}^2 
\end{equation}

## $H\neq 0$ hysteresis

To show how the total magnetisation varied with some nonzero external field, $H$ was varied between the range $[-1. 1]$ in a linear fashion (triangle wave) and the resultant total magnetisation and energy were plotted as a function of $H$. We incremented $H$ sufficiently slowly to ensure equilibrium is reached within a few MC steps between each transition, 10 MC steps were performed between changes in $H$.

## Performance

Due to the large $N$ used (up to 55), the whole programme takes ~5 hours to run if ran separately. However, as most results are written out to separate `.csv` files, we could easily run the code for multiple values of $N$ at once. Initially, to speed up calculations, the `multiprocessing` package was used to parallelise the programme, however, it rendered the PC unusable and was therefore ultimately abandoned.

In an attempt the lower the computation cost of each MC step, the probabilities $\exp(-\beta\Delta E)$ were stored separately in an array to prevent having to repeat the same calculation multiple times. However, this did not prove to speed up the process much suggesting that the `python` interpreter either knows to stash these calculations automatically, or the retrieval of values from an array is slower than the calculation of an exponent (the former of which sounds more plausible).

Finally, the code was \`vectorised' such that the MC steps for multiple values of $T$ would be computed in unison. This was perhaps the most notable improvement, exemplifying the infamously inefficient for loops in `python`.

# Results

## Time evolution of spin configurations, and initial conditions \label{sec:time_evole}

We first consider how the total magnetisation of the system evolves in time if the initial spins are chosen randomly.

\begin{figure}[H]
\vspace{-10pt}
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{./Mag/Mag.pdf}}
\caption{How total magnetisation (total spin) of a system of $50\times 50$ spins evolves in time for temperatures above, below, and around the critical point,corresponding to $T = 4.0$, 1.0, and 2.7, starting from a random configuration.}
\label{fig:totMag}
\vspace{-10pt}
\end{figure}

It is clear that for low temperatures, the system tends to an extreme as all spins become aligned, displaying spontaneous magnetisation. At temperatures above the critical point, the system appears to behave randomly and fluctuates rapidly about $M=0$. Close to the critical point, some order can be observed over short ranges, causing the total magnetisation to alternate more slowly about $M=0$.

\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[$T = 1.0$]{
    {\includegraphics[width=2in]{./spin/T10-arr3.pdf}}}\quad
  \subfloat[$T = 2.7$]{
    \includegraphics[width=2in]{./spin/T27-arr3.pdf} \label{singleError}}
  \subfloat[$T = 4.0$]{
    \includegraphics[width=2in]{./spin/T40-arr3.pdf} \label{singleError}}
  \caption{Plot of the spin orientations within a $50\times50$ matrix after 2 steps for temperatures above, below, and around the critical point, corresponding to $T = 4.0$, 1.0, and 2.7, starting from a random configuration.}
  \label{fig:spinArr3}
\vspace{-10pt}
\end{figure}


\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[$T = 1.0$]{
    {\includegraphics[width=2in]{./spin/T10-arr21.pdf}}}\quad
  \subfloat[$T = 2.7$]{
    \includegraphics[width=2in]{./spin/T27-arr21.pdf} \label{singleError}}
  \subfloat[$T = 4.0$]{
    \includegraphics[width=2in]{./spin/T40-arr21.pdf} \label{singleError}}
  \caption{Plot of the spin orientations within a $50\times50$ matrix after 20 steps for temperatures above, below, and around the critical point, corresponding to $T = 4.0$, 1.0, and 2.7, starting from a random configuration.}
  \label{fig:spinArr21}
\vspace{-10pt}
\end{figure}

Starting from a random collection of spins, we can see that for low temperatures, regions of different spins form, which limits the system's interaction energy to the zone boundaries. This explains why a large number of steps are required to reach an equilibrium (figure \ref{fig:totMag}), as the formation of zones drastically hinders the thermalising process. In the long run, the smaller regions will shrink until eventually the whole system is dominated by one spin at equilibrium. This problem is exacerbated for larger $N$ as more zones are able to form meaning that the number of steps require to bring a subcritical system to equilibrium from a random initial state scales with $N$.

Close to the critical temperature, small amounts of ordering can be seen at equilibrium but the boundaries between such zones are more ambiguous than the low temperature case. For higher temperatures, the spins remain in an almost random configuration and exhibits no long range order as expected. 

If instead we initialise the system with a homogeneous configuration, the problem with zone boundaries vanishes as the system starts off in its ground state. Furthermore, for supercritical temperatures, the rate at which the system thermalises is not hindered, demonstrated using a $50\times 50$ system below in figure \ref{fig:spinHomo} below:

\begin{figure} [H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[1 step]{
    {\includegraphics[width=2in]{./spin/T40-arr2ones.pdf}}}\quad
  \subfloat[2 steps]{
    \includegraphics[width=2in]{./spin/T40-arr3ones.pdf} \label{singleError}}
  \subfloat[20 steps]{
    \includegraphics[width=2in]{./spin/T40-arr19ones.pdf} \label{singleError}}
  \caption{Plot of the spin orientations within a $50\times50$ matrix at $T = 4$ after 1, 2, and 20 steps, starting from a homogeneous configuration.}
  \label{fig:spinHomo}
\vspace{-10pt}
\end{figure}

It is abundantly clear that the system reaches equilibrium within 20 steps at $T=4$.

To reinforce this point, animations of the first set of spin evolutions were created, (see Appendix C). These demonstrate how the effectiveness of the Metropolis algorithm is sensitive to initial conditions, suggesting they should be carefully chosen to best facilitate the investigation.

## Mean magnetisation and energy \label{sec:meanMag}

We move on to the mean magnetisation and energy of a system in equilibrium and how they depend upon the temperature. 50000 iterations were performed for each of these results, where the first 5000 were treated as thermalising steps and therefore not considered in equilibrium.

Firstly, we must acknowledge that taking a mean over the total magnetisation will always be 0 given a long enough time for finite lattices and nonzero $T$ (finite probability of whole lattice flipping). To overcome this, we find the average over the absolute total magnetisation instead.

\begin{figure}[H]
\vspace{-10pt}
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{3.pdf}}
\caption{How the mean of the absolute magnetisation of a system of $N\times N$ spins depends on temperature plotted for $N=15$, 35, and 55. The highlighted region is the area in which the mean magnetisation drops the most rapidly for $N=55$, giving a rudimentary estimate for $T_\mathrm{c}$. Some error bars were omitted to maintain readability.}
\label{fig:meanMag}
\vspace{-10pt}
\end{figure}

We expect the total magnetisation to drop around the critical point (highlighted region in figure \ref{fig:meanMag}). For the larger $N$, this drop is more defined, meaning we are able to find $T_\mathrm{c}$ more accurately. For the $N=55$ case demonstrated in figure \ref{fig:meanMag}, $T_\mathrm{c} = 2.30\pm0.05$. Furthermore, we can see that the fluctuation in magnetisation (as shown by the error bars) is much greater for smaller lattices as well. 

Next, we have the energy per spin as a function of temperature. The total energy was normalised by the size of the matrix to aid comparison between different $N$.

\begin{figure}[h]
\vspace{-10pt}
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{2.pdf}}
\caption{How total energy of a system of $N\times N$ spins depends on temperature plotted for $N=15$, 35 and 55. Some error bars were omitted to maintain readability.}
\label{fig:totEng}
\vspace{-10pt}
\end{figure}

Unlike the mean magnetisation, the energy of the system does not exhibit a sharp transition around the critical temperature. The difference between the two sizes of lattice also does not appear to greatly affect the average energy per site. Note that for low temperatures, the energy per site tends towards $-2J$ reinforcing the fact that spins tend to be fully aligned at low temperatures (4 neighbours times $1/2$ to avoid double counting). The main difference between the two cases is the size of fluctuations that occur, where much like the mean magnetisation case, larger lattices give rise to smaller fluctuations.

## Heat capacity, $C$

\begin{figure}[H]
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{4.pdf}}
\caption{How the specific heat, $C$, of a system of $N\times N$ spins depends on temperature plotted for $N=15$, 35, and 55. Some error bars were omitted to maintain readability.}
\label{fig:heatCap}
\vspace{-10pt}
\end{figure}

The heat capacity was plotted as a function of temperature and normalised with respect to the lattice size to aid comparison. We see a clear spike in $C$ near the critical temperature providing an easy way of determining the critical temperature by identifying the temperature at which $C$ is maximum, $T_\mathrm{c}(N=55) = 2.264 \pm 0.014$. Errors obtained from the bootstrap method (section \ref{sec:bootstrap}) are also shown.

## Magnetic susceptibility, $\chi$

\begin{figure}[H]
\vspace{-10pt}
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{6.pdf}}
\caption{How the magnetic susceptibility, $\chi$ (using the variance of $\abs{M}$), of a system of $N\times N$ spins depends on temperature plotted for $N=15$, 35, and 55. Some error bars were omitted to maintain readability.}
\label{fig:magSus}
\vspace{-10pt}
\end{figure}

Comparing figures \ref{fig:heatCap} and \ref{fig:magSus} clearly shows that the magnetic susceptibility of the ferromagnet has a much more pronounced discontinuity around the critical temperature producing a more prominent spike. For the $N=55$ case, the peak was located at $T_\mathrm{c}(N = 55) = 2.236 \pm 0.014$.

Despite the distinctive peak, the results are clearly subpar for smaller $N$ making it an unideal candidate for determining $T_{\mathrm{c},\infty}$ (equation \ref{eq:TcInf}). This is discussed further in section \ref{sec:chiDiss}. A better method is to take the variance in the absolute of magnetisation, $\abs{M}$, instead, eliminating the probability of spontaneous flips at low temperatures.

\begin{figure}[H]
\vspace{-10pt}
\captionsetup{width=0.8\textwidth}
\centering
{\includegraphics[width=3.5in]{8.pdf}}
\caption{How the magnetic susceptibility, $\chi'$ (using the variance of $\abs{M}$), of a system of $N\times N$ spins depends on temperature plotted for $N=15$, 35, and 55. Some error bars were omitted to maintain readability.}
\label{fig:magSusAbs}
\vspace{-10pt}
\end{figure}

Again, looking at the peak of $\chi'$ gives yields the critical temperature: $T_\mathrm{c}(N = 55) = 2.314 \pm 0.014$.

## Finite size scaling 
Using the peaks of the specific heat for a range of different lattice sizes, we can determine how the critical temperature depends on $N$.

\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[Plot of $T_\mathrm{c}$ against $N$]{
    {\includegraphics[width=3in]{5.pdf}}\label{fig:Tc1}}\quad
  \subfloat[Plot of $T_\mathrm{c}$ against $1/N$]{
    \includegraphics[width=3in]{7.pdf} \label{fig:Tc2}}
  \caption{Plot of the critical temperature (blue)  from peaks of $C$, as a function of $N$ and $1/N$, the fitted curve (orange) and Onsager's result (Onsager 1944) (green) is also plotted.}
  \label{fig:Tc}
\vspace{-10pt}
\end{figure}

We first fitted the data to equation \ref{eq:TcInf}, shown in figure \ref{fig:Tc1} (orange), which showed that the critical exponent, $\nu$ is closes to 1. From there we refit the same data against $1/N$ ($\nu$ fixed to 1), shown in figure \ref{fig:Tc2} (orange). The combination of these results gives $T_{\mathrm{c}, \infty} = 2.275\pm 0.007$ (error obtained from regression).

We then turn to the peaks of the modified magnetic susceptibility, $\chi'$ as shown in figure \ref{fig:magSusAbs}. The critical temperatures for these are summarised below:

\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[Plot of $T_\mathrm{c}$ against $N$]{
    {\includegraphics[width=3in]{9.pdf}}\label{fig:Tc1}}\quad
  \subfloat[Plot of $T_\mathrm{c}$ against $1/N$]{
    \includegraphics[width=3in]{10.pdf} \label{fig:Tc2}}
  \caption{Plot of the critical temperature (blue) from peaks of $\chi'$, as a function of $N$ and $1/N$, the fitted curve (orange) and Onsager's result (Onsager 1944) (green) is also plotted.}
  \label{fig:Tc}
\vspace{-10pt}
\end{figure}

The fitting procedure adopted remains unchanged. The intersection was found to be $T_{\mathrm{c}, \infty} = 2.262 \pm 0.006$ (error obtained from regression). This also provided the best critical exponent $\nu = 0.89 \pm 0.06$.

## Hysteresis for $H\neq0$

If we now turn on an external magnetic field and then vary it slowly between $-1<H<1$, we expect a ferromagnet to display a hysteresis loop with non-zero area. In addition, we also expect said ferromagnetic properties to be more prominent at lower temperatures. 

The hysteresis loop for 3 different temperatures were plotted, showing that the ferromagnetic properties are only present for $T<T_\mathrm{c}$.

We also found the energy of the system under the same conditions and plotted it against the external field below.

\begin{figure}[H]
\vspace{-10pt}
  \centering
	\captionsetup{width=0.95\textwidth}
  \subfloat[Magnetisation per spin, $M$, against $H$]{
    {\includegraphics[width=3in]{hyst.pdf}}\label{fig:hyst}}\quad
  \subfloat[Energy per spin, $E$, against $H$]{
    \includegraphics[width=3in]{hystEng.pdf} \label{fig:hystEng}}
  \caption{How the properties of a $30 \times 30$ spin matrix depends on the external field $H$ which varies periodically between -1 to 1, for different temperatures.}
  \label{fig:Tc}
\vspace{-10pt}
\end{figure}


# Discussion

## Initialisation of spins

From figure \ref{fig:spinArr21} we know that spins in a ferromagnet tend to form zones of opposing spins if the spins are initialised without a bias. This makes the process of reaching equilibrium (when all spins are aligned) much more time consuming than it ought to be for large matrices. To overcome this problem, we initialise the spins homogeneously meaning the whole matrix starts of as up spins ($+1$). Qualitatively, the appearance of zone boundaries would cause large spikes in $C$ even at temperatures well below $T_\mathrm{c}$ as the system fails to reach equilibrium in the 5000 allocated MC steps. 

Fortunately, at high temperatures, the thermalisation of the system is much less dependent on initial conditions and will readily reach equilibrium within 5000 steps. Therefore, choosing a homogeneous starting configuration is in fact rather sensible as it yields useful results most consistently.

## Corrections for $\chi$ \label{sec:chiDiss}

As previously mentioned, if we applied equation \ref{eq:calc_obsv} directly, $\chi$ would be skewed towards lower temperatures. This resulted from the finite probability for all spins to flip simultaneously at subcritical temperatures:
\begin{equation}
    \begin{aligned}
        P_\mathrm{flip} & \sim \prod_{i=1}^{N^2}\exp(-4\beta J) \\
        & = \exp(-4N^2\beta J)
    \end{aligned}
\end{equation}
We see that this probability shrinks rapidly as $N$ and $\beta$ grows. However, at temperatures just below $T_\mathrm{c}$, this effect is significant and has to be compensated. An obvious solution would be to modify \ref{eq:calc_obsv} and replace the total magnetisation, $M$, with its absolute value, $\abs{M}$ (see figure 
\ref{fig:magSusAbs} such that the modified $\chi$ will not be affected by these flips. Comparing figures \ref{fig:magSus} and \ref{fig:magSusAbs} shows a few key differences:

-   the peaks have moved back to the right, towards the analytical solution.
-   $\chi$ is relatively larger for higher temperatures. 
-   The peak temperature now decreases with increasing $N$, a trend resembling $C$ suggesting the scaling relation is now correct.

Ultimately, the use of this modified susceptibility is justified by the fact that it produces a better estimate for $T_{\mathrm{c},\infty}$ and obeys the proper scaling relations.

## Finite size scaling

Plotting the critical temperature against $N$ for $\chi'$ yielded our best value of $\nu$. However, it was still $\sim2\sigma$ less than the theoretical result of $\nu=1$. This discrepancy can be attributed to the fact that equation \ref{eq:TcInf} is only exact in the limit $N\to\infty$ [@landau], whereas we have only sampled up to $N=55$.

## Hysteresis for $H\neq0$

The results for the hysteresis loops were as expected and again strongly suggested that ferromagnetic behaviour only exists at subcritical temperatures. It also confirmed that the \`hardness' of a magnet increases the further below $T_\mathrm{c}$ we go.

The energy of the system also behaves as we expect, varying linearly with $\pm H$ from a minimum of $-4J$ (extra $2J$ from the field contribution) and \`hopping' to the $\mp H$ line when $M$ transitions from one sign to another.

## Improvements

Looking at figures \ref{fig:Tc} shows clearly that the errors when determining $T_\mathrm{c}$ using $C$ is rather significant. While the deviations from the trend are not too pronounced, the quality of the fit could be improved were smaller $N$ also included. In particular, as the function of $1/N$ flattens out, more data points towards the lower end of $N$ would have been helpful in attaining a better value for the critical exponent, $\nu$. Furthermore, perhaps a more extreme solution would be to increase the sampling rate of $T$, improving the precision of the measurement. However, since it is not straightforward to determine the underlying error in the maximal value, the marginal gain from this may be limited.

It is also evident from our results that the Metropolis algorithm partially breaks down $T\to T_\mathrm{c}$ as the system fails to equilibrate. From literature, we know that there are other algorithms that work by \`clustering' spins together and flipping them as groups at a time. The earliest example is the Swendsen-Wang cluster algorithm which is particularly useful for $T$ close the $T_\mathrm{c}$ as it is able to reduce the relaxation time significantly [@wang].

# Conclusion

Our investigation has shown that the Ising model is adept at describing the properties of a ferromagnet despite its massive simplifications. In particular, we are able to obtain a clear transition temperature from the model by looking at the heat capacity and magnetic susceptibility for varying values of $N$. It is important to note however, the process of performing such calculations is computationally expensive, while still failing to provide the most best results around the transition point. Hence, if at all possible, alternatives to the Metropolis algorithm should be considered when implementing the Ising model numerically.

Overall the results obtained were reasonable and matched expected theoretical results. The best estimate for the asymptotic value of the critical temperature is $T_{\mathrm{c},\infty} = 2.262 \pm 0.006$ which is remarkably close to the true value of $2.269$ with only a 0.23% error. However, for the critical exponent $\nu$, the quality of our results were not quite good enough and gave $\nu = 0.86\pm 0.06$ which is $\sim 2\sigma$ from the expected value of 1 which is likely a problem to do with the small lattices used in the computation [@landau].

# References

