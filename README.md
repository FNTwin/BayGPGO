# BayGPGO - Bayesian multi-objective Gaussian Process GO
A custom (and from scratch) implementation of a Black box multi-objective Optimization with Gaussian Process as a surrogate model.
It is still in development but it was successfully used to achieve a bottom up optimization of the Dissipative
Particle Dynamics force field for a complex system of polymers chains functionalized gold nanoparticles in a water solvent and a PEG Brownian dynamics model by
doing both single and multi objective routine. 

# Hyperparameters
The Hyperparameters of the GP are optimized by the common technique of maximizing the Log Marginal Likelihood. In this repository this is achieved by using a search grid (although not in an efficient way) or by using the scipy optimizer module (L-BFGS-B, TNC, SLSCP).
The analytical gradient is implemented for the Radial Basis Function kernel and it is possible to use the derivate of the Log Marginal Likelihood to optimize the hyperparameters.

<a href="https://ibb.co/D8yvW3x"><img src="https://i.ibb.co/pR8MwCt/Figure-6.png" alt="Figure-6" border="0"></a>

# Acquisition function
As it is there are two different acquisition function implemented right now:

-Expected Improvement (EI)

-UCB (Upper Confidence Bound)

# Maximizing the Acquisition function 
In this little package right now there are 3 ways to run an optimization task with Gaussian Processes:

-NAIVE : AkA sampling the acquisition function with a grid of some kind or a quasi random methods as LHS (require smt package)

-BFGS : optimize the Acquisition function by using the L-BFGS-B optimizer

-DIRECT : optimize the Acquisition function by using the DIRECT optimizer (require DIRECT python package)

-PSO : optimize the Acquisition function by a Particle Swarm Optimization genetic algorithm

<a href="https://ibb.co/GPSM0cm"><img src="https://i.ibb.co/f0wN24J/Figure-7.png" alt="Figure-7" border="0"></a>

# Made for experiments
Easy to use it with a shell procedure!
Load the data and just .suggest_location() to get the next points of your experiment! 

# Multi-objective
The package contains an implementation of the NSGAII genetic solver that allows to solve multi objective 
problems. It has also an early version of a Multi-objective Bayesian optimization that uses the NSGAII and optimize 
the Acquisition function EI (or whatever acquisition function you choose) or the mean function of the GP. 
It will follow a more precise implementation of the maximization of Hypervolume improvement for batch processing.

<a href="https://ibb.co/MhC92Yc"><img src="https://i.ibb.co/7z1bYwn/pareto.png" alt="pareto" border="0"></a>





