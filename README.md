# diffusion-notes
Notes on machine learning diffusion models

### History
1) Sohl-Dickstein et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)
    > *Introduced diffusion for machine learning*
2) Ho et al. Denoising Diffusion Probabilistic Models (2020)
    > *Popularized diffusion models*

### Background
In generative machine learning, the goal is to model the true underlying probability distribution of some data *x*

![Equation](https://latex.codecogs.com/png.latex?p(x))

where the sum of probabilities across all states *p*(*x*<sub>1</sub>) + *p*(*x*<sub>2</sub>) + ... + *p*(*x*<sub>n</sub>) must equal 1. For example, the probability distribution of a fair coin can be modeled explicitly with two states *p*(*x*<sub>heads</sub>) = 0.5 and *p*(*x*<sub>tails</sub>) = 0.5.
