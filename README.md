# diffusion-notes
Notes on machine learning diffusion models

### History
1) Sohl-Dickstein et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)
    > *Introduced diffusion for machine learning*
2) Ho et al. Denoising Diffusion Probabilistic Models (2020)
    > *Popularized diffusion models*
3) Ramesh et al. Hierarchical Text-Conditional Image Generation with CLIP Latents (2022)
    > *Revolutionized text-to-image generation with DALL·E 2*
4) Watson et al. De novo design of protein structure and function with RFdiffusion (2023)
    > *Revolutionized protein backbone design with RFdiffusion*

### Background
In generative machine learning, the goal is to model the **true underlying probability distribution** of some variable *x*

![Equation](https://latex.codecogs.com/png.latex?p(x))

where the sum of probabilities across all states *p*(*x*<sub>1</sub>) + *p*(*x*<sub>2</sub>) + ... + *p*(*x*<sub>n</sub>) must equal 1. For example, the probability distribution of a fair coin can be modeled explicitly with two states *p*(*x*<sub>heads</sub>) = 0.5 and *p*(*x*<sub>tails</sub>) = 0.5.

However, most real-world systems cannot be explicitly modeled this way due to the astronomically large number of possible states. For example, modeling the probability distribution of cat images with this approach would require the intractable task of assigning probabilities to every possible cat image. Given that randomization of just 12 pixels in an image results in 256<sub>intensities</sub><sup>3<sub>colors</sub><sup>12<sub>pixels</sub></sup></sup> ≈ 10<sup>86</sup> possible states which surpasses the ~10<sup>80</sup> atoms in the universe, the number of possible cat images is clearly astronomical in size.

Thus, the probability distribution must be implicitly modeled with an approximate parameterized function

![Equation](https://latex.codecogs.com/png.latex?p_\theta(x))

which predicts the **likelihood** of any given state from learned parameters *θ*.

### Gaussian Diffusion
...

### Multinomial Diffusion
The probabilities of a categorical variable *x* are described by the **categorical probability distribution**

![Equation](https://latex.codecogs.com/png.latex?C(x|p))

where the probabilites of each state *x*<sub>1</sub>, *x*<sub>2</sub>, ... *x*<sub>n</sub> are described by some function *p*. For example, a fair coin would be represented as *C*(*x*<sub>heads</sub>|*p*<sub>heads</sub>=0.5) and *C*(*x*<sub>tails</sub>|*p*<sub>tails</sub>=0.5). If, however, variable *x* is categorical across multiple dimensions
