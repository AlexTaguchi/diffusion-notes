# diffusion-notes
Notes on machine learning diffusion models

### History
1) Sohl-Dickstein et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)
    > *Introduced diffusion for machine learning*
2) Ho et al. Denoising Diffusion Probabilistic Models (2020)
    > *Popularized diffusion models*
3) Hoogeboom et al. Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions (2021)
    > *Introduced multinomial diffusion for categorical distributions*
4) Ramesh et al. Hierarchical Text-Conditional Image Generation with CLIP Latents (2022)
    > *Revolutionized text-to-image generation with DALL·E 2*
5) Watson et al. De novo design of protein structure and function with RFdiffusion (2023)
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
The probabilities of a single categorical variable *x* are described by a **categorical distribution**

![Equation](https://latex.codecogs.com/png.latex?C(x|p))

where the probabilites of each state are described by function *p*. For example, a coin flip would be represented as *C*(*x*|*p*<sub>heads</sub>,*p*<sub>tails</sub>), where *p*<sub>heads</sub> + *p*<sub>tails</sub> = 1. Flipping a coin multiple times results in a **binomial distribution**

![Equation](https://latex.codecogs.com/png.latex?B(x|p)=\binom{n}{x}p^x(1-p)^{n-x})

where the probability of any state of *x* is calculated from the probability *p* of that state occuring once and the total number of coin flips *n*. For example, the probability of observing 2 heads and 1 tails is

![Equation](https://latex.codecogs.com/png.latex?\binom{3}{2}0.5^2(1-0.5)^{3-2}=\frac{3!}{2!(3-2)!}0.5^2(0.5)^{1}=0.375)

Rolling a dice with more than 2 sides requires a more general **multinomial distribution**

![Equation](https://latex.codecogs.com/png.latex?M(x|p))

where there are no longer any restrictions on the number of possible states or events, so long as all events sample from the same categorical distribution.
