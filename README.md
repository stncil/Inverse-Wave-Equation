# Inverse-Wave-Equation

## Research Methodology:
In this work, I propose an image-generative model by reversing the wave equation. 
I am investigating this because the previous works(mentioned in the references) show a very clear trend that physics-based equations have a promising generative modeling counterpart.

First, a general solution to the wave equation is of the following form:

![solution to the wave equation](./gifs/wave_sol_eq.png)

where exp is the exponential function, A is any complex number, and ω, kx, and ky are three numbers such that ω = f(kx,ky). The number ω is called the angular frequency of the wave, and the vector k = (kx,ky) its wave vector. This relation between the angular frequency and wave vector is called the dispersion relation. It can be used to infer some important properties of the waves like the speed at which the information and energy it carries propagate.
Additionally, if we add -iγ|k| to the angular frequency, it functions as a dissipative component, which works by removing energy from the wave thereby decaying it.

The idea is to use this equation to propagate any image forward in time which will lead it to a prior distribution. Then the neural network can reverse the flow, essentially exhibiting a generative process by reversing any randomly drawn image from the said prior.

## Procedure
Forward Process

![Forward Process](./gifs/forward.gif)

-------------------------------------------------------------
Reverse Process

![Reverse Process](./gifs/reverse.gif)

-------------------------------------------------------------

## Model Definition

![Model Definition](./gifs/model_definition.png)

## Results

<img src="./gifs/process1.gif" width="250" height="250"/>
<img src="./gifs/process2.gif" width="250" height="250"/>
<img src="./gifs/process3.gif" width="250" height="250"/>

## Interpolation

<img src="./gifs/process4_9.gif" width="250" height="250"/>

<img src="./gifs/process135.gif" width="250" height="250"/>


## References: 
1. Inverse Heat Equation (https://arxiv.org/pdf/2206.13397.pdf)
2. Simulating wave propagation with the optical Fourier engine (https://medium.com/optalysys/simulating-wave-propagation-with-the-optical-fourier-engine-f4a9f2e74d28)
3. GENPHYS: From Physical Processes to Generative Models (https://arxiv.org/pdf/2304.02637.pdf)
