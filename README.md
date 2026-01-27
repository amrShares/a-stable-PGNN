# A stable PGNN
A mini library for recreating (and generalizing) the experiments in the paper  
[A stable physics-guided neural networks approach of electromagnetic problems](https://www.sciencedirect.com/science/article/abs/pii/S0021999125008502).

# Structure
Code divides the modelling task into 5 main classes:
* guides: A wrapper on the waveguiding structure under study. It contains classes for a generalized guiding structure, and some specific premade ones, such as rib and channel waveguides.
* models: The type of PINN to use. The main is a simple wrapper around FCNNs, but allows DCSNs (Discontinuity-capturing Shallow Networks) as well.
* optimizers: The optimizer to use. The user is allowed to directly use any of PyTorch's optimizers, but we add SOAP and LM optimizers as well, which are not native to PyTorch.
* samplers: Facilitates choosing certain sampling schemes, including uniform, random (uniform and Sobol), and random (Gaussian).
* trainers: Takes care of setting up the training loop, including sampling, constructing the composite loss function, and using the optimizer to update model parameters.

In addition to those classes, a utilities module exists to implement supporting functionality, such as modelling TE/TM modes in 1D.

# Types of experiments supported
* All experiments appearing in the original paper (2D scalar modal analysis of channel and rib waveguides)
* Semi-vectorial E-field analysis of 2D waveguides
* Modal analysis of 1D slab waveguides
