# a stable PGNN
A mini library for recreating (and generalizing) the experiments in the paper "A stable physics-guided neural networks approach of electromagnetic problems".

# Structure
Code divides the modelling task into 5 main classes:
* guides: A wrapper on the waveguiding structure under study. It contains classes for a a generaliezd guiding structure, and some specific premade ones, such as rib and channel waveguides.
* models: The type of PINN to use. The main is a simple wrapper around the FCNNs, but allows DCSNs (Discontinuity-capturing Shallow Networks) as well.
* optimizers: The optimizer to use. The user is allowed to directly use any of Pytorch's optimizers, but we add SOAP and and LM optimizer in this one as well, which are not native to PyTorch.
* samplers: Facilitates choosing certain sampling schemes, including uniform, random (uniform and Soboll), and randomn(Gaussian)
* Trainers: Takes care of setting up the training loop, including using the sampling, constructing the composite loss function, and using the optimizer to update model parameters.

in addition to those classes a utilities module exists to implement some supporting functionality in some cases such as modelling TE/TM in 1D.

# The types of Exepriments supported
* All experiments appearing in the original paper (2D scalar modal analysis of Channel and Rib waveguides)
* E-field Semi-vectorial analysis of 2D waveguides
* Modal Analysis of 1D slab waveguides
  
