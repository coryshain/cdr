Introduction
============

Deconvolutional time series regression (DTSR) is a regression technique for time series that directly models temporal diffusion of effects.
DTSR recasts the streams of independent and dependent variables as `signals` and learns impulse response functions (IRF) that mediate the relationship between them.
Given data and a model template specifying the functional form(s) of the IRF kernel(s), DTSR finds convolution parameters that optimize some objective function.

The ``dtsr`` package documented here provides two Python implementations of DTSR, a neural-network based implementation (NNDTSR) and a Bayesian implementation (BDTSR).
NNDTSR is implemented in TensorFlow, learns point estimates for the parameters, and generally trains more quickly.
BDTSR is implemented in TensorFlow+Edward and learns posterior distributions over the parameters using one of a number of variational and MCMC inference techniques, providing direct quantification of uncertainty in the parameter estimates.
