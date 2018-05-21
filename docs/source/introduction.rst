.. _introduction:

Introduction
============

Deconvolutional time series regression (DTSR) is a regression technique for time series that directly models temporal diffusion of effects.
DTSR recasts the streams of independent and dependent variables as `signals` and learns impulse response functions (IRF) that mediate the relationship between them.
Given data and a model template specifying the functional form(s) of the IRF kernel(s), DTSR finds IRF parameters that optimize some objective function.

The ``dtsr`` package documented here provides two Python implementations of DTSR, a neural-network based implementation (DTSRMLE) and a Bayesian implementation (DTSRBayes).
DTSRMLE is implemented in TensorFlow and learns point estimates for the parameters.
DTSRMLE usually reaches a good solution more quickly than DTSRBayes, but often takes many more training epochs to settle on a final optimum.
DTSRBayes is implemented in TensorFlow+Edward and learns posterior distributions over the parameters using one of a number of variational and MCMC inference techniques, providing direct quantification of uncertainty in the parameter estimates.

This package provides (1) an API for programming with DTSR and (2) several executables that allow users to train and evaluate DTSR models out of the box, without needing to write any code.


This package was built and tested using Python 3.6.4, Tensorflow 1.6.0, and Edward 1.3.4.
Python 2.7.* support is not guaranteed, although certain DTSR features may still work.
If you run into issues using DTSR with other versions of these tools, you may report them in the issue tracker on `Github <https://github.com/coryshain/dtsr>`_.