.. _introduction:

Introduction
============

.. image:: rtdbanner.png

Continuous-time deconvolutional regression (CDR) is a regression technique for time series that directly models temporal diffusion of effects (Shain & Schuler, 2018, 2019).
CDR recasts the streams of independent and dependent variables as `signals` and learns impulse response functions (IRF) that mediate the relationship between them.
Given data and a model template specifying the functional form(s) of the IRF kernel(s), CDR finds IRF parameters that optimize some objective function.

The ``cdr`` package documented here provides Python implementations of two broad inference algorithms for CDR models: a maximum likelihood implementation (CDRMLE) and a Bayesian implementation (CDRBayes).
CDRMLE is implemented in TensorFlow, learns point estimates for the parameters, and generally trains more quickly.
However, it does not quantify uncertainty in the parameter estimates.
CDRBayes is implemented in TensorFlow+Edward, learns posterior distributions over the parameters using one of a number of variational and MCMC inference techniques, and generally trains more slowly.
However, CDRBayes models directly quantify uncertainty in the parameter estimates.
An intermediary approach is to run a CDRBayes model using variational inference with implicit improper uniform priors.
Such a model considers parameters to be random variables but uses maximum likelihood estimation to fit their means and variances, permitting quantification of uncertainty without injecting a learning bias through the prior.
See :ref:`config` for more information on how to do this.

This package provides (1) an API for programming with CDR and (2) executables that allow users to train and evaluate CDR models out of the box, without needing to write any code.

This package was built and tested using Python 3.6.4, Tensorflow 1.6.0, and Edward 1.3.5.
Python 2.7.* support is not guaranteed, although certain CDR features may still work.
If you run into issues using CDR with other versions of these tools, you may report them in the issue tracker on `Github <https://github.com/coryshain/cdr>`_.


References
----------
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. *EMNLP18*.
Shain, Cory and Schuler, William (2019). Continuous-time deconvolutional regression for psycholinguistic modeling. *PsyArXiv*. [https://doi.org/10.31234/osf.io/whvk5](https://doi.org/10.31234/osf.io/whvk5).