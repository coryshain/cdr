.. _getting_started:

Getting Started
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   config
   train
   evaluate

The ``dtsr`` package provides utilities that support easy training and evaluation of DTSR models.
Most of these utilities depend on a config file ``*.ini`` containing metadata needed to construct the DTSR model(s).
At training time, a copy of the config file is placed in the output directory for the experiment, which can be useful for record keeping.
The approach of using config files rather than command line arguments to define experiments has several advantages:

- Users only need to define the experiment once, and settings can be reused for arbitrarily many models
- The config file contains a permanent record of the settings used in the experiment, facilitating later reproduction
- The config file can be shared across many utilities (training, evaluation, statistical testing, plotting, etc.)

This Getting Started guide describes how to write the config file and use it to train and evaluate DTSR models.
