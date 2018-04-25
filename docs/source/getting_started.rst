.. _getting_started:

Getting Started
===============

The ``dtsr`` package provides utilities that support easy training and evaluation of DTSR models.
Most of these utilities depend on a config file ``*.ini`` containing metadata needed to construct the DTSR model(s).
At training time, a copy of the config file is placed in the output directory for the experiment, which can be useful for record keeping.
The approach of using config files rather than command line arguments to define experiments has several advantages:

- Users only need to define the experiment once, and settings can be reused for arbitrarily many models
- The config file contains a permanent record of the settings used in the experiment, facilitating later reproduction
- The config file can be shared across many utilities (training, evaluation, statistical testing, plotting, etc.)

This guide describes how to write the config file and use it to train and evaluate DTSR models.




Configuring DTSR Experiments
----------------------------

A set of DTSR experiments is definied in a config file ``*.ini``, which can then be passed to DTSR executables.
For example, once a config file ``foo.ini`` has been created, the models defined in it can be trained using the included ``train`` executable, like so:

``python -m dtsr.bin.train foo.ini``

The config file must at minimum contain the headings ``[data]`` (with pointers to training/evaluation data) and ``[settings]`` (with arguments used to construct the model).
In addition, the config file should contain at least one model heading, consisting of the prefix ``model_DTSR_`` followed by a custom name for the model.
For example, to define a DTSR model called ``readingtimes``, use the heading ``[model_DTSR_readingtimes]``.
The string ``DTSR_readingtimes`` will then be used by a number of utilities to designate this model
(The reason ``DTSR`` must be included in the prefix is that this repository distributes with unofficial support for running LM, LME, and GAM models from the same interface).
The data and settings configurations will be shared between all models defined a single config file.
Each model heading should contain a single field: ``form``.
The value provided to ``form`` must be a valid DTSR model string.
To run experiments with different data and/or settings, multiple config files must be created.

For more details on the available configuration parameters, see :ref:`config`.
For more details on DTSR model formulae, see :ref:`formula`.



Running DTSR executables
------------------------



Training DTSR Models
--------------------



Evaluating DTSR Models
----------------------





