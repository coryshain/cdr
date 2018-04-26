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

The config file must at minimum contain the sections ``[data]`` (with pointers to training/evaluation data) and ``[settings]`` (with arguments used to construct the model).
In addition, the config file should contain at least one model section, consisting of the prefix ``model_DTSR_`` followed by a custom name for the model.
For example, to define a DTSR model called ``readingtimes``, use the section heading ``[model_DTSR_readingtimes]``.
The identifier ``DTSR_readingtimes`` will then be used by a number of utilities to designate this model
(the reason ``DTSR`` must be included in the prefix is that this repository distributes with unofficial support for running LM, LME, and GAM models from the same interface).
The data and settings configurations will be shared between all models defined a single config file.
Each model section heading should contain a single field: ``formula``.
The value provided to ``formula`` must be a valid DTSR model string.
To run experiments with different data and/or settings, multiple config files must be created.

For more details on the available configuration parameters, see :ref:`config`.
For more details on DTSR model formulae, see :ref:`formula`.



Running DTSR executables
------------------------

A number of executable utilities are contained in the module ``dtsr.bin`` and can be executed using Python's ``-m`` flag.
For example, the following call runs the ``train`` utility on all models specified in a config file ``config.ini``:

``python -m dtsr.bin.train config.ini``

Usage help for each utility can be viewed by running:

``python -m dtsr.bin.<UTIL-NAME> -h``

for the utility in question.
Or usage help for all utilities can be printed at once using:

``python -m dtsr.bin.help``

The following sections go into more detail about training and evaluation utilities that will likely be useful to most users.




Training DTSR Models
--------------------

Once the config file has been written, training a DTSR model is simple using the ``train`` utility, which takes as its principle argument a path to the config file.
For example, if the config file is names ``experiment.ini``, all models defined in it can be trained with the call:

``python -m dstr.bin.train experiment.ini``

To restrict training to some subset of models, the ``-m`` flag is used.
For example, the following call trains models ``A`` and ``B`` only:

``python -m dtsr.bin.train experiment.ini -m DTSR_A DTSR_B``


Evaluating DTSR Models
----------------------

This package provides several utilities for inspecting and evaluating fitted DTSR models.
The principal evaluation utility is ``predict``.
The following generates predictions on test data from the model ``DTSR_A`` defined in ``experiment.ini``:

``python -m dtsr.bin.predict experiment.ini -m DTSR_A -p test``

This call will save files containing elementwise predictions, errors, and likelihoods, along with a performance summary.
For more details on usage, run:

``python -m dtsr.bin.predict -h``

Once ``predict`` has been run for multiple models, statistical model comparison (permutation test) can be performed using ``compare``, as shown:

``python -m dtsr.bin.compare experiment.ini -p test``

The above call will permutation test pairwise differences in mean squared error on test data for all unique pairs of models defined in ``experiment.ini``.

In addition to these core utilities, the ``convolve`` convolves the input predictors using the fitted DTSR data transform and saves the data table, and ``make_plots`` generates IRF plots with some degree of customization afforded by the command line arguments.




