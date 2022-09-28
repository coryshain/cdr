.. _config:

CDR Configuration Files
========================

The CDR utilities in this module read config files that follow the INI standard.
Basic information about INI file syntax can be found e.g. `here <https://en.wikipedia.org/wiki/INI_file>`_.
This reference assumes familiarity with the INI protocol.

CDR configuration files contain the sections and fields described below.



Section: ``[data]``
-------------------

The ``[data]`` section supports the following fields:

**REQUIRED**

- **X_train**: ``str``; Path to training data (predictor matrix)
- **y_train**: ``str``; Path to training data (response matrix)
- **series_ids**: space-delimited list of ``str``; Names of columns used to define unique time series

Note that, unlike e.g. linear models, CDR does not require synchronous predictors and responses, which is why separate data objects must be provided for each of these components.
If the predictors and responses are synchronous, this is fine.
The ``X_train`` and ``y_train`` fields can point to the same file.
The system will treat each unique combination of values in the columns given in ``series_ids`` as constituting a unique time series.

**OPTIONAL**

- **X_dev**: ``str``; Path to dev data (predictor matrix)
- **y_dev**: ``str``; Path to dev data (response matrix)
- **X_test**: ``str``; Path to test data (predictor matrix)
- **y_test**: ``str``; Path to test data (response matrix)
- **history_length**: ``int``; Length of history window in timesteps (default: ``128``)
- **filters**: ``str``; List of filters to apply to response data (``;``-delimited).
All variables used in a filter must be contained in the data files indicated by the ``y_*`` parameters in the ``[data]`` section of the config file.
The variable name is specified as an INI field, and the condition is specified as its value.
Supported logical operators are ``<``, ``<=``, ``>``, ``>=``, ``==``, and ``!=``.
For example, to keep only data points for which column ``foo`` is less or equal to 100, the following filter can be added::

    filters = foo <= 100

To keep only data points for which the column ``foo`` does not equal ``bar``, the following filter can be added::

    filters = foo != bar

Filters can be conjunctively combined::

    filters = foo > 5; foo <= 100

Count-based filters are also supported, using the designated ``nunique`` suffix.
For example, if the column ``subject`` is being used to define a random effects level and we want to exclude all subjects with fewer than 100 data points, this can be accomplished using the following filter::

    filters = subjectsnunique > 100

More complex filtration conditions are not supported automatically in CDR but can be applied to the data by the user as a preprocess.

Several CDR utilities (e.g. for prediction and evaluation) are designed to handle train, dev, and test partitions of the input data, but these partitions must be constructed in advance.
This package also provides a ``partition`` utility that can be used to partition input data by applying modular arithmetic to some subset of the variables in the data.
For usage details run:

``python -m cdr.bin.partition -h``

**IMPORTANT NOTES**

- The files indicated in ``X_*`` must contain the following columns:

  - **time**: Timestamp associated with each observation
  - A column for each variable in ``series_ids``
  - A column for each predictor variable indicated in the model formula

- The file in ``y_*`` must contain the following columns:

  - **time**: Timestamp associated with each observation
  - A column for the response variable in the model formula
  - A column for each variable in ``series_ids``
  - A column for each random grouping factor in the in the model formula
  - A column for each variable used for data filtration (see below)

- Data in ``y_*`` may be filtered/partitioned, but data in ``X_*`` **must be uncensored** unless independent reason exists to assume that certain observations never have an impact on the response.



Section: ``[global_settings]``
------------------------------
The ``[global_settings]`` section supports the following fields:

- **outdir**: ``str``; Path to output directory where checkpoints, plots, and Tensorboard logs should be saved (default: ``./cdr_model/``).
  If it does not exist, this directory will be created.
  At runtime, the ``train`` utility will copy the config file to this directory as ``config.ini``, serving as a record of the settings used to generate the analysis.
- **use_gpu_if_available**: ``bool``; If available, run on GPU. If ``False``, always runs on CPU even when system has compatible GPU.



Section: ``[cdr_settings]``
----------------------------

The ``[cdr_settings]`` section supports the following fields:

.. exec::
    from cdr.kwargs import cdr_kwarg_docstring
    print(cdr_kwarg_docstring())



Section: ``[irf_name_map]``
---------------------------

The optional ``[irf_name_map]`` section simply permits prettier variable naming in plots.
For example, the internal name for a convolution applied to predictor ``A`` may be ``ShiftedGammaKgt1.s(A)-Terminal.s(A)``, which is not very readable.
To address this, the string above can be mapped to a more readable name using an INI key-value pair, as shown::

    ShiftedGammaKgt1.s(A)-Terminal.s(A) = A

The model will then print ``A`` in plots rather than ``ShiftedGammaKgt1.s(A)-Terminal.s(A)``.
Unused entries in the name map are ignored, and model variables that do not have an entry in the name map print with their default internal identifier.



Sections: ``[model_CDR_*]``
----------------------------

Arbitrarily many sections named ``[model_CDR_*]`` can be provided in the config file, where ``*`` stands in for a unique identifier.
Each such section defines a different CDR model and must contain at least one field --- ``formula`` --- whose value is a CDR model formula (see :ref:`formula` for more on CDR formula syntax)
The identifier ``CDR_*`` will be used by the CDR utilities to reference the fitted model and its output files.

For example, to define a CDR model called ``readingtimes``, the section header ``[model_CDR_readingtimes]`` is included in the config file along with an appropriate ``formula`` specification.
To use this specific model once fitted, it can be referenced using the identifier ``CDR_readingtimes``.
For example, the following call will extract predictions on dev data from a fitted ``CDR_readingtimes`` defined in config file **config.ini**::

    python -m cdr.bin.predict config.ini -m CDR_readingtimes -p dev

Additional fields from ``[cdr_settings]`` may be specified for a given model, in which case the locally-specified setting (rather than the globally specified setting or the default value) will be used to train the model.
For example, imagine that ``[cdr_settings]`` contains the field ``n_iter = 1000``.
All CDR models subsequently specified in the config file will train for 1000 iterations.
However, imagine that model ``[model_CDR_longertrain]`` should train for 5000 iterations instead.
This can be specified within the same config file as::

    [model_CDR_longertrain]
    n_iter = 5000
    formula = ...

This setup allows a single config file to define a variety of CDR models, as long as they all share the same data.
Distinct datasets require distinct config files.

For hypothesis testing, fixed effect ablation can be conveniently automated using the ``ablate`` model field.
For example, the following specification implicitly defines 7 unique models, one for each of the ``|powerset(a, b, c)| - 1 = 7``
non-null ablations of ``a``, ``b``, and ``c``::

    [model_CDR_example]
    n_iter = 5000
    ablate = a b c
    formula = C(a + b + c, Normal()) + (C(a + b + c, Normal()) | subject)

The ablated models are named using ``'!'`` followed by the name for each ablated predictor.
Therefore, the above specification is equivalent to (and much easier to write than) the following::

    [model_CDR_example]
    n_iter = 5000
    formula = C(a + b + c, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!a]
    n_iter = 5000
    formula = C(b + c, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!b]
    n_iter = 5000
    formula = C(a + c, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!c]
    n_iter = 5000
    formula = C(a + b, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!a!b]
    n_iter = 5000
    formula = C(c, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!a!c]
    n_iter = 5000
    formula = C(b, Normal()) + (C(a + b + c, Normal()) | subject)

    [model_CDR_example!b!c]
    n_iter = 5000
    formula = C(a, Normal()) + (C(a + b + c, Normal()) | subject)