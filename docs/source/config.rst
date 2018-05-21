.. _config:

DTSR Configuration File Reference
=================================

The DTSR utilities in this module read config files that follow the INI standard.
Basic information about INI file syntax can be found e.g. `here <https://en.wikipedia.org/wiki/INI_file>`_.
This reference assumes familiarity with the INI protocol.

DTSR configuration files contain the sections and fields described below.


Section: ``[data]``
-------------------

The ``[data]`` section supports the following fields:

**REQUIRED**

- **X_train**: ``str``; Path to training data (impulse matrix)
- **y_train**: ``str``; Path to training data (response matrix)
- **series_ids**: space-delimited list of ``str``; Names of columns used to define unique time series

Note that, unlike e.g. linear models, DTSR does not require synchronous predictors (impulses) and responses, which is why separate data objects must be provided for each of these components.
If the predictors and responses are synchronous, this is fine.
The ``X_train`` and ``y_train`` fields can point to the same file.
The system will treat each unique combination of values in the columns given in ``series_ids`` as constituting a unique time series.

**OPTIONAL**

- **X_dev**: ``str``; Path to dev data (impulse matrix)
- **y_dev**: ``str``; Path to dev data (response matrix)
- **X_test**: ``str``; Path to test data (impulse matrix)
- **y_test**: ``str``; Path to test data (response matrix)

Several DTSR utilities (e.g. for prediction and evaluation) are designed to handle train, dev, and test partitions of the input data, but these partitions must be constructed in advance.
This package also provides a ``partition`` utility that can be used to partition input data by applying modular arithmetic to some subset of the variables in the data.
For usage details run:

``python -m dtsr.bin.partition -h``

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




Section: ``[settings]``
-----------------------

The ``[settings]`` section supports the following fields:

**Both DTSRMLE and DTSRBayes**

- **outdir**: ``str``; Path to output directory where checkpoints, plots, and Tensorboard logs should be saved (default: ``./dtsr_model/``).
  If it does not exist, this directory will be created.
  At runtime, the ``train`` utility will copy the config file to this directory as ``config.ini``, serving as a record of the settings used to generate the analysis.
- **network_type**: ``str``; Type of inference to use (one of ``bayes`` or ``mle``, default: ``bayes``)
- **history_length**: ``int`` or ``None``; Maximum number of preceding timepoints to use from the history (if ``None``, no clipping; default: ``128``)
- **low_memory**: ``bool``; Whether to use the low memory setting for deconvolution (temporarily discontinued, only ``False`` is supported)
- **init_sd**: ``float``; Standard deviation of parameter initialization distribution (default: ``0.1``)
- **ema_decay**: ``str``; Decay rate to use in the exponential moving average of parameters, used for prediction (default: ``0.999``)
- **optim**: ``str``; Name of optimizer to use, one of the following (default: ``Adam``)

  - ``SGD``
  - ``Momentum``
  - ``AdaGrad``
  - ``AdaDelta``
  - ``Adam``
  - ``FTRL``
  - ``RMSProp``
  - ``Nadam``
  - ``None`` (only supported for ``DTSRBayes`` and not recommended; uses the default optimizer defined by Edward, which currently includes steep learning rate decay)

- **learning_rate**: ``float``; Learning rate (default: ``0.001``)
- **lr_decay_family**: ``str`` or ``None``; Name of learning rate decay family to use; supports all available decays in Tensorflow's ``train`` module, or no decay if ``None`` (default: ``None``)
  **Note**: Decaying the learning rate can give a false impression of convergence, since stability in model parameters can be artificially induced by a vanishing learning rate.
- **learning_rate_min**: ``float`` or ``None``; Minimum learning rate (no minimum if ``None``), ignored if ``lr_decay_family`` is ``None`` (default: ``1e-5``)
- **lr_decay_steps**: ``int``; Decay rate for learning rate, ignored if **lr_decay_family** is ``None`` (default: ``0.5``)
- **lr_decay_rate**: ``float``; Number of training epochs per decay level, ignored if **lr_decay_family** is ``None`` (default: ``100``)
- **lr_decay_staircase**: ``bool``; Whether to staircase the learning rate decay (default: ``False)
- **n_iter**: ``int``; Number of training epochs (default: ``1000``)
- **minibatch_size**: ``int`` or ``None``; Size of training minibatches, full batch training if ``None`` (default: ``1024``)
- **eval_minibatch_size**: ``int`` or ``None``; Size of evaluation minibatches, full batch evaluation if ``None`` (default: ``100000``)
- **n_interp**: ``int``; Number of interpolation points to use for approximate deconvolution, only used if the model formula flags at least one input as continuous (see :ref:`formula`, default: ``64``)
- **float_type**: ``str``; Type of floating point representation to use (default: ``float32``)
- **int_type**: ``str``; Type of integer representation to use (default: ``int32``)
- ``use_gpu_if_available``; ``bool``; Whether to use GPU if available (default: ``True``)
- **log_freq**: ``int``; Frequency (in epochs) with which to write Tensorboard logs during training (default: ``1``)
- **pc**: ``bool``; Whether to use principle components regression (experimental; default: ``False``)
- **save_freq**: ``int``; Frequency (in epochs) with which to save model checkpoints and plots during training (default: ``1``)
- **log_random**: ``bool``; Whether to write Tensorboard logs for random effects (default: ``True``)
- **plot_n_time_units**: ``float``; Number of time units to include in IRF plots (default: ``2.5``)
- **plot_n_points_per_time_unit**: ``float``; Number of plot points to write per time unit in IRF plots (default: ``500``)
- **plot_x_inches**: ``float``; Width of IRF plots in inches (default: ``500``)
- **plot_y_inches**: ``float``; Height of IRF plots in inches (default: ``500``)
- **cmap**: ``str``; Name of ``matplotlib`` colormap scheme to use for plotting (default: ``500``)

**DTSRMLE only**

- **loss**: ``str``; Name of loss to use (one of ``mse`` or ``mae``; default: ``mse``)
- **regularizer**: ``str`` or ``None``; Name of regularizer to use; supports all regularizer layers in Tensorflow's ``contrib.layers`` module, or no regularization if ``None`` (default: ``None``)
- **regularizer_scale**: ``float``; Regularization constant; ignored if **regularizer** is ``None`` (default: ``0.01``)

**DTSRBayes only**

- **inference_name**: ``str``; Name of inference to use; supports most inferences provided by Edward (default: ``KLqp``)
- **n_samples**: ``int`` or ``None``; Number of samples to use, use Edward defaults if ``None``. If using MCMC, the number of samples is set deterministically as ``n_iter * n_minibatch``, so this user-supplied parameter is ignored (default: ``1``)
- **n_samples_eval**: ``int`` or ``None``; Number of samples to use for evaluation, can be overridden by DTSR evaluation utilities (default: ``128``)
- **y_scale**: ``float`` or ``None``; Fixed value for the standard deviation of the output distribution, or ``None`` to fit this as a parameter (default: ``None``)
- **intercept_prior_sd**: ``float``; Standard deviation of prior on the intercept (default: ``1``)
- **coef_prior_sd**: ``float``; Standard deviation of prior on the model coefficients (default: ``1``)
- **conv_prior_sd**: ``float``; Standard deviation of prior on the IRF parameters (default: ``1``)
- **y_scale_prior_sd**: ``float``; Standard deviation of prior on the standard deviation of the output distribution, ignored if **y_scale** is not ``None`` (default: ``1``)
- **mh_proposal_sd**: ``float``; Standard deviation of the proposal distribution for Metropolis-Hastings inference, ignored unless **inference_name** is ``MetropolisHastings`` (default: ``1``)
- **mv**: ``bool``; Whether to use a MVN prior on fixed effects (otherwise fixed effects priors are independent normal, default: ``False``)
- **mv_ran**: ``bool``; Whether to use a MVN prior on random effects (otherwise random effects priors are independent normal, default: ``False``)
- **asymmetric_error**: ``boolean``; Whether to apply the ``SinhArcsinh`` transform to the normal error, allowing fitting of skewness and tailweight (default: ``False``)



Section: ``[filters]``
----------------------

The optional ``[filters]`` section allows specification of simple data censoring, which will be applied only to the vector of regression targets.
All variables used in a filter must be contained in the data files indicated by the ``y_*`` parameters in the ``[data]`` section of the config file.
The variable name is specified as an INI field, and the condition is specified as its value.
Supported logical operators are ``<``, ``<=``, ``>``, ``>=``, ``==``, and ``!=``.
For example, to keep only data points for which column ``foo`` is less or equal to 100, the following filter can be added:

``foo = <= 100``

To keep only data points for which the column ``foo`` does not equal ``bar``, the following filter can be added:

``foo = != bar``

More complex filtration conditions are not supported automatically in DTSR but can be applied to the data by the user as a preprocess.



Section: ``[irf_name_map]``
---------------------------

The optional ``[irf_name_map]`` section simply permits prettier variable naming in plots.
For example, the internal name for a convolution applied to predictor ``A`` may be ``ShiftedGammaKgt1.s(A)-Terminal.s(A)``, which is not very readable.
To address this, the string above can be mapped to a more readable name using an INI key-value pair, as shown:

``ShiftedGammaKgt1.s(A)-Terminal.s(A) = A``

The model will then print ``A`` in plots rather than ``ShiftedGammaKgt1.s(A)-Terminal.s(A)``.
Unused entries in the name map are ignored, and model variables that do not have an entry in the name map print with their default internal identifier.



Sections: ``[model_DTSR_*]``
----------------------------

Arbitrarily many sections named ``[model_DTSR_*]`` can be provided in the config file, where ``*`` stands in for a unique identifier.
Each such section defines a different DTSR model and must contain exactly one field --- ``formula`` --- whose value is a DTSR model formula (see :ref:`formula` for more on DTSR formula syntax)
The identifier ``DTSR_*`` will be used by the DTSR utilities to reference the fitted model and its output files.

For example, to define a DTSR model called ``readingtimes``, the section header ``[model_DTSR_readingtimes]`` is included in the config file along with an appropriate ``formula`` specification.
To use this specific model once fitted, it can be referenced using the identifier ``DTSR_readingtimes``.
For example, the following call will extract predictions on dev data from a fitted ``DTSR_readingtimes`` defined in config file **config.ini**:

``python -m dtsr.bin.predict config.ini -m DTSR_readingtimes -p dev``

