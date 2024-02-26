from functools import cmp_to_key


def docstring_from_kwarg(kwarg):
    """
    Generate docstring from CDR keyword argument object.

    :param kwarg: Keyword argument object.
    :return: ``str``; docstring.
    """

    out = ''
    if not kwarg.suppress:
        out += '- **%s**: %s; %s' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)
        if kwarg.default_value == kwarg.default_value_cdrnn:
            out += ' **Default**: %s\n' % kwarg.default_value
        else:
            out += ' **Default (CDR)**: %s; **Default (CDRNN)**: %s\n' \
                   % (kwarg.default_value, kwarg.default_value_cdrnn)

    return out


def cdr_kwarg_docstring():
    """
    Generate docstring snippet summarizing all CDR kwargs, dtypes, and defaults.

    :return: ``str``; docstring snippet
    """

    out = "All Models\n^^^^^^^^^^\n\n"

    for kwarg in MODEL_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += "\nVariational Bayes\n^^^^^^^^^^^^^^^^^\n\n"

    for kwarg in BAYES_KWARGS:
        if kwarg.key not in ['history_length', 'future_length']:
            out += docstring_from_kwarg(kwarg)

    out += '\nNeural Network Components\n^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'

    for kwarg in NN_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\nVariational Bayesian Neural Network Components\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'

    for kwarg in NN_BAYES_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\n'

    return out


def plot_kwarg_docstring():
    """
    Generate docstring snippet summarizing all plotting kwargs, dtypes, and defaults.

    :return: ``str``; docstring snippet
    """

    out = "**Plotting options**\n\n"

    for kwarg in PLOT_KWARGS_CORE + PLOT_KWARGS_OTHER:
        out += '- **%s**: %s; %s **Default:** %s\n' % \
               (kwarg.key, kwarg.dtypes_str(), kwarg.descr, kwarg.default_value)

    out += '\n'

    return out


class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    :param default_value_cdrnn: Any; Default value for CDRNN if distinct from CDR. If ``'same'``, CDRNN uses **default_value**.
    :param suppress: ``bool``; Whether to print documentation for this kwarg. Useful for hiding deprecated or little-used kwargs in order to simplify autodoc output.
    """

    def __init__(
            self,
            key,
            default_value,
            dtypes,
            descr,
            aliases=None,
            default_value_cdrnn='same',
            suppress=False
    ):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if default_value_cdrnn == 'same':
            self.default_value_cdrnn = self.default_value
        else:
            self.default_value_cdrnn = default_value_cdrnn
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases
        self.suppress = suppress

    def dtypes_str(self):
        """
        String representation of dtypes permitted for kwarg.

        :return: ``str``; dtypes string.
        """

        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        """
        String representation of name of a dtype

        :param x: dtype; the dtype to name.
        :return: ``str``; name of dtype.
        """

        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        """
        Check whether kwarg is specified in a settings object parsed from a config file.

        :param settings: settings from a ``ConfigParser`` object.
        :return: ``bool``; whether kwarg is found in **settings**.
        """

        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def get_typed_val(self, key, dtype, settings, default=None):
        if key in settings:
            if isinstance(settings, dict):
                if dtype == str:
                    return settings.get(key)
                elif dtype == int:
                    return int(settings.get(key))
                elif dtype == float:
                    return float(settings.get(key))
                elif dtype == bool:
                    return bool(settings.get(key))
                else:
                    raise ValueError('Unrecognized dtype: %s' % dtype)
            else:
                if dtype == str:
                    return settings.get(key)
                elif dtype == int:
                    return settings.getint(key)
                elif dtype == float:
                    return settings.getfloat(key)
                elif dtype == bool:
                    return settings.getboolean(key)
                else:
                    raise ValueError('Unrecognized dtype: %s' % dtype)
        return default


    def kwarg_from_config(self, settings, is_cdrnn=False):
        """
        Given a settings object parsed from a config file, return value of kwarg cast to appropriate dtype.
        If missing from settings, return default.

        :param settings: settings from a ``ConfigParser`` object or ``dict``.
        :param is_cdrnn: ``bool``; whether this is for a CDRNN model.
        :return: value of kwarg
        """

        val = None

        if is_cdrnn:
            default_value = self.default_value_cdrnn
        else:
            default_value = self.default_value

        if len(self.dtypes) == 1:

            val = self.get_typed_val(self.key, self.dtypes[0], settings, default=None)

            if val is None:
                for alias in self.aliases:
                    val = self.get_typed_val(alias, self.dtypes[0], settings, default=default_value)
                    if val is not None:
                        break

            if val is None:
                val = default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except ValueError:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val


    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``

        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0


MODEL_INITIALIZATION_KWARGS = [
    # DATA SETTINGS
    Kwarg(
        'outdir',
        './cdr_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'use_distributional_regression',
        False,
        bool,
        "Whether to model all parameters of the response distribution as dependent on IRFs of the impulses (distributional regression). If ``False``, only the mean depends on the predictors (other parameters of the response distribution are treated as constant).",
        aliases=['heteroscedastic', 'heteroskedastic'],
        default_value_cdrnn=True
    ),
    Kwarg(
        'response_distribution_map',
        None,
        [str, None],
        "Definition of response distribution. Can be a single distribution name (shared across all response variables), a space-delimited list of distribution names (one per response variable), a space-delimited list of ';'-delimited tuples matching response variables to distribution names (e.g. ``response;Bernoulli``), or ``None``, in which case the response distribution will be inferred as ``JohnsonSU`` for continuous variables and ``Categorical`` for categorical variables.",
        aliases=['response_distribution', 'predictive_distribution_map', 'predictive_distribution', 'pred_dist']
    ),
    Kwarg(
        'center_inputs',
        False,
        bool,
        "DISCOURAGED UNLESS YOU HAVE A GOOD REASON, since this can distort rate estimates. Center inputs by subtracting training set means. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values."
    ),
    Kwarg(
        'rescale_inputs',
        True,
        bool,
        "Rescale input features by dividing by training set standard deviation. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values.",
        aliases=['scale_inputs'],
        default_value_cdrnn=True
    ),
    Kwarg(
        'history_length',
        128,
        int,
        "Length of the history (backward) window (in timesteps)."
    ),
    Kwarg(
        'future_length',
        0,
        int,
        "Length of the future (forward) window (in timesteps). Note that causal IRF kernels cannot be used if **future_length** > 0."
    ),
    Kwarg(
        't_delta_cutoff',
        None,
        [float, None],
        "Maximum distance in time to consider (can help improve training stability on data with large gaps in time). If ``0`` or ``None``, no cutoff."
    ),

    # MODEL DEFINITION
    Kwarg(
        'constraint',
        'softplus',
        str,
        "Constraint function to use for bounded variables. One of ``['abs', 'square', 'softplus']``."
    ),
    Kwarg(
        'random_variables',
        'default',
        str,
        "Space-delimited list of model components to instantiate as (variationally optimized) random variables, rather than point estimates. Can be any combination of: `['intercept', 'coefficient', 'interaction', 'irf_param', 'nn']`. Can also be `'all'`, `'none'`, or `'default'`, which defaults to all components except `'nn'`.",
        aliases=['rvs', 'as_rv', 'as_random_variable']
    ),

    # OPTIMIZATION SETTINGS
    Kwarg(
        'scale_loss_with_data',
        True,
        bool,
        "Whether to multiply the scale of the LL loss by N, where N is num batches. This turns the loss into an expectation over training set likelihood."
    ),
    Kwarg(
        'scale_regularizer_with_data',
        False,
        bool,
        "Whether to multiply the scale of all weight regularization by B * N, where B is batch size and N is num batches. If **scale_loss_with_data** is true, this approach ensures a stable regularization strength (relative to the loss) across datasets and batch sizes."
    ),
    Kwarg(
        'n_iter',
        100000,
        int,
        "Number of training iterations. If using variational inference, this becomes the `expected` number of training iterations and is used only for Tensorboard logging, with no impact on training behavior.",
        aliases=['niter']
    ),
    Kwarg(
        'minibatch_size',
        1024,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``).",
        aliases=['batch_size']
    ),
    Kwarg(
        'eval_minibatch_size',
        1024,
        int,
        "Size of minibatches to use for prediction/evaluation."
    ),
    Kwarg(
        'n_samples_eval',
        1000,
        int,
        "Number of posterior predictive samples to draw for prediction/evaluation. Ignored for evaluating CDR MLE models."
    ),
    Kwarg(
        'optim_name',
        'Adam',
        [str, None],
        """Name of the optimizer to use. Must be one of:
        
            - ``'SGD'``
            - ``'Momentum'``
            - ``'AdaGrad'``
            - ``'AdaDelta'``
            - ``'Adam'``
            - ``'FTRL'``
            - ``'RMSProp'``
            - ``'Nadam'``"""
    ),
    Kwarg(
        'max_gradient',
        None,
        [float, None],
        'Maximum allowable value for the gradient, which will be clipped as needed. If ``None``, no max gradient.'
    ),
    Kwarg(
        'max_global_gradient_norm',
        1.,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no max global norm for the gradient.'
    ),
    Kwarg(
        'use_safe_optimizer',
        False,
        bool,
        'Stabilize training by preventing the optimizer from applying updates involving NaN gradients (affected weights will remain unchanged after the update). Incurs slight additional computational overhead and can lead to bias in the training process.'
    ),
    Kwarg(
        'epsilon',
        1e-5,
        float,
        "Epsilon parameter to use for numerical stability in bounded parameter estimation (imposes a positive lower bound on the parameter)."
    ),
    Kwarg(
        'response_dist_epsilon',
        1e-5,
        float,
        "Epsilon parameter to use for numerical stability in bounded parameters of the response distribution (imposes a positive lower bound on the parameter).",
        aliases=['pred_dist_epsilon', 'epsilon']
    ),
    Kwarg(
        'optim_epsilon',
        1e-8,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.001,
        float,
        "Initial value for the learning rate.",
        default_value_cdrnn=0.01
    ),
    Kwarg(
        'learning_rate_min',
        0.,
        float,
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        [str, None],
        "Functional family for the learning rate decay schedule (no decay if ``None``).",
        aliases=['learning_rate_decay_family']
    ),
    Kwarg(
        'lr_decay_rate',
        1.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``).",
        aliases=['learning_rate_decay_rate']
    ),
    Kwarg(
        'lr_decay_steps',
        100,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``).",
        aliases=['learning_rate_decay_steps']
    ),
    Kwarg(
        'lr_decay_iteration_power',
        0.5,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay.",
        aliases=['learning_rate_decay_iteration_power']
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``).",
        aliases=['learning_rate_decay_staircase']
    ),
    Kwarg(
        'filter_outlier_losses',
        False,
        [float, bool, None],
        "Whether outlier large losses are filtered out while training continues. If ``False``, outlier losses trigger a restart from the most recent save point. Ignored unless *loss_cutoff_n_sds* is specified. Using this option avoids restarts, but can lead to bias if training instances are systematically dropped. If ``None``, ``False``, or ``0``, no loss filtering.",
        aliases=['loss_filter_n_sds']
    ),
    Kwarg(
        'loss_cutoff_n_sds',
        1000,
        [float, None],
        "How many moving standard deviations above the moving mean of the loss to use as a cut-off for stability (if outlier large losses are detected, training restarts from the preceding checkpoint). If ``None``, or ``0``, no loss cut-off."
    ),
    Kwarg(
        'ema_decay',
        0.999,
        float,
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),

    # CONVERGENCE
    Kwarg(
        'convergence_n_iterates',
        500,
        [int, None],
        "Number of timesteps over which to average parameter movements for convergence diagnostics. If ``None`` or ``0``, convergence will not be programmatically checked (reduces memory overhead, but convergence must then be visually diagnosed).",
        default_value_cdrnn=100
    ),
    Kwarg(
        'convergence_stride',
        1,
        int,
        "Stride (in iterations) over which to compute convergence. If larger than 1, iterations within a stride are averaged with the most recently saved value. Larger values increase the receptive field of the slope estimates, making convergence diagnosis less vulnerable to local perturbations but also increasing the number of post-convergence iterations necessary in order to identify convergence. If ``early_stopping`` is ``True``, ``convergence_stride`` will implicitly be multiplied by ``eval_freq``."
    ),
    Kwarg(
        'convergence_alpha',
        0.5,
        [float, None],
        "Significance threshold above which to fail to reject the null of no correlation between convergence basis and training time. Larger values are more stringent."
    ),
    Kwarg(
        'early_stopping',
        True,
        bool,
        "Whether to diagnose convergence based on dev set performance (``True``) or training set performance (``False``)."
    ),

    # REGULARIZATION
    Kwarg(
        'regularizer_name',
        None,
        [str, None],
        "Name of global regularizer; can be overridden by more regularizers for more specific parameters (e.g. ``l1_regularizer``, ``l2_regularizer``). If ``None``, no regularization."
    ),
    Kwarg(
        'regularizer_scale',
        0.,
        [str, float],
        "Scale of global regularizer; can be overridden by more regularizers for more specific parameters (ignored if ``regularizer_name==None``)."
    ),
    Kwarg(
        'intercept_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of intercept regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'intercept_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of intercept regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'coefficient_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of coefficient regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'coefficient_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of coefficient regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'irf_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of IRF parameter regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'irf_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of IRF parameter regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'ranef_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of random effects regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization. Regularization only applies to random effects without variational priors.",
        default_value_cdrnn='l2_regularizer'
    ),
    Kwarg(
        'ranef_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of random effects regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**. Regularization only applies to random effects without variational priors.",
        default_value_cdrnn=100.
    ),
    Kwarg(
        'regularize_mean',
        False,
        bool,
        "Mean-aggregate regularized variables. If ``False``, use sum aggregation."
    ),

    # INCREMENTAL SAVING AND LOGGING
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'plot_freq',
        10,
        int,
        "Frequency (in iterations) with which to plot model estimates (or ``0`` to turn off incremental plotting)."
    ),
    Kwarg(
        'eval_freq',
        10,
        int,
        "Frequency (in iterations) with which to evaluate on dev data (or ``0`` to turn off incremental evaluation)."
    ),
    Kwarg(
        'log_freq',
        1,
        int,
        "Frequency (in iterations) with which to log model params to Tensorboard."
    ),
    Kwarg(
        'log_fixed',
        True,
        bool,
        "Log random fixed to Tensorboard. Can slow training of models with many fixed effects."
    ),
    Kwarg(
        'log_random',
        True,
        bool,
        "Log random effects to Tensorboard. Can slow training of models with many random effects."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    ),

    # PLOTTING
    Kwarg(
        'indicator_names',
        '',
        str,
        "Space-delimited list of predictors that are indicators (0 or 1). Used for plotting and effect estimation (value 0 is always used as reference, rather than mean)."
    ),
    Kwarg(
        'default_reference_type',
        0.,
        ['mean', 0.],
        "Reference stimulus to use by default for plotting and effect estimation. If `0`, zero vector. If `mean`, training set mean by predictor.",
        default_value_cdrnn='mean'
    ),
    Kwarg(
        'reference_values',
        '',
        str,
        "Predictor values to use as a reference in plotting and effect estimation. Structured as space-delimited pairs ``NAME=FLOAT``. Any predictor without a specified reference value will use either 0 or the training set mean, depending on **plot_mean_as_reference**."
    ),
    Kwarg(
        'plot_step',
        '',
        str,
        "Size of step by predictor to take above reference in univariate IRF plots. Structured as space-delimited pairs ``NAME=FLOAT``. Any predictor without a specified step size will inherit from **plot_step_default**."
    ),
    Kwarg(
        'plot_step_default',
        'sd',
        [str, float],
        "Default size of step to take above reference in univariate IRF plots, if not specified in **plot_step**. Either a float or the string ``'sd'``, which indicates training sample standard deviation."
    ),
    Kwarg(
        'reference_time',
        0.,
        float,
        "Timepoint at which to plot interactions."
    ),
    Kwarg(
        'plot_n_time_units',
        1,
        float,
        "Number of time units to use for plotting."
    ),
    Kwarg(
        'plot_n_time_points',
        1024,
        int,
        "Resolution of plot axis (for 3D plots, uses sqrt of this number for each axis)."
    ),
    Kwarg(
        'plot_dirac',
        False,
        bool,
        "Whether to include any Dirac delta IRF's (stick functions at t=0) in plot."
    ),
    Kwarg(
        'plot_x_inches',
        6.,
        float,
        "Width of plot in inches."
    ),
    Kwarg(
        'plot_y_inches',
        4.,
        float,
        "Height of plot in inches."
    ),
    Kwarg(
        'plot_legend',
        True,
        bool,
        "Whether to include a legend in plots with multiple components.",
        aliases=['use_legend', 'legend']
    ),
    Kwarg(
        'generate_univariate_irf_plots',
        True,
        bool,
        "Whether to plot univariate IRFs over time.",
        aliases=['generate_univariate_IRF_plots']
    ),
    Kwarg(
        'generate_univariate_irf_heatmaps',
        False,
        bool,
        "Whether to plot univariate IRF heatmaps over time.",
        aliases=['generate_univariate_IRF_heatmaps']
    ),
    Kwarg(
        'generate_curvature_plots',
        True,
        bool,
        "Whether to plot IRF curvature at time **reference_time**."
    ),
    Kwarg(
        'generate_irf_surface_plots',
        False,
        bool,
        "Whether to plot IRF surfaces."
    ),
    Kwarg(
        'generate_interaction_surface_plots',
        False,
        bool,
        "Whether to plot IRF interaction surfaces at time **reference_time**."
    ),
    Kwarg(
        'generate_err_dist_plots',
        True,
        bool,
        "Whether to plot the average error distribution for real-valued responses."
    ),
    Kwarg(
        'generate_nonstationarity_surface_plots',
        False,
        bool,
        "Whether to plot IRF surfaces showing non-stationarity in the response."
    ),
    Kwarg(
        'cmap',
        'gist_rainbow',
        str,
        "Name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot)."
    ),
    Kwarg(
        'dpi',
        300,
        int,
        "Dots per inch of saved plot image file."
    ),
    Kwarg(
        'keep_plot_history',
        False,
        bool,
        "Keep IRF plots from each checkpoint of a run, which can help visualize learning trajectories but can also consume a lot of disk space. If ``False``, only the most recent plot of each type is kept."
    ),

    # DEPRECATED OR RARELY USED
    Kwarg(
        'validate_irf_args',
        True,
        bool,
        "Check whether inputs and parameters to IRF obey constraints. Imposes a small performance cost but helps catch and report bugs in the model.",
        suppress=True
    ),
    Kwarg(
        'use_jtps',
        False,
        bool,
        "Whether to modify the base optimizer using JTPS.",
        suppress=True
    ),
    Kwarg(
        'n_samples',
        1,
        int,
        "**DEPRECATED**",
        suppress=True
    ),
    Kwarg(
        'n_interp',
        1000,
        int,
        "Number of interpolation points to use for discrete approximation of composed IRF. Ignored unless IRF composition is required by the model.",
        suppress=True
    ),
    Kwarg(
        'interp_hz',
        1000,
        float,
        "Frequency in Hz at which to interpolate continuous impulses for approximating the IRF integral.",
        suppress=True
    ),
    Kwarg(
        'interp_step',
        0.1,
        float,
        "Step length for resampling from interpolated continuous predictors.",
        suppress=True
    ),
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network.",
        suppress=True
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing).",
        suppress=True
    )
]

BAYES_KWARGS = [
    # PRIORS
    Kwarg(
        'declare_priors_fixef',
        True,
        bool,
        "Specify Gaussian priors for all fixed model parameters (if ``False``, use implicit improper uniform priors).",
        aliases=['declare_priors']
    ),
    Kwarg(
        'declare_priors_ranef',
        True,
        bool,
        "Specify Gaussian priors for all random model parameters (if ``False``, use implicit improper uniform priors).",
        aliases=['declare_priors']
    ),
    Kwarg(
        'intercept_prior_sd',
        None,
        [str, float, None],
        "Standard deviation of prior on fixed intercept. Can be a space-delimited list of ``;``-delimited floats (one per distributional parameter per response variable), a ``float`` (applied to all responses), or ``None``, in which case the prior is inferred from **prior_sd_scaling_coefficient** and the empirical variance of the response on the training set.",
        aliases=['prior_sd']
    ),
    Kwarg(
        'coef_prior_sd',
        None,
        [str, float, None],
        "Standard deviation of prior on fixed coefficients. Can be a space-delimited list of ``;``-delimited floats (one per distributional parameter per response variable), a ``float`` (applied to all responses), or ``None``, in which case the prior is inferred from **prior_sd_scaling_coefficient** and the empirical variance of the response on the training set.",
        aliases=['coefficient_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'irf_param_prior_sd',
        1.,
        [str, float],
        "Standard deviation of prior on convolutional IRF parameters. Can be either a space-delimited list of ``;``-delimited floats (one per distributional parameter per response variable) or a ``float`` (applied to all responses)",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'y_sd_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on standard deviation of output model. If ``None``, inferred as **y_sd_prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['y_scale_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'prior_sd_scaling_coefficient',
        1,
        float,
        "Factor by which to multiply priors on intercepts and coefficients if inferred from the empirical variance of the data (i.e. if **intercept_prior_sd** or **coef_prior_sd** is ``None``). Ignored for any prior widths that are explicitly specified."
    ),
    Kwarg(
        'y_sd_prior_sd_scaling_coefficient',
        1,
        float,
        "Factor by which to multiply prior on output model variance if inferred from the empirical variance of the data (i.e. if **y_sd_prior_sd** is ``None``). Ignored if prior width is explicitly specified.",
        aliases=['y_scale_prior_sd_scaling_coefficient', 'prior_sd_scaling_coefficient']
    ),
    Kwarg(
        'ranef_to_fixef_prior_sd_ratio',
        0.1,
        float,
        "Ratio of widths of random to fixed effects priors. I.e. if less than 1, random effects have tighter priors."
    ),

    # INITIALIZATION
    Kwarg(
        'posterior_to_prior_sd_ratio',
        0.01,
        float,
        "Ratio of posterior initialization SD to prior SD. Low values are often beneficial to stability, convergence speed, and quality of final fit by avoiding erratic sampling and divergent behavior early in training."
    )
]

NN_KWARGS = [
    # DATA SETTINGS
    Kwarg(
        'center_X_time',
        False,
        bool,
        "Whether to center time values as inputs under the hood. Times are automatically shifted back to the source location for plotting and model criticism.",
        aliases=['center_time', 'center_time_X']
    ),
    Kwarg(
        'center_t_delta',
        False,
        bool,
        "Whether to center time offset values under the hood. Offsets are automatically shifted back to the source location for plotting and model criticism.",
        aliases=['center_time', 'center_tdelta']
    ),
    Kwarg(
        'rescale_X_time',
        True,
        bool,
        "Whether to rescale time values as inputs by their training SD under the hood. Times are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time', 'rescale_time_X'],
        default_value_cdrnn=True
    ),
    Kwarg(
        'rescale_t_delta',
        False,
        bool,
        "Whether to rescale time offset values by their training SD under the hood. Offsets are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time', 'rescale_tdelta']
    ),
    Kwarg(
        'nn_use_input_scaler',
        False,
        bool,
        "Whether to apply a Hadamard scaling layer to the inputs to any NN components.",
        aliases=['use_input_scaler']
    ),
    Kwarg(
        'log_transform_t_delta',
        False,
        bool,
        "Whether to log-modulus transform time offset values for stability under the hood (log-modulus is used to handle negative values in non-causal models). Offsets are automatically reconverted back to the source scale for plotting and model criticism."
    ),
    Kwarg(
        'nonstationary',
        True,
        bool,
        "Whether to model non-stationarity in NN components by feeding impulse timestamps as input."
    ),

    # MODEL SIZE
    Kwarg(
        'n_layers_ff',
        2,
        [int, None],
        "Number of hidden layers in feedforward encoder. If ``None``, inferred from length of **n_units_ff**.",
        aliases=['n_layers', 'n_layers_encoder', 'n_layers_input_projection']
    ),
    Kwarg(
        'n_units_ff',
        128,
        [int, str, None],
        "Number of units per feedforward encoder hidden layer. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_layers_rnn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no feedforward encoder.",
        aliases=['n_units', 'n_units_encoder', 'n_units_input_projection']
    ),
    Kwarg(
        'n_layers_rnn',
        None,
        [int, None],
        "Number of RNN layers. If ``None``, inferred from length of **n_units_rnn**."
    ),
    Kwarg(
        'n_units_rnn',
        None,
        [int, str, None],
        "Number of units per RNN layer. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_layers_rnn** space-delimited integers, one for each layer in order from bottom to top. Can also be ``'infer'``, which infers the size from the number of predictors, or ``'inherit'``, which uses size **n_units_hidden_state**. If ``0`` or ``None``, no RNN encoding (i.e. use a context-independent convolution kernel)."
    ),
    Kwarg(
        'n_layers_rnn_projection',
        None,
        [int, None],
        "Number of hidden layers in projection of RNN state (or of timestamp + predictors if no RNN). If ``None``, inferred automatically."
    ),
    Kwarg(
        'n_units_rnn_projection',
        None,
        [int, str, None],
        "Number of units per hidden layer in projection of RNN state. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_rnn_projection** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers in RNN projection."
    ),
    Kwarg(
        'n_layers_irf',
        2,
        [int, None],
        "Number of IRF hidden layers. If ``None``, inferred from length of **n_units_irf**.",
        aliases=['n_layers', 'n_layers_decoder']
    ),
    Kwarg(
        'n_units_irf',
        128,
        [int, str, None],
        "Number of units per hidden layer in IRF. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_irf** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers.",
        aliases=['n_units', 'n_units_decoder']
    ),
    Kwarg(
        'input_dependent_irf',
        True,
        bool,
        "Whether or not NN IRFs are input-dependent (can modify their shape at different values of the predictors)."
    ),
    Kwarg(
        'ranef_l1_only',
        False,
        bool,
        "Whether to include random effects only on first layer of feedforward transforms (``True``) or on all neural components."
    ),
    Kwarg(
        'ranef_bias_only',
        True,
        bool,
        "Whether to include random effects only on bias terms of neural components (``True``) or also on weight matrices.",
        aliases=['ranef_biases_only']
    ),
    Kwarg(
        'normalizer_use_ranef',
        False,
        bool,
        "Whether to include random effects in normalizer layers (``True``) or not."
    ),

    # ACTIVATION FUNCTIONS
    Kwarg(
        'ff_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in feedforward encoder.",
        aliases=['activation', 'input_projection_inner_activation']
    ),
    Kwarg(
        'ff_activation',
        None,
        [str, None],
        "Name of activation function to use for output of feedforward encoder.",
        aliases=['input_projection_activation']
    ),
    Kwarg(
        'rnn_activation',
        'tanh',
        [str, None],
        "Name of activation to use in RNN layers.",
    ),
    Kwarg(
        'recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of recurrent activation to use in RNN layers.",
    ),
    Kwarg(
        'rnn_projection_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in projection of RNN state.",
        aliases=['activation']
    ),
    Kwarg(
        'rnn_projection_activation',
        None,
        [str, None],
        "Name of activation function to use for final layer in projection of RNN state."
    ),
    Kwarg(
        'irf_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in IRF.",
        aliases=['activation']
    ),
    Kwarg(
        'irf_activation',
        None,
        [str, None],
        "Name of activation function to use for final layer in IRF."
    ),

    # INITIALIZATION
    Kwarg(
        'kernel_initializer',
        'glorot_uniform_initializer',
        [str, None],
        "Name of initializer to use in encoder kernels.",
    ),
    Kwarg(
        'recurrent_initializer',
        'orthogonal_initializer',
        [str, None],
        "Name of initializer to use in encoder recurrent kernels.",
    ),
    Kwarg(
        'weight_sd_init',
        'glorot',
        [str, float, None],
        "Standard deviation of kernel initialization distribution (Normal, mean=0). Can also be ``'glorot'``, which uses the SD of the Glorot normal initializer. If ``None``, inferred from other hyperparams."
    ),

    # NORMALIZATION
    Kwarg(
        'batch_normalization_decay',
        None,
        [bool, float, None],
        "Decay rate to use for batch normalization in internal layers. If ``True``, uses decay ``0.999``. If ``False`` or ``None``, no batch normalization.",
        aliases=['batch_normalization', 'batch_norm']
    ),
    Kwarg(
        'layer_normalization_type',
        'z',
        [bool, str, None],
        "Type of layer normalization, one of ``['z', 'length', None]``. If ``'z'``, classical z-transform-based normalization. If ``'length'``, normalize by the norm of the activation vector. If ``True``, uses ``'z'``. If ``False`` or ``None``, no layer normalization.",
        aliases=['layer_normalization', 'layer_norm']
    ),
    Kwarg(
        'normalize_ff',
        True,
        bool,
        "Whether to apply normalization (if applicable) to hidden layers of feedforward encoders.",
        aliases=['normalize_input_projection']
    ),
    Kwarg(
        'normalize_irf',
        True,
        bool,
        "Whether to apply normalization (if applicable) to non-initial internal IRF layers.",
    ),
    Kwarg(
        'normalize_after_activation',
        False,
        bool,
        "Whether to apply normalization (if applicable) after the non-linearity (otherwise, applied before).",
    ),
    Kwarg(
        'shift_normalized_activations',
        True,
        bool,
        "Whether to use trainable shift in batch/layer normalization layers.",
        aliases=['normalization_use_beta', 'batch_normalization_use_beta', 'layer_normalization_use_beta']
    ),
    Kwarg(
        'rescale_normalized_activations',
        True,
        bool,
        "Whether to use trainable scale in batch/layer normalization layers.",
        aliases=['normalization_use_gamma', 'batch_normalization_use_gamma', 'layer_normalization_use_gamma']
    ),
    Kwarg(
        'normalize_inputs',
        False,
        bool,
        "Whether to apply normalization (if applicable) to the inputs.",
    ),
    Kwarg(
        'normalize_final_layer',
        False,
        bool,
        "Whether to apply normalization (if applicable) to the final layer.",
    ),

    # REGULARIZATION
    Kwarg(
        'nn_regularizer_name',
        None,
        [str, 'inherit', None],
        "Name of weight regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'nn_regularizer_scale',
        1.,
        [str, float, 'inherit'],
        "Scale of weight regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'activity_regularizer_name',
        None,
        [str, 'inherit', None],
        "Name of activity regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no activity regularization."
    ),
    Kwarg(
        'activity_regularizer_scale',
        5.,
        [str, float, 'inherit'],
        "Scale of activity regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'ff_regularizer_name',
        None,
        [str, None],
        "Name of weight regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``) on output layer of feedforward encoders; overrides **regularizer_name**. If ``None``, inherits from **nn_regularizer_name**.",
        aliases=['input_projection_regularizer_name']
    ),
    Kwarg(
        'ff_regularizer_scale',
        5.,
        [str, float],
        "Scale of weight regularizer (ignored if ``regularizer_name==None``) on output layer of feedforward encoders. If ``None``, inherits from **nn_regularizer_scale**.",
        aliases=['input_projection_regularizer_scale']
    ),
    Kwarg(
        'regularize_initial_layer',
        True,
        bool,
        "Whether to regulare the first layer of NN components."
    ),
    Kwarg(
        'regularize_final_layer',
        False,
        bool,
        "Whether to regulare the last layer of NN components."
    ),
    Kwarg(
        'rnn_projection_regularizer_name',
        None,
        [str, None],
        "Name of weight regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``) on output layer of RNN projection; overrides **regularizer_name**. If ``None``, inherits from **nn_regularizer_name**."
    ),
    Kwarg(
        'rnn_projection_regularizer_scale',
        5.,
        [str, float],
        "Scale of weight regularizer (ignored if ``regularizer_name==None``) on output layer of RNN projection. If ``None``, inherits from **nn_regularizer_scale**."
    ),
    Kwarg(
        'context_regularizer_name',
        'l1_l2_regularizer',
        [str, 'inherit', None],
        "Name of regularizer on contribution of context (RNN) to hidden state (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'context_regularizer_scale',
        10.,
        [float, 'inherit'],
        "Scale of weight regularizer (ignored if ``context_regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'maxnorm',
        None,
        [float, None],
        "Bound on norm of dense kernel dimensions for max-norm regularization. If ``None``, no max-norm regularization."
    ),

    # DROPOUT
    Kwarg(
        'input_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop input_features."
    ),
    Kwarg(
        'ff_dropout_rate',
        0.5,
        [float, None],
        "Rate at which to drop neurons of FF projection.",
        aliases=['dropout', 'dropout_rate', 'input_projection_dropout_rate', 'h_in_dropout_rate']
    ),
    Kwarg(
        'rnn_h_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of RNN hidden state."
    ),
    Kwarg(
        'rnn_c_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of RNN cell state."
    ),
    Kwarg(
        'h_rnn_dropout_rate',
        0.5,
        [float, None],
        "Rate at which to drop neurons of h_rnn.",
        aliases=['dropout', 'dropout_rate']
    ),
    Kwarg(
        'rnn_dropout_rate',
        0.5,
        [float, None],
        "Rate at which to entirely drop the RNN.",
        aliases=['dropout', 'dropout_rate']
    ),
    Kwarg(
        'irf_dropout_rate',
        0.5,
        [float, None],
        "Rate at which to drop neurons of IRF layers.",
        aliases=['dropout', 'dropout_rate']
    ),
    Kwarg(
        'ranef_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop random effects indicators.",
        aliases=['dropout', 'dropout_rate']
    ),
    Kwarg(
        'dropout_final_layer',
        False,
        bool,
        "Whether to apply dropout to the last layer of NN components."
    ),
    Kwarg(
        'fixed_dropout',
        True,
        bool,
        "Whether to fix the dropout mask over the time dimension during training, " +
        "ensuring that each training instance is processed by the same resampled model."
    ),

    # DEPRECATED OR RARELY USED
    Kwarg(
        'input_jitter_level',
        None,
        [float, None],
        "Standard deviation of jitter injected into inputs (predictors and timesteps) during training. " +
        "If ``0`` or ``None``, no input jitter.",
        suppress=True
    ),
    Kwarg(
        'ff_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into FF projection.",
        suppress=True
    ),
    Kwarg(
        'h_rnn_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into h_rnn.",
        suppress=True
    )
]


NN_BAYES_KWARGS = [
    # PRIORS
    Kwarg(
        'declare_priors_weights',
        True,
        bool,
        "Specify Gaussian priors for all fixed model parameters (if ``False``, use implicit improper uniform priors).",
        aliases=['declare_priors']
    ),
    Kwarg(
        'declare_priors_biases',
        True,
        bool,
        "Specify Gaussian priors for model biases (if ``False``, use implicit improper uniform priors)."
    ),
    Kwarg(
        'declare_priors_gamma',
        True,
        bool,
        "Specify Gaussian priors for gamma parameters of any batch normalization layers (if ``False``, use implicit improper uniform priors).",
        aliases=['declare_priors']
    ),
    Kwarg(
        'weight_prior_sd',
        'glorot',
        [str, float],
        "Standard deviation of prior on CDRNN hidden weights. A ``float``, ``'glorot'``, or ``'he'``.",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'bias_prior_sd',
        1.,
        [str, float],
        "Standard deviation of prior on CDRNN hidden biases. A ``float``, ``'glorot'``, or ``'he'``.",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'gamma_prior_sd',
        1,
        [str, float],
        "Standard deviation of prior on batch norm gammas. A ``float``, ``'glorot'``, or ``'he'``. Ignored unless batch normalization is used",
        aliases=['conv_prior_sd', 'prior_sd']
    ),

    # INITIALIZATION
    Kwarg(
        'bias_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over biases. If ``None``, inferred from other hyperparams."
    ),
    Kwarg(
        'gamma_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over batch norm gammas. If ``None``, inferred from other hyperparams. Ignored unless batch normalization is used."
    )
]

MODEL_INITIALIZATION_KWARGS += BAYES_KWARGS + NN_KWARGS + NN_BAYES_KWARGS

PLOT_KWARGS_CORE = [
    # PLOT DATA GENERATION
    Kwarg(
        'responses',
        None,
        [str, None],
        "Name(s) of response variable(s) to plot. If ``None``, plots all univariate responses."
    ),
    Kwarg(
        'response_params',
        None,
        [str, None],
        "Name(s) of parameter(s) of response distribution to plot for each response variable. If ``None``, plots the first parameter only. Parameter names not present in a given distribution will be skipped."
    ),
    Kwarg(
        'generate_univariate_irf_plots',
        None,
        [bool, None],
        "Whether to plot univariate IRFs over time. If ``None``, use model defaults.",
        aliases=['generate_univariate_IRF_plots']
    ),
    Kwarg(
        'generate_univariate_irf_heatmaps',
        None,
        [bool, None],
        "Whether to plot univariate IRF heatmaps over time. If ``None``, use model defaults.",
        aliases=['generate_univariate_IRF_heatmaps']
    ),
    Kwarg(
        'generate_curvature_plots',
        None,
        [bool, None],
        "Whether to plot IRF curvature at time **reference_time**. If ``None``, use model defaults."
    ),
    Kwarg(
        'generate_irf_surface_plots',
        None,
        [bool, None],
        "Whether to plot IRF surfaces. If ``None``, use model defaults.",
        aliases=['generate_IRF_surface_plots']
    ),
    Kwarg(
        'generate_interaction_surface_plots',
        None,
        [bool, None],
        "Whether to plot IRF interaction surfaces at time **reference_time**. If ``None``, use model defaults."
    ),
    Kwarg(
        'generate_err_dist_plots',
        None,
        [bool, None],
        "Whether to plot the average error distribution for real-valued responses. If ``None``, use model defaults."
    ),
    Kwarg(
        'generate_nonstationarity_surface_plots',
        None,
        [bool, None],
        "Whether to plot IRF surfaces showing non-stationarity in the response. If ``None``, use model defaults."
    ),
    Kwarg(
        'n_samples',
        1000,
        [int, None],
        "Number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults."
    ),
    Kwarg(
        'level',
        95.,
        float,
        "Significance level for confidence/credible intervals, if supported."
    ),
    Kwarg(
        'plot_rangf',
        False,
        bool,
        "Whether to plot all (marginal) random effects."
    ),
    Kwarg(
        'plot_step',
        None,
        [str, None],
        "Size of step by predictor to take above reference in univariate IRF plots. Structured as space-delimited pairs ``NAME=FLOAT``. Any predictor without a specified step size will inherit from **plot_step_default**. If ``None``, use model defaults."
    ),
    Kwarg(
        'plot_step_default',
        None,
        [str, float, None],
        "Default size of step to take above reference in univariate IRF plots, if not specified in **plot_step**. Either a float or the string ``'sd'``, which indicates training sample standard deviation. If ``None``, use model defaults."
    ),
    Kwarg(
        'reference_time',
        None,
        [float, None],
        "Timepoint at which to plot interactions. If ``None``, use model defaults."
    ),
    Kwarg(
        'reference_type',
        None,
        [str, None],
        "Type of plotting reference to use. One of ``'sampling'``, ``'mean'``, ``'zero'``, or ``None`` for default."
    ),
    Kwarg(
        'plot_quantile_range',
        0.9,
        float,
        "Quantile range to use for plotting. E.g., 0.9 uses the interdecile range."
    ),
    Kwarg(
        'x_axis_transform',
        None,
        [str, None],
        "String description of transform to apply to x-axis prior to plotting. Currently supported: ``'exp'``, ``'log'``, ``'neglog'``. If ``None``, no x-axis transform."
    ),
    Kwarg(
        'y_axis_transform',
        None,
        [str, None],
        "String description of transform to apply to y-axis (in 3d plots only) prior to plotting. Currently supported: ``'exp'``, ``'log'``, ``'neglog'``. If ``None``, no y-axis transform."
    ),

    # CONTROLS FOR UNIVARIATE IRF PLOTS
    Kwarg(
        'pred_names',
        None,
        [str, None],
        "List of names of predictors to include in univariate IRF plots. If ``None``, all predictors are plotted."
    ),
    Kwarg(
        'sort_names',
        True,
        bool,
        "Whether to alphabetically sort IRF names."
    ),
    Kwarg(
        'prop_cycle_length',
        None,
        [int, None],
        "Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **pred_names**."
    ),
    Kwarg(
        'prop_cycle_map',
        None,
        [str, None],
        "Integer indices to use in the properties cycle for each entry in **pred_names**. Can be (1) a space-delimited list of ``;``-delimited pairs mapping from predictor names to ``int``; (2) a space-delimited list of ``int`` which is assumed to align one-to-one with predictor names, or (3) ``None``, in which case indices are automatically assigned."
    ),
    Kwarg(
        'plot_dirac',
        False,
        bool,
        "Whether to include any Dirac delta IRF's (stick functions at t=0) in plot."
    ),

    # AESTHETICS
    Kwarg(
        'plot_n_time_units',
        None,
        [float, None],
        "Number of time units to use for plotting. If ``None``, use model defaults."
    ),
    Kwarg(
        'plot_n_time_points',
        None,
        [int, None],
        "Resolution of plot axis (for 3D plots, uses sqrt of this number for each axis). If ``None``, use model defaults."
    ),
    Kwarg(
        'plot_x_inches',
        None,
        [float, None],
        "Width of plot in inches. If ``None``, use model defaults."
    ),
    Kwarg(
        'plot_y_inches',
        None,
        [float, None],
        "Height of plot in inches. If ``None``, use model defaults"
    ),
    Kwarg(
        'ylim',
        None,
        [str, None],
        "Space-delimited ``lower_bound upper_bound`` to use for y axis. If ``None``, automatically inferred."
    ),
    Kwarg(
        'use_horiz_axlab',
        True,
        bool,
        "Whether to include horizontal axis label(s) (x axis in 2D plots, x/y axes in 3D plots)."
    ),
    Kwarg(
        'use_vert_axlab',
        True,
        bool,
        "Whether to include vertical axis label (y axis in 2D plots, z axis in 3D plots)."
    ),
    Kwarg(
        'use_legend',
        None,
        [bool, None],
        "Whether to add legend to univariate IRF plots. If ``None``, use model defaults.",
        aliases=['legend']
    ),
    Kwarg(
        'use_line_markers',
        False,
        bool,
        "Whether to add markers to lines in univariate IRF plots."
    ),
    Kwarg(
        'transparent_background',
        False,
        bool,
        "Whether to use a transparent background. If ``False``, uses a white background."
    ),
    Kwarg(
        'cmap',
        None,
        [str, None],
        "Name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot). If ``None``, use model defaults."
    ),
    Kwarg(
        'dpi',
        None,
        [int, None],
        "Dots per inch of saved plot image file. If ``None``, use model defaults."
    ),
    Kwarg(
        'prefix',
        None,
        [str, None],
        "Prefix string to prepend to plot image files."
    ),
    Kwarg(
        'suffix',
        '.png',
        str,
        "File extension to use for plot outputs."
    ),
    Kwarg(
        'key',
        None,
        [str, None],
        "Any additional string key to add to the filename to distinguish the plot."
    )
]

PLOT_KWARGS_OTHER = [
    # SYNTHETIC DATA
    Kwarg(
        'plot_true_synthetic',
        False,
        bool,
        "If the models are fitted to synthetic data, whether to additionally generate plots of the true IRF."
    ),

    # QUANTILE-QUANTILE PLOTS
    Kwarg(
        'qq_partition',
        None,
        [str, None],
        "Partition over which to generate Q-Q plot for errors. Ignored if ``None`` or model directory does not contain saved errors for the requested partition."
    ),
    Kwarg(
        'qq_use_axis_labels',
        False,
        bool,
        "Whether to add axis labels to Q-Q plots."
    ),
    Kwarg(
        'qq_use_ticks',
        False,
        bool,
        "Whether to add ticks to Q-Q plots."
    ),
    Kwarg(
        'qq_use_legend',
        False,
        bool,
        "Whether to add legend to Q-Q plots."
    )
]
