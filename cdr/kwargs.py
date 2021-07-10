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
            out += ' **Default (CDR)**: %s; **Default (CDRNN)**: %s\n' % (kwarg.default_value, kwarg.default_value_cdrnn)

    return out


def cdr_kwarg_docstring():
    """
    Generate docstring snippet summarizing all CDR kwargs, dtypes, and defaults.

    :return: ``str``; docstring snippet
    """

    out = "All models\n^^^^^^^^^^\n\n"

    for kwarg in MODEL_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += "\nAll CDR models\n^^^^^^^^^^^^^^\n\n"

    for kwarg in CDR_INITIALIZATION_KWARGS:
        if kwarg.key not in ['history_length']:
            out += docstring_from_kwarg(kwarg)

    out += '\nCDRMLE\n^^^^^^\n\n'

    for kwarg in CDRMLE_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\nCDRBayes\n^^^^^^^^\n\n'

    for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\nAll CDRNN models\n^^^^^^^^^^^^^^^^\n\n'

    for kwarg in CDRNN_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\nCDRNNMLE\n^^^^^^^^\n\n'

    for kwarg in CDRNNMLE_INITIALIZATION_KWARGS:
        out += docstring_from_kwarg(kwarg)

    out += '\nCDRNNBayes\n^^^^^^^^^^\n\n'

    for kwarg in CDRNNBAYES_INITIALIZATION_KWARGS:
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
        out += '- **%s**: %s; %s **Default:** %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr, kwarg.default_value)

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

    def kwarg_from_config(self, settings, is_cdrnn=False):
        """
        Given a settings object parsed from a config file, return value of kwarg cast to appropriate dtype.
        If missing from settings, return default.

        :param settings: settings from a ``ConfigParser`` object.
        :param is_cdrnn: ``bool``; whether this is for a CDRNN model.
        :return: value of kwarg
        """

        val = None

        if is_cdrnn:
            default_value = self.default_value_cdrnn
        else:
            default_value = self.default_value

        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, default_value)
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
        "Whether to model all parameters of the predictive distribution as dependent on IRFs of the impulses (distributional regression). If ``False``, only the mean depends on the predictors (other parameters of the predictive distribution are treated as constant).",
        aliases=['heteroskedastic'],
        default_value_cdrnn=True
    ),
    Kwarg(
        'predictive_distribution_map',
        None,
        [str, None],
        "Map defining predictive distribution. Can be a space-delimited list of distribution names (one per response variable), a space-delimited list of ';'-delimited tuples matching response variables to distribution names (e.g. ``response;Bernoulli``), or ``None``, in which case the predictive distribution will be inferred as ``Normal`` for continuous variables and ``Categorical`` for categorical variables."
    ),
    Kwarg(
        'center_inputs',
        False,
        bool,
        "DISCOURAGED UNLESS YOU HAVE A GOOD REASON, since this can distort rate estimates. Center inputs by subtracting training set means. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values."
    ),
    Kwarg(
        'rescale_inputs',
        False,
        bool,
        "Rescale input features by dividing by training set standard deviation. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values.",
        aliases=['scale_inputs'],
        default_value_cdrnn=True
    ),
    Kwarg(
        'standardize_response',
        True,
        bool,
        "Standardize (Z-transform) the response variable implicitly during training using training set mean and variance. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values."
    ),
    Kwarg(
        'history_length',
        None,
        int,
        "Length of the history window (in timesteps)."
    ),

    # MODEL DEFINITION
    Kwarg(
        'asymmetric_error',
        False,
        bool,
        "Whether to model numeric responses by default with an (asymmetric) SinhArcshin transform of the Normal distribution. Otherwise, defaults to a Normal distribution. Only affects response variables whose distributions have not been explicitly specified using **predictive_distribution_map**."
    ),
    Kwarg(
        'constraint',
        'softplus',
        str,
        "Constraint function to use for bounded variables. One of ``['abs', 'square', 'softplus']``."
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
        "Number of training iterations. If using variational inference, this becomes the `expected` number of training iterations and is used only for Tensorboard logging, with no impact on training behavior."
    ),
    Kwarg(
        'minibatch_size',
        1024,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        10000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'n_samples_eval',
        1000,
        int,
        "Number of posterior predictive samples to draw for prediction/evaluation. Ignored for evaluating CDR MLE models."
    ),
    Kwarg(
        'optim_name',
        'Nadam',
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
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.',
        default_value_cdrnn=1.,
    ),
    Kwarg(
        'epsilon',
        1e-5,
        float,
        "Epsilon parameter to use for numerical stability in bounded parameter estimation.",
        default_value_cdrnn=1e-2,
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
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        0.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_steps',
        25,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'loss_filter_n_sds',
        None,
        [float, None],
        "How many moving standard deviations above the moving mean of the loss to use as a cut-off for stability (suppressing large losses). If ``None``, or ``0``, no loss filtering.",
        default_value_cdrnn=1000.,
    ),
    Kwarg(
        'ema_decay',
        0.999,
        [float, None],
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
        "Stride (in iterations) over which to compute convergence. If larger than 1, iterations within a stride are averaged with the most recently saved value. Larger values increase the receptive field of the slope estimates, making convergence diagnosis less vulnerable to local perturbations but also increasing the number of post-convergence iterations necessary in order to identify convergence."
    ),
    Kwarg(
        'convergence_alpha',
        0.5,
        [float, None],
        "Significance threshold above which to fail to reject the null of no correlation between convergence basis and training time. Larger values are more stringent."
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
        'ranef_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of random effects regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization.",
        default_value_cdrnn='l2_regularizer'
    ),
    Kwarg(
        'ranef_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of random effects regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**.",
        default_value_cdrnn=10.
    ),

    # INCREMENTAL SAVING AND LOGGING
    Kwarg(
        'save_freq',
        100,
        int,
        "Frequency (in iterations) with which to save model checkpoints.",
        default_value_cdrnn=10
    ),
    Kwarg(
        'log_freq',
        100,
        int,
        "Frequency (in iterations) with which to log model params to Tensorboard.",
        default_value_cdrnn=1
    ),
    Kwarg(
        'log_random',
        True,
        bool,
        "Log random effects to Tensorboard."
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
        "Size of step by predictor to take above reference in univariate IRF plots. Structured as space-delimited pairs ``NAME=FLOAT``. Any predictor without a specified step size will step 1 SD from training set."
    ),
    Kwarg(
        'plot_step_default',
        1.,
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
        5,
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
        "Whether to include a legend in plots with multiple components."
    ),
    Kwarg(
        'generate_curvature_plots',
        False,
        bool,
        "Whether to plot IRF curvature at time **reference_time**.",
        default_value_cdrnn=True
    ),
    Kwarg(
        'generate_irf_surface_plots',
        False,
        bool,
        "Whether to plot IRF surfaces.",
        default_value_cdrnn=True
    ),
    Kwarg(
        'generate_interaction_surface_plots',
        True,
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
        "Whether to plot IRF surfaces showing non-stationarity in the response.",
        default_value_cdrnn=True
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

MODEL_BAYES_INITIALIZATION_KWARGS = [
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
    )
]

CDR_INITIALIZATION_KWARGS = [
    # REGULARIZATION
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

    # DEPRECATED OR RARELY USED
    Kwarg(
        'validate_irf_args',
        True,
        bool,
        "Check whether inputs and parameters to IRF obey constraints. Imposes a small performance cost but helps catch and report bugs in the model.",
        suppress=True
    )
]

CDRMLE_INITIALIZATION_KWARGS = []

CDRBAYES_INITIALIZATION_KWARGS = [
    # PRIORS
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

CDRNN_INITIALIZATION_KWARGS = [
    # DATA SETTINGS
    Kwarg(
        'center_time_X',
        False,
        bool,
        "Whether to center time values as inputs under the hood. Times are automatically shifted back to the source location for plotting and model criticism.",
        aliases=['center_time']
    ),
    Kwarg(
        'center_t_delta',
        False,
        bool,
        "Whether to center time offset values under the hood. Offsets are automatically shifted back to the source location for plotting and model criticism.",
        aliases=['center_time', 'center_tdelta']
    ),
    Kwarg(
        'rescale_time_X',
        True,
        bool,
        "Whether to rescale time values as inputs by their training SD under the hood. Times are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time']
    ),
    Kwarg(
        'rescale_t_delta',
        True,
        bool,
        "Whether to rescale time offset values by their training SD under the hood. Offsets are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time', 'rescale_tdelta']
    ),

    # MODEL SIZE
    Kwarg(
        'n_layers_input_projection',
        2,
        [int, None],
        "Number of hidden layers in input projection. If ``None``, inferred from length of **n_units_input_projection**."
    ),
    Kwarg(
        'n_units_input_projection',
        32,
        [int, str, None],
        "Number of units per input projection hidden layer. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_layers_rnn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers in input projection."
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
        'n_units_hidden_state',
        32,
        [int, str],
        "Number of units in CDRNN hidden state. Must be an ``int``."
    ),
    Kwarg(
        'n_layers_irf',
        2,
        [int, None],
        "Number of IRF hidden layers. If ``None``, inferred from length of **n_units_irf**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_irf',
        32,
        [int, str, None],
        "Number of units per hidden layer in IRF. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_irf** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers.",
        aliases=['n_units_decoder']
    ),

    # ACTIVATION FUNCTIONS
    Kwarg(
        'input_projection_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in input projection.",
        aliases=['activation']
    ),
    Kwarg(
        'input_projection_activation',
        None,
        [str, None],
        "Name of activation function to use for output of input projection."
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
        'hidden_state_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for CDRNN hidden state.",
        aliases=['activation']
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

    # NORMALIZATION
    Kwarg(
        'batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal layers. If ``None``, no batch normalization.",
    ),
    Kwarg(
        'normalize_input_projection',
        True,
        bool,
        "Whether to apply normalization (if applicable) to hidden layers of the input projection.",
    ),
    Kwarg(
        'normalize_h',
        True,
        bool,
        "Whether to apply normalization (if applicable) to the hidden state.",
    ),
    Kwarg(
        'normalize_irf_l1',
        False,
        bool,
        "Whether to apply normalization (if applicable) to the first IRF layer.",
    ),
    Kwarg(
        'normalize_irf',
        False,
        bool,
        "Whether to apply normalization (if applicable) to non-initial internal IRF layers.",
    ),
    Kwarg(
        'normalize_after_activation',
        True,
        bool,
        "Whether to apply normalization (if applicable) after the non-linearity (otherwise, applied before).",
    ),
    Kwarg(
        'normalization_use_gamma',
        True,
        bool,
        "Whether to use trainable scale in batch/layer normalization layers.",
        aliases=['batch_normalization_use_gamma', 'layer_normalization_use_gamma']
    ),
    Kwarg(
        'layer_normalization_type',
        None,
        [str, None],
        "Type of layer normalization, one of ``['z', 'length', None]``. If ``'z'``, classical z-transform-based normalization. If ``'length'``, normalize by the norm of the activation vector. If ``None``, no layer normalization. Incompatible with batch normalization.",
    ),

    # REGULARIZATION
    Kwarg(
        'nn_regularizer_name',
        'l2_regularizer',
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
        'context_regularizer_name',
        'l2_regularizer',
        [str, 'inherit', None],
        "Name of regularizer on contribution of context (RNN) to hidden state (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'context_regularizer_scale',
        1.,
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
        'input_projection_dropout_rate',
        0.2,
        [float, None],
        "Rate at which to drop neurons of input projection layers.",
        aliases=['dropout_rate']
    ),
    Kwarg(
        'rnn_h_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of RNN hidden state.",
        # aliases=['dropout_rate']
    ),
    Kwarg(
        'rnn_c_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of RNN cell state.",
        # aliases=['dropout_rate']
    ),
    Kwarg(
        'h_in_dropout_rate',
        0.2,
        [float, None],
        "Rate at which to drop neurons of h_in.",
        # aliases=['dropout_rate']
    ),
    Kwarg(
        'h_rnn_dropout_rate',
        0.2,
        [float, None],
        "Rate at which to drop neurons of h_rnn.",
        # aliases=['dropout_rate']
    ),
    Kwarg(
        'h_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of h.",
        aliases=['dropout_rate']
    ),
    Kwarg(
        'rnn_dropout_rate',
        None,
        [float, None],
        "Rate at which to entirely drop the RNN.",
        aliases=['dropout_rate']
    ),
    Kwarg(
        'irf_dropout_rate',
        0.2,
        [float, None],
        "Rate at which to drop neurons of IRF layers.",
        aliases=['dropout_rate']
    ),
    Kwarg(
        'ranef_dropout_rate',
        0.2,
        [float, None],
        "Rate at which to drop random effects indicators.",
        aliases=['dropout_rate']
    ),

    # DEPRECATED OR RARELY USED
    Kwarg(
        'rnn_type',
        'LSTM',
        str,
        "**DEPRECATED** (only LSTM is supported).",
        suppress=True
    ),
    Kwarg(
        'forget_rate',
        None,
        [float, None],
        "Rate at which to drop recurrent connection entirely.",
        suppress=True
    ),
    Kwarg(
        'input_jitter_level',
        None,
        [float, None],
        "Standard deviation of jitter injected into inputs (predictors and timesteps) during training. If ``0`` or ``None``, no input jitter.",
        suppress=True
    ),
    Kwarg(
        'h_in_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into h_in.",
        suppress=True
    ),
    Kwarg(
        'h_rnn_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into h_rnn.",
        suppress=True
    ),
]

CDRNNMLE_INITIALIZATION_KWARGS = [
    # INITIALIZATION
    Kwarg(
        'weight_sd_init',
        'glorot',
        [float, str],
        "Standard deviation of kernel initialization distribution (Normal, mean=0). Can also be ``'glorot'``, which uses the SD of the Glorot normal initializer."
    )
]

CDRNNBAYES_INITIALIZATION_KWARGS = [
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
        False,
        bool,
        "Specify Gaussian priors for model biases (if ``False``, use implicit improper uniform priors)."
    ),
    Kwarg(
        'declare_priors_gamma',
        False,
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
        'glorot',
        [str, float],
        "Standard deviation of prior on batch norm gammas. A ``float``, ``'glorot'``, or ``'he'``. Ignored unless batch normalization is used",
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
        'weight_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over weights. If ``None``, inferred from other hyperparams."
    ),
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
    ),
    Kwarg(
        'posterior_to_prior_sd_ratio',
        0.01,
        float,
        "Ratio of posterior initialization SD to prior SD. Low values are often beneficial to stability, convergence speed, and quality of final fit by avoiding erratic sampling and divergent behavior early in training."
    )
]

PLOT_KWARGS_CORE = [
    # PLOT DATA GENERATION
    Kwarg(
        'resvar',
        'y_mean',
        str,
        "Name of parameter of predictive distribution to plot as response variable. One of ``'y_mean'``, ``'y_sd'``, ``'y_skewness'``, or ``'y_tailweight'``. Only ``'y_mean'`` is interesting for CDR, since the others are assumed scalar. CDRNN fits all predictive parameters via IRFs."
    ),
    Kwarg(
        'generate_univariate_IRF_plots',
        True,
        bool,
        "Whether to plot univariate IRFs over time."
    ),
    Kwarg(
        'generate_curvature_plots',
        True,
        bool,
        "Whether to plot IRF curvature at time **reference_time**."
    ),
    Kwarg(
        'generate_irf_surface_plots',
        True,
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
        False,
        bool,
        "Whether to plot the average error distribution for real-valued responses."
    ),
    Kwarg(
        'generate_nonstationarity_surface_plots',
        True,
        bool,
        "Whether to plot IRF surfaces showing non-stationarity in the response."
    ),
    Kwarg(
        'n_samples',
        None,
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
        'reference_time',
        0.,
        float,
        "Timepoint at which to plot interactions."
    ),
    Kwarg(
        'standardize_response',
        False,
        bool,
        "Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``."
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
        2.5,
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
        True,
        bool,
        "Whether to add legend to univariate IRF plots."
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
        'prefix',
        None,
        [str, None],
        "Name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot)."
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
