from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

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

    def kwarg_from_config(self, settings):
        """
        Given a settings object parsed from a config file, return value of kwarg cast to appropriate dtype.
        If missing from settings, return default.

        :param settings: settings from a ``ConfigParser`` object.
        :return: value of kwarg
        """

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
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
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
    Kwarg(
        'outdir',
        './cdr_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
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
        aliases=['scale_inputs']
    ),
    Kwarg(
        'standardize_response',
        True,
        bool,
        "Standardize (Z-transform) the response variable using training set mean and variance. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- prediction, likelihood computation, and plotting are reported on the source values."
    ),
    Kwarg(
        'asymmetric_error',
        False,
        bool,
        "Allow an asymmetric error distribution by fitting a SinArcsinh transform of the normal error, adding trainable skewness and tailweight parameters."
    ),
    Kwarg(
        'history_length',
        None,
        int,
        "Length of the history window to use."
    ),
    Kwarg(
        'n_iter',
        100000,
        int,
        "Number of training iterations. If using variational inference, this becomes the `expected` number of training iterations and is used only for Tensorboard logging, with no impact on training behavior."
    ),
    Kwarg(
        'n_samples',
        1,
        int,
        "For MLE models, number of samples from joint distribution (used only if either **covarying_fixef** or **covarying_ranef** is ``True``, otherwise ignored). For BBVI models, number of posterior samples to draw at each training step during variational inference. If using MCMC inference, the number of samples is set deterministically as ``n_iter * n_minibatch``, so this user-supplied parameter is overridden."
    ),
    Kwarg(
        'n_interp',
        1000,
        int,
        "Number of interpolation points to use for discrete approximation of composed IRF. Ignored unless IRF composition is required by the model."
    ),
    Kwarg(
        'interp_hz',
        1000,
        float,
        "Frequency in Hz at which to interpolate continuous impulses for approximating the IRF integral."
    ),
    Kwarg(
        'interp_step',
        0.1,
        float,
        "Step length for resampling from interpolated continuous predictors."
    ),
    Kwarg(
        'intercept_init',
        None,
        [float, None],
        "Initial value to use for the intercept (if ``None``, use mean response in training data)"
    ),
    Kwarg(
        'y_sd_init',
        None,
        [float, None],
        "Initial value for the standard deviation of the output model. If ``None``, inferred as the empirical standard deviation of the response on the training set.",
        aliases=['y_scale_init']
    ),
    Kwarg(
        'constraint',
        'softplus',
        str,
        "Constraint function to use for bounded variables. One of ``['abs', 'square', 'softplus']``."
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
        'use_jtps',
        False,
        bool,
        "Whether to modify the base optimizer using JTPS."
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'epsilon',
        1e-8,
        float,
        "Epsilon parameter to use for numerical stability in bounded parameter estimation."
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
        "Initial value for the learning rate."
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
        "How many moving standard deviations above the moving mean of the loss to use as a cut-off for stability (suppressing large losses). If ``None``, or ``0``, no loss filtering."
    ),
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
        'scale_loss_with_data',
        False,
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
        "Name of random effects regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'ranef_regularizer_scale',
        'inherit',
        [str, float, 'inherit'],
        "Scale of random effects regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'ema_decay',
        0.999,
        [float, None],
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
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
    # Kwarg(
    #     'normalize_error_params_fn',
    #     False,
    #     bool,
    #     "Whether to apply normalization (if applicable) to internal layers of the error parameter function.",
    # ),
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
    Kwarg(
        'convergence_n_iterates',
        500,
        [int, None],
        "Number of timesteps over which to average parameter movements for convergence diagnostics. If ``None`` or ``0``, convergence will not be programmatically checked (reduces memory overhead, but convergence must then be visually diagnosed)."
    ),
    Kwarg(
        'convergence_stride',
        1,
        int,
        "Stride (in iterations) over which to compute convergence. If larger than 1, iterations within a stride are averaged with the most recently saved value. Larger values increase the receptive field of the slope estimates, making convergence diagnosis less vulnerable to local perturbations but also increasing the number of post-convergence iterations necessary in order to identify convergence."
    ),
    Kwarg(
        'convergence_basis',
        'loss',
        str,
        "Basis of convergence diagnostic, one of ``['parameters', 'loss']``. If ``parameters``, slopes of all parameters with respect to time must be within the tolerance of 0. If ``loss``, slope of loss with respect to time must be within the tolerance of 0 (even if parameters are still moving). The loss-based criterion is less stringent."
    ),
    Kwarg(
        'convergence_alpha',
        0.5,
        [float, None],
        "Significance threshold above which to fail to reject the null of no correlation between convergence basis and training time. Larger values are more stringent."
    ),
    Kwarg(
        'minibatch_size',
        1024,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        100000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'log_freq',
        1,
        int,
        "Frequency (in iterations) with which to log model params to Tensorboard."
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
    Kwarg(
        'indicator_names',
        '',
        str,
        "Space-delimited list of predictors that are indicators (0 or 1). Used for plotting and effect estimation (value 0 is always used as reference, rather than mean)."
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
        'plot_interactions',
        '',
        str,
        "Space-delimited list of all implicit interactions to plot."
    ),
    Kwarg(
        'reference_time',
        0.,
        float,
        "Timepoint at which to plot interactions."
    ),
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
        'plot_legend',
        True,
        bool,
        "Whether to include a legend in plots with multiple components."
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
    Kwarg(
        'pc',
        False,
        bool,
        "Transform input variables using principal components analysis. BROKEN, DO NOT USE."
    )
]


CDR_INITIALIZATION_KWARGS = [
    Kwarg(
        'covarying_fixef',
        False,
        bool,
        "Use multivariate model that fits covariances between fixed parameters. Experimental, not thoroughly tested. If ``False``, fixed parameter distributions are treated as independent.",
        aliases=['mv']
    ),
    Kwarg(
        'covarying_ranef',
        False,
        bool,
        "Use multivariate model that fits covariances between random parameters within a random grouping factor. Experimental, not thoroughly tested. If ``False``, random parameter distributions are treated as independent.",
        aliases=['mv_ran']
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
        'validate_irf_args',
        True,
        bool,
        "Check whether inputs and parameters to IRF obey constraints. Imposes a small performance cost but helps catch and report bugs in the model."
    ),
    Kwarg(
        'default_reference_type',
        0.,
        ['mean', 0.],
        "Reference stimulus to use by default for plotting and effect estimation. If `0`, zero vector. If `mean`, training set mean by predictor."
    ),
    Kwarg(
        'generate_curvature_plots',
        False,
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
        'generate_nonstationarity_surface_plots',
        False,
        bool,
        "Whether to plot IRF surfaces showing non-stationarity in the response."
    )
]

CDRMLE_INITIALIZATION_KWARGS = [
    Kwarg(
        'intercept_joint_sd',
        None,
        [float, None],
        "Square root of variance of intercept in initial variance-covariance matrix of joint distributions. Used only if either **covarying_fixef** or **covarying_ranef** is ``True``, otherwise ignored. If ``None``, inferred as **joint_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['intercept_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'coef_joint_sd',
        None,
        [float, None],
        "Square root of variance of coefficients in initial variance-covariance matrix of joint distributions. Used only if either **covarying_fixef** or **covarying_ranef** is ``True``, otherwise ignored. If ``None``, inferred as **joint_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['coef_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'irf_param_joint_sd',
        1.,
        float,
        "Square root of variance of intercept in initial variance-covariance matrix of joint distributions. Used only if either **covarying_fixef** or **covarying_ranef** is ``True``, otherwise ignored.",
        aliases=['irf_param_prior_sd', 'conv_param_joint_sd', 'conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'joint_sd_scaling_coefficient',
        1.,
        float,
        "Factor by which to multiply square roots of variances on intercepts and coefficients if inferred from the empirical variance of the data (i.e. if **intercept_joint_sd** or **coef_joint_sd** is ``None``). Ignored for any prior widths that are explicitly specified.",
        aliases=['prior_sd_scaling_coefficient']
    ),
    Kwarg(
        'ranef_to_fixef_joint_sd_ratio',
        0.1,
        float,
        "Ratio of widths of random to fixed effects root-variances in joint distributions. I.e. if less than 1, random effects have tighter distributions. Used only if either **covarying_fixef** or **covarying_ranef** is ``True``, otherwise ignored.",
        aliases=['ranef_to_fixef_prior_sd_ratio']
    )
]


CDRBAYES_INITIALIZATION_KWARGS = [
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
        'n_samples_eval',
        1024,
        int,
        "Number of posterior predictive samples to draw for prediction/evaluation."
    ),
    Kwarg(
        'intercept_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on fixed intercept. If ``None``, inferred as **prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['prior_sd']
    ),
    Kwarg(
        'coef_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on fixed coefficients. If ``None``, inferred as **prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['coefficient_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'irf_param_prior_sd',
        1.,
        float,
        "Standard deviation of prior on convolutional IRF parameters",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'y_sd_trainable',
        True,
        bool,
        "Tune the standard deviation of the output model during training. If ``False``, remains fixed at ``y_sd_init``.",
        aliases=['y_scale_trainable']
    ),
    Kwarg(
        'y_sd_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on standard deviation of output model. If ``None``, inferred as **y_sd_prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['y_scale_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'y_skewness_prior_sd',
        1,
        float,
        "Standard deviation of prior on skewness parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored.",
        aliases=['prior_sd']
    ),
    Kwarg(
        'y_tailweight_prior_sd',
        1,
        float,
        "Standard deviation of prior on tailweight parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored.",
        aliases=['prior_sd']
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
    Kwarg(
        'posterior_to_prior_sd_ratio',
        0.01,
        float,
        "Ratio of posterior initialization SD to prior SD. Low values are often beneficial to stability, convergence speed, and quality of final fit by avoiding erratic sampling and divergent behavior early in training."
    )
]

CDRNN_INITIALIZATION_KWARGS = [
    Kwarg(
        'rnn_type',
        'LSTM',
        str,
        "Type of RNN unit to use. One of ``['LSTM', 'GRU', 'SimpleRNN']."
    ),
    Kwarg(
        'n_samples_eval',
        1024,
        int,
        "Number of posterior predictive samples to draw for prediction/evaluation."
    ),
    Kwarg(
        'direct_irf',
        True,
        bool,
        "Whether to generate the response directly as the output of the IRF (``True``). Otherwise, IRF provides weights on the input dimensions."
    ),
    Kwarg(
        'use_coefficient',
        False,
        bool,
        "Whether to apply a trainable coefficient vector to the input dimensions."
    ),
    Kwarg(
        'heteroskedastic',
        True,
        bool,
        "Whether to parameterize the error distribution using a neural net. Otherwise, constant error parameters are used."
    ),
    Kwarg(
        'split_h',
        False,
        bool,
        "Whether to split the hidden state between IRF and error params fn. Ignored unless **heteroskedastic** is ``True``."
    ),
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
        False,
        bool,
        "Whether to rescale time values as inputs by their training SD under the hood. Times are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time']
    ),
    Kwarg(
        'rescale_t_delta',
        False,
        bool,
        "Whether to rescale time offset values by their training SD under the hood. Offsets are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_time', 'rescale_tdelta']
    ),
    Kwarg(
        'n_layers_input_projection',
        None,
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
        32,
        [int, str, None],
        "Number of units per RNN layer. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_layers_rnn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no RNN encoding (i.e. use a context-independent convolution kernel)."
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
        None,
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
    Kwarg(
        'nonstationary_intercept',
        False,
        bool,
        "Whether to implement a time-varying intercept term, using a feedforward network architecturally matched to the IRF."
    ),
    # Kwarg(
    #     'n_layers_error_params_fn',
    #     None,
    #     [int, None],
    #     "Number of hidden layers mapping hidden state to parameters of error distribution (e.g. variance). If ``None``, inferred from length of **n_units_error_params_fn**.",
    #     aliases=['n_layers_decoder']
    # ),
    # Kwarg(
    #     'n_units_error_params_fn',
    #     32,
    #     [int, str, None],
    #     "Number of units per hidden layer in mapping from hidden state to parameters of error distribution. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_variance_fn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers.",
    #     aliases=['n_units_decoder']
    # ),
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
    # Kwarg(
    #     'error_params_fn_inner_activation',
    #     'gelu',
    #     [str, None],
    #     "Name of activation function to use for hidden layers of error params function.",
    #     aliases=['activation']
    # ),
    # Kwarg(
    #     'error_params_fn_activation',
    #     None,
    #     [str, None],
    #     "Name of activation function to use for final layer in error params function."
    # ),
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
        'context_regularizer_name',
        None,
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
    Kwarg(
        'input_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop input_features."
    ),
    Kwarg(
        'input_projection_dropout_rate',
        None,
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
        None,
        [float, None],
        "Rate at which to drop neurons of h_in.",
        # aliases=['dropout_rate']
    ),
    Kwarg(
        'h_rnn_dropout_rate',
        None,
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
        None,
        [float, None],
        "Rate at which to drop neurons of IRF layers.",
        aliases=['dropout_rate']
    ),
    # Kwarg(
    #     'error_params_fn_dropout_rate',
    #     None,
    #     [float, None],
    #     "Rate at which to drop neurons of error params function.",
    #     aliases=['dropout_rate']
    # ),
    Kwarg(
        'ranef_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop random effects indicators.",
        aliases=['dropout_rate']
    ),
    Kwarg(
        'h_in_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into h_in."
    ),
    Kwarg(
        'h_rnn_noise_sd',
        None,
        [float, None],
        "SD of white-out noise to inject into h_rnn."
    ),
    Kwarg(
        'forget_rate',
        None,
        [float, None],
        "Rate at which to drop recurrent connection entirely."
    ),
    Kwarg(
        'input_jitter_level',
        None,
        [float, None],
        "Standard deviation of jitter injected into inputs (predictors and timesteps) during training. If ``0`` or ``None``, no input jitter."
    ),
    Kwarg(
        'default_reference_type',
        'mean',
        ['mean', 0.],
        "Reference stimulus to use by default for plotting and effect estimation. If `0`, zero vector. If `mean`, training set mean by predictor."
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
        'generate_nonstationarity_surface_plots',
        False,
        bool,
        "Whether to plot IRF surfaces showing non-stationarity in the response."
    )
]

CDRNNMLE_INITIALIZATION_KWARGS = [
    Kwarg(
        'weight_sd_init',
        'he',
        [float, str],
        "Standard deviation of kernel initialization distribution (Normal, mean=0). Can also be ``'glorot'``, which uses the SD of the Glorot normal initializer."
    )
]

CDRNNBAYES_INITIALIZATION_KWARGS = [
    Kwarg(
        'declare_priors_fixef',
        True,
        bool,
        "Specify Gaussian priors for all fixed model parameters (if ``False``, use implicit improper uniform priors).",
        aliases=['declare_priors']
    ),
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
        [float, None],
        "Standard deviation of prior on fixed intercept. If ``None``, inferred as **prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['prior_sd']
    ),
    Kwarg(
        'weight_prior_sd',
        'glorot',
        [str, float],
        "Standard deviation of prior on CDRNN hidden weights. A ``float``, ``'glorot'``, or ``'he'``.",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'weight_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over weights. If ``None``, inferred from other hyperparams."
    ),
    Kwarg(
        'bias_prior_sd',
        1.,
        [str, float],
        "Standard deviation of prior on CDRNN hidden biases. A ``float``, ``'glorot'``, or ``'he'``.",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'bias_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over biases. If ``None``, inferred from other hyperparams."
    ),
    Kwarg(
        'gamma_prior_sd',
        'glorot',
        [str, float],
        "Standard deviation of prior on batch norm gammas. A ``float``, ``'glorot'``, or ``'he'``. Ignored unless batch normalization is used",
        aliases=['conv_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'gamma_sd_init',
        None,
        [str, float, None],
        "Initial standard deviation of variational posterior over batch norm gammas. If ``None``, inferred from other hyperparams. Ignored unless batch normalization is used."
    ),
    Kwarg(
        'y_sd_trainable',
        True,
        bool,
        "Tune the standard deviation of the output model during training. If ``False``, remains fixed at ``y_sd_init``.",
        aliases=['y_scale_trainable']
    ),
    Kwarg(
        'y_sd_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on standard deviation of output model. If ``None``, inferred as **y_sd_prior_sd_scaling_coefficient** times the empirical variance of the response on the training set.",
        aliases=['y_scale_prior_sd', 'prior_sd']
    ),
    Kwarg(
        'y_skewness_prior_sd',
        1,
        float,
        "Standard deviation of prior on skewness parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored.",
        aliases=['prior_sd']
    ),
    Kwarg(
        'y_tailweight_prior_sd',
        1,
        float,
        "Standard deviation of prior on tailweight parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored.",
        aliases=['prior_sd']
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
    Kwarg(
        'posterior_to_prior_sd_ratio',
        0.01,
        float,
        "Ratio of posterior initialization SD to prior SD. Low values are often beneficial to stability, convergence speed, and quality of final fit by avoiding erratic sampling and divergent behavior early in training."
    )
]


def cdr_kwarg_docstring():
    """
    Generate docstring snippet summarizing all CDR kwargs, dtypes, and defaults.

    :return: ``str``; docstring snippet
    """

    out = "**All models**\n\n"

    for kwarg in MODEL_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out = "**All CDR models**\n\n"

    for kwarg in CDR_INITIALIZATION_KWARGS:
        if kwarg.key not in ['history_length']:
            out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**CDRMLE only**\n\n'

    for kwarg in CDRMLE_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**CDRBayes only**\n\n'

    for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**All CDRNN models**\n\n'

    for kwarg in CDRNN_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**CDRNNMLE only**\n\n'

    for kwarg in CDRNNMLE_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**CDRNNBayes only**\n\n'

    for kwarg in CDRNNBAYES_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)


    out += '\n'

    return out
