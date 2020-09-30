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
                        except TypeError:
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
        'standardize_response',
        True,
        bool,
        "Standardize (Z-transform) the response variable. Can improve convergence speed and reduce vulnerability to local optima. Only affects fitting -- the original response scale is still used for prediction, likelihood computation, and plotting."
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
        "Constraint function to use for bounded variables. One of ``['abs', 'softplus']``."
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
            - ``'Nadam'``
            - ``None`` (CDRBayes only; uses the default optimizer defined by Edward, which currently includes steep learning rate decay and is therefore not recommended in the general case)"""
    ),
    Kwarg(
        'use_jtps',
        False,
        bool,
        "Whether to modify the base optimizer using JTPS. If ``False``, runs a baseline model. If ``True``, runs a test model."
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'epsilon',
        1e-2,
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
        'regularizer_name',
        None,
        [str, None],
        "Name of global regularizer; can be overridden by more regularizers for more specific parameters (e.g. ``l1_regularizer``, ``l2_regularizer``). If ``None``, no regularization."
    ),
    Kwarg(
        'regularizer_scale',
        0.01,
        float,
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
        [float, 'inherit'],
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
        [float, 'inherit'],
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
        'keep_plot_history',
        False,
        bool,
        "Keep IRF plots from each checkpoint of a run, which can help visualize learning trajectories but can also consume a lot of disk space. If ``False``, only the most recent plot of each type is kept."
    )
]


CDR_INITIALIZATION_KWARGS = [
    Kwarg(
        'pc',
        False,
        bool,
        "Transform input variables using principal components analysis. Experimental, not thoroughly tested."
    ),
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
        'coefficient_regularizer_name',
        'inherit',
        [str, 'inherit', None],
        "Name of coefficient regularizer (e.g. ``l1_regularizer``, ``l2_regularizer``); overrides **regularizer_name**. If ``'inherit'``, inherits **regularizer_name**. If ``None``, no regularization."
    ),
    Kwarg(
        'coefficient_regularizer_scale',
        'inherit',
        [float, 'inherit'],
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
        [float, 'inherit'],
        "Scale of IRF parameter regularizer (ignored if ``regularizer_name==None``). If ``'inherit'``, inherits **regularizer_scale**."
    ),
    Kwarg(
        'oob_regularizer_threshold',
        None,
        [float, None],
        "Threshold (in units time) for out-of-bounds regularizer. If ``None``, default to 75th percentile of temporal offsets seen in training."
    ),
    Kwarg(
        'oob_regularizer_scale',
        None,
        [float, 'inherit', None],
        "Scale of out-of-bounds regularizer. If ``'inherit'``, inherits **regularizer_scale**. If ``None``, no out-of-bounds regularization."
    ),
    Kwarg(
        'validate_irf_args',
        True,
        bool,
        "Check whether inputs and parameters to IRF obey constraints. Imposes a small performance cost but helps catch and report bugs in the model."
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
        'inference_name',
        'KLqp',
        str,
        "The Edward inference class to use for fitting."
    ),
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
        'n_iter',
        100000,
        int,
        "Number of training iterations. If using variational inference, this becomes the `expected` number of training iterations and is used only for Tensorboard logging, with no impact on training behavior."
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
        'mh_proposal_sd',
        None,
        [float, None],
        "Standard deviation of proposal distribution. If ``None``, inferred as standard deviation of corresponding prior. Only used if ``inference_name == 'MetropolisHastings'``, otherwise ignored."
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
        'rescale_time',
        False,
        bool,
        "Whether to rescale time values by their training SD under the hood, which can help with convergence. Times are automatically reconverted back to the source scale for plotting and model criticism.",
        aliases=['rescale_t_delta']
    ),
    Kwarg(
        'n_layers_input_projection',
        None,
        [int, None],
        "Number of hidden layers in input projection. If ``None``, inferred from length of **n_units_input_projection**."
    ),
    Kwarg(
        'n_units_input_projection',
        None,
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
        "Number of units per RNN layer. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_layers_rnn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no RNN encoding (i.e. use a stationary convolution kernel)."
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
        None,
        [int, None],
        "Number of units in CDRNN hidden. Must be an ``int``. If ``None``, no RNN encoding (i.e. use a stationary convolution kernel)."
    ),
    Kwarg(
        'n_units_t_delta_embedding',
        None,
        [int, None],
        "Number of units in the embedding of the temporal offset t_delta. If ``None``, inferred automatically.",
        aliases=['n_units_decoder', 'n_units_embedding']
    ),
    Kwarg(
        'n_layers_irf',
        None,
        [int, None],
        "Number of IRF layers. If ``None``, inferred from length of **n_units_irf**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_irf',
        None,
        [int, str, None],
        "Number of units per hidden layer in IRF. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_irf** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers.",
        aliases=['n_units_decoder']
    ),
    Kwarg(
        'n_layers_error_params_fn',
        None,
        [int, None],
        "Number of hidden layers mapping hidden state to parameters of error distribution (e.g. variance). If ``None``, inferred from length of **n_units_error_params_fn**.",
        aliases=['n_layers_decoder']
    ),
    Kwarg(
        'n_units_error_params_fn',
        None,
        [int, str, None],
        "Number of units per hidden layer in mapping from hidden state to parameters of error distribution. Can be an ``int``, which will be used for all layers, or a ``str`` with **n_units_variance_fn** space-delimited integers, one for each layer in order from bottom to top. If ``0`` or ``None``, no hidden layers.",
        aliases=['n_units_decoder']
    ),
    Kwarg(
        'input_projection_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in input projection."
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
        "Name of activation function to use for hidden layers in projection of RNN state."
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
        "Name of activation function to use for CDRNN hidden state."
    ),
    Kwarg(
        'irf_inner_activation',
        'gelu',
        [str, None],
        "Name of activation function to use for hidden layers in IRF.",
        aliases=['decoder_inner_activation']
    ),
    Kwarg(
        'irf_activation',
        None,
        [str, None],
        "Name of activation function to use for final layer in IRF.",
        aliases=['decoder_activation']
    ),
    Kwarg(
        'error_params_fn_inner_activation',
        None,
        [str, None],
        "Name of activation function to use for hidden layers of error params function.",
        aliases=['decoder_inner_activation']
    ),
    Kwarg(
        'error_params_fn_activation',
        None,
        [str, None],
        "Name of activation function to use for final layer in error params function.",
        aliases=['decoder_activation']
    ),
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
        [float, 'inherit'],
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
        'predictor_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop predictors.",
        aliases=['input_dropout_rate']
    ),
    Kwarg(
        'event_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop events (input vectors).",
        aliases=['input_dropout_rate']
    ),
    Kwarg(
        'tail_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop the tail of events (input vectors).",
        aliases=['input_dropout_rate']
    ),
    Kwarg(
        'tail_dropout_max',
        None,
        [int, None],
        "Maximum number of final events to randomly dropout from time series during training."
    ),
    Kwarg(
        'rangf_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop random grouping factors.",
        aliases=['input_dropout_rate']
    ),
    Kwarg(
        'hidden_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop bottom-up neurons in the encoder."
    ),
    Kwarg(
        'recurrent_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop recurrent neurons in the encoder."
    ),
    Kwarg(
        'input_projection_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of input projection layers."
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
        'h_in_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of h_in."
    ),
    Kwarg(
        'h_rnn_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of h_rnn."
    ),
    Kwarg(
        'irf_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of IRF layers."
    ),
    Kwarg(
        'error_params_fn_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop neurons of error params function."
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
    )
]

CDRNNMLE_INITIALIZATION_KWARGS = [

]

CDRNNBAYES_INITIALIZATION_KWARGS = [
    Kwarg(
        'n_samples_eval',
        1024,
        int,
        "Number of posterior predictive samples to draw for prediction/evaluation."
    ),
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
