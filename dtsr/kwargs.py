from functools import cmp_to_key
import configparser

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr):
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, self.default_value)
        else:
            from_settings = settings.get(self.key, None)
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
                        except:
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





DTSR_INITIALIZATION_KWARGS = [
    Kwarg(
        'outdir',
        './dtsr_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'history_length',
        None,
        int,
        "Length of the history window to use."
    ),
    Kwarg(
        'pc',
        False,
        bool,
        "Transform input variables using principal components analysis (experimental, not thoroughly tested)."
    ),
    Kwarg(
        'intercept_init',
        None,
        [float, None],
        "Initial value to use for the intercept (if ``None``, use mean response in training data)"
    ),
    Kwarg(
        'init_sd',
        .001,
        float,
        "Standard deviation of Gaussian initialization distribution for trainable variables."
    ),
    Kwarg(
        'n_interp',
        64,
        int,
        "Number of interpolation points (ignored unless the model formula specification contains continuous inputs)."
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
            - ``None`` (DTSRBayes only; uses the default optimizer defined by Edward, which currently includes steep learning rate decay and is therefore not recommended in the general case)"""
    ),
    Kwarg(
        'optim_epsilon',
        0.01,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.01,
        float,
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        1e-4,
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
        float,
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
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
        'validate_irf_args',
        True,
        bool,
        "Check whether inputs and parameters to IRF obey constraints. Imposes a small performance cost but helps catch and report bugs in the model."
    ),
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'log_random',
        True,
        bool,
        "Log random effects to Tensorboard."
    ),
    Kwarg(
        'log_freq',
        1,
        int,
        "Frequency (in iterations) with which to log model params to Tensorboard."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    )
]

DTSRMLE_INITIALIZATION_KWARGS = [
    Kwarg(
        'loss_name',
        'mse',
        str,
        "The optimization objective."
    )
]


DTSRBAYES_INITIALIZATION_KWARGS = [
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
        "Specify Gaussian priors for all fixed model parameters (if ``False``, use implicit improper uniform priors)."
    ),
    Kwarg(
        'declare_priors_ranef',
        True,
        bool,
        "Specify Gaussian priors for all random model parameters (if ``False``, use implicit improper uniform priors)."
    ),
    Kwarg(
        'n_iter',
        1000,
        int,
        "Number of training iterations. If using variational inference, this becomes the `expected` number of training iterations and is used only for Tensorboard logging, with no impact on training behavior."
    ),
    Kwarg(
        'n_samples',
        None,
        int,
        "Number of posterior samples to draw at each training step during variational inference. If using MCMC inferences, the number of samples is set deterministically as ``n_iter * n_minibatch``, so this user-supplied parameter is overridden."
    ),
    Kwarg(
        'n_samples_eval',
        1024,
        [int],
        "Number of posterior predictive samples to draw for prediction/evaluation."
    ),
    Kwarg(
        'intercept_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on fixed intercept. If ``None``, inferred as ``prior_sd_scaling_coefficient`` times the empirical variance of the response on the training set."
    ),
    Kwarg(
        'coef_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on fixed coefficients. If ``None``, inferred as ``prior_sd_scaling_coefficient`` times the empirical variance of the response on the training set."
    ),
    Kwarg(
        'conv_prior_sd',
        1,
        float,
        "Standard deviation of prior on convolutional IRF parameters"
    ),
    Kwarg(
        'y_scale_init',
        None,
        [float, None],
        "Initial value for the standard deviation of the output model. If ``None``, inferred as the empirical standard deviation of the response on the training set."
    ),
    Kwarg(
        'y_scale_trainable',
        True,
        bool,
        "Tune the standard deviation of the output model during training. If ``False``, remains fixed at ``y_scale_init``."
    ),
    Kwarg(
        'y_scale_prior_sd',
        None,
        [float, None],
        "Standard deviation of prior on standard deviation of output model. If ``None``, inferred as ``y_scale_prior_sd_scaling_coefficient`` times the empirical variance of the response on the training set."
    ),
    Kwarg(
        'y_skewness_prior_sd',
        1,
        float,
        "Standard deviation of prior on skewness parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored."
    ),
    Kwarg(
        'y_tailweight_prior_sd',
        1,
        float,
        "Standard deviation of prior on tailweight parameter of output model. Only used if ``asymmetric_error == True``, otherwise ignored."
    ),
    Kwarg(
        'mh_proposal_sd',
        None,
        [float, None],
        "Standard deviation of proposal distribution. If ``None``, inferred as standard deviation of corresponding prior. Only used if ``inference_name == 'MetropolisHastings', otherwise ignored."
    ),
    Kwarg(
        'prior_sd_scaling_coefficient',
        1,
        float,
        "Factor by which to multiply priors on intercepts and coefficients if inferred from the empirical variance of the data (i.e. if ``intercept_prior_sd`` or ``coef_prior_sd`` is ``None``). Ignored for any prior widths that are explicitly specified."
    ),
    Kwarg(
        'y_scale_prior_sd_scaling_coefficient',
        1,
        float,
        "Factor by which to multiply prior on output model variance if inferred from the empirical variance of the data (i.e. if ``y_scale_prior_sd`` is ``None``). Ignored if prior width is explicitly specified."
    ),
    Kwarg(
        'ranef_to_fixef_prior_sd_ratio',
        1,
        float,
        "Ratio of widths of random to fixed effects priors. I.e. if less than 1, random effects have tighter priors."
    ),
    Kwarg(
        'posterior_to_prior_sd_ratio',
        0.01,
        float,
        "Ratio of widths of priors to posterior initializations. Low values are often beneficial to stability, convergence speed, and optimality of the final model by avoiding erratic sampling and divergent behavior early in training."
    ),
    Kwarg(
        'asymmetric_error',
        False,
        bool,
        "Allow an asymmetric error distribution by fitting a SinArcsinh transform of the normal error, adding trainable skewness and tailweight parameters."
    ),
    Kwarg(
        'mv',
        False,
        bool,
        "Use multivariate model that fits covariances between fixed effects (experimental, not thoroughly tested). If ``False``, parameter distributions are treated as independent."
    ),
    Kwarg(
        'mv_ran',
        False,
        bool,
        "Use multivariate model that fits covariances between random effects within a random grouping factor (experimental, not thoroughly tested). If ``False``, random parameter distributions are treated as independent."
    )
]


def dtsr_kwarg_docstring():
    out = "**Both DTSRMLE and DTSRBayes**\n\n"

    for kwarg in DTSR_INITIALIZATION_KWARGS:
        if kwarg.key not in ['outdir', 'history_length']:
            out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**DTSRMLE only**\n\n'

    for kwarg in DTSRMLE_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n**DTSRBayes only**\n\n'

    for kwarg in DTSRBAYES_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n'

    return out
