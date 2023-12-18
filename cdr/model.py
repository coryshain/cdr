import textwrap
import time as pytime
from collections import defaultdict
import subprocess

import scipy.interpolate
import scipy.linalg
import scipy.signal
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score

from .backend import *
from .data import build_CDR_impulse_data, build_CDR_response_data, corr, get_first_last_obs_lists, \
    split_cdr_outputs, concat_nested
from .formula import *
from .kwargs import MODEL_INITIALIZATION_KWARGS
from .opt import *
from .plot import *

NN_KWARG_BY_KEY = {x.key: x for x in NN_KWARGS + NN_BAYES_KWARGS}
ENSEMBLE = re.compile('\.m\d+')
CROSSVAL = re.compile('\.CV([^.~]+)~([^.~]+)')
N_MCIFIED_DIST_RESAMP = 10000

import tensorflow as tf
if int(tf.__version__.split('.')[0]) == 1:
    from tensorflow.contrib.distributions import Distribution, Normal, SinhArcsinh, Bernoulli, Categorical, \
        Exponential, TransformedDistribution
    import tensorflow.contrib.distributions as tfd
    AffineScalar = tfd.bijectors.AffineScalar

    # TODO: Implement
    class ExponentiallyModifiedGaussian(TransformedDistribution):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('The exgaussian distribution is not supported in TensorFlow v1. Switch to a TensorFlow v2 release in order to use this feature.')

    # TODO: Implement
    class JohnsonSU(Distribution):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('The JohnsonSU distribution is not supported in TensorFlow v1. Switch to a TensorFlow v2 release in order to use this feature.')

    # Much of this is stolen from the TF2 source, since TF1 lacks LogNormal
    class LogNormal(TransformedDistribution):
        def __init__(self,
                loc,
                scale,
                validate_args=False,
                name='LogNormal'
        ):
            with tf.name_scope(name) as name:
                super(LogNormal, self).__init__(
                    distribution=Normal(loc=loc, scale=scale),
                    bijector=tfd.bijectors.Exp(),
                    validate_args=validate_args,
                    name=name
                )

        @property
        def loc(self):
            """Distribution parameter for the pre-transformed mean."""
            return self.distribution.loc

        @property
        def scale(self):
            """Distribution parameter for the pre-transformed standard deviation."""
            return self.distribution.scale

        def _mean(self):
            return tf.exp(self.distribution.mean() + 0.5 * self.distribution.variance())

        def _variance(self):
            variance = self.distribution.variance()
            return ((tf.exp(variance) - 1) *
                    tf.exp(2. * self.distribution.mean() + variance))

        def _mode(self):
            return tf.exp(self.distribution.mean() - self.distribution.variance())

        def _entropy(self):
            return (self.distribution.mean() + 0.5 +
                    tf.log(self.distribution.stddev()) + 0.5 * np.log(2 * np.pi))

        def _default_event_space_bijector(self):
            return tfd.bijectors.Exp(validate_args=self.validate_args)

        @classmethod
        def _maximum_likelihood_parameters(cls, value):
            log_x = tf.log(value)
            return {'loc': tf.reduce_mean(log_x, axis=0),
                    'scale': tf.math.reduce_std(log_x, axis=0)}

    from tensorflow.contrib.opt import NadamOptimizer
    from tensorflow.contrib.framework import argsort as tf_argsort
    from tensorflow.contrib import keras
    from tensorflow import check_numerics as tf_check_numerics
    parameter_properties = None

    def tf_quantile(x, q, **kwargs):
        return tfd.percentile(x, q * 100, **kwargs)

    def histogram(*args, **kwargs):
        raise NotImplementedError('Bootstrapping the mode is not supported in TensorFlow v1. Switch to a TensorFlow v2 release in order to use this feature.')

    TF_MAJOR_VERSION = 1
elif int(tf.__version__.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow_probability import distributions as tfd
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import stats as tfs
    from tensorflow_probability.python.internal import parameter_properties

    Distribution = tfd.Distribution
    Normal = tfd.Normal
    LogNormal = tfd.LogNormal
    Bernoulli = tfd.Bernoulli
    Categorical = tfd.Categorical
    Exponential = tfd.Exponential
    ExponentiallyModifiedGaussian = tfd.ExponentiallyModifiedGaussian
    SinhArcsinh = tfd.SinhArcsinh
    JohnsonSU = tfd.JohnsonSU
    TransformedDistribution = tfd.TransformedDistribution
    Shift = tfb.Shift
    Scale = tfb.Scale
    Chain = tfb.Chain
    Identity = tfb.Identity
    histogram = tfs.histogram

    def AffineScalar(shift, scale, *args, **kwargs):
        chain = []
        if shift is not None:
            m = Shift(shift)
            chain.append(m)
        if scale is not None:
            s = Scale(scale)
            chain.append(s)
        if len(chain) == 0:
            return Identity(*args, **kwargs)
        elif len(chain) == 1:
            return chain[0]
        return Chain(chain, *args, **kwargs)

    from tensorflow_probability import math as tfm
    tf_erfcx = tfm.erfcx
    from tensorflow.compat.v1.keras.optimizers import Nadam as NadamOptimizer
    from tensorflow import argsort as tf_argsort
    from tensorflow import keras
    from tensorflow.debugging import check_numerics as tf_check_numerics

    def tf_quantile(x, q, **kwargs):
        return tfs.percentile(x, q * 100, **kwargs)

    TF_MAJOR_VERSION = 2
else:
    raise ImportError('Unsupported TensorFlow version: %s. Must be 1.x.x or 2.x.x.' % tf.__version__)


class SkewNormal(Distribution):
    def __init__(self,
                 loc,
                 scale,
                 shape,
                 validate_args=False,
                 name='SkewNormal'
                 ):
        with tf.name_scope(name) as name:
            self._loc = tf.convert_to_tensor(loc, dtype=tf.float32)
            self._scale = tf.convert_to_tensor(scale, dtype=tf.float32)
            self._shape = tf.convert_to_tensor(shape, dtype=tf.float32)
            super(SkewNormal, self).__init__(
                validate_args=validate_args,
                name=name
            )

            stdnorm = Normal(loc=0., scale=1.)
            self.stdnorm_logpdf = stdnorm.log_prob
            self.stdnorm_logcdf = stdnorm.log_cdf
            self.log2 = tf.convert_to_tensor(np.log(2), dtype=tf.float32)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed standard deviation."""
        return self._scale

    @property
    def shape(self):
        """Distribution parameter for the pre-transformed standard deviation."""
        return self._shape

    def _log_prob(self, x):
        z = self._z(x)
        return tf.log(2 / self.scale) + self.stdnorm_logpdf(z) + self.stdnorm_logcdf(self.shape * z)

    def _mean(self):
        # TODO
        pass

    def _variance(self):
        # TODO
        pass

    def _mode(self):
        # TODO
        pass

    def _z(self, x, scale=None):
        """Standardize input `x` to a unit normal."""
        with tf.name_scope('standardize'):
            return (x - self.loc) / (self.scale if scale is None else scale)

    @classmethod
    def _maximum_likelihood_parameters(cls, value):
        log_x = tf.log(value)
        return {'loc': tf.reduce_mean(log_x, axis=0),
                'scale': tf.math.reduce_std(log_x, axis=0)}


def mcify(dist):
    class MCifiedDistribution(dist):
        """
        A wrapper that provides non-differentiable Monte Carlo approximations to key statistics
        like mean() and quantile() in case the source distribution lacks analytical implementations
        of these.
        """
        def __init__(
                self,
                *args,
                n_resamp=N_MCIFIED_DIST_RESAMP,
                **kwargs
        ):
            super(MCifiedDistribution, self).__init__(*args, **kwargs)
            self.n_resamp = n_resamp
            self.dist_name = dist.__name__

        def _mean(self):
            try:
                return super(MCifiedDistribution, self)._mean()
            except NotImplementedError:
                samp = self.sample(sample_shape=self.n_resamp)
                return tf.reduce_mean(samp, axis=0)

        def _mode(self):
            try:
                return super(MCifiedDistribution, self)._mode()
            except NotImplementedError:
                # Bootstrap the mode, which is surprisingly involved.
                # Returns the midpoint of the largest bin in a histogram of samples from the distribution.

                # Get argmax indices of sample histogram
                samp = self.sample(sample_shape=self.n_resamp)
                nbins = self.n_resamp // 100
                bins = tf.linspace(
                    tf.reduce_min(samp, axis=0),
                    tf.reduce_max(samp, axis=0),
                    nbins + 1
                )
                vals = (bins[:-1] + bins[1:]) / 2 # Midpoints of each bin, which define possible return values
                h = histogram(samp, bins, axis=0, extend_lower_interval=True, extend_upper_interval=True)
                ix = tf.cast(tf.argmax(h, axis=0), tf.int32)

                # Index into the bootstrapped mode
                gather_ix = [ix]
                ndim = len(ix.shape)
                ix_shape = tf.shape(ix)
                for i in range(ndim):
                    _gather_ix = tf.range(ix_shape[i])
                    for j in range(i):
                        _gather_ix = _gather_ix[None, ...]
                    for j in range(i + 1, ndim):
                        _gather_ix = _gather_ix[..., None]
                    tile_ix = [ix_shape[j] if j != i else 1 for j in range(ndim)]
                    _gather_ix = tf.tile(_gather_ix, tile_ix)
                    gather_ix.append(_gather_ix)
                gather_ix = tf.stack(gather_ix, axis=-1)
                mode = tf.gather_nd(vals, gather_ix)

                return mode

        def _quantile(self, q, **kwargs):
            try:
                return super(MCifiedDistribution, self)._quantile(q, **kwargs)
            except NotImplementedError:
                samp = self.sample(sample_shape=self.n_resamp)
                return tf_quantile(samp, q, **kwargs)

        def has_analytical_mean(self):
            try:
                super(MCifiedDistribution, self)._mean()
                return True
            except NotImplementedError:
                return False

    return MCifiedDistribution


class LogNormalV2(LogNormal):
    """
    A LogNormal distribution parameterized by its mean and standard deviation,
    rather than the more common parameterization by the mean and standard deviation of
    the underlying normal distribution.

    :param loc: TF tensor; mean.
    :param scale: TF tensor; standard deviation
    :param epsilon: ``float``; stabilizing constant
    :param kwargs: ``dict``; additional kwargs to the LogNormal distribution
    :return: LogNormalV2 distribution object
    """
    def __init__(self, loc, scale, epsilon=1e-5, **kwargs):
        scale = tf.sqrt(tf.log(tf.square(scale / (loc + epsilon)) + 1) + epsilon)
        loc = tf.log(tf.maximum(loc, epsilon)) - tf.square(scale) / 2.
        parameters = dict(locals())
        super(LogNormalV2, self).__init__(loc, scale, **kwargs)
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return LogNormal._parameter_properties(dtype, num_classes=num_classes)


class ExponentiallyModifiedGaussianInvRate(ExponentiallyModifiedGaussian):
    """
    An ExponentiallyModifiedGaussian distribution that uses the inverse rate rather than the rate
    parameterization. The inverse rate parameterization has the advantage that the mean is linear in the location
    and scale parameters.

    :param loc: TF tensor; mean.
    :param scale: TF tensor; standard deviation
    :param rate: TF tensor; inverse rate parameter
    :param epsilon: ``float``; stabilizing constant
    :param kwargs: ``dict``; additional kwargs to the LogNormal distribution
    :return: LogNormalV2 distribution object
    """
    def __init__(self, loc, scale, rate, *args, **kwargs):
        rate = 1. / rate
        parameters = dict(locals())
        super(ExponentiallyModifiedGaussianInvRate, self).__init__(loc, scale, rate, *args, **kwargs)
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return ExponentiallyModifiedGaussian._parameter_properties(dtype, num_classes=num_classes)


class ShiftedScaledDistribution(TransformedDistribution):
    def __init__(self,
                 dist,
                 shift,
                 scale,
                 validate_args=False,
                 name='ShiftedScaledDistribution'
                 ):
        with tf.name_scope(name) as name:
            parameters = dict(locals())
            self._shift = tf.convert_to_tensor(shift, tf.float32)
            self._scale = tf.convert_to_tensor(scale, tf.float32)
            bijector = AffineScalar(
                shift,
                scale
            )
            super(ShiftedScaledDistribution, self).__init__(
                distribution=dist,
                bijector=bijector,
                validate_args=validate_args,
                name=name
            )
            self._parameters = parameters

    @property
    def shift(self):
        return self._shift

    @property
    def scale(self):
        return self._scale

    def _mean(self):
        return self.bijector.forward(self.distribution.mean())

    def _variance(self):
        return tf.square(self.scale) * self.distribution.variance()

    def _mode(self):
        return self.bijector.forward(self.distribution.mode())

    def _entropy(self):
        return self.distribution.entropy()

    def _quantile(self, q):
        return self.bijector.forward(self.distribution.quantile(q))

    def has_analytical_mean(self):
        return self.distribution.has_analytical_mean()

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            shift=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties()
        )


class ZeroMeanDistribution(ShiftedScaledDistribution):
    def __init__(self,
                 dist,
                 validate_args=False,
                 name='ShiftedScaledDistribution'
                 ):
        with tf.name_scope(name) as name:

            shift = -dist.mean()
            scale = tf.ones_like(shift)
            super(ZeroMeanDistribution, self).__init__(
                dist,
                shift,
                scale,
                validate_args=validate_args,
                name=name
            )

            self._properties = dict()

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


class CDRModel(object):
    _INITIALIZATION_KWARGS = MODEL_INITIALIZATION_KWARGS

    _doc_header = """
        Class implementing a continuous-time deconvolutional regression model.
    """
    _doc_args = """
        :param form_str: An R-style string representing the model formula.
        :param X: ``pandas`` table or ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Often only one table will be used.
            Support for multiple tables allows simultaneous use of independent variables that are measured at different times (e.g. word features and sound power in Shain, Blank, et al. (2020).
            Each ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the ``form_str`` provided at initialization
        :param Y: ``pandas`` table or ``list`` of ``pandas`` tables; matrices of response variables, grouped by series and temporally sorted.
            Each ``Y`` must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs(_<K>)``:  Index in the design matrix `X` of the first observation in the time series associated with each observation in ``y``. If multiple ``X``, must be zero-indexed for each of the K dataframes in X.
            * ``last_obs(_<K>)``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each observation in ``y``. If multiple ``X``, must be zero-indexed for each of the K dataframes in X.
            * A column with the same name as each response variable specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs


    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    IRF_KERNELS = {
        'DiracDelta': [],
        'Exp': [
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ExpRateGT1': [
            ('beta', {'lb': 1., 'default': 2.})
        ],
        'Gamma': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'GammaShapeGT1': [
            ('alpha', {'lb': 1., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ShiftedGamma': [
            ('alpha', {'lb': 0., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'ShiftedGammaShapeGT1': [
            ('alpha', {'lb': 1., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'Normal': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.})
        ],
        'SkewNormal': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.}),
            ('alpha', {'default': 0.})
        ],
        'EMG': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'BetaPrime': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ShiftedBetaPrime': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'HRFSingleGamma': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma1': [
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma2': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma3': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
        'HRFDoubleGamma4': [
            ('alpha_main', {'lb': 1., 'default': 6.}),
            ('alpha_undershoot', {'lb': 1., 'default': 16.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
        'HRFDoubleGamma5': [
            ('alpha_main', {'lb': 1., 'default': 6.}),
            ('alpha_undershoot', {'lb': 1., 'default': 16.}),
            ('beta_main', {'lb': 0., 'default': 1.}),
            ('beta_undershoot', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
        'NN': []
    }

    N_QUANTILES = 41
    RESPONSE_DISTRIBUTIONS = Formula.RESPONSE_DISTRIBUTIONS.copy()
    for x in RESPONSE_DISTRIBUTIONS:
        RESPONSE_DISTRIBUTIONS[x]['dist'] = globals()[RESPONSE_DISTRIBUTIONS[x]['dist']]

    def __init__(self, form, X, Y, ablated=None, build=True, **kwargs):
        super(CDRModel, self).__init__()

        ## Store initialization settings
        for kwarg in CDRModel._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        stderr('  Collecting summary statistics...\n')

        assert self.n_samples == 1, 'n_samples is now deprecated and must be left at its default of 1'

        if not isinstance(X, list):
            X = [X]
        if not isinstance(Y, list):
            Y = [Y]

        # Cross validation settings
        for crossval_kwarg in (
            'crossval_factor',
            'crossval_folds',
            'crossval_fold',
            'crossval_dev_fold'
        ):
            setattr(self, crossval_kwarg, kwargs[crossval_kwarg])
            del kwargs[crossval_kwarg]
        use_crossval = bool(self.crossval_factor)
        if use_crossval:
            Y = [_Y[_Y[self.crossval_factor] != self.crossval_fold] for _Y in Y]

        # Plot default settings
        self.irf_name_map = kwargs['irf_name_map']
        del kwargs['irf_name_map']

        # Parse and store model data from formula
        if isinstance(form, str):
            self.form_str = form
            form = Formula(form)
        else:
            self.form_str = str(form)
        form = form.re_transform(X)
        form = form.categorical_transform(X)
        self.form = form
        if self.future_length:
            assert self.form.t.supports_non_causal(), 'If future_length > 0, causal IRF kernels (kernels which require that t > 0) cannot be used.'

        responses = form.responses()
        response_names = [x.name() for x in responses]
        response_is_categorical = {}
        response_ndim = {}
        response_category_maps = {}
        for _response in responses:
            if _response.categorical(Y):
                is_categorical = True
                found = False
                for _Y in Y:
                    if _response.name() in _Y:
                        cats = sorted(list(_Y[_response.name()].unique()))
                        category_map = dict(zip(cats, range(len(cats))))
                        n_dim = len(cats)
                        found = True
                        break
                assert found, 'Response %s not found in data.' % _response.name()
            else:
                is_categorical = False
                category_map = {}
                n_dim = 1
            response_is_categorical[_response.name()] = is_categorical
            response_ndim[_response.name()] = n_dim
            response_category_maps[_response.name()] = category_map
        response_expanded_bounds = {}
        s = 0
        for _response in response_names:
            n_dim = response_ndim[_response]
            e = s + n_dim
            response_expanded_bounds[_response] = (s, e)
            s = e
        self.response_is_categorical = response_is_categorical
        self.response_ndim = response_ndim
        self.response_category_maps = response_category_maps
        self.response_expanded_bounds = response_expanded_bounds

        rangf = form.rangf

        # Store ablation info
        if ablated is None:
            self.ablated = set()
        elif isinstance(ablated, str):
            self.ablated = {ablated}
        else:
            self.ablated = set(ablated)

        q = np.linspace(0.0, 1, self.N_QUANTILES)

        # Collect stats for response variable(s)
        self.n_train = 0.
        Y_all = {x: [] for x in response_names}
        for i, _Y in enumerate(Y):
            to_add = True
            for j, _response in enumerate(response_names):
                if _response in _Y:
                    if to_add:
                        self.n_train += len(_Y)
                        to_add = False
                    Y_all[_response].append(_Y[_response])

        Y_train_means = {}
        Y_train_sds = {}
        Y_train_quantiles = {}

        for i, _response_name in enumerate(Y_all):
            stderr('\r    Processing response %d/%d...' % (i + 1, len(Y_all)))
            _response = Y_all[_response_name]
            if len(_response):
                if response_is_categorical[_response_name]:
                    _response = pd.concat(_response)
                    _map = response_category_maps[_response_name]
                    _response = _response.map(_map).values
                    # To 1-hot
                    __response = np.zeros((len(_response), len(_map.values())))
                    __response[np.arange(len(_response)), _response] = 1
                    _response = __response
                else:
                    _response = np.concatenate(_response, axis=0)[..., None]
                _mean = np.nanmean(_response, axis=0)
                _sd = np.nanstd(_response, axis=0)
                assert np.all(_sd > 0), 'Some responses (%s) had no variance. SD vector: %s.' % (_response_name, _sd)
                _quantiles = np.nanquantile(_response, q, axis=0)
            else:
                _mean = 0.
                _sd = 0.
                _quantiles = np.zeros_like(q)

            Y_train_means[_response_name] = _mean
            Y_train_sds[_response_name] = _sd
            Y_train_quantiles[_response_name] = _quantiles

        self.Y_train_means = Y_train_means
        self.Y_train_sds = Y_train_sds
        self.Y_train_quantiles = Y_train_quantiles

        impulse_means = {}
        impulse_sds = {}
        impulse_medians = {}
        impulse_quantiles = {}
        impulse_lq = {}
        impulse_uq = {}
        impulse_min = {}
        impulse_max = {}
        indicators = set()
        impulse_df_ix = []
        impulse_blocks = {}
        impulses = self.form.t.impulses(include_interactions=True)

        for impulse_ix, impulse in enumerate(impulses):
            stderr('\r    Processing predictor %d/%d...' % (impulse_ix + 1, len(impulses)))
            name = impulse.name()
            is_interaction = type(impulse).__name__ == 'ImpulseInteraction'
            found = False
            i = 0
            if name.lower() == 'rate':
                found = True
                impulse_means[name] = 1.
                impulse_sds[name] = 1.
                quantiles = np.ones_like(q)
                impulse_quantiles[name] = quantiles
                impulse_medians[name] = 1.
                impulse_lq[name] = 1.
                impulse_uq[name] = 1.
                impulse_min[name] = 1.
                impulse_max[name] = 1.
            else:
                for i, df in enumerate(X + Y):
                    if name in df and not name.lower() == 'rate':
                        column = df[name].values
                        impulse_means[name] = np.nanmean(column)
                        impulse_sds[name] = np.nanstd(column)
                        assert impulse_sds[name] > 0, 'Predictor %s had no variance' % name
                        quantiles = np.nanquantile(column, q)
                        impulse_quantiles[name] = quantiles
                        impulse_medians[name] = np.nanquantile(column, 0.5)
                        impulse_lq[name] = np.nanquantile(column, 0.1)
                        impulse_uq[name] = np.nanquantile(column, 0.9)
                        impulse_min[name] = np.nanmin(column)
                        impulse_max[name] = np.nanmax(column)

                        if self._vector_is_indicator(column):
                            indicators.add(name)

                        if i not in impulse_blocks:
                            impulse_blocks[i] = {}
                        impulse_blocks[i][name] = column

                        found = True
                        break
                    elif is_interaction:
                        found = True
                        impulse_names = [x.name() for x in impulse.impulses()]
                        for x in impulse.impulses():
                            if not x.name() in df:
                                found = False
                                break
                        if found:
                            column = df[impulse_names].product(axis=1).values
                            impulse_means[name] = np.nanmean(column)
                            impulse_sds[name] = np.nanstd(column)
                            assert impulse_sds[name] > 0, 'Predictor %s had no variance' % name
                            quantiles = np.nanquantile(column, q)
                            impulse_quantiles[name] = quantiles
                            impulse_medians[name] = np.nanquantile(column, 0.5)
                            impulse_lq[name] = np.nanquantile(column, 0.1)
                            impulse_uq[name] = np.nanquantile(column, 0.9)
                            impulse_min[name] = np.nanmin(column)
                            impulse_max[name] = np.nanmax(column)

                            if i not in impulse_blocks:
                                impulse_blocks[i] = {}
                            impulse_blocks[i][name] = column

                            if self._vector_is_indicator(column):
                                indicators.add(name)
                            break
            if not found:
                raise ValueError('Impulse %s was not found in an input file.' % name)

            impulse_df_ix.append(i)

        stderr('\n')

        self.impulse_df_ix = impulse_df_ix
        impulse_df_ix_unique = set(self.impulse_df_ix)

        self.impulse_means = impulse_means
        self.impulse_sds = impulse_sds
        self.impulse_medians = impulse_medians
        self.impulse_quantiles = impulse_quantiles
        self.impulse_lq = impulse_lq
        self.impulse_uq = impulse_uq
        self.impulse_min = impulse_min
        self.impulse_max = impulse_max
        self.indicators = indicators

        if len(self.impulse_means) > 1 and len(self.impulse_means) < 100:
            stderr('\r    Computing predictor covariances...\n')
            names = []
            corr_blocks = []
            cov_blocks = []
            for k in sorted(impulse_blocks.keys()):
                all_scalar = True
                for _k in impulse_blocks[k]:
                    if hasattr(impulse_blocks[k][_k], '__len__') and len(impulse_blocks[k][_k]) > 0:
                        all_scalar = False
                        break
                if all_scalar:
                    block = pd.DataFrame({_k: [impulse_blocks[k][_k]] for _k in impulse_blocks[k]})
                else:
                    block = pd.DataFrame(impulse_blocks[k])
                corr_blocks.append(block.corr().values)
                cov_blocks.append(block.cov().values)
                names += list(block.columns)
            corr = scipy.linalg.block_diag(*corr_blocks)
            if corr.shape == (1, 0):
                corr = np.zeros((0, 0))
            corr = pd.DataFrame(corr, index=names, columns=names)
            cov = scipy.linalg.block_diag(*cov_blocks)
            if cov.shape == (1, 0):
                cov = np.zeros((0, 0))
            cov = pd.DataFrame(cov, index=names, columns=names)
            means = pd.DataFrame([self.impulse_means[x] for x in cov.index], index=cov.index, columns=['val'])
            self.impulse_corr = corr
            self.impulse_cov = cov
            self.impulse_sampler_means = means
        else:
            self.impulse_corr = None
            self.impulse_cov = None
            self.impulse_sampler_means = None

        self.response_to_df_ix = {}
        for _response in response_names:
            self.response_to_df_ix[_response] = []
            for i, _Y in enumerate(Y):
                if _response in _Y:
                    self.response_to_df_ix[_response].append(i)

        # Collect stats for temporal features
        stderr('\r    Computing temporal statistics...\n')
        t_deltas = []
        t_delta_maxes = []
        X_time = []
        Y_time = []
        for _Y in Y:
            first_obs, last_obs = get_first_last_obs_lists(_Y)
            _Y_time = _Y.time.values
            Y_time.append(_Y_time)
            for i, cols in enumerate(zip(first_obs, last_obs)):
                if i in impulse_df_ix_unique or (not impulse_df_ix_unique and i == 0):
                    _first_obs, _last_obs = cols
                    _first_obs = np.array(_first_obs, dtype=getattr(np, self.int_type))
                    _last_obs = np.array(_last_obs, dtype=getattr(np, self.int_type))
                    _X_time = np.array(X[i].time, dtype=getattr(np, self.float_type))
                    X_time.append(_X_time)
                    n_cells = len(_first_obs) * self.history_length
                    step = max(1, np.round(n_cells / 1e8))
                    if step > 1:
                        stderr('\r      Dataset too large to compute exact temporal quantiles.' + 
                               '\n      Approximating using %0.02f%% of data.\n' % (1./step * 100))
                    for j, (s, e) in enumerate(zip(_first_obs, _last_obs)):
                        if j % step == 0:
                            _X_time_slice = _X_time[s:e]
                            t_delta = _Y_time[j] - _X_time_slice
                            t_deltas.append(t_delta)
                            t_delta_maxes.append(_Y_time[j] - _X_time[s])
        if not len(X_time):
            X_time = Y_time
        X_time = np.concatenate(X_time, axis=0)
        assert np.all(np.isfinite(X_time)), 'Stimulus sequence contained non-finite timestamps'
        Y_time = np.concatenate(Y_time, axis=0)
        assert np.all(np.isfinite(Y_time)), 'Response sequence contained non-finite timestamps'

        self.X_time_limit = np.quantile(X_time, 0.75)
        self.X_time_quantiles = np.quantile(X_time, q)
        self.X_time_max = X_time.max()
        self.X_time_mean = X_time.mean()
        self.X_time_sd = X_time.std()

        self.Y_time_quantiles = np.quantile(Y_time, q)
        self.Y_time_mean = Y_time.mean()
        self.Y_time_sd = Y_time.std()

        if len(t_deltas):
            t_deltas = np.concatenate(t_deltas, axis=0)
            t_delta_maxes = np.array(t_delta_maxes)
            t_delta_quantiles = np.quantile(t_deltas, q)
            self.t_delta_limit = np.quantile(t_deltas, 0.75)
            self.t_delta_quantiles = t_delta_quantiles
            self.t_delta_max = t_deltas.max()
            self.t_delta_mean_max = t_delta_maxes.mean()
            self.t_delta_mean = t_deltas.mean()
            self.t_delta_sd = t_deltas.std()
        else:
            self.t_delta_limit = self.epsilon
            self.t_delta_quantiles = np.zeros(len(q))
            self.t_delta_max = 0.
            self.t_delta_mean_max = 0.
            self.t_delta_mean = 0.
            self.t_delta_sd = 1.

        ## Set up hash table for random effects lookup
        stderr('\r    Computing random effects statistics...\n')
        self.rangf_map_base = []
        self.rangf_n_levels = []
        for i, gf in enumerate(rangf):
            rangf_counts = {}
            for _Y in Y:
                _rangf_counts = dict(zip(*np.unique(_Y[gf].astype('str'), return_counts=True)))
                for k in _rangf_counts:
                    if k in rangf_counts:
                        rangf_counts[k] += _rangf_counts[k]
                    else:
                        rangf_counts[k] = _rangf_counts[k]

            keys = sorted(list(rangf_counts.keys()))
            counts = np.array([rangf_counts[k] for k in keys])

            sd = counts.std()
            if np.isfinite(sd):
                mu = counts.mean()
                lb = mu - 2 * sd
                too_few = []
                for v, c in zip(keys, counts):
                    if c < lb:
                        too_few.append((v, c))
                if len(too_few) > 0:
                    report = '\nWARNING: Some random effects levels had fewer than 2 standard deviations (%.2f)\nbelow the mean number of data points per level (%.2f):\n' % (
                    sd * 2, mu)
                    for t in too_few:
                        report += ' ' * 4 + str(t[0]) + ': %d\n' % t[1]
                    report += 'Having too few instances for some levels can lead to poor quality random effects estimates.\n'
                    stderr(report)
            vals = np.arange(len(keys), dtype=getattr(np, self.int_type))
            rangf_map = pd.DataFrame({'id': vals}, index=keys).to_dict()['id']
            self.rangf_map_base.append(rangf_map)
            self.rangf_n_levels.append(len(keys) + 1)

        try:
            self.git_hash = subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            ).strip().decode()
            self.pip_version = None
        except (FileNotFoundError, subprocess.CalledProcessError): # CDR not installed via git
            self.git_hash = None
            try:
                pip_info = subprocess.check_output(
                    ["pip", "show", "cdr"]
                ).strip().decode()
                version = None
                for line in pip_info.split('\n'):
                    line = line.strip()
                    if line.startswith('Version:'):
                        version = line.split()[1]
                self.pip_version = version
            except (FileNotFoundError, subprocess.CalledProcessError): # pip not available
                self.pip_version = None

        self._initialize_session()
        tf.keras.backend.set_session(self.session)

        if build:
            self._initialize_metadata()
            self.build()

    def __getstate__(self):
        md = self._pack_metadata()
        return md

    def __setstate__(self, state):
        self._initialize_session()

        self._unpack_metadata(state)
        self._initialize_metadata()

        self.log_graph = False

    def _initialize_session(self):
        self.g = tf.Graph()
        self._session = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        ## Compute secondary data from intialization settings

        assert TF_MAJOR_VERSION == 1 or self.optim_name.lower() != 'nadam', 'Nadam optimizer is not supported when using TensorFlow 2.X.X'

        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        self.prop_bwd = self.history_length / (self.history_length + self.future_length)
        self.prop_fwd = self.future_length / (self.history_length + self.future_length)

        f = self.form
        self.responses = f.responses()
        self.response_names = f.response_names()
        self.has_intercept = f.has_intercept
        self.rangf = f.rangf
        self.ranef_group2ix = {x: i for i, x in enumerate(self.rangf)}

        self.X_weighted_unscaled = {}  # Key order: <response>; Value: nbatch x ntime x ncoef x nparam x ndim tensor of IRF-weighted values at each timepoint of each predictor for each response distribution parameter of the response
        self.X_weighted_unscaled_sumT = {}  # Key order: <response>; Value: nbatch x ncoef x nparam x ndim tensor of IRF-weighted values of each predictor for each response distribution parameter of the response
        self.X_weighted_unscaled_sumK = {}  # Key order: <response>; Value: nbatch x time x nparam x ndim tensor of IRF-weighted values at each timepoint for each response distribution parameter of the response
        self.X_weighted_unscaled_sumTK = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of IRF-weighted values for each response distribution parameter of the response
        self.X_weighted = {}  # Key order: <response>; Value: nbatch x ntime x ncoef x nparam x ndim tensor of IRF-weighted values at each timepoint of each predictor for each response distribution parameter of the response
        self.X_weighted_sumT = {}  # Key order: <response>; Value: nbatch x ncoef x nparam x ndim tensor of IRF-weighted values of each predictor for each response distribution parameter of the response
        self.X_weighted_sumK = {}  # Key order: <response>; Value: nbatch x ntime x nparam x ndim tensor of IRF-weighted values at each timepoint for each response distribution parameter of the response
        self.X_weighted_sumTK = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of IRF-weighted values for each response distribution parameter of the response
        self.layers = [] # List of NN layers
        self.kl_penalties = {} # Key order: <variable>; Value: scalar KL divergence
        self.ema_ops = [] # Container for any exponential moving average updates to run at each training step

        if np.isfinite(self.minibatch_size):
            self.n_train_minibatch = math.ceil(float(self.n_train) / self.minibatch_size)
            self.minibatch_scale = float(self.n_train) / self.minibatch_size
        else:
            self.n_train_minibatch = 1
            self.minibatch_scale = 1
        self.regularizer_losses = []
        self.regularizer_losses_names = []
        self.regularizer_losses_scales = []
        self.regularizer_losses_varnames = []

        # Initialize model metadata

        self.t = self.form.t
        t = self.t
        self.node_table = t.node_table()
        self.coef_names = t.coef_names()
        self.fixed_coef_names = t.fixed_coef_names()
        self.unary_nonparametric_coef_names = t.unary_nonparametric_coef_names()
        self.interaction_list = t.interactions()
        self.interaction_names = t.interaction_names()
        self.fixed_interaction_names = t.fixed_interaction_names()
        self.impulse_names = t.impulse_names(include_interactions=True)
        self.response_names = self.form.response_names()
        self.n_impulse = len(self.impulse_names)
        self.n_response = len(self.response_names)
        self.impulse_names_to_ix = {}
        self.impulse_names_printable = {}
        for i, x in enumerate(self.impulse_names):
            self.impulse_names_to_ix[x] = i
            self.impulse_names_printable[x] = ':'.join([get_irf_name(x, self.irf_name_map) for y in x.split(':')])
        self.terminal_names = t.terminal_names()
        self.terminals_by_name = t.terminals_by_name()
        self.terminal_names_to_ix = {}
        self.terminal_names_printable = {}
        self.non_dirac_impulses = set()
        for i, x in enumerate(self.terminal_names):
            # if self.is_cdrnn or not x.startswith('DiracDelta'):
            if not x.startswith('DiracDelta'):
                for y in self.terminals_by_name[x].impulses():
                    self.non_dirac_impulses.add(y.name())
            self.terminal_names_to_ix[x] = i
            self.terminal_names_printable[x] = ':'.join([get_irf_name(x, self.irf_name_map) for y in x.split(':')])
        self.coef2impulse = t.coef2impulse()
        self.impulse2coef = t.impulse2coef()
        self.coef2terminal = t.coef2terminal()
        self.terminal2coef = t.terminal2coef()
        self.impulse2terminal = t.impulse2terminal()
        self.terminal2impulse = t.terminal2impulse()
        self.interaction2inputs = t.interactions2inputs()
        self.coef_by_rangf = t.coef_by_rangf()
        self.interaction_by_rangf = t.interaction_by_rangf()
        self.interactions_list = t.interactions()
        self.atomic_irf_names_by_family = t.atomic_irf_by_family()
        self.atomic_irf_family_by_name = {}
        for family in self.atomic_irf_names_by_family:
            for id in self.atomic_irf_names_by_family[family]:
                assert id not in self.atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' % id
                self.atomic_irf_family_by_name[id] = family
        self.atomic_irf_param_init_by_family = t.atomic_irf_param_init_by_family()
        self.atomic_irf_param_trainable_by_family = t.atomic_irf_param_trainable_by_family()
        self.irf = {}
        self.nn_irf = {} # Key order: <response, nn_id>
        self.irf_by_rangf = t.irf_by_rangf()
        self.nns_by_id = self.form.nns_by_id

        self.parametric_irf_terminals = [self.node_table[x] for x in self.terminal_names if self.node_table[x].p.family != 'NN']
        self.parametric_irf_terminal_names = [x.name() for x in self.parametric_irf_terminals]

        self.nn_irf_ids = sorted([x for x in self.nns_by_id if self.nns_by_id[x].nn_type == 'irf'])
        self.nn_irf_preterminals = {}
        self.nn_irf_preterminal_names = {}
        self.nn_irf_terminals = {}
        self.nn_irf_terminal_names = {}
        self.nn_irf_impulses = {}
        self.nn_irf_impulse_names = {}
        self.nn_irf_input_names = {}
        self.nn_irf_output_names = {}
        for nn_id in self.nn_irf_ids:
            self.nn_irf_preterminals[nn_id] = self.nns_by_id[nn_id].nodes
            self.nn_irf_preterminal_names[nn_id] = [x.name() for x in self.nn_irf_preterminals[nn_id]]
            self.nn_irf_terminals[nn_id] = [self.node_table[x] for x in self.terminal_names if self.node_table[x].p.name() in self.nn_irf_preterminal_names[nn_id]]
            self.nn_irf_terminal_names[nn_id] = [x.name() for x in self.nn_irf_terminals[nn_id]]
            self.nn_irf_impulses[nn_id] = None
            self.nn_irf_impulse_names[nn_id] = self.nns_by_id[nn_id].all_impulse_names()
            self.nn_irf_input_names[nn_id] = self.nns_by_id[nn_id].input_impulse_names()
            self.nn_irf_output_names[nn_id] = self.nns_by_id[nn_id].output_impulse_names()

        self.nn_impulse_ids = sorted([x for x in self.nns_by_id if self.nns_by_id[x].nn_type == 'impulse'])
        self.nn_impulse_impulses = {}
        self.nn_impulse_impulse_names = {}
        for nn_id in self.nn_impulse_ids:
            self.nn_impulse_impulses[nn_id] = None
            assert len(self.nns_by_id[nn_id].nodes) == 1, 'NN impulses should have exactly 1 associated node. Got %d.' % len(self.nns_by_id[nn_id].nodes)
            self.nn_impulse_impulse_names[nn_id] = self.nns_by_id[nn_id].input_impulse_names()
        self.nn_transformed_impulses = []
        self.nn_transformed_impulse_t_delta = []
        self.nn_transformed_impulse_X_time = []
        self.nn_transformed_impulse_X_mask = []
        self.nn_transformed_impulse_dirac_delta_mask = []

        # Initialize response distribution metadata

        response_distribution = {}
        response_distribution_map = {}
        if self.response_distribution_map is not None:
            _response_distribution_map = self.response_distribution_map.split()
            if len(_response_distribution_map) == 1:
                _response_distribution_map = _response_distribution_map * len(self.response_names)
            has_delim = [';' in x for x in _response_distribution_map]
            assert np.all(has_delim) or (not np.any(has_delim) and len(has_delim) == len(self.response_names)), 'response_distribution must contain a single distribution name, a one-to-one list of distribution names, one per response variable, or a list of ``;``-delimited pairs mapping <response, distribution>.'
            for i, x in enumerate(_response_distribution_map):
                if has_delim[i]:
                    _response, _dist = x.split(';')
                    response_distribution_map[_response] = _dist
                else:
                    response_distribution_map[self.response_names[i]] = x

        for _response in self.response_names:
            if _response in response_distribution_map:
                response_distribution[_response] = self.RESPONSE_DISTRIBUTIONS[response_distribution_map[_response]]
            elif self.response_is_categorical[_response]:
                response_distribution[_response] = self.RESPONSE_DISTRIBUTIONS['categorical']
            else:
                response_distribution[_response] = self.RESPONSE_DISTRIBUTIONS['johnsonsu']
        self.response_distribution_config = response_distribution

        self.response_category_to_ix = self.response_category_maps
        self.response_ix_to_category = {}
        for _response in self.response_category_to_ix:
            self.response_ix_to_category[_response] = {}
            for _cat in self.response_category_to_ix[_response]:
                self.response_ix_to_category[_response][self.response_category_to_ix[_response][_cat]] = _cat

        # Initialize random effects metadata
        # Can't pickle defaultdict because it requires a lambda term for the default value,
        # so instead we pickle a normal dictionary (``rangf_map_base``) and compute the defaultdict
        # from it.

        self.rangf_map = []
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(defaultdict((lambda x: lambda: x)(self.rangf_n_levels[i] - 1), self.rangf_map_base[i]))

        self.rangf_map_ix_2_levelname = []

        for i in range(len(self.rangf_map_base)):
            ix_2_levelname = [None] * self.rangf_n_levels[i]
            for level in self.rangf_map_base[i]:
                ix_2_levelname[self.rangf_map_base[i][level]] = level
            assert ix_2_levelname[-1] is None, 'Non-null value found in rangf map for overall/unknown level'
            ix_2_levelname[-1] = 'Overall'
            self.rangf_map_ix_2_levelname.append(ix_2_levelname)

        self.ranef_ix2level = {}
        self.ranef_level2ix = {}
        ranef_group_names = [None]
        ranef_group_ix = [None]
        ranef_level_names = [None]
        ranef_level_ix = [None]
        for i, gf in enumerate(self.rangf):
            if gf not in self.ranef_ix2level:
                self.ranef_ix2level[gf] = {}
            if gf not in self.ranef_level2ix:
                self.ranef_level2ix[gf] = {}
            if self.has_nn_irf or self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                self.ranef_ix2level[gf][self.rangf_n_levels[i] - 1] = None
                self.ranef_level2ix[gf][None] = self.rangf_n_levels[i] - 1
                for j, k in enumerate(self.rangf_map[i].keys()):
                    self.ranef_ix2level[gf][j] = str(k)
                    self.ranef_level2ix[gf][str(k)] = j
                    ranef_group_names.append(gf)
                    ranef_group_ix.append(self.rangf[i])
                    ranef_level_names.append(str(k))
                    ranef_level_ix.append(self.rangf_map[i][k])
        self.ranef_group_names = ranef_group_names
        self.ranef_level_names = ranef_level_names
        self.ranef_group_ix = ranef_group_ix
        self.ranef_level_ix = ranef_level_ix

        # Initialize objects derived from training data stats

        if self.impulse_df_ix is None:
            self.impulse_df_ix = np.zeros(len(self.impulse_names))
        self.impulse_df_ix = np.array(self.impulse_df_ix, dtype=self.INT_NP)
        self.impulse_df_ix_unique = sorted(list(set(self.impulse_df_ix)))
        self.n_impulse_df = len(self.impulse_df_ix_unique)
        self.impulse_indices = []
        if self.impulse_df_ix_unique:
            max_impulse_df_ix_unique = max(self.impulse_df_ix_unique)
        else:
            max_impulse_df_ix_unique = 0
        for i in range(max_impulse_df_ix_unique + 1):
            arange = np.arange(len(self.form.t.impulses(include_interactions=True)))
            ix = arange[np.where(self.impulse_df_ix == i)[0]]
            self.impulse_indices.append(ix)
        if self.response_to_df_ix is None:
            self.response_to_df_ix = {x: [0] for x in self.response_names}
        self.n_response_df = 0
        for _response in self.response_to_df_ix:
            self.n_response_df = max(self.n_response_df, max(self.response_to_df_ix[_response]))
        self.n_response_df += 1

        if self.impulse_sampler_means is None or not len(self.impulse_sampler_means):
            self.impulse_sampler = None
        else:
            self.impulse_sampler = scipy.stats.multivariate_normal(
                mean=self.impulse_sampler_means.val,
                cov=self.impulse_cov,
                allow_singular=True
            )
        
        impulse_dfs_noninteraction = set()
        terminal_names = [x for x in self.terminal_names if self.node_table[x].p.family == 'NN']
        for x in terminal_names:
            impulse = self.terminal2impulse[x][0]
            is_nn = self.node_table[x].impulse.is_nn_impulse()
            if is_nn:
                for _impulse in self.node_table[x].impulse.impulses():
                    _impulse_name = _impulse.name()
                    ix = self.impulse_names.index(_impulse_name)
                    df_ix = self.impulse_df_ix[ix]
                    impulse_dfs_noninteraction.add(df_ix)
            else:
                ix = self.impulse_names.index(impulse)
                df_ix = self.impulse_df_ix[ix]
                impulse_dfs_noninteraction.add(df_ix)
        self.n_impulse_df_noninteraction = len(impulse_dfs_noninteraction)

        self.use_crossval = bool(self.crossval_factor)
        self.crossval_use_dev_fold = bool(self.crossval_dev_fold)

        for x in self.indicator_names.split():
            self.indicators.add(x)

        m = self.impulse_means
        m = np.array([m[x] for x in self.impulse_names])
        self.impulse_means_arr = m
        while len(m.shape) < 3:
            m = m[None, ...]
        self.impulse_means_arr_expanded = m

        m = self.impulse_means
        m = np.array([0. if x in self.indicators else m[x] for x in self.impulse_names])
        self.impulse_shift_arr = m
        while len(m.shape) < 3:
            m = m[None, ...]
        self.impulse_shift_arr_expanded = m

        s = self.impulse_sds
        s = np.array([s[x] for x in self.impulse_names])
        self.impulse_sds_arr = s
        while len(s.shape) < 3:
            s = s[None, ...]
        self.impulse_sds_arr_expanded = s

        s = self.impulse_sds
        s = np.array([1. if x in self.indicators else s[x] for x in self.impulse_names])
        self.impulse_scale_arr = s
        while len(s.shape) < 3:
            s = s[None, ...]
        self.impulse_scale_arr_expanded = s

        q = self.impulse_quantiles
        q = [q[x] for x in self.impulse_names]
        if len(q):
            q = np.stack(q, axis=1)
        else:
            q = np.zeros((self.N_QUANTILES, 0))
        self.impulse_quantiles_arr = q
        while len(s.shape) < 3:
            q = np.expand_dims(q, axis=1)
        self.impulse_quantiles_arr_expanded = q

        self.reference_map = self.get_reference_map()
        r = self.reference_map
        r = np.array([r[x] for x in self.impulse_names])
        self.reference_arr = r

        self.plot_step_map = self.get_plot_step_map()
        s = self.plot_step_map
        s = np.array([s[x] for x in self.impulse_names])
        self.plot_step_arr = s

        self._initialize_nn_metadata()

        # NN IRF layers/transforms
        self.input_dropout_layer = {}
        self.X_time_dropout_layer = {}
        self.ff_layers = {}
        self.ff_fn = {}
        self.rnn_layers = {}
        self.rnn_h_ema = {}
        self.rnn_c_ema = {}
        self.rnn_encoder = {}
        self.rnn_projection_layers = {}
        self.rnn_projection_fn = {}
        self.h_rnn_dropout_layer = {}
        self.rnn_dropout_layer = {}
        self.nn_irf_l1 = {}
        self.nn_irf_layers = {}
        self.nn_t_delta_scale = {}
        self.nn_t_delta_shift = {}

        with self.session.as_default():
            with self.session.graph.as_default():

                # Initialize constraint functions

                self.constraint_fn, \
                self.constraint_fn_np, \
                self.constraint_fn_inv, \
                self.constraint_fn_inv_np = get_constraint(self.constraint)

                # Initialize variational metadata

                model_components = {'intercept', 'coefficient', 'irf_param', 'interaction', 'nn'}
                if self.random_variables.strip().lower() == 'all':
                    self.rvs = model_components
                elif self.random_variables.strip().lower() == 'none':
                    self.rvs = set()
                elif self.random_variables.strip().lower() == 'default':
                    self.rvs = set([x for x in model_components if x != 'nn'])
                else:
                    self.rvs = set()
                    for x in self.random_variables.strip().split():
                        if x in model_components:
                            self.rvs.add(x)
                        else:
                            stderr('WARNING: Unrecognized random variable value "%s". Skipping...\n' % x)

                if 'nn' not in self.rvs and self.weight_sd_init in (None, 'None'):
                    self.weight_sd_init = 'glorot'

                if self.is_bayesian:
                    self._intercept_prior_sd, \
                    self._intercept_posterior_sd_init, \
                    self._intercept_ranef_prior_sd, \
                    self._intercept_ranef_posterior_sd_init = self._process_prior_sd(self.intercept_prior_sd)

                    self._coef_prior_sd, \
                    self._coef_posterior_sd_init, \
                    self._coef_ranef_prior_sd, \
                    self._coef_ranef_posterior_sd_init = self._process_prior_sd(self.coef_prior_sd)

                    assert isinstance(self.irf_param_prior_sd, str) or isinstance(self.irf_param_prior_sd, float), 'irf_param_prior_sd must either be a string or a float'

                    self._irf_param_prior_sd, \
                    self._irf_param_posterior_sd_init, \
                    self._irf_param_ranef_prior_sd, \
                    self._irf_param_ranef_posterior_sd_init = self._process_prior_sd(self.irf_param_prior_sd)

                # Initialize intercept initial values

                self.intercept_init = {}
                for _response in self.response_names:
                    self.intercept_init[_response] = self._get_intercept_init(
                        _response,
                        has_intercept=self.has_intercept[None]
                    )

                # Initialize convergence checking

                if self.convergence_n_iterates and self.convergence_alpha is not None:
                    self.d0 = []
                    self.d0_names = []
                    self.d0_saved = []
                    self.d0_saved_update = []
                    self.d0_assign = []

                    convergence_stride = self.convergence_stride
                    if self.early_stopping and self.eval_freq > 0:
                        convergence_stride *= self.eval_freq

                    self.convergence_history = tf.Variable(
                        tf.zeros([int(self.convergence_n_iterates / convergence_stride), 1]), trainable=False,
                        dtype=self.FLOAT_NP, name='convergence_history')
                    self.convergence_history_update = tf.placeholder(self.FLOAT_TF, shape=[
                        int(self.convergence_n_iterates / convergence_stride), 1],
                                                                     name='convergence_history_update')
                    self.convergence_history_assign = tf.assign(self.convergence_history,
                                                                self.convergence_history_update)
                    self.proportion_converged = tf.reduce_mean(self.convergence_history)

                    self.last_convergence_check = tf.Variable(0, trainable=False, dtype=self.INT_NP,
                                                              name='last_convergence_check')
                    self.last_convergence_check_update = tf.placeholder(self.INT_NP, shape=[],
                                                                        name='last_convergence_check_update')
                    self.last_convergence_check_assign = tf.assign(self.last_convergence_check,
                                                                   self.last_convergence_check_update)
                    self.check_convergence = True
                else:
                    self.check_convergence = False

        self.predict_mode = False
        
    def _initialize_nn_metadata(self):
        self.nn_meta = {}
        nn_ids = [None] + list(self.nns_by_id.keys())
        for nn_id in nn_ids:
            if nn_id is None:
                nn_meta = {}
            else:
                nn_meta = self.nns_by_id[nn_id].nn_config.copy()
                for key in nn_meta:
                    nn_meta[key] = NN_KWARG_BY_KEY[key].kwarg_from_config(nn_meta, is_cdrnn=True)
            self.nn_meta[nn_id] = nn_meta
            nn_meta['use_batch_normalization'] = bool(self.get_nn_meta('batch_normalization_decay', nn_id))
            nn_meta['use_layer_normalization'] = bool(self.get_nn_meta('layer_normalization_type', nn_id))
            assert not (self.get_nn_meta('use_batch_normalization', nn_id) and self.get_nn_meta('use_layer_normalization', nn_id)), 'Cannot batch normalize and layer normalize the same model.'
            nn_meta['normalize_activations'] = nn_meta['use_batch_normalization'] or nn_meta['use_layer_normalization']

            n_units_ff = self.get_nn_meta('n_units_ff', nn_id)
            n_layers_ff = self.get_nn_meta('n_layers_ff', nn_id)

            if n_units_ff:
                if isinstance(n_units_ff, str):
                    if n_units_ff.lower() == 'infer':
                        nn_meta['n_units_ff'] = [len(self.terminal_names) + len(self.ablated)]
                    else:
                        nn_meta['n_units_ff'] = [int(x) for x in n_units_ff.split()]
                elif isinstance(n_units_ff, int):
                    if n_layers_ff is None:
                        nn_meta['n_units_ff'] = [n_units_ff]
                    else:
                        nn_meta['n_units_ff'] = [n_units_ff] * n_layers_ff
                else:
                    nn_meta['n_units_ff'] = n_units_ff
                if n_layers_ff is None:
                    nn_meta['n_layers_ff'] = len(nn_meta['n_units_ff'])
                else:
                    nn_meta['n_layers_ff'] = n_layers_ff
                if len(nn_meta['n_units_ff']) == 1 and nn_meta['n_layers_ff'] != 1:
                    nn_meta['n_units_ff'] = nn_meta['n_units_ff'] * nn_meta['n_layers_ff']
            else:
                nn_meta['n_units_ff'] = []
                nn_meta['n_layers_ff'] = 0
            assert nn_meta['n_layers_ff'] == len(nn_meta['n_units_ff']), 'Inferred n_layers_ff and n_units_ff must have the same number of layers. Saw %d and %d, respectively.' % (nn_meta['n_layers_ff'], len(nn_meta['n_units_ff']))

            n_units_rnn = self.get_nn_meta('n_units_rnn', nn_id)
            n_layers_rnn = self.get_nn_meta('n_layers_rnn', nn_id)

            if n_units_rnn:
                if isinstance(n_units_rnn, str):
                    if n_units_rnn.lower() == 'infer':
                        nn_meta['n_units_rnn'] = [len(self.terminal_names) + len(self.ablated)]
                    elif n_units_rnn.lower() == 'inherit':
                        nn_meta['n_units_rnn'] = ['inherit']
                    else:
                        nn_meta['n_units_rnn'] = [int(x) for x in n_units_rnn.split()]
                elif isinstance(n_units_rnn, int):
                    if n_layers_rnn is None:
                        nn_meta['n_units_rnn'] = [n_units_rnn]
                    else:
                        nn_meta['n_units_rnn'] = [n_units_rnn] * n_layers_rnn
                else:
                    nn_meta['n_units_rnn'] = n_units_rnn
                if n_layers_rnn is None:
                    nn_meta['n_layers_rnn'] = len(nn_meta['n_units_rnn'])
                else:
                    nn_meta['n_layers_rnn'] = n_layers_rnn
                if len(nn_meta['n_units_rnn']) == 1 and nn_meta['n_layers_rnn'] != 1:
                    nn_meta['n_units_rnn'] = nn_meta['n_units_rnn'] * nn_meta['n_layers_rnn']
            else:
                nn_meta['n_units_rnn'] = []
                nn_meta['n_layers_rnn'] = 0
            assert nn_meta['n_layers_rnn'] == len(nn_meta['n_units_rnn']), 'Inferred n_layers_rnn and n_units_rnn must have the same number of layers. Saw %d and %d, respectively.' % (nn_meta['n_layers_rnn'], len(nn_meta['n_units_rnn']))

            n_units_rnn_projection = self.get_nn_meta('n_units_rnn_projection', nn_id)
            n_layers_rnn_projection = self.get_nn_meta('n_layers_rnn_projection', nn_id)

            if n_units_rnn_projection:
                if isinstance(n_units_rnn_projection, str):
                    if n_units_rnn_projection.lower() == 'infer':
                        nn_meta['n_units_rnn_projection'] = [len(self.terminal_names) + len(self.ablated)]
                    else:
                        nn_meta['n_units_rnn_projection'] = [int(x) for x in n_units_rnn_projection.split()]
                elif isinstance(n_units_rnn_projection, int):
                    if n_layers_rnn_projection is None:
                        nn_meta['n_units_rnn_projection'] = [n_units_rnn_projection]
                    else:
                        nn_meta['n_units_rnn_projection'] = [n_units_rnn_projection] * n_layers_rnn_projection
                else:
                    nn_meta['n_units_rnn_projection'] = n_units_rnn_projection
                if n_layers_rnn_projection is None:
                    nn_meta['n_layers_rnn_projection'] = len(nn_meta['n_units_rnn_projection'])
                else:
                    nn_meta['n_layers_rnn_projection'] = n_layers_rnn_projection
                if len(nn_meta['n_units_rnn_projection']) == 1 and nn_meta['n_layers_rnn_projection'] != 1:
                    nn_meta['n_units_rnn_projection'] = nn_meta['n_units_rnn_projection'] * nn_meta['n_layers_rnn_projection']
            else:
                nn_meta['n_units_rnn_projection'] = []
                nn_meta['n_layers_rnn_projection'] = 0
            assert nn_meta['n_layers_rnn_projection'] == len(nn_meta['n_units_rnn_projection']), 'Inferred n_layers_rnn_projection and n_units_rnn_projection must have the same number of layers. Saw %d and %d, respectively.' % (nn_meta['n_layers_rnn_projection'], len(nn_meta['n_units_rnn_projection']))

            n_units_irf = self.get_nn_meta('n_units_irf', nn_id)
            n_layers_irf = self.get_nn_meta('n_layers_irf', nn_id)

            if n_units_irf:
                if isinstance(n_units_irf, str):
                    if n_units_irf.lower() == 'infer':
                        nn_meta['n_units_irf'] = [len(self.terminal_names) + len(self.ablated)]
                    else:
                        nn_meta['n_units_irf'] = [int(x) for x in n_units_irf.split()]
                elif isinstance(n_units_irf, int):
                    if n_layers_irf is None:
                        nn_meta['n_units_irf'] = [n_units_irf]
                    else:
                        nn_meta['n_units_irf'] = [n_units_irf] * n_layers_irf
                else:
                    nn_meta['n_units_irf'] = n_units_irf
                if n_layers_irf is None:
                    nn_meta['n_layers_irf'] = len(nn_meta['n_units_irf'])
                else:
                    nn_meta['n_layers_irf'] = n_layers_irf
                if len(nn_meta['n_units_irf']) == 1 and nn_meta['n_layers_irf'] != 1:
                    nn_meta['n_units_irf'] = nn_meta['n_units_irf'] * nn_meta['n_layers_irf']
            else:
                nn_meta['n_units_irf'] = []
                nn_meta['n_layers_irf'] = 0
            assert nn_meta['n_layers_irf'] == len(nn_meta['n_units_irf']), 'Inferred n_layers_irf and n_units_irf must have the same number of layers. Saw %d and %d, respectively.' % (nn_meta['n_layers_irf'], len(nn_meta['n_units_irf']))

            if nn_meta['n_units_rnn'] and nn_meta['n_units_rnn'][-1] == 'inherit':
                if nn_meta['n_units_ff']:
                    nn_meta['n_units_rnn'] = nn_meta['n_units_ff'][0]
                if nn_meta['n_units_irf']:
                    nn_meta['n_units_rnn'] = nn_meta['n_units_irf'][0]
                else:
                    nn_meta['n_units_rnn'] = [len(self.terminal_names) + len(self.ablated)]

            if nn_id is None:
                nn_meta['response_params'] = None
            else:
                nn_meta['response_params'] = self.nns_by_id[nn_id].response_params

            if self.has_nn_irf and len(self.interaction_names) and self.get_nn_meta('input_dependent_irf', nn_id):
                stderr('WARNING: Be careful about interaction terms in models with input-dependent neural net IRFs. Interactions can be implicit in such models (if one or more variables are present in both the input to the NN and the interaction), rendering explicit interaction terms uninterpretable.\n')

    def _pack_metadata(self):
        md = {
            'form_str': self.form_str,
            'form': self.form,
            'n_train': self.n_train,
            'ablated': self.ablated,
            'Y_train_means': self.Y_train_means,
            'Y_train_sds': self.Y_train_sds,
            'Y_train_quantiles': self.Y_train_quantiles,
            'response_is_categorical': self.response_is_categorical,
            'response_ndim': self.response_ndim,
            'response_category_maps': self.response_category_maps,
            'response_expanded_bounds': self.response_expanded_bounds,
            't_delta_max': self.t_delta_max,
            't_delta_mean_max': self.t_delta_mean_max,
            't_delta_mean': self.t_delta_mean,
            't_delta_sd': self.t_delta_sd,
            't_delta_quantiles': self.t_delta_quantiles,
            't_delta_limit': self.t_delta_limit,
            'impulse_df_ix': self.impulse_df_ix,
            'response_to_df_ix': self.response_to_df_ix,
            'X_time_max': self.X_time_max,
            'X_time_mean': self.X_time_mean,
            'X_time_sd': self.X_time_sd,
            'X_time_quantiles': self.X_time_quantiles,
            'X_time_limit': self.X_time_limit,
            'Y_time_mean': self.Y_time_mean,
            'Y_time_sd': self.Y_time_sd,
            'Y_time_quantiles': self.Y_time_quantiles,
            'rangf_map_base': self.rangf_map_base,
            'rangf_n_levels': self.rangf_n_levels,
            'impulse_means': self.impulse_means,
            'impulse_sds': self.impulse_sds,
            'impulse_medians': self.impulse_medians,
            'impulse_quantiles': self.impulse_quantiles,
            'impulse_lq': self.impulse_lq,
            'impulse_uq': self.impulse_uq,
            'impulse_min': self.impulse_min,
            'impulse_max': self.impulse_max,
            'impulse_corr': self.impulse_corr,
            'impulse_cov': self.impulse_cov,
            'impulse_sampler_means': self.impulse_sampler_means,
            'indicators': self.indicators,
            'outdir': self.outdir,
            'crossval_factor': self.crossval_factor,
            'crossval_folds': self.crossval_folds,
            'crossval_fold': self.crossval_fold,
            'crossval_dev_fold': self.crossval_dev_fold,
            'irf_name_map': self.irf_name_map,
            'git_hash': self.git_hash,
            'pip_version': self.pip_version
        }
        for kwarg in CDRModel._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        self.form_str = md.pop('form_str')
        self.form = md.pop('form', Formula(self.form_str))
        self.n_train = md.pop('n_train')
        self.ablated = md.pop('ablated', set())
        self.Y_train_means = md.pop('Y_train_means', md.pop('y_train_mean', None))
        self.Y_train_sds = md.pop('Y_train_sds', md.pop('y_train_sd', None))
        self.Y_train_quantiles = md.pop('Y_train_quantiles', md.pop('y_train_quantiles', None))
        self.response_is_categorical = md.pop('response_is_categorical', {x: 'False' for x in self.form.response_names()})
        self.response_ndim = md.pop('response_ndim', md.pop('response_n_dim', {x: 1 for x in self.form.response_names()}))
        self.response_category_maps = md.pop('response_category_maps', {x: {} for x in self.form.response_names()})
        self.response_expanded_bounds = md.pop('response_expanded_bounds', {x: (i, i + 1) for i, x in enumerate(self.form.response_names())})
        self.t_delta_max = md.pop('t_delta_max', md.pop('max_tdelta', None))
        self.t_delta_mean_max = md.pop('t_delta_mean_max', self.t_delta_max)
        self.t_delta_sd = md.pop('t_delta_sd', 1.)
        self.t_delta_mean = md.pop('t_delta_mean', 1.)
        self.t_delta_quantiles = md.pop('t_delta_quantiles', None)
        self.t_delta_limit = md.pop('t_delta_limit', self.t_delta_max)
        self.impulse_df_ix = md.pop('impulse_df_ix', None)
        self.response_to_df_ix = md.pop('response_to_df_ix', None)
        self.X_time_max = md.pop('X_time_max', md.pop('time_X_max', md.pop('max_time_X', None)))
        self.X_time_sd = md.pop('X_time_sd', md.pop('time_X_sd', 1.))
        self.X_time_mean = md.pop('X_time_mean', md.pop('time_X_mean', 1.))
        self.X_time_quantiles = md.pop('X_time_quantiles', md.pop('time_X_quantiles', None))
        self.X_time_limit = md.pop('X_time_limit', md.pop('time_X_limit', self.t_delta_max))
        self.Y_time_mean = md.pop('Y_time_mean', md.pop('time_y_mean', 0.))
        self.Y_time_sd = md.pop('Y_time_sd', md.pop('time_y_sd', 1.))
        self.Y_time_quantiles = md.pop('Y_time_quantiles', md.pop('time_y_quantiles', None))
        self.rangf_map_base = md.pop('rangf_map_base')
        self.rangf_n_levels = md.pop('rangf_n_levels')
        self.impulse_means = md.pop('impulse_means', {})
        self.impulse_sds = md.pop('impulse_sds', {})
        self.impulse_medians = md.pop('impulse_medians', {})
        self.impulse_quantiles = md.pop('impulse_quantiles', {})
        self.impulse_lq = md.pop('impulse_lq', {})
        self.impulse_uq = md.pop('impulse_uq', {})
        self.impulse_min = md.pop('impulse_min', {})
        self.impulse_max = md.pop('impulse_max', {})
        self.impulse_corr = md.pop('impulse_corr', None)
        self.impulse_cov = md.pop('impulse_cov', None)
        self.impulse_sampler_means = md.pop('impulse_sampler_means', None)
        self.indicators = md.pop('indicators', set())
        self.outdir = md.pop('outdir', './cdr_model/')
        self.crossval_factor = md.pop('crossval_factor', None)
        self.crossval_folds = md.pop('crossval_folds', [])
        self.crossval_fold = md.pop('crossval_fold', None)
        self.crossval_dev_fold = md.pop('crossval_dev_fold', None)
        self.irf_name_map = md.pop('irf_name_map', {})
        self.git_hash = md.pop('git_hash', None)
        self.pip_version = md.pop('pip_version', None)

        # Convert response statistics to vectors if needed (for backward compatibility)
        response_names = [x.name() for x in self.form.responses()]
        if not isinstance(self.Y_train_means, dict):
            self.Y_train_means = {x: self.form.self.Y_train_means[x] for x in response_names}
        if not isinstance(self.Y_train_sds, dict):
            self.Y_train_sds = {x: self.form.self.Y_train_sds[x] for x in response_names}
        if not isinstance(self.Y_train_quantiles, dict):
            self.Y_train_quantiles = {x: self.form.self.Y_train_quantiles[x] for x in response_names}

        for kwarg in CDRModel._INITIALIZATION_KWARGS:
            if kwarg.key == 'response_distribution_map' and 'predictive_distribution_map' in md:
                setattr(self, kwarg.key, md.pop('predictive_distribution_map', kwarg.default_value))
            elif kwarg.key == 'response_dist_epsilon' and 'pred_dist_epsilon' in md:
                setattr(self, kwarg.key, md.pop('pred_dist_epsilon', kwarg.default_value))
            else:
                setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                # Boolean switches
                self.training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name='training')
                self.use_MAP_mode = tf.placeholder_with_default(tf.logical_not(self.training), shape=[], name='use_MAP_mode')
                self.sum_outputs_along_T = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='reduce_preds_along_T')
                self.sum_outputs_along_K = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='reduce_preds_along_K')

                # Impulses
                self.X = tf.placeholder(
                    shape=[None, None, self.n_impulse],
                    dtype=self.FLOAT_TF,
                    name='X'
                )
                X_shape = tf.shape(self.X)
                self.X_batch_dim = X_shape[0]
                self.X_time_dim = X_shape[1]
                X_processed = self.X
                if self.center_inputs:
                    X_processed -= self.impulse_shift_arr_expanded
                if self.rescale_inputs:
                    scale = self.impulse_scale_arr_expanded
                    scale = np.where(scale != 0, scale, 1.)
                    X_processed /= scale
                self.X_processed = X_processed
                self.X_time = tf.placeholder_with_default(
                    tf.zeros(
                        tf.convert_to_tensor([
                            self.X_batch_dim,
                            self.history_length + self.future_length,
                            self.n_impulse
                        ]),
                        dtype=self.FLOAT_TF
                    ),
                    shape=[None, None, self.n_impulse],
                    name='X_time'
                )
                self.X_mask = tf.placeholder_with_default(
                    tf.ones(
                        tf.convert_to_tensor([
                            self.X_batch_dim,
                            self.history_length + self.future_length,
                            self.n_impulse
                        ]),
                        dtype=self.FLOAT_TF
                    ),
                    shape=[None, None, self.n_impulse],
                    name='X_mask'
                )

                # Responses
                self.Y = tf.placeholder(
                    shape=[None, self.n_response],
                    dtype=self.FLOAT_TF,
                    name=sn('Y')
                )
                Y_shape = tf.shape(self.Y)
                self.Y_batch_dim = Y_shape[0]
                self.Y_time = tf.placeholder_with_default(
                    tf.ones(tf.convert_to_tensor([self.Y_batch_dim]), dtype=self.FLOAT_TF),
                    shape=[None],
                    name=sn('Y_time')
                )
                self.Y_mask = tf.placeholder_with_default(
                    tf.ones(tf.convert_to_tensor([self.Y_batch_dim, self.n_response]), dtype=self.FLOAT_TF),
                    shape=[None, self.n_response],
                    name='Y_mask'
                )

                # Compute tensor of temporal offsets
                # shape (B,)
                _Y_time = self.Y_time
                # shape (B, 1, 1)
                _Y_time = _Y_time[..., None, None]
                # shape (B, T, n_impulse)
                _X_time = self.X_time
                # shape (B, T, n_impulse)
                t_delta = _Y_time - _X_time
                if self.history_length and not self.future_length:
                    # Floating point precision issues can allow the response to precede the impulse for simultaneous x/y,
                    # which can break causal IRFs where t_delta must be >= 0. The correction below prevents this.
                    t_delta = tf.maximum(t_delta, 0)
                self.t_delta = t_delta
                self.gf_defaults = np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                self.Y_gf = tf.placeholder_with_default(
                    tf.cast(self.gf_defaults, dtype=self.INT_TF),
                    shape=[None, len(self.rangf)],
                    name='Y_gf'
                )
                self.Y_gf_dropout = {}
                self.ranef_dropout_layer = {}
                for nn_id in self.nns_by_id:
                    ranef_dropout_rate = self.get_nn_meta('ranef_dropout_rate', nn_id)
                    if ranef_dropout_rate and not ranef_dropout_rate in self.Y_gf_dropout:
                        ranef_dropout_layer = get_dropout(
                            ranef_dropout_rate,
                            training=self.training,
                            use_MAP_mode=tf.constant(True, dtype=tf.bool),
                            rescale=False,
                            constant=self.gf_defaults,
                            name='ranef_dropout',
                            session=self.session
                        )
                        self.ranef_dropout_layer[ranef_dropout_rate] = ranef_dropout_layer
                        self.Y_gf_dropout[ranef_dropout_rate] = ranef_dropout_layer(self.Y_gf)

                self.dirac_delta_mask = tf.cast(
                    tf.abs(self.t_delta) < self.epsilon,
                    self.FLOAT_TF
                )

                self.max_tdelta_batch = tf.reduce_max(self.t_delta)

                # Tensor used for interpolated IRF composition
                self.interpolation_support = tf.linspace(0., self.max_tdelta_batch, self.n_interp)[..., None]

                # Linspace tensor used for plotting
                self.support_start = tf.placeholder_with_default(
                    tf.cast(0., self.FLOAT_TF),
                    shape=[],
                    name='support_start'
                )
                self.n_time_units = tf.placeholder_with_default(
                    tf.cast(self.t_delta_limit, self.FLOAT_TF),
                    shape=[],
                    name='n_time_units'
                )
                self.n_time_points = tf.placeholder_with_default(
                    tf.cast(self.interp_hz, self.FLOAT_TF),
                    shape=[],
                    name='n_time_points'
                )
                self.support = tf.lin_space(
                    self.support_start,
                    self.n_time_units+self.support_start,
                    tf.cast(self.n_time_points, self.INT_TF) + 1,
                    name='support'
                )[..., None]
                self.support = tf.cast(self.support, dtype=self.FLOAT_TF)
                self.dd_support = tf.concat(
                    [
                        tf.ones((1, 1), dtype=self.FLOAT_TF),
                        tf.zeros(tf.convert_to_tensor([tf.shape(self.support)[0] - 1, 1]), dtype=self.FLOAT_TF)
                    ],
                    axis=0
                )

                # Error vector for probability plotting
                self.errors = {}
                self.n_errors = {}
                for response in self.response_names:
                    if self.is_real(response):
                        self.errors[response] = tf.placeholder(
                            self.FLOAT_TF,
                            shape=[None],
                            name='errors_%s' % sn(response)
                        )
                        self.n_errors[response] = tf.placeholder(
                            self.INT_TF,
                            shape=[],
                            name='n_errors_%s' % sn(response)
                        )

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

                self.training_wall_time = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.FLOAT_TF,
                    name='training_wall_time'
                )
                self.training_wall_time_in = tf.placeholder(
                    self.FLOAT_TF,
                    shape=[],
                    name='training_wall_time_in'
                )
                self.set_training_wall_time_op = tf.assign_add(self.training_wall_time, self.training_wall_time_in)

                self.training_complete = tf.Variable(
                    False,
                    trainable=False,
                    dtype=tf.bool,
                    name='training_complete'
                )
                self.training_complete_true = tf.assign(self.training_complete, True)
                self.training_complete_false = tf.assign(self.training_complete, False)

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                self.reg_loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='reg_loss_total')
                if self.is_bayesian:
                    self.kl_loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='kl_loss_total')
                self.n_dropped_in = tf.placeholder(shape=[], dtype=self.INT_TF, name='n_dropped_in')
                if self.eval_freq > 0:
                    self.dev_ll_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='dev_ll_total')
                    self.dev_ll_max = tf.Variable(-np.inf, trainable=False)
                    self.set_dev_ll_max = tf.assign(self.dev_ll_max, tf.maximum(self.dev_ll_total, self.dev_ll_max))
                    self.dev_metrics = {}
                    if len(self.response_names) > 1:
                        self.dev_metrics['full_log_lik'] = self.dev_ll_total
                    for response in self.response_names:
                        file_ix = self.response_to_df_ix[response]
                        multiple_files = len(file_ix) > 1
                        suffix = response
                        for ix in file_ix:
                            if multiple_files:
                                suffix = response + '_file%d' % ix
                            else:
                                suffix = response
                            if 'log_lik' not in self.dev_metrics:
                                self.dev_metrics['log_lik'] = {}
                            if response not in self.dev_metrics['log_lik']:
                                self.dev_metrics['log_lik'][response] = {}
                            self.dev_metrics['log_lik'][response][ix] = tf.placeholder(
                                shape=[], dtype=self.FLOAT_TF, name=sn('log_lik_%s' % suffix)
                            )
                            if self.is_binary(response) or self.is_categorical(response):
                                if 'f1' not in self.dev_metrics:
                                    self.dev_metrics['f1'] = {}
                                if response not in self.dev_metrics['f1']:
                                    self.dev_metrics['f1'][response] = {}
                                self.dev_metrics['f1'][response][ix] = tf.placeholder(
                                    shape=[], dtype=self.FLOAT_TF, name=sn('f1_%s' % suffix)
                                )
                                if 'acc' not in self.dev_metrics:
                                    self.dev_metrics['acc'] = {}
                                if response not in self.dev_metrics['acc']:
                                    self.dev_metrics['acc'][response] = {}
                                self.dev_metrics['acc'][response][ix] = tf.placeholder(
                                    shape=[], dtype=self.FLOAT_TF, name=sn('acc_%s' % suffix)
                                )
                            else:
                                if 'mse' not in self.dev_metrics:
                                    self.dev_metrics['mse'] = {}
                                if response not in self.dev_metrics['mse']:
                                    self.dev_metrics['mse'][response] = {}
                                self.dev_metrics['mse'][response][ix] = tf.placeholder(
                                    shape=[], dtype=self.FLOAT_TF, name=sn('mse_%s' % suffix)
                                )
                                if 'rho' not in self.dev_metrics:
                                    self.dev_metrics['rho'] = {}
                                if response not in self.dev_metrics['rho']:
                                    self.dev_metrics['rho'][response] = {}
                                self.dev_metrics['rho'][response][ix] = tf.placeholder(
                                    shape=[], dtype=self.FLOAT_TF, name=sn('rho_%s' % suffix)
                                )
                                if 'percent_variance_explained' not in self.dev_metrics:
                                    self.dev_metrics['percent_variance_explained'] = {}
                                if response not in self.dev_metrics['percent_variance_explained']:
                                    self.dev_metrics['percent_variance_explained'][response] = {}
                                self.dev_metrics['percent_variance_explained'][response][ix] = tf.placeholder(
                                    shape=[], dtype=self.FLOAT_TF, name=sn('percent_variance_explained_%s' % suffix)
                                )

                # Initialize vars for saving training set stats upon completion.
                # Allows these numbers to be reported in later summaries without access to the training data.
                self.training_loglik_full_in = tf.placeholder(
                    self.FLOAT_TF,
                    shape=[],
                    name='training_loglik_full_in'
                )
                self.training_loglik_full_tf = tf.Variable(
                    np.nan,
                    dtype=self.FLOAT_TF,
                    trainable=False,
                    name='training_loglik_full'
                )
                self.set_training_loglik_full = tf.assign(self.training_loglik_full_tf, self.training_loglik_full_in)
                self.training_loglik_in = {}
                self.training_loglik_tf = {}
                self.set_training_loglik = {}
                self.training_mse_in = {}
                self.training_mse_tf = {}
                self.set_training_mse = {}
                self.training_percent_variance_explained_tf = {}
                self.training_rho_in = {}
                self.training_rho_tf = {}
                self.set_training_rho = {}
                for response in self.response_names:
                    # log likelihood
                    self.training_loglik_in[response] = {}
                    self.training_loglik_tf[response] = {}
                    self.set_training_loglik[response] = {}

                    file_ix = self.response_to_df_ix[response]
                    multiple_files = len(file_ix) > 1
                    for ix in file_ix:
                        if multiple_files:
                            name_base = '%s_f%s' % (sn(response), ix + 1)
                        else:
                            name_base = sn(response)
                        self.training_loglik_in[response][ix] = tf.placeholder(
                            self.FLOAT_TF,
                            shape=[],
                            name='training_loglik_in_%s' % name_base
                        )
                        self.training_loglik_tf[response][ix] = tf.Variable(
                            np.nan,
                            dtype=self.FLOAT_TF,
                            trainable=False,
                            name='training_loglik_%s' % name_base
                        )
                        self.set_training_loglik[response][ix] = tf.assign(
                            self.training_loglik_tf[response][ix],
                            self.training_loglik_in[response][ix]
                        )

                    if self.is_real(response):
                        self.training_mse_in[response] = {}
                        self.training_mse_tf[response] = {}
                        self.set_training_mse[response] = {}
                        self.training_percent_variance_explained_tf[response] = {}
                        self.training_rho_in[response] = {}
                        self.training_rho_tf[response] = {}
                        self.set_training_rho[response] = {}

                        for ix in range(self.n_response_df):
                            # MSE
                            self.training_mse_in[response][ix] = tf.placeholder(
                                self.FLOAT_TF,
                                shape=[],
                                name='training_mse_in_%s' % name_base
                            )
                            self.training_mse_tf[response][ix] = tf.Variable(
                                np.nan,
                                dtype=self.FLOAT_TF,
                                trainable=False,
                                name='training_mse_%s' % name_base
                            )
                            self.set_training_mse[response][ix] = tf.assign(
                                self.training_mse_tf[response][ix],
                                self.training_mse_in[response][ix]
                            )

                            # % variance explained
                            full_variance = self.Y_train_sds[response] ** 2
                            if self.get_response_ndim(response) == 1:
                                full_variance = np.squeeze(full_variance, axis=-1)
                            self.training_percent_variance_explained_tf[response][ix] = tf.maximum(
                                0.,
                                (1. - self.training_mse_tf[response][ix] / full_variance) * 100.
                            )

                            # rho
                            self.training_rho_in[response][ix] = tf.placeholder(
                                self.FLOAT_TF,
                                shape=[], name='training_rho_in_%s' % name_base
                            )
                            self.training_rho_tf[response][ix] = tf.Variable(
                                np.nan,
                                dtype=self.FLOAT_TF,
                                trainable=False,
                                name='training_rho_%s' % name_base
                            )
                            self.set_training_rho[response][ix] = tf.assign(
                                self.training_rho_tf[response][ix],
                                self.training_rho_in[response][ix]
                            )

                # convergence
                self._add_convergence_tracker(self.loss_total, 'loss_total')
                self.converged_in = tf.placeholder(tf.bool, shape=[], name='converged_in')
                self.converged = tf.Variable(False, trainable=False, dtype=tf.bool, name='converged')
                self.set_converged = tf.assign(self.converged, self.converged_in)

                # Initialize regularizers
                self.regularizer = self._initialize_regularizer(
                    self.regularizer_name,
                    self.regularizer_scale
                )
                self.intercept_regularizer = self._initialize_regularizer(
                    self.intercept_regularizer_name,
                    self.intercept_regularizer_scale
                )
                self.coefficient_regularizer = self._initialize_regularizer(
                    self.coefficient_regularizer_name,
                    self.coefficient_regularizer_scale
                )
                self.irf_regularizer = self._initialize_regularizer(
                    self.irf_regularizer_name,
                    self.irf_regularizer_scale
                )
                self.ranef_regularizer = self._initialize_regularizer(
                    self.ranef_regularizer_name,
                    self.ranef_regularizer_scale
                )
                self.nn_regularizer = {}
                self.ff_regularizer = {}
                self.rnn_projection_regularizer = {}
                self.activity_regularizer = {}
                self.context_regularizer = {}
                for nn_id in self.nns_by_id:
                    nn_regularizer_name = self.get_nn_meta('nn_regularizer_name', nn_id)
                    nn_regularizer_scale = self.get_nn_meta('nn_regularizer_scale', nn_id)
                    self.nn_regularizer[nn_id] = self._initialize_regularizer(
                        nn_regularizer_name,
                        nn_regularizer_scale
                    )

                    ff_regularizer_name = self.get_nn_meta('ff_regularizer_name', nn_id)
                    ff_regularizer_scale = self.get_nn_meta('ff_regularizer_scale', nn_id)
                    if ff_regularizer_name is None:
                        ff_regularizer_name = nn_regularizer_name
                    if ff_regularizer_scale is None:
                        ff_regularizer_scale = nn_regularizer_scale
                    self.ff_regularizer[nn_id] = self._initialize_regularizer(
                        ff_regularizer_name,
                        ff_regularizer_scale
                    )

                    rnn_projection_regularizer_name = self.get_nn_meta('rnn_projection_regularizer_name', nn_id)
                    rnn_projection_regularizer_scale = self.get_nn_meta('rnn_projection_regularizer_scale', nn_id)
                    if rnn_projection_regularizer_name is None:
                        rnn_projection_regularizer_name = nn_regularizer_name
                    if rnn_projection_regularizer_scale is None:
                        rnn_projection_regularizer_scale = nn_regularizer_scale
                    self.rnn_projection_regularizer[nn_id] = self._initialize_regularizer(
                        rnn_projection_regularizer_name,
                        rnn_projection_regularizer_scale
                    )

                    activity_regularizer_name = self.get_nn_meta('activity_regularizer_name', nn_id)
                    activity_regularizer_scale = self.get_nn_meta('activity_regularizer_scale', nn_id)
                    if activity_regularizer_name is None:
                        self.activity_regularizer[nn_id] = None
                    elif activity_regularizer_name == 'inherit':
                        self.activity_regularizer[nn_id] = self.regularizer
                    else:
                        if self.regularize_mean:
                            scale = activity_regularizer_scale
                        else:
                            scale = activity_regularizer_scale / (
                                (self.history_length + self.future_length) * max(1, self.n_impulse_df_noninteraction)
                            ) # Average over time
                        self.activity_regularizer[nn_id] = self._initialize_regularizer(
                            activity_regularizer_name,
                            scale,
                            per_item=True
                        )
                        
                    context_regularizer_name = self.get_nn_meta('context_regularizer_name', nn_id)
                    context_regularizer_scale = self.get_nn_meta('context_regularizer_scale', nn_id)
                    if context_regularizer_name is None:
                        self.context_regularizer[nn_id] = None
                    elif context_regularizer_name == 'inherit':
                        self.context_regularizer[nn_id] = self.regularizer
                    else:
                        if self.regularize_mean:
                            scale = context_regularizer_scale
                        else:
                            scale = context_regularizer_scale / (
                                (self.history_length + self.future_length) * max(1, self.n_impulse_df_noninteraction)
                            ) # Average over time
                        self.context_regularizer[nn_id] = self._initialize_regularizer(
                            context_regularizer_name,
                            scale,
                            per_item=True
                        )

                self.resample_ops = [] # Only used by CDRNN, defined here for global API
                self.regularizable_layers = {} # Only used by CDRNN, defined here for global API

    def _get_prior_sd(self, response_name):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = []
                ndim = self.get_response_ndim(response_name)
                for param in self.get_response_params(response_name):
                    out.append(np.ones((1, ndim)))

                out = np.concatenate(out)

                return out

    def _process_prior_sd(self, prior_sd_in):
        prior_sd = {}
        if isinstance(prior_sd_in, str):
            _prior_sd = prior_sd_in.split()
            for i, x in enumerate(_prior_sd):
                _response = self.response_names[i]
                nparam = self.get_response_nparam(_response)
                ndim = self.get_response_ndim(_response)
                _param_sds = x.split(';')
                assert len(_param_sds) == nparam, 'Expected %d priors for the %s response to variable %s, got %d.' % (nparam, self.get_response_dist_name(_response), _response, len(_param_sds))
                _prior_sd = np.array([float(_param_sd) for _param_sd in _param_sds])
                _prior_sd = _prior_sd[..., None] * np.ones([1, ndim])
                prior_sd[_response] = _prior_sd
        elif isinstance(prior_sd_in, float):
            for _response in self.response_names:
                nparam = self.get_response_nparam(_response)
                ndim = self.get_response_ndim(_response)
                prior_sd[_response] = np.ones([nparam, ndim]) * prior_sd_in
        elif prior_sd_in is None:
            for _response in self.response_names:
                prior_sd[_response] = self._get_prior_sd(_response)
        else:
            raise ValueError('Unsupported type %s found for prior_sd.' % type(prior_sd_in))
        for _response in self.response_names:
            assert _response in prior_sd, 'No entry for response %s provided in prior_sd' % _response

        posterior_sd_init = {x: prior_sd[x] * self.posterior_to_prior_sd_ratio for x in prior_sd}
        ranef_prior_sd = {x: prior_sd[x] * self.ranef_to_fixef_prior_sd_ratio for x in prior_sd}
        ranef_posterior_sd_init = {x: posterior_sd_init[x] * self.ranef_to_fixef_prior_sd_ratio for x in posterior_sd_init}

        # outputs all have shape [nparam, ndim]

        return prior_sd, posterior_sd_init, ranef_prior_sd, ranef_posterior_sd_init

    def _get_intercept_init(self, response_name, has_intercept=True):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = []
                ndim = self.get_response_ndim(response_name)
                for param in self.get_response_params(response_name):
                    if self.get_response_dist_name(response_name).startswith('lognormal'):
                        # Given response mean m and standard deviation s,
                        # initialize lognormal mu and sigma to instantiate the
                        # desired mean and variance of the distribution.
                        mean = self.Y_train_means[response_name]
                        sd = self.Y_train_sds[response_name]
                        if self.get_response_dist_name(response_name) == 'lognormal':
                            m = mean / sd
                            s = 1
                            sigma = np.sqrt(np.log((s / m) ** 2 + 1))
                            mu = np.log(m) + (sigma ** 2) / 2
                        else:  # self.get_response_dist_name(response_name) == 'lognormalv2'
                            mu = self.constraint_fn_inv_np(mean / sd)
                            sigma = np.ones_like(mu)
                        sigma = sigma[None, ...]
                        mu = mu[None, ...]
                    else:
                        mu = np.zeros((1, ndim))
                        sigma = np.ones((1, ndim))
                    if param == 'mu':
                        _out = mu
                    elif param == 'sigma':
                        _out = self.constraint_fn_inv_np(sigma)
                    elif param in ['beta', 'tailweight']:
                        _out = self.constraint_fn_inv_np(np.ones((1, ndim)))
                    elif param == 'skewness':
                        _out = np.zeros((1, ndim))
                    elif param == 'logit':
                        if has_intercept:
                            _out = np.log(self.Y_train_means[response_name][None, ...])
                        else:
                            _out = np.zeros((1, ndim))
                    else:
                        raise ValueError('Unrecognized response distributional parameter %s.' % param)

                    out.append(_out)

                out = np.concatenate(out, axis=0)

                return out

    def _get_nonparametric_irf_params(self, family):
        param_names = []
        param_kwargs = []
        bases = Formula.bases(family)
        x_init = np.zeros(bases)

        for _param_name in Formula.irf_params(family):
            _param_kwargs = {}
            if _param_name.startswith('x'):
                n = int(_param_name[1:])
                _param_kwargs['default'] = x_init[n - 1]
                _param_kwargs['lb'] = None
            elif _param_name.startswith('y'):
                n = int(_param_name[1:])
                if n == 1:
                    _param_kwargs['default'] = 1.
                else:
                    _param_kwargs['default'] = 0.
                _param_kwargs['lb'] = None
            else:
                n = int(_param_name[1:])
                _param_kwargs['default'] = n
                _param_kwargs['lb'] = 0.
            param_names.append(_param_name)
            param_kwargs.append(_param_kwargs)

        return param_names, param_kwargs
    
    def _get_irf_param_metadata(self, param_name, family, lb=None, ub=None, default=0.):
        irf_ids = self.atomic_irf_names_by_family[family]
        param_init = self.atomic_irf_param_init_by_family[family]
        param_trainable = self.atomic_irf_param_trainable_by_family[family]

        # Process and store initial/prior means
        param_mean = self._get_mean_init_vector(irf_ids, param_name, param_init, default=default)
        param_mean_unconstrained, param_lb, param_ub = self._process_mean(param_mean, lb=lb, ub=ub)

        # Select out irf IDs for which this param is trainable
        trainable_ix, untrainable_ix = self._get_trainable_untrainable_ix(
            param_name,
            irf_ids,
            trainable=param_trainable
        )

        return param_mean, param_mean_unconstrained, param_lb, param_ub, trainable_ix, untrainable_ix

    # PARAMETER INITIALIZATION

    def _initialize_base_params(self):
        with self.session.as_default():
            with self.session.graph.as_default():

                # Intercept

                # Key order: response, ?(ran_gf)
                self.intercept_fixed_base = {}
                self.intercept_random_base = {}
                for _response in self.response_names:
                    # Fixed
                    if self.has_intercept[None]:
                        x = self._initialize_intercept(_response)
                        intercept_fixed = x['value']
                        if 'kl_penalties' in x:
                            self.kl_penalties.update(x['kl_penalties'])
                        if 'eval_resample' in x:
                            self.resample_ops.append(x['eval_resample'])
                    else:
                        intercept_fixed = tf.constant(self.intercept_init[_response], dtype=self.FLOAT_TF)
                    self.intercept_fixed_base[_response] = intercept_fixed

                    # Random
                    for gf in self.rangf:
                        if self.has_intercept[gf]:
                            x = self._initialize_intercept(_response, ran_gf=gf)
                            _intercept_random = x['value']
                            if 'kl_penalties' in x:
                                self.kl_penalties.update(x['kl_penalties'])
                            if 'eval_resample' in x:
                                self.resample_ops.append(x['eval_resample'])
                            if _response not in self.intercept_random_base:
                                self.intercept_random_base[_response] = {}
                            self.intercept_random_base[_response][gf] = _intercept_random

                # Coefficients

                # Key order: response, ?(ran_gf)
                self.coefficient_fixed_base = {}
                self.coefficient_random_base = {}
                for response in self.response_names:
                    # Fixed
                    coef_ids = self.fixed_coef_names
                    if len(coef_ids) > 0:
                        x = self._initialize_coefficient(
                            response,
                            coef_ids=coef_ids
                        )
                        _coefficient_fixed_base = x['value']
                        if 'kl_penalties' in x:
                            self.kl_penalties.update(x['kl_penalties'])
                        if 'eval_resample' in x:
                            self.resample_ops.append(x['eval_resample'])
                    else:
                        if self.use_distributional_regression:
                            nparam = self.get_response_nparam(response)
                        else:
                            nparam = 1
                        ndim = self.get_response_ndim(response)
                        ncoef = 0
                        _coefficient_fixed_base = tf.zeros([ncoef, nparam, ndim])
                    self.coefficient_fixed_base[response] = _coefficient_fixed_base

                    # Random
                    for gf in self.rangf:
                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids):
                            x = self._initialize_coefficient(
                                response,
                                coef_ids=coef_ids,
                                ran_gf=gf
                            )
                            _coefficient_random_base = x['value']
                            if 'kl_penalties' in x:
                                self.kl_penalties.update(x['kl_penalties'])
                            if 'eval_resample' in x:
                                self.resample_ops.append(x['eval_resample'])
                            if response not in self.coefficient_random_base:
                                self.coefficient_random_base[response] = {}
                            self.coefficient_random_base[response][gf] = _coefficient_random_base

                # Parametric IRF parameters

                # Key order: family, param
                self.irf_params_means = {}
                self.irf_params_means_unconstrained = {}
                self.irf_params_lb = {}
                self.irf_params_ub = {}
                self.irf_params_trainable_ix = {}
                self.irf_params_untrainable_ix = {}
                # Key order: response, ?(ran_gf,) family, param
                self.irf_params_fixed_base = {}
                self.irf_params_random_base = {}
                for family in [x for x in self.atomic_irf_names_by_family]:
                    # Collect metadata for IRF params
                    self.irf_params_means[family] = {}
                    self.irf_params_means_unconstrained[family] = {}
                    self.irf_params_lb[family] = {}
                    self.irf_params_ub[family] = {}
                    self.irf_params_trainable_ix[family] = {}
                    self.irf_params_untrainable_ix[family] = {}

                    param_names = []
                    param_kwargs = []
                    if family in self.IRF_KERNELS:
                        for x in self.IRF_KERNELS[family]:
                            param_names.append(x[0])
                            param_kwargs.append(x[1])
                    elif Formula.is_LCG(family):
                        _param_names, _param_kwargs = self._get_nonparametric_irf_params(family)
                        param_names += _param_names
                        param_kwargs += _param_kwargs
                    else:
                        raise ValueError('Unrecognized IRF kernel family "%s".' % family)

                    # Process and store metadata for IRF params
                    for _param_name, _param_kwargs in zip(param_names, param_kwargs):
                        param_mean, param_mean_unconstrained, param_lb, param_ub, trainable_ix, \
                            untrainable_ix = self._get_irf_param_metadata(_param_name, family, **_param_kwargs)

                        self.irf_params_means[family][_param_name] = param_mean
                        self.irf_params_means_unconstrained[family][_param_name] = param_mean_unconstrained
                        self.irf_params_lb[family][_param_name] = param_lb
                        self.irf_params_ub[family][_param_name] = param_ub
                        self.irf_params_trainable_ix[family][_param_name] = trainable_ix
                        self.irf_params_untrainable_ix[family][_param_name] = untrainable_ix

                    # Initialize IRF params
                    for response in self.response_names:
                        for _param_name in param_names:
                            # Fixed
                            x = self._initialize_irf_param(
                                response,
                                family,
                                _param_name
                            )
                            _param = x['value']
                            if 'kl_penalties' in x:
                                self.kl_penalties.update(x['kl_penalties'])
                            if 'eval_resample' in x:
                                self.resample_ops.append(x['eval_resample'])
                            if _param is not None:
                                if response not in self.irf_params_fixed_base:
                                    self.irf_params_fixed_base[response] = {}
                                if family not in self.irf_params_fixed_base[response]:
                                    self.irf_params_fixed_base[response][family] = {}
                                self.irf_params_fixed_base[response][family][_param_name] = _param

                            # Random
                            for gf in self.irf_by_rangf:
                                x = self._initialize_irf_param(
                                    response,
                                    family,
                                    _param_name,
                                    ran_gf=gf
                                )
                                _param = x['value']
                                if 'kl_penalties' in x:
                                    self.kl_penalties.update(x['kl_penalties'])
                                if 'eval_resample' in x:
                                    self.resample_ops.append(x['eval_resample'])
                                if _param is not None:
                                    if response not in self.irf_params_random_base:
                                        self.irf_params_random_base[response] = {}
                                    if gf not in self.irf_params_random_base[response]:
                                        self.irf_params_random_base[response][gf] = {}
                                    if family not in self.irf_params_random_base[response][gf]:
                                        self.irf_params_random_base[response][gf][family] = {}
                                    self.irf_params_random_base[response][gf][family][_param_name] = _param

                # Interactions

                # Key order: response, ?(ran_gf)
                self.interaction_fixed_base = {}
                self.interaction_random_base = {}
                for response in self.response_names:
                    if len(self.interaction_names):
                        interaction_ids = self.fixed_interaction_names
                        if len(interaction_ids):
                            # Fixed
                            x = self._initialize_interaction(
                                response,
                                interaction_ids=interaction_ids
                            )
                            _interaction_fixed_base = x['value']
                            if 'kl_penalties' in x:
                                self.kl_penalties.update(x['kl_penalties'])
                            if 'eval_resample' in x:
                                self.resample_ops.append(x['eval_resample'])
                            self.interaction_fixed_base[response] = _interaction_fixed_base

                        # Random
                        for gf in self.rangf:
                            interaction_ids = self.interaction_by_rangf.get(gf, [])
                            if len(interaction_ids):
                                x = self._initialize_interaction(
                                    response,
                                    interaction_ids=interaction_ids,
                                    ran_gf=gf
                                )
                                _interaction_random_base = x['value']
                                if 'kl_penalties' in x:
                                    self.kl_penalties.update(x['kl_penalties'])
                                if 'eval_resample' in x:
                                    self.resample_ops.append(x['eval_resample'])
                                if response not in self.interaction_random_base:
                                    self.interaction_random_base[response] = {}
                                self.interaction_random_base[response][gf] = _interaction_random_base

                # NN components are initialized elsewhere

    # INTERCEPT INITIALIZATION

    def _initialize_intercept_mle(self, response, ran_gf=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                init = self.intercept_init[response]
                name = sn(response)
                if ran_gf is None:
                    intercept = tf.Variable(
                        init,
                        dtype=self.FLOAT_TF,
                        name='intercept_%s' % name
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.use_distributional_regression:
                        nparam = self.get_response_nparam(response)
                    else:
                        nparam = 1
                    ndim = self.get_response_ndim(response)
                    shape = [rangf_n_levels, nparam, ndim]
                    intercept = tf.Variable(
                        tf.zeros(shape, dtype=self.FLOAT_TF),
                        name='intercept_%s_by_%s' % (name, sn(ran_gf))
                    )

                return {'value': intercept}

    def _initialize_intercept_bayes(self, response, ran_gf=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                init = self.intercept_init[response]

                name = sn(response)
                if ran_gf is None:
                    sd_prior = self._intercept_prior_sd[response]
                    sd_posterior = self._intercept_posterior_sd_init[response]

                    rv_dict = get_random_variable(
                        'intercept_%s' % name,
                        init.shape,
                        sd_posterior,
                        init=init,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )

                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_fixef:
                        out['kl_penalties'] = rv_dict['kl_penalties']

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    sd_prior = self._intercept_ranef_prior_sd[response]
                    sd_posterior = self._intercept_ranef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        sd_prior = sd_prior[:1]
                        sd_posterior = sd_posterior[:1]
                    sd_prior = np.ones((rangf_n_levels, 1, 1)) * sd_prior[None, ...]
                    sd_posterior = np.ones((rangf_n_levels, 1, 1)) * sd_posterior[None, ...]

                    rv_dict = get_random_variable(
                        'intercept_%s_by_%s' % (name, sn(ran_gf)),
                        sd_posterior.shape,
                        sd_posterior,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )

                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_ranef:
                        out['kl_penalties'] = rv_dict['kl_penalties']

                return out

    def _initialize_intercept(self, *args, **kwargs):
        if 'intercept' in self.rvs:
            return self._initialize_intercept_bayes(*args, **kwargs)
        return self._initialize_intercept_mle(*args, **kwargs)

    def _compile_intercepts(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.intercept = {}
                self.intercept_fixed = {}
                self.intercept_random = {}
                for response in self.response_names:
                    # Fixed
                    response_params = self.get_response_params(response)
                    nparam = len(response_params)
                    ndim = self.get_response_ndim(response)

                    intercept_fixed = self.intercept_fixed_base[response]
                    self.intercept_fixed[response] = intercept_fixed

                    self._regularize(
                        intercept_fixed,
                        center=self.intercept_init[response],
                        regtype='intercept',
                        var_name='intercept_%s' % sn(response)
                    )

                    intercept = intercept_fixed[None, ...]

                    if self.log_fixed:
                        for i, response_param in enumerate(response_params):
                            dim_names = self.expand_param_name(response, response_param)
                            for j, dim_name in enumerate(dim_names):
                                val = intercept_fixed[i, j]
                                if self.has_intercept[None]:
                                    tf.summary.scalar(
                                        'intercept/%s_%s' % (sn(response), sn(dim_name)),
                                        val,
                                        collections=['params']
                                    )

                    # Random
                    for i, gf in enumerate(self.rangf):
                        # Random intercepts
                        if self.has_intercept[gf]:
                            intercept_random = self.intercept_random_base[response][gf]
                            intercept_random_means = tf.reduce_mean(intercept_random, axis=0, keepdims=True)
                            intercept_random -= intercept_random_means
                            if response not in self.intercept_random:
                                self.intercept_random[response] = {}
                            self.intercept_random[response][gf] = intercept_random

                            if 'intercept' not in self.rvs or not self.declare_priors_ranef:
                                self._regularize(
                                    intercept_random,
                                    regtype='ranef',
                                    var_name='intercept_%s_by_%s' % (sn(response), sn(gf))
                                )

                            if self.log_random:
                                for j, response_param in enumerate(response_params):
                                    if j == 0 or self.use_distributional_regression:
                                        dim_names = self.expand_param_name(response, response_param)
                                        for k, dim_name in enumerate(dim_names):
                                            val = intercept_random[:, j, k]
                                            tf.summary.histogram(
                                                'by_%s/intercept/%s_%s' % (sn(gf), sn(response), sn(dim_name)),
                                                val,
                                                collections=['random']
                                            )

                            if not self.use_distributional_regression:
                                # Pad out any unmodeled params of response distribution
                                intercept_random = tf.pad(
                                    intercept_random,
                                    # ranef   pred param    pred dim
                                    [(0, 0), (0, nparam - 1), (0, 0)]
                                )

                            # Add final 0 vector for population-level effect
                            intercept_random = tf.concat(
                                [
                                    intercept_random,
                                    tf.zeros([1, nparam, ndim])
                                ],
                                axis=0
                            )

                            intercept = intercept + tf.gather(intercept_random, self.Y_gf[:, i])

                    self.intercept[response] = intercept

    # COEFFICIENT INITIALIZATION

    def _initialize_coefficient_mle(self, response, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names

        if self.use_distributional_regression:
            nparam = self.get_response_nparam(response)
        else:
            nparam = 1
        ndim = self.get_response_ndim(response)
        ncoef = len(coef_ids)

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    coefficient = tf.Variable(
                        tf.zeros([ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s' % sn(response)
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    coefficient = tf.Variable(
                        tf.zeros([rangf_n_levels, ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s_by_%s' % (sn(response), sn(ran_gf))
                    )

                # shape: (?rangf_n_levels, ncoef, nparam, ndim)

                return {'value': coefficient}

    def _initialize_coefficient_bayes(self, response, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names

        ncoef = len(coef_ids)

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    sd_prior = self._coef_prior_sd[response]
                    sd_posterior = self._coef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        sd_prior = sd_prior[:1]
                        sd_posterior = sd_posterior[:1]
                    sd_prior = np.ones((ncoef, 1, 1)) * sd_prior[None, ...]
                    sd_posterior = np.ones((ncoef, 1, 1)) * sd_posterior[None, ...]

                    rv_dict = get_random_variable(
                        'coefficient_%s' % sn(response),
                        sd_posterior.shape,
                        sd_posterior,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )

                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_fixef:
                        out['kl_penalties'] = rv_dict['kl_penalties']

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    sd_prior = self._coef_ranef_prior_sd[response]
                    sd_posterior = self._coef_ranef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        sd_prior = sd_prior[:1]
                        sd_posterior = sd_posterior[:1]
                    sd_prior = np.ones((rangf_n_levels, ncoef, 1, 1)) * sd_prior[None, None, ...]
                    sd_posterior = np.ones((rangf_n_levels, ncoef, 1, 1)) * sd_posterior[None, None, ...]

                    rv_dict = get_random_variable(
                        'coefficient_%s_by_%s' % (sn(response), sn(ran_gf)),
                        sd_posterior.shape,
                        sd_posterior,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )

                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_ranef:
                        out['kl_penalties'] = rv_dict['kl_penalties']

                # shape: (?rangf_n_levels, ncoef, nparam, ndim)

                return out

    def _initialize_coefficient(self, *args, **kwargs):
        if 'coefficient' in self.rvs:
            return self._initialize_coefficient_bayes(*args, **kwargs)
        return self._initialize_coefficient_mle(*args, **kwargs)

    def _compile_coefficients(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.coefficient = {}
                self.coefficient_fixed = {}
                self.coefficient_random = {}
                for response in self.response_names:
                    response_params = self.get_response_params(response)
                    if not self.use_distributional_regression:
                        response_params = response_params[:1]
                    nparam = len(response_params)
                    ndim = self.get_response_ndim(response)
                    fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)
                    coef_ids = self.coef_names

                    coefficient_fixed = self._scatter_along_axis(
                        fixef_ix,
                        self.coefficient_fixed_base[response],
                        [len(coef_ids), nparam, ndim]
                    )
                    self.coefficient_fixed[response] = coefficient_fixed

                    self._regularize(
                        self.coefficient_fixed_base[response],
                        regtype='coefficient',
                        var_name='coefficient_%s' % response
                    )

                    coefficient = coefficient_fixed[None, ...]

                    if self.log_fixed:
                        for i, coef_name in enumerate(self.coef_names):
                            for j, response_param in enumerate(response_params):
                                dim_names = self.expand_param_name(response, response_param)
                                for k, dim_name in enumerate(dim_names):
                                    val = coefficient_fixed[i, j, k]
                                    tf.summary.scalar(
                                        'coefficient' + '/%s/%s_%s' % (
                                            sn(coef_name),
                                            sn(response),
                                            sn(dim_name)
                                        ),
                                        val,
                                        collections=['params']
                                    )
                    for i, gf in enumerate(self.rangf):
                        levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                        coefs = self.coef_by_rangf.get(gf, [])
                        if len(coefs) > 0:
                            nonzero_coef_ix = names2ix(coefs, self.coef_names)

                            coefficient_random = self.coefficient_random_base[response][gf]
                            coefficient_random_means = tf.reduce_mean(coefficient_random, axis=0, keepdims=True)
                            coefficient_random -= coefficient_random_means
                            if response not in self.coefficient_random:
                                self.coefficient_random[response] = {}
                            self.coefficient_random[response][gf] = coefficient_random

                            if 'coefficient' not in self.rvs or not self.declare_priors_ranef:
                                self._regularize(
                                    coefficient_random,
                                    regtype='ranef',
                                    var_name='coefficient_%s_by_%s' % (sn(response),sn(gf))
                                )

                            if self.log_random:
                                for j, coef_name in enumerate(coefs):
                                    for k, response_param in enumerate(response_params):
                                        dim_names = self.expand_param_name(response, response_param)
                                        for l, dim_name in enumerate(dim_names):
                                            val = coefficient_random[:, j, k, l]
                                            tf.summary.histogram(
                                                'by_%s/coefficient/%s/%s_%s' % (
                                                    sn(gf),
                                                    sn(coef_name),
                                                    sn(response),
                                                    sn(dim_name)
                                                ),
                                                val,
                                                collections=['random']
                                            )

                            coefficient_random = self._scatter_along_axis(
                                nonzero_coef_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    coefficient_random,
                                    [self.rangf_n_levels[i], len(coefs), nparam, ndim]
                                ),
                                [self.rangf_n_levels[i], len(self.coef_names), nparam, ndim],
                                axis=1
                            )

                            coefficient = coefficient + tf.gather(coefficient_random, self.Y_gf[:, i], axis=0)

                    self.coefficient[response] = coefficient

    # IRF PARAMETER INITIALIZATION

    def _initialize_irf_param_mle(self, response, family, param_name, ran_gf=None):
        param_mean_unconstrained = self.irf_params_means_unconstrained[family][param_name]
        trainable_ix = self.irf_params_trainable_ix[family][param_name]
        mean = param_mean_unconstrained[trainable_ix]
        irf_ids_all = self.atomic_irf_names_by_family[family]
        param_trainable = self.atomic_irf_param_trainable_by_family[family]

        if self.use_distributional_regression:
            response_nparam = self.get_response_nparam(response) # number of params of response dist, not IRF
        else:
            response_nparam = 1
        response_ndim = self.get_response_ndim(response)

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    trainable_ids = [x for x in irf_ids_all if param_name in param_trainable[x]]
                    nirf = len(trainable_ids)

                    if nirf:
                        param = tf.Variable(
                            tf.ones([nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF) * tf.constant(mean[..., None, None], dtype=self.FLOAT_TF),
                            name=sn('%s_%s_%s' % (param_name, '-'.join(trainable_ids), sn(response)))
                        )
                    else:
                        param = None
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_ids_gf = self.irf_by_rangf[ran_gf]
                    trainable_ids = [x for x in irf_ids_all if (param_name in param_trainable[x] and x in irf_ids_gf)]
                    nirf = len(trainable_ids)

                    if nirf:
                        param = tf.Variable(
                            tf.zeros([rangf_n_levels, nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF),
                            name=sn('%s_%s_%s_by_%s' % (param_name, '-'.join(trainable_ids), sn(response), sn(ran_gf)))
                        )
                    else:
                        param = None

                # shape: (?rangf_n_levels, nirf, nparam, ndim)

                return {'value': param}

    def _initialize_irf_param_bayes(self, response, family, param_name, ran_gf=None):
        param_mean_unconstrained = self.irf_params_means_unconstrained[family][param_name]
        trainable_ix = self.irf_params_trainable_ix[family][param_name]
        mean = param_mean_unconstrained[trainable_ix]
        irf_ids_all = self.atomic_irf_names_by_family[family]
        param_trainable = self.atomic_irf_param_trainable_by_family[family]

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    trainable_ids = [x for x in irf_ids_all if param_name in param_trainable[x]]
                    nirf = len(trainable_ids)

                    if nirf:
                        sd_prior = self._irf_param_prior_sd[response]
                        sd_posterior = self._irf_param_posterior_sd_init[response]
                        if not self.use_distributional_regression:
                            sd_prior = sd_prior[:1]
                            sd_posterior = sd_posterior[:1]
                        sd_prior = np.ones((nirf, 1, 1)) * sd_prior[None, ...]
                        sd_posterior = np.ones((nirf, 1, 1)) * sd_posterior[None, ...]
                        while len(mean.shape) < len(sd_posterior.shape):
                            mean = mean[..., None]
                        mean = np.ones_like(sd_posterior) * mean

                        rv_dict = get_random_variable(
                            '%s_%s_%s' % (param_name, sn(response), sn('-'.join(trainable_ids))),
                            sd_posterior.shape,
                            sd_posterior,
                            init=mean,
                            constraint=self.constraint,
                            sd_prior=sd_prior,
                            training=self.training,
                            use_MAP_mode=self.use_MAP_mode,
                            epsilon=self.epsilon,
                            session=self.session
                        )
                        out = {
                            'value': rv_dict['v'],
                            'eval_resample': rv_dict['v_eval_resample']
                        }
                        if self.declare_priors_fixef:
                            out['kl_penalties'] = rv_dict['kl_penalties']
                    else:
                        out = {
                            'v': None,
                            'eval_resample': None
                        }
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_ids_gf = self.irf_by_rangf[ran_gf]
                    trainable_ids = [x for x in irf_ids_all if (param_name in param_trainable[x] and x in irf_ids_gf)]
                    nirf = len(trainable_ids)

                    if nirf:
                        sd_prior = self._irf_param_ranef_prior_sd[response]
                        sd_posterior = self._irf_param_ranef_posterior_sd_init[response]
                        if not self.use_distributional_regression:
                            sd_prior = sd_prior[:1]
                            sd_posterior = sd_posterior[:1]
                        sd_prior = np.ones((rangf_n_levels, nirf, 1, 1)) * sd_prior[None, None, ...]
                        sd_posterior = np.ones((rangf_n_levels, nirf, 1, 1)) * sd_posterior[None, None, ...]

                        rv_dict = get_random_variable(
                            '%s_%s_%s_by_%s' % (param_name, sn(response), sn('-'.join(trainable_ids)), sn(ran_gf)),
                            sd_posterior.shape,
                            sd_posterior,
                            constraint=self.constraint,
                            sd_prior=sd_prior,
                            training=self.training,
                            use_MAP_mode=self.use_MAP_mode,
                            epsilon=self.epsilon,
                            session=self.session
                        )
                        out = {
                            'value': rv_dict['v'],
                            'eval_resample': rv_dict['v_eval_resample']
                        }
                        if self.declare_priors_ranef:
                            out['kl_penalties'] = rv_dict['kl_penalties']
                    else:
                        out = {
                            'v': None,
                            'v_eval_resample': None
                        }

                # shape: (?rangf_n_levels, nirf, nparam, ndim)

                return out

    def _initialize_irf_param(self, *args, **kwargs):
        if 'irf_param' in self.rvs:
            return self._initialize_irf_param_bayes(*args, **kwargs)
        return self._initialize_irf_param_mle(*args, **kwargs)

    def _compile_irf_params(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                # Base IRF params are saved as tensors with shape (nid, npredparam, npreddim),
                # one for each irf_param of each IRF kernel family.
                # Here fixed and random IRF params are summed, constraints are applied, and new tensors are stored
                # with shape (batch, 1, npredparam, npreddim), one for each parameter of each IRF ID for each response variable.
                # The 1 in the 2nd dim supports broadcasting over the time dimension.

                # Key order: response, ?(ran_gf), irf_id, irf_param
                self.irf_param = {}
                self.irf_param_fixed = {}
                self.irf_param_random = {}
                for response in self.response_names:
                    for family in self.atomic_irf_names_by_family:
                        if family in ('DiracDelta', 'NN'):
                            continue

                        irf_ids = self.atomic_irf_names_by_family[family]
                        trainable = self.atomic_irf_param_trainable_by_family[family]

                        for irf_param_name in Formula.irf_params(family):
                            response_params = self.get_response_params(response)
                            if not self.use_distributional_regression:
                                response_params = response_params[:1]
                            nparam_response = len(response_params)  # number of params of response dist, not IRF
                            ndim = self.get_response_ndim(response)
                            irf_param_lb = self.irf_params_lb[family][irf_param_name]
                            if irf_param_lb is not None:
                                irf_param_lb = tf.constant(irf_param_lb, dtype=self.FLOAT_TF)
                            irf_param_ub = self.irf_params_ub[family][irf_param_name]
                            if irf_param_ub is not None:
                                irf_param_ub = tf.constant(irf_param_ub, dtype=self.FLOAT_TF)
                            trainable_ix = self.irf_params_trainable_ix[family][irf_param_name]
                            untrainable_ix = self.irf_params_untrainable_ix[family][irf_param_name]

                            irf_param_means = self.irf_params_means_unconstrained[family][irf_param_name]
                            irf_param_trainable_means = tf.constant(
                                irf_param_means[trainable_ix][..., None, None],
                                dtype=self.FLOAT_TF
                            )

                            self._regularize(
                                self.irf_params_fixed_base[response][family][irf_param_name],
                                irf_param_trainable_means,
                                regtype='irf', var_name='%s_%s' % (irf_param_name, sn(response))
                            )

                            irf_param_untrainable_means = tf.constant(
                                irf_param_means[untrainable_ix][..., None, None],
                                dtype=self.FLOAT_TF
                            )
                            irf_param_untrainable_means = tf.broadcast_to(
                                irf_param_untrainable_means,
                                [len(untrainable_ix), nparam_response, ndim]
                            )

                            irf_param_trainable = self._scatter_along_axis(
                                trainable_ix,
                                self.irf_params_fixed_base[response][family][irf_param_name],
                                [len(irf_ids), nparam_response, ndim]
                            )
                            irf_param_untrainable = self._scatter_along_axis(
                                untrainable_ix,
                                irf_param_untrainable_means,
                                [len(irf_ids), nparam_response, ndim]
                            )
                            is_trainable = np.zeros(len(irf_ids), dtype=bool)
                            is_trainable[trainable_ix] = True

                            irf_param_fixed = tf.where(
                                is_trainable,
                                irf_param_trainable,
                                irf_param_untrainable
                            )

                            # Add batch dimension
                            irf_param = irf_param_fixed[None, ...]

                            for i, irf_id in enumerate(irf_ids):
                                _irf_param_fixed = irf_param_fixed[i]
                                if response not in self.irf_param_fixed:
                                    self.irf_param_fixed[response] = {}
                                if irf_id not in self.irf_param_fixed[response]:
                                    self.irf_param_fixed[response][irf_id] = {}
                                self.irf_param_fixed[response][irf_id][irf_param_name] = _irf_param_fixed

                                if self.log_fixed:
                                    for j, response_param in enumerate(response_params):
                                        dim_names = self.expand_param_name(response, response_param)
                                        for k, dim_name in enumerate(dim_names):
                                            val = _irf_param_fixed[j, k]
                                            tf.summary.scalar(
                                                '%s/%s/%s_%s' % (
                                                    irf_param_name,
                                                    sn(irf_id),
                                                    sn(response),
                                                    sn(dim_name)
                                                ),
                                                val,
                                                collections=['params']
                                            )

                            for i, gf in enumerate(self.rangf):
                                if gf in self.irf_by_rangf:
                                    irf_ids_all = [x for x in self.irf_by_rangf[gf] if self.node_table[x].family == family]
                                    irf_ids_ran = [x for x in irf_ids_all if irf_param_name in trainable[x]]
                                    if len(irf_ids_ran):
                                        irfs_ix = names2ix(irf_ids_all, irf_ids)
                                        levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                                        irf_param_random = self.irf_params_random_base[response][gf][family][irf_param_name]
                                        irf_param_random_mean = tf.reduce_mean(irf_param_random, axis=0, keepdims=True)
                                        irf_param_random -= irf_param_random_mean

                                        if 'irf_param' not in self.rvs or not self.declare_priors_ranef:
                                            self._regularize(
                                                irf_param_random,
                                                regtype='ranef',
                                                var_name='%s_%s_by_%s' % (irf_param_name, sn(response), sn(gf))
                                            )

                                        for j, irf_id in enumerate(irf_ids_ran):
                                            _irf_param_random = irf_param_random[:, j]
                                            if response not in self.irf_param_random:
                                                self.irf_param_random[response] = {}
                                            if gf not in self.irf_param_random[response]:
                                                self.irf_param_random[response][gf] = {}
                                            if irf_id not in self.irf_param_random[response][gf]:
                                                self.irf_param_random[response][gf][irf_id] = {}
                                            self.irf_param_random[response][gf][irf_id][irf_param_name] = _irf_param_random

                                            if self.log_random:
                                                if irf_id in irf_ids_ran:
                                                    for k, response_param in enumerate(response_params):
                                                        dim_names = self.expand_param_name(response, response_param)
                                                        for l, dim_name in enumerate(dim_names):
                                                            val = _irf_param_random[:, k, l]
                                                            tf.summary.histogram(
                                                                'by_%s/%s/%s/%s_%s' % (
                                                                    sn(gf),
                                                                    sn(irf_id),
                                                                    irf_param_name,
                                                                    sn(dim_name),
                                                                    sn(response)
                                                                ),
                                                                val,
                                                                collections=['random']
                                                            )

                                        irf_param_random = self._scatter_along_axis(
                                            irfs_ix,
                                            self._scatter_along_axis(
                                                levels_ix,
                                                irf_param_random,
                                                [self.rangf_n_levels[i], len(irfs_ix), nparam_response, ndim]
                                            ),
                                            [self.rangf_n_levels[i], len(irf_ids), nparam_response, ndim],
                                            axis=1
                                        )

                                        irf_param = irf_param + tf.gather(irf_param_random, self.Y_gf[:, i], axis=0)

                            if irf_param_lb is not None and irf_param_ub is None:
                                irf_param = irf_param_lb + self.constraint_fn(irf_param) + self.epsilon
                            elif irf_param_lb is None and irf_param_ub is not None:
                                irf_param = irf_param_ub - self.constraint_fn(irf_param) - self.epsilon
                            elif irf_param_lb is not None and irf_param_ub is not None:
                                irf_param = self._sigmoid(irf_param, a=irf_param_lb, b=irf_param_ub) * (1 - 2 * self.epsilon) + self.epsilon

                            for j, irf_id in enumerate(irf_ids):
                                if irf_param_name in trainable[irf_id]:
                                    if response not in self.irf_param:
                                        self.irf_param[response] = {}
                                    if irf_id not in self.irf_param[response]:
                                        self.irf_param[response][irf_id] = {}
                                    # id is -3 dimension
                                    self.irf_param[response][irf_id][irf_param_name] = irf_param[..., j, :, :]

    def _initialize_irf_lambdas(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.irf_lambdas = {}
                if self.future_length: # Non-causal
                    support_lb = None
                else: # Causal
                    support_lb = 0.
                support_ub = None

                def exponential(**params):
                    return exponential_irf_factory(
                        **params,
                        session=self.session
                    )

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['ExpRateGT1'] = exponential

                def gamma(**params):
                    return gamma_irf_factory(
                        **params,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['HRFSingleGamma'] = gamma

                def shifted_gamma_lambdas(**params):
                    return shifted_gamma_irf_factory(
                        **params,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma_lambdas
                self.irf_lambdas['ShiftedGammaShapeGT1'] = shifted_gamma_lambdas

                def normal(**params):
                    return normal_irf_factory(
                        **params,
                        support_lb=support_lb,
                        support_ub=support_ub,
                        session=self.session
                    )

                self.irf_lambdas['Normal'] = normal

                def skew_normal(**params):
                    return skew_normal_irf_factory(
                        **params,
                        support_lb=support_lb,
                        support_ub=self.t_delta_limit.astype(dtype=self.FLOAT_NP) if support_ub is None else support_ub,
                        session=self.session
                    )

                self.irf_lambdas['SkewNormal'] = skew_normal

                def emg(**kwargs):
                    return emg_irf_factory(
                        **kwargs,
                        support_lb=support_lb,
                        support_ub=support_ub,
                        session=self.session
                    )

                self.irf_lambdas['EMG'] = emg

                def beta_prime(**kwargs):
                    return beta_prime_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session
                    )

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(**kwargs):
                    return shifted_beta_prime_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session
                    )

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

                def double_gamma_1(**kwargs):
                    return double_gamma_1_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma1'] = double_gamma_1

                def double_gamma_2(**kwargs):
                    return double_gamma_2_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma2'] = double_gamma_2

                def double_gamma_3(**kwargs):
                    return double_gamma_3_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma3'] = double_gamma_3

                def double_gamma_4(**kwargs):
                    return double_gamma_4_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma4'] = double_gamma_4

                def double_gamma_5(**kwargs):
                    return double_gamma_5_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.session,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma'] = double_gamma_5
                self.irf_lambdas['HRFDoubleGamma5'] = double_gamma_5

    def _initialize_LCG_irf(
            self,
            bases
    ):
        if self.future_length: # Non-causal
            support_lb = None
        else: # Causal
            support_lb = 0.
        support_ub = None

        def f(
                bases=bases,
                int_type=self.INT_TF,
                float_type=self.FLOAT_TF,
                session=self.session,
                support_lb=support_lb,
                support_ub=support_ub,
                **params
        ):
            return LCG_irf_factory(
                bases,
                int_type=int_type,
                float_type=float_type,
                support_lb=support_lb,
                support_ub=support_ub,
                session=session,
                **params
            )

        return f

    def _get_irf_lambdas(self, family):
        if family in self.irf_lambdas:
            return self.irf_lambdas[family]
        elif Formula.is_LCG(family):
            bases = Formula.bases(family)
            return self._initialize_LCG_irf(
                bases
            )
        else:
            raise ValueError('No IRF lamdba found for family "%s"' % family)

    def _initialize_irfs(self, t, response):
        with self.session.as_default():
            with self.session.graph.as_default():
                if response not in self.irf:
                    self.irf[response] = {}
                if t.family is None:
                    self.irf[response][t.name()] = []
                elif t.family in ('Terminal', 'DiracDelta'):
                    if t.p.family != 'NN': # NN IRFs are computed elsewhere, skip here
                        assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                        if t.family == 'DiracDelta':
                            assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in CDR formula strings'
                            assert not t.impulse == 'rate', '"rate" is a reserved keyword in CDR formula strings and cannot be used under DiracDelta'
                        self.irf[response][t.name()] = self.irf[response][t.p.name()][:]
                elif t.family != 'NN': # NN IRFs are computed elsewhere, skip here
                    params = self.irf_param[response][t.irf_id()]
                    atomic_irf = self._get_irf_lambdas(t.family)(**params)
                    if t.p.name() in self.irf:
                        irf = self.irf[response][t.p.name()][:] + [atomic_irf]
                    else:
                        irf = [atomic_irf]
                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[response][t.name()] = irf

                for c in t.children:
                    self._initialize_irfs(c, response)

    # NN INITIALIZATION

    def _initialize_bias_mle(
            self,
            nn_id,
            rangf_map=None,
            use_ranef=None,
            name=None
    ):
        if use_ranef is None:
            use_ranef = True
        if not use_ranef:
            rangf_map = None

        with self.session.as_default():
            with self.session.graph.as_default():
                bias = ScaleLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    rangf_map=rangf_map,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return bias

    def _initialize_bias_bayes(
            self,
            nn_id,
            rangf_map=None,
            use_ranef=None,
            name=None
    ):
        if use_ranef is None:
            use_ranef = True
        if not use_ranef:
            rangf_map = None

        declare_priors = self.get_nn_meta('declare_priors_biases', nn_id)
        sd_prior = self.get_nn_meta('bias_prior_sd', nn_id)
        sd_init = self.get_nn_meta('bias_sd_init', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                bias = ScaleLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    rangf_map=rangf_map,
                    declare_priors=declare_priors,
                    sd_prior=sd_prior,
                    sd_init=sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    ranef_to_fixef_prior_sd_ratio=self.ranef_to_fixef_prior_sd_ratio,
                    constraint=self.constraint,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return bias

    def _initialize_bias(self, *args, **kwargs):
        if 'nn' in self.rvs:
            return self._initialize_bias_bayes(*args, **kwargs)
        return self._initialize_bias_mle(*args, **kwargs)


    def _initialize_scale_mle(
            self,
            nn_id,
            rangf_map=None,
            use_ranef=None,
            name=None
    ):
        if use_ranef is None:
            use_ranef = True
        if not use_ranef:
            rangf_map = None

        with self.session.as_default():
            with self.session.graph.as_default():
                bias = ScaleLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    rangf_map=rangf_map,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return bias

    def _initialize_scale_bayes(
            self,
            nn_id,
            rangf_map=None,
            use_ranef=None,
            name=None
    ):
        if use_ranef is None:
            use_ranef = True
        if not use_ranef:
            rangf_map = None

        declare_priors = self.get_nn_meta('declare_priors_weights', nn_id)
        sd_prior = self.get_nn_meta('weight_prior_sd', nn_id)
        sd_init = self.get_nn_meta('weight_sd_init', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                bias = ScaleLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    rangf_map=rangf_map,
                    declare_priors=declare_priors,
                    sd_prior=sd_prior,
                    sd_init=sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    ranef_to_fixef_prior_sd_ratio=self.ranef_to_fixef_prior_sd_ratio,
                    constraint=self.constraint,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return bias

    def _initialize_scale(self, *args, **kwargs):
        if 'nn' in self.rvs:
            return self._initialize_bias_bayes(*args, **kwargs)
        return self._initialize_bias_mle(*args, **kwargs)

    def _initialize_feedforward_mle(
            self,
            nn_id,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            rangf_map=None,
            weights_use_ranef=None,
            biases_use_ranef=None,
            normalizer_use_ranef=None,
            final=False,
            name=None
    ):
        if weights_use_ranef is None:
            weights_use_ranef = not self.get_nn_meta('ranef_bias_only', nn_id)
        if biases_use_ranef is None:
            biases_use_ranef = True
        if normalizer_use_ranef is None:
            normalizer_use_ranef = self.get_nn_meta('normalizer_use_ranef', nn_id)
        normalize_after_activation = self.get_nn_meta('normalize_after_activation', nn_id)
        shift_normalized_activations = self.get_nn_meta('shift_normalized_activations', nn_id) and use_bias
        rescale_normalized_activations = self.get_nn_meta('rescale_normalized_activations', nn_id)
        weight_sd_init = self.get_nn_meta('weight_sd_init', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                projection = DenseLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    use_bias=use_bias,
                    activation=activation,
                    dropout=dropout,
                    maxnorm=maxnorm,
                    batch_normalization_decay=batch_normalization_decay,
                    layer_normalization_type=layer_normalization_type,
                    normalize_after_activation=normalize_after_activation,
                    shift_normalized_activations=shift_normalized_activations,
                    rescale_normalized_activations=rescale_normalized_activations,
                    rangf_map=rangf_map,
                    weights_use_ranef=weights_use_ranef,
                    biases_use_ranef=biases_use_ranef,
                    normalizer_use_ranef=normalizer_use_ranef,
                    kernel_sd_init=weight_sd_init,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return projection

    def _initialize_feedforward_bayes(
            self,
            nn_id,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            rangf_map=None,
            weights_use_ranef=None,
            biases_use_ranef=None,
            normalizer_use_ranef=None,
            final=False,
            name=None
    ):
        if weights_use_ranef is None:
            weights_use_ranef = not self.get_nn_meta('ranef_bias_only', nn_id)
        if biases_use_ranef is None:
            biases_use_ranef = True
        if normalizer_use_ranef is None:
            normalizer_use_ranef = self.get_nn_meta('normalizer_use_ranef', nn_id)
        normalize_after_activation = self.get_nn_meta('normalize_after_activation', nn_id)
        shift_normalized_activations = self.get_nn_meta('shift_normalized_activations', nn_id) and use_bias
        rescale_normalized_activations = self.get_nn_meta('rescale_normalized_activations', nn_id)
        weight_sd_init = self.get_nn_meta('weight_sd_init', nn_id)
        bias_sd_init = self.get_nn_meta('bias_sd_init', nn_id)
        gamma_sd_init = self.get_nn_meta('gamma_sd_init', nn_id)
        declare_priors_biases = self.get_nn_meta('declare_priors_biases', nn_id)
        declare_priors_gamma = self.get_nn_meta('declare_priors_gamma', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                if final:
                    weight_sd_prior = 1.
                    bias_sd_prior = 1.
                    gamma_sd_prior = 1.
                    declare_priors_weights = self.declare_priors_fixef
                else:
                    weight_sd_prior = self.get_nn_meta('weight_prior_sd', nn_id)
                    bias_sd_prior = self.get_nn_meta('bias_prior_sd', nn_id)
                    gamma_sd_prior = self.get_nn_meta('gamma_prior_sd', nn_id)
                    declare_priors_weights = self.get_nn_meta('declare_priors_weights', nn_id)
                    declare_priors_gamma = self.get_nn_meta('declare_priors_gamma', nn_id)

                projection = DenseLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    use_bias=use_bias,
                    activation=activation,
                    dropout=dropout,
                    maxnorm=maxnorm,
                    batch_normalization_decay=batch_normalization_decay,
                    layer_normalization_type=layer_normalization_type,
                    normalize_after_activation=normalize_after_activation,
                    shift_normalized_activations=shift_normalized_activations,
                    rescale_normalized_activations=rescale_normalized_activations,
                    rangf_map=rangf_map,
                    weights_use_ranef=weights_use_ranef,
                    biases_use_ranef=biases_use_ranef,
                    normalizer_use_ranef=normalizer_use_ranef,
                    declare_priors_weights=declare_priors_weights,
                    declare_priors_biases=declare_priors_biases,
                    declare_priors_gamma=declare_priors_gamma,
                    kernel_sd_prior=weight_sd_prior,
                    kernel_sd_init=weight_sd_init,
                    bias_sd_prior=bias_sd_prior,
                    bias_sd_init=bias_sd_init,
                    gamma_sd_prior=gamma_sd_prior,
                    gamma_sd_init=gamma_sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    ranef_to_fixef_prior_sd_ratio=self.ranef_to_fixef_prior_sd_ratio,
                    constraint=self.constraint,
                    epsilon=self.epsilon,
                    session=self.session,
                    name=name
                )

                return projection

    def _initialize_feedforward(self, *args, **kwargs):
        if 'nn' in self.rvs:
            return self._initialize_feedforward_bayes(*args, **kwargs)
        return self._initialize_feedforward_mle(*args, **kwargs)

    def _initialize_rnn_mle(
            self,
            nn_id,
            l,
            rangf_map=None,
            weights_use_ranef=None,
            biases_use_ranef=None,
            normalizer_use_ranef=None,
    ):
        if weights_use_ranef is None:
            weights_use_ranef = not self.get_nn_meta('ranef_bias_only', nn_id)
        if biases_use_ranef is None:
            if self.get_nn_meta('ranef_l1_only', nn_id):
                if l == 0:
                    biases_use_ranef = True
                else:
                    biases_use_ranef = True
            else:
                biases_use_ranef = True
        if normalizer_use_ranef is None:
            normalizer_use_ranef = self.get_nn_meta('normalizer_use_ranef', nn_id)
        n_units_rnn = self.get_nn_meta('n_units_rnn', nn_id)
        rnn_activation = self.get_nn_meta('rnn_activation', nn_id)
        recurrent_activation = self.get_nn_meta('recurrent_activation', nn_id)
        weight_sd_init = self.get_nn_meta('weight_sd_init', nn_id)
        ff_dropout_rate = self.get_nn_meta('ff_dropout_rate', nn_id)
        rnn_h_dropout_rate = self.get_nn_meta('rnn_h_dropout_rate', nn_id)
        rnn_c_dropout_rate = self.get_nn_meta('rnn_c_dropout_rate', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                units = n_units_rnn[l]
                rnn = RNNLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    activation=rnn_activation,
                    recurrent_activation=recurrent_activation,
                    kernel_sd_init=weight_sd_init,
                    rangf_map=rangf_map,
                    weights_use_ranef=weights_use_ranef,
                    biases_use_ranef=biases_use_ranef,
                    normalizer_use_ranef=normalizer_use_ranef,
                    bottomup_dropout=ff_dropout_rate,
                    h_dropout=rnn_h_dropout_rate,
                    c_dropout=rnn_c_dropout_rate,
                    return_sequences=True,
                    name='%s_rnn_l%d' % (nn_id, l + 1),
                    epsilon=self.epsilon,
                    session=self.session
                )

                return rnn

    def _initialize_rnn_bayes(
            self,
            nn_id,
            l,
            rangf_map=None,
            weights_use_ranef=None,
            biases_use_ranef=None,
            normalizer_use_ranef=None,
    ):

        if weights_use_ranef is None:
            weights_use_ranef = not self.get_nn_meta('ranef_bias_only', nn_id)
        if biases_use_ranef is None:
            if self.get_nn_meta('ranef_l1_only', nn_id):
                if l == 0:
                    biases_use_ranef = True
                else:
                    biases_use_ranef = True
            else:
                biases_use_ranef = True
        if normalizer_use_ranef is None:
            normalizer_use_ranef = self.get_nn_meta('normalizer_use_ranef', nn_id)
        n_units_rnn = self.get_nn_meta('n_units_rnn', nn_id)
        rnn_activation = self.get_nn_meta('rnn_activation', nn_id)
        recurrent_activation = self.get_nn_meta('recurrent_activation', nn_id)
        ff_dropout_rate = self.get_nn_meta('ff_dropout_rate', nn_id)
        rnn_h_dropout_rate = self.get_nn_meta('rnn_h_dropout_rate', nn_id)
        rnn_c_dropout_rate = self.get_nn_meta('rnn_c_dropout_rate', nn_id)
        declare_priors_weights = self.get_nn_meta('declare_priors_weights', nn_id)
        declare_priors_biases = self.get_nn_meta('declare_priors_biases', nn_id)
        weight_prior_sd = self.get_nn_meta('weight_prior_sd', nn_id)
        weight_sd_init = self.get_nn_meta('weight_sd_init', nn_id)
        bias_prior_sd = self.get_nn_meta('bias_prior_sd', nn_id)
        bias_sd_init = self.get_nn_meta('bias_sd_init', nn_id)

        with self.session.as_default():
            with self.session.graph.as_default():
                units = n_units_rnn[l]
                rnn = RNNLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    activation=rnn_activation,
                    recurrent_activation=recurrent_activation,
                    bottomup_dropout=ff_dropout_rate,
                    h_dropout=rnn_h_dropout_rate,
                    c_dropout=rnn_c_dropout_rate,
                    return_sequences=True,
                    declare_priors_weights=declare_priors_weights,
                    declare_priors_biases=declare_priors_biases,
                    kernel_sd_prior=weight_prior_sd,
                    kernel_sd_init=weight_sd_init,
                    rangf_map=rangf_map,
                    weights_use_ranef=weights_use_ranef,
                    biases_use_ranef=biases_use_ranef,
                    normalizer_use_ranef=normalizer_use_ranef,
                    bias_sd_prior=bias_prior_sd,
                    bias_sd_init=bias_sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    ranef_to_fixef_prior_sd_ratio=self.ranef_to_fixef_prior_sd_ratio,
                    constraint=self.constraint,
                    name='%s_rnn_l%d' % (nn_id, l + 1),
                    epsilon=self.epsilon,
                    session=self.session
                )

                return rnn

    def _initialize_rnn(self, *args, **kwargs):
        if 'nn' in self.rvs:
            return self._initialize_rnn_bayes(*args, **kwargs)
        return self._initialize_rnn_mle(*args, **kwargs)

    def _initialize_nn(self, nn_id):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.regularizable_layers[nn_id] = []

                # Collect metadata for this NN component
                n_layers_ff = self.get_nn_meta('n_layers_ff', nn_id)
                n_units_ff = self.get_nn_meta('n_units_ff', nn_id)
                ff_inner_activation = self.get_nn_meta('ff_inner_activation', nn_id)
                ff_activation = self.get_nn_meta('ff_activation', nn_id)
                ff_dropout_rate = self.get_nn_meta('ff_dropout_rate', nn_id)
                normalize_ff = self.get_nn_meta('normalize_ff', nn_id)
                n_layers_rnn = self.get_nn_meta('n_layers_rnn', nn_id)
                n_units_rnn = self.get_nn_meta('n_units_rnn', nn_id)
                n_layers_rnn_projection = self.get_nn_meta('n_layers_rnn_projection', nn_id)
                n_units_rnn_projection = self.get_nn_meta('n_units_rnn_projection', nn_id)
                rnn_dropout_rate = self.get_nn_meta('rnn_dropout_rate', nn_id)
                rnn_projection_inner_activation = self.get_nn_meta('rnn_projection_inner_activation', nn_id)
                rnn_projection_activation = self.get_nn_meta('rnn_projection_activation', nn_id)
                h_rnn_dropout_rate = self.get_nn_meta('h_rnn_dropout_rate', nn_id)
                n_layers_irf = self.get_nn_meta('n_layers_irf', nn_id)
                n_units_irf = self.get_nn_meta('n_units_irf', nn_id)
                irf_inner_activation = self.get_nn_meta('irf_inner_activation', nn_id)
                irf_activation = self.get_nn_meta('irf_activation', nn_id)
                irf_dropout_rate = self.get_nn_meta('irf_dropout_rate', nn_id)
                ranef_dropout_rate = self.get_nn_meta('ranef_dropout_rate', nn_id)
                input_dropout_rate = self.get_nn_meta('input_dropout_rate', nn_id)
                maxnorm = self.get_nn_meta('maxnorm', nn_id)
                ranef_l1_only = self.get_nn_meta('ranef_l1_only', nn_id)
                dropout_final_layer = self.get_nn_meta('dropout_final_layer', nn_id)
                regularize_initial_layer = self.get_nn_meta('regularize_initial_layer', nn_id)
                regularize_final_layer = self.get_nn_meta('regularize_final_layer', nn_id)
                batch_normalization_decay = self.get_nn_meta('batch_normalization_decay', nn_id)
                layer_normalization_type = self.get_nn_meta('layer_normalization_type', nn_id)
                normalize_inputs = self.get_nn_meta('normalize_inputs', nn_id)
                normalize_irf = self.get_nn_meta('normalize_irf', nn_id)
                normalize_final_layer = self.get_nn_meta('normalize_final_layer', nn_id)
                nn_use_input_scaler = self.get_nn_meta('nn_use_input_scaler', nn_id)

                rangf_map = {}
                if ranef_dropout_rate:
                    Y_gf = self.Y_gf_dropout[ranef_dropout_rate]
                else:
                    Y_gf = self.Y_gf
                for i, gf in enumerate(self.rangf):
                    if gf in self.nns_by_id[nn_id].rangf:
                        _Y_gf = Y_gf[:, i]
                        rangf_map[gf] = (self.rangf_n_levels[self.rangf.index(gf)], _Y_gf)
                rangf_map_l1 = rangf_map
                if ranef_l1_only:
                    rangf_map_other = None
                else:
                    rangf_map_other = rangf_map

                if input_dropout_rate:
                    self.input_dropout_layer[nn_id] = get_dropout(
                        input_dropout_rate,
                        fixed=self.fixed_dropout,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        rescale=False,
                        name='%s_input_dropout' % nn_id,
                        session=self.session
                    )
                    self.X_time_dropout_layer[nn_id] = get_dropout(
                        input_dropout_rate,
                        fixed=self.fixed_dropout,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        rescale=False,
                        name='%s_X_time_dropout' % nn_id,
                        session=self.session
                    )

                # FEEDFORWARD ENCODER
                if self.has_ff(nn_id):
                    ff_layers = []
                    if nn_use_input_scaler:
                        scale_layer = self._initialize_scale(
                            nn_id,
                            rangf_map=rangf_map_l1,
                            name='%s_ff_input_scaler' % nn_id
                        )
                        self.regularizable_layers[nn_id].append(scale_layer)
                        ff_layers.append(scale_layer)

                    if normalize_inputs:
                        layer = self._initialize_feedforward(
                            nn_id,
                            0,  # Passing units=0 creates an identity kernel
                            batch_normalization_decay=batch_normalization_decay,
                            layer_normalization_type=layer_normalization_type,
                            rangf_map=rangf_map_l1,
                            name='%s_ff_input_normalization' % nn_id
                        )
                        ff_layers.append(layer)
                    for l in range(n_layers_ff + 1):
                        if l == 0 or not ranef_l1_only:
                            _rangf_map = rangf_map_l1
                        else:
                            _rangf_map = rangf_map_other

                        if l < n_layers_ff:
                            units = n_units_ff[l]
                            activation = ff_inner_activation
                            dropout = ff_dropout_rate
                            if normalize_ff:
                                bn = batch_normalization_decay
                                ln = layer_normalization_type
                            else:
                                bn = ln = None
                            use_bias = True
                        else:
                            units = 1
                            activation = ff_activation
                            if dropout_final_layer:
                                dropout = ff_dropout_rate
                            else:
                                dropout = None
                            if normalize_final_layer:
                                bn = batch_normalization_decay
                                ln = layer_normalization_type
                            else:
                                bn = ln = None
                            use_bias = False
                        mn = maxnorm

                        projection = self._initialize_feedforward(
                            nn_id,
                            units,
                            use_bias=use_bias,
                            activation=activation,
                            dropout=dropout,
                            maxnorm=mn,
                            batch_normalization_decay=bn,
                            layer_normalization_type=ln,
                            rangf_map=_rangf_map,
                            name='%s_ff_l%s' % (nn_id, l + 1)
                        )
                        self.layers.append(projection)
                        ff_layers.append(make_lambda(projection, session=self.session, use_kwargs=False))

                        if 'nn' not in self.rvs and \
                                (regularize_initial_layer or l > 0) and \
                                (regularize_final_layer or l < n_layers_ff):
                            self.regularizable_layers[nn_id].append(projection)
                            ff_layers.append(
                                make_lambda(
                                    lambda x: self._regularize(
                                        x,
                                        regtype='activity',
                                        var_name=reg_name('%s_ff_l%s_activity' % (nn_id, l + 1)),
                                        nn_id=nn_id
                                    ),
                                    session=self.session,
                                    use_kwargs=False
                                )
                            )

                    ff_fn = compose_lambdas(ff_layers)

                    self.ff_layers[nn_id] = ff_layers
                    self.ff_fn[nn_id] = ff_fn

                # RNN ENCODER
                if self.has_rnn(nn_id):
                    rnn_layers = []
                    rnn_h_ema = []
                    rnn_c_ema = []

                    if nn_use_input_scaler:
                        scale_layer = self._initialize_scale(
                            nn_id,
                            rangf_map=rangf_map_l1,
                            name='%s_rnn_input_scaler' % nn_id
                        )
                        self.regularizable_layers[nn_id].append(scale_layer)
                        rnn_layers.append(scale_layer)

                    for l in range(n_layers_rnn):
                        units = n_units_rnn[l]
                        if l == 0:
                            _rangf_map = rangf_map_l1
                        else:
                            _rangf_map = rangf_map_other
                        layer = self._initialize_rnn(nn_id, l, rangf_map=_rangf_map)
                        _rnn_h_ema = tf.Variable(tf.zeros(units), trainable=False, name='%s_rnn_h_ema_l%d' % (nn_id, l+1))
                        rnn_h_ema.append(_rnn_h_ema)
                        _rnn_c_ema = tf.Variable(tf.zeros(units), trainable=False, name='%s_rnn_c_ema_l%d' % (nn_id, l+1))
                        rnn_c_ema.append(_rnn_c_ema)
                        self.layers.append(layer)
                        rnn_layers.append(make_lambda(layer, session=self.session, use_kwargs=True))
                        if 'nn' not in self.rvs:
                            self.regularizable_layers[nn_id].append(layer)
                            rnn_layers.append(
                                make_lambda(
                                    lambda x, nn_id=nn_id, l=l: self._regularize(
                                        x,
                                        regtype='activity',
                                        var_name=reg_name('%s_rnn_l%d_activity' % (nn_id, l + 1)),
                                        nn_id=nn_id
                                    ),
                                    session=self.session,
                                    use_kwargs=False
                                )
                            )

                    rnn_encoder = compose_lambdas(rnn_layers)

                    self.rnn_layers[nn_id] = rnn_layers
                    self.rnn_h_ema[nn_id] = rnn_h_ema
                    self.rnn_c_ema[nn_id] = rnn_c_ema
                    self.rnn_encoder[nn_id] = rnn_encoder

                    rnn_projection_layers = []
                    for l in range(n_layers_rnn_projection + 1):
                        if l < n_layers_rnn_projection:
                            units = n_units_rnn_projection[l]
                            activation = rnn_projection_inner_activation
                            bn = batch_normalization_decay
                            ln = layer_normalization_type
                            use_bias = True
                        else:
                            if nn_id in self.nn_irf_ids:
                                units = len([x for x in self.nn_irf_input_names[nn_id] if x != 'rate'])
                            else:
                                units = 1
                            activation = rnn_projection_activation
                            if normalize_final_layer:
                                bn = batch_normalization_decay
                                ln = layer_normalization_type
                            else:
                                bn = ln = None
                            use_bias = False
                        mn = maxnorm

                        projection = self._initialize_feedforward(
                            nn_id,
                            units,
                            use_bias=use_bias,
                            activation=activation,
                            dropout=None,
                            maxnorm=mn,
                            batch_normalization_decay=bn,
                            layer_normalization_type=ln,
                            rangf_map=rangf_map,
                            name='%s_rnn_projection_l%s' % (nn_id, l + 1)
                        )
                        self.layers.append(projection)

                        if 'nn' not in self.rvs:
                            self.regularizable_layers[nn_id].append(projection)
                            rnn_layers.append(
                                make_lambda(
                                    lambda x, nn_id=nn_id, l=l: self._regularize(
                                        x,
                                        regtype='activity',
                                        var_name=reg_name('%s_rnn_projection_l%s_activity' % (nn_id, l + 1)),
                                        nn_id=nn_id
                                    ),
                                    session=self.session,
                                    use_kwargs=False
                                )
                            )
                        rnn_projection_layers.append(make_lambda(projection, session=self.session, use_kwargs=False))

                    rnn_projection_fn = compose_lambdas(rnn_projection_layers)

                    self.rnn_projection_layers[nn_id] = rnn_projection_layers
                    self.rnn_projection_fn[nn_id] = rnn_projection_fn

                    self.h_rnn_dropout_layer[nn_id] = get_dropout(
                        h_rnn_dropout_rate,
                        fixed=self.fixed_dropout,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        name='%s_h_rnn_dropout' % nn_id,
                        session=self.session
                    )
                    self.rnn_dropout_layer[nn_id] = get_dropout(
                        rnn_dropout_rate,
                        noise_shape=[None, None, 1],
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        rescale=False,
                        name='%s_rnn_dropout' % nn_id,
                        session=self.session
                    )

                # IRF
                if nn_id in self.nn_irf_ids:
                    output_ndim = self.get_nn_irf_output_ndim(nn_id)
                    if output_ndim:
                        irf_layers = []

                        if nn_use_input_scaler:
                            scale_layer = self._initialize_scale(
                                nn_id,
                                rangf_map=rangf_map_l1,
                                name='%s_irf_input_scaler' % nn_id
                            )
                            self.regularizable_layers[nn_id].append(scale_layer)
                            irf_layers.append(scale_layer)

                        if normalize_inputs:
                            layer = self._initialize_feedforward(
                                nn_id,
                                0,  # Passing units=0 creates an identity kernel
                                batch_normalization_decay=batch_normalization_decay,
                                layer_normalization_type=layer_normalization_type,
                                rangf_map=rangf_map_l1,
                                name='%s_irf_input_normalization' % nn_id
                            )
                            irf_layers.append(layer)

                        for l in range(n_layers_irf + 1):
                            if l == 0 or not ranef_l1_only:
                                _rangf_map = rangf_map_l1
                            else:
                                _rangf_map = rangf_map_other

                            if l < n_layers_irf:
                                units = n_units_irf[l]
                                activation = irf_inner_activation
                                dropout = irf_dropout_rate
                                if normalize_irf:
                                    bn = batch_normalization_decay
                                    ln = layer_normalization_type
                                else:
                                    bn = None
                                    ln = None
                                use_bias = True
                                final = False
                                mn = maxnorm
                            else:
                                units = output_ndim
                                activation = irf_activation
                                if dropout_final_layer:
                                    dropout = irf_dropout_rate
                                else:
                                    dropout = None
                                if normalize_final_layer:
                                    bn = batch_normalization_decay
                                    ln = layer_normalization_type
                                else:
                                    bn = ln = None
                                use_bias = False
                                final = True
                                mn = None

                            projection = self._initialize_feedforward(
                                nn_id,
                                units,
                                use_bias=use_bias,
                                activation=activation,
                                dropout=dropout,
                                maxnorm=mn,
                                batch_normalization_decay=bn,
                                layer_normalization_type=ln,
                                rangf_map=_rangf_map,
                                final=final,
                                name='%s_irf_l%s' % (nn_id, l + 1)
                            )
                            self.layers.append(projection)
                            irf_layers.append(projection)

                            if 'nn' not in self.rvs and \
                                    (regularize_initial_layer or l > 0) and \
                                    (regularize_final_layer or l < n_layers_irf):
                                self.regularizable_layers[nn_id].append(projection)
                                irf_layers.append(
                                    make_lambda(
                                        lambda x, nn_id=nn_id, l=l: self._regularize(
                                            x,
                                            regtype='activity',
                                            var_name=reg_name('%s_irf_l%s_activity' % (nn_id, l + 1)),
                                            nn_id=nn_id
                                        ),
                                        session=self.session,
                                        use_kwargs=False
                                    )
                                )
                            if l == 0:
                                self.nn_irf_l1[nn_id] = projection

                        self.nn_irf_layers[nn_id] = irf_layers

    def _compile_nn(self, nn_id):
        with self.session.as_default():
            with self.session.graph.as_default():
                # Collect metadata for this NN
                center_X_time = self.get_nn_meta('center_X_time', nn_id)
                center_t_delta = self.get_nn_meta('center_t_delta', nn_id)
                rescale_X_time = self.get_nn_meta('rescale_X_time', nn_id)
                rescale_t_delta = self.get_nn_meta('rescale_t_delta', nn_id)
                log_transform_t_delta = self.get_nn_meta('log_transform_t_delta', nn_id)
                ff_noise_sd = self.get_nn_meta('ff_noise_sd', nn_id)
                n_layers_rnn = self.get_nn_meta('n_layers_rnn', nn_id)
                rnn_dropout_rate = self.get_nn_meta('rnn_dropout_rate', nn_id)
                h_rnn_noise_sd = self.get_nn_meta('h_rnn_noise_sd', nn_id)
                h_rnn_dropout_rate = self.get_nn_meta('h_rnn_dropout_rate', nn_id)
                input_jitter_level = self.get_nn_meta('input_jitter_level', nn_id)
                input_dropout_rate = self.get_nn_meta('input_dropout_rate', nn_id)
                nonstationary = self.get_nn_meta('nonstationary', nn_id)
                input_dependent_irf = self.get_nn_meta('input_dependent_irf', nn_id)

                if nn_id in self.nn_impulse_ids:
                    impulse_names = self.nn_impulse_impulse_names[nn_id]
                    input_names = impulse_names
                    output_names = []
                else:  # nn_id in self.nn_irf_ids
                    impulse_names = self.nn_irf_impulse_names[nn_id]
                    input_names = self.nn_irf_input_names[nn_id]
                    output_names = self.nn_irf_output_names[nn_id]

                X = []
                t_delta = []
                X_time = []
                X_mask = []
                dirac_delta_mask = []
                impulse_names_ordered = []

                # Collect non-neural impulses
                non_nn_impulse_names = [x for x in impulse_names if x in self.impulse_names]
                if len(non_nn_impulse_names):
                    impulse_ix = names2ix(non_nn_impulse_names, self.impulse_names)
                    X.append(tf.gather(self.X_processed, impulse_ix, axis=2))
                    t_delta.append(tf.gather(self.t_delta, impulse_ix, axis=2))
                    X_time.append(tf.gather(self.X_time, impulse_ix, axis=2))
                    X_mask.append(tf.gather(self.X_mask, impulse_ix, axis=2))
                    impulse_names_ordered += non_nn_impulse_names

                # Collect neurally transformed impulses
                nn_impulse_names = [x for x in impulse_names if x not in self.impulse_names]
                assert not len(nn_impulse_names) or nn_id in self.nn_irf_ids, 'NN impulse transforms may not be nested.'
                if len(nn_impulse_names):
                    all_nn_impulse_names = [self.nns_by_id[x].name() for x in self.nn_impulse_ids]
                    impulse_ix = names2ix(nn_impulse_names, all_nn_impulse_names)
                    X.append(tf.gather(self.nn_transformed_impulses, impulse_ix, axis=2))
                    t_delta.append(tf.gather(self.nn_transformed_impulse_t_delta, impulse_ix, axis=2))
                    X_time.append(tf.gather(self.nn_transformed_impulse_X_time, impulse_ix, axis=2))
                    X_mask.append(tf.gather(self.nn_transformed_impulse_X_mask, impulse_ix, axis=2))
                    impulse_names_ordered += nn_impulse_names

                assert len(impulse_names_ordered), 'NN transform must get at least one input'
                # Pad and concatenate impulses, deltas, timestamps, and masks
                if len(X) == 1:
                    X = X[0]
                else:
                    max_len = tf.reduce_max([tf.shape(x)[1] for x in X])  # Get maximum timesteps
                    X = [
                        tf.pad(x, ((0, 0), (max_len - tf.shape(x)[1], 0), (0, 0))) for x in X
                    ]
                    X = tf.concat(X, axis=2)
                if len(t_delta) == 1:
                    t_delta = t_delta[0]
                else:
                    max_len = tf.reduce_max([tf.shape(x)[1] for x in t_delta])  # Get maximum timesteps
                    t_delta = [
                        tf.pad(x, ((0, 0), (max_len - tf.shape(x)[1], 0), (0, 0))) for x in t_delta
                    ]
                    t_delta = tf.concat(t_delta, axis=2)
                if len(X_time) == 1:
                    X_time = X_time[0]
                else:
                    max_len = tf.reduce_max([tf.shape(x)[1] for x in X_time])  # Get maximum timesteps
                    X_time = [
                        tf.pad(x, ((0, 0), (max_len - tf.shape(x)[1], 0), (0, 0))) for x in X_time
                    ]
                    X_time = tf.concat(X_time, axis=2)
                if len(X_mask) == 1:
                    X_mask = X_mask[0]
                else:
                    max_len = tf.reduce_max([tf.shape(x)[1] for x in X_mask])  # Get maximum timesteps
                    X_mask = [
                        tf.pad(x, ((0, 0), (max_len - tf.shape(x)[1], 0), (0, 0))) for x in X_mask
                    ]
                    X_mask = tf.concat(X_mask, axis=2)
                    
                # Reorder impulses if needed (i.e. if both neural and non-neural impulses are included, they
                # will be out of order relative to impulse_names)
                if len(non_nn_impulse_names) and len(nn_impulse_names):
                    impulse_ix = names2ix(impulse_names_ordered, impulse_names)
                    X = tf.gather(X, impulse_ix, axis=2)
                    t_delta = tf.gather(t_delta, impulse_ix, axis=2)
                    X_time = tf.gather(X_time, impulse_ix, axis=2)
                    X_mask = tf.gather(X_mask, impulse_ix, axis=2)

                if center_X_time:
                    X_time -= self.X_time_mean
                if center_t_delta:
                    t_delta -= self.t_delta_mean

                if rescale_X_time:
                    X_time /= self.X_time_sd
                if rescale_t_delta:
                    t_delta /= self.t_delta_sd

                # Handle multiple impulse streams with different timestamps
                # by interleaving the impulses in temporal order
                if self.n_impulse_df_noninteraction > 1:
                    X_cdrnn = []
                    t_delta_cdrnn = []
                    X_time_cdrnn = []
                    X_mask_cdrnn = []

                    X_shape = tf.shape(X)
                    B = X_shape[0]
                    T = X_shape[1]

                    for i, ix in enumerate(self.impulse_indices):
                        if len(ix) > 0:
                            dim_mask = np.zeros(len(self.impulse_names))
                            dim_mask[ix] = 1
                            dim_mask = tf.constant(dim_mask, dtype=self.FLOAT_TF)
                            while len(dim_mask.shape) < len(X.shape):
                                dim_mask = dim_mask[None, ...]
                            dim_mask = tf.gather(dim_mask, impulse_ix, axis=2)
                            X_cur = X * dim_mask

                            if t_delta.shape[-1] > 1:
                                t_delta_cur = t_delta[..., ix[0]:ix[0] + 1]
                            else:
                                t_delta_cur = t_delta

                            if X_time.shape[-1] > 1:
                                _X_time = X_time[..., ix[0]:ix[0] + 1]
                            else:
                                _X_time = X_time

                            if X_mask is not None and X_mask.shape[-1] > 1:
                                _X_mask = X_mask[..., ix[0]]
                            else:
                                _X_mask = X_mask

                            X_cdrnn.append(X_cur)
                            t_delta_cdrnn.append(t_delta_cur)
                            X_time_cdrnn.append(_X_time)
                            if X_mask is not None:
                                X_mask_cdrnn.append(_X_mask)

                    X_cdrnn = tf.concat(X_cdrnn, axis=1)
                    t_delta_cdrnn = tf.concat(t_delta_cdrnn, axis=1)
                    X_time_cdrnn = tf.concat(X_time_cdrnn, axis=1)
                    if X_mask is not None:
                        X_mask_cdrnn = tf.concat(X_mask_cdrnn, axis=1)

                    sort_ix = tf_argsort(tf.squeeze(X_time_cdrnn, axis=-1), axis=1)
                    B_ix = tf.tile(
                        tf.range(B)[..., None],
                        [1, T * self.n_impulse_df_noninteraction]
                    )
                    gather_ix = tf.stack([B_ix, sort_ix], axis=-1)

                    X = tf.gather_nd(X_cdrnn, gather_ix)
                    t_delta = tf.gather_nd(t_delta_cdrnn, gather_ix)
                    X_time = tf.gather_nd(X_time_cdrnn, gather_ix)
                    if X_mask is not None:
                        X_mask = tf.gather_nd(X_mask_cdrnn, gather_ix)
                else:
                    t_delta = t_delta[..., :1]
                    X_time = X_time[..., :1]
                    if X_mask is not None and len(X_mask.shape) == 3:
                        X_mask = X_mask[..., 0]
                    
                if input_jitter_level:
                    jitter_sd = input_jitter_level
                    X = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(X), X, jitter_sd),
                        lambda: X
                    )
                    t_delta = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(t_delta), t_delta, jitter_sd),
                        lambda: t_delta
                    )
                    X_time = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(X_time), X_time, jitter_sd),
                        lambda: X_time
                    )

                if input_dropout_rate:
                    X = self.input_dropout_layer[nn_id](X)
                    X_time = self.X_time_dropout_layer[nn_id](X_time)

                impulse_ix = names2ix([x for x in input_names if x != 'rate'], impulse_names)
                X_gathered = tf.gather(X, impulse_ix, axis=2)
                X_in = X_gathered
                if nonstationary:
                    X_in = tf.concat([X_in, X_time], axis=-1)

                # Compute hidden state
                h = h_ff = h_rnn = rnn_hidden = rnn_cell = None

                if self.has_ff(nn_id) or self.has_rnn(nn_id):
                    _X_in = X_in
                    if self.has_ff(nn_id):
                        h_ff = self.ff_fn[nn_id](_X_in)
                        if ff_noise_sd:
                            def ff_train_fn(ff=h_ff):
                                return tf.random_normal(tf.shape(ff), ff, stddev=ff_noise_sd[nn_id])
                            def ff_eval_fn(ff=h_ff):
                                return ff
                            h_ff = tf.cond(self.training, ff_train_fn, ff_eval_fn)
                        h = h_ff

                    if self.has_rnn(nn_id):
                        rnn_hidden = []
                        rnn_cell = []
                        for l in range(n_layers_rnn):
                            _rnn_hidden, _rnn_cell = self.rnn_layers[nn_id][l](
                                _X_in,
                                return_state=True,
                                mask=X_mask
                            )
                            rnn_hidden.append(_rnn_hidden)
                            rnn_cell.append(_rnn_cell)
                            _X_in = _rnn_hidden

                        h_rnn = self.rnn_projection_fn[nn_id](rnn_hidden[-1])

                        if rnn_dropout_rate:
                            h_rnn = self.rnn_dropout_layer[nn_id](h_rnn)

                        if h_rnn_noise_sd:
                            def h_rnn_train_fn(h_rnn=h_rnn):
                                return tf.random_normal(tf.shape(h_rnn), h_rnn, stddev=h_rnn_noise_sd)
                            def h_rnn_eval_fn(h_rnn=h_rnn):
                                return h_rnn
                            h_rnn = tf.cond(self.training, h_rnn_train_fn, h_rnn_eval_fn)
                        if h_rnn_dropout_rate:
                            h_rnn = self.h_rnn_dropout_layer[nn_id](h_rnn)

                        if h is None:
                            h = h_rnn
                        else:
                            h += h_rnn

                if nn_id in self.nn_impulse_ids:
                    assert h is not None, 'NN impulse transforms must involve a feedforward component, an RNN component, or both.'
                    self.nn_transformed_impulses.append(h)
                    self.nn_transformed_impulse_t_delta.append(t_delta)
                    self.nn_transformed_impulse_X_time.append(X_time)
                    self.nn_transformed_impulse_X_mask.append(X_mask[..., None])
                    dirac_delta_mask.append(
                        tf.cast(tf.abs(t_delta) < self.epsilon, dtype=self.FLOAT_TF), 
                    )
                    self.nn_transformed_impulse_dirac_delta_mask.append(dirac_delta_mask)

                else:  # nn_id in self.nn_irf_ids
                    # Compute IRF outputs

                    output_ndim = self.get_nn_irf_output_ndim(nn_id)
                    if output_ndim:

                        impulse_ix = names2ix(output_names, impulse_names)
                        nn_irf_impulses = tf.gather(X, impulse_ix, axis=2) # IRF output dims, includes rate

                        if log_transform_t_delta:
                            t_delta = tf.sign(t_delta) * tf.log1p(tf.abs(t_delta))

                        irf_out = [t_delta]
                        if nonstationary:
                            irf_out.append(X_time)
                        if input_dependent_irf:
                            _X_gathered = X_gathered
                            if h_rnn is not None:
                                _X_gathered = _X_gathered + h_rnn
                            irf_out.append(_X_gathered)  # IRF inputs, no rate
                        irf_out = tf.concat(irf_out, axis=2)
                        for layer in self.nn_irf_layers[nn_id]:
                            irf_out = layer(
                                irf_out
                            )

                        stabilizing_constant = (self.history_length + self.future_length) * len(output_names)
                        irf_out = irf_out / stabilizing_constant

                        nn_irf_impulses = nn_irf_impulses[..., None, None] # Pad out for ndim of response distribution(s)
                        self.nn_irf_impulses[nn_id] = nn_irf_impulses

                        # Slice and apply IRF outputs
                        slices, shapes = self.get_nn_irf_output_slice_and_shape(nn_id)
                        if X_mask is None:
                            X_mask_out = None
                        else:
                            X_mask_out = X_mask[..., None, None, None] # Pad out for impulses plus nparam, ndim of response distribution(s)
                        _X_time = X_time[..., None, None]

                        for i, response in enumerate(self.response_names):
                            _slice = slices[response]
                            _shape = shapes[response]

                            _irf_out = tf.reshape(irf_out[..., _slice], _shape)
                            if X_mask_out is not None:
                                _irf_out = _irf_out * X_mask_out

                            if response not in self.nn_irf:
                                self.nn_irf[response] = {}
                            self.nn_irf[response][nn_id] = _irf_out

                # Set up EMA for RNN
                ema_rate = self.ema_decay
                if ema_rate is None:
                    ema_rate = 0.

                mask = X_mask[..., None]
                denom = tf.reduce_sum(mask)

                if h_rnn is not None:
                    h_rnn_masked = h_rnn * mask
                    self._regularize(h_rnn_masked, regtype='context', var_name=reg_name('context'), nn_id=nn_id)

                if nn_id in self.nn_impulse_ids or input_dependent_irf:
                    for l in range(n_layers_rnn):
                        reduction_axes = list(range(len(rnn_hidden[l].shape) - 1))

                        h_sum = tf.reduce_sum(rnn_hidden[l] * mask, axis=reduction_axes)
                        h_mean = h_sum / (denom + self.epsilon)
                        h_ema = self.rnn_h_ema[nn_id][l]
                        h_ema_op = tf.assign(
                            h_ema,
                            ema_rate * h_ema + (1. - ema_rate) * h_mean
                        )
                        self.ema_ops.append(h_ema_op)

                        c_sum = tf.reduce_sum(rnn_cell[l] * mask, axis=reduction_axes)
                        c_mean = c_sum / (denom + self.epsilon)
                        c_ema = self.rnn_c_ema[nn_id][l]
                        c_ema_op = tf.assign(
                            c_ema,
                            ema_rate * c_ema + (1. - ema_rate) * c_mean
                        )
                        self.ema_ops.append(c_ema_op)

                if input_dropout_rate:
                    self.resample_ops += self.input_dropout_layer[nn_id].resample_ops() + self.X_time_dropout_layer[nn_id].resample_ops()
                if rnn_dropout_rate and n_layers_rnn:
                    self.resample_ops += self.h_rnn_dropout_layer[nn_id].resample_ops()
                    self.resample_ops += self.rnn_dropout_layer[nn_id].resample_ops()
                if h_rnn_dropout_rate and n_layers_rnn and nn_id in self.h_rnn_dropout_layer:
                    self.resample_ops += self.h_rnn_dropout_layer[nn_id].resample_ops()

    def _concat_nn_impulses(self):
        if len(self.nn_transformed_impulses):
            if len(self.nn_transformed_impulses) == 1:
                self.nn_transformed_impulses = self.nn_transformed_impulses[0]
            else:
                self.nn_transformed_impulses = tf.concat(self.nn_transformed_impulses, axis=2)
            if len(self.nn_transformed_impulse_t_delta) == 1:
                self.nn_transformed_impulse_t_delta = self.nn_transformed_impulse_t_delta[0]
            else:
                self.nn_transformed_impulse_t_delta = tf.concat(self.nn_transformed_impulse_t_delta, axis=2)
            if len(self.nn_transformed_impulse_X_time) == 1:
                self.nn_transformed_impulse_X_time = self.nn_transformed_impulse_X_time[0]
            else:
                self.nn_transformed_impulse_X_time = tf.concat(self.nn_transformed_impulse_X_time, axis=2)
            if len(self.nn_transformed_impulse_X_mask) == 1:
                self.nn_transformed_impulse_X_mask = self.nn_transformed_impulse_X_mask[0]
            else:
                self.nn_transformed_impulse_X_mask = tf.concat(self.nn_transformed_impulse_X_mask, axis=2)
            if len(self.nn_transformed_impulse_dirac_delta_mask) == 1:
                self.nn_transformed_impulse_dirac_delta_mask = self.nn_transformed_impulse_dirac_delta_mask[0]
            else:
                self.nn_transformed_impulse_dirac_delta_mask = tf.concat(self.nn_transformed_impulse_dirac_delta_mask, axis=2)

    def _collect_layerwise_ops(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                for x in self.layers:
                    self.ema_ops += x.ema_ops()
                    self.resample_ops += x.resample_ops()

    def _initialize_interaction_mle(self, response, interaction_ids=None, ran_gf=None):
        if interaction_ids is None:
            interaction_ids = self.interaction_names

        if self.use_distributional_regression:
            nparam = self.get_response_nparam(response)
        else:
            nparam = 1
        ndim = self.get_response_ndim(response)
        ninter = len(interaction_ids)

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    interaction = tf.Variable(
                        tf.zeros([ninter, nparam, ndim], dtype=self.FLOAT_TF),
                        name='interaction_%s' % sn(response)
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    interaction = tf.Variable(
                        tf.zeros([rangf_n_levels, ninter, nparam, ndim], dtype=self.FLOAT_TF),
                        name='interaction_%s_by_%s' % (sn(response), sn(ran_gf))
                    )

                # shape: (?rangf_n_levels, ninter, nparam, ndim)

                return {'value': interaction}

    def _initialize_interaction_bayes(self, response, interaction_ids=None, ran_gf=None):
        if interaction_ids is None:
            interaction_ids = self.interaction_names

        if self.use_distributional_regression:
            nparam = self.get_response_nparam(response)
        else:
            nparam = 1
        ndim = self.get_response_ndim(response)
        ninter = len(interaction_ids)

        with self.session.as_default():
            with self.session.graph.as_default():
                if ran_gf is None:
                    sd_prior = self._coef_prior_sd[response]
                    sd_posterior = self._coef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        sd_prior = sd_prior[:1]
                        sd_posterior = sd_posterior[:1]
                    sd_prior = np.ones((ninter, 1, 1)) * sd_prior[None, ...]
                    sd_posterior = np.ones((ninter, 1, 1)) * sd_posterior[None, ...]

                    rv_dict = get_random_variable(
                        'interaction_%s' % sn(response),
                        sd_posterior.shape,
                        sd_posterior,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )
                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_fixef:
                        out['kl_penalties'] = rv_dict['kl_penalties']
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    sd_prior = self._coef_ranef_prior_sd[response]
                    sd_posterior = self._coef_ranef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        sd_prior = sd_prior[:1]
                        sd_posterior = sd_posterior[:1]
                    sd_prior = np.ones((rangf_n_levels, ninter, 1, 1)) * sd_prior[None, None, ...]
                    sd_posterior = np.ones((rangf_n_levels, ninter, 1, 1)) * sd_posterior[None, None, ...]

                    rv_dict = get_random_variable(
                        'interaction_%s_by_%s' % (sn(response), sn(ran_gf)),
                        sd_posterior.shape,
                        sd_posterior,
                        constraint=self.constraint,
                        sd_prior=sd_prior,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        epsilon=self.epsilon,
                        session=self.session
                    )
                    out = {
                        'value': rv_dict['v'],
                        'eval_resample': rv_dict['v_eval_resample']
                    }
                    if self.declare_priors_ranef:
                        out['kl_penalties'] = rv_dict['kl_penalties']

                # shape: (?rangf_n_levels, ninter, nparam, ndim)

                return out

    def _initialize_interaction(self, *args, **kwargs):
        if 'interaction' in self.rvs:
            return self._initialize_interaction_bayes(*args, **kwargs)
        return self._initialize_interaction_mle(*args, **kwargs)

    def _compile_interactions(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.interaction = {}
                self.interaction_fixed = {}
                self.interaction_random = {}
                fixef_ix = names2ix(self.fixed_interaction_names, self.interaction_names)
                if len(self.interaction_names) > 0:
                    for response in self.response_names:
                        response_params = self.get_response_params(response)
                        if not self.use_distributional_regression:
                            response_params = response_params[:1]
                        nparam = len(response_params)
                        ndim = self.get_response_ndim(response)
                        interaction_ids = self.interaction_names

                        interaction_fixed = self._scatter_along_axis(
                            fixef_ix,
                            self.interaction_fixed_base[response],
                            [len(interaction_ids), nparam, ndim]
                        )
                        self.interaction_fixed[response] = interaction_fixed

                        self._regularize(
                            self.interaction_fixed_base[response],
                            regtype='coefficient',
                            var_name='interaction_%s' % response
                        )

                        interaction = interaction_fixed[None, ...]

                        if self.log_fixed:
                            for i, interaction_name in enumerate(self.interaction_names):
                                for j, response_param in enumerate(response_params):
                                    dim_names = self.expand_param_name(response, response_param)
                                    for k, dim_name in enumerate(dim_names):
                                        val = interaction_fixed[i, j, k]
                                        tf.summary.scalar(
                                            'interaction' + '/%s/%s_%s' % (
                                                sn(interaction_name),
                                                sn(response),
                                                sn(dim_name)
                                            ),
                                            val,
                                            collections=['params']
                                        )

                        for i, gf in enumerate(self.rangf):
                            levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                            interactions = self.interaction_by_rangf.get(gf, [])
                            if len(interactions) > 0:
                                interaction_ix = names2ix(interactions, self.interaction_names)

                                interaction_random = self.interaction_random_base[response][gf]
                                interaction_random_means = tf.reduce_mean(interaction_random, axis=0, keepdims=True)
                                interaction_random -= interaction_random_means
                                if response not in self.interaction_random:
                                    self.interaction_random[response] = {}
                                self.interaction_random[response][gf] = interaction_random

                                self._regularize(
                                    interaction_random,
                                    regtype='ranef',
                                    var_name='interaction_%s_by_%s' % (sn(response), sn(gf))
                                )

                                if self.log_random:
                                    for j, interaction_name in enumerate(interactions):
                                        for k, response_param in enumerate(response_params):
                                            dim_names = self.expand_param_name(response, response_param)
                                            for l, dim_name in enumerate(dim_names):
                                                val = interaction_random[:, j, k, l]
                                                tf.summary.histogram(
                                                    'by_%s/interaction/%s/%s_%s' % (
                                                        sn(gf),
                                                        sn(interaction_name),
                                                        sn(response),
                                                        sn(dim_name)
                                                    ),
                                                    val,
                                                    collections=['random']
                                                )

                                interaction_random = self._scatter_along_axis(
                                    interaction_ix,
                                    self._scatter_along_axis(
                                        levels_ix,
                                        interaction_random,
                                        [self.rangf_n_levels[i], len(interactions), nparam, ndim]
                                    ),
                                    [self.rangf_n_levels[i], len(self.interaction_names), nparam, ndim],
                                    axis=1
                                )

                                interaction = interaction + tf.gather(interaction_random, self.Y_gf[:, i], axis=0)

                        self.interaction[response] = interaction

    def _apply_interactions(self, response):
        with self.session.as_default():
            with self.session.graph.as_default():
                if len(self.interaction_names) > 0:
                    interaction_coefs = tf.expand_dims(self.interaction[response], axis=1)
                    interaction_inputs = []
                    terminal_names = self.terminal_names[:]
                    impulse_names = self.impulse_names
                    nn_impulse_names = [self.nns_by_id[x].name() for x in self.nn_impulse_ids]
                    nparam = self.get_response_nparam(response)
                    ndim = self.get_response_ndim(response)
                    dd = {}

                    for i, interaction in enumerate(self.interaction_list):
                        assert interaction.name() == self.interaction_names[i], 'Mismatched sort order between self.interaction_names and self.interaction_list. This should not have happened, so please report it in issue tracker on Github.'
                        irf_input_names = [x.name() for x in interaction.irf_responses()]
                        nn_impulse_input_names = [x.name() for x in interaction.nn_impulse_responses()]
                        dirac_delta_input_names = [x.name() for x in interaction.dirac_delta_responses()]

                        inputs_cur = []

                        if len(irf_input_names) > 0:
                            irf_input_ix = names2ix(irf_input_names, terminal_names)
                            irf_inputs = self.X_weighted_unscaled_sumT[response]
                            irf_inputs = tf.gather(
                                irf_inputs,
                                irf_input_ix,
                                axis=2
                            )
                            inputs_cur.append(irf_inputs)

                        if len(nn_impulse_input_names):
                            nn_impulse_input_ix = names2ix(nn_impulse_input_names, nn_impulse_names)
                            nn_impulse_inputs = tf.gather(self.nn_transformed_impulses, nn_impulse_input_ix, axis=2)
                            nn_impulse_mask = tf.gather(self.nn_transformed_impulse_dirac_delta_mask, nn_impulse_input_ix, axis=2)
                            nn_impulse_inputs = tf.reduce_sum(nn_impulse_inputs * nn_impulse_mask, axis=1, keepdims=True)
                            # Expand out response_param and response_param_dim axes
                            nn_impulse_inputs = nn_impulse_inputs[..., None, None]
                            nn_impulse_inputs = tf.pad(
                                nn_impulse_inputs,
                                paddings=[
                                    (0, 0),
                                    (0, 0),
                                    (0, 0),
                                    (0, nparam - 1),
                                    (0, ndim - 1)
                                ]
                            )
                            inputs_cur.append(nn_impulse_inputs)
                                
                        if len(dirac_delta_input_names):
                            dd_key = tuple(dirac_delta_input_names)
                            if dd_key not in dd:
                                dirac_delta_input_ix = names2ix(dirac_delta_input_names, impulse_names)
                                # Expand out response_param and response_param_dim axes
                                dirac_delta_inputs = tf.gather(self.X_processed, dirac_delta_input_ix, axis=2)
                                dirac_delta_mask = tf.gather(self.dirac_delta_mask, dirac_delta_input_ix, axis=2)
                                dirac_delta_inputs = tf.reduce_sum(dirac_delta_inputs * dirac_delta_mask, axis=1, keepdims=True)
                                dirac_delta_inputs = dirac_delta_inputs[..., None, None]
                                dirac_delta_inputs = tf.pad(
                                    dirac_delta_inputs,
                                    paddings=[
                                        (0, 0),
                                        (0, 0),
                                        (0, 0),
                                        (0, nparam - 1),
                                        (0, ndim - 1)
                                    ]
                                )
                                dd[dd_key] = dirac_delta_inputs
                            dirac_delta_inputs = dd[dd_key]
                            inputs_cur.append(dirac_delta_inputs)

                        inputs_cur = tf.concat(inputs_cur, axis=1)
                        inputs_cur = tf.reduce_prod(inputs_cur, axis=1)

                        interaction_inputs.append(inputs_cur)
                    interaction_inputs = tf.stack(interaction_inputs, axis=2)

                    return interaction_coefs * interaction_inputs

    def _compile_irf_impulses(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                # Parametric IRFs with non-neural impulses
                irf_impulses = []
                terminal_names = []
                parametric_terminals = [x for x in self.parametric_irf_terminals if not x.impulse.is_nn_impulse()]
                parametric_terminal_names = [x.name() for x in parametric_terminals]
                impulse_names = [x.impulse.name() for x in parametric_terminals]
                if len(impulse_names):
                    impulse_ix = names2ix(impulse_names, self.impulse_names)
                    parametric_irf_impulses = tf.gather(self.X_processed, impulse_ix, axis=2)
                    parametric_irf_impulses = parametric_irf_impulses[..., None, None] # Pad out for response distribution param,dim
                    irf_impulses.append(parametric_irf_impulses)
                    terminal_names += parametric_terminal_names

                # Parametric IRFs with neural impulses
                parametric_terminals = [x for x in self.parametric_irf_terminals if x.impulse.is_nn_impulse()]
                parametric_terminal_names = [x.name() for x in parametric_terminals]
                nn_impulse_names = [self.nns_by_id[x].name() for x in self.nn_impulse_ids]
                impulse_names = [x.impulse.name() for x in parametric_terminals]
                if len(impulse_names):
                    impulse_ix = names2ix(impulse_names, nn_impulse_names)
                    parametric_irf_impulses = tf.gather(self.nn_transformed_impulses, impulse_ix, axis=2)
                    parametric_irf_impulses = parametric_irf_impulses[..., None, None] # Pad out for response distribution param,dim
                    irf_impulses.append(parametric_irf_impulses)
                    terminal_names += parametric_terminal_names

                for nn_id in self.nn_irf_ids:
                    if self.nn_irf_impulses[nn_id] is not None:
                        irf_impulses.append(self.nn_irf_impulses[nn_id])
                        terminal_names += self.nn_irf_terminal_names[nn_id]

                if len(irf_impulses):
                    if len(irf_impulses) == 1:
                        irf_impulses = irf_impulses[0]
                    else:
                        max_len = tf.reduce_max([tf.shape(x)[1] for x in irf_impulses]) # Get maximum timesteps
                        irf_impulses = [
                            tf.pad(x, ((0,0), (max_len-tf.shape(x)[1], 0), (0,0), (0,0), (0,0))) for x in irf_impulses
                        ]
                        irf_impulses = tf.concat(irf_impulses, axis=2)
                else:
                    irf_impulses = None

                if irf_impulses is not None:
                    assert irf_impulses.shape[2] == len(self.terminal_names), 'There should be exactly 1 IRF impulse per terminal. Got %d impulses and %d terminals.' % (irf_impulses.shape[2], len(self.terminal_names))
                    terminal_ix = names2ix(self.terminal_names, terminal_names)
                    irf_impulses = tf.gather(irf_impulses, terminal_ix, axis=2)
                
                self.irf_impulses = irf_impulses

    def _compile_X_weighted_by_irf(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.X_weighted_by_irf = {}
                for i, response in enumerate(self.response_names):
                    self.X_weighted_by_irf[response] = {}
                    irf_weights = []
                    terminal_names = []
                    nparam = self.get_response_nparam(response)
                    for name in self.parametric_irf_terminal_names:
                        terminal_names.append(name)
                        t = self.node_table[name]
                        if type(t.impulse).__name__ == 'NNImpulse':
                            impulse_names = [x.name() for x in t.impulse.impulses()]
                        else:
                            impulse_names = self.terminal2impulse[name]
                        impulse_ix = names2ix(impulse_names, self.impulse_names)

                        if t.p.family == 'DiracDelta':
                            if self.use_distributional_regression:
                                _nparam = nparam
                            else:
                                _nparam = 1
                            ndim = self.get_response_ndim(response)
                            irf_seq = tf.gather(self.dirac_delta_mask, impulse_ix, axis=2)
                            irf_seq = irf_seq[..., None, None]
                            irf_seq = tf.tile(irf_seq, [1, 1, 1, _nparam, ndim])
                        else:
                            t_delta = self.t_delta[:, :, impulse_ix[0]]

                            irf = self.irf[response][name]
                            if len(irf) > 1:
                                irf = self._compose_irf(irf)
                            else:
                                irf = irf[0]

                            # Put batch dim last
                            t_delta = tf.transpose(t_delta, [1, 0])
                            # Add broadcasting for response nparam, ndim
                            t_delta = t_delta[..., None, None]
                            # Apply IRF
                            irf_seq = irf(t_delta)
                            # Put batch dim first
                            irf_seq = tf.transpose(irf_seq, [1, 0, 2, 3])
                            # Add terminal dim
                            irf_seq = tf.expand_dims(irf_seq, axis=2)
                        if not self.use_distributional_regression:
                            irf_seq = tf.pad(
                                irf_seq,
                                paddings=[
                                    (0, 0),
                                    (0, 0),
                                    (0, 0),
                                    (0, nparam - 1),
                                    (0, 0)
                                ]
                            )

                        irf_weights.append(irf_seq)

                    for nn_id in self.nn_irf_ids:
                        if self.nn_irf_terminal_names[nn_id]:
                            if response in self.nn_irf:
                                irf_seq = self.nn_irf[response][nn_id]
                                response_params = self.get_nn_meta('response_params', nn_id)
                                if response_params:
                                    dist_name = self.get_response_dist_name(response)
                                    if dist_name in response_params:
                                        response_params = response_params[dist_name]
                                    else:
                                        response_params = response_params[None]
                                    all_param_names = self.get_response_params(response)
                                    param_names = [x for x in all_param_names if x in response_params]
                                    param_ix = names2ix(param_names, all_param_names)
                                    if len(param_names) < len(all_param_names):
                                        seq_shape = tf.shape(irf_seq)
                                        new_shape = [
                                            seq_shape[0],
                                            seq_shape[1],
                                            seq_shape[2],
                                            len(all_param_names),
                                            seq_shape[4]
                                        ]
                                        irf_seq = self._scatter_along_axis(
                                            param_ix,
                                            irf_seq,
                                            new_shape,
                                            axis=3
                                        )
                                elif not self.use_distributional_regression:
                                    irf_seq = tf.pad(
                                        irf_seq,
                                        paddings=[
                                            (0, 0),
                                            (0, 0),
                                            (0, 0),
                                            (0, nparam - 1),
                                            (0, 0)
                                        ]
                                    )

                                irf_weights.append(irf_seq)
                                terminal_names += self.nn_irf_terminal_names[nn_id]

                    if len(irf_weights):
                        if len(irf_weights) == 1:
                            irf_weights = irf_weights[0]
                        else:
                            max_len = tf.reduce_max([tf.shape(x)[1] for x in irf_weights])  # Get maximum timesteps
                            irf_weights = [
                                tf.pad(x, ((0, 0), (max_len - tf.shape(x)[1], 0), (0, 0), (0, 0), (0, 0))) for x in irf_weights
                            ]
                            irf_weights = tf.concat(irf_weights, axis=2)
                    else:
                        irf_weights = None

                    if irf_weights is not None:
                        terminal_ix = names2ix(self.terminal_names, terminal_names)
                        irf_weights = tf.gather(irf_weights, terminal_ix, axis=2)
                        X_weighted_by_irf = self.irf_impulses * irf_weights
                    else:
                        X_weighted_by_irf = tf.zeros_like(self.X)[..., None, None] # Expand param and param_dim dimensions

                    X_weighted_unscaled = X_weighted_by_irf
                    X_weighted_unscaled_sumT = tf.reduce_sum(X_weighted_by_irf, axis=1, keepdims=True)
                    X_weighted_unscaled_sumK = tf.reduce_sum(X_weighted_by_irf, axis=2, keepdims=True)
                    X_weighted_unscaled_sumTK = tf.reduce_sum(X_weighted_unscaled_sumT, axis=1, keepdims=True)
                    self.X_weighted_unscaled[response] = X_weighted_unscaled
                    self.X_weighted_unscaled_sumT[response] = X_weighted_unscaled_sumT
                    self.X_weighted_unscaled_sumK[response] = X_weighted_unscaled_sumK
                    self.X_weighted_unscaled_sumTK[response] = X_weighted_unscaled_sumTK

                    coef_names = [self.node_table[x].coef_id() for x in self.terminal_names]
                    coef_ix = names2ix(coef_names, self.coef_names)
                    coef = tf.gather(self.coefficient[response], coef_ix, axis=1)
                    coef = tf.expand_dims(coef, axis=1)
                    X_weighted = X_weighted_unscaled
                    X_weighted = X_weighted * coef
                    X_weighted_sumT = tf.reduce_sum(X_weighted, axis=1, keepdims=True)
                    X_weighted_sumK = tf.reduce_sum(X_weighted, axis=2, keepdims=True)
                    X_weighted_sumTK = tf.reduce_sum(X_weighted_sumT, axis=2, keepdims=True)
                    self.X_weighted[response] = X_weighted
                    self.X_weighted_sumT[response] = X_weighted_sumT
                    self.X_weighted_sumK[response] = X_weighted_sumK
                    self.X_weighted_sumTK[response] = X_weighted_sumTK

    def _initialize_response_distribution(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.output_base = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of baseline values at response distribution parameter of the response
                self.output_delta = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of stimulus-driven offsets at response distribution parameter of the response (summing over predictors and time)
                self.output_delta_w_interactions = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of stimulus-driven offsets (including interactions) at response distribution parameter of the response (summing over predictors and time)
                self.interaction_delta = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of interaction-driven offsets at response distribution parameter of the response (summing over predictors and time)
                self.output = {}  # Key order: <response>; Value: nbatch x nparam x ndim tensor of predictoins at response distribution parameter of the response (summing over predictors and time)
                self.response_distribution = {}
                self.has_analytical_mean = {}
                self.response_distribution_delta = {} # IRF-driven changes in each parameter of the response distribution
                self.response_distribution_delta_w_interactions = {} # IRF-driven changes in each parameter of the response distribution
                self.prediction = {}
                self.prediction_over_time = {}
                self.ll_by_var = {}
                self.error_distribution = {}
                self.error_distribution_theoretical_quantiles = {}
                self.error_distribution_theoretical_cdf = {}
                self.error_distribution_plot = {}
                self.error_distribution_plot_lb = {}
                self.error_distribution_plot_ub = {}
                self.response_params_ema = {}
                self.X_conv_ema = {}
                self.X_conv_ema_debiased = {}
                self.z_bijectors = {}
                self.s_bijectors = {}

                for i, response in enumerate(self.response_names):
                    self.response_distribution[response] = {}
                    self.response_distribution_delta[response] = {}
                    self.response_distribution_delta_w_interactions[response] = {}
                    self.ll_by_var[response] = {}
                    self.error_distribution[response] = {}
                    self.error_distribution_theoretical_quantiles[response] = {}
                    self.error_distribution_theoretical_cdf[response] = {}
                    ndim = self.get_response_ndim(response)

                    if self.is_real(response):
                        response_means = tf.convert_to_tensor(
                            np.squeeze(self.Y_train_means[response]),
                            dtype=self.FLOAT_TF
                        )
                        response_sds = tf.convert_to_tensor(
                            np.squeeze(self.Y_train_sds[response]),
                            dtype=self.FLOAT_TF
                        )
                        z_bijector = AffineScalar(
                            shift=response_means,
                            scale=response_sds,
                            name='z_bijector_%s' % sn(response)
                        )
                        self.z_bijectors[response] = z_bijector
                        s_bijector = AffineScalar(
                            shift=tf.zeros_like(response_means),
                            scale=response_sds,
                            name='s_bijector_%s' % sn(response)
                        )
                        self.s_bijectors[response] = s_bijector

                    response_param_names = self.get_response_params(response)
                    response_params = self.intercept[response] # (batch, param, dim)
                    
                    response_dist_kwargs = {}
                    if self.get_response_dist_name(response) == 'lognormalv2':
                        response_dist_kwargs['epsilon'] = self.response_dist_epsilon

                    # Base output deltas
                    X_weighted = self.X_weighted[response] # (batch, time, impulse, param, dim)
                    X_weighted_sumT = self.X_weighted_sumT[response] # (batch, 1, impulse, param, dim)
                    X_weighted_sumK = self.X_weighted_sumK[response] # (batch, time, 1, param, dim)
                    X_weighted_sumTK = self.X_weighted_sumTK[response] # (batch, 1, 1, param, dim)
                    nparam = int(response_params.shape[-2])

                    # Interactions
                    if len(self.interaction_names):
                        interaction_delta = self._apply_interactions(response)
                    else:
                        interaction_delta = None

                    # Prediction targets
                    Y = self.Y[..., i]
                    Y_mask = self.Y_mask[..., i]

                    # Expand T and K dimensions of intercept
                    response_params = tf.expand_dims(tf.expand_dims(response_params, axis=1), axis=1)

                    # Get output deltas
                    output_delta = tf.cond(
                        self.sum_outputs_along_K,
                        lambda: tf.cond(
                            self.sum_outputs_along_T,
                            lambda: X_weighted_sumTK,
                            lambda: X_weighted_sumK
                        ),
                        lambda: tf.cond(
                            self.sum_outputs_along_T,
                            lambda: X_weighted_sumT,
                            lambda: X_weighted
                        )
                    )

                    # Conditionally tile Y along T and K
                    time_tile_shape = [1, self.history_length + self.future_length, 1]
                    n_impulse = self.n_impulse
                    impulse_tile_shape = [1, 1, n_impulse]
                    Y = tf.expand_dims(tf.expand_dims(Y, axis=1), axis=1)
                    Y_mask = tf.expand_dims(tf.expand_dims(Y_mask, axis=1), axis=1)
                    Y = tf.cond(
                        self.sum_outputs_along_T,
                        lambda: Y,
                        lambda: tf.tile(Y, time_tile_shape)
                    )
                    Y_mask = tf.cond(
                        self.sum_outputs_along_T,
                        lambda: Y_mask,
                        lambda: tf.tile(Y_mask, time_tile_shape)
                    )
                    Y = tf.cond(
                        self.sum_outputs_along_K,
                        lambda: Y,
                        lambda: tf.tile(Y, impulse_tile_shape)
                    )
                    Y_mask = tf.cond(
                        self.sum_outputs_along_K,
                        lambda: Y_mask,
                        lambda: tf.tile(Y_mask, impulse_tile_shape)
                    )

                    output_delta_w_interactions = output_delta
                    if interaction_delta is not None:
                        output_delta_w_interactions += interaction_delta

                    output_base = response_params
                    response_params += output_delta_w_interactions

                    self.output_base[response] = output_base
                    self.output_delta[response] = output_delta
                    self.output_delta_w_interactions[response] = output_delta_w_interactions
                    self.interaction_delta[response] = interaction_delta
                    self.output[response] = response_params

                    for j, response_param_name in enumerate(response_param_names):
                        dim_names = self.expand_param_name(response, response_param_name)
                        for k, dim_name in enumerate(dim_names):
                            if self.use_distributional_regression or j == 0:
                                self.response_distribution_delta[response][dim_name] = output_delta[:, 0, 0, j, k]
                                self.response_distribution_delta_w_interactions[response][dim_name] = output_delta_w_interactions[:, 0, 0, j, k]
                            else:
                                self.response_distribution_delta[response][dim_name] = tf.zeros_like(
                                    output_delta[:, 0, 0, 0, k]
                                )
                                self.response_distribution_delta_w_interactions[response][dim_name] = tf.zeros_like(
                                    output_delta_w_interactions[:, 0, 0, 0, k]
                                )

                    response_dist, response_dist_src, response_params = self._initialize_response_distribution_inner(
                        response,
                        param_type='output'
                    )
                    pred_mean = response_dist.mean()
                    self.response_distribution[response] = response_dist
                    self.has_analytical_mean[response] = response_dist.has_analytical_mean()
                    if not self.is_categorical(response):
                        base_response_dist, _, _ = self._initialize_response_distribution_inner(
                            response,
                            param_type='output_base'
                        )
                        base_response_mean = base_response_dist.mean()
                        delta_response_dist, _, _ = self._initialize_response_distribution_inner(
                            response,
                            param_type='output_delta'
                        )
                        delta_response_mean = delta_response_dist.mean()
                        self.response_distribution_delta[response]['mean'] = (delta_response_mean - base_response_mean)[:, 0, 0]
                        self.response_distribution_delta_w_interactions[response]['mean'] = (pred_mean - base_response_mean)[:, 0, 0]
                        if self.get_response_dist_name(response).startswith('lognormal'):
                            bijector = self.s_bijectors[response]
                        else:
                            bijector = self.z_bijectors[response]
                    else:
                        bijector = None

                    # Define prediction tensors
                    dist_name = self.get_response_dist_name(response)

                    def MAP_predict(
                            self=self,
                            response_dist=response_dist_src,
                            bijector=bijector
                    ):
                        mode = response_dist.mode()
                        if self.is_real(response) and bijector is not None:
                            mode = bijector.forward(mode)

                        return mode

                    prediction = tf.cond(self.use_MAP_mode, MAP_predict, response_dist.sample)

                    if dist_name in ['bernoulli', 'categorical']:
                        prediction = tf.cast(prediction, self.INT_TF) * \
                                                    tf.cast(Y_mask, self.INT_TF)
                    else: # Treat as continuous regression, use the first (location) parameter
                        prediction = prediction * Y_mask
                    prediction = tf.cond(
                        self.sum_outputs_along_T,
                        lambda: tf.cond(
                            self.sum_outputs_along_K,
                            lambda: prediction[:, :, 0],
                            lambda: prediction
                        )[:, 0],
                        lambda: tf.cond(
                            self.sum_outputs_along_K,
                            lambda: prediction[:, :, 0],
                            lambda: prediction
                        )
                    )
                    self.prediction[response] = prediction

                    # Get elementwise log likelihood
                    if self.is_real(response):
                        # Ensure a safe value for Y
                        sel = tf.cast(Y_mask, tf.bool)
                        Y_safe = tf.ones_like(Y) * self.Y_train_means[response]
                        _Y = tf.where(sel, Y, Y_safe)
                    else:
                        _Y = Y
                    ll = response_dist.log_prob(_Y)

                    # Mask out likelihoods of predictions for missing response variables.
                    ll = ll * Y_mask
                    ll = tf.cond(
                        self.sum_outputs_along_T,
                        lambda: tf.cond(
                            self.sum_outputs_along_K,
                            lambda: ll[:, :, 0],
                            lambda: ll
                        )[:, 0],
                        lambda: tf.cond(
                            self.sum_outputs_along_K,
                            lambda: ll[:, :, 0],
                            lambda: ll
                        )
                    )
                    self.ll_by_var[response] = ll

                    # Define EMA over response distribution
                    beta = self.ema_decay
                    step = tf.cast(self.global_batch_step, self.FLOAT_TF)
                    response_params_ema_cur = []
                    # These will only ever be used in training mode, so un-standardize if needed
                    for j , response_param_name in enumerate(response_param_names):
                        response_params_ema_cur.append(response_params[j][:, 0, 0]) # Index into spurious T and K dimensions
                    response_params_ema_cur = tf.stack(response_params_ema_cur, axis=1)
                    response_params_ema_cur = tf.reduce_mean(response_params_ema_cur, axis=0)
                    self.response_params_ema[response] = tf.Variable(
                        tf.zeros((nparam, ndim)),
                        trainable=False,
                        name='response_params_ema_%s' % sn(response)
                    )
                    response_params_ema_prev = self.response_params_ema[response]
                    response_params_ema_debiased = response_params_ema_prev / (1. - beta ** step)
                    ema_update = beta * response_params_ema_prev + \
                                 (1. - beta) * response_params_ema_cur
                    response_params_ema_op = tf.assign(
                        self.response_params_ema[response],
                        ema_update
                    )
                    self.ema_ops.append(response_params_ema_op)
                    for j, response_param_name in enumerate(response_param_names):
                        dim_names = self.expand_param_name(response, response_param_name)
                        for k, dim_name in enumerate(dim_names):
                            tf.summary.scalar(
                                'ema' + '/%s/%s_%s' % (
                                    sn(response_param_name),
                                    sn(response),
                                    sn(dim_name)
                                ),
                                response_params_ema_debiased[j, k],
                                collections=['params']
                            )

                    # Define error distribution
                    if self.is_real(response):
                        empirical_quantiles = tf.linspace(0., 1., self.n_errors[response])
                        err_dist_params = [response_params_ema_debiased[j] for j in range(len(response_param_names))]
                        if self.get_response_ndim(response) == 1:
                            err_dist_params = [tf.squeeze(x, axis=-1) for x in err_dist_params]
                        err_dist_kwargs = {x: y for x, y in zip(self.get_response_params_tf(response), err_dist_params)}
                        err_dist_kwargs.update(**response_dist_kwargs)
                        response_dist_fn = self.get_response_dist(response)
                        err_dist = response_dist_fn(**err_dist_kwargs)
                        # Rescale
                        err_dist = ShiftedScaledDistribution(
                            err_dist,
                            tf.zeros_like(response_means),
                            response_sds
                        )
                        # Shift to 0
                        err_dist = ZeroMeanDistribution(
                            err_dist
                        )
                        err_dist_theoretical_cdf = err_dist.cdf(self.errors[response])
                        err_dist_theoretical_quantiles = err_dist.quantile(empirical_quantiles)
                        err_dist_lb = err_dist.quantile(.025)
                        err_dist_ub = err_dist.quantile(.975)
                        self.error_distribution_theoretical_quantiles[response] = err_dist_theoretical_quantiles

                        err_dist_plot = tf.exp(err_dist.log_prob(self.support))

                        self.error_distribution[response] = err_dist
                        self.error_distribution_theoretical_cdf[response] = err_dist_theoretical_cdf
                        self.error_distribution_plot[response] = err_dist_plot
                        self.error_distribution_plot_lb[response] = err_dist_lb
                        self.error_distribution_plot_ub[response] = err_dist_ub

                self.ll = tf.add_n([self.ll_by_var[x] for x in self.ll_by_var])

    def _initialize_response_distribution_inner(self, response, param_type='output'):
        with self.session.as_default():
            with self.session.graph.as_default():
                response_param_names = self.get_response_params(response)
                response_dist_fn = self.get_response_dist(response)
                ndim = self.get_response_ndim(response)
                response_dist_kwargs = {}
                if self.get_response_dist_name(response) == 'lognormalv2':
                    response_dist_kwargs['epsilon'] = self.response_dist_epsilon

                if param_type == 'output_base':
                    response_params = self.output_base[response]
                elif param_type == 'output_delta':
                    response_params = self.output_base[response] + self.output_delta[response]
                elif param_type == 'output_delta_w_interactions':
                    response_params = self.output_base[response] + self.output_delta_w_interactions[response]
                elif param_type == 'output':
                    response_params = self.output[response]
                else:
                    raise ValueError('Unrecognized param_type %s' % param_type)

                nparam = self.get_response_nparam(response)

                response_params = [response_params[..., j, :] for j in range(nparam)]

                # Post process response params
                for j, response_param_name in enumerate(response_param_names):
                    _response_param = response_params[j]
                    if self.is_real(response) and (response_param_name in ['sigma', 'tailweight', 'beta'] or
                            (response_param_name == 'mu' and self.get_response_dist_name(response) == 'lognormalv2')):
                        _response_param = self.constraint_fn(_response_param) + self.response_dist_epsilon
                    response_params[j] = _response_param

                # Define response distribution
                # Squeeze params if needed
                if ndim == 1:
                    _response_params = [tf.squeeze(x, axis=-1) for x in response_params]
                else:
                    _response_params = response_params
                _response_dist_kwargs = {x: y for x, y in zip(self.get_response_params_tf(response), _response_params)}
                _response_dist_kwargs.update(response_dist_kwargs)
                response_dist = response_dist_fn(**_response_dist_kwargs)
                response_dist_src = response_dist

                if self.is_real(response):
                    scale = tf.constant(np.squeeze(self.Y_train_sds[response]), self.FLOAT_TF)
                    if self.get_response_dist_name(response).startswith('lognormal'):
                        shift = tf.zeros_like(scale)
                    else:
                        shift = tf.constant(np.squeeze(self.Y_train_means[response]), self.FLOAT_TF)
                    response_dist = ShiftedScaledDistribution(
                        response_dist,
                        shift,
                        scale
                    )

                return response_dist, response_dist_src, response_params

    def _initialize_regularizer(self, regularizer_name, regularizer_scale, per_item=False):
        with self.session.as_default():
            with self.session.graph.as_default():
                if regularizer_name is None:
                    regularizer = None
                elif regularizer_name == 'inherit':
                    regularizer = self.regularizer
                else:
                    scale = regularizer_scale
                    if isinstance(scale, str):
                        scale = [float(x) for x in scale.split(';')]
                    else:
                        scale = [scale]
                    if self.scale_regularizer_with_data:
                        if per_item:
                            scale = [x * self.minibatch_scale for x in scale]
                        else:
                            scale = [x * self.minibatch_size * self.minibatch_scale for x in scale]
                    elif per_item:
                        scale = [x / self.minibatch_size for x in scale]

                    regularizer = get_regularizer(
                        regularizer_name,
                        scale=scale,
                        regularize_mean=self.regularize_mean,
                        session=self.session
                    )

                return regularizer

    def _initialize_optimizer(self):
        name = self.optim_name.lower()
        use_jtps = self.use_jtps
        safe_mode = self.use_safe_optimizer

        with self.session.as_default():
            with self.session.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.global_batch_step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.global_batch_step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        schedule = getattr(tf.keras.optimizers.schedules, self.lr_decay_family)(
                            lr,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                        self.lr = schedule(t)
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_gradient
                clip_global = self.max_global_gradient_norm

                optimizer_args = [self.lr]
                optimizer_kwargs = {}
                if name == 'momentum':
                    optimizer_args += [0.9]
                if name.endswith('fast'):
                    optimizer_kwargs['beta2'] = 0.9
                if name in ('adadelta', 'adam', 'nadam'):
                    optimizer_kwargs['epsilon'] = self.optim_epsilon
                    if name == 'adadelta':
                        optimizer_kwargs['rho'] = 0.999

                optimizer_class = {
                    'sgd': tf.train.GradientDescentOptimizer,
                    'momentum': tf.train.MomentumOptimizer,
                    'adagrad': tf.train.AdagradOptimizer,
                    'adadelta': tf.train.AdadeltaOptimizer,
                    'ftrl': tf.train.FtrlOptimizer,
                    'rmsprop': tf.train.RMSPropOptimizer,
                    'adam': tf.train.AdamOptimizer,
                    'adamfast': tf.train.AdamOptimizer,
                    'nadam': NadamOptimizer,
                    'nadamfast': NadamOptimizer,
                    'amsgrad': AMSGradOptimizer
                }[name]

                if clip or clip_global:
                    optimizer_class = get_clipped_optimizer_class(optimizer_class, session=self.session)
                    optimizer_kwargs['max_grad'] = clip
                    optimizer_kwargs['max_global_norm'] = clip_global

                if use_jtps:
                    optimizer_class = get_JTPS_optimizer_class(optimizer_class, session=self.session)
                    optimizer_kwargs['meta_learning_rate'] = 1

                if safe_mode:
                    optimizer_class = get_safe_optimizer_class(optimizer_class, session=self.session)

                optim = optimizer_class(*optimizer_args, **optimizer_kwargs)

                return optim

    def _initialize_objective(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                loss_func = -self.ll

                # Average over number of dependent variables for stability
                loss_func /= self.n_response

                # Filter
                if self.loss_cutoff_n_sds:
                    assert self.ema_decay, '``ema_decay`` must be provided if ``loss_cutoff_n_sds`` is used'
                    beta = self.ema_decay
                    ema_warm_up = int(1/(1 - self.ema_decay))
                    n_sds = self.loss_cutoff_n_sds
                    step = tf.cast(self.global_batch_step, self.FLOAT_TF)

                    self.loss_m1_ema = tf.Variable(0., trainable=False, name='loss_m1_ema')
                    self.loss_m2_ema = tf.Variable(0., trainable=False, name='loss_m2_ema')
                    self.n_dropped_ema = tf.Variable(0., trainable=False, name='n_dropped_ema')

                    # Debias
                    loss_m1_ema = self.loss_m1_ema / (1. - beta ** step)
                    loss_m2_ema = self.loss_m2_ema / (1. - beta ** step)

                    sd = tf.sqrt(loss_m2_ema - loss_m1_ema**2)
                    loss_cutoff = loss_m1_ema + n_sds * sd
                    loss_func_filter = tf.cast(loss_func < loss_cutoff, dtype=self.FLOAT_TF)
                    loss_func_filtered = loss_func * loss_func_filter
                    n_batch = tf.cast(tf.shape(loss_func)[0], dtype=self.FLOAT_TF)
                    n_retained = tf.reduce_sum(loss_func_filter)

                    loss_func, n_retained = tf.cond(
                        self.global_batch_step > ema_warm_up,
                        lambda loss_func_filtered=loss_func_filtered, n_retained=n_retained: (loss_func_filtered, n_retained),
                        lambda loss_func=loss_func: (loss_func, n_batch),
                    )

                    self.n_dropped = n_batch - n_retained

                    denom = n_retained + self.epsilon
                    loss_m1_cur = tf.reduce_sum(loss_func) / denom
                    loss_m2_cur = tf.reduce_sum(loss_func**2) / denom

                    loss_m1_ema_update = beta * self.loss_m1_ema + (1 - beta) * loss_m1_cur
                    loss_m2_ema_update = beta * self.loss_m2_ema + (1 - beta) * loss_m2_cur
                    n_dropped_ema_update = beta * self.n_dropped_ema + (1 - beta) * self.n_dropped

                    loss_m1_ema_op = tf.assign(self.loss_m1_ema, loss_m1_ema_update)
                    loss_m2_ema_op = tf.assign(self.loss_m2_ema, loss_m2_ema_update)
                    n_dropped_ema_op = tf.assign(self.n_dropped_ema, n_dropped_ema_update)

                    self.ema_ops += [loss_m1_ema_op, loss_m2_ema_op, n_dropped_ema_op]

                loss_func = tf.reduce_sum(loss_func)

                # Rescale
                if self.scale_loss_with_data:
                    loss_func = loss_func * self.minibatch_scale

                # Regularize
                for nn_id in self.regularizable_layers:
                    for l in self.regularizable_layers[nn_id]:
                        if hasattr(l, 'regularizable_weights'):
                            vars = l.regularizable_weights
                        else:
                            vars = [l]
                        for v in vars:
                            is_ranef = False
                            for gf in self.rangf:
                                if '_by_%s' % sn(gf) in v.name:
                                    is_ranef = True
                                    break
                            if is_ranef:
                                if 'nn' not in self.rvs or not self.declare_priors_ranef:
                                    self._regularize(
                                        v,
                                        regtype='ranef',
                                        var_name=reg_name(v.name)
                                    )
                            elif 'bias' not in v.name:
                                if 'ff_l' in v.name:
                                    regtype = 'ff'
                                elif 'rnn_projection_l' in v.name:
                                    regtype = 'rnn_projection'
                                else:
                                    regtype = 'nn'
                                self._regularize(v, regtype=regtype, var_name=reg_name(v.name), nn_id=nn_id)

                reg_loss = tf.constant(0., dtype=self.FLOAT_TF)
                if len(self.regularizer_losses_varnames) > 0:
                    reg_loss += tf.add_n(self.regularizer_losses)
                    loss_func += reg_loss

                kl_loss = tf.constant(0., dtype=self.FLOAT_TF)
                if self.is_bayesian and len(self.kl_penalties):
                    for layer in self.layers:
                        self.kl_penalties.update(layer.kl_penalties())
                    kl_loss += tf.reduce_sum([tf.reduce_sum(self.kl_penalties[k]['val']) for k in self.kl_penalties])
                    loss_func += kl_loss

                self.loss_func = loss_func
                self.reg_loss = reg_loss
                self.kl_loss = kl_loss

                self.optim = self._initialize_optimizer()
                assert self.optim_name is not None, 'An optimizer name must be supplied'

                self.train_op = self.optim.minimize(self.loss_func, var_list=tf.trainable_variables())

    def _initialize_logging(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                tf.summary.scalar('opt/loss_by_iter', self.loss_total, collections=['opt'])
                tf.summary.scalar('opt/reg_loss_by_iter', self.reg_loss_total, collections=['opt'])
                if self.is_bayesian:
                    tf.summary.scalar('opt/kl_loss_by_iter', self.kl_loss_total, collections=['opt'])
                if self.filter_outlier_losses and self.loss_cutoff_n_sds:
                    tf.summary.scalar('opt/n_dropped', self.n_dropped_in, collections=['opt'])
                if self.eval_freq > 0:
                    for metric in self.dev_metrics:
                        if metric == 'full_log_lik':
                            tf.summary.scalar(
                                'dev/%s' % metric, self.dev_metrics[metric], collections=['dev']
                            )
                        else:
                            for response in self.dev_metrics[metric]:
                                name = '%s_%s' % (metric, response)
                                for ix in self.dev_metrics[metric][response]:
                                    if len(self.dev_metrics[metric][response]) > 1:
                                        name += '_file%d' % ix
                                    tf.summary.scalar(
                                        'dev/%s' % sn(name), self.dev_metrics[metric][response][ix], collections=['dev']
                                    )
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr', self.session.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr')
                self.summary_opt = tf.summary.merge_all(key='opt')
                if self.eval_freq > 0:
                    self.summary_dev = tf.summary.merge_all(key='dev')
                self.summary_params = tf.summary.merge_all(key='params')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def _initialize_saver(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf_check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.ema_vars = tf.get_collection('trainable_variables')
                self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay if self.ema_decay else 0.)
                ema_op = self.ema.apply(self.ema_vars)
                self.ema_ops.append(ema_op)
                self.ema_map = {}
                for v in self.ema_vars:
                    self.ema_map[self.ema.average_name(v)] = v
                for v in tf.get_collection('batch_norm'):
                    name = ':'.join(v.name.split(':')[:-1])
                    self.ema_map[name] = v
                self.ema_saver = tf.train.Saver(self.ema_map)

    def _initialize_convergence_checking(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.check_convergence:
                    self.rho_t = tf.placeholder(self.FLOAT_TF, name='rho_t_in')
                    self.p_rho_t = tf.placeholder(self.FLOAT_TF, name='p_rho_t_in')
                    tf.summary.scalar('convergence/rho_t', self.rho_t, collections=['convergence'])
                    tf.summary.scalar('convergence/p_rho_t', self.p_rho_t, collections=['convergence'])
                    tf.summary.scalar('convergence/proportion_converged', self.proportion_converged, collections=['convergence'])
                    self.summary_convergence = tf.summary.merge_all(key='convergence')




    ######################################################
    #
    #  Utility methods
    #
    ######################################################

    def _vector_is_indicator(self, a):
        vals = set(np.unique(a))
        if len(vals) != 2:
            return False
        return vals in (
            {0,1},
            {'0','1'},
            {True, False},
            {'True', 'False'},
            {'TRUE', 'FALSE'},
            {'true', 'false'},
            {'T', 'F'},
            {'t', 'f'},
        )

    def _tril_diag_ix(self, n):
        return (np.arange(1, n + 1).cumsum() - 1).astype(self.INT_NP)

    def _scatter_along_axis(self, axis_indices, updates, shape, axis=0):
        # Except for axis, updates and shape must be identically shaped
        with self.session.as_default():
            with self.session.graph.as_default():
                if axis != 0:
                    transpose_axes = [axis] + list(range(axis)) + list(range(axis + 1, len(updates.shape)))
                    inverse_transpose_axes = list(range(1, axis + 1)) + [0] + list(range(axis + 1, len(updates.shape)))
                    updates_transposed = tf.transpose(updates, transpose_axes)
                    scatter_shape = [shape[axis]] + shape[:axis] + shape[axis + 1:]
                else:
                    updates_transposed = updates
                    scatter_shape = shape

                out = tf.scatter_nd(
                    tf.expand_dims(axis_indices, -1),
                    updates_transposed,
                    scatter_shape
                )

                if axis != 0:
                    out = tf.transpose(out, inverse_transpose_axes)

                return out

    def _softplus_sigmoid(self, x, a=-1., b=1.):
        with self.session.as_default():
            with self.session.graph.as_default():
                f = tf.nn.softplus
                c = b - a

                g = (-f(-f(x - a) + c) + f(c)) * c / f(c) + a
                return g

    def _softplus_sigmoid_inverse(self, x, a=-1., b=1.):
        with self.session.as_default():
            with self.session.graph.as_default():
                f = tf.nn.softplus
                ln = tf.log
                exp = tf.exp
                c = b - a

                g = ln(exp(c) / ( (exp(c) + 1) * exp( -f(c) * (x - a) / c ) - 1) - 1) + a
                return g

    def _sigmoid(self, x, lb=0., ub=1.):
        with self.session.as_default():
            with self.session.graph.as_default():
                return tf.sigmoid(x) * (ub - lb) + lb

    def _sigmoid_np(self, x, lb=0., ub=1.):
        return (1. / (1. + np.exp(-x))) * (ub - lb) + lb

    def _logit(self, x, lb=0., ub=1.):
        with self.session.as_default():
            with self.session.graph.as_default():
                x = (x - lb) / (ub - lb)
                x = x * (1 - 2 * self.epsilon) + self.epsilon
                return tf.log(x / (1 - x))

    def _logit_np(self, x, lb=0., ub=1.):
        with self.session.as_default():
            with self.session.graph.as_default():
                x = (x - lb) / (ub - lb)
                x = x * (1 - 2 * self.epsilon) + self.epsilon
                return np.log(x / (1 - x))

    def _piecewise_linear_interpolant(self, c, v):
        # c: knot locations, shape=[B, Q, K], B = batch, Q = query points or 1, K = n knots
        # v: knot values, shape identical to c
        with self.session.as_default():
            with self.session.graph.as_default():
                if len(c.shape) == 1:
                    # No batch or query dim
                    c = c[None, None, ...]
                elif len(c.shape) == 2:
                    # No query dim
                    c = tf.expand_dims(c, axis=-2)
                elif len(c.shape) > 3:
                    # Too many dims
                    raise ValueError(
                        'Rank of knot location tensor c to piecewise resampler must be >= 1 and <= 3. Saw "%d"' % len(
                            c.shape))
                if len(v.shape) == 1:
                    # No batch or query dim
                    v = v[None, None, ...]
                elif len(v.shape) == 2:
                    # No query dim
                    v = tf.expand_dims(v, axis=-2)
                elif len(v.shape) > 3:
                    # Too many dims
                    raise ValueError(
                        'Rank of knot amplitude tensor c to piecewise resampler must be >= 1 and <= 3. Saw "%d"' % len(
                            v.shape))

                c_t = c[..., 1:]
                c_tm1 = c[..., :-1]
                y_t = v[..., 1:]
                y_tm1 = v[..., :-1]

                # Compute intercepts a_ and slopes b_ of line segments
                a_ = (y_t - y_tm1) / (c_t - c_tm1)
                valid = c_t > c_tm1
                a_ = tf.where(valid, a_, tf.zeros_like(a_))
                b_ = y_t - a_ * c_t

                # Handle points beyond final knot location (0 response)
                a_ = tf.concat([a_, tf.zeros_like(a_[..., -1:])], axis=-1)
                b_ = tf.concat([b_, tf.zeros_like(b_[..., -1:])], axis=-1)
                c_ = tf.concat([c, tf.ones_like(c[..., -1:]) * np.inf], axis=-1)

                def make_piecewise(a, b, c):
                    def select_segment(x, c):
                        c_t = c[..., 1:]
                        c_tm1 = c[..., :-1]
                        select = tf.cast(tf.logical_and(x >= c_tm1, x < c_t), dtype=self.FLOAT_TF)
                        return select

                    def piecewise(x):
                        select = select_segment(x, c)
                        # a_select = tf.reduce_sum(a * select, axis=-1, keepdims=True)
                        # b_select = tf.reduce_sum(b * select, axis=-1, keepdims=True)
                        # response = a_select * x + b_select
                        response = tf.reduce_sum((a * x + b) * select, axis=-1, keepdims=True)
                        return response

                    return piecewise

                out = make_piecewise(a_, b_, c_)

                return out





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _new_irf(self, irf_lambda, params):
        irf = irf_lambda(params)
        def new_irf(x):
            return irf(x)
        return new_irf

    def _compose_irf(self, f_list):
        if not isinstance(f_list, list):
            f_list = [f_list]
        with self.session.as_default():
            with self.session.graph.as_default():
                f = f_list[0](self.interpolation_support)[..., 0]
                for g in f_list[1:]:
                    _f = tf.spectral.rfft(f)
                    _g = tf.spectral.rfft(g(self.interpolation_support)[..., 0])
                    f = tf.spectral.irfft(
                        _f * _g
                    ) * self.max_tdelta_batch / tf.cast(self.n_interp, dtype=self.FLOAT_TF)

                def make_composed_irf(seq):
                    def composed_irf(t):
                        squeezed = 0
                        while t.shape[-1] == 1:
                            t = tf.squeeze(t, axis=-1)
                            squeezed += 1
                        ix = tf.cast(tf.round(t * tf.cast(self.n_interp - 1, self.FLOAT_TF) / self.max_tdelta_batch), dtype=self.INT_TF)
                        row_ix = tf.tile(tf.range(tf.shape(t)[0])[..., None], [1, tf.shape(t)[1]])
                        ix = tf.stack([row_ix, ix], axis=-1)
                        out = tf.gather_nd(seq, ix)

                        for _ in range(squeezed):
                            out = out[..., None]

                        return out

                    return composed_irf

                return make_composed_irf(f)

    def _get_mean_init_vector(self, irf_ids, param_name, irf_param_init, default=0.):
        mean = np.zeros(len(irf_ids))
        for i in range(len(irf_ids)):
            mean[i] = irf_param_init[irf_ids[i]].get(param_name, default)
        return mean

    def _process_mean(self, mean, lb=None, ub=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if lb is not None and ub is None:
                    # Lower-bounded support only
                    mean = self.constraint_fn_inv_np(mean - lb - self.epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    mean = self.constraint_fn_inv_np(-(mean - ub + self.epsilon))
                elif lb is not None and ub is not None:
                    # Finite-interval bounded support
                    mean = self._logit_np(mean, lb, ub)

        return mean, lb, ub

    def _get_trainable_untrainable_ix(self, param_name, ids, trainable=None):
        if trainable is None:
            trainable_ix = np.array(list(range(len(ids))), dtype=self.INT_NP)
            untrainable_ix = []
        else:
            trainable_ix = []
            untrainable_ix = []
            for i in range(len(ids)):
                name = ids[i]
                if param_name in trainable[name]:
                    trainable_ix.append(i)
                else:
                    untrainable_ix.append(i)
            trainable_ix = np.array(trainable_ix, dtype=self.INT_NP)
            untrainable_ix = np.array(untrainable_ix, dtype=self.INT_NP)

        return trainable_ix, untrainable_ix

    def _regularize(self, var, center=None, regtype=None, var_name=None, nn_id=None):
        assert regtype in [
            None, 'intercept', 'coefficient', 'irf', 'ranef', 'nn', 'ff', 'rnn_projection', 'activity', 'context',
            'unit_integral', 'conv_output']

        if regtype is None:
            regularizer = self.regularizer
        else:
            regularizer = getattr(self, '%s_regularizer' % regtype)
        if regtype in ['nn', 'ff', 'rnn_projection', 'activity', 'context']:
            regularizer = regularizer[nn_id]

        if regularizer is not None and str(var_name) not in self.regularizer_losses_varnames:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if center is None:
                        reg = regularizer(var)
                    else:
                        reg = regularizer(var - center)
                    self.regularizer_losses.append(reg)
                    self.regularizer_losses_varnames.append(str(var_name))
                    if regtype is None:
                        reg_name = self.regularizer_name
                        reg_scale = self.regularizer_scale
                        if self.scale_regularizer_with_data:
                            reg_scale *= self.minibatch_size * self.minibatch_scale
                    elif regtype in ['ff', 'rnn_projection'] and getattr(self, '%s_regularizer_name' % regtype) is None:
                        reg_name = self.get_nn_meta('nn_regularizer_name', nn_id)
                        reg_scale = self.get_nn_meta('nn_regularizer_scale', nn_id)
                    elif regtype == 'unit_integral':
                        reg_name = 'l1_regularizer'
                        reg_scale = getattr(self, '%s_regularizer_scale' % regtype)
                    else:
                        reg_name = getattr(self, '%s_regularizer_name' % regtype)
                        reg_scale = getattr(self, '%s_regularizer_scale' % regtype)
                    if reg_name == 'inherit':
                        reg_name = self.regularizer_name
                    if reg_scale == 'inherit':
                        reg_scale = self.regularizer_scale
                        if self.scale_regularizer_with_data:
                            reg_scale *= self.minibatch_size * self.minibatch_scale
                    self.regularizer_losses_names.append(reg_name)
                    self.regularizer_losses_scales.append(reg_scale)

        return var

    def _add_convergence_tracker(self, var, name, alpha=0.9):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.convergence_n_iterates:
                    # Flatten the variable for easy argmax
                    var = tf.reshape(var, [-1])
                    self.d0.append(var)

                    self.d0_names.append(name)

                    # Initialize tracker of parameter iterates
                    convergence_stride = self.convergence_stride
                    if self.early_stopping and self.eval_freq > 0:
                        convergence_stride *= self.eval_freq

                    var_d0_iterates = tf.Variable(
                        tf.zeros([int(self.convergence_n_iterates / convergence_stride)] + list(var.shape)),
                        name=name + '_d0',
                        trainable=False
                    )

                    var_d0_iterates_update = tf.placeholder(self.FLOAT_TF, shape=var_d0_iterates.shape)
                    self.d0_saved.append(var_d0_iterates)
                    self.d0_saved_update.append(var_d0_iterates_update)
                    self.d0_assign.append(tf.assign(var_d0_iterates, var_d0_iterates_update))

    def _compute_and_test_corr(self, iterates, twotailed=True):
        convergence_stride = self.convergence_stride
        if self.early_stopping and self.eval_freq > 0:
            convergence_stride *= self.eval_freq

        x = np.arange(0, len(iterates)*convergence_stride, convergence_stride).astype('float')[..., None]
        y = iterates

        n_iterates = int(self.convergence_n_iterates / convergence_stride)

        rt = corr(x, y)[0]
        tt = rt * np.sqrt((n_iterates - 2) / (1 - rt ** 2))
        if twotailed:
            p_tt = 1 - (scipy.stats.t.cdf(np.fabs(tt), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(tt), n_iterates - 2))
        else:
            p_tt = scipy.stats.t.cdf(tt, n_iterates - 2)
        p_tt = np.where(np.isfinite(p_tt), p_tt, np.zeros_like(p_tt))

        ra = corr(y[1:], y[:-1])[0]
        ta = ra * np.sqrt((n_iterates - 2) / (1 - ra ** 2))
        if twotailed:
            p_ta = 1 - (scipy.stats.t.cdf(np.fabs(ta), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(ta), n_iterates - 2))
        else:
            p_ta = scipy.stats.t.cdf(ta, n_iterates - 2)
        p_ta = np.where(np.isfinite(p_ta), p_ta, np.zeros_like(p_ta))

        return rt, p_tt, ra, p_ta

    def run_convergence_check(self, verbose=True, feed_dict=None):
        with self.session.as_default():
            with self.session.graph.as_default():
                if self.check_convergence:
                    min_p = 1.
                    min_p_ix = 0
                    rt_at_min_p = 0
                    ra_at_min_p = 0
                    p_ta_at_min_p = 0
                    fd_assign = {}

                    cur_step = self.global_step.eval(session=self.session)
                    last_check = self.last_convergence_check.eval(session=self.session)
                    convergence_stride = self.convergence_stride
                    if self.early_stopping and self.eval_freq > 0:
                        convergence_stride *= self.eval_freq
                    offset = cur_step % convergence_stride
                    update = last_check < cur_step and convergence_stride > 0
                    if update and feed_dict is None:
                        update = False
                        if verbose:
                            stderr('Skipping convergence history update because no feed_dict provided.\n')

                    push = update and offset == 0

                    if self.check_convergence:
                        if update:
                            var_d0, var_d0_iterates = self.session.run([self.d0, self.d0_saved], feed_dict=feed_dict)
                        else:
                            var_d0_iterates = self.session.run(self.d0_saved)

                        start_ix = int(self.convergence_n_iterates / convergence_stride) - int((cur_step - 1) / convergence_stride) - 1
                        start_ix = max(0, start_ix)
                        
                        twotailed = not (self.early_stopping and self.eval_freq > 0)

                        for i in range(len(var_d0_iterates)):
                            if update:
                                new_d0 = var_d0[i]
                                iterates_d0 = var_d0_iterates[i]
                                if push:
                                    iterates_d0[:-1] = iterates_d0[1:]
                                    iterates_d0[-1] = new_d0
                                else:
                                    new_d0 = (new_d0 + offset * iterates_d0[-1]) / (offset + 1)
                                    iterates_d0[-1] = new_d0
                                fd_assign[self.d0_saved_update[i]] = iterates_d0

                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(
                                    iterates_d0[start_ix:],
                                    twotailed=twotailed
                                )
                            else:
                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(
                                    var_d0_iterates[i][start_ix:],
                                    twotailed=twotailed
                                )

                            new_min_p_ix = p_tt.argmin()
                            new_min_p = p_tt[new_min_p_ix]
                            if new_min_p < min_p:
                                min_p = new_min_p
                                min_p_ix = i
                                rt_at_min_p = rt[new_min_p_ix]
                                ra_at_min_p = ra[new_min_p_ix]
                                p_ta_at_min_p = p_ta[new_min_p_ix]

                        if update:
                            fd_assign[self.last_convergence_check_update] = self.global_step.eval(session=self.session)
                            to_run = [self.d0_assign, self.last_convergence_check_assign]
                            self.session.run(to_run, feed_dict=fd_assign)

                    if push:
                        locally_converged = cur_step > self.convergence_n_iterates and \
                                    (min_p > self.convergence_alpha)
                        convergence_history = self.convergence_history.eval(session=self.session)
                        convergence_history[:-1] = convergence_history[1:]
                        convergence_history[-1] = locally_converged
                        self.session.run(self.convergence_history_assign, {self.convergence_history_update: convergence_history})

                    if self.log_freq > 0 and self.global_step.eval(session=self.session) % self.log_freq == 0:
                        fd_convergence = {
                                self.rho_t: rt_at_min_p,
                                self.p_rho_t: min_p
                            }
                        summary_convergence = self.session.run(
                            self.summary_convergence,
                            feed_dict=fd_convergence
                        )
                        self.writer.add_summary(summary_convergence, self.global_step.eval(session=self.session))

                    proportion_converged = self.proportion_converged.eval(session=self.session)
                    converged = cur_step > self.convergence_n_iterates and \
                                (min_p > self.convergence_alpha) and \
                                (proportion_converged > self.convergence_alpha)
                                # (p_ta_at_min_p > self.convergence_alpha)

                    if verbose:
                        stderr('rho_t: %s.\n' % rt_at_min_p)
                        stderr('p of rho_t: %s.\n' % min_p)
                        stderr('Location: %s.\n\n' % self.d0_names[min_p_ix])
                        stderr('Iterate meets convergence criteria: %s.\n\n' % converged)
                        stderr('Proportion of recent iterates converged: %s.\n' % proportion_converged)

                else:
                    min_p_ix = min_p = rt_at_min_p = ra_at_min_p = p_ta_at_min_p = None
                    proportion_converged = 0
                    converged = False
                    if verbose:
                        stderr('Convergence checking off.\n')

                self.session.run(self.set_converged, feed_dict={self.converged_in: converged})

                return min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged

    def run_train_step(self, feed_dict):
        """
        Update the model from a batch of training data.

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                to_run = [self.train_op]
                to_run += self.ema_ops

                to_run += [self.loss_func, self.reg_loss]
                to_run_names = ['loss', 'reg_loss']

                if self.loss_cutoff_n_sds:
                    to_run_names.append('n_dropped')
                    to_run.append(self.n_dropped)

                if self.is_bayesian:
                    to_run_names.append('kl_loss')
                    to_run.append(self.kl_loss)

                out = self.session.run(
                    to_run,
                    feed_dict=feed_dict
                )
                self.session.run(self.incr_global_batch_step)

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict




    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    @property
    def name(self):
        return os.path.basename(self.outdir)
    
    @property
    def session(self):
        return self._session

    @property
    def is_bayesian(self):
        """
        Whether the model is defined using variational Bayes.

        :return: ``bool``; whether the model is defined using variational Bayes.
        """

        return len(self.rvs) > 0

    @property
    def has_nn_irf(self):
        """
        Whether the model has any neural network IRFs.

        :return: ``bool``; whether the model has any neural network IRFs.
        """

        return 'NN' in self.form.t.atomic_irf_by_family()

    @property
    def has_nn_impulse(self):
        """
        Whether the model has any neural network impulse transforms.

        :return: ``bool``; whether the model has any neural network impulse transforms.
        """

        for nn_id in self.form.nns_by_id:
            if self.form.nns_by_id[nn_id].nn_type == 'impulse':
                return True
        return False

    @property
    def has_dropout(self):
        """
        Whether the model uses dropout

        :return: ``bool``; whether the model uses dropout.
        """

        for nn_id in self.nns_by_id:
            if self.has_nn_irf and (
                        self.get_nn_meta('ff_dropout_rate', nn_id) or
                        self.get_nn_meta('rnn_h_dropout_rate', nn_id) or
                        self.get_nn_meta('rnn_c_dropout_rate', nn_id) or
                        self.get_nn_meta('h_rnn_dropout_rate', nn_id) or
                        self.get_nn_meta('rnn_dropout_rate', nn_id) or
                        self.get_nn_meta('irf_dropout_rate', nn_id) or
                        self.get_nn_meta('ranef_dropout_rate', nn_id)
                    ):
                return True
            if self.has_nn_impulse and self.get_nn_meta('irf_dropout_rate', nn_id):
                return True
        return False

    @property
    def is_mixed_model(self):
        """
        Whether the model is mixed (i.e. has any random effects).

        :return: ``bool``; whether the model is mixed.
        """

        return len(self.rangf) > 0

    @property
    def training_loglik_full(self):
        """
        Overall training likelihood

        :return: ``float``; Overall training likelihood.
        """

        return self.training_loglik_full_tf.eval(session=self.session)

    @property
    def training_loglik(self):
        """
        Response-wise training likelihood

        :return: ``dict`` of ``list``; Dictionary of training likelihoods, one for each response variable, for each file in the response data.
        """

        out = {}
        for x in self.training_loglik_tf:
            out[x] = {}
            for y in self.training_loglik_tf[x]:
                out[x][y] = self.training_loglik_tf[x][y].eval(session=self.session)

        return out

    @property
    def training_mse(self):
        """
        Response-wise training mean squared error

        :return: ``dict`` of ``list``; Dictionary of training MSEs, one for each response variable, for each file in the response data.
        """

        out = {}
        for x in self.training_mse_tf:
            out[x] = {}
            for y in self.training_mse_tf[x]:
                out[x][y] = self.training_mse_tf[x][y].eval(session=self.session)

        return out

    @property
    def training_rho(self):
        """
        Response-wise training rho (prediction-response correlation)

        :return: ``dict`` of ``list``; Dictionary of training rhos, one for each response variable, for each file in the response data.
        """

        out = {}
        for x in self.training_rho_tf:
            out[x] = {}
            for y in self.training_rho_tf[x]:
                out[x][y] = self.training_rho_tf[x][y].eval(session=self.session)

        return out

    @property
    def training_percent_variance_explained(self):
        """
        Response-wise training percent variance explained

        :return: ``dict`` of ``list``; Dictionary of training training percent variance explained, one for each response variable, for each file in the response data.
        """

        out = {}
        for x in self.training_percent_variance_explained_tf:
            out[x] = {}
            for y in self.training_percent_variance_explained_tf[x]:
                out[x][y] = self.training_percent_variance_explained_tf[x][y].eval(session=self.session)

        return out

    def get_nn_meta(self, key, nn_id=None):
        if key in self.nn_meta[nn_id]:
            return self.nn_meta[nn_id][key]
        if key in self.nn_meta[None]:
            return self.nn_meta[None][key]
        return getattr(self, key)

    def get_nn_irf_output_ndim(self, nn_id):
        """
        Get the number of output dimensions for a given neural network component

        :param nn_id: ``str``; ID of neural network component
        :return: ``int``; number of output dimensions
        """

        assert nn_id in self.nn_irf_ids, 'Unrecognized nn_id for NN IRF: %s.' % nn_id

        n = 0
        n_irf = len(self.nn_irf_output_names[nn_id])
        response_params = self.get_nn_meta('response_params', nn_id)
        for response in self.response_names:
            if response_params:
                dist_name = self.get_response_dist_name(response)
                all_response_params = self.get_response_params(response)
                if dist_name in response_params:
                    _pred_prams = response_params[dist_name]
                elif None in response_params:
                    _response_params = response_params[None]
                else:
                    _response_params = []
                _response_params = [x for x in all_response_params if x in _response_params]
                nparam = len(_response_params)
            elif self.use_distributional_regression:
                nparam = self.get_response_nparam(response)
            else:
                nparam = 1
            ndim = self.get_response_ndim(response)
            n += n_irf * nparam * ndim

        return n

    def get_nn_irf_output_slice_and_shape(self, nn_id):
        """
        Get slice and shape objects that will select out and reshape the elements of an NN's output that are relevant
        to each response.

        :param nn_id: ``str``; ID of neural network component
        :return: ``dict``; map from response name to 2-tuple <slice, shape> containing slice and shape objects
        """

        assert nn_id in self.nn_irf_ids, 'Unrecognized nn_id for NN IRF: %s.' % nn_id

        with self.session.as_default():
            with self.session.graph.as_default():
                slices = {}
                shapes = {}
                n = 0
                n_irf = len(self.nn_irf_output_names[nn_id])
                response_params = self.get_nn_meta('response_params', nn_id)
                for response in self.response_names:
                    if response_params:
                        dist_name = self.get_response_dist_name(response)
                        all_response_params = self.get_response_params(response)
                        if dist_name in response_params:
                            _pred_prams = response_params[dist_name]
                        elif None in response_params:
                            _response_params = response_params[None]
                        else:
                            _response_params = []
                        _response_params = [x for x in all_response_params if x in _response_params]
                        nparam = len(_response_params)
                    elif self.use_distributional_regression:
                        nparam = self.get_response_nparam(response)
                    else:
                        nparam = 1
                    if nparam:
                        ndim = self.get_response_ndim(response)
                        slices[response] = slice(n, n + n_irf * nparam * ndim)
                        shapes[response] = tf.convert_to_tensor((
                            self.X_batch_dim,
                            # Predictor files get tiled out over the time dimension:
                            self.X_time_dim * self.n_impulse_df_noninteraction,
                            n_irf,
                            nparam,
                            ndim
                        ))
                        n += n_irf * nparam * ndim
                    else:
                        slices[response] = None
                        shapes[response] = None

                return slices, shapes

    def build(self, outdir=None, restore=True, report_time=False, verbose=True):
        """
        Construct the CDR(NN) network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param outdir: ``str``; Output directory. If ``None``, inferred.
        :param restore: ``bool``; Restore saved network parameters if model checkpoint exists in the output directory.
        :param report_time: ``bool``; Whether to report the time taken for each initialization step.
        :param verbose: ``bool``; Whether to report progress to stderr.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './cdr_model/'
        else:
            self.outdir = outdir

        with self.session.as_default():
            with self.session.graph.as_default():
                if verbose:
                    stderr('  Initializing input nodes...\n')
                t0 = pytime.time()
                self._initialize_inputs()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_inputs took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing base params...\n')
                t0 = pytime.time()
                self._initialize_base_params()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_base_params took %.2fs\n' % dur)

                for nn_id in self.nn_impulse_ids:
                    if verbose:
                        stderr('  Initializing %s...\n' % nn_id)
                    t0 = pytime.time()
                    self._initialize_nn(nn_id)
                    dur = pytime.time() - t0
                    if report_time:
                        stderr('_initialize_nn for %s took %.2fs\n' % (nn_id, dur))

                    if verbose:
                        stderr('  Compiling %s...\n' % nn_id)
                    t0 = pytime.time()
                    self._compile_nn(nn_id)
                    dur = pytime.time() - t0
                    if report_time:
                        stderr('_compile_nn for %s took %.2fs\n' % (nn_id, dur))

                if verbose:
                    stderr('  Concatenating impulses...\n')
                t0 = pytime.time()
                self._concat_nn_impulses()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_concat_nn_impulses took %.2fs\n' % dur)

                if verbose:
                    stderr('  Compiling intercepts...\n')
                t0 = pytime.time()
                self._compile_intercepts()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_intercepts took %.2fs\n' % dur)

                if verbose:
                    stderr('  Compiling coefficients...\n')
                t0 = pytime.time()
                self._compile_coefficients()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_coefficients took %.2fs\n' % dur)

                if verbose:
                    stderr('  Compiling interactions...\n')
                t0 = pytime.time()
                self._compile_interactions()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_interactions took %.2fs\n' % dur)

                if verbose:
                    stderr('  Compiling IRF params...\n')
                t0 = pytime.time()
                self._compile_irf_params()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_irf_params took %.2fs\n' % dur)

                for nn_id in self.nn_irf_ids:
                    if verbose:
                        stderr('  Initializing %s...\n' % nn_id)
                    t0 = pytime.time()
                    self._initialize_nn(nn_id)
                    dur = pytime.time() - t0
                    if report_time:
                        stderr('_initialize_nn for %s took %.2fs\n' % (nn_id, dur))

                    if verbose:
                        stderr('  Compiling %s...\n' % nn_id)
                    t0 = pytime.time()
                    self._compile_nn(nn_id)
                    dur = pytime.time() - t0
                    if report_time:
                        stderr('_compile_nn for %s took %.2fs\n' % (nn_id, dur))

                if verbose:
                    stderr('  Collecting layerwise ops...\n')
                t0 = pytime.time()
                self._collect_layerwise_ops()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_collect_layerwise_ops took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing IRF lambdas...\n')
                t0 = pytime.time()
                self._initialize_irf_lambdas()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_irf_lambdas took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing IRFs...\n')
                for resix, response in enumerate(self.response_names):
                    if verbose:
                        stderr('\r    Processing response %d/%d...' % (resix + 1, self.n_response))
                    t0 = pytime.time()
                    self._initialize_irfs(self.t, response)
                    dur = pytime.time() - t0
                    if report_time:
                        stderr('_initialize_irfs for %s took %.2fs\n' % (response, dur))
                if verbose:
                    stderr('\n')

                if verbose:
                    stderr('  Compiling IRF impulses...\n')
                t0 = pytime.time()
                self._compile_irf_impulses()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_irf_impulses took %.2fs\n' % dur)

                if verbose:
                    stderr('  Compiling IRF-weighted impulses...\n')
                t0 = pytime.time()
                self._compile_X_weighted_by_irf()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_compile_X_weighted_by_irf took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing response distribution...\n')
                t0 = pytime.time()
                self._initialize_response_distribution()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_response_distribution took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing objective...\n')
                t0 = pytime.time()
                self._initialize_objective()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_objective took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing Tensorboard logging...\n')
                t0 = pytime.time()
                self._initialize_logging()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_logging took %.2fs\n' % dur)

                if verbose:
                    stderr('  Initializing moving averages...\n')
                t0 = pytime.time()
                self._initialize_ema()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_ema took %.2fs\n' % dur)

                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )

                if verbose:
                    stderr('  Initializing saver...\n')
                t0 = pytime.time()
                self._initialize_saver()
                dur = pytime.time() - t0
                if report_time:
                    stderr('_initialize_saver took %.2fs\n' % dur)

                if verbose:
                    stderr('  Loading weights...\n')
                self.load(restore=restore)

                self._initialize_convergence_checking()

                # self.sess.graph.finalize()

    def check_numerics(self):
        """
        Check that all trainable parameters are finite. Throws an error if not.

        :return: ``None``
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                for op in self.check_numerics_ops:
                    self.session.run(op)

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                uninitialized = self.session.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None, suffix=''):
        """
        Save the CDR model.

        :param dir: ``str``; output directory. If ``None``, use model default.
        :param suffix: ``str``; file suffix.
        :return: ``None``
        """

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.session.as_default():
            with self.session.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.session, dir + '/model%s.ckpt' % suffix)
                        with open(dir + '/m%s.obj' % suffix, 'wb') as f:
                            pickle.dump(self, f)
                        self.saver.save(self.session, dir + '/model%s_backup.ckpt' % suffix)
                        with open(dir + '/m%s_backup.obj' % suffix, 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except:
                        stderr('Write failure during save. Retrying...\n')
                        pytime.sleep(1)
                        i += 1
                if i >= 10:
                    stderr('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.session, dir + '/model%s_backup.ckpt' % suffix)
                    with open(dir + '/m%s_backup.obj' % suffix, 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, suffix='', predict=False, restore=True, allow_missing=True):
        """
        Load weights from a CDR checkpoint and/or initialize the CDR model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param suffix: ``str``; file suffix.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """

        if outdir is None:
            outdir = self.outdir
        with self.session.as_default():
            with self.session.graph.as_default():
                if not self.initialized():
                    self.session.run(tf.global_variables_initializer())
                if restore and os.path.exists(outdir + '/checkpoint'):
                    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround for missing vars
                    path = outdir + '/model%s.ckpt' % suffix
                    if (self.early_stopping and self.eval_freq > 0) and \
                            (self.has_converged() or
                             self.global_step.eval(session=self.session) >= self.n_iter):
                        pred_path = outdir + '/model%s_maxval.ckpt' % suffix
                        if not os.path.exists(pred_path + '.meta'):
                            pred_path = path
                    else:
                        pred_path = path
                    try:
                        self.saver.restore(self.session, path)
                        if predict and self.ema_decay:
                            self.ema_saver.restore(self.session, pred_path)
                    except tf.errors.DataLossError:
                        stderr('Read failure during load. Trying from backup...\n')
                        self.saver.restore(self.session, path[:-5] + '%s_backup.ckpt' % suffix)
                        if predict:
                            self.ema_saver.restore(self.session, pred_path[:-5] + '%s_backup.ckpt' % suffix)
                    except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                        if allow_missing:
                            reader = tf.train.NewCheckpointReader(path)
                            saved_shapes = reader.get_variable_to_shape_map()
                            model_var_names = sorted(
                                [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                            ckpt_var_names = sorted(
                                [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                 if var.name.split(':')[0] in saved_shapes])

                            model_var_names_set = set([x[1] for x in model_var_names])
                            ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                            missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                            if len(missing_in_ckpt) > 0:
                                stderr(
                                    'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                        sorted(list(missing_in_ckpt))))
                            missing_in_model = ckpt_var_names_set - model_var_names_set
                            if len(missing_in_model) > 0:
                                stderr(
                                    'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                        sorted(list(missing_in_ckpt))))

                            restore_vars = []
                            name2var = dict(
                                zip(map(lambda x: x.name.split(':')[0], tf.global_variables()),
                                    tf.global_variables()))

                            with tf.variable_scope('', reuse=True):
                                for var_name, saved_var_name in ckpt_var_names:
                                    curr_var = name2var[saved_var_name]
                                    var_shape = curr_var.get_shape().as_list()
                                    if var_shape == saved_shapes[saved_var_name]:
                                        restore_vars.append(curr_var)

                            saver_tmp = tf.train.Saver(restore_vars)
                            saver_tmp.restore(self.session, path)

                            if predict:
                                self.ema_map = {}
                                for v in restore_vars:
                                    self.ema_map[self.ema.average_name(v)] = v
                                saver_tmp = tf.train.Saver(self.ema_map)
                                saver_tmp.restore(self.session, pred_path)

                        else:
                            raise err
                else:
                    if predict:
                        stderr('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def resample_model(self):
        """
        Run any ops required to resample the model (e.g. resampling from posteriors and dropout distributions).

        :return: ``None``
        """

        if self.resample_ops:
            self.session.run(self.resample_ops)

    def finalize(self):
        """
        Close the CDR instance to prevent memory leaks.

        :return: ``None``
        """

        self.session.close()
        tf.reset_default_graph()

    def set_predict_mode(self, mode):
        """
        Set predict mode.
        If set to ``True``, the model enters predict mode and replaces parameters with the exponential moving average of their training iterates.
        If set to ``False``, the model exits predict mode and replaces parameters with their most recently saved values.
        To avoid data loss, always save the model before entering predict mode.

        :param mode: ``bool``; if ``True``, enter predict mode. If ``False``, exit predict mode.
        :return: ``None``
        """

        if mode != self.predict_mode:
            with self.session.as_default():
                with self.session.graph.as_default():
                    self.load(predict=mode)

            self.predict_mode = mode

    def get_training_wall_time(self):
        """
        Returns current training wall time.
        Typically run at the end of an iteration.
        Value of ``time_to_add`` will be added to the current training wall time accumulator.

        :return: ``float``; current training wall time
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                return self.training_wall_time.eval(self.session)

    def update_training_wall_time(self, time_to_add):
        """
        Update (increment) training wall time.
        Typically run at the end of an iteration.
        Value of ``time_to_add`` will be added to the current training wall time accumulator.

        :param time_to_add: ``float``; amount of time (in seconds) to add to current wall time.
        :return: ``None``
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                self.session.run(self.set_training_wall_time_op, feed_dict={self.training_wall_time_in: time_to_add})

    def has_converged(self):
        """
        Check whether model has reached its automatic convergence criteria

        :return: ``bool``; whether the model has converged
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                if self.check_convergence:
                    return self.session.run(self.converged)
                else:
                    return False

    def set_training_complete(self, status):
        """
        Change internal record of whether training is complete.
        Training is recorded as complete when fit() terminates.
        If fit() is called again with a larger number of iterations, training is recorded as incomplete and will not change back to complete until either fit() is called or set_training_complete() is called and the model is saved.

        :param status: ``bool``; Target state (``True`` if training is complete, ``False`` otherwise).
        :return: ``None``
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                if status:
                    self.session.run(self.training_complete_true)
                else:
                    self.session.run(self.training_complete_false)

    def get_response_dist(self, response):
        """
        Get the TensorFlow distribution class for the response distribution assigned to a given response.

        :param response: ``str``; name of response
        :return: TensorFlow distribution object; class of response distribution
        """

        return mcify(self.response_distribution_config[response]['dist'])

    def get_response_dist_name(self, response):
        """
        Get name of the response distribution assigned to a given response.

        :param response: ``str``; name of response
        :return: ``str``; name of response distribution
        """

        return self.response_distribution_config[response]['name']

    def get_response_params(self, response):
        """
        Get tuple of names of parameters of the response distribution for a given response.

        :param response: ``str``; name of response
        :return: ``tuple`` of ``str``; parameters of response distribution
        """

        return self.response_distribution_config[response]['params']

    def get_response_params_tf(self, response):
        """
        Get tuple of TensorFlow-internal names of parameters of the response distribution for a given response.

        :param response: ``str``; name of response
        :return: ``tuple`` of ``str``; parameters of response distribution
        """

        return self.response_distribution_config[response]['params_tf']

    def expand_param_name(self, response, response_param):
        """
        Expand multivariate response distribution parameter names.
        Returns an empty list if the param is not used by the response.
        Returns the unmodified param if the response is univariate.
        Returns the concatenation "<param_name>.<dim_name>" if the response is multivariate.

        :param response: ``str``; name of response variable
        :param response_param: ``str``; name of response distribution parameter
        :return:
        """

        ndim = self.get_response_ndim(response)
        out = []
        if ndim == 1:
            if response_param == 'mean' or response_param in self.get_response_params(response):
                out.append(response_param)
        else:
            for i in range(ndim):
                cat = self.response_ix_to_category[response].get(i, i)
                out.append('%s.%s' % (response_param, cat))

        return out

    def get_response_support(self, response):
        """
        Get the name of the distributional support of the response distribution assigned to a given response

        :param response: ``str``; name of response
        :return: ``str``; label of distributional support
        """

        return self.response_distribution_config[response]['support']

    def get_response_nparam(self, response):
        """
        Get the number of parameters in the response distrbution assigned to a given response

        :param response: ``str``; name of response
        :return: ``int``; number of parameters in the response distribution
        """

        return len(self.get_response_params(response))

    def get_response_ndim(self, response):
        """
        Get the number of dimensions for a given response

        :param response: ``str``; name of response
        :return: ``int``; number of dimensions in the response
        """

        return self.response_ndim[response]

    def is_real(self, response):
        """
        Check whether a given response name is real-valued

        :param response: ``str``; name of response
        :return: ``bool``; whether the response is real-valued
        """

        return self.get_response_support(response) in ('real', 'positive', 'negative')

    def is_categorical(self, response):
        """
        Check whether a given response name has a (multiclass) categorical distribution

        :param response: ``str``; name of response
        :return: ``bool``; whether the response has a categorical distribution
        """

        return self.get_response_dist_name(response) == 'categorical'

    def is_binary(self, response):
        """
        Check whether a given response name is binary (has a Bernoulli distribution)

        :param response: ``str``; name of response
        :return: ``bool``; whether the response has a categorical distribution
        """

        return self.get_response_dist_name(response) == 'bernoulli'

    def has_param(self, response, param):
        """
        Check whether a given parameter name is present in the response distribution assigned to a given response

        :param response: ``str``; name of response
        :param param: ``str``; name of parameter to query
        :return: ``bool``; whether the parameter is present in the response distribution
        """

        return param in self.response_distribution_config[response]['params']

    def has_rnn(self, nn_id):
        """
        Check whether a given NN component includes an RNN transform

        :param nn_id: ``str``; id of NN component
        :return: ``bool``; whether the NN includes an RNN transform
        """

        has_rnn = nn_id in self.nn_impulse_ids or self.get_nn_meta('input_dependent_irf', nn_id)
        has_rnn &= bool(self.get_nn_meta('n_layers_rnn', nn_id))

        return has_rnn

    def has_ff(self, nn_id):
        """
        Check whether a given NN component includes a feedforward transform

        :param nn_id: ``str``; id of NN component
        :return: ``bool``; whether the NN includes a feedforward transform
        """

        has_ff = nn_id in self.nn_impulse_ids
        has_ff &= bool(self.get_nn_meta('n_layers_ff', nn_id)) or not self.has_rnn(nn_id)

        return has_ff

    def report_formula_string(self, indent=0):
        """
        Generate a string representation of the model formula.

        :param indent: ``int``; indentation level
        :return: ``str``; the formula report
        """

        out = ' ' * indent + 'MODEL FORMULA:\n'
        form_str = textwrap.wrap(str(self.form), 150)
        for line in form_str:
            out += ' ' * indent + '  ' + line + '\n'

        out += '\n'

        return out

    def report_settings(self, indent=0):
        """
        Generate a string representation of the model settings.

        :param indent: ``int``; indentation level
        :return: ``str``; the settings report
        """

        out = ' ' * indent + 'MODEL SETTINGS:\n'
        for kwarg in MODEL_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)
        out += ' ' * (indent + 2) + '%s: %s\n' % ('crossval_factor', "\"%s\"" % self.crossval_factor)
        out += ' ' * (indent + 2) + '%s: %s\n' % ('crossval_fold', self.crossval_fold)
        if self.crossval_use_dev_fold:
            out += ' ' * (indent + 2) + '%s: %s\n' % ('crossval_dev_fold', self.crossval_dev_fold)
        if self.git_hash:
            out += ' ' * (indent + 2) + 'Git hash: %s\n' % self.git_hash
        if self.pip_version:
            out += ' ' * (indent + 2) + 'Pip version: %s\n' % self.pip_version

        return out

    def report_parameter_values(self, random=False, level=95, n_samples='default', indent=0):
        """
        Generate a string representation of the model's parameter table.

        :param random: ``bool``; report random effects estimates.
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :param indent: ``int``; indentation level.
        :return: ``str``; the parameter table report
        """

        left_justified_formatter = lambda df, col: '{{:<{}s}}'.format(df[col].str.len().max()).format

        pd.set_option("display.max_colwidth", 10000)
        out = ' ' * indent + 'FITTED PARAMETER VALUES:\n'
        out += ' ' * indent + 'NOTE: Fixed effects for bounded parameters are reported on the constrained space, but\n'
        out += ' ' * indent + '      random effects for bounded parameters are reported on the unconstrained space.\n'
        out += ' ' * indent + '      Therefore, they cannot be directly added. To obtain parameter estimates\n'
        out += ' ' * indent + '      for a bounded variable in a given random effects configuration, first invert the\n'
        out += ' ' * indent + '      bounding transform (e.g. apply inverse softplus), then add random offsets, then\n'
        out += ' ' * indent + '      re-apply the bounding transform (e.g. apply softplus).\n'
        parameter_table = self.parameter_table(
            fixed=True,
            level=level,
            n_samples=n_samples
        )
        formatters = {
            'Parameter': left_justified_formatter(parameter_table, 'Parameter')
        }
        parameter_table_str = parameter_table.to_string(
            index=False,
            justify='left',
            formatters=formatters
        )

        out += ' ' * (indent + 2) + 'Fixed:\n'
        for line in parameter_table_str.splitlines():
            out += ' ' * (indent + 4) + line + '\n'
        out += '\n'

        if random:
            parameter_table = self.parameter_table(
                fixed=False,
                level=level,
                n_samples=n_samples
            )
            formatters = {
                'Parameter': left_justified_formatter(parameter_table, 'Parameter')
            }
            parameter_table_str = parameter_table.to_string(
                index=False,
                justify='left',
                formatters=formatters
            )

            out += ' ' * (indent + 2) + 'Random:\n'
            for line in parameter_table_str.splitlines():
                out += ' ' * (indent + 4) + line + '\n'
            out += '\n'

        pd.set_option("display.max_colwidth", 50)

        return out

    def report_irf_integrals(self, random=False, level=95, n_samples='default', integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's IRF integrals (effect sizes)

        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the IRF integrals report
        """

        pd.set_option("display.max_colwidth", 10000)
        left_justified_formatter = lambda df, col: '{{:<{}s}}'.format(df[col].str.len().max()).format

        if integral_n_time_units is None:
            integral_n_time_units = self.t_delta_limit

        if n_samples == 'default':
            if self.is_bayesian or self.has_dropout:
                n_samples = self.n_samples_eval

        irf_integrals = self.irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            n_time_units=integral_n_time_units,
            n_time_points=1000
        )

        out = ''

        if len(irf_integrals):
            formatters = {
                'IRF': left_justified_formatter(irf_integrals, 'IRF')
            }

            out += ' ' * indent + 'IRF INTEGRALS (EFFECT SIZES):\n'
            out += ' ' * (indent + 2) + 'Integral upper bound (time): %s\n\n' % integral_n_time_units

            ci_str = irf_integrals.to_string(
                index=False,
                justify='left',
                formatters=formatters
            )

            for line in ci_str.splitlines():
                out += ' ' * (indent + 2) + line + '\n'

            out += '\n'

        return out

    def parameter_summary(self, random=False, level=95, n_samples='default', integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's effect sizes and parameter values.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the parameter summary
        """

        out = ' ' * indent + '-----------------\n'
        out += ' ' * indent + 'PARAMETER SUMMARY\n'
        out += ' ' * indent + '-----------------\n\n'

        out += self.report_irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            integral_n_time_units=integral_n_time_units,
            indent=indent+2
        )

        out += self.report_parameter_values(
            random=random,
            level=level,
            n_samples=n_samples,
            indent=indent+2
        )

        return out

    def summary(self, random=False, level=95, n_samples='default', integral_n_time_units=None, indent=0):
        """
        Generate a summary of the fitted model.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :return: ``str``; the model summary
        """

        out = '  ' * indent + '*' * 100 + '\n\n'
        out += ' ' * indent + '############################\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '#    CDR MODEL SUMMARY    #\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '############################\n\n\n'

        out += self.initialization_summary(indent=indent + 2)
        out += '\n'
        out += self.training_evaluation_summary(indent=indent + 2)
        out += '\n'
        out += self.convergence_summary(indent=indent + 2)
        out += '\n'
        out += ' ' * (indent + 2) + 'Response Statistics:\n'
        for response in self.Y_train_sds:
            out += ' ' * (indent + 4) + response + ':\n'
            out += ' ' * (indent + 6) + 'Mean: ' + ', '.join([str(x) for x in self.Y_train_means[response].flatten()]) + '\n'
            out += ' ' * (indent + 6) + 'SD:   ' + ', '.join([str(x) for x in self.Y_train_sds[response].flatten()]) + '\n'
        out += '\n'
        out += self.parameter_summary(
            random=random,
            level=level,
            n_samples=n_samples,
            integral_n_time_units=integral_n_time_units,
            indent=indent + 2
        )
        out += '\n'
        out += '  ' * indent + '*' * 100 + '\n\n'

        return out

    def report_irf_tree(self, indent=0):
        """
        Generate a string representation of the model's IRF tree structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the IRF tree report
        """

        out = ''

        out += ' ' * indent + 'IRF TREE:\n'
        tree_str = str(self.t)
        new_tree_str = ''
        for line in tree_str.splitlines():
            new_tree_str += ' ' * (indent + 2) + line + '\n'
        out += new_tree_str + '\n'

        return out

    def report_n_params(self, indent=0):
        """
        Generate a string representation of the number of trainable model parameters

        :param indent: ``int``; indentation level
        :return: ``str``; the num. parameters report
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.session.run(tf.trainable_variables())
                vars_and_vals = zip(var_names, var_vals)
                vars_and_vals = sorted(list(vars_and_vals), key=lambda x: x[0])
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for v_name, v_val in vars_and_vals:
                    cur_params = int(np.prod(np.array(v_val).shape))
                    n_params += cur_params
                    out += ' ' * indent + '  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params)
                out +=  ' ' * indent + '  TOTAL: %d\n\n' % n_params

                return out

    def report_regularized_variables(self, indent=0):
        """
        Generate a string representation of the model's regularization structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the regularization report
        """

        with self.session.as_default():
            with self.session.graph.as_default():
                assert len(self.regularizer_losses) == len(self.regularizer_losses_names), 'Different numbers of regularized variables found in different places'

                out = ' ' * indent + 'REGULARIZATION:\n'

                if len(self.regularizer_losses_names) == 0:
                    out +=  ' ' * indent + '  No regularized variables.\n\n'
                else:
                    regs = sorted(
                        list(zip(self.regularizer_losses_varnames, self.regularizer_losses_names, self.regularizer_losses_scales)),
                        key=lambda x: x[0]
                    )
                    for name, reg_name, reg_scale in regs:
                        out += ' ' * indent + '  %s:\n' % name
                        out += ' ' * indent + '    Regularizer: %s\n' % reg_name
                        out += ' ' * indent + '    Scale: %s\n' % reg_scale

                    out += '\n'

                if self.is_bayesian:
                    out += ' ' * indent + 'VARIATIONAL PRIORS:\n'

                    kl_penalties = self.kl_penalties

                    if len(kl_penalties) == 0:
                        out +=  ' ' * indent + '  No variational priors.\n\n'
                    else:
                        for name in sorted(list(kl_penalties.keys())):
                            out += ' ' * indent + '  %s:\n' % name
                            for k in sorted(list(kl_penalties[name].keys())):
                                if not k == 'val':
                                    val = str(kl_penalties[name][k])
                                    if len(val) > 100:
                                        val = val[:100] + '...'
                                    out += ' ' * indent + '    %s: %s\n' % (k, val)

                        out += '\n'

                return out

    def report_evaluation(
            self,
            mse=None,
            mae=None,
            f1=None,
            f1_baseline=None,
            acc=None,
            acc_baseline=None,
            rho=None,
            loglik=None,
            loss=None,
            percent_variance_explained=None,
            true_variance=None,
            ks_results=None,
            indent=0
    ):
        """
        Generate a string representation of pre-comupted evaluation metrics.

        :param mse: ``float`` or ``None``; mean squared error, skipped if ``None``.
        :param mae: ``float`` or ``None``; mean absolute error, skipped if ``None``.
        :param f1: ``float`` or ``None``; macro f1 score, skipped if ``None``.
        :param f1_baseline: ``float`` or ``None``; macro f1 score of a baseline (e.g. chance), skipped if ``None``.
        :param acc: ``float`` or ``None``; acuracy, skipped if ``None``.
        :param acc_baseline: ``float`` or ``None``; acuracy of a baseline (e.g. chance), skipped if ``None``.
        :param rho: ``float`` or ``None``; Pearson correlation of predictions with observed response, skipped if ``None``.
        :param loglik: ``float`` or ``None``; log likelihood, skipped if ``None``.
        :param loss: ``float`` or ``None``; loss per training objective, skipped if ``None``.
        :param true_variance: ``float`` or ``None``; variance of targets, skipped if ``None``.
        :param percent_variance_explained: ``float`` or ``None``; percent variance explained, skipped if ``None``.
        :param true_variance: ``float`` or ``None``; true variance, skipped if ``None``.
        :param ks_results: pair of ``float`` or ``None``; if non-null, pair containing ``(D, p_value)`` from Kolmogorov-Smirnov test of errors againts fitted error distribution ; skipped if ``None``.
        :param indent: ``int``; indentation level
        :return: ``str``; the evaluation report
        """

        out = ' ' * indent + 'MODEL EVALUATION STATISTICS:\n'
        if loglik is not None:
            out += ' ' * (indent+2) + 'Loglik:              %s\n' % np.squeeze(loglik)
        if f1 is not None:
            out += ' ' * (indent+2) + 'Macro F1:            %s\n' % np.squeeze(f1)
        if f1_baseline is not None:
            out += ' ' * (indent+2) + 'Macro F1 (baseline): %s\n' % np.squeeze(f1_baseline)
        if acc is not None:
            out += ' ' * (indent+2) + 'Accuracy:            %s\n' % np.squeeze(acc)
        if acc_baseline is not None:
            out += ' ' * (indent+2) + 'Accuracy (baseline): %s\n' % np.squeeze(acc_baseline)
        if mse is not None:
            out += ' ' * (indent+2) + 'MSE:                 %s\n' % np.squeeze(mse)
        if mae is not None:
            out += ' ' * (indent+2) + 'MAE:                 %s\n' % np.squeeze(mae)
        if rho is not None:
            out += ' ' * (indent+2) + 'r(true, pred):       %s\n' % np.squeeze(rho)
        if loss is not None:
            out += ' ' * (indent+2) + 'Loss:                %s\n' % np.squeeze(loss)
        if true_variance is not None and percent_variance_explained is not None:  # No point reporting data variance if there's no comparison
            out += ' ' * (indent+2) + 'True variance:       %s\n' % np.squeeze(true_variance)
        if percent_variance_explained is not None:
            out += ' ' * (indent+2) + '%% var expl:          %.2f%%\n' % np.squeeze(percent_variance_explained)
        out += ' ' * (indent+2) + 'Training wall time:  %.02fs\n' % self.get_training_wall_time()
        if ks_results is not None:
            out += ' ' * (indent+2) + 'Kolmogorov-Smirnov test of goodness of fit of modeled to true error:\n'
            out += ' ' * (indent+4) + 'D value: %s\n' % np.squeeze(ks_results[0])
            out += ' ' * (indent+4) + 'p value: %s\n' % np.squeeze(ks_results[1])
            if ks_results[1] < 0.05:
                out += '\n'
                out += ' ' * (indent+4) + 'NOTE: KS tests will likely reject on large datasets.\n'
                out += ' ' * (indent+4) + 'This does not entail that the model is fatally flawed.\n'
                out += ' ' * (indent+4) + "Check the Q-Q plot in the model's output directory.\n"

        out += '\n'

        return out

    def initialization_summary(self, indent=0):
        """
        Generate a string representation of the model's initialization details

        :param indent: ``int``; indentation level.
        :return: ``str``; the initialization summary
        """

        out = ' ' * indent + '----------------------\n'
        out += ' ' * indent + 'INITIALIZATION SUMMARY\n'
        out += ' ' * indent + '----------------------\n\n'

        out += self.report_formula_string(indent=indent+2)
        out += self.report_settings(indent=indent+2)
        out += '\n' + ' ' * (indent + 2) + 'Training iterations completed: %d\n\n' %self.global_step.eval(session=self.session)
        out += self.report_irf_tree(indent=indent+2)
        out += self.report_n_params(indent=indent+2)
        out += self.report_regularized_variables(indent=indent+2)

        return out

    def training_evaluation_summary(self, indent=0):
        """
        Generate a string representation of the model's training metrics.
        Correctness is not guaranteed until fit() has successfully exited.

        :param indent: ``int``; indentation level.
        :return: ``str``; the training evaluation summary
        """

        out = ' ' * indent + '---------------------------\n'
        out += ' ' * indent + 'TRAINING EVALUATION SUMMARY\n'
        out += ' ' * indent + '---------------------------\n\n'

        if len(self.response_names) > 1:
            out += ' ' * indent + 'Full loglik: %s\n\n' % self.training_loglik_full

        training_loglik = self.training_loglik
        training_mse = self.training_mse
        training_rho = self.training_rho
        training_percent_variance_explained = self.training_percent_variance_explained

        for response in self.response_names:
            file_ix = self.response_to_df_ix[response]
            multiple_files = len(file_ix) > 1
            out += ' ' * indent + 'Response variable: %s\n\n' % response
            for ix in file_ix:
                if multiple_files:
                    out += ' ' * indent + 'File: %s\n\n' % ix
                out += ' ' * indent + 'MODEL EVALUATION STATISTICS:\n'
                out += ' ' * indent +     'Loglik:        %s\n' % training_loglik[response][ix]
                if response in training_mse and ix in training_mse[response]:
                    out += ' ' * indent + 'MSE:           %s\n' % training_mse[response][ix]
                if response in training_rho and ix in training_rho[response]:
                    out += ' ' * indent + 'r(true, pred): %s\n' % training_rho[response][ix]
                if response in training_percent_variance_explained and ix in training_percent_variance_explained[response]:
                    out += ' ' * indent + '%% var expl:    %s\n' % training_percent_variance_explained[response][ix]
                out += '\n'

        return out

    def convergence_summary(self, indent=0):
        """
        Generate a string representation of model's convergence status.

        :param indent: ``int``; indentation level
        :return: ``str``; the convergence report
        """

        out = ' ' * indent + '-------------------\n'
        out += ' ' * indent + 'CONVERGENCE SUMMARY\n'
        out += ' ' * indent + '-------------------\n\n'

        if self.check_convergence:
            n_iter = self.global_step.eval(session=self.session)
            min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged = self.run_convergence_check(verbose=False)
            training_wall_time = self.get_training_wall_time()

            out += ' ' * (indent * 2) + 'Converged:                   %s\n' % converged
            out += ' ' * (indent * 2) + 'Training wall time (s):      %.02f\n' % training_wall_time
            out += ' ' * (indent * 2) + 'Convergence n iterates:      %s\n' % self.convergence_n_iterates
            out += ' ' * (indent * 2) + 'Convergence stride:          %s\n' % self.convergence_stride
            out += ' ' * (indent * 2) + 'Convergence alpha:           %s\n' % self.convergence_alpha
            out += ' ' * (indent * 2) + 'Convergence min p of rho_t:  %s\n' % min_p
            out += ' ' * (indent * 2) + 'Convergence rho_t at min p:  %s\n' % rt_at_min_p
            out += ' ' * (indent * 2) + 'Proportion converged:        %s\n' % proportion_converged
            out += ' ' * (indent * 2) + 'N iterations at convergence: %s\n' % n_iter

            if self.filter_outlier_losses and self.loss_cutoff_n_sds:
                n_dropped = self.n_dropped_ema.eval(self.session)
                out += '\n'
                out += ' ' * (indent * 2) + 'Number of outlier losses dropped per iteration (exp mov avg): %.4f\n' % n_dropped
                if n_dropped > 1:
                    out += ' ' * (indent * 2) + 'WARNING: Moving average of outlier losses dropped per iteration'
                    out += ' ' * (indent * 2) + '         exceeded 1 at convergence. It is possible that training data'
                    out += ' ' * (indent * 2) + '         was systematically excluded. Check the n_dropped field of'
                    out += ' ' * (indent * 2) + '         the TensorFlow logs to determine whether the large average'
                    out += ' ' * (indent * 2) + '         was driven by outlier iterations. If few/no iterations near'
                    out += ' ' * (indent * 2) + '         the end of training had n_dropped=0, some training data was'
                    out += ' ' * (indent * 2) + '         likely ignored at every iteration. Consider using'
                    out += ' ' * (indent * 2) + '         setting filter_outlier_losses to ``False``, which '
                    out += ' ' * (indent * 2) + '         restarts training from the last checkpoint after encountering'
                    out += ' ' * (indent * 2) + '         outlier losses rather than removing them, avoiding bias.'
                out += '\n'

            if converged:
                out += ' ' * (indent + 2) + 'NOTE:\n'
                out += ' ' * (indent + 4) + 'Programmatic diagnosis of convergence in CDR is error-prone because of stochastic optimization.\n'
                out += ' ' * (indent + 4) + 'It is possible that the convergence diagnostics used are too permissive given the stochastic dynamics of the model.\n'
                out += ' ' * (indent + 4) + 'Consider visually checking the learning curves in Tensorboard to see whether the losses have flatlined:\n'
                out += ' ' * (indent + 6) + 'python -m tensorboard.main --logdir=<path_to_model_directory>\n'
                out += ' ' * (indent + 4) + 'If not, consider raising **convergence_alpha** and resuming training.\n'

            else:
                out += ' ' * (indent + 2) + 'Model did not reach convergence criteria in %s iterations.\n' % n_iter
                out += ' ' * (indent + 2) + 'NOTE:\n'
                out += ' ' * (indent + 4) + 'Programmatic diagnosis of convergence in CDR is error-prone because of stochastic optimization.\n'
                out += ' ' * (indent + 4) + 'It is possible that the convergence diagnostics used are too conservative given the stochastic dynamics of the model.\n'
                out += ' ' * (indent + 4) + 'Consider visually checking the learning curves in Tensorboard to see whether thelosses have flatlined:\n'
                out += ' ' * (indent + 6) + 'python -m tensorboard.main --logdir=<path_to_model_directory>\n'
                out += ' ' * (indent + 4) + 'If so, consider the model converged.\n'

        else:
            out += ' ' * (indent + 2) + 'Convergence checking is turned off.\n'

        return out

    def is_non_dirac(self, impulse_name):
        """
        Check whether an impulse is associated with a non-Dirac response function

        :param impulse_name: ``str``; name of impulse
        :return: ``bool``; whether the impulse is associated with a non-Dirac response function
        """

        return impulse_name in self.non_dirac_impulses

    def sample_impulses(self, *args, **kwargs):
        """
        Resample impulses from an empirical multivariate normal distribution derived from the training data.

        :param args: ``list`` or ``tuple``; args to pass to scipy's resampling method
        :param kwargs: ``dict``; kwargs to pass to scipy's resampling method
        :return: ``numpy`` array; samples
        """

        sample = self.impulse_sampler.rvs(*args, **kwargs)
        if len(sample.shape) < 2:
            sample = sample[None, ...]

        return pd.DataFrame(sample, columns=self.impulse_cov.columns)

    def fit(
            self,
            X,
            Y,
            X_dev=None,
            Y_dev=None,
            X_in_Y_names=None,
            n_iter=None,
            force_training_evaluation=True,
            optimize_memory=False
    ):
        """
        Fit the model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y: ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **Y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **Y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **Y**
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **Y**)
            * A column for each random grouping factor in the model formula

            In general, **Y** will be identical to the parameter **Y** provided at model initialization.
        :param X_dev: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X_dev**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y_dev: ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **Y_dev** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **Y_dev**
            * ``first_obs``:  Index in the design matrix **X_dev** of the first observation in the time series associated with each entry in **Y_dev**
            * ``last_obs``:  Index in the design matrix **X_dev** of the immediately preceding observation in the time series associated with each entry in **Y_dev**
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **Y_dev**)
            * A column for each random grouping factor in the model formula

            In general, **Y_dev** will be identical to the parameter **Y_dev** provided at model initialization.
        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param n_iter: ``int`` or ``None``; maximum number of training iterations. Training will stop either at convergence or **n_iter**, whichever happens first. If ``None``, uses model default.
        :param force_training_evaluation: ``bool``; (Re-)run post-fitting evaluation, even if resuming a model whose training is already complete.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        """

        if not isinstance(X, list):
            X = [X]
        if Y is not None and not isinstance(Y, list):
            Y = [Y]

        cv_exclude = []
        if self.crossval_use_dev_fold:
            assert X_dev is None, '``X_dev`` cannot be specified for models that use ``crossval_dev_fold``.' + \
                                  'Either provide a value for ``X_dev`` or set ``crossval_dev_fold`` to ``None.'
            assert Y_dev is None, '``Y_dev`` cannot be specified for models that use ``crossval_dev_fold``. ' + \
                                  'Either provide a value for ``Y_dev`` or set ``crossval_dev_fold`` to ``None.'
            X_dev = X
            Y_dev = [_Y[_Y[self.crossval_factor] == self.crossval_dev_fold] for _Y in Y]
            cv_exclude.append(self.crossval_dev_fold)

        if self.early_stopping and self.eval_freq > 0:
            assert X_dev is not None, '``X_dev`` must be specified if early stopping is used'
            assert Y_dev is not None, '``Y_dev`` must be specified if early stopping is used'

        # Preprocess data
        # Training data
        if self.use_crossval:
            cv_exclude.append(self.crossval_fold)
            Y = [_Y[~_Y[self.crossval_factor].isin(cv_exclude)] for _Y in Y]
        lengths = [len(_Y) for _Y in Y]
        n = sum(lengths)
        if not np.isfinite(self.minibatch_size):
            minibatch_size = n
        else:
            minibatch_size = self.minibatch_size
        n_minibatch = int(math.ceil(n / minibatch_size))

        stderr('*' * 100 + '\n' + self.initialization_summary() + '*' * 100 + '\n\n')
        with open(self.outdir + '/initialization_summary.txt', 'w') as i_file:
            i_file.write(self.initialization_summary())

        if not n_iter:
            n_iter = self.n_iter

        X_in = X
        Y_in = Y
        if X_in_Y_names:
            X_in_Y_names = [x for x in X_in_Y_names if x in self.impulse_names]

        usingGPU = tf.test.is_gpu_available()
        stderr('Using GPU: %s\nNumber of training samples: %d\n\n' % (usingGPU, n))

        Y, first_obs, last_obs, Y_time, Y_mask, Y_gf, X_in_Y = build_CDR_response_data(
            self.response_names,
            Y=Y_in,
            X_in_Y_names=X_in_Y_names,
            Y_category_map=self.response_category_to_ix,
            response_to_df_ix=self.response_to_df_ix,
            gf_names=self.rangf,
            gf_map=self.rangf_map
        )

        if not optimize_memory:
            # Training data
            X, X_time, X_mask = build_CDR_impulse_data(
                X_in,
                first_obs,
                last_obs,
                X_in_Y_names=X_in_Y_names,
                X_in_Y=X_in_Y,
                history_length=self.history_length,
                future_length=self.future_length,
                impulse_names=self.impulse_names,
                int_type=self.int_type,
                float_type=self.float_type,
            )

        if False:
            self.make_plots(prefix='plt')

        with self.session.as_default():
            with self.session.graph.as_default():
                self.run_convergence_check(verbose=False)

                if (self.global_step.eval(session=self.session) < n_iter) and not self.has_converged():
                    self.set_training_complete(False)

                if self.training_complete.eval(session=self.session):
                    stderr('Model training is already complete; no additional updates to perform.' + \
                           'To train for additional iterations, re-run fit() with a larger n_iter.\n\n')
                else:
                    if self.global_step.eval(session=self.session) == 0:
                        if not type(self).__name__.startswith('CDRNN'):
                            summary_params = self.session.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.session))
                            if self.log_random and self.is_mixed_model:
                                summary_random = self.session.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.session))
                            self.writer.flush()
                    else:
                        stderr('Resuming training from most recent checkpoint...\n\n')

                    if self.global_step.eval(session=self.session) == 0:
                        stderr('Saving initial weights...\n')
                        self.save()

                    # Counter for number of times an attempted iteration has failed due to outlier
                    # losses or failed numerics checks
                    n_failed = 0
                    failed = False

                    t0_iter = pytime.time()

                    while not self.has_converged() and \
                            self.global_step.eval(session=self.session) < n_iter:
                        if failed:
                            stderr('Restarting from most recent checkpoint (restart #%d from this checkpoint).\n' % n_failed)
                            self.load() # Reload from previous save point
                        p, p_inv = get_random_permutation(n)
                        stderr('-' * 50 + '\n')
                        stderr('Iteration %d\n' % int(self.global_step.eval(session=self.session) + 1))
                        stderr('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            stderr('Learning rate: %s\n' % self.lr.eval(session=self.session))

                        pb = keras.utils.Progbar(n_minibatch)

                        loss_total = 0.
                        reg_loss_total = 0.
                        if self.is_bayesian:
                            kl_loss_total = 0.
                        if self.loss_cutoff_n_sds:
                            n_dropped = 0.

                        failed = False
                        for i in range(0, n, minibatch_size):
                            indices = p[i:i+minibatch_size]
                            if optimize_memory:
                                _Y = Y[indices]
                                _first_obs = [x[indices] for x in first_obs]
                                _last_obs = [x[indices] for x in last_obs]
                                _Y_time = Y_time[indices]
                                _Y_mask = Y_mask[indices]
                                _Y_gf = None if Y_gf is None else Y_gf[indices]
                                _X_in_Y = None if X_in_Y is None else X_in_Y[indices]
                                _X, _X_time, _X_mask = build_CDR_impulse_data(
                                    X_in,
                                    _first_obs,
                                    _last_obs,
                                    X_in_Y_names=X_in_Y_names,
                                    X_in_Y=_X_in_Y,
                                    history_length=self.history_length,
                                    future_length=self.future_length,
                                    impulse_names=self.impulse_names,
                                    int_type=self.int_type,
                                    float_type=self.float_type,
                                )
                                fd = {
                                    self.X: _X,
                                    self.X_time: _X_time,
                                    self.X_mask: _X_mask,
                                    self.Y: _Y,
                                    self.Y_time: _Y_time,
                                    self.Y_mask: _Y_mask,
                                    self.Y_gf: _Y_gf,
                                    self.training: not self.predict_mode
                                }
                            else:
                                fd = {
                                    self.X: X[indices],
                                    self.X_time: X_time[indices],
                                    self.X_mask: X_mask[indices],
                                    self.Y: Y[indices],
                                    self.Y_time: Y_time[indices],
                                    self.Y_mask: Y_mask[indices],
                                    self.Y_gf: None if Y_gf is None else Y_gf[indices],
                                    self.training: not self.predict_mode
                                }

                            try:
                                info_dict = self.run_train_step(fd)
                            except tf.errors.InvalidArgumentError as e:
                                failed = True
                                stderr('\nDid not pass stability check.\nNon-finite gradients.\n')
                                break

                            try:
                                self.check_numerics()
                            except tf.errors.InvalidArgumentError as e:
                                failed = True
                                stderr('\nDid not pass stability check.\nNon-finite parameter values.\n')
                                break

                            if self.loss_cutoff_n_sds:
                                n_dropped += info_dict['n_dropped']
                                if not self.filter_outlier_losses and n_dropped:
                                    failed = True
                                    stderr('\nDid not pass stability check.\nLarge outlier losses.\n')
                                    break

                            loss_cur = info_dict['loss']
                            if not np.isfinite(loss_cur):
                                loss_cur = 0
                            loss_total += loss_cur

                            pb_update = [('loss', loss_cur)]
                            if 'reg_loss' in info_dict:
                                reg_loss_cur = info_dict['reg_loss']
                                reg_loss_total += reg_loss_cur
                                pb_update.append(('reg', reg_loss_cur))
                            if 'kl_loss' in info_dict:
                                kl_loss_cur = info_dict['kl_loss']
                                kl_loss_total += kl_loss_cur
                                pb_update.append(('kl', kl_loss_cur))

                            pb.update((i/minibatch_size) + 1, values=pb_update)

                        if failed:
                            n_failed += 1
                            assert n_failed <= 1000, '1000 restarts in a row from the same save point ' \
                                                     'failed to pass stability checks. Model training ' \
                                                     'has failed.'
                            continue

                        self.session.run(self.incr_global_step)

                        if self.eval_freq > 0 and \
                                self.global_step.eval(session=self.session) % self.eval_freq == 0:
                            self.save()
                            if self.crossval_use_dev_fold:
                                partition_name = 'CVdev'
                            else:
                                partition_name = 'dev'
                            dev_results, _ = self.evaluate(
                                X_dev,
                                Y_dev,
                                X_in_Y_names=X_in_Y_names,
                                partition=partition_name,
                                optimize_memory=optimize_memory
                            )
                            dev_ll = dev_results['full_log_lik']
                            dev_ll_max_prev = self.dev_ll_max.eval(session=self.session)
                            log_fd = {self.dev_ll_total: dev_ll}
                            for metric in self.dev_metrics:
                                if metric != 'full_log_lik':
                                    for response in self.dev_metrics[metric]:
                                        for ix in self.dev_metrics[metric][response]:
                                            log_fd[self.dev_metrics[metric][response][ix]] = np.squeeze(
                                                dev_results[metric][response][ix]
                                            )
                            summary_dev, _ = self.session.run(
                                [self.summary_dev, self.set_dev_ll_max],
                                feed_dict=log_fd
                            )
                            self.writer.add_summary(
                                summary_dev,
                                self.global_step.eval(session=self.session)
                            )
                        else:
                            dev_ll = None

                        if self.check_convergence:
                            if self.early_stopping and self.eval_freq > 0:
                                if dev_ll is not None:
                                    fd = {self.loss_total: -dev_ll}
                                    self.run_convergence_check(verbose=False, feed_dict=fd)
                                    if dev_ll > dev_ll_max_prev:
                                        self.save(suffix='_maxval')
                            else:
                                fd = {self.loss_total: loss_total/n_minibatch}
                                self.run_convergence_check(verbose=False, feed_dict=fd)

                        if self.log_freq > 0 and \
                                self.global_step.eval(session=self.session) % self.log_freq == 0:
                            loss_total /= n_minibatch
                            reg_loss_total /= n_minibatch
                            log_fd = {self.loss_total: loss_total, self.reg_loss_total: reg_loss_total}
                            if self.is_bayesian:
                                kl_loss_total /= n_minibatch
                                log_fd[self.kl_loss_total] = kl_loss_total
                            if self.filter_outlier_losses and self.loss_cutoff_n_sds:
                                log_fd[self.n_dropped_in] = n_dropped
                            summary_train_loss = self.session.run(self.summary_opt, feed_dict=log_fd)
                            self.writer.add_summary(
                                summary_train_loss,
                                self.global_step.eval(session=self.session)
                            )
                            summary_params = self.session.run(self.summary_params)
                            self.writer.add_summary(
                                summary_params,
                                self.global_step.eval(session=self.session)
                            )
                            if self.log_random and self.is_mixed_model:
                                summary_random = self.session.run(self.summary_random)
                                self.writer.add_summary(
                                    summary_random,
                                    self.global_step.eval(session=self.session)
                                )
                            self.writer.flush()

                        t1_iter = pytime.time()
                        t_iter = t1_iter - t0_iter
                        t0_iter = t1_iter
                        self.update_training_wall_time(t_iter)
                        stderr('Iteration time: %.2fs\n' % t_iter)

                        if self.save_freq > 0 and \
                                self.global_step.eval(session=self.session) % self.save_freq == 0:
                            n_failed = 0
                            self.save()
                        if self.plot_freq > 0 and \
                                self.global_step.eval(session=self.session) % self.plot_freq == 0:
                            self.make_plots(prefix='plt')

                        if self.check_convergence:
                            stderr('Convergence:    %.2f%%\n' %
                                   (100 * self.session.run(self.proportion_converged) /
                                    self.convergence_alpha))


                    assert not failed, 'Training loop completed without passing stability checks. Model training has failed.'

                    self.save()

                    # End of training plotting and evaluation.
                    # For CDRMLE, this is a crucial step in the model definition because it provides the
                    # variance of the output distribution for computing log likelihood.

                    self.make_plots(prefix='plt')

                    if self.is_bayesian or self.has_dropout:
                        # Generate plots with 95% credible intervals
                        self.make_plots(n_samples=self.n_samples_eval, prefix='plt')

                if not self.training_complete.eval(session=self.session) or force_training_evaluation:
                    # Extract and save predictions
                    if self.crossval_use_dev_fold:
                        train_name = 'CVtrain'
                        dev_name = 'CVdev'
                    else:
                        train_name = 'train'
                        dev_name = 'dev'
                    metrics, summary = self.evaluate(
                        X_in,
                        Y_in,
                        X_in_Y_names=X_in_Y_names,
                        dump=True,
                        partition=train_name,
                        optimize_memory=optimize_memory
                    )

                    if self.eval_freq > 0:
                        dev_results, _ = self.evaluate(
                            X_dev,
                            Y_dev,
                            X_in_Y_names=X_in_Y_names,
                            dump=True,
                            partition=dev_name,
                            optimize_memory=optimize_memory
                        )
                        
                    # Extract and save losses
                    ll_full = sum([_ll for r in self.response_names for _ll in metrics['log_lik'][r]])
                    self.session.run(self.set_training_loglik_full, feed_dict={self.training_loglik_full_in: ll_full})
                    fd = {}
                    to_run = []
                    for response in self.training_loglik_in:
                        for ix in self.training_loglik_in[response]:
                            tensor = self.training_loglik_in[response][ix]
                            fd[tensor] = np.squeeze(metrics['log_lik'][response][ix])
                            to_run.append(self.set_training_loglik[response][ix])
                    for response in self.training_mse_in:
                        if self.is_real(response):
                            for ix in self.training_mse_in[response]:
                                tensor = self.training_mse_in[response][ix]
                                fd[tensor] = np.squeeze(metrics['mse'][response][ix])
                                to_run.append(self.set_training_mse[response][ix])
                    for response in self.training_rho_in:
                        if self.is_real(response):
                            for ix in self.training_rho_in[response]:
                                tensor = self.training_rho_in[response][ix]
                                fd[tensor] = np.squeeze(metrics['rho'][response][ix])
                                to_run.append(self.set_training_rho[response][ix])

                    self.session.run(to_run, feed_dict=fd)
                    self.save()

                    self.set_training_complete(True)
                    self.save()

    def run_predict_op(
            self,
            feed_dict,
            responses=None,
            n_samples=None,
            algorithm='MAP',
            return_preds=True,
            return_loglik=False,
            verbose=True
    ):
        """
        Generate predictions from a batch of data.

        :param feed_dict: ``dict``; A dictionary mapping string input names (e.g. ``'X'``, ``'Y'``) to their values.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) of response variable(s) to predict. If ``None``, predicts all responses.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param return_preds: ``bool``; whether to return predictions.
        :param return_loglik: ``bool``; whether to return elementwise log likelihoods. Requires that **Y** is not ``None``.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``dict`` of ``numpy`` arrays; Predicted responses and/or log likelihoods, one for each training sample. Key order: <('preds'|'log_lik'), response>.
        """

        assert 'Y' in feed_dict or not return_loglik, 'Cannot return log likelihood when Y is not provided.'

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict['use_MAP_mode'] = use_MAP_mode

        if responses is None:
            responses = self.response_names
        if not isinstance(responses, list):
            responses = [responses]

        if return_preds or return_loglik:
            if use_MAP_mode:
                to_run = {}
                if return_preds:
                    to_run_preds = {x: self.prediction[x] for x in responses}
                    to_run['preds'] = to_run_preds
                if return_loglik:
                    to_run_loglik = {x: self.ll_by_var[x] for x in responses}
                    to_run['log_lik'] = to_run_loglik
                fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
                out = self.session.run(to_run, feed_dict=fd)
            else:
                if n_samples is None:
                    n_samples = self.n_samples_eval

                if verbose:
                    pb = keras.utils.Progbar(n_samples)

                out = {}
                if return_preds:
                    out['preds'] = {x: np.zeros((len(feed_dict['Y_time']), n_samples)) for x in responses}
                if return_loglik:
                    out['log_lik'] = {x: np.zeros((len(feed_dict['Y_time']), n_samples)) for x in responses}

                for i in range(n_samples):
                    self.resample_model()

                    to_run = {}
                    if return_preds:
                        to_run_preds = {x: self.prediction[x] for x in responses}
                        to_run['preds'] = to_run_preds
                    else:
                        to_run_preds = None
                    if return_loglik:
                        to_run_loglik = {x: self.ll_by_var[x] for x in responses}
                        to_run['log_lik'] = to_run_loglik
                    else:
                        to_run_loglik = None
                    fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
                    _out = self.session.run(to_run, feed_dict=fd)
                    if to_run_preds:
                        _preds = _out['preds']
                        for _response in _preds:
                            out['preds'][_response][:, i] = _preds[_response]
                    if to_run_loglik:
                        _log_lik = _out['log_lik']
                        for _response in _log_lik:
                            out['log_lik'][_response][:, i] = _log_lik[_response]
                    if verbose:
                        pb.update(i + 1)

                if return_preds:
                    for _response in out['preds']:
                        _preds = out['preds'][_response]
                        dist_name = self.get_response_dist_name(_response)
                        if dist_name == 'bernoulli':  # Majority vote
                            _preds = np.round(np.mean(_preds, axis=1)).astype('int')
                        elif dist_name == 'categorical':  # Majority vote
                            _preds = scipy.stats.mode(_preds, axis=1)
                        else:  # Average
                            _preds = _preds.mean(axis=1)
                        out['preds'][_response] = _preds

                if return_loglik:
                    for _response in out['log_lik']:
                        out['log_lik'][_response] = out['log_lik'][_response].mean(axis=1)

            return out

    def run_loss_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        """
        Compute the elementwise training loss of a batch of data.

        :param feed_dict: ``dict``; A dictionary mapping string input names (e.g. ``'X'``, ``'Y'``) to their values.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; total training loss for batch
        """

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict['use_MAP_mode'] = use_MAP_mode

        if use_MAP_mode:
            fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
            loss = self.session.run(self.loss_func, feed_dict=fd)
        else:
            feed_dict[self.use_MAP_mode] = False
            if n_samples is None:
                n_samples = self.n_samples_eval

            if verbose:
                pb = keras.utils.Progbar(n_samples)

            loss = np.zeros((len(feed_dict[self.Y_time]), n_samples))

            for i in range(n_samples):
                self.resample_model()
                fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
                loss[:, i] = self.session.run(self.loss_func, feed_dict=fd)
                if verbose:
                    pb.update(i + 1)

            loss = loss.mean(axis=1)

        return loss

    def run_conv_op(self, feed_dict, responses=None, response_param=None, n_samples=None, algorithm='MAP', verbose=True):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.

        :param feed_dict: ``dict``; A dictionary mapping string input names (e.g. ``'X'``, ``'X_time'``) to their values.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to convolve toward. If ``None``, convolves toward all univariate responses. Multivariate convolution (e.g. of categorical responses) is supported but turned off by default to avoid excessive computation. When convolving toward a multivariate response, a set of convolved predictors will be generated for each dimension of the response.
        :param response_param: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of response distribution(s) to convolve toward per response variable. Any param names not used by the response distribution for a given response will be ignored. If ``None``, convolves toward the first parameter of each response distribution.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``dict`` of ``numpy`` arrays; The convolved inputs, one per **response_param** per **response**. Each element has shape (batch, terminals)
        """

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict['use_MAP_mode'] = use_MAP_mode

        if responses is None:
            responses = [x for x in self.response_names if self.get_response_ndim(x) == 1]
        if isinstance(responses, str):
            responses = [responses]

        if response_param is None:
            response_param = set()
            for _response in responses:
                response_param.add(self.get_response_params(_response)[0])
            response_param = sorted(list(response_param))
        if isinstance(response_param, str):
            response_param = [response_param]

        if use_MAP_mode:
            to_run = {}
            for _response in responses:
                to_run[_response] = self.X_conv_delta[_response]
            fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
            X_conv = self.session.run(to_run, feed_dict=fd)
        else:
            X_conv = {}
            for _response in responses:
                nparam = self.get_response_nparam(_response)
                ndim = self.get_response_ndim(_response)
                X_conv[_response] = np.zeros(
                    (len(feed_dict[self.Y_time]), len(self.terminal_names), nparam, ndim, n_samples)
                )

            if n_samples is None:
                n_samples = self.n_samples_eval
            if verbose:
                pb = keras.utils.Progbar(n_samples)

            for i in range(0, n_samples):
                self.resample_model()
                to_run = {}
                for _response in responses:
                    to_run[_response] = self.X_conv_delta[_response]
                fd = {getattr(self, x): feed_dict[x] for x in feed_dict}
                _X_conv = self.session.run(to_run, feed_dict=fd)
                for _response in _X_conv:
                    X_conv[_response][..., i] = _X_conv[_response]
                if verbose:
                    pb.update(i + 1, force=True)

            for _response in X_conv:
                X_conv[_response] = X_conv[_response].mean(axis=2)

        # Break things out by response dimension
        out = {}
        for _response in X_conv:
            for i, _response_param in enumerate(response_param):
                if self.has_param(_response, _response_param):
                    dim_names = self.expand_param_name(_response, _response_param)
                    for j, _dim_name in enumerate(dim_names):
                        if _response not in out:
                            out[_response] = {}
                        out[_response][_dim_name] = X_conv[_response][..., i, j]

        return out


    def predict(
            self,
            X,
            Y=None,
            first_obs=None,
            last_obs=None,
            Y_time=None,
            Y_gf=None,
            responses=None,
            X_in_Y_names=None,
            X_in_Y=None,
            n_samples=None,
            algorithm='MAP',
            return_preds=True,
            return_loglik=False,
            sum_outputs_along_T=True,
            sum_outputs_along_K=True,
            dump=False,
            extra_cols=False,
            partition=None,
            optimize_memory=False,
            verbose=True
    ):
        """
        Predict from the pre-trained CDR model.
        Predictions are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.
        Can also be used to generate log likelihoods when targets **Y** are provided (see options below).

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y (optional): ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            This parameter is optional and responses are not directly used. It simply allows the user to omit the
            inputs **Y_time**, **Y_gf**, **first_obs**, and **last_obs**, since they can be inferred from **Y**
            If supplied, each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **y**
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model formula

        :param first_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of first observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the first impulse in the time series associated with each response. If ``None``, inferred from **Y**.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of last observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the last impulse in the time series associated with each response. If ``None``, inferred from **Y**.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param Y_time: ``list`` of response timestamp vectors (``list``, ``pandas`` series, or ``numpy`` vector); vector(s) of response timestamps, one for each response array. Needed to timestamp any response-aligned predictors (ignored if none in model).
        :param Y_gf: ``list`` of random grouping factor values (``list``, ``pandas`` series, or ``numpy`` vector); random grouping factor values (if applicable), one for each response dataframe.
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) of response(s) to predict. If ``None``, predicts all responses.
        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param X_in_Y: ``list`` of ``pandas`` ``DataFrame`` or ``None``; tables (one per response array) of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, inferred from **Y** and **X_in_Y_names**.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param return_preds: ``bool``; whether to return predictions.
        :param return_loglik: ``bool``; whether to return elementwise log likelihoods. Requires that **Y** is not ``None``.
        :param sum_outputs_along_T: ``bool``; whether to sum IRF-weighted predictors along the time dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for timestep-specific evaluation.
        :param sum_outputs_along_K: ``bool``; whether to sum IRF-weighted predictors along the predictor dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for impulse-specific evaluation.
        :param dump: ``bool``; whether to save generated predictions (and log likelihood vectors if applicable) to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert Y is not None or not return_loglik, 'Cannot return log likelihood when Y is not provided.'
        assert not dump or (
                sum_outputs_along_T and sum_outputs_along_K), 'dump=True is only supported if sum_outputs_along_T=True and sum_outputs_along_K=True'

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)
            stderr('Computing predictions...\n')

        if responses is None:
            responses = self.response_names
        if not isinstance(responses, list):
            responses = [responses]

        if algorithm.lower() == 'map':
            for response in responses:
                dist_name = self.get_response_dist_name(response)
                if dist_name.lower() == 'exgaussian':
                    stderr('WARNING: The exact mode of the ExGaussian distribution is currently not implemented,\n' +
                           'and an approximation is used that degrades when the skew is larger than the scale.\n' +
                           'Predictions/errors from ExGaussian models should be treated with caution.\n')
                    break

        # Preprocess data
        if not isinstance(X, list):
            X = [X]
        X_in = X
        if Y is None:
            assert Y_time is not None, 'Either Y or Y_time must be provided.'
            lengths = [len(_Y_time) for _Y_time in Y_time]
        else:
            if not isinstance(Y, list):
                Y = [Y]
            lengths = [len(_Y) for _Y in Y]
        n = sum(lengths)
        Y_in = Y
        if Y_time is None:
            Y_time_in = [_Y.time for _Y in Y]
        else:
            Y_time_in = Y_time
        if Y_gf is None:
            assert Y is not None, 'Either Y or Y_gf must be provided.'
            Y_gf_in = Y
        else:
            Y_gf_in = Y_gf
        if X_in_Y_names:
            X_in_Y_names = [x for x in X_in_Y_names if x in self.impulse_names]
        X_in_Y_in = X_in_Y

        Y, first_obs, last_obs, Y_time, Y_mask, Y_gf, X_in_Y = build_CDR_response_data(
            self.response_names,
            Y=Y_in,
            first_obs=first_obs,
            last_obs=last_obs,
            Y_gf=Y_gf_in,
            X_in_Y_names=X_in_Y_names,
            X_in_Y=X_in_Y_in,
            Y_category_map=self.response_category_to_ix,
            response_to_df_ix=self.response_to_df_ix,
            gf_names=self.rangf,
            gf_map=self.rangf_map
        )

        if not optimize_memory:
            X, X_time, X_mask = build_CDR_impulse_data(
                X_in,
                first_obs,
                last_obs,
                X_in_Y_names=X_in_Y_names,
                X_in_Y=X_in_Y,
                history_length=self.history_length,
                future_length=self.future_length,
                impulse_names=self.impulse_names,
                int_type=self.int_type,
                float_type=self.float_type,
            )

        if return_preds or return_loglik:
            with self.session.as_default():
                with self.session.graph.as_default():
                    self.set_predict_mode(True)

                    out = {}
                    out_shape = (n,)
                    if not sum_outputs_along_T:
                        out_shape = out_shape + (self.history_length + self.future_length,)
                    if not sum_outputs_along_K:
                        n_impulse = self.n_impulse
                        out_shape = out_shape + (n_impulse,)

                    if return_preds:
                        out['preds'] = {}
                        for _response in responses:
                            if self.is_real(_response):
                                dtype = self.FLOAT_NP
                            else:
                                dtype = self.INT_NP
                            out['preds'][_response] = np.zeros(out_shape, dtype=dtype)
                    if return_loglik:
                        out['log_lik'] = {x: np.zeros(out_shape) for x in responses}

                    B = self.eval_minibatch_size
                    n_eval_minibatch = math.ceil(n / B)
                    for i in range(0, n, B):
                        if verbose:
                            stderr('\rMinibatch %d/%d' % ((i / B) + 1, n_eval_minibatch))
                        if optimize_memory:
                            _Y = None if Y is None else Y[i:i + B]
                            _first_obs = [x[i:i + B] for x in first_obs]
                            _last_obs = [x[i:i + B] for x in last_obs]
                            _Y_time = Y_time[i:i + B]
                            _Y_mask = Y_mask[i:i + B]
                            _Y_gf = None if Y_gf is None else Y_gf[i:i + B]
                            _X_in_Y = None if X_in_Y is None else X_in_Y[i:i + B]

                            _X, _X_time, _X_mask = build_CDR_impulse_data(
                                X_in,
                                _first_obs,
                                _last_obs,
                                X_in_Y_names=X_in_Y_names,
                                X_in_Y=_X_in_Y,
                                history_length=self.history_length,
                                future_length=self.future_length,
                                impulse_names=self.impulse_names,
                                int_type=self.int_type,
                                float_type=self.float_type,
                            )
                            fd = {
                                'X': _X,
                                'X_time': _X_time,
                                'X_mask': _X_mask,
                                'Y_time': _Y_time,
                                'Y_mask': _Y_mask,
                                'Y_gf': _Y_gf,
                                'training': not self.predict_mode,
                                'sum_outputs_along_T': sum_outputs_along_T,
                                'sum_outputs_along_K': sum_outputs_along_K
                            }
                            if return_loglik:
                                fd['Y'] = _Y
                        else:
                            fd = {
                                'X': X[i:i + B],
                                'X_time': X_time[i:i + B],
                                'X_mask': X_mask[i:i + B],
                                'Y_time': Y_time[i:i + B],
                                'Y_mask': Y_mask[i:i + B],
                                'Y_gf': None if Y_gf is None else Y_gf[i:i + B],
                                'training': not self.predict_mode,
                                'sum_outputs_along_T': sum_outputs_along_T,
                                'sum_outputs_along_K': sum_outputs_along_K
                            }
                            if return_loglik:
                                fd['Y'] = Y[i:i + B]
                        _out = self.run_predict_op(
                            fd,
                            responses=responses,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            return_preds=return_preds,
                            return_loglik=return_loglik,
                            verbose=verbose
                        )

                        if return_preds:
                            for _response in _out['preds']:
                                out['preds'][_response][i:i + B] = _out['preds'][_response]
                        if return_loglik:
                            for _response in _out['log_lik']:
                                out['log_lik'][_response][i:i + B] = _out['log_lik'][_response]

                    # Convert predictions to category labels, if applicable
                    if return_preds:
                        for _response in out['preds']:
                            if self.is_categorical(_response):
                                mapper = np.vectorize(lambda x: self.response_ix_to_category[_response].get(x, x))
                                out['preds'][_response] = mapper(out['preds'][_response])

                    # Split into per-file predictions.
                    # Exclude the length of last file because it will be inferred.
                    out = split_cdr_outputs(out, [x for x in lengths[:-1]])

                    if verbose:
                        stderr('\n\n')

                    self.set_predict_mode(False)

                    if dump:
                        response_keys = responses[:]

                        if partition and not partition.startswith('_'):
                            partition_str = '_' + partition
                        else:
                            partition_str = ''

                        for _response in response_keys:
                            file_ix = self.response_to_df_ix[_response]
                            multiple_files = len(file_ix) > 1
                            for ix in file_ix:
                                df = {}
                                if return_preds and _response in out['preds']:
                                    df['CDRpreds'] = out['preds'][_response][ix]
                                if return_loglik:
                                    df['CDRloglik'] = out['log_lik'][_response][ix]
                                if Y is not None and _response in Y[ix]:
                                    df['CDRobs'] = Y[ix][_response]
                                df = pd.DataFrame(df)
                                if extra_cols:
                                    if Y is None:
                                        df_new = {x: Y_gf_in[i] for i, x in enumerate(self.rangf)}
                                        df_new['time'] = Y_time_in[ix]
                                        df_new = pd.DataFrame(df_new)
                                    else:
                                        df_new = Y[ix]
                                    df = pd.concat([df.reset_index(drop=True), df_new.reset_index(drop=True)], axis=1)

                                if multiple_files:
                                    name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                                else:
                                    name_base = '%s%s' % (sn(_response), partition_str)
                                df.to_csv(self.outdir + '/CDRpreds_%s.csv' % name_base, sep=' ', na_rep='NaN',
                                          index=False)
        else:
            out = {}

        return out

    def log_lik(
            self,
            X,
            Y,
            sum_outputs_along_T=True,
            sum_outputs_along_K=True,
            dump=False,
            extra_cols=False,
            partition=None,
            **kwargs
    ):
        """
        Compute log-likelihood of data from predictive posterior.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y: ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables.`
        :param sum_outputs_along_T: ``bool``; whether to sum IRF-weighted predictors along the time dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for timestep-specific evaluation.
        :param sum_outputs_along_K: ``bool``; whether to sum IRF-weighted predictors along the predictor dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for impulse-specific evaluation.
        :param dump; ``bool``; whether to save generated log likelihood vectors to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param **kwargs; Any additional keyword arguments accepted by ``predict()`` (see docs for ``predict()`` for details).
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        assert not dump or (sum_outputs_along_T and sum_outputs_along_K), 'dump=True is only supported if sum_outputs_along_T=True and sum_outputs_along_K=True'

        out = self.predict(
            X,
            Y=Y,
            return_preds=False,
            return_loglik=True,
            sum_outputs_along_T=sum_outputs_along_T,
            sum_outputs_along_K=sum_outputs_along_K,
            dump=False,
            **kwargs
        )['log_lik']

        if dump:
            response_keys = list(out['log_lik'].keys())

            Y_gf = [_Y[self.rangf] for _Y in Y]
            Y_time = [_Y.time for _Y in Y]

            if partition and not partition.startswith('_'):
                partition_str = '_' + partition
            else:
                partition_str = ''

            for _response in response_keys:
                file_ix = self.response_to_df_ix[_response]
                multiple_files = len(file_ix) > 1
                for ix in file_ix:
                    df = {'CDRloglik': out[_response][ix]}
                    if extra_cols:
                        if Y is None:
                            df_new = {x: Y_gf[i] for i, x in enumerate(self.rangf)}
                            df_new['time'] = Y_time[ix]
                            df_new = pd.DataFrame(df_new)
                        else:
                            df_new = Y[ix]
                        df = pd.concat([df.reset_index(drop=True), df_new.reset_index(drop=True)], axis=1)

                    if multiple_files:
                        name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                    else:
                        name_base = '%s%s' % (sn(_response), partition_str)
                    df.to_csv(self.outdir + '/output_%s.csv' % name_base, sep=' ', na_rep='NaN', index=False)

        return out

    def loss(
            self,
            X,
            Y,
            X_in_Y_names=None,
            n_samples=None,
            algorithm='MAP',
            training=None,
            optimize_memory=False,
            verbose=True
    ):
        """
        Compute the elementsize loss over a dataset using the model's optimization objective.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y: ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param training: ``bool``; Whether to compute loss in training mode.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)
            stderr('Computing loss...\n')

        # Preprocess data
        if not isinstance(X, list):
            X = [X]
        X_in = X
        if Y is not None and not isinstance(Y, list):
            Y = [Y]
        lengths = [len(_Y) for _Y in Y]
        n = sum(lengths)
        Y_in = Y
        if X_in_Y_names:
            X_in_Y_names = [x for x in X_in_Y_names if x in self.impulse_names]

        Y, first_obs, last_obs, Y_time, Y_mask, Y_gf, X_in_Y = build_CDR_response_data(
            self.response_names,
            Y=Y_in,
            X_in_Y_names=X_in_Y_names,
            Y_category_map=self.response_category_to_ix,
            response_to_df_ix=self.response_to_df_ix,
            gf_names=self.rangf,
            gf_map=self.rangf_map
        )

        if not optimize_memory:
            X, X_time, X_mask = build_CDR_impulse_data(
                X_in,
                first_obs,
                last_obs,
                X_in_Y_names=X_in_Y_names,
                X_in_Y=X_in_Y,
                history_length=self.history_length,
                future_length=self.future_length,
                impulse_names=self.impulse_names,
                int_type=self.int_type,
                float_type=self.float_type,
            )

        with self.session.as_default():
            with self.session.graph.as_default():
                self.set_predict_mode(True)

                if training is None:
                    training = not self.predict_mode

                B = self.eval_minibatch_size
                n = sum([len(_Y) for _Y in Y])
                n_minibatch = math.ceil(n / B)
                loss = np.zeros((n,))
                for i in range(0, n, B):
                    if verbose:
                        stderr('\rMinibatch %d/%d' % (i + 1, n_minibatch))
                    if optimize_memory:
                        _Y = Y[i:i + B]
                        _first_obs = [x[i:i + B] for x in first_obs]
                        _last_obs = [x[i:i + B] for x in last_obs]
                        _Y_time = Y_time[i:i + B]
                        _Y_mask = Y_mask[i:i + B]
                        _Y_gf = None if Y_gf is None else Y_gf[i:i + B]
                        _X_in_Y = None if X_in_Y is None else X_in_Y[i:i + B]

                        _X, _X_time, _X_mask = build_CDR_impulse_data(
                            X_in,
                            _first_obs,
                            _last_obs,
                            X_in_Y_names=X_in_Y_names,
                            X_in_Y=_X_in_Y,
                            history_length=self.history_length,
                            future_length=self.future_length,
                            impulse_names=self.impulse_names,
                            int_type=self.int_type,
                            float_type=self.float_type,
                        )
                        _Y = None if Y is None else [_y[i:i + B] for _y in Y]
                        _Y_gf = None if Y_gf is None else Y_gf[i:i + B]

                        fd = {
                            'X': _X,
                            'X_time': _X_time,
                            'X_mask': _X_mask,
                            'Y': _Y,
                            'Y_time': _Y_time,
                            'Y_mask': _Y_mask,
                            'Y_gf': _Y_gf,
                            'training': not self.predict_mode
                        }
                    else:
                        fd = {
                            'X': X[i:i + B],
                            'X_time': X_time[i:i + B],
                            'X_mask': X_mask[i:i + B],
                            'Y_time': Y_time[i:i + B],
                            'Y_mask': Y_mask[i:i + B],
                            'Y_gf': None if Y_gf is None else Y_gf[i:i + B],
                            'Y': Y[i:i + B],
                            'training': training
                        }
                    loss[i:i + B] = self.run_loss_op(
                        fd,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                loss = loss.mean()

                if verbose:
                    stderr('\n\n')

                self.set_predict_mode(False)

                return loss

    def evaluate(
            self,
            X,
            Y,
            X_in_Y_names=None,
            n_samples=None,
            algorithm='MAP',
            return_preds=None,
            ks_test=False,
            sum_outputs_along_T=True,
            sum_outputs_along_K=True,
            dump=False,
            extra_cols=False,
            partition=None,
            optimize_memory=False,
            verbose=True
    ):
        """
        Compute and evaluate CDR model outputs relative to targets, optionally saving generated data and evaluations to disk.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y: ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs_<K>``:  Index in the Kth (zero-indexed) element of `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param return_preds: ``bool``; whether to return predictions as well as likelihoods. If ``None``, defaults are chosen based on response distribution(s).
        :param ks_test: ``bool``; whether to return results of a Kolmogorov-Smirnov test between empirical and modeled data distributions.
        :param sum_outputs_along_T: ``bool``; whether to sum IRF-weighted predictors along the time dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for timestep-specific evaluation.
        :param sum_outputs_along_K: ``bool``; whether to sum IRF-weighted predictors along the predictor dimension. Must be ``True`` for valid convolution. Setting to ``False`` is useful for impulse-specific evaluation.
        :param dump: ``bool``; whether to save generated data and evaluations to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: pair of <``dict``, ``str``>; Dictionary of evaluation metrics, human-readable evaluation summary string.
        """

        assert not dump or (sum_outputs_along_T and sum_outputs_along_K), 'dump=True is only supported if sum_outputs_along_T=True and sum_outputs_along_K=True'

        if partition and not partition.startswith('_'):
            partition_str = '_' + partition
        else:
            partition_str = ''

        if return_preds is None:
            return_preds = True
            for response in self.response_names:
                if self.get_response_dist_name(response) in ('sinharcsinh', 'johnsonsu'):
                    return_preds = False  # These distributions currently must bootstrap the mode, which is slow, so turned off by default.

        cdr_out = self.predict(
            X,
            Y=Y,
            X_in_Y_names=X_in_Y_names,
            n_samples=n_samples,
            algorithm=algorithm,
            return_preds=return_preds,
            return_loglik=True,
            sum_outputs_along_T=sum_outputs_along_T,
            sum_outputs_along_K=sum_outputs_along_K,
            dump=False,
            optimize_memory=optimize_memory,
            verbose=verbose
        )

        if return_preds:
            preds = cdr_out['preds']
        log_lik = cdr_out['log_lik']

        # Expand arrays to be B x T x K
        if return_preds:
            for response in preds:
                for ix in preds[response]:
                    arr = preds[response][ix]
                    while len(arr.shape) < 3:
                        arr = arr[..., None]
                    preds[response][ix] = arr
        for response in log_lik:
            for ix in log_lik[response]:
                arr = log_lik[response][ix]
                while len(arr.shape) < 3:
                    arr = arr[..., None]
                log_lik[response][ix] = arr

        if sum_outputs_along_T:
            T = 1
        else:
            T = self.history_length + self.future_length

        if sum_outputs_along_K:
            K = 1
        else:
            K = self.n_impulse

        metrics = {
            'mse': {},
            'rho': {},
            'f1': {},
            'f1_baseline': {},
            'acc': {},
            'acc_baseline': {},
            'log_lik': {},
            'percent_variance_explained': {},
            'true_variance': {},
            'ks_results': {},
            'full_log_lik': 0.
        }

        response_names = self.response_names[:]
        for _response in response_names:
            metrics['mse'][_response] = {}
            metrics['rho'][_response] = {}
            metrics['f1'][_response] = {}
            metrics['f1_baseline'][_response] = {}
            metrics['acc'][_response] = {}
            metrics['acc_baseline'][_response] = {}
            metrics['log_lik'][_response] = {}
            metrics['percent_variance_explained'][_response] = {}
            metrics['true_variance'][_response] = {}
            metrics['ks_results'][_response] = {}

            file_ix_all = list(range(len(Y)))
            file_ix = self.response_to_df_ix[_response]
            multiple_files = len(file_ix_all) > 1

            for ix in file_ix_all:
                metrics['mse'][_response][ix] = None
                metrics['rho'][_response][ix] = None
                metrics['f1'][_response][ix] = None
                metrics['f1_baseline'][_response][ix] = None
                metrics['acc'][_response][ix] = None
                metrics['acc_baseline'][_response][ix] = None
                metrics['log_lik'][_response][ix] = None
                metrics['percent_variance_explained'][_response][ix] = None
                metrics['true_variance'][_response][ix] = None
                metrics['ks_results'][_response][ix] = None

                if ix in file_ix:
                    _Y = Y[ix]
                    sel = None
                    if _response in _Y:
                        _y = _Y[_response]
                        dtype = _y.dtype
                        if dtype.name not in ('object', 'category') and np.issubdtype(dtype, np.number):
                            is_numeric = True
                        else:
                            is_numeric = False
                        if is_numeric:
                            sel = np.isfinite(_y)
                        else:
                            sel = np.ones(len(_y), dtype=bool)
                        index = _y.index[sel]
                        if return_preds:
                            _preds = preds[_response][ix][sel]
                        else:
                            _preds = None
                        _y = _y[sel]

                        resid_theoretical_q = None

                        if self.is_binary(_response):
                            baseline = np.ones((len(_y),))
                            metrics['f1_baseline'][_response][ix] = f1_score(_y, baseline, average='binary')
                            metrics['acc_baseline'][_response][ix] = accuracy_score(_y, baseline)
                            err_col_name = 'CDRcorrect'
                            for t in range(T):
                                for k in range(K):
                                    __preds = _preds[:, t, k]
                                    error = (_y == __preds).astype('int')
                                    if metrics['f1'][_response][ix] is None:
                                        metrics['f1'][_response][ix] = np.zeros((T, K))
                                    metrics['f1'][_response][ix][t, k] = f1_score(_y, __preds, average='binary')
                                    if metrics['acc'][_response][ix] is None:
                                        metrics['acc'][_response][ix] = np.zeros((T, K))
                                    metrics['acc'][_response][ix][t, k] = accuracy_score(_y, __preds)
                        elif self.is_categorical(_response):
                            classes, counts = np.unique(_y, return_counts=True)
                            majority = classes[np.argmax(counts)]
                            baseline = [majority] * len(_y)
                            metrics['f1_baseline'][_response][ix] = f1_score(_y, baseline, average='macro')
                            metrics['acc_baseline'][_response][ix] = accuracy_score(_y, baseline)
                            err_col_name = 'CDRcorrect'
                            for t in range(T):
                                for k in range(K):
                                    __preds = _preds[:, t, k]
                                    error = (_y == __preds).astype('int')
                                    if metrics['f1'][_response][ix] is None:
                                        metrics['f1'][_response][ix] = np.zeros((T, K))
                                    metrics['f1'][_response][ix][t, k] = f1_score(_y, __preds, average='macro')
                                    if metrics['acc'][_response][ix] is None:
                                        metrics['acc'][_response][ix] = np.zeros((T, K))
                                    metrics['acc'][_response][ix][t, k] = accuracy_score(_y, __preds)
                        else:
                            err_col_name = 'CDRsquarederror'
                            metrics['true_variance'][_response][ix] = np.std(_y) ** 2
                            for t in range(T):
                                for k in range(K):
                                    if return_preds:
                                        __preds = _preds[:, t, k]
                                        error = np.array(_y - __preds) ** 2
                                        score = error.mean()
                                        resid = np.sort(_y - __preds)
                                        if ks_test:
                                            if self.error_distribution_theoretical_quantiles[_response] is None:
                                                resid_theoretical_q = None
                                            else:
                                                resid_theoretical_q = self.error_theoretical_quantiles(len(resid), _response)
                                                valid = np.isfinite(resid_theoretical_q)
                                                resid = resid[valid]
                                                resid_theoretical_q = resid_theoretical_q[valid]
                                            D, p_value = self.error_ks_test(resid, _response)
    
                                        if metrics['mse'][_response][ix] is None:
                                            metrics['mse'][_response][ix] = np.zeros((T, K))
                                        metrics['mse'][_response][ix][t, k] = score
                                        if metrics['rho'][_response][ix] is None:
                                            metrics['rho'][_response][ix] = np.zeros((T, K))
                                        metrics['rho'][_response][ix][t, k] = np.corrcoef(_y, __preds, rowvar=False)[0, 1]
                                        if metrics['percent_variance_explained'][_response][ix] is None:
                                            metrics['percent_variance_explained'][_response][ix] = np.zeros((T, K))
                                        metrics['percent_variance_explained'][_response][ix][
                                            t, k] = percent_variance_explained(
                                            _y, __preds)
                                        if ks_test:
                                            if metrics['ks_results'][_response][ix] is None:
                                                metrics['ks_results'][_response][ix] = (np.zeros((T, K)), np.zeros((T, K)))
                                            metrics['ks_results'][_response][ix][0][t, k] = D
                                            metrics['ks_results'][_response][ix][1][t, k] = p_value
                                    else:
                                        error = __preds = None
                    else:
                        err_col_name = error = __preds = _y = None

                    _ll = log_lik[_response][ix]
                    if sel is not None:
                        if __preds is not None:
                            __preds = pd.Series(__preds, index=index)
                        _ll = _ll[sel]
                        _ll = pd.Series(np.squeeze(_ll), index=index)
                        if err_col_name is not None and error is not None:
                            error = pd.Series(error, index=index)
                    _ll_summed = _ll.sum(axis=0)
                    metrics['log_lik'][_response][ix] = _ll_summed
                    metrics['full_log_lik'] += _ll_summed

                    if dump:
                        if multiple_files:
                            name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                        else:
                            name_base = '%s%s' % (sn(_response), partition_str)

                        df = pd.DataFrame()
                        if err_col_name is not None and error is not None:
                            df[err_col_name] = error
                        if __preds is not None:
                            df['CDRpreds'] = __preds
                        if _y is not None:
                            df['CDRobs'] = _y
                        df['CDRloglik'] = _ll

                        if extra_cols:
                            df = pd.concat([_Y.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

                        preds_outfile = self.outdir + '/output_%s.csv' % name_base
                        df.to_csv(preds_outfile, sep=' ', na_rep='NaN', index=False)

                        if _response in self.response_distribution_config and \
                                self.is_real(_response) and resid_theoretical_q is not None:
                            plot_qq(
                                resid_theoretical_q,
                                resid,
                                outdir=self.outdir,
                                filename='error_qq_plot_%s.png' % name_base,
                                xlab='Theoretical',
                                ylab='Empirical'
                            )

        summary = ''
        if sum_outputs_along_T and sum_outputs_along_K:
            summary_header = '=' * 50 + '\n'
            summary_header += 'CDR regression\n\n'
            summary_header += 'Model name: %s\n\n' % self.name
            summary_header += 'Formula:\n'
            summary_header += '  ' + self.form_str + '\n\n'
            summary_header += 'Partition: %s\n' % partition
            summary_header += 'Training iterations completed: %d\n\n' % self.global_step.eval(session=self.session)
            summary_header += 'Full log likelihood: %s\n\n' % np.squeeze(metrics['full_log_lik'])

            summary += summary_header

            for _response in response_names:
                file_ix = self.response_to_df_ix[_response]
                multiple_files = len(file_ix) > 1
                for ix in file_ix:
                    summary += 'Response variable: %s\n\n' % _response
                    _summary = summary_header
                    _summary += 'Response variable: %s\n\n' % _response

                    if multiple_files:
                        summary += 'File index: %s\n\n' % ix
                        name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                        _summary += 'File index: %s\n\n' % ix
                    else:
                        name_base = '%s%s' % (sn(_response), partition_str)

                    summary_eval = self.report_evaluation(
                        mse=metrics['mse'][_response][ix],
                        f1=metrics['f1'][_response][ix],
                        f1_baseline=metrics['f1_baseline'][_response][ix],
                        acc=metrics['acc'][_response][ix],
                        acc_baseline=metrics['acc_baseline'][_response][ix],
                        rho=metrics['rho'][_response][ix],
                        loglik=metrics['log_lik'][_response][ix],
                        percent_variance_explained=metrics['percent_variance_explained'][_response][ix],
                        true_variance=metrics['true_variance'][_response][ix],
                        ks_results=metrics['ks_results'][_response][ix]
                    )

                    summary += summary_eval
                    _summary += summary_eval
                    _summary += '=' * 50 + '\n'
                    with open(self.outdir + '/eval_%s.txt' % name_base, 'w') as f_out:
                        f_out.write(_summary)

            summary += '=' * 50 + '\n'
            if verbose:
                stderr(summary)
                stderr('\n\n')

        return metrics, summary

    def convolve_inputs(
            self,
            X,
            Y=None,
            first_obs=None,
            last_obs=None,
            Y_time=None,
            Y_gf=None,
            responses=None,
            response_params=None,
            X_in_Y_names=None,
            X_in_Y=None,
            n_samples=None,
            algorithm='MAP',
            extra_cols=False,
            dump=False,
            partition=None,
            optimize_memory=False,
            verbose=True
    ):
        """
        Convolve input data using the fitted CDR(NN) model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param Y (optional): ``list`` of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            This parameter is optional and responses are not directly used. It simply allows the user to omit the
            inputs **Y_time**, **Y_gf**, **first_obs**, and **last_obs**, since they can be inferred from **Y**
            If supplied, each element of **Y** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **y**
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model formula

        :param first_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of first observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the first impulse in the time series associated with each response. If ``None``, inferred from **Y**.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of last observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the last impulse in the time series associated with each response. If ``None``, inferred from **Y**.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param Y_time: ``list`` of response timestamp vectors (``list``, ``pandas`` series, or ``numpy`` vector); vector(s) of response timestamps, one for each response array. Needed to timestamp any response-aligned predictors (ignored if none in model).
        :param Y_gf: ``list`` of random grouping factor values (``list``, ``pandas`` series, or ``numpy`` vector); random grouping factor values (if applicable), one for each response dataframe.
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to convolve toward. If ``None``, convolves toward all univariate responses. Multivariate convolution (e.g. of categorical responses) is supported but turned off by default to avoid excessive computation. When convolving toward a multivariate response, a set of convolved predictors will be generated for each dimension of the response.
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of response distribution(s) to convolve toward per response variable. Any param names not used by the response distribution for a given response will be ignored. If ``None``, convolves toward the first parameter of each response distribution.
        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param X_in_Y: ``list`` of ``pandas`` ``DataFrame`` or ``None``; tables (one per response array) of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, inferred from **Y** and **X_in_Y_names**.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables.
        :param dump; ``bool``; whether to save generated log likelihood vectors to disk.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)
            stderr('Computing convolutions...\n')

        if partition and not partition.startswith('_'):
            partition_str = '_' + partition
        else:
            partition_str = ''

        if responses is None:
            responses = [x for x in self.response_names if self.get_response_ndim(x) == 1]
        if isinstance(responses, str):
            responses = [responses]

        if response_params is None:
            response_params = set()
            for _response in responses:
                response_params.add(self.get_response_params(_response)[0])
            response_params = sorted(list(response_params))
        if isinstance(response_params, str):
            response_params = [response_params]

        # Preprocess data
        if not isinstance(X, list):
            X = [X]
        X_in = X
        if Y is None:
            assert Y_time is not None, 'Either Y or Y_time must be provided.'
            lengths = [len(_Y_time) for _Y_time in Y_time]
        else:
            if not isinstance(Y, list):
                Y = [Y]
            lengths = [len(_Y) for _Y in Y]
        n = sum(lengths)
        Y_in = Y
        if Y_time is None:
            Y_time_in = [_Y.time for _Y in Y]
        else:
            Y_time_in = Y_time
        if Y_gf is None:
            assert Y is not None, 'Either Y or Y_gf must be provided.'
            Y_gf_in = Y
        else:
            Y_gf_in = Y_gf
        if X_in_Y_names:
            X_in_Y_names = [x for x in X_in_Y_names if x in self.impulse_names]
        X_in_Y_in = X_in_Y

        Y, first_obs, last_obs, Y_time, Y_mask, Y_gf, X_in_Y = build_CDR_response_data(
            self.response_names,
            Y=Y_in,
            first_obs=first_obs,
            last_obs=last_obs,
            Y_gf=Y_gf_in,
            X_in_Y_names=X_in_Y_names,
            X_in_Y=X_in_Y_in,
            Y_category_map=self.response_category_to_ix,
            response_to_df_ix=self.response_to_df_ix,
            gf_names=self.rangf,
            gf_map=self.rangf_map
        )

        if not optimize_memory or not np.isfinite(self.minibatch_size):
            X, X_time, X_mask = build_CDR_impulse_data(
                X_in,
                first_obs,
                last_obs,
                X_in_Y_names=X_in_Y_names,
                X_in_Y=X_in_Y,
                history_length=self.history_length,
                future_length=self.future_length,
                impulse_names=self.impulse_names,
                int_type=self.int_type,
                float_type=self.float_type,
            )

        with self.session.as_default():
            with self.session.graph.as_default():
                self.set_predict_mode(True)
                B = self.eval_minibatch_size
                n_eval_minibatch = math.ceil(n / B)
                X_conv = {}
                for _response in responses:
                    X_conv[_response] = {}
                    for _response_param in response_params:
                        dim_names = self.expand_param_name(_response, _response_param)
                        for _dim_name in dim_names:
                            X_conv[_response][_dim_name] = np.zeros(
                                (n, len(self.terminal_names))
                            )
                for i in range(0, n, B):
                    if verbose:
                        stderr('\rMinibatch %d/%d' % ((i / B) + 1, n_eval_minibatch))
                    if optimize_memory:
                        _Y = None if Y is None else Y[i:i + B]
                        _first_obs = [x[i:i + B] for x in first_obs]
                        _last_obs = [x[i:i + B] for x in last_obs]
                        _Y_time = Y_time[i:i + B]
                        _Y_mask = Y_mask[i:i + B]
                        _Y_gf = None if Y_gf is None else Y_gf[i:i + B]
                        _X_in_Y = None if X_in_Y is None else X_in_Y[i:i + B]

                        _X, _X_time, _X_mask = build_CDR_impulse_data(
                            X_in,
                            _first_obs,
                            _last_obs,
                            X_in_Y_names=X_in_Y_names,
                            X_in_Y=_X_in_Y,
                            history_length=self.history_length,
                            future_length=self.future_length,
                            impulse_names=self.impulse_names,
                            int_type=self.int_type,
                            float_type=self.float_type,
                        )
                        fd = {
                            self.X: _X,
                            self.X_time: _X_time,
                            self.X_mask: _X_mask,
                            self.Y_time: _Y_time,
                            self.Y_mask: _Y_mask,
                            self.Y_gf: _Y_gf,
                            self.training: not self.predict_mode
                        }
                    else:
                        fd = {
                            self.X: X[i:i + B],
                            self.X_time: X_time[i:i + B],
                            self.X_mask: X_mask[i:i + B],
                            self.Y_time: Y_time[i:i + B],
                            self.Y_mask: Y_mask[i:i + B],
                            self.Y_gf: None if Y_gf is None else Y_gf[i:i + B],
                            self.training: not self.predict_mode
                        }
                    if verbose:
                        stderr('\rMinibatch %d/%d' % ((i / B) + 1, n_eval_minibatch))
                    _X_conv = self.run_conv_op(
                        fd,
                        responses=responses,
                        response_param=response_params,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                    for _response in _X_conv:
                        for _dim_name in _X_conv[_response]:
                            _X_conv_batch = _X_conv[_response][_dim_name]
                            X_conv[_response][_dim_name][i:i + B] = _X_conv_batch

                # Split into per-file predictions.
                # Exclude the length of last file because it will be inferred.
                X_conv = split_cdr_outputs(X_conv, [x for x in lengths[:-1]])

                if verbose:
                    stderr('\n\n')

                self.set_predict_mode(False)

                out = {}
                names = []
                for x in self.terminal_names:
                    if self.node_table[x].p.irfID is None:
                        names.append(sn(''.join(x.split('-')[:-1])))
                    else:
                        names.append(sn(x))
                for _response in responses:
                    out[_response] = {}
                    file_ix = self.response_to_df_ix[_response]
                    multiple_files = len(file_ix) > 1
                    for ix in file_ix:
                        for dim_name in X_conv[_response]:
                            if dim_name not in out[_response]:
                                out[_response][dim_name] = []

                            df = pd.DataFrame(X_conv[_response][dim_name][ix], columns=names, dtype=self.FLOAT_NP)
                            if extra_cols:
                                if Y is None:
                                    df_extra = {x: Y_gf_in[i] for i, x in enumerate(self.rangf)}
                                    df_extra['time'] = Y_time_in[ix]
                                    df_extra = pd.DataFrame(df_extra)
                                else:
                                    new_cols = []
                                    for c in Y[ix].columns:
                                        if c not in df:
                                            new_cols.append(c)
                                    df_extra = Y[ix][new_cols].reset_index(drop=True)
                                df = pd.concat([df, df_extra], axis=1)
                            out[_response][dim_name].append(df)

                        if dump:
                            if multiple_files:
                                name_base = '%s_%s_f%s%s' % (sn(_response), sn(dim_name), ix, partition_str)
                            else:
                                name_base = '%s_%s%s' % (sn(_response), sn(dim_name), partition_str)
                            df.to_csv(self.outdir + '/X_conv_%s.csv' % name_base, sep=' ', na_rep='NaN', index=False)

                return out

    def error_theoretical_quantiles(
            self,
            n_errors,
            response
    ):
        with self.session.as_default():
            with self.session.graph.as_default():
                self.set_predict_mode(True)
                fd = {
                    self.n_errors[response]: n_errors,
                    self.training: not self.predict_mode
                }
                err_q = self.session.run(self.error_distribution_theoretical_quantiles[response], feed_dict=fd)
                self.set_predict_mode(False)

                return err_q

    def error_theoretical_cdf(
            self,
            errors,
            response
    ):
        with self.session.as_default():
            with self.session.graph.as_default():
                fd = {
                    self.errors[response]: errors,
                    self.training: not self.predict_mode
                }
                err_cdf = self.session.run(self.error_distribution_theoretical_cdf[response], feed_dict=fd)

                return err_cdf

    def error_ks_test(
            self,
            errors,
            response
    ):
        with self.session.as_default():
            with self.session.graph.as_default():
                err_cdf = self.error_theoretical_cdf(errors, response)

                D, p_value = scipy.stats.kstest(errors, lambda x: err_cdf)

                return D, p_value

    def get_plot_data(
            self,
            xvar='t_delta',
            yvar=None,
            responses=None,
            response_params=None,
            X_ref=None,
            X_time_ref=None,
            t_delta_ref=None,
            gf_y_ref=None,
            ref_varies_with_x=False,
            ref_varies_with_y=False,
            manipulations=None,
            pair_manipulations=False,
            include_interactions=False,
            reference_type=None,
            plot_quantile_range=0.9,
            xaxis=None,
            xmin=None,
            xmax=None,
            xres=None,
            yaxis=None,
            ymin=None,
            ymax=None,
            yres=None,
            n_samples=None,
            level=95
    ):
        """
        Compute arrays of plot data by passing input manipulations through the model, relative to a reference input.
        The reference can be a point, a matrix evolving over one of the plot axes, or (in the case of 3d plots) a
        tensor evolving over both axes. The response to the reference is subtracted from the responses to the remaining
        variations, so responses to manipulations represent deviation from the reference response.

        The final dimension of return arrays will have size ``len(manipulations) + 1``. If the reference
        varies with all input axes, the first element of the final dimension will be the reference response. Otherwise,
        the first element of the final dimension will be the un-manipulated covariate. All post-initial elements of the
        final dimension will be the responses to manipulations, in the order provided.

        This method supports a large space of queries. Any continuous input variable can be provided as an axis,
        including all predictors (impulses), as well as ``'rate'``, ``'X_time'``, and ``'t_delta'``, respectively the
        deconvolutional intercept, the stimulus timestamp, and the delay from stimulus onset (i.e. the input to the
        IRF). The **manipulations** parameter supports arbitrary lambda functions on any combination of these variables,
        as well as on the random effects levels. Values for all of these variables can also be set for the reference
        response, enabling comparison to arbitrary references.

        Note that most of these queries are only of interest for CDRNN, since CDR assumes their structure (e.g.
        additive effects and non-stationarity). For CDR, the primary estimate of interest (the IRF) can be obtained by
        setting ``xvar = 't_delta'``, using a zero-vectored reference, and constructing a list of manipulations that
        adds ``1`` to each of the predictors independently.

        :param xvar: ``str``; Name of continuous variable for x-axis. Can be a predictor (impulse), ``'rate'``, ``'t_delta'``, or ``'X_time'``.
        :param yvar: ``str``; Name of continuous variable for y-axis in 3D plots. Can be a predictor (impulse), ``'rate'``, ``'t_delta'``, or ``'X_time'``. If ``None``, 2D plot.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to plot.
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of response distribution(s) to plot per response variable. Any param names not used by the response distribution for a given response will be ignored.
        :param X_ref: ``dict`` or ``None``; Dictionary mapping impulse names to numeric values for use in constructing the reference. Any impulses not specified here will take default values.
        :param X_time_ref: ``float`` or ``None``; Timestamp to use for constructing the reference. If ``None``, use default value.
        :param t_delta_ref: ``float`` or ``None``; Delay/offset to use for constructing the reference. If ``None``, use default value.
        :param gf_y_ref: ``dict`` or ``None``; Dictionary mapping random grouping factor names to random grouping factor levels for use in constructing the reference. Any random effects not specified here will take default values.
        :param ref_varies_with_x: ``bool``; Whether the reference varies along the x-axis. If ``False``, use the scalar reference value for the x-axis.
        :param ref_varies_with_y: ``bool``; Whether the reference varies along the y-axis. If ``False``, use the scalar reference value for the y-axis. Ignored if **yvar** is ``None``.
        :param manipulations: ``list`` of ``dict``; A list of manipulations, where each manipulation is constructed as a dictionary mapping a variable name (e.g. ``'predictorX'``, ``'t_delta'``) to either a float offset or a function that transforms the reference value for that variable (e.g. multiplies it by ``2``). Alternatively, the keyword ``'ranef'`` can be used to manipulate random effects. The ``'ranef'`` entry must map to a ``dict`` that itself maps random grouping factor names (e.g. ``'subject'``) to levels (e.g. ``'subjectA'``).
        :param pair_manipulations: ``bool``; Whether to apply the manipulations to the reference input. If ``False``, all manipulations are compared to the same reference. For example, when plotting by-subject IRFs by subject, each subject might have a difference base response. In this case, set **pair_manipulations** to ``True`` in order to match the random effects used to compute the reference response and the response of interest.
        :param include_interactions: ``bool``; Whether to include interaction terms in plotting the influence of a manipulation.
        :param reference_type: ``bool``; Type of reference to use. If ``0``, use a zero-valued reference. If ``'mean'``, use the training set mean for all variables. If ``None``, use the default reference vector specified in the model's configuration file.
        :param plot_quantile_range: ``float``; Quantile range to use for plotting. E.g., 0.9 uses the interdecile range.
        :param xaxis: ``list``, ``numpy`` vector, or ``None``; Vector of values to use for the x-axis. If ``None``, inferred.
        :param xmin: ``float`` or ``None``; Minimum value for x-axis (if axis inferred). If ``None``, inferred.
        :param xmax: ``float`` or ``None``; Maximum value for x-axis (if axis inferred). If ``None``, inferred.
        :param xres: ``int`` or ``None``; Resolution (number of plot points) on x-axis. If ``None``, inferred.
        :param yaxis: `list``, ``numpy`` vector, or ``None``; Vector of values to use for the y-axis. If ``None``, inferred.
        :param ymin: ``float`` or ``None``; Minimum value for y-axis (if axis inferred). If ``None``, inferred.
        :param ymax: ``float`` or ``None``; Maximum value for y-axis (if axis inferred). If ``None``, inferred.
        :param yres: ``int`` or ``None``; Resolution (number of plot points) on y-axis. If ``None``, inferred.
        :param n_samples: ``int`` or ``None``; Number of plot samples to draw for computing intervals. If ``None``, ``0``, ``1``, or if the model type does not support uncertainty estimation, the maximum likelihood estimate will be returned.
        :param level: ``float``; The confidence level of any intervals (i.e. ``95`` indicates 95% confidence/credible intervals).
        :return: 5-tuple (plot_axes, mean, lower, upper, samples); Let RX, RY, S, and K respectively be the x-axis resolution, y-axis resolution, number of samples, and number of output dimensions (manipulations). If plot is 2D, ``plot_axes`` is an array with shape ``(RX,)``, ``mean``, ``lower``, and ``upper`` are dictionaries of arrays with shape ``(RX, K)``, one for each **response_param** of each **response**,  and ``samples is a dictionary of arrays with shape ``(S, RX, K)``,  one for each **response_param** of each **response**. If plot is 3D, ``plot_axes`` is a pair of arrays each with shape ``(RX, RY)`` (i.e. a meshgrid), ``mean``, ``lower``, and ``upper`` are dictionaries of arrays with shape ``(RX, RY, K)``, one for each **response_param** of each **response**, and ``samples`` is a dictionary of arrays with shape ``(S, RX, RY, K)``, one for each **response_param** of each **response**.
        """

        assert xvar is not None, 'Value must be provided for xvar'
        assert xvar != yvar, 'Cannot vary two axes along the same variable'

        if level is None:
            level = 95

        plot_quantile_ix = int((1 - plot_quantile_range) / 2 * self.N_QUANTILES)

        if responses is None:
            if self.n_response == 1:
                responses = self.response_names
            else:
                responses = [x for x in self.response_names if self.get_response_ndim(x) == 1]
        if isinstance(responses, str):
            responses = [responses]

        if response_params is None:
            response_params = set()
            for response in responses:
                if self.has_analytical_mean[response]:
                    response_params.add('mean')
                else:
                    response_params.add(self.get_response_params(response)[0])
            response_params = sorted(list(response_params))
        if isinstance(response_params, str):
            response_params = [response_params]
            
        is_3d = yvar is not None
        if manipulations is None:
            manipulations = []

        if xaxis is None:
            if is_3d:
                if xres is None:
                    xres = 32
            else:
                if xres is None:
                    xres = 1024
            xvar_base = np.linspace(0., 1., xres)
        else:
            xres = len(xaxis)

        if is_3d:
            if yaxis is None:
                if yres is None:
                    yres = 32
                yvar_base = np.linspace(0., 1., yres)
            else:
                yres = len(yaxis)

            T = xres * yres
        else:
            T = xres

        if n_samples and (self.is_bayesian or self.has_dropout):
            resample = True
            S = n_samples
        else:
            resample = False
            S = 1

        ref_as_manip = ref_varies_with_x and (not is_3d or ref_varies_with_y)  # Only return ref as manip if it fully varies along all axes

        n_impulse = len(self.impulse_names)
        n_manip = int(not ref_as_manip) + len(manipulations) # If ref is not returned, return default variation as first manip
        assert not (ref_as_manip and pair_manipulations), "Cannot both vary reference along all axes and pair manipulations, since doing so will cause all responses to cancel."

        if is_3d:
            sample_shape = (xres, yres, n_manip)
            if pair_manipulations:
                ref_shape = sample_shape
                B_ref = T
            elif ref_varies_with_x or ref_varies_with_y:
                ref_shape = (xres, yres, 1)
                B_ref = T
            else:
                ref_shape = tuple()
                B_ref = 1
        else:
            sample_shape = (T, n_manip)
            if pair_manipulations:
                ref_shape = sample_shape
                B_ref = T
            elif ref_varies_with_x:
                ref_shape = (T, 1)
                B_ref = T
            else:
                ref_shape = tuple()
                B_ref = 1

        # Initialize predictor reference
        if reference_type is None:
            X_ref_arr = np.copy(self.reference_arr)
        elif reference_type == 'mean':
            X_ref_arr = np.copy(self.impulse_means_arr)
        else:
            X_ref_arr = np.zeros_like(self.reference_arr)
        if X_ref is None:
            X_ref = {}
        for x in X_ref:
            ix = self.impulse_names_to_ix[x]
            X_ref_arr[ix] = X_ref[x]
        X_ref = X_ref_arr[None, None, ...]

        # Initialize timestamp reference
        if X_time_ref is None:
            X_time_ref = self.X_time_mean
        assert np.isscalar(X_time_ref), 'X_time_ref must be a scalar'
        X_time_ref = np.reshape(X_time_ref, (1, 1, 1))
        X_time_ref = np.tile(X_time_ref, [1, 1, max(n_impulse, 1)])

        # Initialize offset reference
        if t_delta_ref is None:
            t_delta_ref = self.reference_time
        assert np.isscalar(t_delta_ref), 't_delta_ref must be a scalar'
        t_delta_ref = np.reshape(t_delta_ref, (1, 1, 1))
        t_delta_ref = np.tile(t_delta_ref, [1, 1, max(n_impulse, 1)])

        # Initialize random effects reference
        gf_y_ref_arr = np.copy(self.gf_defaults)
        if gf_y_ref is None:
            gf_y_ref = []
        for x in gf_y_ref:
            if x is not None:
                if isinstance(x, str):
                    g_ix = self.ranef_group2ix[x]
                else:
                    g_ix = x
                val = gf_y_ref[x]
                if isinstance(val, str):
                    l_ix = self.ranef_level2ix[x][val]
                else:
                    l_ix = val
                gf_y_ref_arr[0, g_ix] = l_ix
        gf_y_ref = gf_y_ref_arr

        # Construct x-axis manipulation
        xdict = {
            'axis_var': xvar,
            'axis': xaxis,
            'ax_min': xmin,
            'ax_max': xmax,
            'base': xvar_base,
            'ref_varies': ref_varies_with_x,
            'tile_3d': None
        }
        params = [xdict]
        
        if is_3d:
            xdict['tile_3d'] = [1, yres, 1]
            
            ydict = {
                'axis_var': yvar,
                'axis': yaxis,
                'ax_min': ymin,
                'ax_max': ymax,
                'base': yvar_base,
                'ref_varies': ref_varies_with_y,
                'tile_3d': [xres, 1, 1]
            }
            params.append(ydict)

        plot_axes = []

        X_base = None
        X_time_base = None
        t_delta_base = None
        X_ref_mask = np.ones(self.n_impulse)
        X_main_mask = np.ones(self.n_impulse)

        for par in params:
            axis_var = par['axis_var']
            axis = par['axis']
            ax_min = par['ax_min']
            ax_max = par['ax_max']
            base = par['base']
            ref_varies = par['ref_varies']
            tile_3d = par['tile_3d']
            plot_axis = None

            if X_base is None:
                X_base = np.tile(X_ref, (T, 1, 1))
            if X_time_base is None:
                X_time_base = np.tile(X_time_ref, (T, 1, 1))
            if t_delta_base is None:
                t_delta_base = np.tile(t_delta_ref, (T, 1, 1))

            if axis_var in self.impulse_names_to_ix:
                ix = self.impulse_names_to_ix[axis_var]
                X_main_mask[ix] = 0
                if ref_varies:
                    X_ref_mask[ix] = 0
                if axis is None:
                    qix = plot_quantile_ix
                    lq = self.impulse_quantiles_arr[qix][ix]
                    uq = self.impulse_quantiles_arr[self.N_QUANTILES - qix - 1][ix]
                    select = np.isclose(uq - lq, 0)
                    while qix > 0 and np.any(select):
                        qix -= 1
                        lq = self.impulse_quantiles_arr[qix][ix]
                        uq = self.impulse_quantiles_arr[self.N_QUANTILES - qix - 1][ix]
                        select = np.isclose(uq - lq, 0)
                    if np.any(select):
                        lq = lq - self.epsilon
                        uq = uq + self.epsilon
                    if ax_min is None:
                        ax_min = lq
                    if ax_max is None:
                        ax_max = uq
                    axis = (base * (ax_max - ax_min) + ax_min)
                else:
                    axis = np.array(axis)
                assert len(axis.shape) == 1, 'axis must be a (1D) vector. Got a tensor of rank %d.' % len(axis.shape)
                plot_axis = axis
                plot_axes.append(axis)
                X_delta = np.pad(axis[..., None, None] - X_ref[0, 0, ix], ((0, 0), (0, 0), (ix, n_impulse - (ix + 1))))
                if is_3d:
                    X_delta = np.tile(X_delta, tile_3d).reshape((T, 1, max(n_impulse, 1)))
                X_base += X_delta
                if ref_varies:
                    X_ref = X_ref + X_delta

            if axis_var == 'X_time':
                if axis is None:
                    if ax_min is None:
                        ax_min = 0.
                    if ax_max is None:
                        ax_max = self.X_time_mean + self.X_time_sd
                    axis = (base * (ax_max - ax_min) + ax_min)
                else:
                    axis = np.array(axis)
                assert len(axis.shape) == 1, 'axis must be a (1D) vector. Got a tensor of rank %d.' % len(axis.shape)
                plot_axis = axis
                plot_axes.append(axis)
                X_time_base = np.tile(axis[..., None, None], (1, 1, max(n_impulse, 1)))
                if is_3d:
                    X_time_base = np.tile(X_time_base, tile_3d).reshape((T, 1, max(n_impulse, 1)))
                if ref_varies:
                    X_time_ref = X_time_base

            if axis_var == 't_delta':
                if axis is None:
                    xinterval = self.plot_n_time_units
                    if ax_min is None:
                        ax_min = -xinterval * self.prop_fwd
                    if ax_max is None:
                        ax_max = xinterval * self.prop_bwd
                    axis = (base * (ax_max - ax_min) + ax_min)
                else:
                    axis = np.array(axis)
                assert len(axis.shape) == 1, 'axis must be a (1D) vector. Got a tensor of rank %d.' % len(axis.shape)
                plot_axis = axis
                plot_axes.append(axis)
                t_delta_base = np.tile(axis[..., None, None], (1, 1, max(n_impulse, 1)))
                if is_3d:
                    t_delta_base = np.tile(t_delta_base, tile_3d).reshape((T, 1, max(n_impulse, 1)))
                if ref_varies:
                    t_delta_ref = t_delta_base

            assert plot_axis is not None, 'Unrecognized value for axis variable: "%s"' % axis_var

        gf_y_base = np.tile(gf_y_ref, (T, 1))
        if ref_varies:
            gf_y_ref = gf_y_base

        if is_3d:
            plot_axes = np.meshgrid(*plot_axes)
        else:
            plot_axes = plot_axes[0]

        # Bring reference arrays into conformable shape
        if X_ref.shape[0] == 1 and B_ref > 1:
            X_ref = np.tile(X_ref, (B_ref, 1, 1))
        if X_time_ref.shape[0] == 1 and B_ref > 1:
            X_time_ref = np.tile(X_time_ref, (B_ref, 1, 1))
        if t_delta_ref.shape[0] == 1 and B_ref > 1:
            t_delta_ref = np.tile(t_delta_ref, (B_ref, 1, 1))
        if gf_y_ref.shape[0] == 1 and B_ref > 1:
            gf_y_ref = np.tile(gf_y_ref, (B_ref, 1))

        # The reference will contain 1 entry if not pair_manipulations and len(manipulations) + 1 entries otherwise
        X_ref_in = [X_ref]
        X_time_ref_in = [X_time_ref]
        t_delta_ref_in = [t_delta_ref]
        gf_y_ref_in = [gf_y_ref]

        if ref_as_manip: # Entails not pair_manipulations
            X = []
            X_time = []
            t_delta = []
            gf_y = []
        else: # Ref doesn't vary along all axes, so *_base contains full variation along all axes and is returned as the first manip
            X = [X_base]
            X_time = [X_time_base]
            t_delta = [t_delta_base]
            gf_y = [gf_y_base]

        for manipulation in manipulations:
            X_cur = None
            X_time_cur = X_time_base
            t_delta_cur = t_delta_base
            gf_y_cur = gf_y_base

            if pair_manipulations:
                X_ref_cur = None
                X_time_ref_cur = X_time_ref
                t_delta_ref_cur = t_delta_ref
                gf_y_ref_cur = gf_y_ref

            for k in manipulation:
                if isinstance(manipulation[k], float) or isinstance(manipulation[k], int):
                    manip = lambda x, offset=float(manipulation[k]): x + offset
                elif manipulation[k] == 'sd':
                    manip = lambda x, offset=self.impulse_sds[k]: x + offset
                else:
                    manip = manipulation[k]
                if k in self.impulse_names_to_ix:
                    if X_cur is None:
                        X_cur = np.copy(X_base)
                    ix = self.impulse_names_to_ix[k]
                    X_cur[..., ix] = manip(X_cur[..., ix])
                    if pair_manipulations:
                        if X_ref_cur is None:
                            X_ref_cur = np.copy(X_ref)
                        X_ref_cur[..., ix] = manip(X_ref_cur[..., ix])
                elif k == 'X_time':
                    X_time_cur = manip(X_time_cur)
                    if pair_manipulations:
                        X_time_ref_cur = X_time_cur
                elif k == 't_delta':
                    t_delta_cur = manip(t_delta_cur)
                    if pair_manipulations:
                        t_delta_ref_cur = t_delta_cur
                elif k == 'ranef':
                    gf_y_cur = np.copy(gf_y_cur)
                    if gf_y_ref is None:
                        gf_y_ref = []
                    for x in manip:
                        if x is not None:
                            if isinstance(x, str):
                                g_ix = self.ranef_group2ix[x]
                            else:
                                g_ix = x
                            val = manip[x]
                            if isinstance(val, str):
                                l_ix = self.ranef_level2ix[x][val]
                            else:
                                l_ix = val
                            gf_y_cur[:, g_ix] = l_ix
                    if pair_manipulations:
                        gf_y_ref_cur = gf_y_cur
                else:
                    raise ValueError('Unrecognized manipulation key: "%s"' % k)

            if X_cur is None:
                X_cur = X_base
            X.append(X_cur)
            X_time.append(X_time_cur)
            t_delta.append(t_delta_cur)
            gf_y.append(gf_y_cur)

            if pair_manipulations:
                if X_ref_cur is None:
                    X_ref_cur = X_ref
                X_ref_in.append(X_ref_cur)
                X_time_ref_in.append(X_time_ref_cur)
                t_delta_ref_in.append(t_delta_ref_cur)
                gf_y_ref_in.append(gf_y_ref_cur)

        X_ref_in = np.concatenate(X_ref_in, axis=0)
        X_time_ref_in = np.concatenate(X_time_ref_in, axis=0)
        X_time_ref_in = X_time_ref_in[..., :X_ref_in.shape[-1]]    # Trim in case there are no impulses in the model
        X_mask_ref_in = np.ones_like(X_time_ref_in)
        t_delta_ref_in = np.concatenate(t_delta_ref_in, axis=0)
        t_delta_ref_in = t_delta_ref_in[..., :X_ref_in.shape[-1]]  # Trim in case there are no impulses in the model
        gf_y_ref_in = np.concatenate(gf_y_ref_in, axis=0)

        # Bring manipulations into 1-1 alignment on the batch dimension
        if n_manip:
            X = np.concatenate(X, axis=0)
            X_time = np.concatenate(X_time, axis=0)
            X_mask = np.ones_like(X_time)
            t_delta = np.concatenate(t_delta, axis=0)
            gf_y = np.concatenate(gf_y, axis=0)

        if reference_type == 'sampling':
            X_samples = self.sample_impulses(size=S).values
            X_samples = np.expand_dims(X_samples, axis=(1,2)) # shape (S, 1, 1, K)
            X_ref_samples = X_samples * X_ref_mask[None, None, None, ...]
            if n_manip:
                X_main_samples = X_samples * X_main_mask[None, None, None, ...]

        alpha = 100-float(level)

        samples = {}
        for i in range(S):
            if reference_type == 'sampling':
                _X_ref_in = X_ref_in + X_ref_samples[i]
                if n_manip:
                    _X = X + X_main_samples[i]
                else:
                    _X = X
            else:
                _X_ref_in = X_ref_in
                _X = X

            self.resample_model()
            if include_interactions:
                delta = self.response_distribution_delta_w_interactions
            else:
                delta = self.response_distribution_delta
            to_run = {}
            b = None
            for response in responses:
                to_run[response] = {}
                for response_param in response_params:
                    _b = (self.history_length + self.future_length) * self.eval_minibatch_size
                    if b is None:
                        b = _b
                    if response_param == 'mean' and not self.has_analytical_mean[response]:
                        b = max(1, _b // N_MCIFIED_DIST_RESAMP)
                    if i == 0 and response_param == 'mean' and not self.has_analytical_mean[response]:
                        stderr(
                            'WARNING: The response distribution for %s lacks an analytical mean, '
                            'so the mean is bootstrapped, which is computationally '
                            'intensive.\n' % response
                        )
                    dim_names = self.expand_param_name(response, response_param)
                    for dim_name in dim_names:
                        to_run[response][dim_name] = delta[response][dim_name]

            sample_ref = []
            for j in range(0, len(X_ref_in), b):
                fd_ref = {
                    self.X: _X_ref_in[j:j+b],
                    self.X_time: X_time_ref_in[j:j+b],
                    self.X_mask: X_mask_ref_in[j:j+b],
                    self.t_delta: t_delta_ref_in[j:j+b],
                    self.Y_gf: gf_y_ref_in[j:j+b],
                    self.training: not self.predict_mode
                }
                if resample:
                    fd_ref[self.use_MAP_mode] = False
                sample_ref.append(self.session.run(to_run, feed_dict=fd_ref))
            sample_ref = concat_nested(sample_ref)
            for response in to_run:
                for dim_name in to_run[response]:
                    _sample = sample_ref[response][dim_name]
                    sample_ref[response][dim_name] = np.reshape(_sample, ref_shape, 'F')

            if n_manip:
                sample = {}
                sample_main = []
                for j in range(0, len(_X), b):
                    fd_main = {
                        self.X: _X[j:j+b],
                        self.X_time: X_time[j:j+b],
                        self.X_mask: X_mask[j:j+b],
                        self.t_delta: t_delta[j:j+b],
                        self.Y_gf: gf_y[j:j+b],
                        self.training: not self.predict_mode
                    }
                    if resample:
                        fd_main[self.use_MAP_mode] = False
                    sample_main.append(self.session.run(to_run, feed_dict=fd_main))
                sample_main = concat_nested(sample_main)
                for response in to_run:
                    sample[response] = {}
                    for dim_name in sample_main[response]:
                        sample_main[response][dim_name] = np.reshape(sample_main[response][dim_name], sample_shape, 'F')
                        sample_main[response][dim_name] = sample_main[response][dim_name] - sample_ref[response][dim_name]
                        if ref_as_manip:
                            sample[response][dim_name] = np.concatenate(
                                [sample_ref[response][dim_name], sample_main[response][dim_name]],
                                axis=-1
                            )
                        else:
                            sample[response][dim_name] = sample_main[response][dim_name]
            else:
                sample = sample_ref
            for response in sample:
                if not response in samples:
                    samples[response] = {}
                for dim_name in sample[response]:
                    if not dim_name in samples[response]:
                        samples[response][dim_name] = []
                    samples[response][dim_name].append(sample[response][dim_name])

        lower = {}
        upper = {}
        mean = {}
        for response in samples:
            lower[response] = {}
            upper[response] = {}
            mean[response] = {}
            for dim_name in samples[response]:
                _samples = np.stack(samples[response][dim_name], axis=0)
                rescale = self.is_real(response) and \
                          not self.get_response_dist_name(response) == 'lognormal' and \
                          (dim_name.startswith('mu') or dim_name.startswith('sigma'))
                if rescale:
                    _samples = _samples * self.Y_train_sds[response]
                samples[response][dim_name] = np.stack(_samples, axis=0)
                _mean = _samples.mean(axis=0)
                mean[response][dim_name] = _mean
                if resample:
                    lower[response][dim_name] = np.percentile(_samples, alpha / 2, axis=0)
                    upper[response][dim_name] = np.percentile(_samples, 100 - (alpha / 2), axis=0)
                else:
                    lower = upper = mean
                    samples[response][dim_name] = _mean[None, ...]

        out = (plot_axes, mean, lower, upper, samples)

        return out

    def irf_rmsd(
            self,
            gold_irf_lambda,
            level=95,
            **kwargs
    ):
        """
        Compute root mean squared deviation of estimated from true IRFs over some interval(s) of interest.
        Any plotting configuration available under ``get_plot_data()`` is supported, but **gold_irf_lambda**
        must accept the same inputs and have the same output dimensionality. See documentation for ``get_plot_data()``
        for description of available keyword arguments.

        :param gold_irf_lambda: True IRF mapping inputs to outputs.
        :param **kwargs: Keyword arguments for ``get_plot_data()``.
        :return: 4-tuple (mean, lower, upper, samples); Let S be the number of samples. ``mean``, ``lower``, and ``upper`` are scalars, and ``samples`` is a vector of size S.
        """

        plot_axes, _, _, _, samples = self.get_plot_data(
            level=level,
            **kwargs
        )

        gold = gold_irf_lambda(plot_axes)

        alpha = 100 - float(level)

        rmsd_samples = {}
        rmsd_mean = {}
        rmsd_lower = {}
        rmsd_upper = {}
        for response in samples:
            rmsd_samples[response] = {}
            rmsd_mean[response] = {}
            rmsd_lower[response] = {}
            rmsd_upper[response] = {}
            for dim_name in samples[response]:
                _samples = samples[response][dim_name]
                axis = tuple(range(1, len(_samples.shape)))
                _rmsd_samples = ((gold - _samples)**2).mean(axis=axis)
                rmsd_samples[response] = _rmsd_samples
                rmsd_mean[response][dim_name] = _rmsd_samples.mean()
                rmsd_lower[response][dim_name] = np.percentile(_rmsd_samples, alpha / 2)
                rmsd_upper[response][dim_name] = np.percentile(_rmsd_samples, 100 - (alpha / 2))

        return rmsd_mean, rmsd_lower, rmsd_upper, rmsd_samples

    def irf_integrals(
            self,
            responses=None,
            response_params=None,
            level=95,
            random=False,
            n_samples='default',
            n_time_units=None,
            n_time_points=1000
    ):
        """
        Generate effect size estimates by computing the area under each IRF curve in the model via discrete approximation.

        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to plot.
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of response distribution(s) to plot per response variable. Any param names not used by the response distribution for a given response will be ignored.
        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use mean/MLE model.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``pandas`` DataFrame; IRF integrals, one IRF per row. If Bayesian, array also contains credible interval bounds.
        """

        assert n_time_points > 0, 'n_time_points must be positive.'

        if n_time_units is None:
            n_time_units = self.t_delta_limit

        alpha = 100 - float(level)
        if self.future_length:
            xmin = -n_time_units / 2.
            xmax = n_time_units / 2.
        else:
            xmin = 0.
            xmax = n_time_units

        self.set_predict_mode(True)

        names = self.impulse_names
        has_rate = 'rate' in names
        names = [x for x in names if not self.has_nn_irf or x != 'rate']

        manipulations = []
        step_size = []
        if self.has_nn_irf and has_rate:
            step_size.append(np.ones(n_time_points) * float(n_time_units) / n_time_points)
        for x in names:
            if self.is_non_dirac(x):
                step_size.append(np.ones(n_time_points) * float(n_time_units) / n_time_points)
            else:
                timepoints = np.linspace(xmin, xmax, n_time_points)
                steps = np.abs(timepoints) < self.epsilon
                n_steps = steps.sum()
                if n_steps > 1:
                    steps = steps / n_steps
                step_size.append(steps)
            delta = self.plot_step_map[x]
            manipulations.append({x: delta})
        if len(step_size):
            step_size = np.stack(step_size, axis=1)[None, ...] # Add sample dim

        if random:
            ranef_group_names = self.ranef_group_names
            ranef_level_names = self.ranef_level_names
            ranef_zipped = zip(ranef_group_names, ranef_level_names)
            gf_y_refs = [{x: y} for x, y in ranef_zipped]
        else:
            gf_y_refs = [{None: None}]

        names = [get_irf_name(x, self.irf_name_map) for x in names]
        if has_rate and self.has_nn_irf:
            names = [get_irf_name('rate', self.irf_name_map)] + names
        sort_key_dict = {x: i for i, x in enumerate(names)}
        def sort_key_fn(x, sort_key_dict=sort_key_dict):
            if x.name == 'IRF':
                return x.map(sort_key_dict)
            return x

        out = []

        if responses is None:
            responses = self.response_names
        if response_params is None:
            response_params = {'mean'}
            for _response in responses:
                response_params.add(self.get_response_params(_response)[0])
            response_params = sorted(list(response_params))

        for g, gf_y_ref in enumerate(gf_y_refs):
            _, _, _, _, vals = self.get_plot_data(
                xvar='t_delta',
                responses=responses,
                response_params=response_params,
                X_ref=None,
                X_time_ref=None,
                t_delta_ref=None,
                gf_y_ref=gf_y_ref,
                ref_varies_with_x=True,
                manipulations=manipulations,
                pair_manipulations=False,
                xaxis=None,
                xmin=xmin,
                xmax=xmax,
                xres=n_time_points,
                n_samples=n_samples,
                level=level,
            )

            for _response in vals:
                for _dim_name in vals[_response]:
                    _vals = vals[_response][_dim_name]
                    if not self.has_nn_irf or not has_rate:
                        _vals = _vals[..., 1:]

                    integrals = (_vals * step_size).sum(axis=1)

                    group_name = list(gf_y_ref.keys())[0]
                    level_name = gf_y_ref[group_name]

                    out_cur = pd.DataFrame({
                        'IRF': names,
                        'Group': group_name if group_name is not None else '',
                        'Level': level_name if level_name is not None else '',
                        'Response': _response,
                        'ResponseParam': _dim_name
                    })

                    if n_samples:
                        mean = integrals.mean(axis=0)
                        lower = np.percentile(integrals, alpha / 2, axis=0)
                        upper = np.percentile(integrals, 100 - (alpha / 2), axis=0)

                        out_cur['Mean'] = mean
                        out_cur['%.1f%%' % (alpha / 2)] = lower
                        out_cur['%.1f%%' % (100 - (alpha / 2))] = upper
                    else:
                        out_cur['Estimate'] = integrals[0]
                    out.append(out_cur)

        out = pd.concat(out, axis=0).reset_index(drop=True)
        out.sort_values(
            ['IRF', 'Group', 'Level'],
            inplace=True,
            key=sort_key_fn
        )

        self.set_predict_mode(False)

        return out

    def get_reference_map(
            self,
            reference_values=None,
            default_reference_type=None
    ):
        if reference_values is None:
            reference_values = self.reference_values
        if default_reference_type is None:
            default_reference_type = self.default_reference_type
        reference_map = {}
        for pair in reference_values.split():
            impulse_name, val = pair.split('=')
            reference = float(val)
            reference_map[impulse_name] = reference
        for x in self.impulse_names:
            if not x in reference_map:
                if default_reference_type == 'mean':
                    reference_map[x] = self.impulse_means[x]
                else:
                    reference_map[x] = 0.

        return reference_map

    def get_plot_step_map(
            self,
            plot_step=None,
            plot_step_default=None
    ):
        if plot_step is None:
            plot_step = self.plot_step
        if plot_step_default is None:
            plot_step_default = self.plot_step_default

        plot_step_map = {}
        for pair in plot_step.split():
            impulse_name, val = pair.split('=')
            plot_step = float(val)
            plot_step_map[impulse_name] = plot_step
        for x in self.impulse_names:
            if not x in plot_step_map:
                if x in self.indicators:
                    plot_step_map[x] = 1 - self.reference_map[x]
                elif isinstance(plot_step_default, str) and plot_step_default.lower() == 'sd':
                    plot_step_map[x] = self.impulse_sds[x]
                else:
                    plot_step_map[x] = plot_step_default

        return plot_step_map

    def make_plots(
            self,
            irf_name_map=None,
            responses=None,
            response_params=None,
            pred_names=None,
            sort_names=True,
            prop_cycle_length=None,
            prop_cycle_map=None,
            plot_dirac=None,
            reference_time=None,
            plot_rangf=False,
            plot_n_time_units=None,
            plot_n_time_points=None,
            reference_type=None,
            plot_quantile_range=0.9,
            plot_step=None,
            plot_step_default=None,
            generate_univariate_irf_plots=None,
            generate_univariate_irf_heatmaps=None,
            generate_curvature_plots=None,
            generate_irf_surface_plots=None,
            generate_nonstationarity_surface_plots=None,
            generate_interaction_surface_plots=None,
            generate_err_dist_plots=None,
            x_axis_transform=None,
            y_axis_transform=None,
            plot_x_inches=None,
            plot_y_inches=None,
            ylim=None,
            use_horiz_axlab=True,
            use_vert_axlab=True,
            cmap=None,
            dpi=None,
            level=95,
            n_samples=None,
            prefix=None,
            suffix='.png',
            use_legend=None,
            use_line_markers=False,
            transparent_background=False,
            keep_plot_history=None,
            dump_source=False,
            **kwargs
    ):
        """
        Generate plots of current state of deconvolution.
        CDR distinguishes plots based on two orthogonal criteria: "atomic" vs. "composite" and "scaled" vs. "unscaled".
        The "atomic"/"composite" distinction is only relevant in models containing composed IRF.
        In such models, "atomic" plots represent the shape of the IRF irrespective of any other IRF with which they are composed, while "composite" plots represent the shape of the IRF composed with any upstream IRF in the model.
        In models without composed IRF, only "atomic" plots are generated.
        The "scaled"/"unscaled" distinction concerns whether the impulse coefficients are represented in the plot ("scaled") or not ("unscaled").
        Only pre-terminal IRF (i.e. the final IRF in all IRF compositions) have coefficients, so only preterminal IRF are represented in "scaled" plots, while "unscaled" plots also contain all intermediate IRF.
        In addition, Bayesian CDR implementations also support MC sampling of credible intervals around all curves.
        Outputs are saved to the model's output directory as PNG files with names indicating which plot type is represented.
        All plot types relevant to a given model are generated.

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to plot. If ``None``, plots all univariate responses. Multivariate plotting (e.g. of categorical responses) is supported but turned off by default to avoid excessive computation. When plotting a multivariate response, a set of plots will be generated for each dimension of the response.
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of response distribution(s) to plot per response variable. Any param names not used by the response distribution for a given response will be ignored. If ``None``, plots the first parameter of each response distribution.
        :param summed: ``bool``; whether to plot individual IRFs or their sum.
        :param pred_names: ``list`` or ``None``; list of names of predictors to include in plots. If ``None``, all predictors are plotted.
        :param sort_names: ``bool``; whether to alphabetically sort IRF names.
        :param plot_unscaled: ``bool``; plot unscaled IRFs.
        :param plot_composite: ``bool``; plot any composite IRFs. If ``False``, only plots terminal IRFs.
        :param prop_cycle_length: ``int`` or ``None``; Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **pred_names**.
        :param prop_cycle_map: ``dict``, ``list`` of ``int``, or ``None``; Integer indices to use in the properties cycle for each entry in **pred_names**. If a ``dict``, a map from predictor names to ``int``. If a ``list`` of ``int``, predictors inferred using **pred_names** are aligned to ``int`` indices one-to-one. If ``None``, indices are automatically assigned.
        :param plot_dirac: ``bool`` or ``None``; whether to include any Dirac delta IRF's (stick functions at t=0) in plot. If ``None``, use default setting.
        :param reference_time: ``float`` or ``None``; timepoint at which to plot interactions. If ``None``, use default setting.
        :param plot_rangf: ``bool``; whether to plot all (marginal) random effects.
        :param plot_n_time_units: ``float`` or ``None``; resolution of plot axis (for 3D plots, uses sqrt of this number for each axis). If ``None``, use default setting.
        :param plot_support_start: ``float`` or ``None``; start time for IRF plots. If ``None``, use default setting.
        :param reference_type: ``bool``; whether to use the predictor means as baseline reference (otherwise use zero).
        :param plot_quantile_range: ``float``; quantile range to use for plotting. E.g., 0.9 uses the interdecile range.
        :param plot_step: ``str`` or ``None``; size of step by predictor to take above reference in univariate IRF plots. Structured as space-delimited pairs ``NAME=FLOAT``. Any predictor without a specified step size will inherit from **plot_step_default**.
        :param plot_step_default: ``float``, ``str``, or ``None``; default size of step to take above reference in univariate IRF plots, if not specified in **plot_step**. Either a float or the string ``'sd'``, which indicates training sample standard deviation.
        :param generate_univariate_irf_plots: ``bool``; whether to plot univariate IRFs over time.
        :param generate_univariate_irf_heatmaps: ``bool``; whether to plot univariate IRFs over time.
        :param generate_curvature_plots: ``bool`` or ``None``; whether to plot IRF curvature at time **reference_time**. If ``None``, use default setting.
        :param generate_irf_surface_plots: ``bool`` or ``None``; whether to plot IRF surfaces.  If ``None``, use default setting.
        :param generate_nonstationarity_surface_plots: ``bool`` or ``None``; whether to plot IRF surfaces showing non-stationarity in the response.  If ``None``, use default setting.
        :param generate_interaction_surface_plots: ``bool`` or ``None``; whether to plot IRF interaction surfaces at time **reference_time**.  If ``None``, use default setting.
        :param generate_err_dist_plots: ``bool`` or ``None``; whether to plot the average error distribution for real-valued responses.  If ``None``, use default setting.
        :param x_axis_transform: ``str`` or ``None``; string description of transform to apply to x-axis prior to plotting. Currently supported: ``'exp'``, ``'log'``, ``'neglog'``. If ``None``, no x-axis transform.
        :param y_axis_transform: ``str`` or ``None``; string description of transform to apply to y-axis (in 3d plots only) prior to plotting. Currently supported: ``'exp'``, ``'log'``, ``'neglog'``. If ``None``, no y-axis transform.
        :param plot_x_inches: ``float`` or ``None``; width of plot in inches. If ``None``, use default setting.
        :param plot_y_inches: ``float`` or ``None; height of plot in inches. If ``None``, use default setting.
        :param ylim: 2-element ``tuple`` or ``list``; (lower_bound, upper_bound) to use for y axis. If ``None``, automatically inferred.
        :param use_horiz_axlab: ``bool``; whether to include horizontal axis label(s) (x axis in 2D plots, x/y axes in 3D plots).
        :param use_vert_axlab: ``bool``; whether to include vertical axis label (y axis in 2D plots, z axis in 3D plots).
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param dpi: ``int`` or ``None``; dots per inch of saved plot image file. If ``None``, use default setting.
        :param level: ``float``; significance level for confidence/credible intervals, if supported.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param prefix: ``str`` or ``None``; prefix appended to output filenames. If ``None``, no prefix added.
        :param suffix: ``str``; file extension of plot outputs.
        :param use_legend: ``bool`` or ``None``; whether to include a legend in plots with multiple components. If ``None``, use default setting.
        :param use_line_markers: ``bool``; whether to add markers to lines in univariate IRF plots.
        :param transparent_background: ``bool``; whether to use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool`` or ``None``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot. If ``None``, use default setting.
        :param dump_source: ``bool``; Whether to dump the plot source array to a csv file.
        :param **kwargs: ``dict``; extra kwargs to pass to ``get_plot_data``
        :return: ``None``
        """

        if irf_name_map is None:
            irf_name_map = self.irf_name_map

        if responses is None:
            if self.n_response == 1:
                responses = self.response_names
            else:
                responses = [x for x in self.response_names if self.get_response_ndim(x) == 1]
        if isinstance(responses, str):
            responses = [responses]

        if response_params is None:
            response_params = set()
            for response in responses:
                if self.has_analytical_mean[response]:
                    response_params.add('mean')
                else:
                    response_params.add(self.get_response_params(response)[0])
            response_params = sorted(list(response_params))
        if isinstance(response_params, str):
            response_params = [response_params]

        mc = bool(n_samples) and (self.is_bayesian or self.has_dropout)

        if plot_dirac is None:
            plot_dirac = self.plot_dirac
        if reference_time is None:
            reference_time = self.reference_time
        if plot_n_time_units is None:
            plot_n_time_units = self.plot_n_time_units
        if plot_n_time_points is None:
            plot_n_time_points = self.plot_n_time_points
        if generate_univariate_irf_plots is None:
            generate_univariate_irf_plots = self.generate_univariate_irf_plots
        if generate_univariate_irf_heatmaps is None:
            generate_univariate_irf_heatmaps = self.generate_univariate_irf_heatmaps
        if generate_curvature_plots is None:
            generate_curvature_plots = self.generate_curvature_plots
        if generate_irf_surface_plots is None:
            generate_irf_surface_plots = self.generate_irf_surface_plots
        if generate_nonstationarity_surface_plots is None:
            generate_nonstationarity_surface_plots = self.generate_nonstationarity_surface_plots
        if generate_interaction_surface_plots is None:
            generate_interaction_surface_plots = self.generate_interaction_surface_plots
        if generate_err_dist_plots is None:
            generate_err_dist_plots = self.generate_err_dist_plots
        if x_axis_transform is None:
            def x_axis_transform(x):
                return x
        elif x_axis_transform.lower() == 'exp':
            x_axis_transform = np.exp
        elif x_axis_transform.lower() == 'log':
            x_axis_transform = np.log
        elif x_axis_transform.lower() == 'neglog':
            def x_axis_transform(x):
                return -np.log(x)
        elif x_axis_transform.lower() == 'sqrt':
            def x_axis_transform(x):
                return np.sqrt(x)
        else:
            def x_axis_transform(x, fname=x_axis_transform):
                return getattr(np, fname)(x)
        
        if y_axis_transform is None:
            def y_axis_transform(x):
                return x
        elif y_axis_transform.lower() == 'exp':
            y_axis_transform = np.exp
        elif y_axis_transform.lower() == 'log':
            y_axis_transform = np.log
        elif y_axis_transform.lower() == 'neglog':
            def y_axis_transform(x):
                return -np.log(x)
        else:
            def y_axis_transform(x, fname=y_axis_transform):
                return getattr(np, fname)(x)
        
        if plot_x_inches is None:
            plot_x_inches = self.plot_x_inches
        if plot_y_inches is None:
            plot_y_inches = self.plot_y_inches
        if use_legend is None:
            use_legend = self.plot_legend
        if cmap is None:
            cmap = self.cmap
        if dpi is None:
            dpi = self.dpi
        if keep_plot_history is None:
            keep_plot_history = self.keep_plot_history

        if prefix is None:
            prefix = ''
        if prefix != '' and not prefix.endswith('_'):
            prefix += '_'

        if plot_rangf:
            ranef_level_names = self.ranef_level_names
            ranef_group_names = self.ranef_group_names
        else:
            ranef_level_names = [None]
            ranef_group_names = [None]

        self.set_predict_mode(True)

        # IRF 1D
        if generate_univariate_irf_plots or generate_univariate_irf_heatmaps:
            names = self.impulse_names
            if not plot_dirac:
                names = [x for x in names if self.is_non_dirac(x)]
            if pred_names is not None and len(pred_names) > 0:
                new_names = []
                for i, name in enumerate(names):
                    for ID in pred_names:
                        if ID == name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                            new_names.append(name)
                names = new_names
            has_rate = 'rate' in names
            names = [x for x in names if not (self.has_nn_irf and x == 'rate')]

            plot_step_map = self.get_plot_step_map(
                plot_step,
                plot_step_default
            )
            manipulations = []
            for x in names:
                delta = plot_step_map[x]
                manipulations.append({x: delta})
            gf_y_refs = [{x: y} for x, y in zip(ranef_group_names, ranef_level_names)]

            fixed_impulses = set()
            for x in self.t.terminals():
                if x.fixed:
                    if x.impulse.is_nn_impulse():
                        for y in x.impulse.impulses():
                            if y.name() in names:
                                fixed_impulses.add(y.name())
                    elif x.impulse.name() in names:
                        for y in x.impulse_names():
                            fixed_impulses.add(y)

            names_fixed = [x for x in names if x in fixed_impulses]
            manipulations_fixed = [x for x in manipulations if list(x.keys())[0] in fixed_impulses]

            if has_rate and self.has_nn_irf:
                names = ['rate'] + names
                names_fixed = ['rate'] + names_fixed

            xinterval = plot_n_time_units
            xmin = -xinterval * self.prop_fwd
            xmax = xinterval * self.prop_bwd

            for g, (gf_y_ref, gf_key) in enumerate(zip(gf_y_refs, ranef_level_names)):
                if gf_key is None:
                    names_cur = names_fixed
                    manipulations_cur = manipulations_fixed
                else:
                    names_cur = names
                    manipulations_cur = manipulations

                plot_x, plot_y, lq, uq, samples = self.get_plot_data(
                    xvar='t_delta',
                    responses=responses,
                    response_params=response_params,
                    gf_y_ref=gf_y_ref,
                    ref_varies_with_x=True,
                    manipulations=manipulations_cur,
                    pair_manipulations=False,
                    reference_type=reference_type,
                    xaxis=None,
                    xmin=xmin,
                    xmax=xmax,
                    xres=plot_n_time_points,
                    n_samples=n_samples,
                    level=level,
                    **kwargs
                )

                plot_x = x_axis_transform(plot_x)

                for _response in plot_y:
                    for _dim_name in plot_y[_response]:
                        include_param_name = True

                        plot_name = 'irf_univariate_%s' % sn(_response)
                        if include_param_name:
                            plot_name += '_%s' % _dim_name

                        if use_horiz_axlab:
                            xlab = 't_delta'
                        else:
                            xlab = None
                        if use_vert_axlab:
                            ylab = [get_irf_name(_response, irf_name_map)]
                            if include_param_name and _dim_name != 'mean':
                                ylab.append(_dim_name)
                            ylab = ', '.join(ylab)
                        else:
                            ylab = None

                        filename = prefix + plot_name

                        if ranef_level_names[g]:
                            filename += '_' + ranef_level_names[g]
                        if mc:
                            filename += '_mc'
                        filename += suffix

                        _plot_y = plot_y[_response][_dim_name]
                        _lq = None if lq is None else lq[_response][_dim_name]
                        _uq = None if uq is None else uq[_response][_dim_name]

                        if not (self.has_nn_irf and has_rate):
                            _plot_y = _plot_y[..., 1:]
                            _lq = None if _lq is None else _lq[..., 1:]
                            _uq = None if _uq is None else _uq[..., 1:]

                        assert _plot_y.shape[-1] == len(names_cur), 'Mismatch between the number of impulse names ' + \
                                                                    'and the number of plot dimensions. Got %d ' + \
                                                                    'impulse names and %d plot dimensions.' % \
                                                                    (len(names_cur), _plot_y.shape[-1])

                        if generate_univariate_irf_plots:
                            plot_irf(
                                plot_x,
                                _plot_y,
                                names_cur,
                                lq=_lq,
                                uq=_uq,
                                sort_names=sort_names,
                                prop_cycle_length=prop_cycle_length,
                                prop_cycle_map=prop_cycle_map,
                                outdir=self.outdir,
                                filename=filename,
                                irf_name_map=irf_name_map,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                ylim=ylim,
                                cmap=cmap,
                                dpi=dpi,
                                legend=use_legend,
                                xlab=xlab,
                                ylab=ylab,
                                use_line_markers=use_line_markers,
                                transparent_background=transparent_background,
                                dump_source=dump_source
                            )

                        if generate_univariate_irf_heatmaps:
                            plot_irf_as_heatmap(
                                plot_x,
                                _plot_y,
                                names_cur,
                                sort_names=sort_names,
                                outdir=self.outdir,
                                filename=filename[:-4] + '_hm' + suffix,
                                irf_name_map=irf_name_map,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                ylim=ylim,
                                dpi=dpi,
                                xlab=xlab,
                                ylab=ylab,
                                transparent_background=transparent_background,
                                dump_source=dump_source
                            )

        if plot_rangf:
            manipulations = [{'ranef': {x: y}} for x, y in zip(ranef_group_names[1:], ranef_level_names[1:])]
        else:
            manipulations = None

        # Curvature plots
        if generate_curvature_plots:
            names = [x for x in self.impulse_names if x != 'rate']
            if pred_names is not None and len(pred_names) > 0:
                new_names = []
                for i, name in enumerate(names):
                    for ID in pred_names:
                        if ID == name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                            new_names.append(name)
                names = new_names

            for name in names:
                if self.is_non_dirac(name):
                    _reference_time = reference_time
                else:
                    _reference_time = 0.
                plot_x, plot_y, lq, uq, samples = self.get_plot_data(
                    xvar=name,
                    responses=responses,
                    response_params=response_params,
                    t_delta_ref=_reference_time,
                    ref_varies_with_x=False,
                    manipulations=manipulations,
                    pair_manipulations=True,
                    reference_type=reference_type,
                    plot_quantile_range=plot_quantile_range,
                    xres=plot_n_time_points,
                    n_samples=n_samples,
                    level=level,
                    **kwargs
                )

                plot_x = x_axis_transform(plot_x)

                for _response in plot_y:
                    for _dim_name in plot_y[_response]:
                        include_param_name = True

                        plot_name = 'curvature_%s' % sn(_response)
                        if include_param_name:
                            plot_name += '_%s' % _dim_name

                        plot_name += '_%s_at_delay%s' % (sn(name), _reference_time)

                        if use_horiz_axlab:
                            xlab = name
                        else:
                            xlab = None
                        if use_vert_axlab:
                            ylab = [get_irf_name(_response, irf_name_map)]
                            if include_param_name and _dim_name != 'mean':
                                ylab.append(_dim_name)
                            ylab = ', '.join(ylab)
                        else:
                            ylab = None

                        _plot_y = plot_y[_response][_dim_name]
                        _lq = None if lq is None else lq[_response][_dim_name]
                        _uq = None if uq is None else uq[_response][_dim_name]

                        for g in range(len(ranef_level_names)):
                            filename = prefix + plot_name
                            if ranef_level_names[g]:
                                filename += '_' + ranef_level_names[g]
                            if mc:
                                filename += '_mc'
                            filename += suffix

                            plot_irf(
                                plot_x,
                                _plot_y[:, g:g + 1],
                                [name],
                                lq=None if _lq is None else _lq[:, g:g + 1],
                                uq=None if _uq is None else _uq[:, g:g + 1],
                                sort_names=sort_names,
                                prop_cycle_length=prop_cycle_length,
                                prop_cycle_map=prop_cycle_map,
                                outdir=self.outdir,
                                filename=filename,
                                irf_name_map=irf_name_map,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                cmap=cmap,
                                dpi=dpi,
                                legend=False,
                                xlab=xlab,
                                ylab=ylab,
                                use_line_markers=use_line_markers,
                                transparent_background=transparent_background,
                                dump_source=dump_source
                            )

        # Surface plots
        for plot_type, run_plot in zip(
                ('irf_surface', 'nonstationarity_surface', 'interaction_surface',),
                (generate_irf_surface_plots, generate_nonstationarity_surface_plots, generate_interaction_surface_plots)
        ):
            if run_plot:
                names = [x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]
                if pred_names is not None and len(pred_names) > 0:
                    new_names = []
                    for i, name in enumerate(names):
                        for ID in pred_names:
                            if ID == name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                                new_names.append(name)
                    names = new_names
                if plot_type == 'irf_surface':
                    names = ['t_delta:%s' % x for x in names]
                elif plot_type == 'nonstationarity_surface':
                    names = ['X_time:%s' % x for x in names]
                else: # plot_type == 'interaction_surface'
                    names_src = [x for x in names]
                    names = [':'.join(x) for x in itertools.combinations(names_src, 2)]
                if names:
                    for name in names:
                        xvar, yvar = name.split(':')

                        if plot_type in ('nonstationarity_surface', 'interaction_surface'):
                            ref_varies_with_x = False
                        else:
                            ref_varies_with_x = True

                        if plot_type == 'irf_surface':
                            xinterval = plot_n_time_units
                            xmin = -xinterval * self.prop_fwd
                            xmax = xinterval * self.prop_bwd
                        else:
                            xmin = None
                            xmax = None

                        (plot_x, plot_y), plot_z, lq, uq, _ = self.get_plot_data(
                            xvar=xvar,
                            yvar=yvar,
                            responses=responses,
                            response_params=response_params,
                            t_delta_ref=reference_time,
                            ref_varies_with_x=ref_varies_with_x,
                            manipulations=manipulations,
                            pair_manipulations=True,
                            reference_type=reference_type,
                            plot_quantile_range=plot_quantile_range,
                            xmin=xmin,
                            xmax=xmax,
                            xres=int(np.ceil(np.sqrt(plot_n_time_points))),
                            yres=int(np.ceil(np.sqrt(plot_n_time_points))),
                            n_samples=n_samples,
                            level=level,
                            **kwargs
                        )

                        plot_x = x_axis_transform(plot_x)
                        plot_y = y_axis_transform(plot_y)

                        for _response in plot_z:
                            for _dim_name in plot_z[_response]:
                                param_names = self.get_response_params(_response)
                                include_param_name = True

                                plot_name = 'surface_%s' % sn(_response)
                                if include_param_name:
                                    plot_name += '_%s' % _dim_name

                                if use_horiz_axlab:
                                    xlab = xvar
                                    ylab = yvar
                                else:
                                    xlab = None
                                    ylab = None
                                if use_vert_axlab:
                                    zlab = [get_irf_name(_response, irf_name_map)]
                                    if include_param_name and _dim_name != 'mean':
                                        zlab.append(_dim_name)
                                    zlab = ', '.join(zlab)
                                else:
                                    zlab = None

                                _plot_z = plot_z[_response][_dim_name]
                                _lq = None if lq is None else lq[_response][_dim_name]
                                _uq = None if uq is None else uq[_response][_dim_name]

                                for g in range(len(ranef_level_names)):
                                    filename = prefix + plot_name + '_' + sn(yvar) + '_by_' + sn(xvar)
                                    if plot_type in ('nonstationarity_surface', 'interaction_surface'):
                                        filename += '_at_delay%s' % reference_time
                                    if ranef_level_names[g]:
                                        filename += '_' + ranef_level_names[g]
                                    if mc:
                                        filename += '_mc'
                                    filename += suffix

                                    plot_surface(
                                        plot_x,
                                        plot_y,
                                        _plot_z[..., g],
                                        lq=None if _lq is None else _lq[..., g],
                                        uq=None if _uq is None else _uq[..., g],
                                        outdir=self.outdir,
                                        filename=filename,
                                        irf_name_map=irf_name_map,
                                        plot_x_inches=plot_x_inches,
                                        plot_y_inches=plot_y_inches,
                                        xlab=xlab,
                                        ylab=ylab,
                                        zlab=zlab,
                                        transparent_background=transparent_background,
                                        dpi=dpi,
                                        dump_source=dump_source
                                    )

        if generate_err_dist_plots:
            for _response in self.error_distribution_plot:
                if self.is_real(_response):
                    lb = self.session.run(self.error_distribution_plot_lb[_response])
                    ub = self.session.run(self.error_distribution_plot_ub[_response])
                    n_time_units = ub - lb
                    fd = {
                        self.support_start: lb,
                        self.n_time_units: n_time_units,
                        self.n_time_points: plot_n_time_points,
                        self.training: not self.predict_mode
                    }

                    plot_x = self.session.run(self.support, feed_dict=fd)
                    plot_y = self.session.run(self.error_distribution_plot[_response], feed_dict=fd)

                    plot_name = 'error_distribution_%s%s' % (sn(_response), suffix)

                    lq = None
                    uq = None

                    plot_irf(
                        plot_x,
                        plot_y,
                        ['Error Distribution'],
                        lq=lq,
                        uq=uq,
                        outdir=self.outdir,
                        filename=prefix + plot_name,
                            legend=False,
                    )

        self.set_predict_mode(False)

    def parameter_table(self, fixed=True, level=95, n_samples='default'):
        """
        Generate a pandas table of parameter names and values.

        :param fixed: ``bool``; Return a table of fixed parameters (otherwise returns a table of random parameters).
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :return: ``pandas`` ``DataFrame``; The parameter table.
        """

        assert fixed or self.is_mixed_model, 'Attempted to generate a random effects parameter table in a fixed-effects-only model'

        if n_samples == 'default':
            if self.is_bayesian or self.has_dropout:
                n_samples = self.n_samples_eval
                MAP_mode = False
            else:
                n_samples = 1
                MAP_mode = True
        elif n_samples:
            MAP_mode = False
        else:
            n_samples = 1
            MAP_mode = True

        alpha = 100 - float(level)

        keys = []
        values = []

        if fixed:
            for response in self.intercept_fixed:
                keys.append(('intercept', response))
                values.append(self.intercept_fixed[response])
            for response in self.coefficient_fixed:
                keys.append(('coefficient', response))
                values.append(self.coefficient_fixed[response])
            for response in self.irf_param_fixed:
                for irf_id in self.irf_param_fixed[response]:
                    for irf_param_name in self.irf_param_fixed[response][irf_id]:
                        keys.append(('irf_param_%s_%s' % (irf_param_name, irf_id), response))
                        values.append(self.irf_param_fixed[response][irf_id][irf_param_name])
            for response in self.interaction_fixed:
                keys.append(('interaction', response))
                values.append(self.interaction_fixed[response])
        else:
            for response in self.intercept_random:
                for rangf in self.intercept_random[response]:
                    keys.append(('intercept', response, rangf))
                    values.append(self.intercept_random[response][rangf])
            for response in self.coefficient_random:
                for rangf in self.coefficient_random[response]:
                    keys.append(('coefficient', response, rangf))
                    values.append(self.coefficient_random[response][rangf])
            for response in self.irf_param_random:
                for rangf in self.irf_param_random[response]:
                    for irf_id in self.irf_param_random[response][rangf]:
                        for irf_param_name in self.irf_param_random[response][rangf][irf_id]:
                            keys.append(('irf_param_%s_%s' % (irf_param_name, irf_id), response, rangf))
                            values.append(self.irf_param_random[response][rangf][irf_id][irf_param_name])
            for response in self.interaction_random:
                for rangf in self.interaction_random[response]:
                    keys.append(('interaction', response, rangf))
                    values.append(self.interaction_random[response][rangf])

        self.set_predict_mode(True)
        samples = []
        with self.session.as_default():
            with self.session.graph.as_default():
                for i in range(n_samples):
                    self.resample_model()
                    samples.append(self.session.run(values, feed_dict={self.use_MAP_mode: MAP_mode}))
        self.set_predict_mode(False)

        out = []
        cols = ['Parameter']
        if len(self.response_names) > 1:
            cols.append('Response')
        if not fixed:
            cols += ['Group', 'Level']
        cols += ['ResponseParam', 'Mean', '%0.1f' % (alpha / 2), '%0.1f' % (100 - (alpha / 2))]
        for key_ix, key in enumerate(keys):
            if fixed:
                param_type, response = key
                rangf = gf_ix = levels = None
            else:
                param_type, response, rangf = key
                gf_ix = self.rangf.index(rangf)
                levels = sorted(self.rangf_map_ix_2_levelname[gf_ix][:-1])
            _samples = np.stack([x[key_ix] for x in samples], axis=0)
            mean = _samples.mean(axis=0)
            lower = np.percentile(_samples, alpha / 2, axis=0)
            upper = np.percentile(_samples, 100 - (alpha / 2), axis=0)
            df = []
            response_params = self.get_response_params(response)
            if param_type == 'intercept':
                for i, response_param in enumerate(response_params):
                    dim_names = self.expand_param_name(response, response_param)
                    for j, dim_name in enumerate(dim_names):
                        if fixed:
                            row = (param_type,)
                            if len(self.response_names) > 1:
                                row += (response,)
                            row += (dim_name, mean[i, j], lower[i, j], upper[i, j])
                            df.append(row)
                        else:
                            gf_ix = self.rangf.index(rangf)
                            for k, level in enumerate(levels):
                                row = (param_type,)
                                if len(self.response_names) > 1:
                                    row += (response,)
                                row += (rangf, level, dim_name, mean[k, i, j], lower[k, i, j], upper[k, i, j])
                                df.append(row)
            elif param_type == 'coefficient':
                for i, coef_name in enumerate(self.coef_names):
                    if self.use_distributional_regression:
                        _response_params = response_params
                    else:
                        _response_params = response_params[:1]
                    for j, response_param in enumerate(_response_params):
                        dim_names = self.expand_param_name(response, response_param)
                        for k, dim_name in enumerate(dim_names):
                            if fixed:
                                row = ('coefficient_%s' % coef_name,)
                                if len(self.response_names) > 1:
                                    row += (response,)
                                row += (dim_name, mean[i, j, k], lower[i, j, k], upper[i, j, k])
                                df.append(row)
                            else:
                                for l, level in enumerate(
                                        evels):
                                    row = ('coefficient_%s' % coef_name,)
                                    if len(self.response_names) > 1:
                                        row += (response,)
                                    row += (rangf, level, dim_name, mean[l, i, j, k], lower[l, i, j, k], upper[l, i, j, k])
                                    df.append(row)
            elif param_type.startswith('irf_param_'):
                if self.use_distributional_regression:
                    _response_params = response_params
                else:
                    _response_params = response_params[:1]
                for i, response_param in enumerate(_response_params):
                    dim_names = self.expand_param_name(response, response_param)
                    for j, dim_name in enumerate(dim_names):
                        if fixed:
                            row = (param_type[10:],)
                            if len(self.response_names) > 1:
                                row += (response,)
                            row += (dim_name, mean[i, j], lower[i, j], upper[i, j])
                            df.append(row)
                        else:
                            gf_ix = self.rangf.index(rangf)
                            for k, level in enumerate(levels):
                                row = (param_type[10:],)
                                if len(self.response_names) > 1:
                                    row += (response,)
                                row += (rangf, level, dim_name, mean[k, i, j], lower[k, i, j], upper[k, i, j])
                                df.append(row)
            elif param_type == 'interaction':
                for i, interaction_name in enumerate(self.interaction_names):
                    if self.use_distributional_regression:
                        _response_params = response_params
                    else:
                        _response_params = response_params[:1]
                    for j, response_param in enumerate(_response_params):
                        dim_names = self.expand_param_name(response, response_param)
                        for k, dim_name in enumerate(dim_names):
                            if fixed:
                                row = ('interaction_%s' % interaction_name,)
                                if len(self.response_names) > 1:
                                    row += (response,)
                                row += (dim_name, mean[i, j, k], lower[i, j, k], upper[i, j, k])
                                df.append(row)
                            else:
                                for l, level in enumerate(levels):
                                    row = ('interaction_%s' % interaction_name,)
                                    if len(self.response_names) > 1:
                                        row += (response,)
                                    row += (rangf, level, dim_name, mean[l, i, j, k], lower[l, i, j, k], upper[l, i, j, k])
                                    df.append(row)

            df = pd.DataFrame(df, columns=cols)
            out.append(df)

        out = pd.concat(out, axis=0)

        return out

    def save_parameter_table(self, random=True, level=95, n_samples='default', outfile=None):
        """
        Save space-delimited parameter table to the model's output directory.

        :param random: Include random parameters.
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int``, ``'defalt'``, or ``None``; number of posterior samples to draw if Bayesian.
        :param outfile: ``str``; Path to output file. If ``None``, use model defaults.
        :return: ``None``
        """

        if n_samples == 'default':
            if self.is_bayesian or self.has_dropout:
                n_samples = self.n_samples_eval

        parameter_table = self.parameter_table(
            fixed=True,
            level=level,
            n_samples=n_samples
        )

        if random and self.is_mixed_model:
            parameter_table = pd.concat(
                [
                    parameter_table,
                    self.parameter_table(
                        fixed=False,
                        level=level,
                        n_samples=n_samples
                    )
                ],
            axis=0
            )

        if outfile:
            outname = self.outdir + '/cdr_parameters.csv'
        else:
            outname = outfile

        parameter_table.to_csv(outname, index=False)

    def save_integral_table(self, random=True, level=95, n_samples='default', integral_n_time_units=None, outfile=None):
        """
        Save space-delimited table of IRF integrals (effect sizes) to the model's output directory

        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param outfile: ``str``; Path to output file. If ``None``, use model defaults.
        :return: ``str``; the IRF integrals report
        """

        if integral_n_time_units is None:
            integral_n_time_units = self.t_delta_limit

        if n_samples == 'default':
            if self.is_bayesian or self.has_dropout:
                n_samples = self.n_samples_eval

        irf_integrals = self.irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            n_time_units=integral_n_time_units,
            n_time_points=1000
        )

        if outfile:
            outname = self.outdir + '/cdr_irf_integrals.csv'
        else:
            outname = outfile

        irf_integrals.to_csv(outname, index=False)


class CDREnsemble(CDRModel):
    _doc_header = """
        Class implementing an ensemble of one or more continuous-time deconvolutional regression models.
    """
    _doc_args = """
        :param outdir_top: ``str``; path to the config file's top-level output directory (i.e. the ``outdir`` argument of the config)
        :param name: ``str``; name of the ensemble as defined in the config file
    \n"""

    __doc__ = _doc_header + _doc_args

    def __init__(self, outdir_top, name):

        self.weight_type = 'uniform'
        self.outdir_top = os.path.normpath(outdir_top)
        self.ensemble_name = name
        mpaths = [
            os.path.join(self.outdir_top,x) for x in os.listdir(self.outdir_top) if self.match_path(x)
        ]
        if not len(mpaths):
            mpaths = [os.path.join(self.outdir_top, name)]
        self.models = []
        for i, mpath in enumerate(mpaths):
            stderr('Loading model %s...\n' % mpath)
            self.models.append(load_cdr(mpath))

        assert self.models, 'An ensemble must contain at least one model. Exiting...'
        # self.__setstate__(self.models[0].__getstate__())
        self.model_index = self.sample_model_index()
        self.outdir = os.path.join(self.outdir_top, self.name)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return getattr(self.models[self.model_index], item)

    @property
    def name(self):
        return self.ensemble_name

    @property
    def n_ensemble(self):
        return len(self.models)

    def match_path(self, path):
        name = self.ensemble_name
        match = path.startswith(name)
        match &= os.path.isdir(os.path.join(self.outdir_top, path))

        if match:
            match = ENSEMBLE.match(path[len(name):])
            if not match:
                match = CROSSVAL.match(path[len(name):])
        else:
            match = False

        return match

    def set_weight_type(self, weight_type):
        if weight_type in ('uniform', 'll'):
            self.weight_type = weight_type
        else:
            raise ValueError('Unrecognized weight type "%s" for CDR ensemble.' % weight_type)

    def model_weights(self):
        if self.weight_type.lower() == 'uniform':
            weights = np.ones(self.n_ensemble) / self.n_ensemble
        elif self.weight_type.lower() == 'll':
            lls = []
            for m in self.models:
                lls.append(m.training_loglik_full)

            weights = logsumexp(lls)
        else:
            raise ValueError('Unrecognized weighting type for ensemble: %s.' % self.weight_type)

        return weights

    def sample_model_index(self):
        if len(self.models) == 1:
            ix = 0
        else:
            w = self.model_weights()
            ix = np.random.multinomial(1, w).argmax()
        return ix

    def resample_model(self):
        self.model_index = self.sample_model_index()

        if self.resample_ops:
            self.session.run(self.resample_ops)

    def load(self, *args, **kwargs):
        for model in self.models:
            model.load(*args, **kwargs)

    def save(self, *args, **kwargs):
        raise NotImplementedError('A CDREnsemble is not a trainable object and cannot be saved')

    def fit(self, *args, **kwargs):
        raise NotImplementedError('A CDREnsemble is not a trainable object and cannot be fitted')

    def build(self, *args, **kwargs):
        raise NotImplementedError('A CDREnsemble is not a trainable object and cannot be built')
