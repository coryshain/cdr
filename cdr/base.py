import textwrap
import time as pytime
import scipy.stats
import scipy.signal
import scipy.interpolate
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

from .kwargs import MODEL_INITIALIZATION_KWARGS, MODEL_BAYES_INITIALIZATION_KWARGS
from .formula import *
from .util import *
from .data import build_CDR_impulse_data, build_CDR_response_data, corr, corr_cdr, get_first_last_obs_lists, \
                  split_cdr_outputs
from .opt import *
from .plot import *

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, SinhArcsinh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


class Model(object):
    _INITIALIZATION_KWARGS = MODEL_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for deconvolutional models.
        ``Model`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``Model`` must implement the following instance methods:

            * ``initialize_model()``
            * ``compile_network()``
            * ``run_train_step()``

        Additionally, if the subclass requires any keyword arguments beyond those provided by ``Model``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
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

    N_QUANTILES = 41
    PLOT_QUANTILE_RANGE = 0.9
    PLOT_QUANTILE_IX = int((1 - PLOT_QUANTILE_RANGE) / 2 * N_QUANTILES)
    PREDICTIVE_DISTRIBUTIONS = {
        'normal': {
            'dist': Normal,
            'name': 'normal',
            'params': ('mu', 'sigma'),
            'support': 'real'
        },
        'sinharcsinh': {
            'dist': SinhArcsinh,
            'name': 'sinharcsinh',
            'params': ('mu', 'sigma', 'skewness', 'tailweight'),
            'support': 'real'
        },
        'bernoulli': {
            'dist': tf.contrib.distributions.Bernoulli,
            'name': 'bernoulli',
            'params': ('logit',),
            'support': 'discrete'
        },
        'categorical': {
            'dist': tf.contrib.distributions.Categorical,
            'name': 'categorical',
            'params': ('logit',),
            'support': 'discrete'
        }
    }

    def __new__(cls, *args, **kwargs):
        if cls is Model:
            raise TypeError("``Model`` is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form, X, Y, ablated=None, **kwargs):

        ## Store initialization settings
        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        assert self.n_samples == 1, 'n_samples is now deprecated and must be left at its default of 1'

        if not isinstance(X, list):
            X = [X]
        if not isinstance(Y, list):
            Y = [Y]

        # Cross validation settings
        self.crossval_factor = kwargs['crossval_factor']
        del kwargs['crossval_factor']
        self.crossval_fold = kwargs['crossval_fold']

        # Plot default settings
        del kwargs['crossval_fold']
        self.irf_name_map = kwargs['irf_name_map']
        del kwargs['irf_name_map']

        # Parse and store model data from formula
        if isinstance(form, str):
            self.form_str = form
            form = Formula(form)
        else:
            self.form_str = str(form)
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
        for _response_name in Y_all:
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
                _mean = _response.mean(axis=0)
                _sd = _response.std(axis=0)
                _quantiles = np.quantile(_response, q, axis=0)
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

        # Collect stats for impulses
        impulse_means = {}
        impulse_sds = {}
        impulse_medians = {}
        impulse_quantiles = {}
        impulse_lq = {}
        impulse_uq = {}
        impulse_min = {}
        impulse_max = {}

        impulse_df_ix = []
        for impulse in self.form.t.impulses():
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
                        impulse_means[name] = column.mean()
                        impulse_sds[name] = column.std()
                        quantiles = np.quantile(column, q)
                        impulse_quantiles[name] = quantiles
                        impulse_medians[name] = np.quantile(column, 0.5)
                        impulse_lq[name] = np.quantile(column, 0.1)
                        impulse_uq[name] = np.quantile(column, 0.9)
                        impulse_min[name] = column.min()
                        impulse_max[name] = column.max()

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
                            column = df[impulse_names].product(axis=1)
                            impulse_means[name] = column.mean()
                            impulse_sds[name] = column.std()
                            quantiles = np.quantile(column, q)
                            impulse_quantiles[name] = quantiles
                            impulse_medians[name] = np.quantile(column, 0.5)
                            impulse_lq[name] = np.quantile(column, 0.1)
                            impulse_uq[name] = np.quantile(column, 0.9)
                            impulse_min[name] = column.min()
                            impulse_max[name] = column.max()
            if not found:
                raise ValueError('Impulse %s was not found in an input file.' % name)

            impulse_df_ix.append(i)
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

        self.response_to_df_ix = {}
        for _response in response_names:
            self.response_to_df_ix[_response] = []
            for i, _Y in enumerate(Y):
                if _response in _Y:
                    self.response_to_df_ix[_response].append(i)

        # Collect stats for temporal features
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
                    for j, (s, e) in enumerate(zip(_first_obs, _last_obs)):
                        _X_time_slice = _X_time[s:e]
                        t_delta = _Y_time[j] - _X_time_slice
                        t_deltas.append(t_delta)
                        t_delta_maxes.append(_Y_time[j] - _X_time[s])
        X_time = np.concatenate(X_time, axis=0)
        Y_time = np.concatenate(Y_time, axis=0)
        t_deltas = np.concatenate(t_deltas, axis=0)
        t_delta_maxes = np.array(t_delta_maxes)
        t_delta_quantiles = np.quantile(t_deltas, q)

        self.t_delta_limit = np.quantile(t_deltas, 0.75)
        self.t_delta_quantiles = t_delta_quantiles
        self.t_delta_max = t_deltas.max()
        self.t_delta_mean_max = t_delta_maxes.mean()
        self.t_delta_mean = t_deltas.mean()
        self.t_delta_sd = t_deltas.std()

        self.X_time_limit = np.quantile(X_time, 0.75)
        self.X_time_quantiles = np.quantile(X_time, q)
        self.X_time_max = X_time.max()
        self.X_time_mean = X_time.mean()
        self.X_time_sd = X_time.std()

        self.Y_time_quantiles = np.quantile(Y_time, q)
        self.Y_time_mean = Y_time.mean()
        self.Y_time_sd = Y_time.std()

        ## Set up hash table for random effects lookup
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
                    report += 'Having too few instances for some levels can lead to degenerate random effects estimates.\n'
                    stderr(report)
            vals = np.arange(len(keys), dtype=getattr(np, self.int_type))
            rangf_map = pd.DataFrame({'id': vals}, index=keys).to_dict()['id']
            self.rangf_map_base.append(rangf_map)
            self.rangf_n_levels.append(len(keys) + 1)

        self._initialize_session()
        tf.keras.backend.set_session(self.sess)

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        ## Compute secondary data from intialization settings

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

        self.summed_interactions = None
        self.output = {} # Key order: <response>; Value: nbatch x nparam x ndim tensor of predictive distribution parameters for the response
        self.X_conv = {} # Key order: <response>;
        self.layers = [] # CDRNN only, list of DNN layers
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
            if self.is_cdrnn or not x.startswith('DiracDelta'):
                for y in self.terminals_by_name[x].impulses():
                    self.non_dirac_impulses.add(y.name())
            self.terminal_names_to_ix[x] = i
            self.terminal_names_printable[x] = ':'.join([get_irf_name(x, self.irf_name_map) for y in x.split(':')])

        # Initialize predictive distribution metadata
        predictive_distribution = {}
        predictive_distribution_map = {}
        if self.predictive_distribution_map is not None:
            _predictive_distribution_map = self.predictive_distribution_map.split()
            has_delim = [';' in x for x in _predictive_distribution_map]
            assert np.all(has_delim) or (not np.any(has_delim) and len(has_delim) == len(self.response_names)), 'predictive_distribution_map must either contain a one-to-one list of distribution names, one per response variable, or a list of ``;``-delimited pairs mapping <response, distribution>.'
            for i, x in enumerate(_predictive_distribution_map):
                if has_delim[i]:
                    _response, _dist = x.split(';')
                    predictive_distribution_map[_response] = _dist
                else:
                    predictive_distribution_map[self.response_names[i]] = x

        for _response in self.response_names:
            if _response in predictive_distribution_map:
                predictive_distribution[_response] = predictive_distribution_map[_response]
            elif self.response_is_categorical[_response]:
                predictive_distribution[_response] = self.PREDICTIVE_DISTRIBUTIONS['categorical']
            elif self.asymmetric_error:
                predictive_distribution[_response] = self.PREDICTIVE_DISTRIBUTIONS['sinharcsinh']
            else:
                predictive_distribution[_response] = self.PREDICTIVE_DISTRIBUTIONS['normal']
        self.predictive_distribution_config = predictive_distribution

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
            assert ix_2_levelname[-1] is None, 'Non-null value found in rangf map for unknown level'
            ix_2_levelname[-1] = 'UNK'
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
            if self.is_cdrnn or self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
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

        if self.impulse_df_ix is None:
            self.impulse_df_ix = np.zeros(len(self.form.t.impulses()))
        self.impulse_df_ix = np.array(self.impulse_df_ix, dtype=self.INT_NP)
        self.impulse_df_ix_unique = sorted(list(set(self.impulse_df_ix)))
        self.n_impulse_df = len(self.impulse_df_ix_unique)
        self.impulse_indices = []
        for i in range(self.n_impulse_df):
            arange = np.arange(len(self.form.t.impulses()))
            ix = arange[np.where(self.impulse_df_ix == i)[0]]
            self.impulse_indices.append(ix)
        if self.response_to_df_ix is None:
            self.response_to_df_ix = {x: [0] for x in self.response_names}
        self.n_response_df = 0
        for _response in self.response_to_df_ix:
            self.n_response_df = max(self.n_response_df, max(self.response_to_df_ix[_response]))
        self.n_response_df += 1

        self.use_crossval = bool(self.crossval_factor)

        self.parameter_table_columns = ['Estimate']

        self.indicators = set()
        for x in self.indicator_names.split():
            self.indicators.add(x)

        m = self.impulse_means
        m = np.array([m[x] for x in self.impulse_names])
        self.impulse_means_arr = m
        while len(m.shape) < 3:
            m = m[None, ...]
        self.impulse_means_arr_expanded = m

        s = self.impulse_sds
        s = np.array([s[x] for x in self.impulse_names])
        self.impulse_sds_arr = s
        while len(s.shape) < 3:
            s = s[None, ...]
        self.impulse_sds_arr_expanded = s

        q = self.impulse_quantiles
        q = np.stack([q[x] for x in self.impulse_names], axis=1)
        self.impulse_quantiles_arr = q
        while len(s.shape) < 3:
            q = np.expand_dims(q, axis=1)
        self.impulse_quantiles_arr_expanded = q

        reference_map = {}
        for pair in self.reference_values.split():
            impulse_name, val = pair.split('=')
            reference = float(val)
            reference_map[impulse_name] = reference
        self.reference_map = reference_map
        for x in self.impulse_names:
            if not x in self.reference_map:
                if self.default_reference_type == 'mean' and not x in self.indicators:
                    self.reference_map[x] = self.impulse_means[x]
                else:
                    self.reference_map[x] = 0.
        r = self.reference_map
        r = np.array([r[x] for x in self.impulse_names])
        self.reference_arr = r

        plot_step_map = {}
        for pair in self.plot_step.split():
            impulse_name, val = pair.split('=')
            plot_step = float(val)
            plot_step_map[impulse_name] = plot_step
        self.plot_step_map = plot_step_map
        for x in self.impulse_names:
            if not x in self.plot_step_map:
                if x in self.indicators:
                    self.plot_step_map[x] = 1
                elif isinstance(self.plot_step_default, str) and self.plot_step_default.lower() == 'sd':
                    self.plot_step_map[x] = self.impulse_sds[x]
                else:
                    self.plot_step_map[x] = self.plot_step_default
        s = self.plot_step_map
        s = np.array([s[x] for x in self.impulse_names])
        self.plot_step_arr = s

        self.summed_interactions = None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.constraint.lower() == 'softplus':
                    self.constraint_fn = tf.nn.softplus
                    self.constraint_fn_np = lambda x: np.log(np.exp(x) + 1.)
                    self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
                    self.constraint_fn_inv_np = lambda x: np.log(np.exp(x) - 1.)
                elif self.constraint.lower() == 'square':
                    self.constraint_fn = tf.square
                    self.constraint_fn_np = np.square
                    self.constraint_fn_inv = tf.sqrt
                    self.constraint_fn_inv_np = np.sqrt
                elif self.constraint.lower() == 'abs':
                    self.constraint_fn = self._abs
                    self.constraint_fn_np = np.abs
                    self.constraint_fn_inv = tf.identity
                    self.constraint_fn_inv_np = lambda x: x
                else:
                    raise ValueError('Unrecognized constraint function %s' % self.constraint)

                self.intercept_init = {}
                for _response in self.response_names:
                    self.intercept_init[_response] = self._get_intercept_init(
                        _response,
                        has_intercept=self.has_intercept[None]
                    )

                if self.convergence_n_iterates and self.convergence_alpha is not None:
                    self.d0 = []
                    self.d0_names = []
                    self.d0_saved = []
                    self.d0_saved_update = []
                    self.d0_assign = []

                    self.convergence_history = tf.Variable(
                        tf.zeros([int(self.convergence_n_iterates / self.convergence_stride), 1]), trainable=False,
                        dtype=self.FLOAT_NP, name='convergence_history')
                    self.convergence_history_update = tf.placeholder(self.FLOAT_TF, shape=[
                        int(self.convergence_n_iterates / self.convergence_stride), 1],
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

    def __getstate__(self):
        md = self._pack_metadata()
        return md

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self._unpack_metadata(state)
        self._initialize_metadata()

        self.log_graph = False

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
            'outdir': self.outdir,
            'crossval_factor': self.crossval_factor,
            'crossval_fold': self.crossval_fold,
            'irf_name_map': self.irf_name_map
        }
        for kwarg in Model._INITIALIZATION_KWARGS:
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
        self.response_expanded_bounds = md.pop('response_expanded_bounds', {x: (i, i+1) for i, x in enumerate(self.form.response_names())})
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
        self.outdir = md.pop('outdir', './cdr_model/')
        self.crossval_factor = md.pop('crossval_factor', None)
        self.crossval_fold = md.pop('crossval_fold', [])
        self.irf_name_map = md.pop('irf_name_map', {})

        # Convert response statistics to vectors if needed (for backward compatibility)
        response_names = [x.name() for x in self.form.responses()]
        if not isinstance(self.Y_train_means, dict):
            self.Y_train_means = {x: self.form.self.Y_train_means[x] for x in response_names}
        if not isinstance(self.Y_train_sds, dict):
            self.Y_train_sds = {x: self.form.self.Y_train_sds[x] for x in response_names}
        if not isinstance(self.Y_train_quantiles, dict):
            self.Y_train_quantiles = {x: self.form.self.Y_train_quantiles[x] for x in response_names}

        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    @property
    def name(self):
        return os.path.basename(self.outdir)


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name='training')

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
                    X_processed -= self.impulse_means_arr_expanded
                if self.rescale_inputs:
                    scale = self.impulse_sds_arr_expanded
                    scale = np.where(scale != 0, scale, 1.)
                    X_processed /= scale
                self.X_processed = X_processed
                self.X_time = tf.placeholder_with_default(
                    tf.zeros(
                        [
                            self.X_batch_dim,
                            self.history_length + self.future_length,
                            max(self.n_impulse, 1)
                        ],
                        dtype=self.FLOAT_TF
                    ),
                    shape=[None, None, max(self.n_impulse, 1)],
                    name='X_time'
                )
                self.X_mask = tf.placeholder_with_default(
                    tf.ones(
                        [
                            self.X_batch_dim,
                            self.history_length + self.future_length,
                            max(self.n_impulse, 1)
                        ],
                        dtype=self.FLOAT_TF
                    ),
                    shape=[None, None, max(self.n_impulse, 1)],
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
                    tf.ones([self.Y_batch_dim], dtype=self.FLOAT_TF),
                    shape=[None],
                    name=sn('Y_time')
                )
                self.Y_mask = tf.placeholder_with_default(
                    tf.ones([self.Y_batch_dim, self.n_response], dtype=self.FLOAT_TF),
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
                        tf.zeros((tf.shape(self.support)[0] - 1, 1), dtype=self.FLOAT_TF)
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

                self.training_complete = tf.Variable(
                    False,
                    trainable=False,
                    dtype=tf.bool,
                    name='training_complete'
                )
                self.training_complete_true = tf.assign(self.training_complete, True)
                self.training_complete_false = tf.assign(self.training_complete, False)

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
                    
                self.ranef_regularizer = self._initialize_regularizer(
                    self.ranef_regularizer_name,
                    self.ranef_regularizer_scale
                )

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                self.reg_loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='reg_loss_total')
                if self.is_bayesian:
                    self.kl_loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='kl_loss_total')
                self.n_dropped_in = tf.placeholder(shape=[], dtype=self.INT_TF, name='n_dropped_in')

                # Initialize vars for saving training set stats upon completion.
                # Allows these numbers to be reported in later summaries without access to the training data.
                self.training_loglik_full_in = tf.placeholder(
                    self.FLOAT_TF,
                    shape=[],
                    name='training_loglik_full_in'
                )
                self.training_loglik_full = tf.Variable(
                    np.nan,
                    dtype=self.FLOAT_TF,
                    trainable=False,
                    name='training_loglik_full'
                )
                self.set_training_loglik_full = tf.assign(self.training_loglik_full, self.training_loglik_full_in)
                self.training_loglik_in = {}
                self.training_loglik = {}
                self.set_training_loglik = {}
                self.training_mse_in = {}
                self.training_mse = {}
                self.set_training_mse = {}
                self.training_percent_variance_explained = {}
                self.training_rho_in = {}
                self.training_rho = {}
                self.set_training_rho = {}
                for response in self.response_names:
                    # log likelihood
                    self.training_loglik_in[response] = {}
                    self.training_loglik[response] = {}
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
                        self.training_loglik[response][ix] = tf.Variable(
                            np.nan,
                            dtype=self.FLOAT_TF,
                            trainable=False,
                            name='training_loglik_%s' % name_base
                        )
                        self.set_training_loglik[response][ix] = tf.assign(
                            self.training_loglik[response][ix],
                            self.training_loglik_in[response][ix]
                        )

                    if self.is_real(response):
                        self.training_mse_in[response] = {}
                        self.training_mse[response] = {}
                        self.set_training_mse[response] = {}
                        self.training_percent_variance_explained[response] = {}
                        self.training_rho_in[response] = {}
                        self.training_rho[response] = {}
                        self.set_training_rho[response] = {}

                        for ix in range(self.n_response_df):
                            # MSE
                            self.training_mse_in[response][ix] = tf.placeholder(
                                self.FLOAT_TF,
                                shape=[],
                                name='training_mse_in_%s' % name_base
                            )
                            self.training_mse[response][ix] = tf.Variable(
                                np.nan,
                                dtype=self.FLOAT_TF,
                                trainable=False,
                                name='training_mse_%s' % name_base
                            )
                            self.set_training_mse[response][ix] = tf.assign(
                                self.training_mse[response][ix],
                                self.training_mse_in[response][ix]
                            )

                            # % variance explained
                            full_variance = self.Y_train_sds[response] ** 2
                            if self.get_response_ndim(response) == 1:
                                full_variance = np.squeeze(full_variance, axis=-1)
                            self.training_percent_variance_explained[response][ix] = tf.maximum(
                                0.,
                                (1. - self.training_mse[response][ix] / full_variance) * 100.
                            )

                            # rho
                            self.training_rho_in[response][ix] = tf.placeholder(
                                self.FLOAT_TF,
                                shape=[], name='training_rho_in_%s' % name_base
                            )
                            self.training_rho[response][ix] = tf.Variable(
                                np.nan,
                                dtype=self.FLOAT_TF,
                                trainable=False,
                                name='training_rho_%s' % name_base
                            )
                            self.set_training_rho[response][ix] = tf.assign(
                                self.training_rho[response][ix],
                                self.training_rho_in[response][ix]
                            )

                # convergence
                self._add_convergence_tracker(self.loss_total, 'loss_total')
                self.converged_in = tf.placeholder(tf.bool, shape=[], name='converged_in')
                self.converged = tf.Variable(False, trainable=False, dtype=tf.bool, name='converged')
                self.set_converged = tf.assign(self.converged, self.converged_in)

                # Initialize regularizers
                if self.intercept_regularizer_name is None:
                    self.intercept_regularizer = None
                elif self.intercept_regularizer_name == 'inherit':
                    self.intercept_regularizer = self.regularizer
                else:
                    scale = self.intercept_regularizer_scale
                    if self.scale_regularizer_with_data:
                        scale *= self.minibatch_size * self.minibatch_scale
                    self.intercept_regularizer = getattr(tf.contrib.layers, self.intercept_regularizer_name)(scale)

                if self.ranef_regularizer_name is None:
                    self.ranef_regularizer = None
                elif self.ranef_regularizer_name == 'inherit':
                    self.ranef_regularizer = self.regularizer
                else:
                    scale = self.ranef_regularizer_scale
                    if isinstance(scale, str):
                        scale = [float(x) for x in scale.split(';')]
                    else:
                        scale = [scale]
                    if self.scale_regularizer_with_data:
                        scale = [x * self.minibatch_size * self.minibatch_scale for x in scale]
                    if self.ranef_regularizer_name == 'l1_l2_regularizer':
                        if len(scale) == 1:
                            scale_l1 = scale_l2 = scale[0]
                        else:
                            scale_l1 = scale[0]
                            scale_l2 = scale[1]
                        self.ranef_regularizer = getattr(tf.contrib.layers, self.ranef_regularizer_name)(
                            scale_l1,
                            scale_l2
                        )
                    else:
                        self.ranef_regularizer = getattr(tf.contrib.layers, self.ranef_regularizer_name)(scale[0])

                self.resample_ops = [] # Only used by CDRNN, defined here for global API
                self.regularizable_layers = [] # Only used by CDRNN, defined here for global API

                self.use_MAP_mode = tf.placeholder_with_default(tf.logical_not(self.training), shape=[], name='use_MAP_mode')

    def _get_intercept_init(self, response_name, has_intercept=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = []
                ndim = self.get_response_ndim(response_name)
                for param in self.get_response_params(response_name):
                    if param == 'mu':
                        if has_intercept and not self.standardize_response:
                            _out = self.Y_train_means[response_name][None, ...]
                        else:
                            _out = np.zeros((1, ndim))
                    elif param == 'sigma':
                        if self.standardize_response:
                            _out = self.constraint_fn_inv_np(np.ones((1, ndim)))
                        else:
                            _out = self.constraint_fn_inv_np(self.Y_train_sds[response_name][None, ...])
                    elif param == 'logit':
                        if has_intercept:
                            _out = np.log(self.Y_train_means[response_name][None, ...])
                        else:
                            _out = np.zeros((1, ndim))
                    else:
                        raise ValueError('Unrecognized predictive distributional parameter %s.' % param)

                    out.append(_out)

                out = np.concatenate(out, axis=0)

                return out

    def _initialize_base_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # Intercept
                # Key order: response, ?(ran_gf)
                self.intercept_fixed_base = {}
                self.intercept_fixed_base_summary = {}
                self.intercept_random_base = {}
                self.intercept_random_base_summary = {}
                for _response in self.response_names:
                    # Fixed
                    if self.has_intercept[None]:
                        intercept_fixed, intercept_fixed_summary = self.initialize_intercept(_response)
                    else:
                        intercept_fixed = intercept_fixed_summary = tf.constant(self.intercept_init[_response], dtype=self.FLOAT_TF)
                    self.intercept_fixed_base[_response] = intercept_fixed
                    self.intercept_fixed_base_summary[_response] = intercept_fixed_summary

                    # Random
                    for gf in self.rangf:
                        if self.has_intercept[gf]:
                            if _response not in self.intercept_random_base:
                                self.intercept_random_base[_response] = {}
                            if _response not in self.intercept_random_base_summary:
                                self.intercept_random_base_summary[_response] = {}
                            _intercept_random, _intercept_random_summary = self.initialize_intercept(_response, ran_gf=gf)
                            self.intercept_random_base[_response][gf] = _intercept_random
                            self.intercept_random_base_summary[_response][gf] = _intercept_random_summary

    def _compile_intercepts(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.intercept = {}
                self.intercept_summary = {}
                self.intercept_fixed = {}
                self.intercept_fixed_summary = {}
                self.intercept_random = {}
                self.intercept_random_summary = {}
                for response in self.response_names:
                    self.intercept[response] = {}
                    self.intercept_summary[response] = {}
                    self.intercept_fixed[response] = {}
                    self.intercept_fixed_summary[response] = {}
                    self.intercept_random[response] = {}
                    self.intercept_random_summary[response] = {}

                    # Fixed
                    response_params = self.get_response_params(response)
                    nparam = len(response_params)
                    ndim = self.get_response_ndim(response)

                    intercept_fixed = self.intercept_fixed_base[response]
                    intercept_fixed_summary = self.intercept_fixed_base_summary[response]

                    self._regularize(
                        intercept_fixed,
                        center=self.intercept_init[response],
                        regtype='intercept',
                        var_name='intercept_%s' % sn(response)
                    )

                    intercept = intercept_fixed[None, ...]
                    intercept_summary = intercept_fixed_summary[None, ...]

                    for i, response_param in enumerate(response_params):
                        dim_names = self._expand_param_name_by_dim(response, response_param)
                        _p = intercept_fixed[i]
                        _p_summary = intercept_fixed_summary[i]
                        if self.standardize_response and self.is_real(response):
                            if response_param == 'mu':
                                _p = _p * self.Y_train_sds[response] + self.Y_train_means[response]
                                _p_summary = _p_summary * self.Y_train_sds[response] + self.Y_train_means[response]
                            elif response_param == 'sigma':
                                _p = self.constraint_fn(_p) + self.epsilon
                                _p = _p * self.Y_train_sds[response]
                                _p_summary = self.constraint_fn(_p_summary) + self.epsilon
                                _p_summary = _p_summary * self.Y_train_sds[response]
                        if response_param == 'tailweight':
                            _p = self.constraint_fn(_p) + self.epsilon
                            _p_summary = self.constraint_fn(_p_summary) + self.epsilon
                        for j, dim_name in enumerate(dim_names):
                            val = _p[j]
                            val_summary = _p_summary[j]
                            if self.has_intercept[None]:
                                tf.summary.scalar(
                                    'intercept/%s_%s' % (sn(response), sn(dim_name)),
                                    val_summary,
                                    collections=['params']
                                )
                            self.intercept_fixed[response][dim_name] = val
                            self.intercept_fixed_summary[response][dim_name] = val_summary

                    # Random
                    for i, gf in enumerate(self.rangf):
                        # Random intercepts
                        if self.has_intercept[gf]:
                            self.intercept_random[response][gf] = {}
                            self.intercept_random_summary[response][gf] = {}

                            intercept_random = self.intercept_random_base[response][gf]
                            intercept_random_summary = self.intercept_random_base_summary[response][gf]

                            intercept_random_means = tf.reduce_mean(intercept_random, axis=0, keepdims=True)
                            intercept_random_summary_means = tf.reduce_mean(intercept_random_summary, axis=0, keepdims=True)

                            intercept_random -= intercept_random_means
                            intercept_random_summary -= intercept_random_summary_means

                            self._regularize(
                                intercept_random,
                                regtype='ranef',
                                var_name='intercept_%s_by_%s' % (sn(response), sn(gf))
                            )

                            for j, response_param in enumerate(response_params):
                                if j == 0 or self.use_distributional_regression:
                                    _p = intercept_random[:, j]
                                    _p_summary = intercept_random_summary[:, j]
                                    if self.standardize_response and self.is_real(response):
                                        if response_param == 'mu':
                                            _p = _p * self.Y_train_sds[response] + self.Y_train_means[response]
                                            _p_summary = _p_summary * self.Y_train_sds[response] + self.Y_train_means[response]
                                        elif response_param == 'sigma':
                                            _p = self.constraint_fn(_p) + self.epsilon
                                            _p = _p * self.Y_train_sds[response]
                                            _p_summary = self.constraint_fn(_p_summary) + self.epsilon
                                            _p_summary = _p_summary * self.Y_train_sds[response]
                                    dim_names = self._expand_param_name_by_dim(response, response_param)
                                    for k, dim_name in enumerate(dim_names):
                                        val = _p[:, k]
                                        val_summary = _p_summary[:, k]
                                        if self.log_random:
                                            tf.summary.histogram(
                                                'by_%s/intercept/%s_%s' % (sn(gf), sn(response), sn(dim_name)),
                                                val_summary,
                                                collections=['random']
                                            )
                                        self.intercept_random[response][gf][dim_name] = val
                                        self.intercept_random_summary[response][gf][dim_name] = val_summary

                            if not self.use_distributional_regression:
                                # Pad out any unmodeled params of predictive distribution
                                intercept_random = tf.pad(
                                    intercept_random,
                                    # ranef   pred param    pred dim
                                    [(0, 0), (0, nparam - 1), (0, 0)]
                                )
                                intercept_random_summary = tf.pad(
                                    intercept_random_summary,
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
                            intercept_random_summary = tf.concat(
                                [
                                    intercept_random_summary,
                                    tf.zeros([1, nparam, ndim])
                                ],
                                axis=0
                            )

                            intercept = intercept + tf.gather(intercept_random, self.Y_gf[:, i])
                            intercept_summary = intercept_summary + tf.gather(intercept_random_summary, self.Y_gf[:, i])

                    self.intercept[response] = intercept
                    self.intercept_summary[response] = intercept_summary

    def _initialize_predictive_distribution(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.predictive_distribution = {}
                self.predictive_distribution_delta = {} # IRF-driven changes in each parameter of the predictive distribution
                self.prediction = {}
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

                for i, response in enumerate(self.response_names):
                    self.predictive_distribution[response] = {}
                    self.predictive_distribution_delta[response] = {}
                    self.ll_by_var[response] = {}
                    self.error_distribution[response] = {}
                    self.error_distribution_theoretical_quantiles[response] = {}
                    self.error_distribution_theoretical_cdf[response] = {}
                    ndim = self.get_response_ndim(response)

                    _Y_mask = self.Y_mask[..., i]

                    pred_dist_fn = self.get_response_dist(response)
                    response_param_names = self.get_response_params(response)
                    response_params = self.intercept[response]
                    if self.summed_interactions:
                        response_params = response_params + self.summed_interactions[response]
                    output = self.output[response]
                    nparam = int(response_params.shape[-2])
                    if not self.use_distributional_regression:
                        # Pad out other predictive params
                        output = tf.pad(
                            output,
                            paddings = [
                                (0, 0),
                                (0, nparam - 1),
                                (0, 0)
                            ]
                        )

                    for j, response_param_name in enumerate(response_param_names):
                        dim_names = self._expand_param_name_by_dim(response, response_param_name)
                        for k, dim_name in enumerate(dim_names):
                            self.predictive_distribution_delta[response][dim_name] = output[:, j, k]

                    response_params = response_params + output
                    response_params = tf.unstack(response_params, axis=1)

                    # Post process response params
                    for j, response_param_name in enumerate(response_param_names):
                        _response_param = response_params[j]
                        if self.standardize_response and self.is_real(response):
                            if response_param_name in ['sigma', 'tailweight']:
                                _response_param = self.constraint_fn(_response_param) + self.epsilon
                            if response_param_name == 'mu':
                                _response_param = tf.cond(
                                    self.training,
                                    lambda: _response_param,
                                    lambda: _response_param * self.Y_train_sds[response] + self.Y_train_means[response]
                                )
                            elif response_param_name == 'sigma':
                                _response_param = tf.cond(
                                    self.training,
                                    lambda: _response_param,
                                    lambda: _response_param * self.Y_train_sds[response]
                                )
                        response_params[j] = _response_param

                    # Define predictive distribution
                    # Squeeze params if needed
                    if ndim == 1:
                        _response_params = [tf.squeeze(x, axis=-1) for x in response_params]
                    else:
                        _response_params = response_params
                    response_dist = pred_dist_fn(*_response_params)
                    self.predictive_distribution[response] = response_dist

                    # Define EMA over predictive distribution
                    beta = self.ema_decay
                    step = tf.cast(self.global_batch_step, self.FLOAT_TF)
                    response_params_ema_cur = []
                    # These will only ever be used in training mode, so un-standardize if needed
                    for j , response_param_name in enumerate(response_param_names):
                        _response_param = response_params[j]
                        if self.standardize_response and self.is_real(response):
                            if response_param_name == 'mu':
                                _response_param = _response_param * self.Y_train_sds[response] + self.Y_train_means[response]
                            elif response_param_name == 'sigma':
                                _response_param = _response_param * self.Y_train_sds[response]
                        response_params_ema_cur.append(_response_param)
                    response_params_ema_cur = tf.stack(response_params_ema_cur, axis=1)
                    response_params_ema_cur = tf.reduce_mean(response_params_ema_cur, axis=0)
                    self.response_params_ema[response] = tf.Variable(
                        tf.zeros((nparam, ndim)),
                        trainable=False,
                        name='response_params_ema_%s' % response
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
                        dim_names = self._expand_param_name_by_dim(response, response_param_name)
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

                    # Define EMA over X_conv
                    X_conv = tf.unstack(self.X_conv[response], axis=2) # X_conv is (batch, impulse, param, dim)
                    self.X_conv_ema[response] = {}
                    self.X_conv_ema_debiased[response] = {}
                    for j, response_param_name in enumerate(response_param_names):
                        _X_conv = X_conv[j]
                        if self.standardize_response and self.is_real(response) and response_param_name in ['mu', 'sigma']:
                            _X_conv = _X_conv * self.Y_train_sds[response]
                        _X_conv = tf.reduce_mean(_X_conv, axis=0)
                        dim_names = self._expand_param_name_by_dim(response, response_param_name)
                        for k, dim_name in enumerate(dim_names):
                            print(self.X_conv[response])
                            print(_X_conv)
                            X_conv_ema_cur = _X_conv[:, k]
                            self.X_conv_ema[response][dim_name] = tf.Variable(
                                tf.zeros_like(X_conv_ema_cur),
                                trainable=False,
                                name='response_params_ema_%s_%s' % (response, dim_name)
                            )
                            X_conv_ema_prev = self.X_conv_ema[response][dim_name]
                            X_conv_ema_debiased = X_conv_ema_prev / (1. - beta ** step)
                            self.X_conv_ema_debiased[response][dim_name] = X_conv_ema_debiased
                            ema_update = beta * X_conv_ema_prev + \
                                         (1. - beta) * X_conv_ema_cur
                            X_conv_ema_op = tf.assign(
                                self.X_conv_ema[response][dim_name],
                                ema_update
                            )
                            self.ema_ops.append(X_conv_ema_op)

                    # Define prediction tensors
                    dist_name = self.get_response_dist_name(response)
                    if dist_name == 'bernoulli':
                        self.prediction[response] = tf.cast(tf.round(_response_params[0]), self.INT_TF) * tf.cast(_Y_mask, self.INT_TF)
                    elif dist_name == 'categorical':
                        self.prediction[response] = tf.cast(tf.argmax(_response_params[0], axis=-1), self.INT_TF) * tf.cast(_Y_mask, self.INT_TF)
                    else: # Treat as continuous regression, use the first (location) parameter
                        self.prediction[response] = _response_params[0] * _Y_mask

                    # Define likelihoods
                    _Y = self.Y[..., i]
                    if self.standardize_response and self.is_real(response):
                        _Yz = (_Y - self.Y_train_means[response]) / self.Y_train_sds[response]
                        _Y = tf.cond(self.training, lambda: _Yz, lambda: _Y)

                    ll = response_dist.log_prob(_Y)
                    # Mask out likelihoods of predictions for missing response variables
                    ll *= _Y_mask
                    self.ll_by_var[response] = ll

                    # Define error distribution
                    if self.is_real(response):
                        empirical_quantiles = tf.linspace(0., 1., self.n_errors[response])
                        err_dist_params = []
                        for j, response_param_name in enumerate(response_param_names):
                            if j:
                                val = response_params_ema_debiased[j]
                            else:
                                val = tf.zeros(self.get_response_ndim(response))
                            err_dist_params.append(val)
                        if self.get_response_ndim(response) == 1:
                            err_dist_params = [tf.squeeze(x, axis=-1) for x in err_dist_params]
                        err_dist = pred_dist_fn(*err_dist_params)
                        err_dist_theoretical_quantiles = err_dist.quantile(empirical_quantiles)
                        err_dist_theoretical_cdf = err_dist.cdf(self.errors[response])

                        err_dist_plot = tf.exp(err_dist.log_prob(self.support))
                        err_dist_lb = err_dist.quantile(.025)
                        err_dist_ub = err_dist.quantile(.975)

                        self.error_distribution[response] = err_dist
                        self.error_distribution_theoretical_quantiles[response] = err_dist_theoretical_quantiles
                        self.error_distribution_theoretical_cdf[response] = err_dist_theoretical_cdf
                        self.error_distribution_plot[response] = err_dist_plot
                        self.error_distribution_plot_lb[response] = err_dist_lb
                        self.error_distribution_plot_ub[response] = err_dist_ub

                self.ll = tf.add_n([self.ll_by_var[x] for x in self.ll_by_var])

    def _initialize_regularizer(self, regularizer_name, regularizer_scale):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
                        scale = [x * self.minibatch_size * self.minibatch_scale for x in scale]
                    if regularizer_name == 'l1_l2_regularizer':
                        if len(scale) == 1:
                            scale_l1 = scale_l2 = scale[0]
                        else:
                            scale_l1 = scale[0]
                            scale_l2 = scale[1]
                        regularizer = getattr(tf.contrib.layers, regularizer_name)(
                            scale_l1,
                            scale_l2
                        )
                    else:
                        regularizer = getattr(tf.contrib.layers, regularizer_name)(scale[0])

                return regularizer

    def _initialize_optimizer(self):
        name = self.optim_name.lower()
        use_jtps = self.use_jtps

        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            t,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                optimizer_args = [self.lr]
                optimizer_kwargs = {}
                if name == 'momentum':
                    optimizer_args += [0.9]

                optimizer_class = {
                    'sgd': tf.train.GradientDescentOptimizer,
                    'momentum': tf.train.MomentumOptimizer,
                    'adagrad': tf.train.AdagradOptimizer,
                    'adadelta': tf.train.AdadeltaOptimizer,
                    'ftrl': tf.train.FtrlOptimizer,
                    'rmsprop': tf.train.RMSPropOptimizer,
                    'adam': tf.train.AdamOptimizer,
                    'nadam': tf.contrib.opt.NadamOptimizer,
                    'amsgrad': AMSGradOptimizer
                }[name]

                if clip:
                    optimizer_class = get_clipped_optimizer_class(optimizer_class, session=self.sess)
                    optimizer_kwargs['max_global_norm'] = clip

                if use_jtps:
                    optimizer_class = get_JTPS_optimizer_class(optimizer_class, session=self.sess)
                    optimizer_kwargs['meta_learning_rate'] = 1

                optim = optimizer_class(*optimizer_args, **optimizer_kwargs)

                return optim

    def _initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss_func = -self.ll

                # Filter
                if self.loss_filter_n_sds and self.ema_decay:
                    beta = self.ema_decay
                    ema_warm_up = 0
                    n_sds = self.loss_filter_n_sds
                    step = tf.cast(self.global_batch_step, self.FLOAT_TF)

                    self.loss_m1_ema = tf.Variable(0., trainable=False, name='loss_m1_ema')
                    self.loss_m2_ema = tf.Variable(0., trainable=False, name='loss_m2_ema')

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

                    loss_m1_ema_op = tf.assign(self.loss_m1_ema, loss_m1_ema_update)
                    loss_m2_ema_op = tf.assign(self.loss_m2_ema, loss_m2_ema_update)

                    self.ema_ops += [loss_m1_ema_op, loss_m2_ema_op]

                loss_func = tf.reduce_sum(loss_func)

                # Rescale
                if self.scale_loss_with_data:
                    loss_func = loss_func * self.minibatch_scale

                # Regularize
                for l in self.regularizable_layers: # CDRNN only
                    if hasattr(l, 'weights'):
                        vars = l.weights
                    else:
                        vars = [l]
                    for v in vars:
                        if 'bias' not in v.name:
                            self._regularize(v, regtype='nn', var_name=reg_name(v.name))
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

                self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('opt/loss_by_iter', self.loss_total, collections=['opt'])
                tf.summary.scalar('opt/reg_loss_by_iter', self.reg_loss_total, collections=['opt'])
                if self.is_bayesian:
                    tf.summary.scalar('opt/kl_loss_by_iter', self.kl_loss_total, collections=['opt'])
                if self.loss_filter_n_sds:
                    tf.summary.scalar('opt/n_dropped', self.n_dropped_in, collections=['opt'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr')
                self.summary_opt = tf.summary.merge_all(key='opt')
                self.summary_params = tf.summary.merge_all(key='params')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def _initialize_parameter_tables(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Fixed
                self.parameter_table_fixed_types = []
                self.parameter_table_fixed_responses = []
                self.parameter_table_fixed_response_params = []
                self.parameter_table_fixed_values = []
                if self.has_intercept[None]:
                    for response in self.intercept_fixed:
                        for dim_name in self.intercept_fixed[response]:
                            self.parameter_table_fixed_types.append('intercept')
                            self.parameter_table_fixed_responses.append(response)
                            self.parameter_table_fixed_response_params.append(dim_name)
                            self.parameter_table_fixed_values.append(
                                self.intercept_fixed[response][dim_name]
                            )

                # Random
                self.parameter_table_random_types = []
                self.parameter_table_random_responses = []
                self.parameter_table_random_response_params = []
                self.parameter_table_random_rangf = []
                self.parameter_table_random_rangf_levels = []
                self.parameter_table_random_values = []
                for response in self.intercept_random:
                    for r, gf in enumerate(self.rangf):
                        if gf in self.intercept_random[response] and self.has_intercept[gf]:
                            levels = sorted(self.rangf_map_ix_2_levelname[r][:-1])
                            for dim_name in self.intercept_random[response][gf]:
                                for l, level in enumerate(levels):
                                    self.parameter_table_random_types.append('intercept')
                                    self.parameter_table_random_responses.append(response)
                                    self.parameter_table_random_response_params.append(dim_name)
                                    self.parameter_table_random_rangf.append(gf)
                                    self.parameter_table_random_rangf_levels.append(level)
                                    self.parameter_table_random_values.append(
                                        self.intercept_random[response][gf][dim_name][l]
                                    )


    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.ema_vars = tf.get_collection('trainable_variables')
                self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay if self.ema_decay else 0.)
                ema_op = self.ema.apply(self.ema_vars)
                self.ema_ops.append(ema_op)
                self.ema_map = {}
                for v in self.ema_vars:
                    self.ema_map[self.ema.average_name(v)] = v
                self.ema_saver = tf.train.Saver(self.ema_map)

    def _initialize_convergence_checking(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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

    def _expand_param_name_by_dim(self, response, response_param):
        # Returns an empty list if the param is not used by the response.
        # Returns the unmodified param if the response is univariate.
        # Returns the concatenation "<param_name>.<dim_name>" if the response is multivariate.
        ndim = self.get_response_ndim(response)
        out = []
        if ndim == 1:
            if response_param in self.get_response_params(response):
                out.append(response_param)
        else:
            for i in range(ndim):
                cat = self.response_ix_to_category[response].get(i, i)
                out.append('%s.%s' % (response_param, cat))

        return out

    def _matmul(self, A, B):
        """
        Matmul operation that supports broadcasting of A

        :param A: Left tensor (>= 2D)
        :param B: Right tensor (2D)
        :return: Broadcasted matrix multiplication on the last 2 ranks of A and B
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                A_batch_shape = tf.gather(tf.shape(A), list(range(len(A.shape)-1)))
                A = tf.reshape(A, [-1, A.shape[-1]])
                C = tf.matmul(A, B)
                C_shape = tf.concat([A_batch_shape, [C.shape[-1]]], axis=0)
                C = tf.reshape(C, C_shape)
                return C

    def _tril_diag_ix(self, n):
        return (np.arange(1, n+1).cumsum() - 1).astype(self.INT_NP)

    def _scatter_along_axis(self, axis_indices, updates, shape, axis=0):
        # Except for axis, updates and shape must be identically shaped
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
        with self.sess.as_default():
            with self.sess.graph.as_default():
                f = tf.nn.softplus
                c = b - a

                g = (-f(-f(x - a) + c) + f(c)) * c / f(c) + a
                return g

    def _softplus_sigmoid_inverse(self, x, a=-1., b=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                f = tf.nn.softplus
                ln = tf.log
                exp = tf.exp
                c = b - a

                g = ln(exp(c) / ( (exp(c) + 1) * exp( -f(c) * (x - a) / c ) - 1) - 1) + a
                return g

    def _abs(self, x):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return tf.where(x > 0., x, -x)

    def _sigmoid(self, x, lb=0., ub=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return tf.sigmoid(x) * (ub - lb) + lb

    def _sigmoid_np(self, x, lb=0., ub=1.):
        return (1. / (1. + np.exp(-x))) * (ub - lb) + lb

    def _logit(self, x, lb=0., ub=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                x = (x - lb) / (ub - lb)
                x = x * (1 - 2 * self.epsilon) + self.epsilon
                return tf.log(x / (1 - x))

    def _logit_np(self, x, lb=0., ub=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                x = (x - lb) / (ub - lb)
                x = x * (1 - 2 * self.epsilon) + self.epsilon
                return np.log(x / (1 - x))

    def _piecewise_linear_interpolant(self, c, v):
        # c: knot locations, shape=[B, Q, K], B = batch, Q = query points or 1, K = n knots
        # v: knot values, shape identical to c
        with self.sess.as_default():
            with self.sess.graph.as_default():
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

    def _regularize(self, var, center=None, regtype=None, var_name=None):
        assert regtype in [None, 'intercept', 'coefficient', 'irf', 'ranef', 'nn', 'context', 'conv_output']
        if regtype is None:
            regularizer = self.regularizer
        else:
            regularizer = getattr(self, '%s_regularizer' % regtype)

        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if center is None:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    else:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var - center])
                    self.regularizer_losses.append(reg)
                    self.regularizer_losses_varnames.append(str(var_name))
                    if regtype is None:
                        reg_name = self.regularizer_name
                        reg_scale = self.regularizer_scale
                        if self.scale_regularizer_with_data:
                            reg_scale *= self.minibatch_size * self.minibatch_scale
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

    def _add_convergence_tracker(self, var, name, alpha=0.9):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.convergence_n_iterates:
                    # Flatten the variable for easy argmax
                    var = tf.reshape(var, [-1])
                    self.d0.append(var)

                    self.d0_names.append(name)

                    # Initialize tracker of parameter iterates
                    var_d0_iterates = tf.Variable(
                        tf.zeros([int(self.convergence_n_iterates / self.convergence_stride)] + list(var.shape)),
                        name=name + '_d0',
                        trainable=False
                    )

                    var_d0_iterates_update = tf.placeholder(self.FLOAT_TF, shape=var_d0_iterates.shape)
                    self.d0_saved.append(var_d0_iterates)
                    self.d0_saved_update.append(var_d0_iterates_update)
                    self.d0_assign.append(tf.assign(var_d0_iterates, var_d0_iterates_update))

    def _compute_and_test_corr(self, iterates):
        x = np.arange(0, len(iterates)*self.convergence_stride, self.convergence_stride).astype('float')[..., None]
        y = iterates

        n_iterates = int(self.convergence_n_iterates / self.convergence_stride)

        rt = corr(x, y)[0]
        tt = rt * np.sqrt((n_iterates - 2) / (1 - rt ** 2))
        p_tt = 1 - (scipy.stats.t.cdf(np.fabs(tt), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(tt), n_iterates - 2))
        p_tt = np.where(np.isfinite(p_tt), p_tt, np.zeros_like(p_tt))

        ra = corr(y[1:], y[:-1])[0]
        ta = ra * np.sqrt((n_iterates - 2) / (1 - ra ** 2))
        p_ta = 1 - (scipy.stats.t.cdf(np.fabs(ta), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(ta), n_iterates - 2))
        p_ta = np.where(np.isfinite(p_ta), p_ta, np.zeros_like(p_ta))

        return rt, p_tt, ra, p_ta

    def run_convergence_check(self, verbose=True, feed_dict=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.check_convergence:
                    min_p = 1.
                    min_p_ix = 0
                    rt_at_min_p = 0
                    ra_at_min_p = 0
                    p_ta_at_min_p = 0
                    fd_assign = {}

                    cur_step = self.global_step.eval(session=self.sess)
                    last_check = self.last_convergence_check.eval(session=self.sess)
                    offset = cur_step % self.convergence_stride
                    update = last_check < cur_step and self.convergence_stride > 0
                    if update and feed_dict is None:
                        update = False
                        stderr('Skipping convergence history update because no feed_dict provided.\n')

                    push = update and offset == 0
                    # End of stride if next step is a push
                    end_of_stride = last_check < (cur_step+1) and self.convergence_stride > 0 and ((cur_step+1) % self.convergence_stride == 0)

                    if self.check_convergence:
                        if update:
                            var_d0, var_d0_iterates = self.sess.run([self.d0, self.d0_saved], feed_dict=feed_dict)
                        else:
                            var_d0_iterates = self.sess.run(self.d0_saved)

                        start_ix = int(self.convergence_n_iterates / self.convergence_stride) - int(cur_step / self.convergence_stride)
                        start_ix = max(0, start_ix)

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

                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(iterates_d0[start_ix:])
                            else:
                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(var_d0_iterates[i][start_ix:])

                            new_min_p_ix = p_tt.argmin()
                            new_min_p = p_tt[new_min_p_ix]
                            if new_min_p < min_p:
                                min_p = new_min_p
                                min_p_ix = i
                                rt_at_min_p = rt[new_min_p_ix]
                                ra_at_min_p = ra[new_min_p_ix]
                                p_ta_at_min_p = p_ta[new_min_p_ix]

                        if update:
                            fd_assign[self.last_convergence_check_update] = self.global_step.eval(session=self.sess)
                            to_run = [self.d0_assign, self.last_convergence_check_assign]
                            self.sess.run(to_run, feed_dict=fd_assign)

                    if end_of_stride:
                        locally_converged = cur_step > self.convergence_n_iterates and \
                                    (min_p > self.convergence_alpha)
                        convergence_history = self.convergence_history.eval(session=self.sess)
                        convergence_history[:-1] = convergence_history[1:]
                        convergence_history[-1] = locally_converged
                        self.sess.run(self.convergence_history_assign, {self.convergence_history_update: convergence_history})

                    if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                        fd_convergence = {
                                self.rho_t: rt_at_min_p,
                                self.p_rho_t: min_p
                            }
                        summary_convergence = self.sess.run(
                            self.summary_convergence,
                            feed_dict=fd_convergence
                        )
                        self.writer.add_summary(summary_convergence, self.global_step.eval(session=self.sess))

                    proportion_converged = self.proportion_converged.eval(session=self.sess)
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

                self.sess.run(self.set_converged, feed_dict={self.converged_in: converged})

                return min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged





    ######################################################
    #
    #  Public initialization methods that must be
    #  implemented by subclasses
    #
    ######################################################

    def initialize_model(self):
        """
        Initialize all required weight arrays and callables.

        :return: ``None``
        """

        raise NotImplementedError

    def compile_network(self):
        """
        Chain transforms from impulses to response predictions

        :return: ``None``
        """

        raise NotImplementedError

    def run_train_step(self, feed_dict):
        """
        Update the model from a batch of training data.

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                to_run = [self.train_op]
                to_run += self.ema_ops

                to_run += [self.loss_func, self.reg_loss]
                to_run_names = ['loss', 'reg_loss']

                if self.loss_filter_n_sds:
                    to_run_names.append('n_dropped')
                    to_run.append(self.n_dropped)

                if self.is_bayesian:
                    to_run_names.append('kl_loss')
                    to_run.append(self.kl_loss)

                out = self.sess.run(
                    to_run,
                    feed_dict=feed_dict
                )

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict

    ######################################################
    #
    #  Private model inspection methods
    #
    ######################################################

    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples_eval

        alpha = 100 - float(level)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    param_vector = self.parameter_table_fixed_values
                else:
                    param_vector = self.parameter_table_random_values

                samples = []
                for i in range(n_samples):
                    if self.resample_ops:
                        self.sess.run(self.resample_ops)
                    samples.append(self.sess.run(param_vector, feed_dict={self.use_MAP_mode: False}))
                samples = np.stack(samples, axis=1)

                mean = samples.mean(axis=1)
                lower = np.percentile(samples, alpha / 2, axis=1)
                upper = np.percentile(samples, 100 - (alpha / 2), axis=1)

                out = np.stack([mean, lower, upper], axis=1)

                self.set_predict_mode(False)

                return out




    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    @property
    def is_bayesian(self):
        """
        Whether the model is defined using variational Bayes.

        :return: ``bool``; whether the model is defined using variational Bayes.
        """

        return False

    @property
    def is_cdrnn(self):
        """
        Whether the model is a subtype of CDRNN.

        :return: ``bool``; whether the model is a subtype of CDRNN.
        """

        return False

    @property
    def has_dropout(self):
        """
        Whether the model uses dropout (CDRNN only).

        :return: ``bool``; whether the model uses dropout.
        """

        return False

    @property
    def is_mixed_model(self):
        """
        Whether the model is mixed (i.e. has any random effects).

        :return: ``bool``; whether the model is mixed.
        """

        return len(self.rangf) > 0

    def initialize_intercept(self, response_name, ran_gf=None):
        """
        Initialize intercepts, i.e. bias terms for each parameter of the predictive distribution of a response variable.
        Must be called separately for each response variable in multivariate models.

        :param response_name: ``str``; name of response variable for which to initialize intercepts
        :param ran_gf: ``str`` or ``None``; name of random grouping factor for which to initialize intercepts. If ``None``, initialize fixed effects.

        :return: pair of fixed and random intercept ``Variable``; fixed intercept has identical shape to **init**, random intercept has shape ``[rangf_n_levels] + init.shape``.
        """

        # MLE implementation, overridden by ModelBayes
        with self.sess.as_default():
            with self.sess.graph.as_default():
                init = self.intercept_init[response_name]
                name = sn(response_name)
                if ran_gf is None:
                    intercept = tf.Variable(
                        init,
                        dtype=self.FLOAT_TF,
                        name='intercept_%s' % name
                    )
                    intercept_summary = intercept
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.use_distributional_regression:
                        nparam = self.get_response_nparam(response_name)
                    else:
                        nparam = 1
                    ndim = self.get_response_ndim(response_name)
                    shape = [rangf_n_levels, nparam, ndim]
                    intercept = tf.Variable(
                        tf.zeros(shape, dtype=self.FLOAT_TF),
                        name='intercept_%s_by_%s' % (name, sn(ran_gf))
                    )
                    intercept_summary = intercept

                return intercept, intercept_summary

    def build(self, outdir=None, restore=True):
        """
        Construct the CDR(NN) network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param outdir: Output directory. If ``None``, inferred.
        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './cdr_model/'
        else:
            self.outdir = outdir

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs()
                self._initialize_base_params()
                self._compile_intercepts()
                self.initialize_model()
                self.compile_network()
                self._initialize_predictive_distribution()
                self._initialize_objective()
                self._initialize_parameter_tables()
                self._initialize_logging()
                self._initialize_ema()

                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
                self._initialize_saver()
                self.load(restore=restore)

                self._initialize_convergence_checking()

                # self.sess.graph.finalize()

    def check_numerics(self):
        """
        Check that all trainable parameters are finite. Throws an error if not.

        :return: ``None``
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                for op in self.check_numerics_ops:
                    self.sess.run(op)

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):
        """
        Save the CDR model.

        :param dir: ``str``; output directory. If ``None``, use model default.
        :return: ``None``
        """

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except:
                        stderr('Write failure during save. Retrying...\n')
                        pytime.sleep(1)
                        i += 1
                if i >= 10:
                    stderr('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a CDR checkpoint and/or initialize the CDR model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """

        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                if restore and os.path.exists(outdir + '/checkpoint'):
                    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround for missing vars
                    path = outdir + '/model.ckpt'
                    try:
                        self.saver.restore(self.sess, path)
                        if predict and self.ema_decay:
                            self.ema_saver.restore(self.sess, path)
                    except tf.errors.DataLossError:
                        stderr('Read failure during load. Trying from backup...\n')
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                        if predict:
                            self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
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
                            saver_tmp.restore(self.sess, path)

                            if predict:
                                self.ema_map = {}
                                for v in restore_vars:
                                    self.ema_map[self.ema.average_name(v)] = v
                                saver_tmp = tf.train.Saver(self.ema_map)
                                saver_tmp.restore(self.sess, path)

                        else:
                            raise err
                else:
                    if predict:
                        stderr('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def finalize(self):
        """
        Close the CDR instance to prevent memory leaks.

        :return: ``None``
        """

        self.sess.close()

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
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.load(predict=mode)

            self.predict_mode = mode

    def has_converged(self):
        """
        Check whether model has reached its automatic convergence criteria

        :return: ``bool``; whether the model has converged
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.check_convergence:
                    return self.sess.run(self.converged)
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if status:
                    self.sess.run(self.training_complete_true)
                else:
                    self.sess.run(self.training_complete_false)

    def get_response_dist(self, response):
        """
        Get the TensorFlow distribution class for the predictive distribution assigned to a given response.

        :param response: ``str``; name of response
        :return: TensorFlow distribution object; class of predictive distribution
        """

        return self.predictive_distribution_config[response]['dist']

    def get_response_dist_name(self, response):
        """
        Get name of the predictive distribution assigned to a given response.

        :param response: ``str``; name of response
        :return: ``str``; name of predictive distribution
        """

        return self.predictive_distribution_config[response]['name']

    def get_response_params(self, response):
        """
        Get tuple of names of parameters of the predictive distribution for a given response.

        :param response: ``str``; name of response
        :return: ``tuple`` of ``str``; parameters of predictive distribution
        """

        return self.predictive_distribution_config[response]['params']

    def get_response_support(self, response):
        """
        Get the name of the distributional support of the predictive distribution assigned to a given response

        :param response: ``str``; name of response
        :return: ``str``; label of distributional support
        """

        return self.predictive_distribution_config[response]['support']

    def get_response_nparam(self, response):
        """
        Get the number of parameters in the predictive distrbution assigned to a given response

        :param response: ``str``; name of response
        :return: ``int``; number of parameters in the predictive distribution
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

        return self.get_response_support(response) == 'real'

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
        Check whether a given parameter name is present in the predictive distrbution assigned to a given response

        :param response: ``str``; name of response
        :param param: ``str``; name of parameter to query
        :return: ``bool``; whether the parameter is present in the predictive distribution
        """

        return param in self.predictive_distribution_config[response]['params']

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

        formatters = {
            'IRF': left_justified_formatter(irf_integrals, 'IRF')
        }

        out = ' ' * indent + 'IRF INTEGRALS (EFFECT SIZES):\n'
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

    def report_impulse_types(self, indent=0):
        """
        Generate a string representation of types of impulses (transient or continuous) in the model.

        :param indent: ``int``; indentation level
        :return: ``str``; the impulse type report
        """

        out = ''
        out += ' ' * indent + 'IMPULSE TYPES:\n'

        t = self.t
        for x in t.terminals():
            out += ' ' * (indent + 2) + x.name() + ': ' + ('continuous' if x.cont else 'transient') + '\n'

        out += '\n'

        return out

    def report_n_params(self, indent=0):
        """
        Generate a string representation of the number of trainable model parameters

        :param indent: ``int``; indentation level
        :return: ``str``; the num. parameters report
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                assert len(self.regularizer_losses) == len(self.regularizer_losses_names) == len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 'Different numbers of regularized variables found in different places'

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
            out += ' ' * (indent+2) + 'Loglik:              %s\n' % loglik
        if f1 is not None:
            out += ' ' * (indent+2) + 'Macro F1:            %s\n' % f1
        if f1_baseline is not None:
            out += ' ' * (indent+2) + 'Macro F1 (baseline): %s\n' % f1_baseline
        if acc is not None:
            out += ' ' * (indent+2) + 'Accuracy:            %s\n' % acc
        if acc_baseline is not None:
            out += ' ' * (indent+2) + 'Accuracy (baseline): %s\n' % acc_baseline
        if mse is not None:
            out += ' ' * (indent+2) + 'MSE:                 %s\n' % mse
        if mae is not None:
            out += ' ' * (indent+2) + 'MAE:                 %s\n' % mae
        if rho is not None:
            out += ' ' * (indent+2) + 'r(true, pred):       %s\n' % rho
        if loss is not None:
            out += ' ' * (indent+2) + 'Loss:                %s\n' % loss
        if true_variance is not None:
            out += ' ' * (indent+2) + 'True variance:       %s\n' % true_variance
        if percent_variance_explained is not None:
            out += ' ' * (indent+2) + '%% var expl:    %.2f%%\n' % percent_variance_explained
        if ks_results is not None:
            out += ' ' * (indent+2) + 'Kolmogorov-Smirnov test of goodness of fit of modeled to true error:\n'
            out += ' ' * (indent+4) + 'D value: %s\n' % ks_results[0]
            out += ' ' * (indent+4) + 'p value: %s\n' % ks_results[1]
            if ks_results[1] < 0.05:
                out += '\n'
                out += ' ' * (indent+4) + 'NOTE: KS tests will likely reject on large datasets.\n'
                out += ' ' * (indent+4) + 'This does not entail that the model is fatally flawed.\n'
                out += ' ' * (indent+4) + "Check the Q-Q plot in the model's output directory.\n"
                if not self.asymmetric_error:
                    out += ' ' * (indent+4) + 'Poor error fit can usually be improved without transforming\n'
                    out += ' ' * (indent+4) + 'the response by optimizing using ``asymmetric_error=True``.\n'
                    out += ' ' * (indent+4) + 'Consult the documentation for details.\n'

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
        out += '\n' + ' ' * (indent + 2) + 'Training iterations completed: %d\n\n' %self.global_step.eval(session=self.sess)
        out += self.report_irf_tree(indent=indent+2)
        out += self.report_impulse_types(indent=indent+2)
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
            out += ' ' * indent + 'Full loglik: %s\n\n' % self.training_loglik_full.eval(session=self.sess)

        for response in self.response_names:
            file_ix = self.response_to_df_ix[response]
            multiple_files = len(file_ix) > 1
            out += ' ' * indent + 'Response variable: %s\n\n' % response
            for ix in file_ix:
                if multiple_files:
                    out += ' ' * indent + 'File: %s\n\n' % ix
                out += ' ' * indent + 'MODEL EVALUATION STATISTICS:\n'
                out += ' ' * indent +     'Loglik:        %s\n' % self.training_loglik[response][ix].eval(session=self.sess)
                if response in self.training_mse:
                    out += ' ' * indent + 'MSE:           %s\n' % self.training_mse[response][ix].eval(session=self.sess)
                if response in self.training_rho:
                    out += ' ' * indent + 'r(true, pred): %s\n' % self.training_rho[response][ix].eval(session=self.sess)
                if response in self.training_percent_variance_explained:
                    out += ' ' * indent + '%% var expl:    %s\n' % self.training_percent_variance_explained[response][ix].eval(session=self.sess)
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
            n_iter = self.global_step.eval(session=self.sess)
            min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged = self.run_convergence_check(verbose=False)
            location = self.d0_names[min_p_ix]

            out += ' ' * (indent * 2) + 'Converged: %s\n' % converged
            out += ' ' * (indent * 2) + 'Convergence n iterates: %s\n' % self.convergence_n_iterates
            out += ' ' * (indent * 2) + 'Convergence stride: %s\n' % self.convergence_stride
            out += ' ' * (indent * 2) + 'Convergence alpha: %s\n' % self.convergence_alpha
            out += ' ' * (indent * 2) + 'Convergence min p of rho_t: %s\n' % min_p
            out += ' ' * (indent * 2) + 'Convergence rho_t at min p: %s\n' % rt_at_min_p
            out += ' ' * (indent * 2) + 'Proportion converged: %s\n' % proportion_converged

            if converged:
                out += ' ' * (indent + 2) + 'NOTE:\n'
                out += ' ' * (indent + 4) + 'Programmatic diagnosis of convergence in CDR is error-prone because of stochastic optimization.\n'
                out += ' ' * (indent + 4) + 'It is possible that the convergence diagnostics used are too permissive given the stochastic dynamics of the model.\n'
                out += ' ' * (indent + 4) + 'Consider visually checking the learning curves in Tensorboard to see whether the losses have flatlined:\n'
                out += ' ' * (indent + 6) + 'python -m tensorboard.main --logdir=<path_to_model_directory>\n'
                out += ' ' * (indent + 4) + 'If not, consider raising **convergence_alpha** and resuming training.\n'

            else:
                out += ' ' * (indent + 2) + 'Model did not reach convergence criteria in %s epochs.\n' % n_iter
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

    def fit(self,
            X,
            Y,
            X_in_Y_names=None,
            n_iter=10000,
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

            * ``time``: Timestamp associated with each observation in **y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **y**
            * Columns with a subset of the names of the DVs specified in ``form_str`` (all DVs should be represented somewhere in **y**)
            * A column for each random grouping factor in the model formula

            In general, **Y** will be identical to the parameter **Y** provided at model initialization.
        :param X_in_Y_names: ``list`` of ``str``; names of predictors contained in **Y** rather than **X** (must be present in all elements of **Y**). If ``None``, no such predictors.
        :param n_iter: ``int``; maximum number of training iterations. Training will stop either at convergence or **n_iter**, whichever happens first.
        :param force_training_evaluation: ``bool``; (Re-)run post-fitting evaluation, even if resuming a model whose training is already complete.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        """

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

        usingGPU = tf.test.is_gpu_available()
        stderr('Using GPU: %s\nNumber of training samples: %d\n\n' % (usingGPU, n))

        # Preprocess data
        if not isinstance(X, list):
            X = [X]
        if Y is not None and not isinstance(Y, list):
            Y = [Y]
        if self.use_crossval:
            Y = [_Y[self.crossval_factor].isin(self.crossval_folds) for _Y in Y]
        X_in = X
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

            # impulse_names = self.impulse_names
            # stderr('Correlation matrix for input variables:\n')
            # impulse_names_2d = [x for x in impulse_names if x in X_2d_predictor_names]
            # rho = corr_cdr(X, impulse_names, impulse_names_2d, X_time, X_mask)
            # stderr(str(rho) + '\n\n')

        if False:
            self.make_plots(prefix='plt')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.run_convergence_check(verbose=False)

                if (self.global_step.eval(session=self.sess) < n_iter) and not self.has_converged():
                    self.set_training_complete(False)

                if self.training_complete.eval(session=self.sess):
                    stderr('Model training is already complete; no additional updates to perform. To train for additional iterations, re-run fit() with a larger n_iter.\n\n')
                else:
                    if self.global_step.eval(session=self.sess) == 0:
                        if not type(self).__name__.startswith('CDRNN'):
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                            if self.log_random and self.is_mixed_model:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                            self.writer.flush()
                    else:
                        stderr('Resuming training from most recent checkpoint...\n\n')

                    if self.global_step.eval(session=self.sess) == 0:
                        stderr('Saving initial weights...\n')
                        self.save()

                    while not self.has_converged() and self.global_step.eval(session=self.sess) < n_iter:
                        p, p_inv = get_random_permutation(n)
                        t0_iter = pytime.time()
                        stderr('-' * 50 + '\n')
                        stderr('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        stderr('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            stderr('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                        loss_total = 0.
                        reg_loss_total = 0.
                        if self.is_bayesian:
                            kl_loss_total = 0.
                        if self.loss_filter_n_sds:
                            n_dropped = 0.

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

                            info_dict = self.run_train_step(fd)

                            self.check_numerics()

                            if self.loss_filter_n_sds:
                                n_dropped += info_dict['n_dropped']

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

                            pb.update((i/minibatch_size)+1, values=pb_update)

                            # if self.global_batch_step.eval(session=self.sess) % 1000 == 0:
                            #     self.save()
                            #     self.make_plots(prefix='plt')

                        self.sess.run(self.incr_global_step)

                        if self.check_convergence:
                            self.run_convergence_check(verbose=False, feed_dict={self.loss_total: loss_total/n_minibatch})

                        if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                            loss_total /= n_minibatch
                            reg_loss_total /= n_minibatch
                            log_fd = {self.loss_total: loss_total, self.reg_loss_total: reg_loss_total}
                            if self.is_bayesian:
                                kl_loss_total /= n_minibatch
                                log_fd[self.kl_loss_total] = kl_loss_total
                            if self.loss_filter_n_sds:
                                log_fd[self.n_dropped_in] = n_dropped
                            summary_train_loss = self.sess.run(self.summary_opt, feed_dict=log_fd)
                            self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                            if self.log_random and self.is_mixed_model:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                            self.writer.flush()

                        if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                            self.save()
                            self.make_plots(prefix='plt')

                        t1_iter = pytime.time()
                        if self.check_convergence:
                            stderr('Convergence:    %.2f%%\n' % (100 * self.sess.run(self.proportion_converged) / self.convergence_alpha))
                        stderr('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                    self.save()

                    # End of training plotting and evaluation.
                    # For CDRMLE, this is a crucial step in the model definition because it provides the
                    # variance of the output distribution for computing log likelihood.

                    self.make_plots(prefix='plt')

                    if self.is_bayesian or self.has_dropout:
                        # Generate plots with 95% credible intervals
                        self.make_plots(n_samples=self.n_samples_eval, prefix='plt')


                if not self.training_complete.eval(session=self.sess) or force_training_evaluation:
                    # Extract and save predictions
                    metrics, summary = self.evaluate(
                        X_in,
                        Y_in,
                        X_in_Y_names=X_in_Y_names,
                        dump=True,
                        partition='train'
                    )

                    # Extract and save losses
                    ll_full = sum([_ll for r in self.response_names for _ll in metrics['log_lik'][r]])
                    self.sess.run(self.set_training_loglik_full, feed_dict={self.training_loglik_full_in: ll_full})
                    fd = {}
                    to_run = []
                    for response in self.training_loglik_in:
                        for ix in self.training_loglik_in[response]:
                            tensor = self.training_loglik_in[response][ix]
                            fd[tensor] = metrics['log_lik'][response][ix]
                            to_run.append(self.set_training_loglik[response][ix])
                    for response in self.training_mse_in:
                        if self.is_real(response):
                            for ix in self.training_mse_in[response]:
                                tensor = self.training_mse_in[response][ix]
                                fd[tensor] = metrics['mse'][response][ix]
                                to_run.append(self.set_training_mse[response][ix])
                    for response in self.training_rho_in:
                        if self.is_real(response):
                            for ix in self.training_rho_in[response]:
                                tensor = self.training_rho_in[response][ix]
                                fd[tensor] = metrics['rho'][response][ix]
                                to_run.append(self.set_training_rho[response][ix])

                    self.sess.run(to_run, feed_dict=fd)
                    self.save()

                    self.save_parameter_table()
                    self.save_integral_table()

                    self.set_training_complete(True)

                    self.save()

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
        :param dump: ``bool``; whether to save generated predictions (and log likelihood vectors if applicable) to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert Y is not None or not return_loglik, 'Cannot return log likelihood when Y is not provided.'

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)
            stderr('Computing predictions...\n')

        if responses is None:
            responses = self.response_names
        if not isinstance(responses, list):
            responses = [responses]

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
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.set_predict_mode(True)

                    out = {}

                    if return_preds:
                        out['preds'] = {}
                        for _response in responses:
                            if self.is_real(_response):
                                dtype = self.FLOAT_NP
                            else:
                                dtype = self.INT_NP
                            out['preds'][_response] = np.zeros((n,), dtype=dtype)
                    if return_loglik:
                        out['log_lik'] = {x: np.zeros((n,)) for x in responses}

                    B = self.eval_minibatch_size
                    n_eval_minibatch = math.ceil(n / B)
                    for i in range(0, n, B):
                        if verbose:
                            stderr('\rMinibatch %d/%d' %((i/B)+1, n_eval_minibatch))
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
                            if return_loglik:
                                fd[self.Y] = _Y
                                fd[self.Y_mask]: _Y_mask
                        else:
                            fd = {
                                self.X: X[i:i + B],
                                self.X_time: X_time[i:i + B],
                                self.X_mask: X_mask[i:i + B],
                                self.Y_time: Y_time[i:i + B],
                                self.Y_gf: None if Y_gf is None else Y_gf[i:i + B],
                                self.training: not self.predict_mode
                            }
                            if return_loglik:
                                fd[self.Y] = Y[i:i + B]
                                fd[self.Y_mask]: Y_mask[i:i + B]
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
                                df.to_csv(self.outdir + '/CDRpreds_%s.csv' % name_base, sep=' ', na_rep='NaN', index=False)
        else:
            out = {}

        return out

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

        :param feed_dict: ``dict``; A dictionary of predictor values.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) of response variable(s) to predict. If ``None``, predicts all responses.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param return_preds: ``bool``; whether to return predictions.
        :param return_loglik: ``bool``; whether to return elementwise log likelihoods. Requires that **Y** is not ``None``.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``dict`` of ``numpy`` arrays; Predicted responses and/or log likelihoods, one for each training sample. Key order: <('preds'|'log_lik'), response>.
        """

        assert self.Y in feed_dict or not return_loglik, 'Cannot return log likelihood when Y is not provided.'

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict[self.use_MAP_mode] = use_MAP_mode

        if responses is None:
            responses = self.response_names
        if not isinstance(responses, list):
            responses = [responses]

        to_run = {}
        if return_preds:
            to_run_preds = {x: self.prediction[x] for x in responses}
            to_run['preds'] = to_run_preds
        if return_loglik:
            to_run_loglik = {x: self.ll_by_var[x] for x in responses}
            to_run['log_lik'] = to_run_loglik

        if to_run:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if use_MAP_mode:
                        out = self.sess.run(to_run, feed_dict=feed_dict)
                    else:
                        feed_dict[self.use_MAP_mode] = False
                        if n_samples is None:
                            n_samples = self.n_samples_eval

                        if verbose:
                            pb = tf.contrib.keras.utils.Progbar(n_samples)

                        out = {}
                        if return_preds:
                            out['preds'] = {x: np.zeros((len(feed_dict[self.Y_time]), n_samples)) for x in to_run_preds}
                        if return_loglik:
                            out['log_lik'] = {x: np.zeros((len(feed_dict[self.Y_time]), n_samples)) for x in
                                              to_run_loglik}

                        for i in range(n_samples):
                            if self.resample_ops:
                                self.sess.run(self.resample_ops)

                            _out = self.sess.run(to_run, feed_dict=feed_dict)
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

    def log_lik(
            self,
            X,
            Y,
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
        :param dump; ``bool``; whether to save generated log likelihood vectors to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param **kwargs; Any additional keyword arguments accepted by ``predict()`` (see docs for ``predict()`` for details).
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        out = self.predict(
            X,
            Y=Y,
            dump=False,
            return_preds=False,
            return_loglik=True,
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

    def evaluate(
            self,
            X,
            Y,
            X_in_Y_names=None,
            n_samples=None,
            algorithm='MAP',
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
        :param dump: ``bool``; whether to save generated data and evaluations to disk.
        :param extra_cols: ``bool``; whether to include columns from **Y** in output tables. Ignored unless **dump** is ``True``.
        :param partition: ``str`` or ``None``; name of data partition (or ``None`` if no partition name), used for output file naming. Ignored unless **dump** is ``True``.
        :param optimize_memory: ``bool``; Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: pair of <``dict``, ``str``>; Dictionary of evaluation metrics, human-readable evaluation summary string.
        """

        if partition and not partition.startswith('_'):
            partition_str = '_' + partition
        else:
            partition_str = ''

        cdr_out = self.predict(
            X,
            Y=Y,
            X_in_Y_names=X_in_Y_names,
            n_samples=n_samples,
            algorithm=algorithm,
            return_preds=True,
            return_loglik=True,
            dump=False,
            optimize_memory=optimize_memory,
            verbose=verbose
        )

        preds = cdr_out['preds']
        log_lik = cdr_out['log_lik']

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
            metrics['mse'][_response] = []
            metrics['rho'][_response] = []
            metrics['f1'][_response] = []
            metrics['f1_baseline'][_response] = []
            metrics['acc'][_response] = []
            metrics['acc_baseline'][_response] = []
            metrics['log_lik'][_response] = []
            metrics['percent_variance_explained'][_response] = []
            metrics['true_variance'][_response] = []
            metrics['ks_results'][_response] = []

            file_ix_all = list(range(len(Y)))
            file_ix = self.response_to_df_ix[_response]
            multiple_files = len(file_ix_all) > 1

            for ix in file_ix_all:
                metrics['mse'][_response].append(None)
                metrics['rho'][_response].append(None)
                metrics['f1'][_response].append(None)
                metrics['f1_baseline'][_response].append(None)
                metrics['acc'][_response].append(None)
                metrics['acc_baseline'][_response].append(None)
                metrics['log_lik'][_response].append(None)
                metrics['percent_variance_explained'][_response].append(None)
                metrics['true_variance'][_response].append(None)
                metrics['ks_results'][_response].append(None)

                if ix in file_ix:
                    _Y = Y[ix]
                    if _response in _Y:
                        _y = _Y[_response]

                        _preds = preds[_response][ix]

                        if self.is_binary(_response):
                            error = (_y == _preds).astype('int')
                            metrics['f1'][_response][-1] = f1_score(_y, _preds, average='binary')
                            metrics['f1_baseline'][_response][-1] = f1_score(_y, baseline, average='binary')
                            metrics['acc'][_response][-1] = accuracy_score(_y, _preds)
                            metrics['acc_baseline'][_response][-1] = accuracy_score(_y, baseline)
                            err_col_name = 'CDRcorrect'
                        elif self.is_categorical(_response):
                            error = (_y == _preds).astype('int')
                            classes, counts = np.unique(_y, return_counts=True)
                            majority = classes[np.argmax(counts)]
                            baseline = [majority] * len(_y)
                            metrics['f1'][_response][-1] = f1_score(_y, _preds, average='macro')
                            metrics['f1_baseline'][_response][-1] = f1_score(_y, baseline, average='macro')
                            metrics['acc'][_response][-1] = accuracy_score(_y, _preds)
                            metrics['acc_baseline'][_response][-1] = accuracy_score(_y, baseline)
                            err_col_name = 'CDRcorrect'
                        else:
                            error = np.array(_y - _preds) ** 2
                            score = error.mean()
                            resid = np.sort(_y - _preds)
                            resid_theoretical_q = self.error_theoretical_quantiles(len(resid), _response)
                            valid = np.isfinite(resid_theoretical_q)
                            resid = resid[valid]
                            resid_theoretical_q = resid_theoretical_q[valid]
                            D, p_value = self.error_ks_test(resid, _response)

                            metrics['mse'][_response][-1] = score
                            metrics['rho'][_response][-1] = np.corrcoef(_y, _preds, rowvar=False)[0, 1]
                            metrics['percent_variance_explained'][_response][-1] = percent_variance_explained(_y, _preds)
                            metrics['true_variance'][_response][-1] = np.std(_y) ** 2
                            metrics['ks_results'][_response][-1] = (D, p_value)
                            err_col_name = 'CDRsquarederror'
                    else:
                        err_col_name = error = _preds = _y = None

                    _ll = log_lik[_response][ix]
                    _ll_summed = _ll.sum()
                    metrics['log_lik'][_response][-1] = _ll_summed
                    metrics['full_log_lik'] += _ll_summed

                    if dump:
                        if multiple_files:
                            name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                        else:
                            name_base = '%s%s' % (sn(_response), partition_str)

                        df = {}
                        if err_col_name is not None and error is not None:
                            df[err_col_name] = error
                        if _preds is not None:
                            df['CDRpreds'] = _preds
                        if _y is not None:
                            df['CDRobs'] = _y
                        df['CDRloglik'] = _ll
                        df = pd.DataFrame(df)

                        if extra_cols:
                            df = pd.concat([_Y.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

                        preds_outfile = self.outdir + '/output_%s.csv' % name_base
                        df.to_csv(preds_outfile, sep=' ', na_rep='NaN', index=False)

                        if _response in self.predictive_distribution_config and self.is_real(_response):
                            plot_qq(
                                resid_theoretical_q,
                                resid,
                                dir=self.outdir,
                                filename='error_qq_plot_%s.png' % name_base,
                                xlab='Theoretical',
                                ylab='Empirical'
                            )

        summary_header = '=' * 50 + '\n'
        summary_header += 'CDR regression\n\n'
        summary_header += 'Model name: %s\n\n' % self.name
        summary_header += 'Formula:\n'
        summary_header += '  ' + self.form_str + '\n\n'
        summary_header += 'Partition: %s\n' % partition
        summary_header += 'Training iterations completed: %d\n\n' % self.global_step.eval(session=self.sess)
        summary_header += 'Full log likelihood: %s\n\n' % metrics['full_log_lik']

        summary = summary_header

        for _response in response_names:
            file_ix = self.response_to_df_ix[_response]
            multiple_files = len(file_ix) > 1
            for ix in file_ix:
                summary += 'Response variable: %s\n\n' % _response
                if dump:
                    _summary = summary_header
                    _summary += 'Response variable: %s\n\n' % _response

                if multiple_files:
                    summary += 'File index: %s\n\n' % ix
                    if dump:
                        name_base = '%s_f%s%s' % (sn(_response), ix, partition_str)
                        _summary += 'File index: %s\n\n' % ix
                elif dump:
                    name_base = '%s%s' % (sn(_response), partition_str)
                _summary = summary_header

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
                if dump:
                    _summary += summary_eval
                    _summary += '=' * 50 + '\n'
                    with open(self.outdir + '/eval_%s.txt' % name_base, 'w') as f_out:
                        f_out.write(_summary)

        summary += '=' * 50 + '\n'
        if verbose:
            stderr(summary)
            stderr('\n\n')

        return metrics, summary

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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if training is None:
                    training = not self.predict_mode
                    
                B = self.eval_minibatch_size
                n = sum([len(_Y) for _Y in Y])
                n_minibatch = math.ceil(n / B)
                loss = np.zeros((n,))
                for i in range(0, n, B):
                    if verbose:
                        stderr('\rMinibatch %d/%d' %(i+1, n_minibatch))
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
                            self.X: X[i:i + B],
                            self.X_time: X_time[i:i + B],
                            self.X_mask: X_mask[i:i + B],
                            self.Y_time: Y_time[i:i + B],
                            self.Y_mask: Y_mask[i:i + B],
                            self.Y_gf: None if Y_gf is None else Y_gf[i:i + B],
                            self.Y: Y[i:i + B],
                            self.training: training
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

    def run_loss_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        """
        Compute the elementwise training loss of a batch of data.

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; total training loss for batch
        """

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict[self.use_MAP_mode] = use_MAP_mode

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if use_MAP_mode:
                    loss = self.sess.run(self.loss_func, feed_dict=feed_dict)
                else:
                    feed_dict[self.use_MAP_mode] = False
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    loss = np.zeros((len(feed_dict[self.Y_time]), n_samples))

                    for i in range(n_samples):
                        if self.resample_ops:
                            self.sess.run(self.resample_ops)
                        loss[:, i] = self.sess.run(self.loss_func, feed_dict=feed_dict)
                        if verbose:
                            pb.update(i + 1)

                    loss = loss.mean(axis=1)

                return loss

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
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of predictive distribution(s) to convolve toward per response variable. Any param names not used by the predictive distribution for a given response will be ignored. If ``None``, convolves toward the first parameter of each response distribution.
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)
                B = self.eval_minibatch_size
                n_eval_minibatch = math.ceil(n / B)
                X_conv = {}
                for _response in responses:
                    X_conv[_response] = {}
                    for _response_param in response_params:
                        dim_names = self._expand_param_name_by_dim(_response, _response_param)
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

    def run_conv_op(self, feed_dict, responses=None, response_param=None, n_samples=None, algorithm='MAP', verbose=True):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.

        :param feed_dict: ``dict``; A dictionary of predictor variables
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to convolve toward. If ``None``, convolves toward all univariate responses. Multivariate convolution (e.g. of categorical responses) is supported but turned off by default to avoid excessive computation. When convolving toward a multivariate response, a set of convolved predictors will be generated for each dimension of the response.
        :param response_param: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of predictive distribution(s) to convolve toward per response variable. Any param names not used by the predictive distribution for a given response will be ignored. If ``None``, convolves toward the first parameter of each response distribution.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``dict`` of ``numpy`` arrays; The convolved inputs, one per **response_param** per **response**. Each element has shape (batch, terminals)
        """

        use_MAP_mode = algorithm in ['map', 'MAP']
        feed_dict[self.use_MAP_mode] = use_MAP_mode

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

        to_run = {}
        for _response in responses:
            to_run[_response] = self.X_conv[_response]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if use_MAP_mode:
                    X_conv = self.sess.run(to_run, feed_dict=feed_dict)
                else:
                    X_conv = {}
                    for _response in to_run:
                        nparam = self.get_response_nparam(_response)
                        ndim = self.get_response_ndim(_response)
                        X_conv[_response] = np.zeros(
                            (len(feed_dict[self.Y_time]), len(self.terminal_names), nparam, ndim, n_samples)
                        )

                    if n_samples is None:
                        n_samples = self.n_samples_eval
                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    for i in range(0, n_samples):
                        _X_conv = self.sess.run(to_run, feed_dict=feed_dict)
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
                            dim_names = self._expand_param_name_by_dim(_response, _response_param)
                            for j, _dim_name in enumerate(dim_names):
                                if _response not in out:
                                    out[_response] = {}
                                out[_response][_dim_name] = X_conv[_response][..., i, j]

                return out

    def error_theoretical_quantiles(
            self,
            n_errors,
            response
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)
                fd = {
                    self.n_errors[response]: n_errors,
                    self.training: not self.predict_mode
                }
                err_q = self.sess.run(self.error_distribution_theoretical_quantiles[response], feed_dict=fd)
                self.set_predict_mode(False)

                return err_q

    def error_theoretical_cdf(
            self,
            errors,
            response
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.errors[response]: errors,
                    self.training: not self.predict_mode
                }
                err_cdf = self.sess.run(self.error_distribution_theoretical_cdf[response], feed_dict=fd)

                return err_cdf

    def error_ks_test(
            self,
            errors,
            response
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
            reference_type=None,
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
        including all predictors (impulses), as well as ``'rate'``, ``'time_X'``, and ``'t_delta'``, respectively the
        deconvolutional intercept, the stimulus timestamp, and the delay from stimulus onset (i.e. the input to the
        IRF). The **manipulations** parameter supports arbitrary lambda functions on any combination of these variables,
        as well as on the random effects levels. Values for all of these variables can also be set for the reference
        response, enabling comparison to arbitrary references.

        Note that most of these queries are only of interest for CDRNN, since CDR assumes their structure (e.g.
        additive effects and non-stationarity). For CDR, the primary estimate of interest (the IRF) can be obtained by
        setting ``xvar = 't_delta'``, using a zero-vectored reference, and constructing a list of manipulations that
        adds ``1`` to each of the predictors independently.

        :param xvar: ``str``; Name of continuous variable for x-axis. Can be a predictor (impulse), ``'rate'``, ``'t_delta'``, or ``'time_X'``.
        :param yvar: ``str``; Name of continuous variable for y-axis in 3D plots. Can be a predictor (impulse), ``'rate'``, ``'t_delta'``, or ``'time_X'``. If ``None``, 2D plot.
        :param responses: ``list`` of ``str``, ``str``, or ``None``; Name(s) response variable(s) to plot.
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of predictive distribution(s) to plot per response variable. Any param names not used by the predictive distribution for a given response will be ignored.
        :param X_ref: ``dict`` or ``None``; Dictionary mapping impulse names to numeric values for use in constructing the reference. Any impulses not specified here will take default values.
        :param X_time_ref: ``float`` or ``None``; Timestamp to use for constructing the reference. If ``None``, use default value.
        :param t_delta_ref: ``float`` or ``None``; Delay/offset to use for constructing the reference. If ``None``, use default value.
        :param gf_y_ref: ``dict`` or ``None``; Dictionary mapping random grouping factor names to random grouping factor levels for use in constructing the reference. Any random effects not specified here will take default values.
        :param ref_varies_with_x: ``bool``; Whether the reference varies along the x-axis. If ``False``, use the scalar reference value for the x-axis.
        :param ref_varies_with_y: ``bool``; Whether the reference varies along the y-axis. If ``False``, use the scalar reference value for the y-axis. Ignored if **yvar** is ``None``.
        :param manipulations: ``list`` of ``dict``; A list of manipulations, where each manipulation is constructed as a dictionary mapping a variable name (e.g. ``'predictorX'``, ``'t_delta'``) to either a float offset or a function that transforms the reference value for that variable (e.g. multiplies it by ``2``). Alternatively, the keyword ``'ranef'`` can be used to manipulate random effects. The ``'ranef'`` entry must map to a ``dict`` that itself maps random grouping factor names (e.g. ``'subject'``) to levels (e.g. ``'subjectA'``).
        :param pair_manipulations: ``bool``; Whether to apply the manipulations to the reference input. If ``False``, all manipulations are compared to the same reference. For example, when plotting by-subject IRFs by subject, each subject might have a difference base response. In this case, set **pair_manipulations** to ``True`` in order to match the random effects used to compute the reference response and the response of interest.
        :param reference_type: ``bool``; Type of reference to use. If ``0``, use a zero-valued reference. If ``'mean'``, use the training set mean for all variables. If ``'default'``, use the default reference vector specified in the model's configuration file.
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
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
                else:
                    resample = False
                    n_samples = 1

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
                assert np.isscalar(X_time_ref), 'time_X_ref must be a scalar'
                X_time_ref = np.reshape(X_time_ref, (1, 1, 1))
                X_time_ref = np.tile(X_time_ref, [1, 1, max(n_impulse, 1)])

                # Initialize offset reference
                if t_delta_ref is None:
                    t_delta_ref = self.t_delta_mean
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
                time_X_base = None
                t_delta_base = None

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
                    if time_X_base is None:
                        time_X_base = np.tile(X_time_ref, (T, 1, 1))
                    if t_delta_base is None:
                        t_delta_base = np.tile(t_delta_ref, (T, 1, 1))

                    if axis_var in self.impulse_names_to_ix:
                        ix = self.impulse_names_to_ix[axis_var]
                        X_ref_mask = np.ones_like(X_ref)
                        X_ref_mask[..., ix] = 0
                        if axis is None:
                            qix = self.PLOT_QUANTILE_IX
                            lq = self.impulse_quantiles_arr[qix][ix]
                            uq = self.impulse_quantiles_arr[self.N_QUANTILES - qix - 1][ix]
                            select = np.isclose(uq - lq, 0)
                            while ix > 1 and np.any(select):
                                qix -= 1
                                lq = self.impulse_quantiles_arr[qix][ix]
                                uq = self.impulse_quantiles_arr[self.N_QUANTILES - qix - 1][ix]
                                select = np.isclose(uq - lq, 0)
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

                    if axis_var == 'time_X':
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
                        time_X_base = np.tile(axis[..., None, None], (1, 1, max(n_impulse, 1)))
                        if is_3d:
                            time_X_base = np.tile(time_X_base, tile_3d).reshape((T, 1, max(n_impulse, 1)))
                        if ref_varies:
                            X_time_ref = time_X_base

                    if axis_var == 't_delta':
                        if axis is None:
                            xinterval = self.plot_n_time_units
                            if ax_min is None:
                                ax_min = -xinterval * self.prop_fwd
                            if ax_max is None:
                                ax_min = xinterval * self.prop_bwd
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
                time_X_ref_in = [X_time_ref]
                t_delta_ref_in = [t_delta_ref]
                gf_y_ref_in = [gf_y_ref]

                if ref_as_manip: # Entails not pair_manipulations
                    X = []
                    time_X = []
                    t_delta = []
                    gf_y = []
                else: # Ref doesn't vary along all axes, so *_base contains full variation along all axes and is returned as the first manip
                    X = [X_base]
                    time_X = [time_X_base]
                    t_delta = [t_delta_base]
                    gf_y = [gf_y_base]

                for manipulation in manipulations:
                    X_cur = None
                    time_X_cur = time_X_base
                    t_delta_cur = t_delta_base
                    gf_y_cur = gf_y_base

                    if pair_manipulations:
                        X_ref_cur = None
                        time_X_ref_cur = X_time_ref
                        t_delta_ref_cur = t_delta_ref
                        gf_y_ref_cur = gf_y_ref

                    for k in manipulation:
                        if isinstance(manipulation[k], float) or isinstance(manipulation[k], int):
                            manip = lambda x, offset=float(manipulation[k]): x + offset
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
                        elif k == 'time_X':
                            time_X_cur = manip(time_X_cur)
                            if pair_manipulations:
                                time_X_ref_cur = time_X_cur
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
                    time_X.append(time_X_cur)
                    t_delta.append(t_delta_cur)
                    gf_y.append(gf_y_cur)

                    if pair_manipulations:
                        if X_ref_cur is None:
                            X_ref_cur = X_ref
                        X_ref_in.append(X_ref_cur)
                        time_X_ref_in.append(time_X_ref_cur)
                        t_delta_ref_in.append(t_delta_ref_cur)
                        gf_y_ref_in.append(gf_y_ref_cur)

                X_ref_in = np.concatenate(X_ref_in, axis=0)
                time_X_ref_in = np.concatenate(time_X_ref_in, axis=0)
                t_delta_ref_in = np.concatenate(t_delta_ref_in, axis=0)
                gf_y_ref_in = np.concatenate(gf_y_ref_in, axis=0)

                fd_ref = {
                    self.X: X_ref_in,
                    self.X_time: time_X_ref_in,
                    self.X_mask: np.ones_like(time_X_ref_in),
                    self.t_delta: t_delta_ref_in,
                    self.Y_gf: gf_y_ref_in,
                    self.training: not self.predict_mode
                }

                # Bring manipulations into 1-1 alignment on the batch dimension
                if n_manip:
                    X = np.concatenate(X, axis=0)
                    time_X = np.concatenate(time_X, axis=0)
                    t_delta = np.concatenate(t_delta, axis=0)
                    gf_y = np.concatenate(gf_y, axis=0)

                    fd_main = {
                        self.X: X,
                        self.X_time: time_X,
                        self.X_mask: np.ones_like(time_X),
                        self.t_delta: t_delta,
                        self.Y_gf: gf_y,
                        self.training: not self.predict_mode
                    }

                if resample:
                    fd_ref[self.use_MAP_mode] = False
                    if n_manip:
                        fd_main[self.use_MAP_mode] = False

                alpha = 100-float(level)

                samples = {}
                for i in range(n_samples):
                    to_run = {}
                    for _response in responses:
                        to_run[_response] = {}
                        for _response_param in response_params:
                            dim_names = self._expand_param_name_by_dim(_response, _response_param)
                            for _dim_name in dim_names:
                                to_run[_response][_dim_name] = self.predictive_distribution_delta[_response][_dim_name]

                    if self.resample_ops:
                        self.sess.run(self.resample_ops)
                    sample_ref = self.sess.run(to_run, feed_dict=fd_ref)
                    for _response in to_run:
                        for _dim_name in to_run[_response]:
                            _sample = sample_ref[_response][_dim_name]
                            sample_ref[_response][_dim_name] = np.reshape(_sample, ref_shape, 'F')

                    if n_manip:
                        sample = {}
                        sample_main = self.sess.run(to_run, feed_dict=fd_main)
                        for _response in to_run:
                            sample[_response] = {}
                            for _dim_name in sample_main[_response]:
                                sample_main[_response][_dim_name] = np.reshape(sample_main[_response][_dim_name], sample_shape, 'F')
                                sample_main[_response][_dim_name] = sample_main[_response][_dim_name] - sample_ref[_response][_dim_name]
                                if ref_as_manip:
                                    sample[_response][_dim_name] = np.concatenate(
                                        [sample_ref[_response][_dim_name], sample_main[_response][_dim_name]],
                                        axis=-1
                                    )
                                else:
                                    sample[_response][_dim_name] = sample_main[_response][_dim_name]
                    else:
                        sample = sample_ref
                    for _response in sample:
                        if not _response in samples:
                            samples[_response] = {}
                        for _dim_name in sample[_response]:
                            if not _dim_name in samples[_response]:
                                samples[_response][_dim_name] = []
                            samples[_response][_dim_name].append(sample[_response][_dim_name])

                lower = {}
                upper = {}
                mean = {}
                for _response in samples:
                    lower[_response] = {}
                    upper[_response] = {}
                    mean[_response] = {}
                    for _dim_name in samples[_response]:
                        _samples = np.stack(samples[_response][_dim_name], axis=0)
                        rescale = self.standardize_response and \
                                  self.is_real(_response) and \
                                  (_dim_name.startswith('mu') or _dim_name.startswith('sigma'))
                        if rescale:
                            _samples = _samples * self.Y_train_sds[_response]
                        samples[_response][_dim_name] = np.stack(_samples, axis=0)
                        _mean = _samples.mean(axis=0)
                        mean[_response][_dim_name] = _mean
                        if resample:
                            lower[_response][_dim_name] = np.percentile(_samples, alpha / 2, axis=0)
                            upper[_response][_dim_name] = np.percentile(_samples, 100 - (alpha / 2), axis=0)
                        else:
                            lower = upper = mean
                            samples[_response][_dim_name] = _mean[None, ...]

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

        axis = list(range(1, len(samples)))
        alpha = 100 - float(level)

        rmsd_mean = {}
        rmsd_lower = {}
        rmsd_upper = {}
        rmsd_samples = {}
        for response in samples:
            rmsd_mean[response] = {}
            rmsd_lower[response] = {}
            rmsd_upper[response] = {}
            rmsd_samples[response] = {}
            for dim_name in samples[response]:
                rmsd_samples[response] = ((gold - samples[response][dim_name])**2).mean(axis=axis)
                rmsd_mean[response][dim_name] = rmsd_samples.mean()
                rmsd_lower[response][dim_name] = rmsd_samples.percentile(alpha / 2)
                rmsd_upper[response][dim_name] = rmsd_samples.percentile(100 - (alpha / 2))

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
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of predictive distribution(s) to plot per response variable. Any param names not used by the predictive distribution for a given response will be ignored.
        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use mean/MLE model.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``pandas`` DataFrame; IRF integrals, one IRF per row. If Bayesian, array also contains credible interval bounds.
        """

        if n_time_units is None:
            n_time_units = self.t_delta_limit

        step = float(n_time_units) / n_time_points
        alpha = 100 - float(level)

        self.set_predict_mode(True)

        names = [get_irf_name(x, self.irf_name_map) for x in self.impulse_names]
        if self.is_cdrnn:
            names = [get_irf_name('rate', self.irf_name_map)] + names
        sort_key_dict = {x: i for i, x in enumerate(names)}

        def sort_key_fn(x, sort_key_dict=sort_key_dict):
            if x.name == 'IRF':
                return x.map(sort_key_dict)
            return x

        manipulations = []
        is_non_dirac = []
        if self.is_cdrnn:
            is_non_dirac.append(1.)
        for x in self.impulse_names:
            if self.is_non_dirac(x):
                is_non_dirac.append(1.)
            else:
                is_non_dirac.append(0.)
            delta = self.plot_step_map[x]
            manipulations.append({x: delta})
        is_non_dirac = np.array(is_non_dirac)[None, ...] # Add sample dim
        step = np.where(is_non_dirac, step, 1.)

        if random:
            ranef_group_names = self.ranef_group_names
            ranef_level_names = self.ranef_level_names
            ranef_zipped = zip(ranef_group_names, ranef_level_names)
            gf_y_refs = [{x: y} for x, y in ranef_zipped]
        else:
            gf_y_refs = [{None: None}]

        out = []

        if responses is None:
            responses = self.response_names
        if response_params is None:
            response_params = set()
            for _response in responses:
                response_params.add(self.get_response_params(_response)[0])
            response_params = sorted(list(response_params))

        if self.future_length:
            xmin = -n_time_units
        else:
            xmin = 0.

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
                xmax=n_time_units,
                xres=n_time_points,
                n_samples=n_samples,
                level=level,
            )

            for _response in vals:
                for _dim_name in vals[_response]:
                    _vals = vals[_response][_dim_name]
                    if not self.is_cdrnn:
                        _vals = _vals[..., 1:]

                    integrals = _vals.sum(axis=1) * step

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

    def make_plots(
            self,
            irf_name_map=None,
            responses=None,
            response_params=None,
            pred_names=None,
            sort_names=True,
            prop_cycle_length=None,
            prop_cycle_map=None,
            plot_dirac=False,
            reference_time=None,
            plot_rangf=False,
            plot_n_time_units=None,
            plot_n_time_points=None,
            reference_type=None,
            generate_univariate_IRF_plots=True,
            generate_curvature_plots=None,
            generate_irf_surface_plots=None,
            generate_nonstationarity_surface_plots=None,
            generate_interaction_surface_plots=None,
            generate_err_dist_plots=None,
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
            use_legend=None,
            use_line_markers=False,
            transparent_background=False,
            keep_plot_history=None,
            dump_source=False
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
        :param response_params: ``list`` of ``str``, ``str``, or ``None``; Name(s) of parameter of predictive distribution(s) to plot per response variable. Any param names not used by the predictive distribution for a given response will be ignored. If ``None``, plots the first parameter of each response distribution.
        :param summed: ``bool``; whether to plot individual IRFs or their sum.
        :param pred_names: ``list`` or ``None``; list of names of predictors to include in univariate IRF plots. If ``None``, all predictors are plotted.
        :param sort_names: ``bool``; whether to alphabetically sort IRF names.
        :param plot_unscaled: ``bool``; plot unscaled IRFs.
        :param plot_composite: ``bool``; plot any composite IRFs. If ``False``, only plots terminal IRFs.
        :param prop_cycle_length: ``int`` or ``None``; Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **pred_names**.
        :param prop_cycle_map: ``dict``, ``list`` of ``int``, or ``None``; Integer indices to use in the properties cycle for each entry in **pred_names**. If a ``dict``, a map from predictor names to ``int``. If a ``list`` of ``int``, predictors inferred using **pred_names** are aligned to ``int`` indices one-to-one. If ``None``, indices are automatically assigned.
        :param plot_dirac: ``bool``; whether to include any Dirac delta IRF's (stick functions at t=0) in plot.
        :param reference_time: ``float`` or ``None``; timepoint at which to plot interactions. If ``None``, use default setting.
        :param plot_rangf: ``bool``; whether to plot all (marginal) random effects.
        :param plot_n_time_units: ``float`` or ``None``; resolution of plot axis (for 3D plots, uses sqrt of this number for each axis). If ``None``, use default setting.
        :param plot_support_start: ``float`` or ``None``; start time for IRF plots. If ``None``, use default setting.
        :param reference_type: ``bool``; whether to use the predictor means as baseline reference (otherwise use zero).
        :param generate_univariate_IRF_plots: ``bool``; whether to plot univariate IRFs over time.
        :param generate_curvature_plots: ``bool`` or ``None``; whether to plot IRF curvature at time **reference_time**. If ``None``, use default setting.
        :param generate_irf_surface_plots: ``bool`` or ``None``; whether to plot IRF surfaces.  If ``None``, use default setting.
        :param generate_nonstationarity_surface_plots: ``bool`` or ``None``; whether to plot IRF surfaces showing non-stationarity in the response.  If ``None``, use default setting.
        :param generate_interaction_surface_plots: ``bool`` or ``None``; whether to plot IRF interaction surfaces at time **reference_time**.  If ``None``, use default setting.
        :param generate_err_dist_plots: ``bool`` or ``None``; whether to plot the average error distribution for real-valued responses.  If ``None``, use default setting.
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
        :param use_legend: ``bool`` or ``None``; whether to include a legend in plots with multiple components. If ``None``, use default setting.
        :param use_line_markers: ``bool``; whether to add markers to lines in univariate IRF plots.
        :param transparent_background: ``bool``; whether to use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool`` or ``None``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot. If ``None``, use default setting.
        :param dump_source: ``bool``; Whether to dump the plot source array to a csv file.
        :return: ``None``
        """

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

        if irf_name_map is None:
            irf_name_map = self.irf_name_map

        mc = bool(n_samples) and (self.is_bayesian or self.has_dropout)

        if reference_time is None:
            reference_time = self.reference_time
        if plot_n_time_units is None:
            plot_n_time_units = self.plot_n_time_units
        if plot_n_time_points is None:
            plot_n_time_points = self.plot_n_time_points
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

        with self.sess.as_default():
            with self.sess.graph.as_default():

                self.set_predict_mode(True)

                # IRF 1D
                if generate_univariate_IRF_plots:
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

                    manipulations = []
                    for x in names:
                        delta = self.plot_step_map[x]
                        manipulations.append({x: delta})
                    gf_y_refs = [{x: y} for x, y in zip(ranef_group_names, ranef_level_names)]

                    fixed_impulses = set()
                    for x in self.t.terminals():
                        if x.fixed:
                            for y in x.impulse_names():
                                fixed_impulses.add(y)

                    names_fixed = [x for x in names if x in fixed_impulses]
                    manipulations_fixed = [x for x in manipulations if list(x.keys())[0] in fixed_impulses]

                    if self.is_cdrnn:
                        if 'rate' not in names:
                            names = ['rate'] + names
                        if 'rate' not in names_fixed:
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
                            X_ref=None,
                            X_time_ref=None,
                            t_delta_ref=None,
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
                            level=level
                        )

                        for _response in plot_y:
                            for _dim_name in plot_y[_response]:
                                param_names = self.get_response_params(_response)

                                if _dim_name == param_names[0]:
                                    include_param_name = False
                                else:
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
                                    if include_param_name:
                                        ylab.append(_dim_name)
                                    ylab = ', '.join(ylab)
                                else:
                                    ylab = None

                                filename = prefix + plot_name

                                if ranef_level_names[g]:
                                    filename += '_' + ranef_level_names[g]
                                if mc:
                                    filename += '_mc'
                                filename += '.png'

                                _plot_y = plot_y[_response][_dim_name]
                                _lq = None if lq is None else lq[_response][_dim_name]
                                _uq = None if uq is None else uq[_response][_dim_name]

                                if not self.is_cdrnn:
                                    _plot_y = _plot_y[..., 1:]
                                    _lq = None if _lq is None else _lq[..., 1:]
                                    _uq = None if _uq is None else _uq[..., 1:]

                                plot_irf(
                                    plot_x,
                                    _plot_y,
                                    names_cur,
                                    lq=_lq,
                                    uq=_uq,
                                    sort_names=sort_names,
                                    prop_cycle_length=prop_cycle_length,
                                    prop_cycle_map=prop_cycle_map,
                                    dir=self.outdir,
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

                if plot_rangf:
                    manipulations = [{'ranef': {x: y}} for x, y in zip(ranef_group_names[1:], ranef_level_names[1:])]
                else:
                    manipulations = None

                # Curvature plots
                if generate_curvature_plots:
                    names = [x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]

                    for name in names:
                        plot_x, plot_y, lq, uq, samples = self.get_plot_data(
                            xvar=name,
                            responses=responses,
                            t_delta_ref=reference_time,
                            ref_varies_with_x=False,
                            manipulations=manipulations,
                            pair_manipulations=True,
                            reference_type=reference_type,
                            xres=plot_n_time_points,
                            n_samples=n_samples,
                            level=level
                        )

                        for _response in plot_y:
                            for _dim_name in plot_y[_response]:
                                param_names = self.get_response_params(_response)
                                if _dim_name == param_names[0]:
                                    include_param_name = False
                                else:
                                    include_param_name = True

                                plot_name = 'curvature_%s' % sn(_response)
                                if include_param_name:
                                    plot_name += '_%s' % _dim_name

                                plot_name += '_%s_at_delay%s' % (sn(name), reference_time)

                                if use_horiz_axlab:
                                    xlab = name
                                else:
                                    xlab = None
                                if use_vert_axlab:
                                    ylab = [get_irf_name(_response, irf_name_map)]
                                    if include_param_name:
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
                                    filename += '.png'

                                    plot_irf(
                                        plot_x,
                                        _plot_y[:, g:g+1],
                                        [name],
                                        lq=None if _lq is None else _lq[:, g:g+1],
                                        uq=None if _uq is None else _uq[:, g:g+1],
                                        dir=self.outdir,
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

                # Surface plots (CDRNN only)
                for plot_type, run_plot in zip(
                        ('irf_surface', 'nonstationarity_surface', 'interaction_surface',),
                        (generate_irf_surface_plots, generate_nonstationarity_surface_plots, generate_interaction_surface_plots)
                ):
                    if run_plot:
                        if plot_type == 'irf_surface':
                            names = ['t_delta:%s' % x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]
                        elif plot_type == 'nonstationarity_surface':
                            names = ['time_X:%s' % x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]
                        else: # plot_type == 'interaction_surface'
                            names_src = [x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]
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
                                    t_delta_ref=reference_time,
                                    ref_varies_with_x=ref_varies_with_x,
                                    manipulations=manipulations,
                                    pair_manipulations=True,
                                    reference_type=reference_type,
                                    xmin=xmin,
                                    xmax=xmax,
                                    xres=int(np.ceil(np.sqrt(plot_n_time_points))),
                                    yres=int(np.ceil(np.sqrt(plot_n_time_points))),
                                    n_samples=n_samples,
                                    level=level
                                )

                                for _response in plot_z:
                                    for _dim_name in plot_z[_response]:
                                        param_names = self.get_response_params(_response)
                                        if _dim_name == param_names[0]:
                                            include_param_name = False
                                        else:
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
                                            if include_param_name:
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
                                            filename += '.png'

                                            plot_surface(
                                                plot_x,
                                                plot_y,
                                                _plot_z[..., g],
                                                lq=None if _lq is None else _lq[..., g],
                                                uq=None if _uq is None else _uq[..., g],
                                                dir=self.outdir,
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
                            lb = self.sess.run(self.error_distribution_plot_lb[_response])
                            ub = self.sess.run(self.error_distribution_plot_ub[_response])
                            n_time_units = ub - lb
                            fd = {
                                self.support_start: lb,
                                self.n_time_units: n_time_units,
                                self.n_time_points: plot_n_time_points,
                                self.training: not self.predict_mode
                            }
                            plot_x = self.sess.run(self.support, feed_dict=fd)
                            plot_name = 'error_distribution_%s.png' % sn(_response)

                            plot_y = self.sess.run(self.error_distribution_plot[_response], feed_dict=fd)
                            lq = None
                            uq = None

                            plot_irf(
                                plot_x,
                                plot_y,
                                ['Error Distribution'],
                                lq=lq,
                                uq=uq,
                                dir=self.outdir,
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    types = self.parameter_table_fixed_types
                    responses = self.parameter_table_fixed_responses
                    response_params = self.parameter_table_fixed_response_params
                    values = self._extract_parameter_values(
                        fixed=True,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({'Parameter': types})
                    if len(self.response_names) > 1:
                        out['Response'] = responses
                    out['ResponseParam'] = response_params

                else:
                    types = self.parameter_table_random_types
                    responses = self.parameter_table_random_responses
                    response_params = self.parameter_table_random_response_params
                    rangf = self.parameter_table_random_rangf
                    rangf_levels = self.parameter_table_random_rangf_levels
                    values = self._extract_parameter_values(
                        fixed=False,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({
                        'Parameter': types,
                        'Group': rangf,
                        'Level': rangf_levels
                    })
                    if len(self.response_names) > 1:
                        out['Response'] = responses
                    out['ResponseParam'] = response_params

                columns = self.parameter_table_columns
                out = pd.concat([out, pd.DataFrame(values, columns=columns)], axis=1)

                self.set_predict_mode(False)

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


class ModelBayes(Model):
    _INITIALIZATION_KWARGS = MODEL_BAYES_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for variational Bayesian deconvolutional models.
        ``ModelBayes`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``ModelBayes`` must implement the following instance methods:

            * ``initialize_model()``
            * ``compile_network()``
            * ``run_train_step()``

        Additionally, if the subclass requires any keyword arguments beyond those provided by ``Model``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
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

    def __new__(cls, *args, **kwargs):
        if cls is ModelBayes:
            raise TypeError("``ModelBayes`` is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super(ModelBayes, self).__init__(
            *args,
            **kwargs
        )

        ## Store initialization settings
        for kwarg in ModelBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

    def _pack_metadata(self):
        md = super(ModelBayes, self)._pack_metadata()
        for kwarg in ModelBayes._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(ModelBayes, self)._unpack_metadata(md)

        for kwarg in ModelBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def _initialize_metadata(self):
        super(ModelBayes, self)._initialize_metadata()

        self.parameter_table_columns = ['Mean', '2.5%', '97.5%']

        self._intercept_prior_sd, \
        self._intercept_posterior_sd_init, \
        self._intercept_ranef_prior_sd, \
        self._intercept_ranef_posterior_sd_init = self._process_prior_sd(self.intercept_prior_sd)

    def _get_prior_sd(self, response_name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = []
                ndim = self.get_response_ndim(response_name)
                for param in self.get_response_params(response_name):
                    if param in ['mu', 'sigma']:
                        if self.standardize_response:
                            _out = np.ones((1, ndim))
                        else:
                            _out = self.Y_train_sds[response_name][None, ...]
                    else:
                        _out = np.ones((1, ndim))
                    out.append(_out)

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

        # Invert constraints
        for _response in self.response_names:
            prior_sd[_response] = self.constraint_fn_inv_np(prior_sd[_response])
            posterior_sd_init[_response] = self.constraint_fn_inv_np(posterior_sd_init[_response])
            ranef_prior_sd[_response] = self.constraint_fn_inv_np(ranef_prior_sd[_response])
            ranef_posterior_sd_init[_response] = self.constraint_fn_inv_np(ranef_posterior_sd_init[_response])

        # outputs all have shape [nparam, ndim]

        return prior_sd, posterior_sd_init, ranef_prior_sd, ranef_posterior_sd_init

    @property
    def is_bayesian(self):
        return True

    def initialize_intercept(self, response_name, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                init = self.intercept_init[response_name]

                name = sn(response_name)
                if ran_gf is None:
                    prior_sd = self._intercept_prior_sd[response_name]
                    post_sd = self._intercept_posterior_sd_init[response_name]

                    # Posterior distribution
                    intercept_q_loc = tf.Variable(
                        tf.constant(init, self.FLOAT_TF),
                        name='intercept_%s_q_loc' % name
                    )

                    intercept_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='intercept_%s_q_scale' % name
                    )

                    intercept_q_dist = Normal(
                        loc=intercept_q_loc,
                        scale=self.constraint_fn(intercept_q_scale) + self.epsilon,
                        name='intercept_%s_q' % name
                    )

                    intercept = tf.cond(self.use_MAP_mode, intercept_q_dist.mean, intercept_q_dist.sample)

                    intercept_summary = intercept_q_dist.mean()

                    if self.declare_priors_fixef:
                        # Prior distribution
                        intercept_prior_dist = Normal(
                            loc=tf.constant(init, self.FLOAT_TF),
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='intercept_%s' % name
                        )
                        self.kl_penalties['intercept_%s' % name] = {
                            'loc': init.flatten(),
                            'scale': self.constraint_fn_np(prior_sd).flatten(),
                            'val': intercept_q_dist.kl_divergence(intercept_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.use_distributional_regression:
                        nparam = self.get_response_nparam(response_name)
                    else:
                        nparam = 1
                    ndim = self.get_response_ndim(response_name)
                    shape = [rangf_n_levels, nparam, ndim]

                    prior_sd = self._intercept_ranef_prior_sd[response_name]
                    post_sd = self._intercept_ranef_posterior_sd_init[response_name]
                    if not self.use_distributional_regression:
                        post_sd = post_sd[:1]
                    prior_sd = np.ones((rangf_n_levels, 1, 1)) * prior_sd[None, ...]
                    post_sd = np.ones((rangf_n_levels, 1, 1)) * post_sd[None, ...]

                    # Posterior distribution
                    intercept_q_loc = tf.Variable(
                        tf.zeros(shape, dtype=self.FLOAT_TF),
                        name='intercept_%s_by_%s_q_loc' % (name, sn(ran_gf))
                    )

                    intercept_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='intercept_%s_by_%s_q_scale' % (name, sn(ran_gf))
                    )

                    intercept_q_dist = Normal(
                        loc=intercept_q_loc,
                        scale=self.constraint_fn(intercept_q_scale) + self.epsilon,
                        name='intercept_%s_by_%s_q' % (name, sn(ran_gf))
                    )

                    intercept = tf.cond(self.use_MAP_mode, intercept_q_dist.mean, intercept_q_dist.sample)

                    intercept_summary = intercept_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        intercept_prior_dist = Normal(
                            loc=0.,
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='intercept_%s_by_%s' % (name, sn(ran_gf))
                        )
                        self.kl_penalties['intercept_%s_by_%s' % (name, sn(ran_gf))] = {
                            'loc': 0.,
                            'scale': self.constraint_fn_np(self._intercept_ranef_prior_sd[response_name]).flatten(),
                            'val': intercept_q_dist.kl_divergence(intercept_prior_dist)
                        }

                # shape: (?rangf_n_levels, n_param, n_dim)

                return intercept, intercept_summary

    def report_regularized_variables(self, indent=0):
        """
        Generate a string representation of the model's regularization structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the regularization report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = super(ModelBayes, self).report_regularized_variables(indent)

                out += ' ' * indent + 'VARIATIONAL PRIORS:\n'
                out += ' ' * indent + '  NOTE: If using **standardize_response**, priors are reported\n'
                out += ' ' * indent + '        on the standardized scale where relevant.\n'

                kl_penalties = self.kl_penalties

                if len(kl_penalties) == 0:
                    out +=  ' ' * indent + '  No variational priors.\n\n'
                else:
                    for name in sorted(list(kl_penalties.keys())):
                        out += ' ' * indent + '  %s:\n' % name
                        for k in sorted(list(kl_penalties[name].keys())):
                            if not k == 'val':
                                out += ' ' * indent + '    %s: %s\n' % (k, kl_penalties[name][k])

                    out += '\n'

                return out