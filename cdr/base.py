import os
import textwrap
import time as pytime
import scipy.stats
import scipy.signal
import scipy.interpolate
import pandas as pd
from collections import defaultdict

from .kwargs import MODEL_INITIALIZATION_KWARGS
from .formula import *
from .util import *
from .data import build_CDR_impulses, corr_cdr, get_first_last_obs_lists
from .opt import *
from .plot import *


import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


def corr(A, B):
    # Assumes A and B are n x a and n x b matrices and computes a x b pairwise correlations
    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)

    A_ss = (A_centered ** 2).sum(axis=0)
    B_ss = (B_centered ** 2).sum(axis=0)

    rho = np.dot(A_centered.T, B_centered) / np.sqrt(np.dot(A_ss[..., None], B_ss[None, ...]))
    rho = np.clip(rho, -1, 1)
    return rho


class Model(object):
    _INITIALIZATION_KWARGS = MODEL_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for deconvolutional models.
        ``Model`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``Model`` must implement the following instance methods:

            * ``initialize_objective()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            * ``run_train_step()``

        Additionally, if the subclass requires any keyword arguments beyond those provided by ``Model``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
    """
    _doc_args = """
        :param form_str: An R-style string representing the model formula.
        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the ``form_str`` provided at initialization
        :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each observation in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each observation in ``y``
            * A column with the same name as the DV specified in ``form_str``
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

    def __new__(cls, *args, **kwargs):
        if cls is Model:
            raise TypeError("``Model`` is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form, X, y, ablated=None, **kwargs):

        ## Store initialization settings
        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        assert self.n_samples == 1, 'n_samples is now deprecated and must be left at its default of 1'

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
        form = form.categorical_transform(y)
        self.form = form
        dv = form.dv
        rangf = form.rangf

        # Store ablation info
        if ablated is None:
            self.ablated = set()
        elif isinstance(ablated, str):
            self.ablated = {ablated}
        else:
            self.ablated = set(ablated)

        q = np.linspace(0.0, 1, self.N_QUANTILES)
        q_plot = np.linspace(0.0, 1, 101)

        # Collect stats for response variable
        self.n_train = len(y)
        self.y_train_mean = float(y[dv].mean())
        self.y_train_sd = float(y[dv].std())
        self.y_train_quantiles = np.quantile(y[dv], q)

        # Collect stats for and kernel density estimators for impulses
        impulse_means = {}
        impulse_sds = {}
        impulse_medians = {}
        impulse_quantiles = {}
        impulse_lq = {}
        impulse_uq = {}
        impulse_min = {}
        impulse_max = {}
        # impulse_vectors = {}
        # densities = {}

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
                for i, df in enumerate(X + [y]):
                    if name in df.columns and not name.lower() == 'rate':
                        column = df[name].values
                        # impulse_vectors[name] = column
                        impulse_means[name] = column.mean()
                        impulse_sds[name] = column.std()
                        quantiles = np.quantile(column, q)
                        impulse_quantiles[name] = quantiles
                        impulse_medians[name] = np.quantile(column, 0.5)
                        impulse_lq[name] = np.quantile(column, 0.1)
                        impulse_uq[name] = np.quantile(column, 0.9)
                        impulse_min[name] = column.min()
                        impulse_max[name] = column.max()
                        # if name.lower() != 'rate' and False:
                        #     kde = scipy.stats.gaussian_kde(column)
                        #     density_support = np.linspace(quantiles[0], quantiles[-1], 100)
                        #     density = kde(density_support)
                        #     spline = scipy.interpolate.UnivariateSpline(density_support, density, ext='zeros', s=0.001)
                        #     densities[(name,)] = spline

                        found = True
                        break
                    elif is_interaction:
                        found = True
                        impulse_names = [x.name() for x in impulse.impulses()]
                        for x in impulse.impulses():
                            if not x.name() in df.columns:
                                found = False
                                break
                        if found:
                            column = df[impulse_names].product(axis=1)
                            # impulse_vectors[name] = column.values
                            impulse_means[name] = column.mean()
                            impulse_sds[name] = column.std()
                            quantiles = np.quantile(column, q)
                            impulse_quantiles[name] = quantiles
                            impulse_medians[name] = np.quantile(column, 0.5)
                            impulse_lq[name] = np.quantile(column, 0.1)
                            impulse_uq[name] = np.quantile(column, 0.9)
                            impulse_min[name] = column.min()
                            impulse_max[name] = column.max()
                            # kde = scipy.stats.gaussian_kde(column)
                            # density_support = np.linspace(quantiles[0], quantiles[-1], 100)
                            # density = kde(density_support)
                            # spline = scipy.interpolate.UnivariateSpline(density_support, density, ext='zeros', s=0.001)
                            # densities[(name,)] = spline

            if not found:
                raise ValueError('Impulse %s was not found in an input file.' % name)

            impulse_df_ix.append(i)
        self.impulse_df_ix = impulse_df_ix
        impulse_df_ix_unique = set(self.impulse_df_ix)

        # impulse_vector_names = list(impulse_vectors.keys())
        # for i in range(len(impulse_vectors)):
        #     name1 = impulse_vector_names[i]
        #     column1 = impulse_vectors[name1]
        #     for j in range(i + 1, len(impulse_vectors)):
        #         name2 = impulse_vector_names[j]
        #         column2 = impulse_vectors[name2]
        #         kde_support = np.stack([column1, column2], axis=0)
        #         kde = scipy.stats.gaussian_kde(kde_support)
        #         density_support1 = np.linspace(impulse_quantiles[name1][0], impulse_quantiles[name1][-1], 20)
        #         density_support2 = np.linspace(impulse_quantiles[name2][0], impulse_quantiles[name2][-1], 20)
        #         density_support1, density_support2 = np.meshgrid(density_support1, density_support2)
        #         density_support1 = density_support1.flatten()
        #         density_support2 = density_support2.flatten()
        #         density = kde(np.stack([density_support1, density_support2], axis=0))
        #         spline = scipy.interpolate.SmoothBivariateSpline(density_support1, density_support2, density, s=0.001)
        #         densities[(name1, name2)] = spline

        self.impulse_means = impulse_means
        self.impulse_sds = impulse_sds
        self.impulse_medians = impulse_medians
        self.impulse_quantiles = impulse_quantiles
        self.impulse_lq = impulse_lq
        self.impulse_uq = impulse_uq
        self.impulse_min = impulse_min
        self.impulse_max = impulse_max
        # self.impulse_vectors_train = impulse_vectors
        # self.densities = densities

        # Collect stats for temporal features
        t_deltas = []
        t_delta_maxes = []
        time_X = []
        first_obs, last_obs = get_first_last_obs_lists(y)
        y_time = y.time.values
        for i, cols in enumerate(zip(first_obs, last_obs)):
            if i in impulse_df_ix_unique or (not impulse_df_ix_unique and i == 0):
                first_obs_cur, last_obs_cur = cols
                first_obs_cur = np.array(first_obs_cur, dtype=getattr(np, self.int_type))
                last_obs_cur = np.array(last_obs_cur, dtype=getattr(np, self.int_type))
                time_X_cur = np.array(X[i].time, dtype=getattr(np, self.float_type))
                time_X.append(time_X_cur)
                for j, (s, e) in enumerate(zip(first_obs_cur, last_obs_cur)):
                    s = max(s, e - self.history_length)
                    time_X_slice = time_X_cur[s:e]
                    t_delta = y_time[j] - time_X_slice
                    t_deltas.append(t_delta)
                    t_delta_maxes.append(y_time[j] - time_X_cur[s])
        time_X = np.concatenate(time_X, axis=0)
        t_deltas = np.concatenate(t_deltas, axis=0)
        t_delta_maxes = np.array(t_delta_maxes)
        t_delta_quantiles = np.quantile(t_deltas, q)

        # kde = scipy.stats.gaussian_kde(t_deltas)
        # density_support = np.linspace(t_delta_quantiles[0], t_delta_quantiles[-1], 100)
        # density = kde(density_support)
        # spline = scipy.interpolate.UnivariateSpline(density_support, density, ext='zeros', s=0.001)
        # self.densities[('t_delta',)] = spline

        # self.t_delta_vector_train = t_deltas
        self.t_delta_limit = np.quantile(t_deltas, 0.75)
        self.t_delta_quantiles = t_delta_quantiles
        self.t_delta_max = t_deltas.max()
        self.t_delta_mean_max = t_delta_maxes.mean()
        self.t_delta_mean = t_deltas.mean()
        self.t_delta_sd = t_deltas.std()

        self.time_X_limit = np.quantile(time_X, 0.75)
        self.time_X_quantiles = np.quantile(time_X, q)
        self.time_X_max = time_X.max()
        self.time_X_mean = time_X.mean()
        self.time_X_sd = time_X.std()

        self.time_y_quantiles = np.quantile(y_time, q)
        self.time_y_mean = y_time.mean()
        self.time_y_sd = y_time.std()

        ## Set up hash table for random effects lookup
        self.rangf_map_base = []
        self.rangf_n_levels = []
        for i in range(len(rangf)):
            gf = rangf[i]
            keys = np.sort(y[gf].astype('str').unique())
            k, counts = np.unique(y[gf].astype('str'), return_counts=True)
            sd = counts.std()
            if np.isfinite(sd):
                mu = counts.mean()
                lb = mu - 2 * sd
                too_few = []
                for v, c in zip(k, counts):
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
        if not hasattr(self, 'is_bayesian'):
            self.is_bayesian = False
        if not hasattr(self, 'is_cdrnn'):
            self.is_cdrnn = False
            self.has_dropout = False

        ## Compute secondary data from intialization settings
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        f = self.form
        self.dv = f.dv
        self.has_intercept = f.has_intercept
        self.rangf = f.rangf
        self.ranef_group2ix = {x: i for i, x in enumerate(self.rangf)}
        self.is_mixed_model = len(self.rangf) > 0

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

        if self.pc:
            # Initialize source tree metadata
            self.t_src = self.form.t
            t_src = self.t_src
            self.src_impulse_names = t_src.impulse_names(include_interactions=True)
            self.src_terminal_names = t_src.terminal_names()
            self.src_terminals_by_name = t_src.terminals_by_name()
            self.impulse_names_to_ix = {}
            self.impulse_names_printable = {}
            for i, x in enumerate(self.src_impulse_names):
                self.impulse_names_to_ix[x] = i
                self.impulse_names_printable[x] = ':'.join([get_irf_name(x, self.irf_name_map) for y in x.split(':')])
            self.src_terminal_names = t_src.terminal_names()
            self.terminal_names_to_ix = {}
            self.terminal_names_printable = {}
            self.non_dirac_impulses = set()
            for i, x in enumerate(self.src_terminal_names):
                if self.is_cdrnn or not x.startswith('DiracDelta'):
                    for y in self.src_terminals_by_name[x].impulses():
                        self.non_dirac_impulses.add(y.name())
                self.terminal_names_to_ix[x] = i
                self.terminal_names_printable[x] = ':'.join([get_irf_name(x, self.irf_name_map) for y in x.split(':')])

            self.n_pc = len(self.src_impulse_names)
            self.has_rate = 'rate' in self.src_impulse_names
            if self.has_rate:
                self.n_pc -= 1
            pointers = {}
            self.fw_pointers, self.bw_pointers = IRFNode.pointers2namemmaps(pointers)
            self.form_pc = self.form.pc_transform(self.n_pc, pointers)
            self.t = self.form_pc.t
            t = self.t
            self.impulse_names = t.impulse_names(include_interactions=True)
            self.terminal_names = t.terminal_names()
            self.terminals_by_name = t.terminals_by_name()
        else:
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

        # Initialize model metadata

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

        if self.intercept_init is None:
            if self.standardize_response:
                self.intercept_init = 0.
            else:
                self.intercept_init = self.y_train_mean
        if self.y_sd_init is None:
            if self.standardize_response:
                self.y_sd_init = 1.
            else:
                self.y_sd_init = self.y_train_sd

        self.output_distr_params = ['loc', 'sd']
        if self.asymmetric_error:
            self.output_distr_params += ['skewness', 'tailweight']

        if self.impulse_df_ix is None:
            self.impulse_df_ix = np.zeros(len(self.form.t.impulses()))
        self.impulse_df_ix = np.array(self.impulse_df_ix, dtype=self.INT_NP)
        self.impulse_df_ix_unique = sorted(list(set(self.impulse_df_ix)))
        self.impulse_indices = []
        for i in range(len(self.impulse_df_ix_unique)):
            arange = np.arange(len(self.form.t.impulses()))
            ix = arange[np.where(self.impulse_df_ix == i)[0]]
            self.impulse_indices.append(ix)

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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.intercept_init_tf = tf.constant(self.intercept_init, dtype=self.FLOAT_TF)

                self.y_sd_init_tf = tf.constant(float(self.y_sd_init))
                if self.constraint.lower() == 'softplus':
                    self.constraint_fn = tf.nn.softplus
                    self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
                elif self.constraint.lower() == 'square':
                    self.constraint_fn = tf.square
                    self.constraint_fn_inv = tf.sqrt
                elif self.constraint.lower() == 'abs':
                    self.constraint_fn = self._abs
                    self.constraint_fn_inv = tf.identity
                else:
                    raise ValueError('Unrecognized constraint function %s' % self.constraint)
                self.y_sd_init_unconstrained = self.constraint_fn_inv(self.y_sd_init_tf)
                if self.asymmetric_error:
                    self.y_tailweight_init_unconstrained = self.constraint_fn_inv(1.)

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
            'y_train_mean': self.y_train_mean,
            'y_train_sd': self.y_train_sd,
            'y_train_quantiles': self.y_train_quantiles,
            # 't_delta_vector_train': self.t_delta_vector_train,
            't_delta_max': self.t_delta_max,
            't_delta_mean_max': self.t_delta_mean_max,
            't_delta_mean': self.t_delta_mean,
            't_delta_sd': self.t_delta_sd,
            't_delta_quantiles': self.t_delta_quantiles,
            't_delta_limit': self.t_delta_limit,
            'impulse_df_ix': self.impulse_df_ix,
            'time_X_max': self.time_X_max,
            'time_X_mean': self.time_X_mean,
            'time_X_sd': self.time_X_sd,
            'time_X_quantiles': self.time_X_quantiles,
            'time_X_limit': self.time_X_limit,
            'time_y_mean': self.time_y_mean,
            'time_y_sd': self.time_y_sd,
            'time_y_quantiles': self.time_y_quantiles,
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
            # 'impulse_vectors_train': self.impulse_vectors_train,
            # 'densities': self.densities
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
        self.y_train_mean = md.pop('y_train_mean')
        self.y_train_sd = md.pop('y_train_sd')
        self.y_train_quantiles = md.pop('y_train_quantiles', None)
        # self.t_delta_vector_train = md.pop('t_delta_vector_train', None)
        self.t_delta_max = md.pop('t_delta_max', md.pop('max_tdelta', None))
        self.t_delta_mean_max = md.pop('t_delta_mean_max', self.t_delta_max)
        self.t_delta_sd = md.pop('t_delta_sd', 1.)
        self.t_delta_mean = md.pop('t_delta_mean', 1.)
        self.t_delta_quantiles = md.pop('t_delta_quantiles', None)
        self.t_delta_limit = md.pop('t_delta_limit', self.t_delta_max)
        self.impulse_df_ix = md.pop('impulse_df_ix', None)
        self.time_X_max = md.pop('time_X_max', md.pop('max_time_X', None))
        self.time_X_sd = md.pop('time_X_sd', 1.)
        self.time_X_mean = md.pop('time_X_mean', 1.)
        self.time_X_quantiles = md.pop('time_X_quantiles', None)
        self.time_X_limit = md.pop('time_X_limit', self.t_delta_max)
        self.time_y_mean = md.pop('time_y_mean', 0.)
        self.time_y_sd = md.pop('time_y_sd', 1.)
        self.time_y_quantiles = md.pop('time_y_quantiles', None)
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
        # self.impulse_vectors_train = md.pop('impulse_vectors_train', {})
        # self.densities = md.pop('densities', {})
        self.outdir = md.pop('outdir', './cdr_model/')
        self.crossval_factor = md.pop('crossval_factor', None)
        self.crossval_fold = md.pop('crossval_fold', [])
        self.irf_name_map = md.pop('irf_name_map', {})

        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################
    
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

    def _initialize_inputs(self, n_impulse):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name='training')

                self.X = tf.placeholder(
                    shape=[None, None, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='X'
                )
                X_batch = tf.shape(self.X)[0]
                X_processed = self.X

                if self.center_inputs:
                    X_processed -= self.impulse_means_arr_expanded
                if self.rescale_inputs:
                    scale = self.impulse_sds_arr_expanded
                    scale = np.where(np.logical_not(np.isclose(scale, 0.)), scale, 1.)
                    X_processed /= scale
                # X_processed = tf.Print(X_processed, [X_processed], summarize=100)
                self.X_processed = X_processed

                self.X_batch = X_batch
                self.time_X = tf.placeholder_with_default(
                    tf.zeros([X_batch, self.history_length,  max(n_impulse, 1)], dtype=self.FLOAT_TF),
                    shape=[None, None, max(n_impulse, 1)],
                    name='time_X'
                )

                self.time_X_mask = tf.placeholder_with_default(
                    tf.ones([X_batch, self.history_length, max(n_impulse, 1)], dtype=self.FLOAT_TF),
                    shape=[None, None, max(n_impulse, 1)],
                    name='time_X_mask'
                )

                self.y = tf.placeholder(
                    shape=[None],
                    dtype=self.FLOAT_TF,
                    name=sn('y')
                )
                self.y_batch = tf.shape(self.y)[0]
                self.time_y = tf.placeholder_with_default(
                    tf.ones([self.y_batch], dtype=self.FLOAT_TF),
                    shape=[None],
                    name=sn('time_y')
                )

                # Tensor of temporal offsets with shape (?, history_length, 1)
                self.t_delta = self.time_y[..., None, None] - self.time_X
                self.gf_defaults = np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                self.gf_y = tf.placeholder_with_default(
                    tf.cast(self.gf_defaults, dtype=self.INT_TF),
                    shape=[None, len(self.rangf)],
                    name='gf_y'
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
                self.errors = tf.placeholder(self.FLOAT_TF, shape=[None], name='errors')
                self.n_errors = tf.placeholder(self.INT_TF, shape=[], name='n_errors')

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

                self.training_mse_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mse_in')
                self.training_mse = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mse')
                self.set_training_mse = tf.assign(self.training_mse, self.training_mse_in)
                self.training_percent_variance_explained = tf.maximum(
                    0.,
                    (1. - self.training_mse / (self.y_train_sd ** 2)) * 100.
                )

                self.training_mae_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mae_in')
                self.training_mae = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mae')
                self.set_training_mae = tf.assign(self.training_mae, self.training_mae_in)

                self.training_loglik_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_loglik_in')
                self.training_loglik = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_loglik')
                self.set_training_loglik = tf.assign(self.training_loglik, self.training_loglik_in)

                self.training_rho_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_rho_in')
                self.training_rho = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_rho')
                self.set_training_rho = tf.assign(self.training_rho, self.training_rho_in)

                if self.convergence_basis.lower() == 'loss':
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

                self.dropout_resample_ops = []

    def _process_objective(self, loss_func):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Filter
                if self.loss_filter_n_sds and self.ema_decay:
                    beta = self.ema_decay
                    ema_warm_up = 0
                    n_sds = self.loss_filter_n_sds
                    step = tf.cast(self.global_batch_step, self.FLOAT_TF)

                    self.loss_m1_ema = tf.Variable(0., trainable=False, name='loss_m1_ema')
                    self.loss_m2_ema = tf.Variable(0., trainable=False, name='loss_m2_ema')

                    # Debias
                    loss_m1_ema = self.loss_m1_ema / (1. - beta ** step )
                    loss_m2_ema = self.loss_m2_ema / (1. - beta ** step )

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

                    self.loss_m1_ema_op = tf.assign(self.loss_m1_ema, loss_m1_ema_update)
                    self.loss_m2_ema_op = tf.assign(self.loss_m2_ema, loss_m2_ema_update)

                loss_func = tf.reduce_sum(loss_func)

                # Rescale
                if self.scale_loss_with_data:
                    loss_func = loss_func * self.minibatch_scale

                # Regularize
                reg_loss = tf.constant(0., dtype=self.FLOAT_TF)
                # self.reg_loss = tf.Print(self.reg_loss, ['reg'] + self.regularizer_losses)
                if len(self.regularizer_losses_varnames) > 0:
                    reg_loss += tf.add_n(self.regularizer_losses)
                    loss_func += reg_loss

                kl_loss = tf.constant(0., dtype=self.FLOAT_TF)
                if self.is_bayesian and len(self.kl_penalties):
                    kl_loss += tf.reduce_sum([tf.reduce_sum(self.kl_penalties[k]['val']) for k in self.kl_penalties])
                    loss_func += kl_loss

                return loss_func, reg_loss, kl_loss

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__(*args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer

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
                parameter_table_fixed_keys = []
                parameter_table_fixed_values = []
                if self.has_intercept[None]:
                    parameter_table_fixed_keys.append('intercept')
                    parameter_table_fixed_values.append(
                        tf.expand_dims(self.intercept_fixed, axis=0)
                    )

                self.parameter_table_fixed_keys = parameter_table_fixed_keys
                self.parameter_table_fixed_values = tf.concat(parameter_table_fixed_values, 0)

                parameter_table_random_keys = []
                parameter_table_random_rangf = []
                parameter_table_random_rangf_levels = []
                parameter_table_random_values = []

                if len(self.rangf) > 0:
                    for i in range(len(self.rangf)):
                        gf = self.rangf[i]
                        levels = sorted(self.rangf_map_ix_2_levelname[i][:-1])
                        levels_ix = names2ix([self.rangf_map[i][level] for level in levels], range(self.rangf_n_levels[i]))
                        if self.has_intercept[gf]:
                            for level in levels:
                                parameter_table_random_keys.append('intercept')
                                parameter_table_random_rangf.append(gf)
                                parameter_table_random_rangf_levels.append(level)
                            parameter_table_random_values.append(
                                tf.gather(self.intercept_random[gf], levels_ix)
                            )

                    self.parameter_table_random_keys = parameter_table_random_keys
                    self.parameter_table_random_rangf = parameter_table_random_rangf
                    self.parameter_table_random_rangf_levels = parameter_table_random_rangf_levels
                    self.parameter_table_random_values = tf.concat(parameter_table_random_values, 0)

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
                self.ema_op = self.ema.apply(self.ema_vars)
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
                    if self.convergence_basis.lower() == 'parameters':
                        self.rho_a = tf.placeholder(self.FLOAT_TF, name='rho_a_in')
                        self.p_rho_a = tf.placeholder(self.FLOAT_TF, name='p_rho_a_in')

                    tf.summary.scalar('convergence/rho_t', self.rho_t, collections=['convergence'])
                    tf.summary.scalar('convergence/p_rho_t', self.p_rho_t, collections=['convergence'])
                    if self.convergence_basis.lower() == 'parameters':
                        tf.summary.scalar('convergence/rho_a', self.rho_a, collections=['convergence'])
                        tf.summary.scalar('convergence/p_rho_a', self.p_rho_a, collections=['convergence'])
                    tf.summary.scalar('convergence/proportion_converged', self.proportion_converged, collections=['convergence'])
                    self.summary_convergence = tf.summary.merge_all(key='convergence')




    ######################################################
    #
    #  Math subroutines
    #
    ######################################################

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

    def _reduce_interpolated_sum(self, X, time, axis=0):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                assert len(X.shape) > 0, 'A scalar cannot be interpolated'

                if axis < 0:
                    axis = len(X.shape) + axis

                X_cur_begin = [0] * (len(X.shape) - 1)
                X_cur_begin.insert(axis, 1)
                X_cur_end = [-1] * len(X.shape)
                X_cur = tf.slice(X, X_cur_begin, X_cur_end)

                time_cur_begin = [0] * axis + [1]
                time_cur_end = [-1] * (axis + 1)
                time_cur = tf.slice(time, time_cur_begin, time_cur_end)

                ub = tf.shape(X)[axis] - 1
                X_prev_begin = [0] * len(X.shape)
                X_prev_end = [-1] * (len(X.shape) - 1)
                X_prev_end.insert(axis, ub)
                X_prev = tf.slice(X, X_prev_begin, X_prev_end)

                time_prev_begin = [0] * (axis + 1)
                time_prev_end = [-1] * axis + [ub]
                time_prev = tf.slice(time, time_prev_begin, time_prev_end)

                time_diff = time_cur-time_prev

                for _ in range(axis+1, len(X.shape)):
                    time_diff = tf.expand_dims(time_diff, -1)

                out = tf.reduce_sum((X_prev + X_cur) / 2 * time_diff, axis=axis)

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

    def _linspace_nd(self, B, A=None, axis=0, n_interp=None):
        if n_interp is None:
            n_interp = self.n_interp
        if axis < 0:
            axis = len(B.shape) + axis
        with self.sess.as_default():
            with self.sess.graph.as_default():
                linspace_support = tf.cast(tf.range(n_interp), dtype=self.FLOAT_TF)
                B = tf.expand_dims(B, axis)
                rank = len(B.shape)
                assert axis < rank, 'Tried to perform linspace_nd on axis %s, which exceeds rank %s of tensor' %(axis, rank)
                expansion = ([None] * axis) + [slice(None)] + ([None] * (rank - axis - 1))
                linspace_support = linspace_support[expansion]

                if A is None:
                    out = B * linspace_support / n_interp
                else:
                    A = tf.expand_dims(A, axis)
                    assert A.shape == B.shape, 'A and B must have the same shape, got %s and %s' %(A.shape, B.shape)
                    out = A + ((B-A) * linspace_support / n_interp)
                return out

    def _lininterp_fixed_n_points(self, x, n):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_input = tf.shape(x)[1]
                n_output = n_input * (n+1)
                interp = tf.image.resize_bilinear(tf.expand_dims(tf.expand_dims(x, -1), -1), [n_output, 1])
                interp = tf.squeeze(tf.squeeze(interp, -1), -1)[..., :-n]
                return interp

    def _lininterp_fixed_frequency(self, x, time, time_mask, hz=1000):
        # Performs a linear interpolation at a fixed frequency between impulses that are variably spaced in time.
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Reverse arrays so that max time is left-aligned
                x = tf.reverse(x, axis=[1])
                x_deltas = x[:, 1:, :] - x[:, :-1, :]
                time = tf.reverse(time, axis=[1])
                time_mask = tf.reverse(time_mask, axis=[1])
                time_mask_int = tf.cast(time_mask, dtype=self.INT_TF)

                # Round timestamps to integers so they can be used to index the interpolated array
                time_int = tf.cast(tf.round(time * hz), dtype=self.INT_TF)

                # Compute intervals by subtracting the lower bound
                end_ix = tf.reduce_sum(time_mask_int, axis=1) - 1
                time_minima = tf.gather_nd(time_int, tf.stack([tf.range(tf.shape(x)[0]), end_ix], axis=1))
                time_delta_int = time_int - time_minima[..., None]

                # Compute the minimum number of interpolation points needed for each data point
                impulse_ix_max_batch = tf.reduce_max(time_delta_int, axis=1, keepdims=True)

                # Compute the largest number of interpolation points in the batch, which will be used to compute the length of the interpolation
                impulse_ix_max = tf.reduce_max(impulse_ix_max_batch)

                # Since events are now temporally reversed, reverse indices by subtracting the max and negating
                impulse_ix = -(time_delta_int - impulse_ix_max_batch)

                # Rescale the deltas by the number of timesteps over which they will be interpolated
                n_steps = impulse_ix[:, 1:] - impulse_ix[:, :-1]
                x_deltas = tf.where(
                    tf.tile(tf.not_equal(n_steps, 0)[..., None], [1, 1, tf.shape(x)[2]]),
                    x_deltas / tf.cast(n_steps, dtype=self.FLOAT_TF)[..., None],
                    tf.zeros_like(x_deltas)
                )

                # Pad x_deltas
                x_deltas = tf.concat([x_deltas, tf.zeros([tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)

                # Compute a mask for the interpolated output
                time_mask_interp = tf.cast(tf.range(impulse_ix_max + 1)[None, ...] <= impulse_ix_max_batch, dtype=self.FLOAT_TF)

                # Compute an array of indices for scattering impulses into the interpolation array
                row_ix = tf.tile(tf.range(tf.shape(x)[0])[..., None], [1, tf.shape(x)[1]])
                scatter_ix = tf.stack([row_ix, impulse_ix], axis=2)

                # Create an array for use by gather_nd by taking the cumsum of an array with ones at indices with impulses, zeros otherwise
                gather_ix = tf.cumsum(
                    tf.scatter_nd(
                        scatter_ix,
                        tf.ones_like(impulse_ix, dtype=self.INT_TF) * time_mask_int,
                        [tf.shape(x)[0], impulse_ix_max + 1]
                    ),
                    axis=1
                ) - 1
                row_ix = tf.tile(tf.range(tf.shape(x)[0])[..., None], [1, impulse_ix_max + 1])
                gather_ix = tf.stack([row_ix, gather_ix], axis=2)

                x_interp_base = tf.gather_nd(
                    x,
                    gather_ix
                )

                interp_factor = tf.cast(
                    tf.range(impulse_ix_max + 1)[None, ...] - tf.gather_nd(
                        impulse_ix,
                        gather_ix
                    ),
                    dtype=self.FLOAT_TF
                )


                interp_delta = tf.cast(interp_factor, dtype=self.FLOAT_TF)[..., None] * tf.gather_nd(
                    x_deltas,
                    gather_ix
                )

                x_interp = (x_interp_base + interp_delta) * time_mask_interp[..., None]

                x_interp = tf.reverse(x_interp, axis=[1])
                time_interp = tf.cast(
                    tf.reverse(
                        tf.maximum(
                            (tf.range(0, -impulse_ix_max - 1, delta=-1)[None, ...] + impulse_ix_max_batch),
                            tf.zeros([impulse_ix_max + 1], dtype=self.INT_TF)
                        ) + time_minima[..., None],
                        axis=[1]
                    ),
                    dtype=self.FLOAT_TF
                ) * tf.reverse(time_mask_interp, axis=[1]) / hz

                return x_interp, time_interp

    def _abs(self, x):
        return tf.where(x > 0., x, -x)




    ######################################################
    #
    #  Internal public network initialization methods.
    #  These must be implemented by all subclasses and
    #  should only be called at initialization.
    #
    ######################################################

    def initialize_intercept(self, ran_gf=None):
        """
        Add an intercept.
        This method must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param ran_gf: ``str`` or ``None``; Name of random grouping factor for random intercept (if ``None``, constructs a fixed intercept)
        :return: 2-tuple of ``Tensor`` ``(intercept, intercept_summary)``; ``intercept`` is the intercept for use by the model. ``intercept_summary`` is an identically-shaped representation of the current intercept value for logging and plotting (can be identical to ``intercept``). For fixed intercepts, should return a trainable scalar. For random intercepts, should return batch-length vector of trainable weights. Weights should be initialized around 0.
        """
        raise NotImplementedError

    def initialize_coefficient(self, coef_ids=None, ran_gf=None, suffix=None):
        """
        Add coefficients.
        This method must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :param suffix: ``str`` or ``None``: Suffix to add to coefficient variable name. If ``None``, no suffix.
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of ``len(coef_ids)`` trainable weights. For random coefficients, should return batch-length matrix of trainable weights with ``len(coef_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_objective(self):
        """
        Add an objective function to the CDR model.

        :return: ``None``
        """

        raise NotImplementedError




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
                    if update and self.convergence_basis == 'loss' and feed_dict is None:
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
                        if self.convergence_basis.lower() == 'parameters':
                            locally_converged &= p_ta_at_min_p > self.convergence_alpha
                        convergence_history = self.convergence_history.eval(session=self.sess)
                        convergence_history[:-1] = convergence_history[1:]
                        convergence_history[-1] = locally_converged
                        self.sess.run(self.convergence_history_assign, {self.convergence_history_update: convergence_history})

                    if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                        fd_convergence = {
                                self.rho_t: rt_at_min_p,
                                self.p_rho_t: min_p
                            }
                        if self.convergence_basis.lower() == 'parameters':
                            fd_convergence[self.rho_a] = ra_at_min_p
                            fd_convergence[self.p_rho_a] =  p_ta_at_min_p
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
                        if self.convergence_basis.lower() == 'parameters':
                            stderr('rho_a: %s.\n' % ra_at_min_p)
                            stderr('p of rho_a: %s.\n' % p_ta_at_min_p)
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

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
                        model_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
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
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

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





    ######################################################
    #
    #  Public methods that must be implemented by
    #  subclasses
    #
    ######################################################

    def run_train_step(self, feed_dict):
        """
        Update the model from a batch of training data.
        **All subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """

        raise NotImplementedError

    def run_predict_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Generate predictions from a batch of data.
        **All CDR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor values.
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """
        raise NotImplementedError

    def run_loglik_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Compute the log-likelihoods of a batch of data.
        **All CDR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Pointwise log-likelihoods, one for each training sample
        """

        raise NotImplementedError




    ######################################################
    #
    #  Private model inspection methods
    #
    ######################################################


    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    out = self.parameter_table_fixed_values.eval(session=self.sess)
                else:
                    out = self.parameter_table_random_values.eval(session=self.sess)

                self.set_predict_mode(False)

            return out




    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

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
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
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
            formatters = formatters
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
            parameter_table = pd.concat(
                [
                    pd.DataFrame({'Parameter': parameter_table['Parameter'] + ' | ' + parameter_table['Group'] + ' | ' + parameter_table['Level']}),
                    parameter_table[self.parameter_table_columns]
                ],
                axis=1
            )
            formatters = {
                'Parameter': left_justified_formatter(parameter_table, 'Parameter')
            }
            parameter_table_str = parameter_table.to_string(
                index=False,
                justify='left',
                formatters = formatters
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

        if hasattr(self, 'pc') and self.pc:
            out += ' ' * indent + 'SOURCE IRF TREE:\n'
            tree_str = str(self.t_src)
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

        if hasattr(self, 'pc') and self.pc:
            t = self.t_src
        else:
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

    def report_training_mse(self, indent=0):
        """
        Generate a string representation of the model's training MSE.

        :param indent: ``int``; indentation level
        :return: ``str``; the training MSE report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_mse = self.training_mse.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING MSE:\n'
                out += ' ' * (indent + 2) + str(training_mse)
                out += '\n\n'

                return out

    def report_training_mae(self, indent=0):
        """
        Generate a string representation of the model's training MAE.

        :param indent: ``int``; indentation level
        :return: ``str``; the training MAE report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_mae = self.training_mae.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING MAE:\n'
                out += ' ' * (indent + 2) + str(training_mae)
                out += '\n\n'

                return out

    def report_training_corr(self, indent=0):
        """
        Generate a string representation of the model's training prediction/response correlation.

        :param indent: ``int``; indentation level
        :return: ``str``; the training correlation report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_rho = self.training_rho.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING R(TRUE, PRED):\n'
                out += ' ' * (indent + 2) + str(training_rho)
                out += '\n\n'

                return out

    def report_training_loglik(self, indent=0):
        """
        Generate a string representation of the model's training log likelihood.

        :param indent: ``int``; indentation level
        :return: ``str``; the training log likelihood report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loglik_train = self.training_loglik.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING LOG LIKELIHOOD:\n'
                out += ' ' * (indent + 2) + str(loglik_train)
                out += '\n\n'

                return out

    def report_training_percent_variance_explained(self, indent=0):
        """
        Generate a string representation of the percent variance explained by the model on training data.

        :param indent: ``int``; indentation level
        :return: ``str``; the training percent variance explained report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_percent_variance_explained = self.training_percent_variance_explained.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING PERCENT VARIANCE EXPLAINED:\n'
                out += ' ' * (indent + 2) + '%.2f' %training_percent_variance_explained + '%'
                out += '\n\n'

                return out

    def report_evaluation(
            self,
            mse=None,
            mae=None,
            corr=None,
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
        :param corr: ``float`` or ``None``; Pearson correlation of predictions with observed response, skipped if ``None``.
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
        if mse is not None:
            out += ' ' * (indent+2) + 'MSE: %s\n' %mse
        if mae is not None:
            out += ' ' * (indent+2) + 'MAE: %s\n' %mae
        if corr is not None:
            out += ' ' * (indent+2) + 'r(true, pred): %s\n' %corr
        if loglik is not None:
            out += ' ' * (indent+2) + 'Log likelihood: %s\n' %loglik
        if loss is not None:
            out += ' ' * (indent+2) + 'Loss per training objective: %s\n' %loss
        if true_variance is not None:
            out += ' ' * (indent+2) + 'True variance: %s\n' %true_variance
        if percent_variance_explained is not None:
            out += ' ' * (indent+2) + 'Percent variance explained: %.2f%%\n' %percent_variance_explained
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

        out += self.report_training_mse(indent=indent+2)
        out += self.report_training_mae(indent=indent+2)
        out += self.report_training_corr(indent=indent+2)
        out += self.report_training_loglik(indent=indent+2)
        out += self.report_training_percent_variance_explained(indent=indent+2)

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
            out += ' ' * (indent * 2) + 'Convergence basis: %s\n' % self.convergence_basis.lower()
            out += ' ' * (indent * 2) + 'Convergence n iterates: %s\n' % self.convergence_n_iterates
            out += ' ' * (indent * 2) + 'Convergence stride: %s\n' % self.convergence_stride
            out += ' ' * (indent * 2) + 'Convergence alpha: %s\n' % self.convergence_alpha
            out += ' ' * (indent * 2) + 'Convergence min p of rho_t: %s\n' % min_p
            out += ' ' * (indent * 2) + 'Convergence rho_t at min p: %s\n' % rt_at_min_p
            if self.convergence_basis.lower() == 'parameters':
                out += ' ' * (indent * 2) + 'Convergence rho_a at min p: %s\n' % ra_at_min_p
                out += ' ' * (indent * 2) + 'Convergence p of rho_a at min p: %s\n' % p_ta_at_min_p
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




    ######################################################
    #
    #  High-level methods for training, prediction,
    #  and plotting
    #
    ######################################################

    def build(self, outdir=None, restore=True):
        """
        Construct the CDR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Report model details after initialization.
        :return: ``None``
        """

        raise NotImplementedError

    def is_non_dirac(self, impulse_name):
        """
        Check whether an impulse is associated with a non-Dirac response function

        :param impulse_name: ``str``; name of impulse
        :return: ``bool``; whether the impulse is associated with a non-Dirac response function
        """

        return impulse_name in self.non_dirac_impulses

    def fit(self,
            X,
            y,
            n_iter=100,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            force_training_evaluation=True
            ):
        """
        Fit the model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in **y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **y**
            * A column with the same name as the dependent variable specified in the model formula
            * A column for each random grouping factor in the model formula

            In general, **y** will be identical to the parameter **y** provided at model initialization.
            This must hold for MCMC inference, since the number of minibatches is built into the model architecture.
            However, it is not necessary for variational inference.
        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param force_training_evaluation: ``bool``; (Re-)run post-fitting evaluation, even if resuming a model whose training is already complete.
        :param n_iter: ``int``; the number of training iterations
        """

        if hasattr(self, 'pc') and self.pc:
            impulse_names = self.src_impulse_names
            assert X_2d_predictors is None, 'Principal components regression not supported for models with 2d predictors'
        else:
            impulse_names  = self.impulse_names

        if not np.isfinite(self.minibatch_size):
            minibatch_size = len(y)
        else:
            minibatch_size = self.minibatch_size
        n_minibatch = math.ceil(float(len(y)) / minibatch_size)

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_CDR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        if self.use_crossval:
            sel = ~y[self.crossval_factor].isin(self.crossval_folds)
            first_obs = first_obs[sel]
            last_obs = last_obs[sel]
            time_y = time_y[sel]
            y_dv = y_dv[sel]
            gf_y = gf_y[sel]
            X_2d = X_2d[sel]
            time_X_2d = time_X_2d[sel]
            time_X_mask = time_X_mask[sel]

            n_train = len(y_dv)
        else:
            n_train = len(y)

        stderr('*' * 100 + '\n' + self.initialization_summary() + '*' * 100 + '\n\n')
        with open(self.outdir + '/initialization_summary.txt', 'w') as i_file:
            i_file.write(self.initialization_summary())

        usingGPU = tf.test.is_gpu_available()
        stderr('Using GPU: %s\nNumber of training samples: %d\n\n' % (usingGPU, n_train))

        stderr('Correlation matrix for input variables:\n')
        impulse_names_2d = [x for x in impulse_names if x in X_2d_predictor_names]
        rho = corr_cdr(X_2d, impulse_names, impulse_names_2d, time_X_2d, time_X_mask)
        stderr(str(rho) + '\n\n')

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
                            if self.log_random and len(self.rangf) > 0:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                            self.writer.flush()
                    else:
                        stderr('Resuming training from most recent checkpoint...\n\n')

                    if self.global_step.eval(session=self.sess) == 0:
                        stderr('Saving initial weights...\n')
                        self.save()

                    while not self.has_converged() and self.global_step.eval(session=self.sess) < n_iter:
                        p, p_inv = get_random_permutation(n_train)
                        t0_iter = pytime.time()
                        stderr('-' * 50 + '\n')
                        stderr('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        stderr('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            stderr('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        pb = tf.contrib.keras.utils.Progbar(self.n_train_minibatch)

                        loss_total = 0.
                        reg_loss_total = 0.
                        if self.is_bayesian:
                            kl_loss_total = 0.
                        if self.loss_filter_n_sds:
                            n_dropped = 0.

                        for j in range(0, n_train, minibatch_size):
                            indices = p[j:j+minibatch_size]
                            fd_minibatch = {
                                self.X: X_2d[indices],
                                self.time_X: time_X_2d[indices],
                                self.time_X_mask: time_X_mask[indices],
                                self.y: y_dv[indices],
                                self.time_y: time_y[indices],
                                self.gf_y: gf_y[indices] if len(gf_y > 0) else gf_y,
                                self.training: not self.predict_mode
                            }

                            info_dict = self.run_train_step(fd_minibatch)

                            self.check_numerics()

                            if self.loss_filter_n_sds:
                                n_dropped += info_dict['n_dropped']

                            loss_cur = info_dict['loss']
                            if self.ema_decay:
                                self.sess.run(self.ema_op)
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

                            pb.update((j/minibatch_size)+1, values=pb_update)

                            # if self.global_batch_step.eval(session=self.sess) % 1000 == 0:
                            #     self.save()
                            #     self.make_plots(prefix='plt')

                        self.sess.run(self.incr_global_step)

                        if not type(self).__name__.startswith('CDRNN'):
                            self.verify_random_centering()

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
                            if self.log_random and len(self.rangf) > 0:
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
                    preds = self.predict(
                        X,
                        y.time,
                        y[self.form.rangf],
                        first_obs,
                        last_obs,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors
                    )

                    with open(self.outdir + '/obs_train.txt', 'w') as o_file:
                        obs = y[self.dv].values
                        for i in range(len(obs)):
                            o_file.write(str(obs[i]) + '\n')

                    with open(self.outdir + '/preds_train.txt', 'w') as p_file:
                        for i in range(len(preds)):
                            p_file.write(str(preds[i]) + '\n')

                    # Extract and save losses
                    training_se = np.array((y[self.dv] - preds) ** 2)
                    training_mse = training_se.mean()
                    training_percent_variance_explained = percent_variance_explained(y[self.dv], preds)

                    training_ae = np.array(np.abs(y[self.dv] - preds))
                    training_mae = training_ae.mean()

                    training_rho = np.corrcoef(y[self.dv], preds)[0,1]

                    with open(self.outdir + '/squared_error_train.txt', 'w') as e_file:
                        for i in range(len(training_se)):
                            e_file.write(str(training_se[i]) + '\n')

                    # Extract and save log likelihoods
                    training_logliks = self.log_lik(
                        X,
                        y,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors,
                    )
                    with open(self.outdir + '/loglik_train.txt','w') as l_file:
                        for i in range(len(training_logliks)):
                            l_file.write(str(training_logliks[i]) + '\n')
                    training_loglik = training_logliks.sum()

                    # Store training evaluation statistics in the graph
                    self.sess.run(
                        [self.set_training_mse, self.set_training_mae],
                        feed_dict={
                            self.training_mse_in: training_mse,
                            self.training_mae_in: training_mae,
                        }
                    )

                    self.sess.run(
                        [self.set_training_loglik],
                        feed_dict={
                            self.training_loglik_in: training_loglik
                        }
                    )

                    self.sess.run(
                        [self.set_training_rho],
                        feed_dict={
                            self.training_rho_in: training_rho
                        }
                    )

                    self.save()

                    with open(self.outdir + '/eval_train.txt', 'w') as e_file:
                        eval_train = '------------------------\n'
                        eval_train += 'CDR TRAINING EVALUATION\n'
                        eval_train += '------------------------\n\n'
                        eval_train += self.report_formula_string(indent=2)
                        eval_train += self.report_evaluation(
                            mse=training_mse,
                            mae=training_mae,
                            corr=training_rho,
                            loglik=training_loglik,
                            percent_variance_explained=training_percent_variance_explained,
                            indent=2
                        )

                        e_file.write(eval_train)

                    self.save_parameter_table()
                    self.save_integral_table()

                    self.set_training_complete(True)

                    self.save()

    def predict(
            self,
            X,
            y_time,
            y_rangf,
            first_obs,
            last_obs,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            standardize_response=False,
            verbose=True
    ):
        """
        Predict from the pre-trained CDR model.
        Predictions are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param y_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the regression targets, grouped by series.
        :param y_rangf: ``pandas`` ``Series`` or 1D ``numpy`` array; random grouping factor values (if applicable).
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param first_obs: list of ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: list of ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)

        if hasattr(self, 'pc') and self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            stderr('Computing predictions...\n')

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_CDR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.training: not self.predict_mode
                }

                if not np.isfinite(self.eval_minibatch_size):
                    preds = self.run_predict_op(
                        fd,
                        standardize_response=standardize_response,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    preds = np.zeros((len(y_time),))
                    n_eval_minibatch = math.ceil(len(y_time) / self.eval_minibatch_size)
                    for i in range(0, len(y_time), self.eval_minibatch_size):
                        if verbose:
                            stderr('\rMinibatch %d/%d' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.training: not self.predict_mode
                        }
                        preds[i:i + self.eval_minibatch_size] = self.run_predict_op(
                            fd_minibatch,
                            standardize_response=standardize_response,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )

                if verbose:
                    stderr('\n\n')

                self.set_predict_mode(False)

                return preds

    def log_lik(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            standardize_response=False,
            verbose=True
    ):
        """
        Compute log-likelihood of data from predictive posterior.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)

        if hasattr(self, 'pc') and self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            stderr('Computing likelihoods...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_CDR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if not np.isfinite(self.eval_minibatch_size):
                    fd = {
                        self.X: X_2d,
                        self.time_X: time_X_2d,
                        self.time_X_mask: time_X_mask,
                        self.time_y: time_y,
                        self.gf_y: gf_y,
                        self.y: y_dv,
                        self.training: not self.predict_mode
                    }
                    log_lik = self.run_loglik_op(
                        fd,
                        standardize_response=standardize_response,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    log_lik = np.zeros((len(time_y),))
                    n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                    for i in range(0, len(time_y), self.eval_minibatch_size):
                        if verbose:
                            stderr('\rMinibatch %d/%d' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.eval_minibatch_size],
                            self.training: not self.predict_mode
                        }
                        log_lik[i:i+self.eval_minibatch_size] = self.run_loglik_op(
                            fd_minibatch,
                            standardize_response=standardize_response,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )

                if verbose:
                    stderr('\n\n')

                self.set_predict_mode(False)

                return log_lik

    def loss(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            training=None,
            verbose=True
    ):
        """
        Compute compute the loss over a dataset using the model's optimization objective.
        Useful for checking divergence between optimization objective and other evaluation metrics.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            stderr('Using GPU: %s\n' % usingGPU)

        if hasattr(self, 'pc') and self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            stderr('Computing loss using objective function...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_CDR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if training is None:
                    training = not self.predict_mode

                if not np.isfinite(self.minibatch_size):
                    fd = {
                        self.X: X_2d,
                        self.time_X: time_X_2d,
                        self.time_X_mask: time_X_mask,
                        self.time_y: time_y,
                        self.gf_y: gf_y,
                        self.y: y_dv,
                        self.training: training
                    }
                    loss = self.run_loss_op(
                        fd,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    n_minibatch = math.ceil(len(y) / self.minibatch_size)
                    loss = np.zeros((n_minibatch,))
                    for i in range(0, n_minibatch):
                        if verbose:
                            stderr('\rMinibatch %d/%d' %(i+1, n_minibatch))
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.minibatch_size],
                            self.time_X: time_X_2d[i:i + self.minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.minibatch_size],
                            self.time_y: time_y[i:i + self.minibatch_size],
                            self.gf_y: gf_y[i:i + self.minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.minibatch_size],
                            self.training: training
                        }
                        loss[i] = self.run_loss_op(
                            fd_minibatch,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )
                    loss = loss.mean()

                if verbose:
                    stderr('\n\n')

                self.set_predict_mode(False)

                return loss

    def error_theoretical_quantiles(
            self,
            n_errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)
                fd = {
                    self.n_errors: n_errors,
                    self.training: not self.predict_mode
                }
                err_q = self.sess.run(self.err_dist_summary_theoretical_quantiles, feed_dict=fd)
                self.set_predict_mode(False)

                return err_q

    def error_theoretical_cdf(
            self,
            errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.errors: errors,
                    self.training: not self.predict_mode
                }
                err_cdf = self.sess.run(self.err_dist_summary_theoretical_cdf, feed_dict=fd)

                return err_cdf

    def error_ks_test(
            self,
            errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                err_cdf = self.error_theoretical_cdf(errors)

                D, p_value = scipy.stats.kstest(errors, lambda x: err_cdf)

                return D, p_value

    def get_plot_data(
            self,
            xvar='t_delta',
            yvar=None,
            resvar='y_mean',
            X_ref=None,
            time_X_ref=None,
            t_delta_ref=None,
            gf_y_ref=None,
            ref_varies_with_x=False,
            ref_varies_with_y=False,
            manipulations=None,
            pair_manipulations=False,
            standardize_response=False,
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
        :param resvar: ``str``; Name of parameter of predictive distribution to plot as response variable. One of ``'y_mean'``, ``'y_sd'``, ``'y_skewness'``, or ``'y_tailweight'``. Only ``'y_mean'`` is interesting for CDR, since the others are assumed scalar. CDRNN fits all predictive parameters via IRFs.
        :param X_ref: ``dict`` or ``None``; Dictionary mapping impulse names to numeric values for use in constructing the reference. Any impulses not specified here will take default values.
        :param time_X_ref: ``float`` or ``None``; Timestamp to use for constructing the reference. If ``None``, use default value.
        :param t_delta_ref: ``float`` or ``None``; Delay/offset to use for constructing the reference. If ``None``, use default value.
        :param gf_y_ref: ``dict`` or ``None``; Dictionary mapping random grouping factor names to random grouping factor levels for use in constructing the reference. Any random effects not specified here will take default values.
        :param ref_varies_with_x: ``bool``; Whether the reference varies along the x-axis. If ``False``, use the scalar reference value for the x-axis.
        :param ref_varies_with_y: ``bool``; Whether the reference varies along the y-axis. If ``False``, use the scalar reference value for the y-axis. Ignored if **yvar** is ``None``.
        :param manipulations: ``list`` of ``dict``; A list of manipulations, where each manipulation is constructed as a dictionary mapping a variable name (e.g. ``'predictorX'``, ``'t_delta'``) to either a float offset or a function that transforms the reference value for that variable (e.g. multiplies it by ``2``). Alternatively, the keyword ``'ranef'`` can be used to manipulate random effects. The ``'ranef'`` entry must map to a ``dict`` that itself maps random grouping factor names (e.g. ``'subject'``) to levels (e.g. ``'subjectA'``).
        :param pair_manipulations: ``bool``; Whether to apply the manipulations to the reference input. If ``False``, all manipulations are compared to the same reference. For example, when plotting by-subject IRFs by subject, each subject might have a difference base response. In this case, set **pair_manipulations** to ``True`` in order to match the random effects used to compute the reference response and the response of interest.
        :param standardize_response: ``bool``; If the model uses implicit response standardization, whether to generate plot data on the standardized scale. If ``False``, plots will be rescaled back to the input scale.
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
        :return: 5-tuple (plot_axes, mean, lower, upper, samples); Let RX, RY, S, and O respectively be the x-axis resolution, y-axis resolution, number of samples, and number of output dimensions (manipulations). If plot is 2D, ``plot_axes`` is an array with shape ``(RX,)``, ``mean``, ``lower``, and ``upper`` are arrays with shape ``(RX, O)``,  and ``samples is an array with shape ``(S, RX, O)``. If plot is 3D, ``plot_axes`` is a pair of arrays each with shape ``(RX, RY)`` (i.e. a meshgrid), ``mean``, ``lower``, and ``upper`` are arrays with shape ``(RX, RY, O)``, and ``samples`` is an array with shape ``(S, RX, RY, O)``.
        """
        assert xvar is not None, 'Value must be provided for xvar'
        assert xvar != yvar, 'Cannot vary two axes along the same variable'

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
                if time_X_ref is None:
                    time_X_ref = self.time_X_mean
                assert np.isscalar(time_X_ref), 'time_X_ref must be a scalar'
                time_X_ref = np.reshape(time_X_ref, (1, 1, 1))
                time_X_ref = np.tile(time_X_ref, [1, 1, max(n_impulse, 1)])

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
                        time_X_base = np.tile(time_X_ref, (T, 1, 1))
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
                                ax_max = self.time_X_mean + self.time_X_sd
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
                            time_X_ref = time_X_base

                    if axis_var == 't_delta':
                        if axis is None:
                            if ax_min is None:
                                ax_min = 0
                            if ax_max is None:
                                ax_max = self.plot_n_time_units
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
                if time_X_ref.shape[0] == 1 and B_ref > 1:
                    time_X_ref = np.tile(time_X_ref, (B_ref, 1, 1))
                if t_delta_ref.shape[0] == 1 and B_ref > 1:
                    t_delta_ref = np.tile(t_delta_ref, (B_ref, 1, 1))
                if gf_y_ref.shape[0] == 1 and B_ref > 1:
                    gf_y_ref = np.tile(gf_y_ref, (B_ref, 1))

                if ref_as_manip:
                    X = []
                    time_X = []
                    t_delta = []
                    gf_y = []
                    X_ref_in = [X_ref]
                    time_X_ref_in = [time_X_ref]
                    t_delta_ref_in = [t_delta_ref]
                    gf_y_ref_in = [gf_y_ref]
                else:
                    X = [X_base]
                    time_X = [time_X_base]
                    t_delta = [t_delta_base]
                    gf_y = [gf_y_base]
                    if pair_manipulations:
                        X_ref_in = [X_ref]
                        time_X_ref_in = [time_X_ref]
                        t_delta_ref_in = [t_delta_ref]
                        gf_y_ref_in = [gf_y_ref]

                for manipulation in manipulations:
                    X_cur = None
                    time_X_cur = time_X_base
                    t_delta_cur = t_delta_base
                    gf_y_cur = gf_y_base

                    if pair_manipulations:
                        X_ref_cur = None
                        time_X_ref_cur = time_X_ref
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
                    self.time_X: time_X_ref_in,
                    self.time_X_mask: np.ones_like(time_X_ref_in),
                    self.t_delta: t_delta_ref_in,
                    self.gf_y: gf_y_ref_in,
                    self.training: not self.predict_mode
                }
                
                # print('X_ref_in')
                # print(fd_ref[self.X].shape)
                # print('time_X_ref_in')
                # print(fd_ref[self.time_X].shape)
                # print('time_X_mask_ref_in')
                # print(fd_ref[self.time_X_mask].shape)
                # print('t_delta_ref_in')
                # print(fd_ref[self.t_delta].shape)

                # Bring manipulations into 1-1 alignment on the batch dimension
                if n_manip:
                    X = np.concatenate(X, axis=0)
                    time_X = np.concatenate(time_X, axis=0)
                    t_delta = np.concatenate(t_delta, axis=0)
                    gf_y = np.concatenate(gf_y, axis=0)

                    fd_main = {
                        self.X: X,
                        self.time_X: time_X,
                        self.time_X_mask: np.ones_like(time_X),
                        self.t_delta: t_delta,
                        self.gf_y: gf_y,
                        self.training: not self.predict_mode
                    }

                    # print('X_in')
                    # print(fd_main[self.X].shape)
                    # print('time_X_in')
                    # print(fd_main[self.time_X].shape)
                    # print('time_X_mask_in')
                    # print(fd_main[self.time_X_mask].shape)
                    # print('t_delta_in')
                    # print(fd_main[self.t_delta].shape)

                if resvar == 'y_mean':
                    response = self.y_delta
                elif resvar == 'y_sd':
                    response = self.y_sd_delta
                elif resvar == 'y_skewness':
                    response = self.y_skewness_delta
                elif resvar == 'y_tailweight':
                    response = self.y_tailweight_delta
                else:
                    raise ValueError('')

                if resample:
                    fd_ref[self.use_MAP_mode] = False
                    if n_manip:
                        fd_main[self.use_MAP_mode] = False

                samples = []
                alpha = 100-float(level)

                for i in range(n_samples):
                    self.sess.run(self.dropout_resample_ops)
                    sample_ref = self.sess.run(response, feed_dict=fd_ref)
                    sample_ref = np.reshape(sample_ref, ref_shape, 'F')
                    if n_manip:
                        sample_main = self.sess.run(response, feed_dict=fd_main)
                        sample_main = np.reshape(sample_main, sample_shape, 'F')
                        sample_main -= sample_ref
                        if ref_as_manip:
                            sample = np.concatenate([sample_ref, sample_main], axis=-1)
                        else:
                            sample = sample_main
                    else:
                        sample = sample_ref
                    samples.append(sample)

                samples = np.stack(samples, axis=0)
                if self.standardize_response and not standardize_response:
                    if resvar in ['y_mean', 'y_sd']:
                        samples *= self.y_train_sd
                mean = samples.mean(axis=0)
                if resample:
                    lower = np.percentile(samples, alpha / 2, axis=0)
                    upper = np.percentile(samples, 100 - (alpha / 2), axis=0)
                else:
                    lower = upper = mean
                    samples = mean[None, ...]

                out = (plot_axes, mean, lower, upper, samples)

                return out

    def irf_integrals(self, standardize_response=False, level=95, random=False, n_samples='default', n_time_units=None, n_time_points=1000):
        """
        Generate effect size estimates by computing the area under each IRF curve in the model via discrete approximation.

        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
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
        for x in self.impulse_names:
            delta = self.plot_step_map[x]
            manipulations.append({x: delta})

        if random:
            ranef_group_names = self.ranef_group_names
            ranef_level_names = self.ranef_level_names
            ranef_zipped = zip(ranef_group_names, ranef_level_names)
            gf_y_refs = [{x: y} for x, y in ranef_zipped]
        else:
            gf_y_refs = [{None: None}]

        out = []

        for g, gf_y_ref in enumerate(gf_y_refs):
            _, _, _, _, vals = self.get_plot_data(
                xvar='t_delta',
                X_ref=None,
                time_X_ref=None,
                t_delta_ref=None,
                gf_y_ref=gf_y_ref,
                ref_varies_with_x=True,
                manipulations=manipulations,
                pair_manipulations=False,
                xaxis=None,
                xmin=0,
                xmax=n_time_units,
                xres=n_time_points,
                n_samples=n_samples,
                level=level,
                standardize_response=standardize_response
            )

            if not self.is_cdrnn:
                vals = vals[..., 1:]

            integrals = vals.sum(axis=1) * step

            group_name = list(gf_y_ref.keys())[0]
            level_name = gf_y_ref[group_name]

            out_cur = pd.DataFrame({
                'IRF': names,
                'Group': group_name if group_name is not None else '',
                'Level': level_name if level_name is not None else ''
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
            resvar='y_mean',
            standardize_response=False,
            pred_names=None,
            sort_names=True,
            prop_cycle_length=None,
            prop_cycle_map=None,
            plot_dirac=False,
            plot_interactions=None,
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
            legend=None,
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
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param resvar: ``str``; Name of parameter of predictive distribution to plot as response variable. One of ``'y_mean'``, ``'y_sd'``, ``'y_skewness'``, or ``'y_tailweight'``. Only ``'y_mean'`` is interesting for CDR, since the others are assumed scalar. CDRNN fits all predictive parameters via IRFs.
        :param summed: ``bool``; whether to plot individual IRFs or their sum.
        :param pred_names: ``list`` or ``None``; list of names of predictors to include in univariate IRF plots. If ``None``, all predictors are plotted.
        :param sort_names: ``bool``; whether to alphabetically sort IRF names.
        :param plot_unscaled: ``bool``; plot unscaled IRFs.
        :param plot_composite: ``bool``; plot any composite IRFs. If ``False``, only plots terminal IRFs.
        :param prop_cycle_length: ``int`` or ``None``; Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **pred_names**.
        :param prop_cycle_map: ``dict``, ``list`` of ``int``, or ``None``; Integer indices to use in the properties cycle for each entry in **pred_names**. If a ``dict``, a map from predictor names to ``int``. If a ``list`` of ``int``, predictors inferred using **pred_names** are aligned to ``int`` indices one-to-one. If ``None``, indices are automatically assigned.
        :param plot_dirac: ``bool``; whether to include any Dirac delta IRF's (stick functions at t=0) in plot.
        :param plot_interactions: ``list`` of ``str`` or ``None``; List of all implicit interactions to plot. If ``None``, use default setting.
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
        :param legend: ``bool`` or ``None``; whether to include a legend in plots with multiple components. If ``None``, use default setting.
        :param use_line_markers: ``bool``; whether to add markers to lines in univariate IRF plots.
        :param transparent_background: ``bool``; whether to use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool`` or ``None``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot. If ``None``, use default setting.
        :param dump_source: ``bool``; Whether to dump the plot source array to a csv file.
        :return: ``None``
        """

        if irf_name_map is None:
            irf_name_map = self.irf_name_map

        if not plot_interactions:
            plot_interactions = []

        mc = bool(n_samples) and (self.is_bayesian or self.has_dropout)

        if plot_interactions is None:
            plot_interactions = self.plot_interactions
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
        if plot_x_inches is None:
            plot_x_inches = self.plot_x_inches
        if plot_y_inches is None:
            plot_y_inches = self.plot_y_inches
        if legend is None:
            legend = self.plot_legend
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

                if self.asymmetric_error:
                    lb = self.sess.run(self.err_dist_lb)
                    ub = self.sess.run(self.err_dist_ub)
                    n_time_units = ub - lb
                    fd = {
                        self.support_start: lb,
                        self.n_time_units: n_time_units,
                        self.n_time_points: plot_n_time_points,
                        self.training: not self.predict_mode
                    }
                    plot_x = self.sess.run(self.support, feed_dict=fd)
                    if mc:
                        plot_y, lq, uq, _ = self.ci_curve(
                            self.err_dist_plot,
                            level=level,
                            n_samples=n_samples,
                            support_start=lb,
                            n_time_units=n_time_units,
                            n_time_points=plot_n_time_points,
                        )
                        plot_y = plot_y[0, ..., None]
                        lq = lq[0, ..., None]
                        uq = uq[0, ..., None]
                        plot_name = 'mc_error_distribution_%s.png' % self.global_step.eval(sess=self.sess) \
                            if self.keep_plot_history else 'mc_error_distribution.png'
                    else:
                        plot_y = self.sess.run(self.err_dist_plot_summary, feed_dict=fd)[0]
                        lq = None
                        uq = None
                        plot_name = 'error_distribution_%s.png' % self.global_step.eval(sess=self.sess) \
                            if self.keep_plot_history else 'error_distribution.png'

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

                if plot_rangf:
                    manipulations = [{'ranef': {x: y}} for x, y in zip(ranef_group_names[1:], ranef_level_names[1:])]
                else:
                    manipulations = None

                # Curvature plots
                if generate_curvature_plots:
                    names = [x for x in self.impulse_names if (self.is_non_dirac(x) and x != 'rate')]

                    for name in names:
                        plot_name = 'curvature'

                        if resvar != 'y_mean':
                            plot_name += '_%s' % resvar

                        plot_x, plot_y, lq, uq, samples = self.get_plot_data(
                            xvar=name,
                            resvar=resvar,
                            t_delta_ref=reference_time,
                            ref_varies_with_x=False,
                            manipulations=manipulations,
                            pair_manipulations=True,
                            standardize_response=standardize_response,
                            reference_type=reference_type,
                            xres=plot_n_time_points,
                            n_samples=n_samples,
                            level=level
                        )

                        plot_name += '_%s_at_delay%s' % (sn(name), reference_time)

                        for g in range(len(ranef_level_names)):
                            filename = prefix + plot_name
                            if ranef_level_names[g]:
                                filename += '_' + ranef_level_names[g]
                            if mc:
                                filename += '_mc'
                            filename += '.png'

                            if use_horiz_axlab:
                                xlab = name
                            else:
                                xlab = None
                            if use_vert_axlab:
                                if resvar == 'y_mean':
                                    ylab = self.dv
                                elif resvar == 'y_sd':
                                    ylab = '%s, SD' % get_irf_name(self.dv, irf_name_map)
                                elif resvar == 'y_skewness':
                                    ylab = '%s, skewness' % get_irf_name(self.dv, irf_name_map)
                                elif resvar == 'y_tailweight':
                                    ylab = '%s, tailweight' % get_irf_name(self.dv, irf_name_map)
                                else:
                                    raise ValueError('Unrecognized resvar %s.' % resvar)
                            else:
                                ylab = None

                            plot_irf(
                                plot_x,
                                plot_y[:, g:g+1],
                                [name],
                                lq=None if lq is None else lq[:, g:g+1],
                                uq=None if uq is None else uq[:, g:g+1],
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
                for plot_type, run_plot in zip(('irf_surface', 'nonstationarity_surface', 'interaction_surface',), (generate_irf_surface_plots, generate_nonstationarity_surface_plots, generate_interaction_surface_plots)):
                    if use_vert_axlab:
                        if resvar == 'y_mean':
                            zlab = self.dv
                        elif resvar == 'y_sd':
                            zlab = '%s, SD' % get_irf_name(self.dv, irf_name_map)
                        elif resvar == 'y_skewness':
                            zlab = '%s, skewness' % get_irf_name(self.dv, irf_name_map)
                        elif resvar == 'y_tailweight':
                            zlab = '%s, tailweight' % get_irf_name(self.dv, irf_name_map)
                        else:
                            raise ValueError('Unrecognized resvar %s.' % resvar)
                    else:
                        zlab = None

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

                                plot_name = 'surface'
                                if resvar != 'y_mean':
                                    plot_name += '_%s' % resvar

                                if plot_type in ('nonstationarity_surface', 'interaction_surface'):
                                    ref_varies_with_x = False
                                else:
                                    ref_varies_with_x = True

                                (plot_x, plot_y), plot_z, lq, uq, _ = self.get_plot_data(
                                    xvar=xvar,
                                    yvar=yvar,
                                    resvar=resvar,
                                    t_delta_ref=reference_time,
                                    ref_varies_with_x=ref_varies_with_x,
                                    manipulations=manipulations,
                                    pair_manipulations=True,
                                    standardize_response=standardize_response,
                                    reference_type=reference_type,
                                    xres=int(np.ceil(np.sqrt(plot_n_time_points))),
                                    yres=int(np.ceil(np.sqrt(plot_n_time_points))),
                                    n_samples=n_samples,
                                    level=level
                                )

                                for g in range(len(ranef_level_names)):
                                    filename = prefix + plot_name + '_' + sn(yvar) + '_by_' + sn(xvar)
                                    if plot_type in ('nonstationarity_surface', 'interaction_surface'):
                                        filename += '_at_delay%s' % reference_time
                                    if ranef_level_names[g]:
                                        filename += '_' + ranef_level_names[g]
                                    if mc:
                                        filename += '_mc'
                                    filename += '.png'

                                    if use_horiz_axlab:
                                        xlab = xvar
                                        ylab = yvar
                                    else:
                                        xlab = xvar
                                        ylab = yvar

                                    plot_surface(
                                        plot_x,
                                        plot_y,
                                        plot_z[..., g],
                                        lq=None if lq is None else lq[..., g],
                                        uq=None if uq is None else uq[..., g],
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

                # IRF 1D
                if generate_univariate_IRF_plots:
                    plot_name = 'irf_univariate'
                    if resvar != 'y_mean':
                        plot_name += '_%s' % resvar

                    if use_horiz_axlab:
                        xlab = 't_delta'
                    else:
                        xlab = None
                    if use_vert_axlab:
                        if resvar == 'y_mean':
                            ylab = self.dv
                        elif resvar == 'y_sd':
                            ylab = '%s, SD' % get_irf_name(self.dv, irf_name_map)
                        elif resvar == 'y_skewness':
                            ylab = '%s, skewness' % get_irf_name(self.dv, irf_name_map)
                        elif resvar == 'y_tailweight':
                            ylab = '%s, tailweight' % get_irf_name(self.dv, irf_name_map)
                        else:
                            raise ValueError('Unrecognized resvar %s.' % resvar)
                    else:
                        ylab = None

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

                    for g, (gf_y_ref, gf_key) in enumerate(zip(gf_y_refs, ranef_level_names)):
                        if gf_key is None:
                            names_cur = names_fixed
                            manipulations_cur = manipulations_fixed
                        else:
                            names_cur = names
                            manipulations_cur = manipulations

                        plot_x, plot_y, lq, uq, samples = self.get_plot_data(
                            xvar='t_delta',
                            resvar=resvar,
                            X_ref=None,
                            time_X_ref=None,
                            t_delta_ref=None,
                            gf_y_ref=gf_y_ref,
                            ref_varies_with_x=True,
                            manipulations=manipulations_cur,
                            pair_manipulations=False,
                            standardize_response=standardize_response,
                            reference_type=reference_type,
                            xaxis=None,
                            xmin=0,
                            xmax=plot_n_time_units,
                            xres=plot_n_time_points,
                            n_samples=n_samples,
                            level=level
                        )

                        filename = prefix + plot_name

                        if ranef_level_names[g]:
                            filename += '_' + ranef_level_names[g]
                        if mc:
                            filename += '_mc'
                        filename += '.png'

                        if not self.is_cdrnn:
                            plot_y = plot_y[..., 1:]
                            if lq is not None:
                                lq = lq[..., 1:]
                            if uq is not None:
                                uq = uq[..., 1:]

                        plot_irf(
                            plot_x,
                            plot_y,
                            names_cur,
                            lq=lq,
                            uq=uq,
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
                            legend=legend,
                            xlab=xlab,
                            ylab=ylab,
                            use_line_markers=use_line_markers,
                            transparent_background=transparent_background,
                            dump_source=dump_source
                        )

                    self.set_predict_mode(False)

    def parameter_table(self, standardize_response=False, fixed=True, level=95, n_samples='default'):
        """
        Generate a pandas table of parameter names and values.

        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param fixed: ``bool``; Return a table of fixed parameters (otherwise returns a table of random parameters).
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int``, ``'default'``, or ``None``; number of posterior samples to draw. If ``None``, use MLE/MAP estimate. If ``'default'``, use model defaults.
        :return: ``pandas`` ``DataFrame``; The parameter table.
        """

        assert fixed or len(self.rangf) > 0, 'Attempted to generate a random effects parameter table in a fixed-effects-only model'

        if n_samples == 'default':
            if self.is_bayesian or self.has_dropout:
                n_samples = self.n_samples_eval

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    keys = self.parameter_table_fixed_keys
                    values = self._extract_parameter_values(
                        fixed=True,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({'Parameter': keys})

                else:
                    keys = self.parameter_table_random_keys
                    rangf = self.parameter_table_random_rangf
                    rangf_levels = self.parameter_table_random_rangf_levels
                    values = self._extract_parameter_values(
                        fixed=False,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({'Parameter': keys, 'Group': rangf, 'Level': rangf_levels}, columns=['Parameter', 'Group', 'Level'])

                if self.standardize_response and not standardize_response:
                    for i, (k, v) in enumerate(zip(keys, values)):
                        if 'intercept' in k or 'coefficient' in k or 'interaction' in k:
                            values[i] = values[i] * self.y_train_sd

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
        if random and len(self.rangf) > 0:
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
