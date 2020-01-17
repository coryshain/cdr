import os
import textwrap
import time as pytime
import scipy.stats
import pandas as pd
from collections import defaultdict
from .kwargs import MODEL_INITIALIZATION_KWARGS
from .formula import *
from .util import *
from .data import build_CDR_impulses, corr_cdr, get_first_last_obs_lists
from .plot import *


import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    def __new__(cls, *args, **kwargs):
        if cls is Model:
            raise TypeError("``Model`` is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, y, **kwargs):

        ## Store initialization settings
        self.form_str = form_str
        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        # Parse and store model data from formula
        form = Formula(self.form_str)
        form = form.categorical_transform(X)
        form = form.categorical_transform(y)
        self.form = form
        dv = form.dv
        rangf = form.rangf

        # Compute from training data
        self.n_train = len(y)
        self.y_train_mean = float(y[dv].mean())
        self.y_train_sd = float(y[dv].std())

        t_deltas = []
        first_obs, last_obs = get_first_last_obs_lists(y)
        for i, cols in enumerate(zip(first_obs, last_obs)):
            first_obs_cur, last_obs_cur = cols
            X_time = np.array(X[i].time, dtype=getattr(np, self.float_type))
            last_obs_cur = np.array(last_obs_cur, dtype=getattr(np, self.int_type))
            first_obs_cur = np.maximum(np.array(first_obs_cur, dtype=getattr(np, self.int_type)),
                                       last_obs_cur - self.history_length + 1)
            t_delta = (y.time - X_time[first_obs_cur])
            t_deltas.append(t_delta)
        t_deltas = np.concatenate(t_deltas, axis=0)
        self.t_delta_limit = np.percentile(t_deltas, 75)
        self.max_tdelta = t_deltas.max()
        self.t_delta_sd = t_deltas.std()

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

        # Collect stats for all impulses and all existing interactions of impulses
        impulse_means = {}
        impulse_sds = {}
        impulse_medians = {}
        impulse_lq = {}
        impulse_uq = {}
        impulse_min = {}
        impulse_max = {}
        impulse_names = self.form.t.impulse_names()
        # impulse_names = [('time',)] + list(itertools.chain.from_iterable(
        #     itertools.combinations(impulse_names, n) for n in range(1, len(impulse_names) + 1)
        # ))

        for name in impulse_names:
            found = False
            # varset = set(name)
            # name = ':'.join(name)
            for df in X + [y]:
                # df_name = None
                # for col in df.columns:
                #     colset = set(col.split(':'))
                #     if len(varset.symmetric_difference(colset)) == 0:
                #         df_name = col
                #         break
                if name in df.columns:
                    impulse_means[name] = df[name].mean()
                    impulse_sds[name] = df[name].std()
                    impulse_medians[name] = df[name].quantile(0.5)
                    impulse_lq[name] = df[name].quantile(0.1)
                    impulse_uq[name] = df[name].quantile(0.9)
                    impulse_min[name] = df[name].min()
                    impulse_max[name] = df[name].max()
                    found = True
                    break
            if not found:
                raise ValueError('Impulse %s was not found in an input file.' % name)


        self.impulse_means = impulse_means
        self.impulse_sds = impulse_sds
        self.impulse_medians = impulse_medians
        self.impulse_lq = impulse_lq
        self.impulse_uq = impulse_uq
        self.impulse_min = impulse_min
        self.impulse_max = impulse_max

        # print(self.impulse_means)
        # print(self.impulse_sds)
        # print(self.impulse_medians)
        # print(self.impulse_lq)
        # print(self.impulse_uq)
        # print(self.impulse_min)
        # print(self.impulse_max)

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

        f = self.form
        self.dv = f.dv
        self.has_intercept = f.has_intercept
        self.rangf = f.rangf

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

        # Can't pickle defaultdict because it requires a lambda term for the default value,
        # so instead we pickle a normal dictionary (``rangf_map_base``) and compute the defaultdict
        # from it.
        self.rangf_map = []
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(
                defaultdict((lambda x: lambda: x)(self.rangf_n_levels[i] - 1), self.rangf_map_base[i]))

        self.rangf_map_ix_2_levelname = []
        for i in range(len(self.rangf_map_base)):
            ix_2_levelname = [None] * self.rangf_n_levels[i]
            for level in self.rangf_map_base[i]:
                ix_2_levelname[self.rangf_map_base[i][level]] = level
            assert ix_2_levelname[-1] is None, 'Non-null value found in rangf map for unknown level'
            ix_2_levelname[-1] = 'UNK'
            self.rangf_map_ix_2_levelname.append(ix_2_levelname)

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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.intercept_init_tf = tf.constant(self.intercept_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(4 * np.finfo(self.FLOAT_NP).eps, dtype=self.FLOAT_TF)

                self.y_sd_init_tf = tf.constant(float(self.y_sd_init), dtype=self.FLOAT_TF)
                if self.constraint.lower() == 'softplus':
                    self.y_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_sd_init_tf)
                    if self.asymmetric_error:
                        self.y_tailweight_init_unconstrained = tf.contrib.distributions.softplus_inverse(1.)
                    self.constraint_fn = tf.nn.softplus
                else:
                    self.y_sd_init_unconstrained = self.y_sd_init_tf
                    if self.asymmetric_error:
                        self.y_tailweight_init_unconstrained = 1.
                    self.constraint_fn = self._safe_abs

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
            'y_train_mean': self.y_train_mean,
            'y_train_sd': self.y_train_sd,
            'max_tdelta': self.max_tdelta,
            't_delta_sd': self.t_delta_sd,
            't_delta_limit': self.t_delta_limit,
            'rangf_map_base': self.rangf_map_base,
            'rangf_n_levels': self.rangf_n_levels,
            'impulse_means': self.impulse_means,
            'impulse_sds': self.impulse_sds,
            'impulse_medians': self.impulse_medians,
            'impulse_lq': self.impulse_lq,
            'impulse_uq': self.impulse_uq,
            'impulse_min': self.impulse_min,
            'impulse_max': self.impulse_max,
            'outdir': self.outdir
        }
        for kwarg in Model._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.form_str = md.pop('form_str')
        self.form = md.pop('form', Formula(self.form_str))
        self.n_train = md.pop('n_train')
        self.y_train_mean = md.pop('y_train_mean')
        self.y_train_sd = md.pop('y_train_sd')
        self.max_tdelta = md.pop('max_tdelta')
        self.t_delta_sd = md.pop('t_delta_sd', 1.)
        self.t_delta_limit = md.pop('t_delta_limit', self.max_tdelta)
        self.rangf_map_base = md.pop('rangf_map_base')
        self.rangf_n_levels = md.pop('rangf_n_levels')
        self.impulse_means = md.pop('impulse_means', {})
        self.impulse_sds = md.pop('impulse_sds', {})
        self.impulse_medians = md.pop('impulse_medians', {})
        self.impulse_lq = md.pop('impulse_lq', {})
        self.impulse_uq = md.pop('impulse_uq', {})
        self.impulse_min = md.pop('impulse_min', {})
        self.impulse_max = md.pop('impulse_max', {})
        self.outdir = md.pop('outdir', './cdr_model/')

        for kwarg in Model._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))




    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self, n_impulse):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.training = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='training')

                self.X = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='X'
                )
                self.time_X = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='time_X'
                )

                self.time_X_mask = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=tf.bool,
                    name='time_X_mask'
                )

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))
                # Tensor of temporal offsets with shape (?, history_length, 1)
                self.t_delta = self.time_y[..., None, None] - self.time_X
                # self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)
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
                if self.regularizer_name is None:
                    self.regularizer = None
                else:
                    self.regularizer = getattr(tf.contrib.layers, self.regularizer_name)(self.regularizer_scale)

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')

                self.training_mse_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mse_in')
                self.training_mse = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mse')
                self.set_training_mse = tf.assign(self.training_mse, self.training_mse_in)
                if self.standardize_response:
                    max_sd = 1
                else:
                    max_sd = self.y_train_sd
                self.training_percent_variance_explained = tf.maximum(0.,
                                                                      (1. - self.training_mse / (max_sd ** 2)) * 100.)

                self.training_mae_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mae_in')
                self.training_mae = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mae')
                self.set_training_mae = tf.assign(self.training_mae, self.training_mae_in)

                self.training_loglik_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_loglik_in')
                self.training_loglik = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_loglik')
                self.set_training_loglik = tf.assign(self.training_loglik, self.training_loglik_in)

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
                    self.intercept_regularizer = getattr(tf.contrib.layers, self.intercept_regularizer_name)(
                        self.intercept_regularizer_scale)

                if self.ranef_regularizer_name is None:
                    self.ranef_regularizer = None
                elif self.ranef_regularizer_name == 'inherit':
                    self.ranef_regularizer = self.regularizer
                else:
                    self.ranef_regularizer = getattr(tf.contrib.layers, self.ranef_regularizer_name)(
                        self.ranef_regularizer_scale)

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

    def _initialize_optimizer(self, name):
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
                        t = tf.cast(self.global_step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.global_step

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

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_by_iter', self.loss_total, collections=['loss'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/cdr')
                self.summary_losses = tf.summary.merge_all(key='loss')
                self.summary_params = tf.summary.merge_all(key='params')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                    self.saver = tf.train.Saver()

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

    def _center_and_constrain(self, param_random, param_fixed, lb=None, ub=None, use_softplus_sigmoid=True):
        with self.sess.as_default():
            # If the parameter is constrained, passes random effects matrix first through a non-linearity, then
            # mean-centers its columns. However, centering after constraint can cause the constraint to be violated,
            # so the recentered weights must be renormalized proportionally to their pre-centering mean in order
            # to guarantee that the constraint is obeyed. How this renormalization is done depends on whether
            # the parameter is upper-bounded, lower-bounded, or both.

            with self.sess.graph.as_default():

                if lb is None and ub is None:
                    param_random -= tf.reduce_mean(param_random, axis=0, keepdims=True)

                    return param_random

                elif lb is not None and ub is None:
                    max_lower_offset = param_fixed - (lb + self.epsilon)

                    # Enforce constraint
                    param_random = self.constraint_fn(param_random)

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_lower_offset / (param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

                elif lb is None and ub is not None:
                    max_upper_offset = ub - param_fixed - self.epsilon

                    # Enforce constraint
                    param_random = -self.constraint_fn(param_random)

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_upper_offset / (-param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

                else:
                    max_lower_offset = param_fixed - (lb + self.epsilon)

                    # Enforce constraint
                    if use_softplus_sigmoid:
                        param_random = self._softplus_sigmoid(param_random, a=lb, b=ub)
                    else:
                        max_range_param = ub - lb
                        param_random = tf.sigmoid(param_random) * max_range_param

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_lower_offset / (param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

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

    def _safe_abs(self, x):
        out = tf.where(
            tf.equal(x, 0.),
            x + 1e-8,
            tf.abs(x)
        )

        return out




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

    def _regularize(self, var, center=None, type=None, var_name=None):
        assert type in [None, 'intercept', 'coefficient', 'irf', 'ranef', 'oob']
        if type is None:
            regularizer = self.regularizer
        else:
            regularizer = getattr(self, '%s_regularizer' %type)

        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if center is None:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    else:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var - center])
                    self.regularizer_losses.append(reg)
                    self.regularizer_losses_varnames.append(str(var_name))
                    if type is None:
                        reg_name = self.regularizer_name
                        reg_scale = self.regularizer_scale
                    else:
                        reg_name = getattr(self, '%s_regularizer_name' %type)
                        reg_scale = getattr(self, '%s_regularizer_scale' %type)
                    if reg_name == 'inherit':
                        reg_name = self.regularizer_name
                    if reg_scale == 'inherit':
                        reg_scale = self.regularizer_scale
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
                        tf.zeros([int(self.convergence_n_iterates / self.convergence_stride)] + list(var.shape), dtype=self.FLOAT_TF),
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
                                (p_ta_at_min_p > self.convergence_alpha) and \
                                (proportion_converged > self.convergence_alpha)

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
                    if predict and self.ema_decay:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    stderr('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
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

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

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
    #  Shared public methods
    #
    ######################################################

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
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
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
                    for i, name in enumerate(self.regularizer_losses_varnames):
                        out += ' ' * indent + '  %s:\n' %name
                        out += ' ' * indent + '    Regularizer: %s\n' %self.regularizer_losses_names[i]
                        out += ' ' * indent + '    Scale: %s\n' %self.regularizer_losses_scales[i]

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

    def fit(self,
            X,
            y,
            n_iter=100,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            force_training_evaluation=True,
            irf_name_map=None,
            plot_n_time_units=2.5,
            plot_n_time_points=1000,
            surface_plot_n_time_points=1024,
            generate_irf_surface_plots=False,
            generate_interaction_surface_plots=False,
            generate_curvature_plots=False,
            plot_x_inches=7,
            plot_y_inches=5,
            cmap='gist_rainbow',
            dpi=300
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
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``float``; number of points to use for plotting.
        :param surface_plot_n_time_points: ``float``; number of points to use in any surface plots (CDRNN only).
        :param generate_irf_surface_plots: ``bool``; whether to generate IRF surface plots (CDRNN only)
        :param generate_interaction_surface_plots: ``bool``; whether to generate interaction surface plots (CDRNN only)
        :param generate_curvature_plots: ``bool``; whether to generate GAM-like curvature plots (CDRNN only)
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param dpi: ``int``; dots per inch.
        :return: ``None``
        """

        stderr('*' * 100 + '\n' + self.initialization_summary() + '*' * 100 + '\n\n')

        usingGPU = tf.test.is_gpu_available()
        stderr('Using GPU: %s\nNumber of training samples: %d\n\n' % (usingGPU, len(y)))

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

        stderr('Correlation matrix for input variables:\n')
        impulse_names_2d = [x for x in impulse_names if x in X_2d_predictor_names]
        rho = corr_cdr(X_2d, impulse_names, impulse_names_2d, time_X_2d, time_X_mask)
        stderr(str(rho) + '\n\n')

        self.make_plots(
            irf_name_map=irf_name_map,
            plot_n_time_units=plot_n_time_units,
            plot_n_time_points=plot_n_time_points,
            surface_plot_n_time_points=surface_plot_n_time_points,
            generate_irf_surface_plots=generate_irf_surface_plots,
            generate_interaction_surface_plots=generate_interaction_surface_plots,
            generate_curvature_plots=generate_curvature_plots,
            plot_x_inches=plot_x_inches,
            plot_y_inches=plot_y_inches,
            cmap=cmap,
            dpi=dpi,
            keep_plot_history=self.keep_plot_history
        )

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
                    else:
                        stderr('Resuming training from most recent checkpoint...\n\n')

                    while not self.has_converged() and self.global_step.eval(session=self.sess) < n_iter:
                        p, p_inv = get_random_permutation(len(y))
                        t0_iter = pytime.time()
                        stderr('-' * 50 + '\n')
                        stderr('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        stderr('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            stderr('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        pb = tf.contrib.keras.utils.Progbar(self.n_train_minibatch)

                        loss_total = 0.

                        for j in range(0, len(y), minibatch_size):
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
                            loss_cur = info_dict['loss']
                            if self.ema_decay:
                                self.sess.run(self.ema_op)
                            if not np.isfinite(loss_cur):
                                loss_cur = 0
                            loss_total += loss_cur

                            pb.update((j/minibatch_size)+1, values=[('loss', loss_cur)])

                        self.sess.run(self.incr_global_step)

                        if not type(self).__name__.startswith('CDRNN'):
                            self.verify_random_centering()

                        if self.check_convergence:
                            self.run_convergence_check(verbose=False, feed_dict={self.loss_total: loss_total/n_minibatch})

                        if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                            loss_total /= n_minibatch
                            summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                            self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                            if self.log_random and len(self.rangf) > 0:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))

                        if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                            self.save()
                            self.make_plots(
                                irf_name_map=irf_name_map,
                                plot_n_time_units=plot_n_time_units,
                                plot_n_time_points=plot_n_time_points,
                                surface_plot_n_time_points=surface_plot_n_time_points,
                                generate_irf_surface_plots=generate_irf_surface_plots,
                                generate_interaction_surface_plots=generate_interaction_surface_plots,
                                generate_curvature_plots=generate_curvature_plots,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                cmap=cmap,
                                dpi=dpi,
                                keep_plot_history=self.keep_plot_history
                            )

                        t1_iter = pytime.time()
                        stderr('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                    self.save()

                    # End of training plotting and evaluation.
                    # For CDRMLE, this is a crucial step in the model definition because it provides the
                    # variance of the output distribution for computing log likelihood.

                    self.make_plots(
                        irf_name_map=irf_name_map,
                        plot_n_time_units=plot_n_time_units,
                        plot_n_time_points=plot_n_time_points,
                        surface_plot_n_time_points=surface_plot_n_time_points,
                        generate_irf_surface_plots=generate_irf_surface_plots,
                        generate_interaction_surface_plots=generate_interaction_surface_plots,
                        generate_curvature_plots=generate_curvature_plots,
                        plot_x_inches=plot_x_inches,
                        plot_y_inches=plot_y_inches,
                        cmap=cmap,
                        dpi=dpi,
                        keep_plot_history=self.keep_plot_history
                    )

                    if type(self).__name__ == 'CDRBayes':
                        # Generate plots with 95% credible intervals
                        self.make_plots(
                            irf_name_map=irf_name_map,
                            plot_n_time_units=plot_n_time_units,
                            plot_n_time_points=plot_n_time_points,
                            surface_plot_n_time_points=surface_plot_n_time_points,
                            generate_irf_surface_plots=generate_irf_surface_plots,
                            generate_interaction_surface_plots=generate_interaction_surface_plots,
                            generate_curvature_plots=generate_curvature_plots,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap,
                            dpi=dpi,
                            mc=True,
                            keep_plot_history=self.keep_plot_history
                        )


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

                    with open(self.outdir + '/preds_train.txt', 'w') as p_file:
                        for i in range(len(preds)):
                            p_file.write(str(preds[i]) + '\n')

                    # Extract and save losses
                    training_se = np.array((y[self.dv] - preds) ** 2)
                    training_mse = training_se.mean()
                    training_percent_variance_explained = percent_variance_explained(y[self.dv], preds)

                    training_ae = np.array(np.abs(y[self.dv] - preds))
                    training_mae = training_ae.mean()

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

                    self.save()

                    with open(self.outdir + '/eval_train.txt', 'w') as e_file:
                        eval_train = '------------------------\n'
                        eval_train += 'CDR TRAINING EVALUATION\n'
                        eval_train += '------------------------\n\n'
                        eval_train += self.report_formula_string(indent=2)
                        eval_train += self.report_evaluation(
                            mse=training_mse,
                            mae=training_mae,
                            loglik=training_loglik,
                            percent_variance_explained=training_percent_variance_explained,
                            indent=2
                        )

                        e_file.write(eval_train)

                    if not type(self).__name__.startswith('CDRNN'):
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

                if not np.isfinite(self.minibatch_size):
                    fd = {
                        self.X: X_2d,
                        self.time_X: time_X_2d,
                        self.time_X_mask: time_X_mask,
                        self.time_y: time_y,
                        self.gf_y: gf_y,
                        self.y: y_dv,
                        self.training: not self.predict_mode
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
                            self.training: not self.predict_mode
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
                fd = {
                    self.n_errors: n_errors,
                    self.training: not self.predict_mode
                }
                err_q = self.sess.run(self.err_dist_summary_theoretical_quantiles, feed_dict=fd)

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

    def get_plot_names(self, composite='composite', scaled='scaled', dirac='dirac', plot_type='irf_1d'):
        raise NotImplementedError

    def get_plot_data(
            self,
            name,
            composite='composite',
            scaled='scaled',
            dirac='dirac',
            plot_type='irf_1d',
            support_start=0.,
            n_time_units=2.5,
            n_time_points=1000,
            t_interaction=0.,
            plot_rangf=False,
            rangf_vals=None
    ):
        raise NotImplementedError

    def make_plots(
            self,
            standardize_response=False,
            summed=False,
            irf_name_map=None,
            irf_ids=None,
            sort_names=True,
            plot_unscaled=True,
            plot_composite=False,
            prop_cycle_length=None,
            prop_cycle_ix=None,
            plot_dirac=False,
            plot_interactions=False,
            t_interaction=0.,
            plot_rangf=False,
            plot_n_time_units=2.5,
            plot_n_time_points=1000,
            surface_plot_n_time_points=1024,
            generate_irf_surface_plots=False,
            generate_interaction_surface_plots=False,
            generate_curvature_plots=False,
            plot_x_inches=6.,
            plot_y_inches=4.,
            ylim=None,
            cmap=None,
            dpi=300,
            mc=False,
            level=95,
            n_samples=None,
            prefix=None,
            legend=True,
            xlab=None,
            ylab=None,
            use_line_markers=False,
            transparent_background=False,
            keep_plot_history=False,
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
        :param summed: ``bool``; whether to plot individual IRFs or their sum.
        :param irf_ids: ``list`` or ``None``; list of irf ID's to plot. If ``None``, all IRF's are plotted.
        :param sort_names: ``bool``; alphabetically sort IRF names.
        :param plot_unscaled: ``bool``; plot unscaled IRFs.
        :param plot_composite: ``bool``; plot any composite IRFs. If ``False``, only plots terminal IRFs.
        :param prop_cycle_length: ``int`` or ``None``; Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **irf_names**.
        :param prop_cycle_ix: ``list`` of ``int``, or ``None``; Integer indices to use in the properties cycle for each entry in **irf_names**. If ``None``, indices are automatically assigned.
        :param plot_dirac: ``bool``; include any linear Dirac delta IRF's (stick functions at t=0) in plot.
        :param plot_interactions: ``bool``; plot all predictor interactions (CDRNN only)
        :param t_interaction: ``float``; timepoint at which to plot interactions (CDRNN only)
        :param plot_rangf: ``bool``; plot all (marginal) random effects.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``int``; number of points to use for plotting.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param ylim: 2-element ``tuple`` or ``list``; (lower_bound, upper_bound) to use for y axis. If ``None``, automatically inferred.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param dpi: ``int``; dots per inch.
        :param mc: ``bool``; compute and plot Monte Carlo credible intervals (only supported for CDRBayes).
        :param level: ``float``; significance level for credible intervals, ignored unless **mc** is ``True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param prefix: ``str`` or ``None``; prefix appended to output filenames. If ``None``, no prefix added.
        :param legend: ``bool``; generate a plot legend.
        :param xlab: ``str`` or ``None``; x-axis label. If ``None``, no label.
        :param ylab: ``str`` or ``None``; y-axis label. If ``None``, no label.
        :param transparent_background: ``bool``; use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot.
        :param dump_source: ``bool``; Whether to dump the plot source array to a csv file.
        :return: ``None``
        """

        assert not mc or type(self).__name__ == 'CDRBayes', 'Monte Carlo estimation of credible intervals (mc=True) is only supported for CDRBayes models.'

        if mc and not hasattr(self, 'ci_curve'):
            stderr('Credible intervals are not supported for instances of %s. Re-run ``make_plots`` with ``mc=False``.\n' % type(self))
            mc = False

        if len(self.terminal_names) == 0:
            return

        if plot_dirac:
            dirac = 'dirac'
        else:
            dirac = 'nodirac'

        if prefix is None:
            prefix = ''
        if prefix != '':
            prefix += '_'

        if summed:
            alpha = 100 - float(level)

        rangf_keys = [None]
        rangf_vals = [self.gf_defaults[0]]
        if plot_rangf:
            for i in range(len(self.rangf)):
                if type(self).__name__.startswith('CDRNN') or self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                    for k in self.rangf_map[i].keys():
                        rangf_keys.append(str(k))
                        rangf_vals.append(np.concatenate([self.gf_defaults[0, :i], [self.rangf_map[i][k]], self.gf_defaults[0, i+1:]], axis=0))
        rangf_vals = np.stack(rangf_vals, axis=0)

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

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: plot_n_time_units,
                    self.n_time_points: plot_n_time_points,
                    self.max_tdelta_batch: plot_n_time_units,
                    self.gf_y: rangf_vals,
                    self.training: not self.predict_mode
                }

                if hasattr(self, 'n_surface_plot_points'):
                    fd[self.n_surface_plot_points] = surface_plot_n_time_points

                # CDRNN Plots
                if not mc:
                    if generate_irf_surface_plots and hasattr(self, 'irf_surface_plot'):
                        plot_x = []
                        plot_y = []
                        plot_z = []
                        names = self.get_plot_names(plot_type='irf_surface')
                        for name in names:
                            (x_cur, y_cur), z_cur = self.get_plot_data(
                                name,
                                plot_type='irf_surface',
                                support_start=0.,
                                n_time_units=plot_n_time_units,
                                n_time_points=plot_n_time_points,
                                t_interaction=t_interaction,
                                plot_rangf=plot_rangf,
                                rangf_vals=rangf_vals
                            )

                            plot_x.append(x_cur)
                            plot_y.append(y_cur)
                            plot_z.append(z_cur)

                        if self.standardize_response and not standardize_response:
                            plot_z = [x * self.y_train_sd for x in plot_z]

                        plot_surface(
                            plot_x,
                            plot_y,
                            plot_z,
                            names,
                            sort_names=True,
                            dir=self.outdir,
                            prefix='irf_surface_plot',
                            irf_name_map=irf_name_map,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            ylim=ylim,
                            transparent_background=transparent_background,
                            dpi=dpi,
                            dump_source=dump_source
                        )

                    if generate_curvature_plots and hasattr(self, 'curvature_plot'):
                        names = self.get_plot_names(plot_type='curvature')
                        for name in names:
                            x_cur, y_cur = self.get_plot_data(
                                name,
                                plot_type='curvature',
                                support_start=0.,
                                n_time_units=plot_n_time_units,
                                n_time_points=plot_n_time_points,
                                t_interaction=t_interaction,
                                plot_rangf=plot_rangf,
                                rangf_vals=rangf_vals
                            )

                            xlab_cur = irf_name_map.get(name, name)
                            filename = 'curvature_plot_t%s_%s.png' % (t_interaction, sn(name))

                            if self.standardize_response and not standardize_response:
                                y_cur *= self.y_train_sd

                            plot_irf(
                                x_cur,
                                y_cur,
                                [name],
                                dir=self.outdir,
                                filename=filename,
                                irf_name_map=irf_name_map,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                ylim=ylim,
                                cmap=cmap,
                                dpi=dpi,
                                legend=legend,
                                xlab=xlab_cur,
                                ylab=ylab,
                                use_line_markers=use_line_markers,
                                transparent_background=transparent_background,
                                dump_source=dump_source
                            )

                    if generate_interaction_surface_plots and hasattr(self, 'interaction_surface_plot'):
                        plot_x = []
                        plot_y = []
                        plot_z = []
                        names = self.get_plot_names(plot_type='interaction_surface')
                        print(names)
                        for name in names:
                            (x_cur, y_cur), z_cur = self.get_plot_data(
                                name,
                                plot_type='interaction_surface',
                                support_start=0.,
                                n_time_units=plot_n_time_units,
                                n_time_points=plot_n_time_points,
                                t_interaction=t_interaction,
                                plot_rangf=plot_rangf,
                                rangf_vals=rangf_vals
                            )
                            plot_x.append(x_cur)
                            plot_x.append(y_cur)
                            plot_x.append(z)

                        if self.standardize_response and not standardize_response:
                            plot_z = [x * self.y_train_sd for x in plot_z]

                        plot_surface(
                            plot_x,
                            plot_y,
                            plot_z,
                            names,
                            sort_names=True,
                            dir=self.outdir,
                            prefix='interaction_surface_plot',
                            irf_name_map=irf_name_map,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            ylim=ylim,
                            transparent_background=transparent_background,
                            dpi=dpi,
                            dump_source=dump_source
                        )

                plot_x = self.sess.run(self.support, fd)

                switches = [['atomic'], ['scaled']]
                if plot_composite:
                    switches[0].append('composite')
                if plot_unscaled:
                    switches[1].append('unscaled')

                for a in switches[0]:
                    if self.t.has_composed_irf() or a == 'atomic':
                        for b in switches[1]:
                            if summed:
                                plot_name = 'irf_%s_%s_summed_%d.png' %(a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'irf_%s_%s_summed.png' %(a, b)
                            else:
                                plot_name = 'irf_%s_%s_%d.png' %(a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'irf_%s_%s.png' %(a, b)
                            names = self.get_plot_names(composite=a, scaled=b, dirac=dirac, plot_type='irf_1d')
                            if irf_ids is not None and len(irf_ids) > 0:
                                new_names = []
                                for i, name in enumerate(names):
                                    for ID in irf_ids:
                                        if ID==name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                                            new_names.append(name)
                                names = new_names
                            if len(names) > 0:
                                if mc:
                                    names_cur = names[:]
                                    if summed:
                                        samples = []
                                    else:
                                        plot_y = []
                                        lq = []
                                        uq = []
                                    for name in names:
                                        mean_cur, lq_cur, uq_cur, samples_cur = self.ci_curve(
                                            self.irf_mc[name][a][b],
                                            rangf=rangf_vals,
                                            level=level,
                                            n_samples=n_samples,
                                            n_time_units=plot_n_time_units,
                                            n_time_points=plot_n_time_points,
                                        )

                                        if summed:
                                            samples.append(samples_cur)
                                        else:
                                            plot_y.append(mean_cur)
                                            lq.append(lq_cur)
                                            uq.append(uq_cur)

                                    if summed:
                                        samples = np.stack(samples, axis=3)
                                        samples = samples.sum(axis=3, keepdims=True)
                                        lq = np.percentile(samples, alpha / 2, axis=2)
                                        uq = np.percentile(samples, 100 - (alpha / 2), axis=2)
                                        plot_y = samples.mean(axis=2)
                                    else:
                                        lq = np.stack(lq, axis=2)
                                        uq = np.stack(uq, axis=2)
                                        plot_y = np.stack(plot_y, axis=2)

                                    if self.standardize_response and not standardize_response:
                                        plot_y *= self.y_train_sd
                                        lq *= self.y_train_sd
                                        uq *= self.y_train_sd

                                    plot_name = 'mc_' + plot_name

                                else:
                                    plot_y = []
                                    names_cur = []
                                    for name in names:
                                        _, y_cur = self.get_plot_data(
                                            name,
                                            composite=a,
                                            scaled=b,
                                            dirac=dirac,
                                            plot_type='irf_1d',
                                            support_start=0.,
                                            n_time_units=plot_n_time_units,
                                            n_time_points=plot_n_time_points,
                                            plot_rangf=plot_rangf,
                                            rangf_vals=rangf_vals
                                        )
                                        plot_y.append(y_cur)
                                        names_cur.append(name)
                                    for i, plot_y_cur in enumerate(plot_y):
                                        if len(plot_y_cur) == 1 and len(rangf_vals) > 1:
                                            plot_y_cur = np.repeat(plot_y_cur, len(rangf_vals), axis=0)
                                        plot_y[i] = plot_y_cur
                                    lq = None
                                    uq = None
                                    plot_y = np.concatenate(plot_y, axis=-1)
                                    if summed:
                                        plot_y = plot_y.sum(axis=-1, keepdims=True)
                                    if self.standardize_response and not standardize_response:
                                        plot_y *= self.y_train_sd

                                if summed:
                                    names_cur = ['Sum']

                                for g in range(len(rangf_keys)):
                                    if rangf_keys[g]:
                                        filename = prefix + rangf_keys[g] + '_' + plot_name
                                    else:
                                        filename = prefix + plot_name
                                    plot_irf(
                                        plot_x,
                                        plot_y[g],
                                        names_cur,
                                        lq=None if lq is None else lq[g],
                                        uq=None if uq is None else uq[g],
                                        sort_names=sort_names,
                                        prop_cycle_length=prop_cycle_length,
                                        prop_cycle_ix=prop_cycle_ix,
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

                if hasattr(self, 'pc') and self.pc:
                    for a in switches[0]:
                        if self.t.has_composed_irf() or a == 'atomic':
                            for b in switches[1]:
                                if b == 'scaled':
                                    if summed:
                                        plot_name = 'src_irf_%s_%s_summed_%d.png' % (a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'src_irf_%s_%s_summed.png' % (a, b)
                                    else:
                                        plot_name = 'src_irf_%s_%s_%d.png' % (a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'src_irf_%s_%s.png' % (a, b)
                                    names = self.src_plot_tensors[a][b][dirac]['names']
                                    if irf_ids is not None and len(irf_ids) > 0:
                                        new_names = []
                                        for i, name in enumerate(names):
                                            for ID in irf_ids:
                                                if ID == name or re.match(ID if ID.endswith('$') else ID + '$',
                                                                          name) is not None:
                                                    new_names.append(name)
                                        names = new_names
                                    if len(names) > 0:
                                        if mc:
                                            if summed:
                                                samples = []
                                            else:
                                                plot_y = []
                                                lq = []
                                                uq = []
                                            for name in names:
                                                mean_cur, lq_cur, uq_cur, samples_cur = self.ci_curve(
                                                    self.src_irf_mc[name][a][b],
                                                    level=level,
                                                    n_samples=n_samples,
                                                    n_time_units=plot_n_time_units,
                                                    n_time_points=plot_n_time_points,
                                                )

                                                if summed:
                                                    samples.append(samples_cur)
                                                else:
                                                    plot_y.append(mean_cur)
                                                    lq.append(lq_cur)
                                                    uq.append(uq_cur)

                                            if summed:
                                                samples = np.stack(samples, axis=2)
                                                samples = samples.sum(axis=2, keepdims=True)
                                                lq = np.percentile(samples, alpha / 2, axis=1)
                                                uq = np.percentile(samples, 100 - (alpha / 2), axis=1)
                                                plot_y = samples.mean(axis=1)
                                            else:
                                                lq = np.stack(lq, axis=1)
                                                uq = np.stack(uq, axis=1)
                                                plot_y = np.stack(plot_y, axis=1)

                                            if self.standardize_response and not standardize_response:
                                                plot_y = plot_y * self.y_train_sd
                                                lq = lq * self.y_train_sd
                                                uq = uq * self.y_train_sd

                                            plot_name = 'mc_' + plot_name

                                        else:
                                            plot_y = [self.sess.run(self.src_plot_tensors[a][b][dirac]['plot'][i], feed_dict=fd) for i in range(len(self.src_plot_tensors[a][b][dirac]['plot'])) if self.src_plot_tensors[a][b][dirac]['names'][i] in names]
                                            lq = None
                                            uq = None
                                            plot_y = np.concatenate(plot_y, axis=1)
                                            if summed:
                                                plot_y = plot_y.sum(axis=1, keepdims=True)
                                            if self.standardize_response and not standardize_response:
                                                plot_y = plot_y * self.y_train_sd

                                        if summed:
                                            names_cur = ['Sum']
                                        else:
                                            names_cur = names

                                        plot_irf(
                                            plot_x,
                                            plot_y,
                                            names_cur,
                                            lq=lq,
                                            uq=uq,
                                            sort_names=sort_names,
                                            prop_cycle_length=prop_cycle_length,
                                            prop_cycle_ix=prop_cycle_ix,
                                            dir=self.outdir,
                                            filename=prefix + plot_name,
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