import os
from collections import defaultdict
import textwrap
from numpy import inf
import pandas as pd
import time as pytime

from .formula import *
from .kwargs import DTSR_INITIALIZATION_KWARGS
from .util import *
from .data import build_DTSR_impulses, corr_dtsr
from .plot import *
from .interpolate_spline import interpolate_spline

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


######################################################
#
#  ABSTRACT DTSR CLASS
#
######################################################

class DTSR(object):

    _INITIALIZATION_KWARGS = DTSR_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for DTSR. Bayesian (:ref:`dtsrbayes`) and MLE (:ref:`dtsrmle`) implementations inherit from ``DTSR``.
        ``DTSR`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``DTSR`` must implement the following instance methods:
        
            * ``initialize_intercept()``
            * ``initialize_coefficient()``
            * ``initialize_irf_param_unconstrained()``
            * ``initialize_joint_distribution()``
            * ``initialize_objective()``
            * ``run_conv_op()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            * ``run_train_step()``
            
        Additionally, if the subclass requires any keyword arguments beyond those provided by ``DTSR``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
        
        Example implementations of each of these methods can be found in the source code for :ref:`dtsrmle` and :ref:`dtsrbayes`.
        
    """
    _doc_args = """
        :param form_str: An R-style string representing the DTSR model formula.
        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization
        :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:
    
            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each observation in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each observation in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    def __new__(cls, *args, **kwargs):
        if cls is DTSR:
            raise TypeError("DTSR is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, y, **kwargs):

        ## Store initialization settings
        self.form_str = form_str
        for kwarg in DTSR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        # Parse and store model data from formula
        form = Formula(self.form_str)
        dv = form.dv
        rangf = form.rangf

        # Compute from training data
        self.n_train = len(y)
        self.y_train_mean = float(y[dv].mean())
        self.y_train_sd = float(y[dv].std())
        last_obs = np.array(y.last_obs - 1, dtype=getattr(np, self.int_type))
        first_obs = np.maximum(np.array(y.first_obs, dtype=getattr(np, self.int_type)), last_obs - self.history_length + 1)
        X_time = np.array(X.time, dtype=getattr(np, self.float_type))
        t_delta = X_time[last_obs] - X_time[first_obs]
        self.max_tdelta = (t_delta).max()

        if self.pc:
            _, self.eigenvec, self.eigenval, self.impulse_means, self.impulse_sds = pca(X[self.src_impulse_names_norate])
            self.plot_eigenvectors()
        else:
            self.eigenvec = self.eigenval = self.impulse_means = self.impulse_sds = None

        ## Set up hash table for random effects lookup
        self.rangf_map_base = []
        self.rangf_n_levels = []
        for i in range(len(rangf)):
            gf = rangf[i]
            keys = np.sort(y[gf].astype('str').unique())
            vals = np.arange(len(keys), dtype=getattr(np, self.int_type))
            rangf_map = pd.DataFrame({'id':vals},index=keys).to_dict()['id']
            self.rangf_map_base.append(rangf_map)
            self.rangf_n_levels.append(len(keys) + 1)

        self._initialize_session()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        ## Compute secondary data from intialization settings
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        self.form = Formula(self.form_str)
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

        # Initialize lookup tables of network objects
        self.irf_lambdas = {}
        self.irf_params_means = {} # {family: {param_name: mean_vector}}
        self.irf_params_means_unconstrained = {} # {family: {param_name: mean_init_vector}}
        self.irf_params_random_means = {} # {rangf: {family: {param_name: mean_vector}}}
        self.irf_params_lb = {} # {family: {param_name: value}}
        self.irf_params_ub = {} # {family: {param_name: value}}
        self.irf_params = {} # {irf_id: param_vector}
        self.irf_params_summary = {} # {irf_id: param_summary_vector}
        self.irf_params_fixed = {} # {irf_id: param_vector}
        self.irf_params_fixed_summary = {} # {irf_id: param_summary_vector}
        self.irf_params_fixed_base = {}  # {family: {param_name: param_vector}}
        self.irf_params_fixed_base_summary = {}  # {family: {param_name: param_summary_vector}}
        self.irf_params_random = {} # {rangf: {irf_id: param_matrix}}
        self.irf_params_random_summary = {} # {rangf: {irf_id: param_matrix}}
        self.irf_params_random_base = {}  # {rangf: {family: {irf_id: param_matrix}}
        self.irf_params_random_base_summary = {}  # {rangf: {family: {irf_id: param_summary_matrix}}
        self.irf = {}
        self.irf_plot = {}
        self.irf_mc = {}
        self.irf_integral_tensors = {}
        if self.pc:
            self.src_irf_plot = {}
            self.src_irf_mc = {}
            self.src_irf_integral_tensors = {}
        self.irf_impulses = {}
        self.convolutions = {}

        # Initialize model metadata

        if self.pc:
            # Initialize source tree metadata
            self.t_src = self.form.t
            t_src = self.t_src
            self.src_node_table = t_src.node_table()
            self.src_coef_names = t_src.coef_names()
            self.src_fixed_coef_names = t_src.fixed_coef_names()
            self.src_impulse_names = t_src.impulse_names()
            self.src_terminal_names = t_src.terminal_names()
            self.src_atomic_irf_names_by_family = t_src.atomic_irf_by_family()
            self.src_atomic_irf_family_by_name = {}
            for family in self.src_atomic_irf_names_by_family:
                for id in self.src_atomic_irf_names_by_family[family]:
                    assert id not in self.src_atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' % id
                    self.src_atomic_irf_family_by_name[id] = family
            self.src_param_init_by_family = t_src.atomic_irf_param_init_by_family()
            self.src_param_trainable_by_family = t_src.atomic_irf_param_trainable_by_family()
            self.src_coef2impulse = t_src.coef2impulse()
            self.src_impulse2coef = t_src.impulse2coef()
            self.src_coef2terminal = t_src.coef2terminal()
            self.src_terminal2coef = t_src.terminal2coef()
            self.src_impulse2terminal = t_src.impulse2terminal()
            self.src_terminal2impulse = t_src.terminal2impulse()
            self.src_coef_by_rangf = t_src.coef_by_rangf()
            self.src_irf_by_rangf = t_src.irf_by_rangf()

            # Initialize PC tree metadata
            self.n_pc = len(self.src_impulse_names)
            self.has_rate = 'rate' in self.src_impulse_names
            if self.has_rate:
                self.n_pc -= 1
            pointers = {}
            self.t = self.t_src.pc_transform(self.n_pc, pointers)[0]
            self.fw_pointers, self.bw_pointers = IRFNode.pointers2namemmaps(pointers)
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.impulse_names = t.impulse_names()
            self.terminal_names = t.terminal_names()
            self.atomic_irf_names_by_family = t.atomic_irf_by_family()
            self.atomic_irf_family_by_name = {}
            for family in self.atomic_irf_names_by_family:
                for id in self.atomic_irf_names_by_family[family]:
                    assert id not in self.atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' %id
                    self.atomic_irf_family_by_name[id] = family
            self.atomic_irf_param_init_by_family = t.atomic_irf_param_init_by_family()
            self.atomic_irf_param_trainable_by_family = t.atomic_irf_param_trainable_by_family()
            self.coef2impulse = t.coef2impulse()
            self.impulse2coef = t.impulse2coef()
            self.coef2terminal = t.coef2terminal()
            self.terminal2coef = t.terminal2coef()
            self.impulse2terminal = t.impulse2terminal()
            self.terminal2impulse = t.terminal2impulse()
            self.coef_by_rangf = t.coef_by_rangf()
            self.irf_by_rangf = t.irf_by_rangf()

            # Compute names and indices of source impulses excluding rate term
            self.src_impulse_names_norate = list(filter(lambda x: x != 'rate', self.src_impulse_names))
            self.src_terminal_ix_norate = names2ix(self.src_impulse_names_norate, self.src_impulse_names)
            self.src_terminal_ix_rate = np.setdiff1d(np.arange(len(self.src_impulse_names)),
                                                     self.src_impulse_names_norate)

            # Compute names and indices of PC impulses excluding rate term
            self.impulse_names_norate = list(filter(lambda x: x != 'rate', self.impulse_names))
            self.terminal_ix_norate = names2ix(self.impulse_names_norate, self.impulse_names)
            self.terminal_ix_rate = np.setdiff1d(np.arange(len(self.impulse_names)), self.impulse_names_norate)

            # Compute names and indices of source coefficients excluding rate term
            self.src_coef_names_norate = list(filter(
                lambda x: not ('rate' in self.src_impulse2coef and x in self.src_impulse2coef['rate']),
                self.src_coef_names
            ))
            self.src_coef_ix_norate = names2ix(self.src_coef_names_norate, self.src_coef_names)
            self.src_coef_names_rate = list(filter(
                lambda x: 'rate' in self.src_impulse2coef and x in self.src_impulse2coef['rate'],
                self.src_coef_names
            ))
            self.src_coef_ix_rate = names2ix(self.src_coef_names_rate, self.src_coef_names)

            # Compute names and indices of PC coefficients excluding rate term
            self.coef_names_norate = list(filter(
                lambda x: not ('rate' in self.impulse2coef and x in self.impulse2coef['rate']),
                self.coef_names
            ))
            self.coef_ix_norate = names2ix(self.src_coef_names_norate, self.src_coef_names)
            self.coef_names_rate = list(filter(
                lambda x: 'rate' in self.impulse2coef and x in self.impulse2coef['rate'],
                self.coef_names
            ))
            self.coef_ix_rate = names2ix(self.coef_names_rate, self.coef_names)
        else:
            # Initialize tree metadata
            self.t = self.form.t
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.impulse_names = t.impulse_names()
            self.terminal_names = t.terminal_names()
            self.atomic_irf_names_by_family = t.atomic_irf_by_family()
            self.atomic_irf_family_by_name = {}
            for family in self.atomic_irf_names_by_family:
                for id in self.atomic_irf_names_by_family[family]:
                    assert id not in self.atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' % id
                    self.atomic_irf_family_by_name[id] = family
            self.atomic_irf_param_init_by_family = t.atomic_irf_param_init_by_family()
            self.atomic_irf_param_trainable_by_family = t.atomic_irf_param_trainable_by_family()
            self.coef2impulse = t.coef2impulse()
            self.impulse2coef = t.impulse2coef()
            self.coef2terminal = t.coef2terminal()
            self.terminal2coef = t.terminal2coef()
            self.impulse2terminal = t.impulse2terminal()
            self.terminal2impulse = t.terminal2impulse()
            self.coef_by_rangf = t.coef_by_rangf()
            self.irf_by_rangf = t.irf_by_rangf()

        if self.log_random:
            self.summary_random_writers = {}
            self.summary_random_indexers = {}
            self.summary_random = {}

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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.intercept_init is None:
                    self.intercept_init = self.y_train_mean
                self.intercept_init_tf = tf.constant(self.intercept_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(2 * np.finfo(self.FLOAT_NP).eps, dtype=self.FLOAT_TF)

        self.parameter_table_columns = ['Estimate']

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
            'n_train': self.n_train,
            'y_train_mean': self.y_train_mean,
            'y_train_sd': self.y_train_sd,
            'max_tdelta': self.max_tdelta,
            'rangf_map_base': self.rangf_map_base,
            'rangf_n_levels': self.rangf_n_levels,
            'outdir': self.outdir,
        }
        for kwarg in DTSR._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.form_str = md.pop('form_str')
        self.n_train = md.pop('n_train')
        self.y_train_mean = md.pop('y_train_mean')
        self.y_train_sd = md.pop('y_train_sd')
        self.max_tdelta = md.pop('max_tdelta', 10.)
        self.rangf_map_base = md.pop('rangf_map_base')
        self.rangf_n_levels = md.pop('rangf_n_levels')
        self.outdir = md.pop('outdir', './dtsr_model/')

        for kwarg in DTSR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pc:
                    n_impulse = len(self.src_impulse_names)
                else:
                    n_impulse = len(self.impulse_names)

                self.X = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='X'
                )
                self.time_X = tf.placeholder(
                    shape=[None, self.history_length],
                    dtype=self.FLOAT_TF,
                    name='time_X'
                )

                self.time_X_mask = tf.placeholder(
                    shape=[None, self.history_length],
                    dtype=tf.bool,
                    name='time_X_mask'
                )

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))
                self.t_delta = tf.expand_dims(tf.expand_dims(self.time_y, -1) - self.time_X, -1)  # Tensor of temporal offsets with shape (?, history_length, 1)
                self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)

                # Tensors used for interpolated IRF composition
                self.max_tdelta_batch = tf.reduce_max(self.t_delta)
                self.interpolation_support = tf.linspace(0., self.max_tdelta_batch, self.n_interp)[..., None]

                # Linspace tensor used for plotting
                self.support_start = tf.placeholder(self.FLOAT_TF, shape=[], name='support_start')
                self.n_time_units = tf.placeholder(self.FLOAT_TF, shape=[], name='n_time_units')
                self.n_time_points = tf.placeholder(self.INT_TF, shape=[], name='n_time_points')
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

                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)
                    rate_ix = names2ix('rate', self.src_impulse_names)
                    self.X_rate = tf.gather(self.X, rate_ix, axis=-1)

                # Initialize regularizers
                if self.regularizer_name is None:
                    self.regularizer = None
                else:
                    self.regularizer = getattr(tf.contrib.layers, self.regularizer_name)(self.regularizer_scale)

                if self.intercept_regularizer_name is None:
                    self.intercept_regularizer = None
                elif self.intercept_regularizer_name == 'inherit':
                    self.intercept_regularizer = self.regularizer
                else:
                    self.intercept_regularizer = getattr(tf.contrib.layers, self.intercept_regularizer_name)(self.intercept_regularizer_scale)
                    
                if self.coefficient_regularizer_name is None:
                    self.coefficient_regularizer = None
                elif self.coefficient_regularizer_name == 'inherit':
                    self.coefficient_regularizer = self.regularizer
                else:
                    self.coefficient_regularizer = getattr(tf.contrib.layers, self.coefficient_regularizer_name)(self.coefficient_regularizer_scale)
                    
                if self.irf_regularizer_name is None:
                    self.irf_regularizer = None
                elif self.irf_regularizer_name == 'inherit':
                    self.irf_regularizer = self.regularizer
                else:
                    self.irf_regularizer = getattr(tf.contrib.layers, self.irf_regularizer_name)(self.irf_regularizer_scale)
                    
                if self.ranef_regularizer_name is None:
                    self.ranef_regularizer = None
                elif self.ranef_regularizer_name == 'inherit':
                    self.ranef_regularizer = self.regularizer
                else:
                    self.ranef_regularizer = getattr(tf.contrib.layers, self.ranef_regularizer_name)(self.ranef_regularizer_scale)

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')

                self.training_mse_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mse_in')
                self.training_mse = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mse')
                self.set_training_mse = tf.assign(self.training_mse, self.training_mse_in)
                self.training_percent_variance_explained = tf.maximum(0., (1. - self.training_mse / (self.y_train_sd ** 2)) * 100.)

                self.training_mae_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mae_in')
                self.training_mae = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mae')
                self.set_training_mae = tf.assign(self.training_mae, self.training_mae_in)

                self.training_loglik_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_loglik_in')
                self.training_loglik = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_loglik')
                self.set_training_loglik = tf.assign(self.training_loglik, self.training_loglik_in)

    def _initialize_base_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.has_intercept[None]:
                    if not self.covarying_fixef:
                        self.intercept_fixed_base, self.intercept_fixed_base_summary = self.initialize_intercept()
                else:
                    self.intercept_fixed_base = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')

                coef_ids = self.fixed_coef_names
                if len(coef_ids) > 0 and not self.covarying_fixef:
                    self.coefficient_fixed_base, self.coefficient_fixed_base_summary = self.initialize_coefficient(coef_ids=coef_ids)

                self.intercept_random_base = {}
                self.intercept_random_base_summary = {}

                self.coefficient_random_base = {}
                self.coefficient_random_base_summary = {}

                for i in range(len(self.rangf)):
                    gf = self.rangf[i]

                    if self.has_intercept[gf]:
                        if not self.covarying_ranef:
                            self.intercept_random_base[gf], self.intercept_random_base_summary[gf] = self.initialize_intercept(ran_gf=gf)

                    coef_ids = self.coef_by_rangf.get(gf, [])
                    if len(coef_ids) > 0 and not self.covarying_ranef:
                        self.coefficient_random_base[gf], self.coefficient_random_base_summary[gf] = self.initialize_coefficient(
                            coef_ids=coef_ids,
                            ran_gf=gf,
                        )

                for family in self.atomic_irf_names_by_family:
                    if family == 'DiracDelta':
                        continue

                    elif family == 'Exp':
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ExpRateGT1':
                        self._initialize_base_irf_param('beta', family, lb=1., default=2.)

                    elif family == 'Gamma':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family in ['GammaKgt1', 'GammaShapeGT1']:
                        self._initialize_base_irf_param('alpha', family, lb=1., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ShiftedGamma':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family in ['ShiftedGammaKgt1', 'ShiftedGammaShapeGT1']:
                        self._initialize_base_irf_param('alpha', family, lb=1., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family == 'Normal':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma', family, lb=0., default=1.)

                    elif family == 'SkewNormal':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha', family, default=1.)

                    elif family == 'EMG':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'BetaPrime':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ShiftedBetaPrime':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family == 'HRFSingleGamma':
                        self._initialize_base_irf_param('alpha', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'HRFDoubleGamma':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha_undershoot_offset', family, lb=0., default=10.)
                        # self._initialize_base_irf_param('c', family, lb=0., ub=1., default=1./6.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGammaUnconstrained':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta_main', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=0., default=16.)
                        self._initialize_base_irf_param('beta_undershoot', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif Formula.is_spline(family):
                        bases = Formula.bases(family)
                        spacing_power = Formula.spacing_power(family)

                        x_init = np.cumsum(np.ones(bases-1)) ** spacing_power
                        x_init *= self.max_tdelta / x_init[-1]
                        x_init[1:] -= x_init[:-1]

                        for param_name in Formula.irf_params(family):
                            if param_name.startswith('x'):
                                n = int(param_name[1:])
                                default = x_init[n-2]
                            else:
                                default = 0
                            self._initialize_base_irf_param(param_name, family, default=default)

    def _initialize_intercepts_coefficients(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                if self.has_intercept[None]:
                    self.intercept_fixed = self.intercept_fixed_base
                    self.intercept_fixed_summary = self.intercept_fixed_base_summary
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_fixed_summary,
                        collections=['params']
                    )
                    self._regularize(self.intercept_fixed, type='intercept', var_name='intercept')
                else:
                    self.intercept_fixed = self.intercept_fixed_base

                fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)

                coef_ids = self.coef_names
                self.coefficient_fixed = self._scatter_along_axis(
                    fixef_ix,
                    self.coefficient_fixed_base,
                    [len(coef_ids)]
                )
                self.coefficient_fixed_summary = self._scatter_along_axis(
                    fixef_ix,
                    self.coefficient_fixed_base_summary,
                    [len(coef_ids)]
                )

                self._regularize(self.coefficient_fixed, type='coefficient', var_name='coefficient')

                for i in range(len(self.coef_names)):
                    tf.summary.scalar(
                        'coefficient' + '/%s' % self.coef_names[i],
                        self.coefficient_fixed_summary[i],
                        collections=['params']
                    )

                self.intercept = self.intercept_fixed
                self.intercept_summary = self.intercept_fixed_summary
                self.coefficient = self.coefficient_fixed
                self.coefficient_summary = self.coefficient_fixed_summary

                self.intercept_random = {}
                self.intercept_random_summary = {}
                self.intercept_random_means = {}
                self.coefficient_random = {}
                self.coefficient_random_summary = {}
                self.coefficient_random_means = {}

                self.coefficient = tf.expand_dims(self.coefficient, 0)

                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                    if self.has_intercept[gf]:
                        intercept_random = self.intercept_random_base[gf]
                        intercept_random_summary = self.intercept_random_base_summary[gf]

                        intercept_random_means = tf.reduce_mean(intercept_random, axis=0, keepdims=True)
                        intercept_random_summary_means = tf.reduce_mean(intercept_random_summary, axis=0, keepdims=True)

                        intercept_random -= intercept_random_means
                        intercept_random_summary -= intercept_random_summary_means

                        self._regularize(intercept_random, type='ranef', var_name='intercept_by_%s' % gf)

                        intercept_random = self._scatter_along_axis(
                            levels_ix,
                            intercept_random,
                            [self.rangf_n_levels[i]]
                        )
                        intercept_random_summary = self._scatter_along_axis(
                            levels_ix,
                            intercept_random_summary,
                            [self.rangf_n_levels[i]]
                        )

                        self.intercept_random[gf] = intercept_random
                        self.intercept_random_summary[gf] = intercept_random_summary
                        self.intercept_random_means[gf] = tf.reduce_mean(intercept_random_summary, axis=0)

                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/intercept' % gf),
                                intercept_random_summary,
                                collections=['random']
                            )

                    coefs = self.coef_by_rangf.get(gf, [])
                    if len(coefs) > 0:
                        coef_ix = names2ix(coefs, self.coef_names)

                        coefficient_random = self.coefficient_random_base[gf]
                        coefficient_random_summary = self.coefficient_random_base_summary[gf]

                        coefficient_random_means = tf.reduce_mean(coefficient_random, axis=0, keepdims=True)
                        coefficient_random_summary_means = tf.reduce_mean(coefficient_random_summary, axis=0, keepdims=True)

                        coefficient_random -= coefficient_random_means
                        coefficient_random_summary -= coefficient_random_summary_means
                        self._regularize(coefficient_random, type='ranef', var_name='coefficient_by_%s' % gf)

                        coefficient_random = self._scatter_along_axis(
                            coef_ix,
                            self._scatter_along_axis(
                                levels_ix,
                                coefficient_random,
                                [self.rangf_n_levels[i], len(coefs)]
                            ),
                            [self.rangf_n_levels[i], len(self.coef_names)],
                            axis=1
                        )
                        coefficient_random_summary = self._scatter_along_axis(
                            coef_ix,
                            self._scatter_along_axis(
                                levels_ix,
                                coefficient_random_summary,
                                [self.rangf_n_levels[i], len(coefs)]
                            ),
                            [self.rangf_n_levels[i], len(self.coef_names)],
                            axis=1
                        )

                        self.coefficient_random[gf] = coefficient_random
                        self.coefficient_random_summary[gf] = coefficient_random_summary
                        self.coefficient_random_means[gf] = tf.reduce_mean(coefficient_random_summary, axis=0)

                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(coefs)):
                                coef_name = coefs[j]
                                ix = coef_ix[j]
                                tf.summary.histogram(
                                    sn('by_%s/coefficient/%s' % (gf, coef_name)),
                                    coefficient_random_summary[:, ix],
                                    collections=['random']
                                )

    def _initialize_irf_lambdas(self):

        with self.sess.as_default():
            with self.sess.graph.as_default():
                def exponential(params):
                    pdf = tf.contrib.distributions.Exponential(rate=params[:,0:1]).prob
                    return lambda x: pdf(x + self.epsilon)

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['ExpRateGT1'] = exponential

                def gamma(params):
                    pdf = tf.contrib.distributions.Gamma(
                        concentration=params[:,0:1],
                        rate=params[:,1:2],
                        validate_args=self.validate_irf_args
                    ).prob
                    return lambda x: pdf(x + self.epsilon)

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['SteepGamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['GammaKgt1'] = gamma
                self.irf_lambdas['HRFSingleGamma'] = gamma

                def shifted_gamma(params):
                    pdf = tf.contrib.distributions.Gamma(
                        concentration=params[:,0:1],
                        rate=params[:,1:2],
                        validate_args=self.validate_irf_args
                    ).prob
                    return lambda x: pdf(x - params[:,2:3] + self.epsilon)

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma
                self.irf_lambdas['ShiftedGammaShapeGT1'] = shifted_gamma
                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma

                def normal(params):
                    pdf = tf.contrib.distributions.Normal(loc=params[:,0:1], scale=params[:,1:2]).prob
                    return lambda x: pdf(x)

                self.irf_lambdas['Normal'] = normal

                def skew_normal(params):
                    mu = params[:,0:1]
                    sigma = params[:,1:2]
                    alpha = params[:,2:3]
                    stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
                    stdnorm_pdf = stdnorm.prob
                    stdnorm_cdf = stdnorm.cdf
                    return lambda x: 2 / sigma * stdnorm_pdf((x - mu) / sigma) * stdnorm_cdf(alpha * (x - mu) / sigma)

                self.irf_lambdas['SkewNormal'] = skew_normal

                def emg(params):
                    mu = params[:,0:1]
                    sigma = params[:,1:2]
                    L = params[:,2:3]
                    return lambda x: L / 2 * tf.exp(0.5 * L * (2. * mu + L * sigma ** 2. - 2. * x)) * tf.erfc(
                        (mu + L * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))

                self.irf_lambdas['EMG'] = emg

                def beta_prime(params):
                    alpha = params[:,1:2]
                    beta = params[:,2:3]
                    return lambda x: (x + self.epsilon) ** (alpha - 1.) * (1. + (x + self.epsilon)) ** (-alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.stack([alpha, beta], axis=0))))

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(params):
                    alpha = params[:,0:1]
                    beta = params[:,1:2]
                    delta = params[:,2:3]
                    return lambda x: (x - delta + self.epsilon) ** (alpha - 1) * (1 + (x - delta + self.epsilon)) ** (
                    -alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.stack([alpha, beta], axis=0))))

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

                def double_gamma(params):
                    alpha_main = params[:, 0:1]
                    beta = params[:, 1:2]
                    alpha_undershoot_offset = params[:, 2:3]
                    c = params[:, 3:4]

                    pdf_main = tf.contrib.distributions.Gamma(
                        concentration=alpha_main,
                        rate=beta,
                        validate_args=self.validate_irf_args
                    ).prob
                    pdf_undershoot = tf.contrib.distributions.Gamma(
                        concentration=alpha_main + alpha_undershoot_offset,
                        rate=beta,
                        validate_args=self.validate_irf_args
                    ).prob

                    return lambda x: pdf_main(x + self.epsilon) - c * pdf_undershoot(x + self.epsilon)

                self.irf_lambdas['HRFDoubleGamma'] = double_gamma

                def double_gamma_unconstrained(params):
                    alpha_main = params[:, 0:1]
                    beta_main = params[:, 1:2]
                    alpha_undershoot = params[:, 2:3]
                    beta_undershoot = params[:, 3:4]
                    c = params[:, 4:5]

                    pdf_main = tf.contrib.distributions.Gamma(
                        concentration=alpha_main,
                        rate=beta_main,
                        validate_args=self.validate_irf_args
                    ).prob
                    pdf_undershoot = tf.contrib.distributions.Gamma(
                        concentration=alpha_undershoot,
                        rate=beta_undershoot,
                        validate_args=self.validate_irf_args
                    ).prob

                    return lambda x: pdf_main(x + self.epsilon) - c * pdf_undershoot(x + self.epsilon)

                self.irf_lambdas['HRFDoubleGammaUnconstrained'] = double_gamma_unconstrained

    def _initialize_spline(self, order, bases, instantaneous=True, roughness_penalty=0.):
        def spline(params):
            target_shape = bases * 2 - 2 - (1-int(instantaneous))
            assert params.shape[1] == target_shape, 'Incorrect number of parameters for spline with bases "%d". Should be %s, got %s.' %(bases, target_shape)

            # Build knot locations
            c = params[:, 0:bases-1]
            c_endpoint_shape = [tf.shape(params)[0], 1, tf.shape(params)[2]]
            zero = tf.zeros(c_endpoint_shape, dtype=self.FLOAT_TF)
            c = tf.cumsum(tf.abs(tf.concat([zero, c], axis=1)), axis=1)
            c = tf.unstack(c, axis=2)

            # Build values at knots
            y = [params[:, bases-1:], zero]
            if not instantaneous:
                y = [zero] + y
            y = tf.concat(y, axis=1)
            y = tf.unstack(y, axis=2)

            assert len(c) == len(y), 'c and y coordinates of spline unpacked into lists of different lengths (%s and %s, respectively)' %(len(c), len(y))

            def apply_spline(x):
                splines = []

                if len(x.shape) == 1:
                    x = x[None, :, None]
                elif len(x.shape) == 2:
                    x = x[None, ...]
                if len(x.shape) != 3:
                    raise ValueError('Query to spline IRF must be exactly rank 3')

                for i in range(len(c)):
                    if order == 1:
                        rise = y[i][:,1:] - y[i][:,:-1]
                        run = c[i][:,1:] - c[i][:,:-1]
                        a_ = (rise / run)
                        self.rise_ = rise
                        self.run_ = run
                        self.a_ = a_
                        c_ = c[i][:,:-1]

                        out = lambda x: tf.reduce_sum(tf.cast(x >= c_, dtype=self.FLOAT_TF) * (x - c_) * a_, axis=2, keepdims=True)

                    else:
                        if c[i].shape[0] == 1:
                            c_ = tf.tile(c[i][..., None], [tf.shape(x)[0], 1, 1])
                        else:
                            c_ = c[i][..., None]
                        if y[i].shape[0] == 1:
                            y_ = tf.tile(y[i][..., None], [tf.shape(x)[0], 1, 1])
                        else:
                            y_ = y[i][..., None]

                        out = lambda x: interpolate_spline(
                            c_,
                            y_,
                            x,
                            order,
                            regularization_weight=roughness_penalty
                        )

                    splines.append(out)

                out = tf.concat([s(x) for s in splines], axis=2)
                # out = tf.where(x <= self.max_tdelta, out, tf.zeros_like(out))

                return out

            return apply_spline

        return spline

    def _get_irf_lambda(self, family):
        if family in self.irf_lambdas:
            return self.irf_lambdas[family]
        elif Formula.is_spline(family):
            order = Formula.order(family)
            bases = Formula.bases(family)
            instantaneous = Formula.instantaneous(family)
            roughness_penalty = Formula.roughness_penalty(family)
            return self._initialize_spline(
                order,
                bases,
                instantaneous=instantaneous,
                roughness_penalty=roughness_penalty
            )
        else:
            raise ValueError('No IRF lamdba found for family "%s"' % family)

    def _initialize_irf_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for family in self.atomic_irf_names_by_family:
                    if family == 'DiracDelta':
                        continue

                    irf_ids = self.atomic_irf_names_by_family[family]

                    params = []
                    params_summary = []
                    params_fixed = []
                    params_fixed_summary = []
                    params_random = {}
                    params_random_summary = {}

                    for param_name in Formula.irf_params(family):
                        param_vals = self._initialize_irf_param(param_name, family)
                        params.append(param_vals[0])

                        params_summary.append(param_vals[1])
                        if param_vals[2] is not None:
                            params_fixed.append(param_vals[2])
                        if param_vals[3] is not None:
                            params_fixed_summary.append(param_vals[3])

                        if param_vals[4] is not None and param_vals[5] is not None:
                            assert(set(param_vals[4].keys()) == set(param_vals[5].keys()))

                            for gf in param_vals[4].keys():
                                if gf not in params_random:
                                    params_random[gf] = []
                                params_random[gf].append(param_vals[4][gf])

                                if gf not in params_random_summary:
                                    params_random_summary[gf] = []
                                params_random_summary[gf].append(param_vals[5][gf])

                    has_random_irf = False
                    for param in params:
                        if not param.shape.is_fully_defined():
                            has_random_irf = True
                            break
                    if has_random_irf:
                        for i in range(len(params)):
                            param = params[i]
                            if param.shape.is_fully_defined():
                                assert param.shape[0] == 1, 'Parameter with shape %s not broadcastable to batch length' %param.shape
                                params[i] =  tf.tile(param, [tf.shape(self.time_y)[0], 1])

                    params = tf.stack(params, axis=1)
                    params_summary = tf.stack(params_summary, axis=1)
                    params_fixed = tf.stack(params_fixed, axis=1)
                    params_fixed_summary = tf.stack(params_fixed_summary, axis=1)
                    for gf in params_random:
                        params_random[gf] = tf.stack(params_random[gf], axis=1)

                    for i in range(len(irf_ids)):
                        id = irf_ids[i]
                        ix = names2ix(id, self.atomic_irf_names_by_family[family])
                        assert id not in self.irf_params, 'Duplicate IRF node name already in self.irf_params'
                        self.irf_params[id] = tf.gather(params, ix, axis=2)
                        self.irf_params_summary[id] = tf.gather(params_summary, ix, axis=2)
                        trainable_param_ix = names2ix(self.atomic_irf_param_trainable_by_family[family][id], Formula.irf_params(family))
                        if len(trainable_param_ix) > 0:
                            self.irf_params_fixed[id] = tf.gather(tf.gather(params_fixed, ix, axis=2), trainable_param_ix, axis=1)
                            self.irf_params_fixed_summary[id] = tf.gather(tf.gather(params_fixed_summary, ix, axis=2), trainable_param_ix, axis=1)
                            for gf in params_random:
                                if gf not in self.irf_params_random:
                                    self.irf_params_random[gf] = {}
                                self.irf_params_random[gf][id] = tf.gather(tf.gather(params_random[gf], ix, axis=2), trainable_param_ix, axis=1)
                                if gf not in self.irf_params_random_summary:
                                    self.irf_params_random_summary[gf] = {}
                                self.irf_params_random_summary[gf][id] = tf.gather(tf.gather(params_random_summary[gf], ix, axis=2), trainable_param_ix, axis=1)

    def _initialize_irf_param(self, param_name, family):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                irf_ids = self.atomic_irf_names_by_family[family]
                param_mean = self.irf_params_means[family][param_name]
                param_lb = self.irf_params_lb[family][param_name]
                param_ub = self.irf_params_ub[family][param_name]
                param_mean_unconstrained = self.irf_params_means_unconstrained[family][param_name]
                trainable = self.atomic_irf_param_trainable_by_family[family]

                trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                    param_name,
                    irf_ids,
                    trainable=trainable
                )

                dim = len(trainable_ix)
                trainable_means = tf.expand_dims(tf.gather(param_mean_unconstrained, trainable_ix), 0)

                # Initialize trainable IRF parameters as trainable variables
                if dim > 0:
                    param_fixed = self.irf_params_fixed_base[family][param_name]
                    param_fixed_summary = self.irf_params_fixed_base_summary[family][param_name]

                    if param_lb is not None and param_ub is None:
                        param_fixed = param_lb + self.epsilon + tf.nn.softplus(param_fixed)
                        param_fixed_summary = param_lb + self.epsilon + tf.nn.softplus(param_fixed_summary)
                    elif param_lb is None and param_ub is not None:
                        param_fixed = param_ub - self.epsilon - tf.nn.softplus(param_fixed)
                        param_fixed_summary = param_ub - self.epsilon - tf.nn.softplus(param_fixed_summary)
                    elif param_lb is not None and param_ub is not None:
                        param_fixed = self._softplus_sigmoid(param_fixed, a=param_lb, b=param_ub)
                        param_fixed_summary = self._softplus_sigmoid(param_fixed_summary, a=param_lb, b=param_ub)

                    self._regularize(param_fixed, trainable_means, type='irf', var_name=param_name)
                else:
                    param_fixed = param_fixed_summary = None

                for i in range(dim):
                    tf.summary.scalar(
                        sn('%s/%s' % (param_name, irf_ids[i])),
                        param_fixed_summary[0, i],
                        collections=['params']
                    )

                # Initialize untrainable IRF parameters as constants
                param_untrainable = tf.expand_dims(
                    tf.gather(
                        tf.ones((len(irf_ids),), dtype=self.FLOAT_TF) * param_mean,
                        untrainable_ix)
                    ,
                    axis=0
                )

                param = param_fixed
                param_summary = param_fixed_summary

                # Process any random IRF parameters
                irf_by_rangf = {}
                for id in irf_ids:
                    for gf in self.irf_by_rangf:
                        if id in self.irf_by_rangf[gf]:
                            if gf not in irf_by_rangf:
                                irf_by_rangf[gf] = []
                            irf_by_rangf[gf].append(id)

                param_random_by_rangf = {}
                param_random_summary_by_rangf = {}

                if len(irf_by_rangf) > 0:
                    for i, gf in enumerate(irf_by_rangf):
                        irf_ids_ran = [x for x in irf_by_rangf[gf] if param_name in trainable[x]]
                        if len(irf_ids_ran):
                            irfs_ix = names2ix(irf_by_rangf[gf], irf_ids)
                            levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                            param_random = self._center_and_constrain(
                                self.irf_params_random_base[gf][family][param_name],
                                tf.gather(param, irfs_ix, axis=1),
                                lb=param_lb,
                                ub=param_ub
                            )
                            param_random_summary = self._center_and_constrain(
                                self.irf_params_random_base_summary[gf][family][param_name],
                                tf.gather(param_summary, irfs_ix, axis=1),
                                lb=param_lb,
                                ub=param_ub
                            )

                            self._regularize(param_random, type='ranef', var_name='%s_by_%s' % (param_name, gf))

                            param_random = self._scatter_along_axis(
                                irfs_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    param_random,
                                    [self.rangf_n_levels[i], len(irfs_ix)]
                                ),
                                [self.rangf_n_levels[i], len(irfs_ix)],
                                axis=1
                            )
                            param_random_summary = self._scatter_along_axis(
                                irfs_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    param_random_summary,
                                    [self.rangf_n_levels[i], len(irfs_ix)]
                                ),
                                [self.rangf_n_levels[i], len(irfs_ix)],
                                axis=1
                            )

                            param_random_by_rangf[gf] = param_random
                            param_random_summary_by_rangf[gf] = param_random_summary
                            if gf not in self.irf_params_random_means:
                                self.irf_params_random_means[gf] = {}
                            if family not in self.irf_params_random_means[gf]:
                                self.irf_params_random_means[gf][family] = {}
                            self.irf_params_random_means[gf][family][param_name] = tf.reduce_mean(param_random_summary, axis=0)

                            param += tf.gather(param_random, self.gf_y[:, i], axis=0)

                            if self.log_random:
                                for j in range(len(irf_by_rangf[gf])):
                                    irf_name = irf_by_rangf[gf][j]
                                    ix = irfs_ix[j]
                                    tf.summary.histogram(
                                        'by_%s/%s/%s' % (gf, param_name, irf_name),
                                        param_random_summary[:, ix],
                                        collections=['random']
                                    )
                        else:
                            param_random = param_random_summary = None

                # Combine trainable and untrainable parameters
                if len(untrainable_ix) > 0:
                    if len(trainable_ix) > 0:
                        param = tf.concat([param, param_untrainable], axis=1)
                        param_summary = tf.concat([param_summary, param_untrainable], axis=1)
                    else:
                        param = param_untrainable
                        param_summary = param_untrainable

                    param = tf.gather(param, np.concatenate([trainable_ix, untrainable_ix]), axis=1)
                param_summary = tf.gather(param_summary, np.concatenate([trainable_ix, untrainable_ix]), axis=1)

                return param, param_summary, param_fixed, param_fixed_summary, param_random_by_rangf, param_random_summary_by_rangf

    def _initialize_base_irf_param(self, param_name, family, lb=None, ub=None, default=0.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                irf_ids = self.atomic_irf_names_by_family[family]
                param_init = self.atomic_irf_param_init_by_family[family]
                param_trainable = self.atomic_irf_param_trainable_by_family[family]

                # Process and store initial/prior means
                param_mean = self._get_mean_init_vector(irf_ids, param_name, param_init, default=default)
                param_mean_unconstrained, param_lb, param_ub = self._process_mean(param_mean, lb=lb, ub=ub)

                if family not in self.irf_params_means:
                    self.irf_params_means[family] = {}
                self.irf_params_means[family][param_name] = param_mean

                if family not in self.irf_params_means_unconstrained:
                    self.irf_params_means_unconstrained[family] = {}
                self.irf_params_means_unconstrained[family][param_name] = param_mean_unconstrained

                if family not in self.irf_params_lb:
                    self.irf_params_lb[family] = {}
                self.irf_params_lb[family][param_name] = param_lb

                if family not in self.irf_params_ub:
                    self.irf_params_ub[family] = {}
                self.irf_params_ub[family][param_name] = param_ub


                # Select out irf IDs for which this param is trainable
                trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                    param_name,
                    irf_ids,
                    trainable=param_trainable
                )
                trainable_means = tf.expand_dims(tf.gather(param_mean_unconstrained, trainable_ix), axis=0)


                if not self.covarying_fixef:
                    # Initialize and store fixed params on the unconstrained space
                    if len(trainable_ix) > 0:
                        param_fixed_base, param_fixed_base_summary = self.initialize_irf_param_unconstrained(
                            param_name,
                            [x for x in irf_ids if param_name in param_trainable[x]],
                            mean=trainable_means
                        )

                        if family not in self.irf_params_fixed_base:
                            self.irf_params_fixed_base[family] = {}
                        self.irf_params_fixed_base[family][param_name] = param_fixed_base

                        if family not in self.irf_params_fixed_base_summary:
                            self.irf_params_fixed_base_summary[family] = {}
                        self.irf_params_fixed_base_summary[family][param_name] = param_fixed_base_summary


                if not self.covarying_ranef:
                    # Initialize and store random params on the unconstrained space
                    irf_by_rangf = {}
                    for id in irf_ids:
                        for gf in self.irf_by_rangf:
                            if id in self.irf_by_rangf[gf]:
                                if gf not in irf_by_rangf:
                                    irf_by_rangf[gf] = []
                                irf_by_rangf[gf].append(id)

                    for gf in irf_by_rangf:
                        irf_ids_ran = [x for x in irf_by_rangf[gf] if param_name in param_trainable[x]]
                        if len(irf_ids_ran) > 0:
                            param_random_base, param_random_base_summary = self.initialize_irf_param_unconstrained(
                                param_name,
                                [x for x in irf_by_rangf[gf] if param_name in param_trainable[x]],
                                mean=0.,
                                ran_gf=gf
                            )

                            if gf not in self.irf_params_random_base:
                                self.irf_params_random_base[gf] = {}
                            if family not in self.irf_params_random_base[gf]:
                                self.irf_params_random_base[gf][family] = {}
                            self.irf_params_random_base[gf][family][param_name] = param_random_base

                            if gf not in self.irf_params_random_base_summary:
                                self.irf_params_random_base_summary[gf] = {}
                            if family not in self.irf_params_random_base_summary[gf]:
                                self.irf_params_random_base_summary[gf][family] = {}
                            self.irf_params_random_base_summary[gf][family][param_name] = param_random_base_summary

    def _initialize_joint_distributions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                if self.covarying_fixef:
                    joint_fixed_means = []
                    joint_fixed_sds = []
                    joint_fixed_ix = {}

                    i = 0

                    if self.has_intercept[None]:
                        joint_fixed_means.append(tf.expand_dims(self.intercept_init_tf, axis=0))
                        joint_fixed_sds.append(tf.expand_dims(self.intercept_joint_sd, axis=0))
                        joint_fixed_ix['intercept'] = (i, i + 1)
                        i += 1

                    coef_ids = self.fixed_coef_names
                    joint_fixed_means.append(tf.zeros((len(coef_ids), ), dtype=self.FLOAT_TF))
                    joint_fixed_sds.append(tf.ones((len(coef_ids), ), dtype=self.FLOAT_TF) * self.coef_joint_sd)
                    joint_fixed_ix['coefficient'] = (i, i + len(coef_ids))
                    i += len(coef_ids)

                    for family in self.atomic_irf_names_by_family:
                        if family not in joint_fixed_ix:
                            joint_fixed_ix[family] = {}
                        for param_name in Formula.irf_params(family):
                            irf_ids = self.atomic_irf_names_by_family[family]
                            param_trainable = self.atomic_irf_param_trainable_by_family[family]

                            trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                                param_name,
                                irf_ids,
                                trainable=param_trainable
                            )

                            if len(trainable_ix) > 0:
                                joint_fixed_means.append(
                                    tf.gather(
                                        self.irf_params_means_unconstrained[family][param_name],
                                        trainable_ix
                                    )
                                )
                                joint_fixed_sds.append(
                                    tf.ones((len(trainable_ix),), dtype=self.FLOAT_TF) * self.irf_param_joint_sd
                                )
                                joint_fixed_ix[family][param_name] = (i,  i + len(trainable_ix))
                                i += len(trainable_ix)
                            else:
                                joint_fixed_ix[family][param_name] = None

                    joint_fixed_means = tf.concat(joint_fixed_means, axis=0)
                    joint_fixed_sds = tf.concat(joint_fixed_sds, axis=0)

                    self.joint_fixed, self.joint_fixed_summary = self.initialize_joint_distribution(
                        joint_fixed_means,
                        joint_fixed_sds,
                    )

                    self.joint_fixed_ix = joint_fixed_ix

                if self.covarying_ranef:
                    joint_random = {}
                    joint_random_summary = {}
                    joint_random_means = {}
                    joint_random_sds = {}
                    joint_random_ix = {}

                    for i, gf in enumerate(self.rangf):
                        joint_random_means[gf] = []
                        joint_random_sds[gf] = []
                        joint_random_ix[gf] = {}
                        n_levels = self.rangf_n_levels[i] - 1

                        i = 0

                        if self.has_intercept[gf]:
                            joint_random_means[gf].append(tf.zeros([n_levels,], dtype=self.FLOAT_TF))
                            joint_random_sds[gf].append(tf.ones([n_levels,], dtype=self.FLOAT_TF) * self.intercept_joint_sd)
                            joint_random_ix[gf]['intercept'] = (i, i + n_levels)
                            i += n_levels

                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids) > 0:
                            joint_random_means[gf].append(tf.zeros([n_levels * len(coef_ids)], dtype=self.FLOAT_TF))
                            joint_random_sds[gf].append(tf.ones([n_levels * len(coef_ids)], dtype=self.FLOAT_TF) * self.coef_joint_sd)
                            joint_random_ix[gf]['coefficient'] = (i, i + n_levels * len(coef_ids))
                            i += n_levels * len(coef_ids)

                        for family in self.atomic_irf_names_by_family:
                            for param_name in Formula.irf_params(family):
                                irf_ids_src = self.atomic_irf_names_by_family[family]

                                irf_ids = []
                                for id in irf_ids_src:
                                    if gf in self.irf_by_rangf and id in self.irf_by_rangf[gf]:
                                        irf_ids.append(id)

                                if len(irf_ids) > 0:
                                    if family not in joint_random_ix[gf]:
                                        joint_random_ix[gf][family] = {}
                                    param_trainable = self.atomic_irf_param_trainable_by_family[family]

                                    trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                                        param_name,
                                        irf_ids,
                                        trainable=param_trainable
                                    )

                                    if len(trainable_ix) > 0:
                                        joint_random_means[gf].append(
                                            tf.zeros([n_levels * len(trainable_ix)], dtype=self.FLOAT_TF)
                                        )
                                        joint_random_sds[gf].append(
                                            tf.ones([n_levels * len(trainable_ix)], dtype=self.FLOAT_TF) * self.irf_param_joint_sd
                                        )
                                        joint_random_ix[gf][family][param_name] = (i, i + n_levels * len(trainable_ix))
                                        i += n_levels * len(trainable_ix)
                                    else:
                                        joint_random_ix[gf][family][param_name] = None

                        joint_random_means[gf] = tf.concat(joint_random_means[gf], axis=0)
                        joint_random_sds[gf] = tf.concat(joint_random_sds[gf], axis=0)

                        joint_random[gf], joint_random_summary[gf] = self.initialize_joint_distribution(
                            joint_random_means[gf],
                            joint_random_sds[gf],
                            ran_gf=gf
                        )

                    self.joint_random = joint_random
                    self.joint_random_summary = joint_random_summary
                    self.joint_random_ix = joint_random_ix

    def _initialize_joint_distribution_slices(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.covarying_fixef:
                    if self.has_intercept[None]:
                        s, e = self.joint_fixed_ix['intercept']
                        self.intercept_fixed_base = self.joint_fixed[s]
                        self.intercept_fixed_base_summary = self.joint_fixed_summary[s]

                    s, e = self.joint_fixed_ix['coefficient']
                    self.coefficient_fixed_base = self.joint_fixed[s:e]
                    self.coefficient_fixed_base_summary = self.joint_fixed_summary[s:e]

                    for family in self.atomic_irf_names_by_family:
                        if family not in self.irf_params_fixed_base:
                            self.irf_params_fixed_base[family] = {}
                        if family not in self.irf_params_fixed_base_summary:
                            self.irf_params_fixed_base_summary[family] = {}
                        for param_name in Formula.irf_params(family):
                            bounds = self.joint_fixed_ix[family][param_name]
                            if bounds is not None:
                                s, e = bounds
                                self.irf_params_fixed_base[family][param_name] = tf.expand_dims(self.joint_fixed[s:e], axis=0)
                                self.irf_params_fixed_base_summary[family][param_name] = tf.expand_dims(self.joint_fixed_summary[s:e], axis=0)

                if self.covarying_ranef:
                    for i, gf in enumerate(self.rangf):
                        rangf_n_levels = self.rangf_n_levels[i] - 1

                        if self.has_intercept[gf]:
                            if gf not in self.intercept_random_base:
                                self.intercept_random_base[gf] = {}
                            if gf not in self.intercept_random_base_summary:
                                self.intercept_random_base_summary[gf] = {}
                            s, e = self.joint_random_ix[gf]['intercept']
                            self.intercept_random_base[gf] = self.joint_random[gf][s:e]
                            self.intercept_random_base_summary[gf] = self.joint_random_summary[gf][s:e]

                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids) > 0:
                            if gf not in self.coefficient_random_base:
                                self.coefficient_random_base[gf] = {}
                            if gf not in self.coefficient_random_base_summary:
                                self.coefficient_random_base_summary[gf] = {}
                            s, e = self.joint_random_ix[gf]['coefficient']
                            self.coefficient_random_base[gf] = tf.reshape(
                                self.joint_random[gf][s:e],
                                [rangf_n_levels, len(coef_ids)]
                            )
                            self.coefficient_random_base_summary[gf] = tf.reshape(
                                self.joint_random_summary[gf][s:e],
                                [rangf_n_levels, len(coef_ids)]
                            )

                        if gf not in self.irf_params_random_base:
                            self.irf_params_random_base[gf] = {}
                        if gf not in self.irf_params_random_base_summary:
                            self.irf_params_random_base_summary[gf] = {}

                        for family in self.atomic_irf_names_by_family:
                            irf_ids_src = self.atomic_irf_names_by_family[family]

                            irf_ids = []
                            for id in irf_ids_src:
                                if gf in self.irf_by_rangf and id in self.irf_by_rangf[gf]:
                                    irf_ids.append(id)

                            if len(irf_ids) > 0:
                                if family not in self.irf_params_random_base[gf]:
                                    self.irf_params_random_base[gf][family] = {}
                                if family not in self.irf_params_random_base_summary[gf]:
                                    self.irf_params_random_base_summary[gf][family] = {}
                                for param_name in Formula.irf_params(family):
                                    if gf in self.irf_by_rangf:
                                        if param_name not in self.irf_params_random_base[gf][family]:
                                            self.irf_params_random_base[gf][family][param_name] = {}
                                        if param_name not in self.irf_params_random_base_summary[gf][family]:
                                            self.irf_params_random_base_summary[gf][family][param_name] = {}

                                        bounds = self.joint_random_ix[gf][family][param_name]
                                        if bounds is not None:
                                            s, e = bounds
                                            self.irf_params_random_base[gf][family][param_name] = tf.reshape(
                                                self.joint_random[gf][s:e],
                                                [rangf_n_levels, len(irf_ids)]
                                            )
                                            self.irf_params_random_base_summary[gf][family][param_name] = tf.reshape(
                                                self.joint_random_summary[gf][s:e],
                                                [rangf_n_levels, len(irf_ids)]
                                            )

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
                for coef_name in self.fixed_coef_names:
                    coef_name_str = coef_name.split('-')
                    if len(coef_name_str) > 1:
                        coef_name_str = ' '.join(coef_name_str[:-1])
                    else:
                        coef_name_str = coef_name_str[0]
                    coef_name_str = sn('coefficient_' + coef_name_str)
                    parameter_table_fixed_keys.append(coef_name_str)
                parameter_table_fixed_values.append(
                    tf.gather(self.coefficient_fixed, names2ix(self.fixed_coef_names, self.coef_names))
                )
                for irf_id in self.irf_params_fixed:
                    family = self.atomic_irf_family_by_name[irf_id]
                    for param in self.atomic_irf_param_trainable_by_family[family][irf_id]:
                        param_ix = names2ix(param, Formula.irf_params(family))
                        irf_id_str = irf_id.split('-')
                        if len(irf_id_str) > 1:
                            irf_id_str = ' '.join(irf_id_str[:-1])
                        else:
                            irf_id_str = irf_id_str[0]
                        irf_id_str = sn(irf_id_str)
                        parameter_table_fixed_keys.append(param + '_' + irf_id_str)
                        parameter_table_fixed_values.append(
                            tf.squeeze(
                                tf.gather(self.irf_params_fixed[irf_id], param_ix, axis=1),
                                axis=(0, 2)
                            )
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
                        if gf in self.coefficient_random:
                            coef_names = self.coef_by_rangf.get(gf, [])
                            for coef_name in coef_names:
                                coef_ix = names2ix(coef_name, self.coef_names)
                                coef_name_str = coef_name.split('-')
                                if len(coef_name_str) > 1:
                                    coef_name_str = ' '.join(coef_name_str[:-1])
                                else:
                                    coef_name_str = coef_name_str[0]
                                coef_name_str = sn('coefficient_' + coef_name_str)
                                for level in levels:
                                    parameter_table_random_keys.append(coef_name_str)
                                    parameter_table_random_rangf.append(gf)
                                    parameter_table_random_rangf_levels.append(level)
                                parameter_table_random_values.append(
                                    tf.squeeze(
                                        tf.gather(
                                            tf.gather(self.coefficient_random[gf], coef_ix, axis=1),
                                            levels_ix
                                        )
                                    )
                                )
                        if gf in self.irf_params_random:
                            for irf_id in self.irf_params_random[gf]:
                                family = self.atomic_irf_family_by_name[irf_id]
                                for param in self.atomic_irf_param_trainable_by_family[family][irf_id]:
                                    param_ix = names2ix(param, Formula.irf_params(family))
                                    irf_id_str = irf_id.split('-')
                                    if len(irf_id_str) > 1:
                                        irf_id_str = ' '.join(irf_id_str[:-1])
                                    else:
                                        irf_id_str = irf_id_str[0]
                                    irf_id_str = sn(irf_id_str)
                                    for level in levels:
                                        parameter_table_random_keys.append(param + '_' + irf_id_str)
                                        parameter_table_random_rangf.append(gf)
                                        parameter_table_random_rangf_levels.append(level)
                                    parameter_table_random_values.append(
                                        tf.squeeze(
                                            tf.gather(
                                                tf.gather(self.irf_params_random[gf][irf_id], param_ix, axis=1),
                                                levels_ix,
                                            )
                                        )
                                    )

                    self.parameter_table_random_keys = parameter_table_random_keys
                    self.parameter_table_random_rangf = parameter_table_random_rangf
                    self.parameter_table_random_rangf_levels = parameter_table_random_rangf_levels
                    self.parameter_table_random_values = tf.concat(parameter_table_random_values, 0)

    def _initialize_random_mean_vector(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.rangf) > 0:
                    means = []
                    for gf in self.intercept_random_means:
                        means.append(tf.expand_dims(self.intercept_random_means[gf], 0))
                    for gf in self.coefficient_random_means:
                        means.append(self.coefficient_random_means[gf])
                    for gf in self.irf_params_random_means:
                        for family in self.irf_params_random_means[gf]:
                            for param_name in self.irf_params_random_means[gf][family]:
                                means.append(self.irf_params_random_means[gf][family][param_name])

                    self.random_means = tf.concat(means, axis=0)

    def _initialize_irfs(self, t):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if t.family is None:
                    self.irf[t.name()] = []
                elif t.family == 'Terminal':
                    coef_name = self.node_table[t.name()].coef_id()
                    coef_ix = names2ix(coef_name, self.coef_names)

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = self.irf[t.p.name()][:]

                    assert not t.name() in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'scaled': self.irf_mc[t.p.name()]['atomic']['unscaled'] * tf.gather(self.coefficient_fixed, coef_ix),
                            'unscaled': self.irf_mc[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_mc[t.p.name()]['composite']['unscaled'] * tf.gather(self.coefficient_fixed, coef_ix),
                            'unscaled': self.irf_mc[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.irf_plot[t.p.name()]['atomic']['unscaled'] * tf.gather(self.coefficient_summary, coef_ix),
                            'unscaled': self.irf_plot[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_plot[t.p.name()]['composite']['unscaled'] * tf.gather(self.coefficient_summary, coef_ix),
                            'unscaled': self.irf_plot[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.irf_integral_tensors, 'Duplicate IRF node name already in self.mc_integrals'
                    if t.p.family == 'DiracDelta':
                        self.irf_integral_tensors[t.name()] = tf.gather(self.coefficient_fixed, coef_ix)
                    else:
                        self.irf_integral_tensors[t.name()] = self._reduce_interpolated_sum(
                            self.irf_mc[t.name()]['composite']['scaled'],
                            self.support[:,0],
                            axis=0
                        )

                elif t.family == 'DiracDelta':
                    assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in DTSR formula strings'
                    assert not t.impulse == 'rate', '"rate" is a reserved keyword in DTSR formula strings and cannot be used under DiracDelta'


                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = self.irf[t.p.name()][:]

                    assert not t.name() in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        },
                        'composite' : {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        },
                        'composite' : {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        }
                    }
                else:
                    params = self.irf_params[t.irf_id()]
                    params_summary = self.irf_params_summary[t.irf_id()]

                    atomic_irf = self._new_irf(self._get_irf_lambda(t.family), params)
                    atomic_irf_plot = self._new_irf(self._get_irf_lambda(t.family), params_summary)

                    if t.p.name() in self.irf:
                        irf = self.irf[t.p.name()][:] + [atomic_irf]
                        irf_plot = self.irf[t.p.name()][:] + [atomic_irf_plot]
                    else:
                        irf = [atomic_irf]
                        irf_plot = [atomic_irf_plot]

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = irf

                    atomic_irf_mc = atomic_irf(self.support)[0]
                    atomic_irf_plot = atomic_irf_plot(self.support)[0]

                    if len(irf) > 1:
                        composite_irf_mc = self._compose_irf(irf)(self.support[None, ...])[0]
                    else:
                        composite_irf_mc = atomic_irf_mc
                    if len(irf_plot) > 1:
                        composite_irf_plot = self._compose_irf(irf_plot)(self.support[None, ...])[0]
                    else:
                        composite_irf_plot = atomic_irf_plot

                    assert t.name() not in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'unscaled': atomic_irf_mc,
                            'scaled': atomic_irf_mc
                        },
                        'composite': {
                            'unscaled': composite_irf_mc,
                            'scaled': composite_irf_mc
                        }
                    }

                    assert t.name() not in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'unscaled': atomic_irf_plot,
                            'scaled': atomic_irf_plot
                        },
                        'composite': {
                            'unscaled': composite_irf_plot,
                            'scaled': composite_irf_plot
                        }
                    }

                for c in t.children:
                    self._initialize_irfs(c)

    def _initialize_backtransformed_irf_plot(self, t):
        if self.pc:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if t.name() in self.irf_plot:
                        src_irf_names = self.bw_pointers[t.name()]
                        t_impulse_names = t.impulse_names()
                        if t_impulse_names == ['rate'] and len(src_irf_names) == 1 and self.src_node_table[src_irf_names[0]].impulse_names() == ['rate']:
                            self.src_irf_plot[src_irf_names[0]] = self.irf_plot[t.name()]
                            self.src_irf_mc[src_irf_names[0]] = self.irf_mc[t.name()]
                            if t.name() in self.irf_integral_tensors:
                                self.src_irf_integral_tensors[src_irf_names[0]] = self.irf_integral_tensors[t.name()]
                        else:
                            for src_irf_name in src_irf_names:
                                src_irf = self.src_node_table[src_irf_name]
                                src_impulse_names = src_irf.impulse_names()
                                src_impulse_names_norate = list(filter(lambda x: x != 'rate', src_impulse_names))
                                src_ix = names2ix(src_impulse_names_norate, self.src_impulse_names_norate)
                                if len(src_ix) > 0:
                                    impulse_names = t.impulse_names()
                                    impulse_names_norate = list(filter(lambda x: x != 'rate', impulse_names))
                                    pc_ix = names2ix(impulse_names_norate, self.impulse_names_norate)
                                    if len(pc_ix) > 0:
                                        e = self.e
                                        e = tf.gather(e, src_ix, axis=0)
                                        e = tf.gather(e, pc_ix, axis=1)
                                        e = tf.reduce_sum(e, axis=1)

                                        if src_irf_name in self.src_irf_plot:
                                            self.src_irf_plot[src_irf_name]['atomic']['scaled'] += tf.reduce_sum(self.irf_plot[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['atomic']['unscaled'] += tf.reduce_sum(self.irf_plot[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['composite']['scaled'] += tf.reduce_sum(self.irf_plot[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['composite']['unscaled'] += tf.reduce_sum(self.irf_plot[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                        else:
                                            self.src_irf_plot[src_irf_name] = {
                                                'atomic': {
                                                    'scaled': tf.reduce_sum(self.irf_plot[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_plot[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                                },
                                                'composite': {
                                                    'scaled': tf.reduce_sum(self.irf_plot[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_plot[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                                }
                                            }
                                        if src_irf_name in self.src_irf_mc:
                                            self.src_irf_mc[src_irf_name]['atomic']['scaled'] += tf.reduce_sum(self.irf_mc[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['atomic']['unscaled'] += tf.reduce_sum(self.irf_mc[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['composite']['scaled'] += tf.reduce_sum(self.irf_mc[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['composite']['unscaled'] += tf.reduce_sum(self.irf_mc[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                        else:
                                            self.src_irf_mc[src_irf_name] = {
                                                'atomic': {
                                                    'scaled': tf.reduce_sum(self.irf_mc[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_mc[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                                },
                                                'composite': {
                                                    'scaled': tf.reduce_sum(self.irf_mc[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_mc[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                                }
                                            }
                                        if t.name() in self.irf_integral_tensors:
                                            if src_irf_name in self.src_irf_integral_tensors:
                                                self.src_irf_integral_tensors[src_irf_name] += tf.reduce_sum(self.irf_integral_tensors[t.name()] * e, axis=0, keep_dims=True)
                                            else:
                                                self.src_irf_integral_tensors[src_irf_name] = tf.reduce_sum(self.irf_integral_tensors[t.name()] * e, axis=0, keep_dims=True)

                    for c in t.children:
                        if c.name() in self.irf_plot:
                            self._initialize_backtransformed_irf_plot(c)

    def _initialize_impulses(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for name in self.terminal_names:
                    t = self.node_table[name]
                    impulse_name = t.impulse.name()
                    impulse_ix = names2ix(impulse_name, self.impulse_names)

                    if t.p.family == 'DiracDelta':
                        if self.pc:
                            if impulse_name == 'rate':
                                impulse = self.X_rate[:, -1, :]
                            else:
                                src_term_names = self.bw_pointers[t.name()]
                                src_impulse_names = set()
                                for x in src_term_names:
                                    src_impulse_names.add(self.src_node_table[x].impulse.name())
                                src_impulse_names = list(src_impulse_names)
                                src_impulse_ix = names2ix(src_impulse_names, self.src_impulse_names)
                                X = self.X[:, -1, :]
                                impulse = self._apply_pc(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)[:, -1, :]
                    else:
                        if self.pc:
                            if impulse_name == 'rate':
                                impulse = self.X_rate
                            else:
                                src_term_names = self.bw_pointers[t.name()]
                                src_impulse_names = set()
                                for x in src_term_names:
                                    src_impulse_names.add(self.src_node_table[x].impulse.name())
                                src_impulse_names = list(src_impulse_names)
                                src_impulse_ix = names2ix(src_impulse_names, self.src_impulse_names)
                                X = self.X
                                impulse = self._apply_pc(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)

                    self.irf_impulses[name] = impulse

    def _initialize_convolutions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for name in self.terminal_names:
                    t = self.node_table[name]

                    if t.p.family == 'DiracDelta':
                        self.convolutions[name] = self.irf_impulses[name]
                    else:
                        if t.cont:
                            impulse = self.irf_impulses[name]
                            irf = self.irf[name]
                            if len(irf) > 1:
                                irf = self._compose_irf(irf)
                            else:
                                irf = irf[0]

                            impulse_interp, t_interp = self._lininterp_fixed_frequency(
                                impulse,
                                self.time_X,
                                self.time_X_mask,
                                hz = self.interp_hz
                            )
                            t_delta_interp = tf.expand_dims(tf.expand_dims(self.time_y, -1) - t_interp, -1)
                            irf_seq = irf(t_delta_interp)

                            self.convolutions[name] = tf.reduce_sum(impulse_interp * irf_seq, axis=1)
                        else:
                            impulse = self.irf_impulses[name]

                            irf = self.irf[name]
                            if len(irf) > 1:
                                irf = self._compose_irf(irf)
                            else:
                                irf = irf[0]

                            irf_seq = irf(self.t_delta)

                            self.convolutions[name] = tf.reduce_sum(impulse * irf_seq, axis=1)

    def _construct_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_irfs(self.t)
                self._initialize_impulses()
                self._initialize_convolutions()
                self._initialize_backtransformed_irf_plot(self.t)

                convolutions = [self.convolutions[x] for x in self.terminal_names]
                if len(convolutions) > 0:
                    self.X_conv = tf.concat(convolutions, axis=1)
                else:
                    self.X_conv = tf.zeros((1, 1), dtype=self.FLOAT_TF)

                coef_names = [self.node_table[x].coef_id() for x in self.terminal_names]
                coef_ix = names2ix(coef_names, self.coef_names)
                coef = tf.gather(self.coefficient, coef_ix, axis=1)
                self.X_conv_scaled = self.X_conv*coef

                self.out = self.intercept + tf.reduce_sum(self.X_conv_scaled, axis=1)

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
                    if 'cosine' in self.lr_decay_family:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            self.global_step,
                            lr_decay_steps,
                            name='learning_rate'
                        )
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            self.global_step,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                return {
                    'SGD': lambda x: tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: tf.train.AdamOptimizer(x, epsilon=self.optim_epsilon),
                    'FTRL': lambda x: tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: tf.contrib.opt.NadamOptimizer(x, epsilon=self.optim_epsilon)
                }[name](self.lr)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_by_iter', self.loss_total, collections=['loss'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dtsr', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dtsr')
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                    self.saver = tf.train.Saver()

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                vars = tf.get_collection('trainable_variables')
                self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                self.ema_op = self.ema.apply(vars)
                self.ema_map = {}
                for v in vars:
                    self.ema_map[self.ema.average_name(v)] = v
                self.ema_saver = tf.train.Saver(self.ema_map)





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
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param ran_gf: ``str`` or ``None``; Name of random grouping factor for random intercept (if ``None``, constructs a fixed intercept)
        :return: 2-tuple of ``Tensor`` ``(intercept, intercept_summary)``; ``intercept`` is the intercept for use by the model. ``intercept_summary`` is an identically-shaped representation of the current intercept value for logging and plotting (can be identical to ``intercept``). For fixed intercepts, should return a trainable scalar. For random intercepts, should return batch-length vector of trainable weights. Weights should be initialized around 0.
        """
        raise NotImplementedError

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        """
        Add coefficients.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of ``len(coef_ids)`` trainable weights. For random coefficients, should return batch-length matrix of trainable weights with ``len(coef_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0, ran_gf=None):
        """
        Add IRF parameters in the unconstrained space.
        DTSR will apply appropriate constraint transformations as needed.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param param_name: ``str``; Name of parameter (e.g. ``"alpha"``)
        :param ids: ``list`` of ``str``; Names of IRF nodes to which this parameter applies
        :param mean: ``float`` or ``Tensor``; scalar (broadcasted) or 1D tensor (shape = ``(len(ids),)``) of parameter means on the transformed space.
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(param, param_summary)``; ``param`` is the parameter for use by the model. ``param_summary`` is an identically-shaped representation of the current param values for logging and plotting (can be identical to ``param``). For fixed params, should return a vector of ``len(ids)`` trainable weights. For random params, should return batch-length matrix of trainable weights with ``len(ids)``. Weights should be initialized around **mean** (if fixed) or ``0`` (if random).
        """

        raise NotImplementedError

    def initialize_joint_distribution(self, means, sds, ran_gf=None):
        """
        Add a multivariate joint distribution over the parameters represented by **means**, where **means** are on the unconstrained space for bounded params.
        The variance-covariance matrix is initialized using the square of **sds** as the diagonal.
        This method is required for multivariate mode and must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param means: ``Tensor``; 1-D tensor as MVN mean parameter.
        :param sds: ``Tensor``; 1-D tensor used to construct diagonal of MVN variance-covariance parameter.
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(joint, join_summary)``; ``joint`` is the random variable for use by the model. ``joint_summary`` is an identically-shaped representation of the current joint for logging and plotting (can be identical to ``joint``). Returns a multivariate normal distribution of dimension len(means) in all cases.
        """

        raise NotImplementedError

    def initialize_objective(self):
        """
        Add an objective function to the DTSR model.

        :return: ``None``
        """

        raise NotImplementedError





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _new_irf(self, irf_lambda, params, parent_irf=None):
        irf = irf_lambda(params)
        if parent_irf is None:
            def new_irf(x):
                return irf(x)
        else:
            def new_irf(x):
                return irf(parent_irf(x))
        return new_irf

    def _compose_irf(self, f_list):
        if not isinstance(f_list, list):
            f_list = [f_list]
        with self.sess.as_default():
            with self.sess.graph.as_default():
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

    def _apply_pc(self, inputs, src_ix=None, pc_ix=None, inv=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if src_ix is None:
                    src_ix = np.arange(self.n_pc)
                if pc_ix is None:
                    pc_ix = np.arange(self.n_pc)

                e = self.e
                e = tf.gather(e, src_ix, axis=0)
                e = tf.gather(e, pc_ix, axis=1)
                if inv:
                    X = tf.gather(inputs, pc_ix, axis=-1)
                    e = tf.transpose(e, [0,1])
                else:
                    X = tf.gather(inputs, src_ix, axis=-1)
                expansions = 0
                while len(X.shape) < 2:
                    expansions += 1
                    X = tf.expand_dims(X, 0)
                outputs = self._matmul(X, e)
                if expansions > 0:
                    outputs = tf.squeeze(outputs, axis=list(range(expansions)))
                return outputs

    def _get_mean_init_vector(self, irf_ids, param_name, irf_param_init, default=0.):
        mean = np.zeros(len(irf_ids))
        for i in range(len(irf_ids)):
            mean[i] = irf_param_init[irf_ids[i]].get(param_name, default)
        return mean

    def _process_mean(self, mean, lb=None, ub=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                mean = tf.constant(mean, dtype=self.FLOAT_TF)
                if lb is not None:
                    lb = tf.constant(lb, dtype=self.FLOAT_TF)
                if ub is not None:
                    ub = tf.constant(ub, dtype=self.FLOAT_TF)

                if lb is not None and ub is None:
                    # Lower-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(mean - lb - self.epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(-(mean - ub + self.epsilon))
                elif lb is not None and ub is not None:
                    # Finite-interval bounded support
                    mean = self._softplus_sigmoid_inverse(mean, lb, ub)

        return mean, lb, ub

    def _compute_trainable_untrainable_ix(self, param_name, ids, trainable=None):
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

    def _collect_plots(self):
        switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.plots = {}
                irf_names = [x for x in self.node_table if x in self.irf_plot and not (len(self.node_table[x].children) == 1 and self.node_table[x].children[0].terminal())]
                irf_names_terminal = [x for x in self.node_table if x in self.irf_plot and self.node_table[x].terminal()]

                for a in switches[0]:
                    if a not in self.plots:
                        self.plots[a] = {}
                    for b in switches[1]:
                        plot_y = []
                        names = irf_names if b == 'unscaled' else irf_names_terminal
                        for x in names:
                            plot_y.append(self.irf_plot[x][a][b])

                        self.plots[a][b] = {
                            'names': names,
                            'plot': plot_y
                        }

                if self.pc:
                    self.src_plot_tensors = {}
                    irf_names = [x for x in self.src_node_table if x in self.src_irf_plot and not (len(self.src_node_table[x].children) == 1 and self.src_node_table[x].children[0].terminal())]

                    for a in switches[0]:
                        if a not in self.src_plot_tensors:
                            self.src_plot_tensors[a] = {}
                        for b in switches[1]:
                            plot_y = []
                            names = irf_names if b == 'unscaled' else irf_names_terminal
                            for x in names:
                                plot_y.append(self.src_irf_plot[x][a][b])

                            self.src_plot_tensors[a][b] = {
                                'names': names,
                                'plot': plot_y
                            }

    def _regularize(self, var, center=None, type=None, var_name=None):
        assert type in [None, 'intercept', 'coefficient', 'irf', 'ranef']
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

    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if fixed:
                    out = self.parameter_table_fixed_values.eval(session=self.sess)
                else:
                    out = self.parameter_table_random_values.eval(session=self.sess)

            return out

    def _extract_irf_integral(self, terminal_name, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if n_time_units is None:
                    n_time_units = self.max_tdelta
                fd = {
                    self.support_start: 0.,
                    self.n_time_units: n_time_units,
                    self.n_time_points: n_time_points,
                    self.gf_y: np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1,
                    self.time_y: [n_time_units],
                    self.time_X: np.zeros((1, self.history_length))
                }

                if terminal_name in self.irf_integral_tensors:
                    irf_integral = self.irf_integral_tensors[terminal_name]
                else:
                    irf_integral = self.src_irf_integral_tensors[terminal_name]

                out = self.sess.run(irf_integral, feed_dict=fd)[0]

                return out

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_allow_missing(self, path, predict=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    sys.stderr.write('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    reader = tf.train.NewCheckpointReader(path)
                    saved_shapes = reader.get_variable_to_shape_map()
                    model_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                    ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                             if var.name.split(':')[0] in saved_shapes])

                    model_var_names_set = set([x[1] for x in model_var_names])
                    ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                    missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                    if len(missing_in_ckpt) > 0:
                        sys.stderr.write(
                            'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                            sorted(list(missing_in_ckpt))))
                    missing_in_model = ckpt_var_names_set - model_var_names_set
                    if len(missing_in_model) > 0:
                        sys.stderr.write(
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
                    param_random = tf.nn.softplus(param_random)

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
                    param_random = -tf.nn.softplus(param_random)

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

    def run_predict_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        """
        Generate predictions from a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor values.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """
        raise NotImplementedError

    def run_loglik_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        """
        Compute the log-likelihoods of a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Pointwise log-likelihoods, one for each training sample
        """

        raise NotImplementedError

    def run_conv_op(self, feed_dict, scaled=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor variables
        :param scaled: ``bool``; Whether to scale the outputs using the model's coefficients
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; The convolved inputs
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

    def verify_random_centering(self):
        """
        Assert that all random effects are properly centered (means sufficiently close to zero).

        :return: ``None``
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.rangf) > 0:
                    means = self.random_means.eval(session=self.sess)
                    centered = np.allclose(means, 0., rtol=1e-3, atol=1e-3)
                    assert centered, 'Some random parameters are not properly centered\n. Current random parameter means:\n %s' %means

    def build(self, outdir=None, restore=True):
        """
        Construct the DTSR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Report model details after initialization.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './dtsr_model/'
        else:
            self.outdir = outdir

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs()
                self._initialize_base_params()
                self._initialize_joint_distributions()
                self._initialize_joint_distribution_slices()
                self._initialize_intercepts_coefficients()
                self._initialize_irf_lambdas()
                self._initialize_irf_params()
                self._initialize_parameter_tables()
                self._initialize_random_mean_vector()
                self._construct_network()
                self.initialize_objective()
                self._initialize_logging()
                self._initialize_ema()

                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
                self._initialize_saver()
                self.load(restore=restore)

                self._collect_plots()

    def save(self, dir=None):
        """
        Save the DTSR model.

        :param dir: ``str``; output directory. If ``None``, use model default.
        :return: ``None``
        """
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
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        pytime.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)
    
    def load(self, outdir=None, predict=False, restore=True):
        """
        Load weights from a DTSR checkpoint and/or initialize the DTSR model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_allow_missing(outdir + '/model.ckpt', predict=predict)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def finalize(self):
        """
        Close the DTSR instance to prevent memory leaks.

        :return: ``None``
        """
        self.sess.close()

    def irf_integrals(self, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        """
        Generate effect size estimates by computing the area under each IRF curve in the model via discrete approximation.

        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``numpy`` array; IRF integrals, one IRF per row. If Bayesian, array also contains credible interval bounds.
        """
        if self.pc:
            terminal_names = self.src_terminal_names
        else:
            terminal_names = self.terminal_names
        irf_integrals = []
        for i in range(len(terminal_names)):
            terminal = terminal_names[i]
            integral = np.array(
                self.irf_integral(
                    terminal,
                    level=level,
                    n_samples=n_samples,
                    n_time_units=n_time_units,
                    n_time_points=n_time_points
                )
            )
            irf_integrals.append(integral)
        irf_integrals = np.stack(irf_integrals, axis=0)

        return irf_integrals

    def irf_integral(self, terminal_name, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        """
        Generate effect size estimates by computing the area under a specific IRF curve via discrete approximation.

        :param terminal_name: ``str``; string ID of IRF to extract.
        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``numpy`` array; IRF integral (scalar), or (if Bayesian) IRF 3x1 vector with mean, lower bound, and upper bound of credible interval.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return self._extract_irf_integral(
                    terminal_name,
                    level=level,
                    n_samples=n_samples,
                    n_time_units=n_time_units,
                    n_time_points=n_time_points
                )

    def set_predict_mode(self, mode):
        """
        Set predict mode.
        If set to ``True``, the model enters predict mode and replaces parameters with the exponential moving average of their training iterates.
        If set to ``False``, the model exits predict mode and replaces parameters with their most recently saved values.
        To avoid data loss, always save the model before entering predict mode.

        :param mode: ``bool``; if ``True``, enter predict mode. If ``False``, exit predict mode.
        :return: ``None``
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.load(predict=mode)

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
        for kwarg in DTSR_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def report_irf_tree(self, indent=0):
        """
        Generate a string representation of the model's IRF tree structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the IRF tree report
        """
        out = ' ' * indent + 'IRF TREE:\n'
        tree_str = str(self.t)
        new_tree_str = ''
        for line in tree_str.splitlines():
            new_tree_str += ' ' * (indent + 2) + line + '\n'
        out += new_tree_str + '\n'

        if self.pc:
            out = ' ' * indent + 'SOURCE IRF TREE:\n'
            tree_str = str(self.t_src)
            new_tree_str = ''
            for line in tree_str.splitlines():
                new_tree_str += ' ' * (indent + 2) + line
            out += new_tree_str + '\n'

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
                    cur_params = np.prod(np.array(v_val).shape)
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

    def report_evaluation(self, mse=None, mae=None, loglik=None, percent_variance_explained=None, indent=0):
        """
        Generate a string representation of pre-comupted evaluation metrics.

        :param mse: ``float`` or ``None``; mean squared error, skipped if ``None``.
        :param mae: ``float`` or ``None``; mean absolute error, skipped if ``None``.
        :param loglik: ``float`` or ``None``; log likelihood, skipped if ``None``.
        :param percent_variance_explained: ``float`` or ``None``; percent variance explained, skipped if ``None``.
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
        if percent_variance_explained is not None:
            out += ' ' * (indent+2) + 'Percent variance explained: %.2f%%\n' %percent_variance_explained

        out += '\n'

        return out

    def report_parameter_values(self, random=False, level=95, n_samples=None, indent=0):
        """
        Generate a string representation of the model's parameter table.

        :param random: ``bool``; report random effects estimates.
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
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

    def report_irf_integrals(self, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's IRF integrals (effect sizes)

        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the IRF integrals report
        """

        if integral_n_time_units is None:
            integral_n_time_units = self.max_tdelta

        irf_integrals = self.irf_integrals(
            level=level,
            n_samples=n_samples,
            n_time_units=integral_n_time_units,
            n_time_points=1000
        )
        if self.pc:
            terminal_names = self.src_terminal_names
        else:
            terminal_names = self.terminal_names
        irf_integrals = pd.DataFrame(
            irf_integrals,
            index=terminal_names,
            columns=self.parameter_table_columns
        )

        out = ' ' * indent + 'IRF INTEGRALS (EFFECT SIZES):\n'
        out += ' ' * (indent + 2) + 'Integral upper bound (time): %s\n\n' % integral_n_time_units

        ci_str = irf_integrals.to_string()

        for line in ci_str.splitlines():
            out += ' ' * (indent + 2) + line + '\n'

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

    def parameter_summary(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's effect sizes and parameter values.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the parameter summary
        """

        out = ' ' * indent + '-----------------\n'
        out += ' ' * indent + 'PARAMETER SUMMARY\n'
        out += ' ' * indent + '-----------------\n\n'

        out += self.report_irf_integrals(
            level=level,
            n_samples=n_samples,
            integral_n_time_units=integral_n_time_units,
            indent=indent+2
        )

        out += self.report_parameter_values(
            random=random,
            level=level,
            n_samples=level,
            indent=indent+2
        )

        return out

    def summary(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a summary of the fitted model.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :return: ``str``; the model summary
        """

        out = '  ' * indent + '*' * 100 + '\n\n'
        out += ' ' * indent + '############################\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '#    DTSR MODEL SUMMARY    #\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '############################\n\n\n'

        out += self.initialization_summary(indent =indent + 2)
        out += '\n'
        out += self.training_evaluation_summary(indent =indent + 2)
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


    ######################################################
    #
    #  High-level methods for training, prediction,
    #  and plotting
    #
    ######################################################

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
            plot_x_inches=7,
            plot_y_inches=5,
            cmap='gist_rainbow'
            ):
        """
        Fit the DTSR model.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

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
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose values at each time point differ for each regression target) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param force_training_evaluation: ``bool``; (Re-)run post-fitting evaluation, even if resuming a model whose training is already complete.
        :param n_iter: ``int``; the number of training iterations
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``float``; number of points to use for plotting.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """

        sys.stderr.write('*' * 100 + '\n')
        sys.stderr.write(self.initialization_summary())
        sys.stderr.write('*' * 100 + '\n\n')

        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)
        sys.stderr.write('Number of training samples: %d\n\n' % len(y))

        if self.pc:
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

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            y.first_obs,
            y.last_obs,
            impulse_names,
            history_length=128,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        sys.stderr.write('Correlation matrix for input variables:\n')
        impulse_names_2d = [x for x in impulse_names if x in X_2d_predictor_names]
        rho = corr_dtsr(X_2d, impulse_names, impulse_names_2d, time_X_mask)
        sys.stderr.write(str(rho) + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.global_step.eval(session=self.sess) < n_iter:
                    self.set_training_complete(False)

                if self.training_complete.eval(session=self.sess):
                    sys.stderr.write('Model training is already complete; no additional updates to perform. To train for additional iterations, re-run fit() with a larger n_iter.\n\n')
                else:
                    if self.global_step.eval(session=self.sess) == 0:
                        summary_params = self.sess.run(self.summary_params)
                        self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                        if self.log_random and len(self.rangf) > 0:
                            summary_random = self.sess.run(self.summary_random)
                            self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                    else:
                        sys.stderr.write('Resuming training from most recent checkpoint...\n\n')

                    while self.global_step.eval(session=self.sess) < n_iter:
                        p, p_inv = get_random_permutation(len(y))
                        t0_iter = pytime.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

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
                                self.gf_y: gf_y[indices] if len(gf_y > 0) else gf_y
                            }

                            info_dict = self.run_train_step(fd_minibatch)
                            loss_cur = info_dict['loss']
                            self.sess.run(self.ema_op)
                            if not np.isfinite(loss_cur):
                                loss_cur = 0
                            loss_total += loss_cur

                            pb.update((j/minibatch_size)+1, values=[('loss', loss_cur)])

                        self.sess.run(self.incr_global_step)

                        self.verify_random_centering()

                        if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                            loss_total /= n_minibatch
                            summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                            self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
                            if self.log_random and len(self.rangf) > 0:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))

                        if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                            self.save()
                            self.make_plots(
                                irf_name_map=irf_name_map,
                                plot_n_time_units=plot_n_time_units,
                                plot_n_time_points=plot_n_time_points,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                cmap=cmap,
                                keep_plot_history=self.keep_plot_history
                            )
                            if type(self).__name__ == 'DTSRBayes' and self.asymmetric_error:
                                lb = self.sess.run(self.err_dist_lb)
                                ub = self.sess.run(self.err_dist_ub)
                                n_time_units = ub-lb
                                fd_plot = {
                                    self.support_start: lb,
                                    self.n_time_units: n_time_units,
                                    self.n_time_points: 1000
                                }
                                plot_x = self.sess.run(self.support, feed_dict=fd_plot)
                                plot_y = self.sess.run(self.err_dist_plot, feed_dict=fd_plot)
                                err_dist_filename = 'error_distribution_%s.png' %self.global_step.eval(sess=self.sess) if self.keep_plot_history else 'error_distribution.png'
                                plot_irf(
                                    plot_x,
                                    plot_y,
                                    ['Error Distribution'],
                                    dir=self.outdir,
                                    filename=err_dist_filename,
                                    legend=False,
                                )
                            self.verify_random_centering()

                        t1_iter = pytime.time()
                        sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                    self.save()




                    # End of training plotting and evaluation.
                    # For DTSRMLE, this is a crucial step in the model definition because it provides the
                    # variance of the output distribution for computing log likelihood.

                    self.make_plots(
                        irf_name_map=irf_name_map,
                        plot_n_time_units=plot_n_time_units,
                        plot_n_time_points=plot_n_time_points,
                        plot_x_inches=plot_x_inches,
                        plot_y_inches=plot_y_inches,
                        cmap=cmap,
                        keep_plot_history=self.keep_plot_history
                    )

                    if type(self).__name__ == 'DTSRBayes':
                        # Generate plots with 95% credible intervals
                        self.make_plots(
                            irf_name_map=irf_name_map,
                            plot_n_time_units=plot_n_time_units,
                            plot_n_time_points=plot_n_time_points,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap,
                            mc=True,
                            keep_plot_history=self.keep_plot_history
                        )


                if not self.training_complete.eval(session=self.sess) or force_training_evaluation:
                    # Extract and save predictions
                    preds = self.predict(
                        X,
                        y.time,
                        y[self.form.rangf],
                        y.first_obs,
                        y.last_obs,
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
                    with open(self.outdir + '/mse_losses_train.txt','w') as l_file:
                        for i in range(len(training_se)):
                            l_file.write(str(training_se[i]) + '\n')
                    training_percent_variance_explained = percent_variance_explained(y[self.dv], preds)

                    training_ae = np.array(np.abs(y[self.dv] - preds))
                    training_mae = training_ae.mean()
                    with open(self.outdir + '/mae_losses_train.txt','w') as l_file:
                        for i in range(len(training_ae)):
                            l_file.write(str(training_ae[i]) + '\n')

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

                    with open(self.outdir + '/eval_train.txt', 'w') as e_file:
                        eval_train = '------------------------\n'
                        eval_train += 'DTSR TRAINING EVALUATION\n'
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

                    self.save_parameter_table()

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
            verbose=True
    ):
        """
        Predict from the pre-trained DTSR model.
        Predictions are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the regression targets, grouped by series.
        :param y_rangf: ``pandas`` ``Series`` or 1D ``numpy`` array; random grouping factor values (if applicable).
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param first_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose values at each time point differ for each regression target) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert len(y_time) == len(y_rangf) == len(first_obs) == len(last_obs), 'y_time, y_rangf, first_obs, and last_obs must be of identical length. Got: len(y_time) = %d, len(y_rangf) = %d, len(first_obs) = %d, len(last_obs) = %d' % (len(y_time), len(y_rangf), len(first_obs), len(last_obs))

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            sys.stderr.write('Computing predictions...\n')

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            history_length=128,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }


                if not np.isfinite(self.eval_minibatch_size):
                    preds = self.run_predict_op(fd, n_samples=n_samples, algorithm=algorithm, verbose=verbose)
                else:
                    preds = np.zeros((len(y_time),))
                    n_eval_minibatch = math.ceil(len(y_time) / self.eval_minibatch_size)
                    for i in range(0, len(y_time), self.eval_minibatch_size):
                        if verbose:
                            sys.stderr.write('\rMinibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                            sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y
                        }
                        preds[i:i + self.eval_minibatch_size] = self.run_predict_op(fd_minibatch, n_samples=n_samples, algorithm=algorithm, verbose=verbose)

                if verbose:
                    sys.stderr.write('\n\n')

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
            verbose=True
    ):
        """
        Compute log-likelihood of data from predictive posterior.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose values at each time point differ for each regression target) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            sys.stderr.write('Computing likelihoods...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            y['first_obs'],
            y['last_obs'],
            impulse_names,
            history_length=128,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_X_mask: time_X_mask,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.y: y_dv
                }


                if not np.isfinite(self.eval_minibatch_size):
                    log_lik = self.run_loglik_op(fd, n_samples=n_samples, algorithm=algorithm, verbose=verbose)
                else:
                    log_lik = np.zeros((len(time_y),))
                    n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                    for i in range(0, len(time_y), self.eval_minibatch_size):
                        if verbose:
                            sys.stderr.write('\rMinibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                            sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.eval_minibatch_size]
                        }
                        log_lik[i:i+self.eval_minibatch_size] = self.run_loglik_op(fd_minibatch, n_samples=n_samples, algorithm=algorithm, verbose=verbose)

                if verbose:
                    sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return log_lik

    def convolve_inputs(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            scaled=False,
            n_samples=None,
            algorithm='MAP',
            verbose=True
    ):
        """
        Convolve input data using the fitted DTSR model.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose values at each time point differ for each regression target) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            y['first_obs'],
            y['last_obs'],
            impulse_names,
            history_length=128,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_X_mask: time_X_mask
                }

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                X_conv = []
                n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                for i in range(0, len(y), self.eval_minibatch_size):
                    if verbose:
                        sys.stderr.write('\rMinibatch %d/%d\n' % ((i / self.eval_minibatch_size) + 1, n_eval_minibatch))
                        sys.stderr.flush()
                    fd_minibatch[self.time_y] = time_y[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.gf_y] = gf_y[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.X] = X_2d[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.time_X] = time_X_2d[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.time_X_mask] = time_X_mask[i:i + self.eval_minibatch_size]
                    X_conv_cur = self.run_conv_op(fd_minibatch, scaled=scaled, n_samples=n_samples, algorithm=algorithm, verbose=verbose)
                    X_conv.append(X_conv_cur)
                names = []
                for x in self.terminal_names:
                    if self.node_table[x].p.irfID is None:
                        names.append(sn(''.join(x.split('-')[:-1])))
                    else:
                        names.append(sn(x))
                X_conv = np.concatenate(X_conv, axis=0)
                out = pd.DataFrame(X_conv, columns=names, dtype=self.FLOAT_NP)

                self.set_predict_mode(False)

                convolution_summary = ''
                corr_conv = out.corr()
                convolution_summary += '=' * 50 + '\n'
                convolution_summary += 'Correlation matrix of convolved predictors:\n\n'
                convolution_summary += str(corr_conv) + '\n\n'

                select = np.where(np.isclose(time_X_2d[:,-1], time_y))[0]

                X_input = X_2d[:,-1,:][select]

                extra_cols = []
                for c in y.columns:
                    if c not in out:
                        extra_cols.append(c)
                out = pd.concat([y[extra_cols].reset_index(), out], axis=1)

                if X_input.shape[0] > 0:
                    out_plus = out
                    for i in range(len(self.impulse_names)):
                        c = self.impulse_names[i]
                        if c not in out_plus:
                            out_plus[c] = X_2d[:,-1,i]
                    corr_conv = out_plus.iloc[select].corr()
                    convolution_summary += '-' * 50 + '\n'
                    convolution_summary += 'Full correlation matrix of input and convolved predictors:\n'
                    convolution_summary += 'Based on %d simultaneously sampled impulse/response pairs (out of %d total data points)\n\n' %(select.shape[0], y.shape[0])
                    convolution_summary += str(corr_conv) + '\n\n'
                    convolution_summary += '=' * 50 + '\n'

                return out, convolution_summary

    def make_plots(
            self,
            irf_name_map=None,
            irf_ids=None,
            plot_n_time_units=2.5,
            plot_n_time_points=1000,
            plot_x_inches=6.,
            plot_y_inches=4.,
            cmap=None,
            mc=False,
            level=95,
            n_samples=None,
            prefix=None,
            legend=True,
            xlab=None,
            ylab=None,
            transparent_background=False,
            keep_plot_history=False
    ):
        """
        Generate plots of current state of deconvolution.
        DTSR distinguishes plots based on two orthogonal criteria: "atomic" vs. "composite" and "scaled" vs. "unscaled".
        The "atomic"/"composite" distinction is only relevant in models containing composed IRF.
        In such models, "atomic" plots represent the shape of the IRF irrespective of any other IRF with which they are composed, while "composite" plots represent the shape of the IRF composed with any upstream IRF in the model.
        In models without composed IRF, only "atomic" plots are generated.
        The "scaled"/"unscaled" distinction concerns whether the impulse coefficients are represented in the plot ("scaled") or not ("unscaled").
        Only pre-terminal IRF (i.e. the final IRF in all IRF compositions) have coefficients, so only preterminal IRF are represented in "scaled" plots, while "unscaled" plots also contain all intermediate IRF.
        In addition, Bayesian DTSR implementations also support MC sampling of credible intervals around all curves.
        Outputs are saved to the model's output directory as PNG files with names indicating which plot type is represented.
        All plot types relevant to a given model are generated.

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param irf_ids: ``list`` or ``None``; List of irf ID's to plot. If ``None``, all IRF's are plotted.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``float``; number of points to use for plotting.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param mc: ``bool``; compute and plot Monte Carlo credible intervals (only supported for DTSRBayes).
        :param level: ``float``; significance level for credible intervals, ignored unless **mc** is ``True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param prefix: ``str`` or ``None``; prefix appended to output filenames. If ``None``, no prefix added.
        :param legend: ``bool``; generate a plot legend.
        :param xlab: ``str`` or ``None``; x-axis label. If ``None``, no label.
        :param ylab: ``str`` or ``None``; y-axis label. If ``None``, no label.
        :param transparent_background: ``bool``; use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot.
        :return: ``None``
        """

        assert not mc or type(self).__name__ == 'DTSRBayes', 'Monte Carlo estimation of credible intervals (mc=True) is only supported for DTSRBayes models.'

        if len(self.terminal_names) == 0:
            return

        if prefix is None:
            prefix = ''
        if prefix != '':
            prefix += '_'
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: plot_n_time_units,
                    self.n_time_points: plot_n_time_points,
                    self.gf_y: np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1,
                    self.max_tdelta_batch: plot_n_time_units
                }

                plot_x = self.sess.run(self.support, fd)

                switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

                for a in switches[0]:
                    if self.t.has_composed_irf() or a == 'atomic':
                        for b in switches[1]:
                            plot_name = 'irf_%s_%s_%d.png' %(a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'irf_%s_%s.png' %(a, b)
                            names = self.plots[a][b]['names']
                            if irf_ids is not None and len(irf_ids) > 0:
                                new_names = []
                                for i, name in enumerate(names):
                                    for ID in irf_ids:
                                        if ID==name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                                            new_names.append(name)
                                names = new_names
                            if len(names) > 0:
                                if mc:
                                    plot_y = []
                                    lq = []
                                    uq = []
                                    for name in names:
                                        mean_cur, lq_cur, uq_cur = self.ci_curve(
                                            self.irf_mc[name][a][b],
                                            level=level,
                                            n_samples=n_samples,
                                            n_time_units=plot_n_time_units,
                                            n_time_points=plot_n_time_points,
                                        )
                                        plot_y.append(mean_cur)
                                        lq.append(lq_cur)
                                        uq.append(uq_cur)
                                    lq = np.stack(lq, axis=1)
                                    uq = np.stack(uq, axis=1)
                                    plot_y = np.stack(plot_y, axis=1)
                                    plot_name = 'mc_' + plot_name
                                else:
                                    plot_y = [self.sess.run(self.plots[a][b]['plot'][i], feed_dict=fd) for i in range(len(self.plots[a][b]['plot'])) if self.plots[a][b]['names'][i] in names]
                                    lq = None
                                    uq = None
                                    plot_y = np.concatenate(plot_y, axis=1)

                                plot_irf(
                                    plot_x,
                                    plot_y,
                                    names,
                                    lq=lq,
                                    uq=uq,
                                    dir=self.outdir,
                                    filename=prefix + plot_name,
                                    irf_name_map=irf_name_map,
                                    plot_x_inches=plot_x_inches,
                                    plot_y_inches=plot_y_inches,
                                    cmap=cmap,
                                    legend=legend,
                                    xlab=xlab,
                                    ylab=ylab,
                                    transparent_background=transparent_background
                                )

                if self.pc:
                    for a in switches[0]:
                        if self.t.has_composed_irf() or a == 'atomic':
                            for b in switches[1]:
                                if b == 'scaled':
                                    plot_name = 'src_irf_%s_%s_%d.png' % (a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'src_irf_%s_%s.png' % (a, b)
                                    names = self.src_plot_tensors[a][b]['names']
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
                                            plot_y = []
                                            lq = []
                                            uq = []
                                            for name in names:
                                                mean_cur, lq_cur, uq_cur = self.ci_curve(
                                                    self.src_irf_mc[name][a][b],
                                                    level=level,
                                                    n_samples=n_samples,
                                                    n_time_units=plot_n_time_units,
                                                    n_time_points=plot_n_time_points,
                                                )
                                                plot_y.append(mean_cur)
                                                lq.append(lq_cur)
                                                uq.append(uq_cur)
                                            lq = np.stack(lq, axis=1)
                                            uq = np.stack(uq, axis=1)
                                            plot_y = np.stack(plot_y, axis=1)
                                            plot_name = 'mc_' + plot_name
                                        else:
                                            plot_y = [self.sess.run(self.src_plot_tensors[a][b]['plot'][i], feed_dict=fd) for i in range(len(self.src_plot_tensors[a][b]['plot'])) if self.src_plot_tensors[a][b]['names'][i] in names]
                                            lq = None
                                            uq = None
                                            plot_y = np.concatenate(plot_y, axis=1)

                                        plot_irf(
                                            plot_x,
                                            plot_y,
                                            names,
                                            lq=lq,
                                            uq=uq,
                                            dir=self.outdir,
                                            filename=prefix + plot_name,
                                            irf_name_map=irf_name_map,
                                            plot_x_inches=plot_x_inches,
                                            plot_y_inches=plot_y_inches,
                                            cmap=cmap,
                                            legend=legend,
                                            xlab=xlab,
                                            ylab=ylab,
                                            transparent_background=transparent_background
                                        )

                self.set_predict_mode(False)

    def plot_eigenvectors(self):
        """
        Save heatmap representation of training data eigenvector matrix to the model's output directory.
        Will throw an error unless ``self.pc == True``.

        :return: ``None``
        """
        plot_heatmap(self.eigenvec, self.src_impulse_names_norate, self.impulse_names_norate, dir=self.outdir)

    def parameter_table(self, fixed=True, level=95, n_samples=None):
        """
        Generate a pandas table of parameter names and values.

        :param fixed: ``bool``; Return a table of fixed parameters (otherwise returns a table of random parameters).
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :return: ``pandas`` ``DataFrame``; The parameter table.
        """

        assert fixed or len(self.rangf) > 0, 'Attempted to generate a random effects parameter table in a fixed-effects-only model'

        if n_samples is None and getattr(self, 'n_samples_eval', None) is not None:
            n_samples = self.n_samples_eval

        with self.sess.as_default():
            with self.sess.graph.as_default():
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

                columns = self.parameter_table_columns
                out = pd.concat([out, pd.DataFrame(values, columns=columns)], axis=1)

                return out

    def save_parameter_table(self, random=True, level=95, n_samples=None):
        """
        Save space-delimited parameter table to the model's output directory.

        :param random: Include random parameters.
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :return: ``None``
        """

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

        parameter_table.to_csv(self.outdir + '/dtsr_parameters.csv', index=False)

