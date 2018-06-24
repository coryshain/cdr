import os
from collections import defaultdict
from numpy import inf
import pandas as pd
import time as pytime

from .formula import *
from .util import *
from .data import build_DTSR_impulses, corr_dtsr
from .plot import *

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their documentation.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param type: ``str``; Description of data type of kwarg
    :param descr: ``str``; Description of kwarg
    """
    def __init__(self, key, default_value, type, descr):
        self.key = key
        self.default_value = default_value
        self.type = type
        self.descr = descr





######################################################
#
#  ABSTRACT DTSR CLASS
#
######################################################

DTSR_INITIALIZATION_KWARGS = [
    Kwarg(
        'outdir',
        './dtsr_model/',
        "``str``",
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'history_length',
        None,
        "``int``",
        "Length of the history window to use."
    ),
    Kwarg(
        'pc',
        False,
        "``bool``",
        "Transform input variables using principal components analysis (experimental, not thoroughly tested)."
    ),
    Kwarg(
        'intercept_init',
        None,
        "``float`` or ``None``",
        "Initial value to use for the intercept (if ``None``, use mean response in training data)"
    ),
    Kwarg(
        'init_sd',
        .01,
        "``float``",
        "Standard deviation of Gaussian initialization distribution for trainable variables."
    ),
    Kwarg(
        'n_interp',
        64,
        "``int``",
        "Number of interpolation points (ignored unless the model formula specification contains continuous inputs)."
    ),
    Kwarg(
        'optim_name',
        'Adam',
        "``str``",
        "Name of the optimizer to use. Choose from ``'SGD'``, ``'AdaGrad'``, ``'AdaDelta'``, ``'Adam'``, ``'FTRL'``, ``'RMSProp'``, ``'Nadam'``."
    ),
    Kwarg(
        'optim_epsilon',
        0.01,
        "``float``",
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.01,
        "``float``",
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        1e-4,
        "``float``",
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        "``str`` or ``None``",
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        0.,
        "``float``",
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_steps',
        25,
        "``int``",
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        "``bool``",
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'regularizer_name',
        None,
        "``str``",
        "Name of regularizer to use (e.g. ``l1``, ``l2``), or if ``None``, no regularization."
    ),
    Kwarg(
        'regularizer_scale',
        0.01,
        "``float``",
        "Regularizer scale (ignored if ``regularizer_name==None``)."
    ),
    Kwarg(
        'ema_decay',
        0.999,
        "``float``",
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
        "``int`` or ``None``",
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        100000,
        "``int`` or ``None``",
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'float_type',
        'float32',
        "``str``",
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        "``str``",
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),
    Kwarg(
        'queue_capacity',
        100000,
        "``int``",
        "Queue capacity for data feeding (currently not used)."
    ),
    Kwarg(
        'num_threads',
        8,
        "``int``",
        "Number of threads for data dequeuing (currently not used)"
    ),
    Kwarg(
        'save_freq',
        1,
        "``int``",
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'log_random',
        True,
        "``bool``",
        "Log random effects to Tensorboard."
    ),
    Kwarg(
        'log_freq',
        1,
        "``int``",
        "Frequency (in iterations) with which to log model params to Tensorboard."
    ),
    Kwarg(
        'log_graph',
        False,
        "``bool``",
        "Log the network graph to Tensorboard"
    )
]

class DTSR(object):

    _INITIALIZATION_KWARGS = DTSR_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for DTSR. Bayesian (:ref:`dtsrbayes`) and MLE (:ref:`dtsrmle`) implementations inherit from ``DTSR``.
        ``DTSR`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``DTSR`` must implement the following instance methods:
        
            * ``initialize_intercept()``
            * ``initialize_coefficient()``
            * ``initialize_irf_param()``
            * ``initialize_objective()``
            * ``run_conv_op()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            * ``run_train_step()``
            * ``summary()``
            
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
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' %x.key + ': ' + '; '.join([x.type, x.descr]) + ' **Default**: ``%s``.' %(x.default_value if not isinstance(x.default_value, str) else "'%s'" %x.default_value) for x in _INITIALIZATION_KWARGS])
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

    def __init__(
            self,
            form_str,
            X,
            y,
            **kwargs
        ):

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

        # Initialize lookup tables of network objects
        self.irf_lambdas = {}
        self.irf_params = {}
        self.irf_params_summary = {}
        self.irf = {}
        self.irf_plot = {}
        self.irf_mc = {}
        self.mc_integrals = {}
        if self.pc:
            self.src_irf_plot = {}
            self.src_irf_mc = {}
            self.src_mc_integrals = {}
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

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.intercept_init is None:
                    self.intercept_init = self.y_train_mean
                self.intercept_init_tf = tf.constant(self.intercept_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(1e-35, dtype=self.FLOAT_TF)

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

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))
                self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)

                # QueueRunner setup for parallelism, seems to have negative performance impact, so turned off

                # self.batch_size = tf.placeholder(shape=[], dtype=self.INT_TF, name='batch_size')
                #
                # self.df_train = tf.data.Dataset.from_generator(
                #     lambda: self.data_generator(),
                #     (self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.INT_TF),
                #     ([self.history_length, n_impulse], [self.history_length], [], [], [len(self.rangf)])
                # )
                # self.df_train = self.df_train.repeat()
                #
                # self.df_eval = tf.data.Dataset.from_generator(
                #     lambda: self.data_generator(),
                #     (self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.INT_TF),
                #     ([self.history_length, n_impulse], [self.history_length], [], [], [len(self.rangf)])
                # )
                #
                # self.iterator = tf.data.Iterator.from_structure(
                #     self.df_train.output_types,
                #     self.df_train.output_shapes
                # )
                # self.iterator_init_train = self.iterator.make_initializer(self.df_train)
                # self.iterator_init_eval = self.iterator.make_initializer(self.df_eval)
                #
                # self.queue = tf.RandomShuffleQueue(
                #     capacity=self.queue_capacity,
                #     min_after_dequeue=int(.9 * self.queue_capacity),
                #     shapes=(
                #         [self.history_length, n_impulse],
                #         [self.history_length],
                #         [],
                #         [],
                #         [len(self.rangf)]
                #     ),
                #     dtypes=(self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.FLOAT_TF, self.INT_TF)
                # )
                # self.enqueue = self.queue.enqueue(self.iterator.get_next())
                # self.qr = tf.train.QueueRunner(self.queue, [self.enqueue] * self.num_threads)
                # tf.train.add_queue_runner(self.qr)
                # self.X, self.time_X, self.y, self.time_y, self.gf_y = self.queue.dequeue_many(self.batch_size)


                # Linspace tensor used for plotting
                self.support_start = tf.placeholder(self.FLOAT_TF, shape=[], name='support_start')
                self.n_time_units = tf.placeholder(self.FLOAT_TF, shape=[], name='n_time_units')
                self.n_points_per_time_unit = tf.placeholder(self.INT_TF, shape=[], name='n_points_per_time_unit')
                self.support = tf.lin_space(
                    self.support_start,
                    self.n_time_units+self.support_start,
                    tf.cast(self.n_time_units * tf.cast(self.n_points_per_time_unit, self.FLOAT_TF), self.INT_TF) + 1,
                    name='support'
                )
                self.support = tf.expand_dims(self.support, -1)
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

                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)
                    rate_ix = names2ix('rate', self.src_impulse_names)
                    self.X_rate = tf.gather(self.X, rate_ix, axis=-1)

                if self.regularizer_name is not None:
                    self.regularizer = getattr(tf.contrib.layers, self.regularizer_name)(self.regularizer_scale)

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')

    def _initialize_intercepts_coefficients(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                if self.has_intercept[None]:
                    self.intercept, self.intercept_summary = self.initialize_intercept()
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_summary,
                        collections=['params']
                    )
                else:
                    self.intercept = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')

                fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)
                coefficient_fixed_mask = np.zeros(len(self.coef_names), dtype=self.FLOAT_NP)
                coefficient_fixed_mask[fixef_ix] = 1.
                coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)

                coef_ids = self.coef_names

                self.coefficient_fixed, self.coefficient_summary = self.initialize_coefficient(coef_ids=coef_ids)

                self.coefficient_fixed *= coefficient_fixed_mask
                self.coefficient_summary *= coefficient_fixed_mask
                self.coefficient = self.coefficient_fixed


                for i in range(len(self.coef_names)):
                    tf.summary.scalar(
                        'coefficient' + '/%s' % self.coef_names[i],
                        self.coefficient_summary[i],
                        collections=['params']
                    )

                # if self.pc:
                #     src_coef = {}
                #     src_coef_summary = {}
                #     for pc_coef in self.coef_names:
                #         for pc_term_name in self.coef2terminal[pc_coef]:
                #             pc_term = self.node_table[pc_term_name]
                #             if pc_term.impulse.name() == 'rate':
                #                 assert len(self.bw_pointers[pc_term.name()]) == 1, 'PC terminal with impulse=rate maps to more than one source terminal'
                #                 term_name = self.bw_pointers[pc_term.name()][0]
                #                 term = self.src_node_table[term_name]
                #                 pc_coef_ix = names2ix(pc_term.coef_id(), self.coef_names)
                #                 computed_coef_summary = tf.gather(self.coefficient_summary, pc_coef_ix)
                #                 if term.coef_id() in src_coef_summary:
                #                     src_coef_summary[term.coef_id()] += computed_coef_summary
                #                 else:
                #                     src_coef_summary[term.coef_id()] = computed_coef_summary
                #             else:
                #                 for term_name in self.bw_pointers[pc_term.name()]:
                #                     term = self.src_node_table[term_name]
                #                     pc_imp_ix = names2ix(pc_term.impulse.name(), self.impulse_names_norate)
                #                     imp_ix = names2ix(term.impulse.name(), self.src_impulse_names_norate)
                #                     computed_coef_summary = self.__apply_pc__(
                #                         self.coefficient_summary,
                #                         src_ix=imp_ix,
                #                         pc_ix=pc_imp_ix,
                #                         inv=True
                #                     )
                #                     if term.coef_id() in src_coef_summary:
                #                         src_coef_summary[term.coef_id()] += computed_coef_summary
                #                     else:
                #                         src_coef_summary[term.coef_id()] = computed_coef_summary
                #
                #     self.src_coefficient_summary = tf.stack([src_coef_summary[x] for x in self.src_coef_names], axis=1)
                #
                #     for i in range(len(self.src_coef_names)):
                #         tf.summary.scalar(
                #             'coefficient_src' + '/%s' % self.src_coef_names[i],
                #             self.src_coefficient_summary[0,i],
                #             collections=['params']
                #         )

                self.coefficient = tf.expand_dims(self.coefficient, 0)

                self.ransl = False
                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    mask_row_np = np.ones(self.rangf_n_levels[i], dtype=self.FLOAT_NP)
                    mask_row_np[self.rangf_n_levels[i] - 1] = 0
                    mask_row = tf.constant(mask_row_np, dtype=self.FLOAT_TF)

                    if self.has_intercept[gf]:
                        intercept_random, intercept_random_summary = self.initialize_intercept(ran_gf=gf)
                        intercept_random *= mask_row
                        intercept_random_summary *= mask_row

                        intercept_random_mean = tf.reduce_sum(intercept_random_summary, axis=0) / tf.reduce_sum(mask_row)
                        intercept_random_centering_vector = mask_row * intercept_random_mean

                        intercept_random -= intercept_random_centering_vector
                        intercept_random_summary -= intercept_random_centering_vector
                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                'by_%s/intercept' % gf,
                                intercept_random_summary,
                                collections=['random']
                            )

                    coefs = self.coef_by_rangf.get(gf, [])
                    if len(coefs) > 0:
                        coef_ix = names2ix(coefs, self.coef_names)
                        mask_col_np = np.zeros(len(self.coef_names))
                        mask_col_np[coef_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)
                        self.ransl = True

                        coefficient_random, coefficient_random_summary = self.initialize_coefficient(
                            coef_ids=coef_ids,
                            ran_gf=gf,
                        )

                        coefficient_random *= mask_col
                        coefficient_random_summary *= mask_col
                        coefficient_random *= tf.expand_dims(mask_row, -1)
                        coefficient_random_summary *= tf.expand_dims(mask_row, -1)

                        coefficient_random_mean = tf.reduce_sum(coefficient_random_summary, axis=0) / tf.reduce_sum(mask_row)
                        coefficient_random_centering_vector = tf.expand_dims(mask_row, -1) * coefficient_random_mean

                        coefficient_random -= coefficient_random_centering_vector
                        coefficient_random_summary -= coefficient_random_centering_vector
                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(coefs)):
                                coef_name = coefs[j]
                                ix = coef_ix[j]
                                coef_name = coefs[j]
                                ix = coef_ix[j]
                                tf.summary.histogram(
                                    'by_%s/coefficient/%s' % (gf, coef_name),
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
                    pdf = tf.contrib.distributions.Gamma(concentration=params[:,0:1],
                                                         rate=params[:,1:2],
                                                         validate_args=True).prob
                    return lambda x: pdf(x + self.epsilon)

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['SteepGamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['GammaKgt1'] = gamma

                def shifted_gamma(params):
                    pdf = tf.contrib.distributions.Gamma(concentration=params[:,0:1],
                                                         rate=params[:,1:2],
                                                         validate_args=True).prob
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

    def _initialize_irf_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for family in self.atomic_irf_names_by_family:
                    if family == 'DiracDelta':
                        continue

                    irf_ids = self.atomic_irf_names_by_family[family]
                    irf_param_init = self.atomic_irf_param_init_by_family[family]
                    irf_param_trainable = self.atomic_irf_param_trainable_by_family[family]

                    irf_by_rangf = {}
                    for id in irf_ids:
                        for gf in self.irf_by_rangf:
                            if id in self.irf_by_rangf[gf]:
                                if gf not in irf_by_rangf:
                                    irf_by_rangf[gf] = []
                                irf_by_rangf[gf].append(id)

                    if family == 'Exp':
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([beta], axis=1)
                        params_summary =  tf.stack([beta_mean], axis=1)
                    elif family == 'ExpRateGT1':
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=2)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=1, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([beta], axis=1)
                        params_summary =  tf.stack([beta_mean], axis=1)
                    elif family == 'Gamma':
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean], axis=1)
                    elif family in ['GammaKgt1', 'GammaShapeGT1']:
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=1, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean], axis=1)
                    elif family == 'SteepGamma':
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=1)
                        beta_init = self._get_mean_init_vector('beta', 25)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean], axis=1)
                    elif family == 'ShiftedGamma':
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                        delta_init = self._get_mean_init_vector('delta', -0.5)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        delta, delta_mean = self.initialize_irf_param('delta', irf_ids, mean=delta_init, ub=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta, delta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean, delta_mean], axis=1)
                    elif family in ['ShiftedGammaKgt1', 'ShiftedGammaShapeGT1']:
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                        delta_init = self._get_mean_init_vector(irf_ids, 'delta', irf_param_init, default=-0.5)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=1, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        delta, delta_mean = self.initialize_irf_param('delta', irf_ids, mean=delta_init, ub=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta, delta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean, delta_mean], axis=1)
                    elif family == 'Normal':
                        mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                        sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                        mu, mu_mean = self.initialize_irf_param('mu', irf_ids, mean=mu_init, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        sigma, sigma_mean = self.initialize_irf_param('sigma', irf_ids, mean=sigma_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([mu, sigma], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean], axis=1)
                    elif family == 'SkewNormal':
                        mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                        sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=0)
                        mu, mu_mean = self.initialize_irf_param('mu', irf_ids, mean=mu_init, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        sigma, sigma_mean = self.initialize_irf_param('sigma', irf_ids, mean=sigma_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, alpha=alpha_init, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([mu, sigma, alpha], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean, alpha_mean], axis=1)
                    elif family == 'EMG':
                        mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                        sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                        mu, mu_mean = self.initialize_irf_param('mu', irf_ids, mean=mu_init, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        sigma, sigma_mean = self.initialize_irf_param('sigma', irf_ids, mean=sigma_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([mu, sigma, beta], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean, beta_mean], axis=1)
                    elif family == 'BetaPrime':
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=1)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean], axis=1)
                    elif family == 'ShiftedBetaPrime':
                        alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=1)
                        beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                        delta_init = self._get_mean_init_vector('delta', -1)
                        alpha, alpha_mean = self.initialize_irf_param('alpha', irf_ids, mean=alpha_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        beta, beta_mean = self.initialize_irf_param('beta', irf_ids, mean=beta_init, lb=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        delta, delta_mean = self.initialize_irf_param('delta', irf_ids, mean=delta_init, ub=0, irf_by_rangf=irf_by_rangf, trainable=irf_param_trainable)
                        params = tf.stack([alpha, beta, delta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean, delta_mean], axis=1)
                    else:
                        raise ValueError('Impulse response function "%s" is not currently supported.' % family)
                    for i in range(len(irf_ids)):
                        id = irf_ids[i]
                        ix = names2ix(id, self.atomic_irf_names_by_family[family])
                        assert id not in self.irf_params, 'Duplicate IRF node name already in self.irf_params'
                        self.irf_params[id] = tf.gather(params, ix, axis=2)
                        self.irf_params_summary[id] = tf.gather(params_summary, ix, axis=2)

    def _get_mean_init_vector(self, irf_ids, param_name, irf_param_init, default=0):
        mean = np.zeros(len(irf_ids))
        for i in range(len(irf_ids)):
            mean[i] = irf_param_init[irf_ids[i]].get(param_name, default)
        return mean

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

                    assert not t.name() in self.mc_integrals, 'Duplicate IRF node name already in self.mc_integrals'
                    if t.p.family == 'DiracDelta':
                        self.mc_integrals[t.name()] = tf.gather(self.coefficient_fixed, coef_ix)
                    else:
                        self.mc_integrals[t.name()] = self._reduce_interpolated_sum(
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

                    atomic_irf = self._new_irf(self.irf_lambdas[t.family], params)
                    atomic_irf_plot = self._new_irf(self.irf_lambdas[t.family], params_summary)
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

                    if len(irf_plot) > 1:
                        composite_irf_mc = irf[0]
                        for p_irf in irf[1:]:
                            composite_irf_mc = self._merge_irf(composite_irf_mc, p_irf, self.t_delta)
                        composite_irf_mc = composite_irf_mc(self.support)[0]

                        composite_irf_plot = irf_plot[0]
                        for p_irf in irf_plot[1:]:
                            composite_irf_plot = self._merge_irf(composite_irf_plot, p_irf, self.t_delta)
                        composite_irf_plot = composite_irf_plot(self.support)[0]

                    else:
                        composite_irf_mc = atomic_irf_mc
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
                            if t.name() in self.mc_integrals:
                                self.src_mc_integrals[src_irf_names[0]] = self.mc_integrals[t.name()]
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
                                        if t.name() in self.mc_integrals:
                                            if src_irf_name in self.src_mc_integrals:
                                                self.src_mc_integrals[src_irf_name] += tf.reduce_sum(self.mc_integrals[t.name()] * e, axis=0, keep_dims=True)
                                            else:
                                                self.src_mc_integrals[src_irf_name] = tf.reduce_sum(self.mc_integrals[t.name()] * e, axis=0, keep_dims=True)

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
                                cur_irf = irf[0]
                                for p_irf in irf[1:]:
                                    cur_irf = self._merge_irf(cur_irf, p_irf, self.t_delta)
                                irf = cur_irf(self.support)[0]
                            else:
                                irf = irf[0]

                            impulse_interp = self._lininterp(impulse, self.n_interp)
                            t_delta_interp = self._lininterp(self.t_delta, self.n_interp)
                            irf_seq = irf(t_delta_interp)

                            self.convolutions[name] = tf.reduce_sum(impulse_interp * irf_seq, axis=1)
                        else:
                            impulse = self.irf_impulses[name]

                            irf = self.irf[name]
                            if len(irf) > 1:
                                cur_irf = irf[0]
                                for p_irf in irf[1:]:
                                    cur_irf = self._merge_irf(cur_irf, p_irf, self.t_delta)
                                irf = cur_irf(self.support)[0]
                            else:
                                irf = irf[0]

                            irf_seq = irf(self.t_delta)

                            self.convolutions[name] = tf.reduce_sum(impulse * irf_seq, axis=1)

    def _construct_network(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.t_delta = tf.expand_dims(tf.expand_dims(self.time_y, -1) - self.time_X, -1)  # Tensor of temporal offsets with shape (?,history_length)
                self._initialize_irfs(self.t)
                self._initialize_impulses()
                self._initialize_convolutions()
                self._initialize_backtransformed_irf_plot(self.t)

                convolutions = [self.convolutions[x] for x in self.terminal_names]
                self.X_conv = tf.concat(convolutions, axis=1)

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
        Add an intercept to the DTSR model.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param ran_gf: ``str`` or ``None``; Name of random grouping factor for random intercept (if ``None``, constructs a fixed intercept)
        :return: 2-tuple of ``Tensor`` ``(intercept, intercept_summary)``; ``intercept`` is the intercept for use by the model. ``intercept_summary`` is an identically-shaped representation of the current intercept value for logging and plotting (can be identical to ``intercept``). For fixed intercepts, should return scalars. For random intercepts, should return batch-length vector of zero-centered random intercept values for each input in the batch.
        """
        raise NotImplementedError

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        """
        Add coefficients to the DTSR model.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of length ``len(coef_ids)``. For random coefficients, should return batch-length matrix of random coefficient values with ``len(coef_ids)`` zero-centered columns for each input in the batch.
        """

        raise NotImplementedError

    def initialize_irf_param(self, param_name, ids, trainable=None, mean=0, lb=None, ub=None, irf_by_rangf=None):
        """
        Add IRF parameters to the DTSR model.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param param_name: ``str``; Name of parameter (e.g. ``"alpha"``)
        :param ids: ``list`` of ``str``; Names of IRF nodes to which this parameter applies
        :param trainable: ``dict`` or ``None``; Keys correspond to IDs in **ids**, each associated with a list of names of trainable parameters. If ``None``, the parameter is treated as trainable for all members of **ids**.
        :param mean: ``float``; Initial value for the parameter.
        :param lb: ``float`` or ``None``; Upper bound on the parameter (if ``None``, no upper bound).
        :param ub: ``float`` or ``None``; Lower bound on the parameter (if ``None``, no lower bound).
        :param irf_by_rangf: ``dict`` or ``None``; Keys correspond to random grouping factors and are associated with a list of IDs (subset of **ids**) -- defines the random IRF structures associated with this parameter. If ``None``, the parameter is treated as fixed.
        :return: 2-tuple of ``Tensor`` ``(param, param_summary)``; ``param`` is the parameter for use by the model. ``param_summary`` is an identically-shaped representation of the current param values for logging and plotting (can be identical to ``param``). For fixed params, should return a vector of length ``len(ids)``. For random params, should return batch-length matrix of **summed** fixed and random param values for each input in the batch.
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
    #  Other private methods
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

    def _merge_irf(self, A, B, t_delta):
        raise ValueError('Hierarchical convolutions are not yet supported.')

        # ode = lambda y, t: A(t) * B(t)
        # A_conv_B = tf.contrib.integrate.odeint(ode, , tf.reverse(self.t_delta))

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

    def _process_mean(self, mean, lb=None, ub=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                mean = tf.constant(mean, dtype=self.FLOAT_TF)
                if lb is not None:
                    lb = tf.constant(lb, dtype=self.FLOAT_TF)
                if ub is not None:
                    ub = tf.constant(ub, dtype=self.FLOAT_TF)

                if lb is None and ub is None:
                    # Unbounded support
                    mean = mean
                elif lb is not None and ub is None:
                    # Lower-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(mean - lb + self.epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(-(mean - ub + self.epsilon))
                else:
                    # Finite-interval bounded support
                    mean = tf.contrib.distributions.bijectors.Sigmoid.inverse(
                        (mean - lb + self.epsilon) / ((ub - self.epsilon) - (lb + self.epsilon))
                    )
        return mean, lb, ub

    def _collect_plots(self):
        switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.plots = {}
                irf_names = [x for x in self.node_table if x in self.irf_plot and not (len(self.node_table[x].children) == 1 and self.node_table[x].children[0].terminal())]

                for a in switches[0]:
                    if a not in self.plots:
                        self.plots[a] = {}
                    for b in switches[1]:
                        plot_y = []
                        for x in irf_names:
                            plot_y.append(self.irf_plot[x][a][b])

                        self.plots[a][b] = {
                            'names': irf_names,
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
                            for x in irf_names:
                                plot_y.append(self.src_irf_plot[x][a][b])

                            self.src_plot_tensors[a][b] = {
                                'names': irf_names,
                                'plot': plot_y
                            }

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

    def _regularize(self, var, center=None):
        if self.regularizer_name is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if center is None:
                        reg = tf.contrib.layers.apply_regularization(self.regularizer, [var])
                    else:
                        reg = tf.contrib.layers.apply_regularization(self.regularizer, [var - center])
                    self.regularizer_losses.append(reg)

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

    def _lininterp(self, x, n):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_input = tf.shape(x)[1]
                n_output = n_input * (n+1)
                interp = tf.image.resize_bilinear(tf.expand_dims(tf.expand_dims(x, -1), -1), [n_output, 1])
                interp = tf.squeeze(tf.squeeze(interp, -1), -1)[..., :-n]
                return interp

    def _lininterp_2(self, x, time, hz):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                time = tf.round(time * hz)
                time_ix = tf.expand_dims(tf.cast(time, dtype=self.INT_TF), -1)
                n = tf.reduce_max(time_ix)

                time_shape = (tf.shape(time), n)
                time_new = tf.scatter_nd(time_ix, time, time_shape)

                x_shape = (tf.shape(x)[0], tf.shape(x)[1], n)

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

    def run_predict_op(self, feed_dict, n_samples=None):
        """
        Generate predictions from a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor values.
        :param n_samples: ``int`` or ``None``; Number of posterior samples to use (ignored by DTSRMLE)
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """
        raise NotImplementedError

    def run_loglik_op(self, feed_dict, n_samples=None):
        """
        Compute the log-likelihoods of a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :param n_samples: ``int`` or ``None``; Number of posterior samples to use (ignored by DTSRMLE)
        :return: ``numpy`` array; Pointwise log-likelihoods, one for each training sample
        """

        raise NotImplementedError

    def run_conv_op(self, feed_dict, scaled=False, n_samples=None):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor variables
        :param scaled: ``bool``; Whether to scale the outputs using the model's coefficients
        :param n_samples: ``int`` or ``None``; Number of posterior samples to use (ignored by DTSRMLE)
        :return: ``numpy`` array; The convolved inputs
        """

        raise NotImplementedError

    def summary(self, fixed=True, random=False):
        """
        Generate a summary of the fitted model.
        **All DTSR subclasses must implement this method.**

        :param fixed: ``bool``; Report fixed effects parameters
        :param random: ``bool``; Report random effects parameters
        :return: ``str``; Formatted model summary for printing
        """

        raise NotImplementedError





    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.load(predict=mode)

    def report_n_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                sys.stderr.write('Trainable variables:\n')
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = np.prod(np.array(v_val).shape)
                    n_params += cur_params
                    sys.stderr.write('  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params))
                sys.stderr.write('Network contains %d total trainable parameters.\n' % n_params)
                sys.stderr.write('\n')

    def build(self, outdir=None, restore=True, verbose=True):
        """
        Construct the DTSR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Show the model tree when called.
        :return: ``None``
        """
        if verbose:
            sys.stderr.write('Constructing network from model tree:\n')
            sys.stdout.write(str(self.t))
            sys.stdout.write('\n\n')

            if self.pc:
                sys.stderr.write('Source model tree:\n')
                sys.stdout.write(str(self.t_src))
                sys.stdout.write('\n\n')

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './dtsr_model/'
        else:
            self.outdir = outdir

        self.reset_training_data()
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs()
                self._initialize_intercepts_coefficients()
                self._initialize_irf_lambdas()
                self._initialize_irf_params()
                self._construct_network()
                self.initialize_objective()
                self._initialize_logging()
                self._initialize_ema()
                self._initialize_saver()
                self.load(restore=restore)

                self._collect_plots()
                self.report_n_params()

    def save(self, dir=None):
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

    def load(self, dir=None, predict=False, restore=True):
        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if restore and os.path.exists(dir + '/checkpoint'):
                    try:
                        if predict:
                            self.ema_saver.restore(self.sess, dir + '/model.ckpt')
                        else:
                            self.saver.restore(self.sess, dir + '/model.ckpt')
                    except:
                        if predict:
                            self.ema_saver.restore(self.sess, dir + '/model_backup.ckpt')
                        else:
                            self.saver.restore(self.sess, dir + '/model_backup.ckpt')
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')
                    self.sess.run(tf.global_variables_initializer())

    def assign_training_data(self, X, time_X, y, time_y, gf_y):
        self.X_in = X
        self.time_X_in = time_X
        self.y_in = y
        self.time_y_in = time_y
        self.gf_y_in = gf_y

    def reset_training_data(self):
        self.X_in = None
        self.time_X_in = None
        self.y_in = None
        self.time_y_in = None
        self.gf_y_in = None

    def data_generator(self):
        j = 0
        while j < self.y_in.shape[0]:
            yield (
                self.X_in[j],
                self.time_X_in[j],
                self.y_in[j],
                self.time_y_in[j],
                self.gf_y_in[j]
            )
            j += 1

    def expand_history(self, X, X_time, first_obs, last_obs):
        last_obs = np.array(last_obs, dtype=self.INT_NP)
        first_obs = np.maximum(np.array(first_obs, dtype=self.INT_NP), last_obs - self.history_length)
        X_time = np.array(X_time, dtype=self.FLOAT_NP)
        X = np.array(X, dtype=self.FLOAT_NP)

        X_history = np.zeros((first_obs.shape[0], self.history_length, X.shape[1]))
        time_X_history = np.zeros((first_obs.shape[0], self.history_length))

        for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
            sX = X[first:last]
            sXt = X_time[first:last]
            X_history[i, -sX.shape[0]:] += sX
            time_X_history[i][-len(sXt):] += sXt

        return X_history, time_X_history

    def ci_curve(
            self,
            posterior,
            level=95,
            n_samples=1024,
            n_time_units=2.5,
            n_points_per_time_unit=1000
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.support_start: 0.,
                    self.n_time_units: n_time_units,
                    self.n_points_per_time_unit: n_points_per_time_unit,
                    self.gf_y: np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                }

                alpha = 100-float(level)

                samples = [self.sess.run(posterior, feed_dict=fd) for _ in range(n_samples)]
                samples = np.concatenate(samples, axis=1)

                mean = samples.mean(axis=1)
                lower = np.percentile(samples, alpha/2, axis=1)
                upper = np.percentile(samples, 100-(alpha/2), axis=1)

                return (mean, lower, upper)

    def ci_integral(
            self,
            terminal_name,
            level=95,
            n_samples=1024,
            n_time_units=2.5,
            n_points_per_time_unit=1000
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.support_start: 0.,
                    self.n_time_units: n_time_units,
                    self.n_points_per_time_unit: n_points_per_time_unit,
                    self.gf_y: np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                }

                alpha = 100 - float(level)

                if terminal_name in self.mc_integrals:
                    posterior = self.mc_integrals[terminal_name]
                else:
                    posterior = self.src_mc_integrals[terminal_name]

                samples = [np.squeeze(self.sess.run(posterior, feed_dict=fd)[0]) for _ in range(n_samples)]
                samples = np.stack(samples, axis=0)

                mean = samples.mean(axis=0)
                lower = np.percentile(samples, alpha / 2, axis=0)
                upper = np.percentile(samples, 100 - (alpha / 2), axis=0)

                return (mean, lower, upper)

    def finalize(self):
        """
        Close the DTSR instance to prevent memory leaks.

        :return: ``None``
        """
        self.sess.close()


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
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            irf_name_map=None,
            plot_n_time_units=2.5,
            plot_n_points_per_time_unit=500,
            plot_x_inches=28,
            plot_y_inches=5,
            cmap='gist_rainbow'):
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
        :param n_epoch_train: ``int``; the number of training iterations
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_n_time_units: ``float``; number if time units to use in plotting.
        :param plot_n_points_per_time_unit: ``float``; number of plot points to use per time unit.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """

        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)
        sys.stderr.write('Number of training samples: %d\n\n' % len(y))

        if self.pc:
            impulse_names = self.src_impulse_names
            assert X_2d_predictors is None, 'Principal components regression not support for models with 3d predictors'
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

        X_2d, time_X_2d, time_mask = build_DTSR_impulses(
            X,
            y.first_obs,
            y.last_obs,
            impulse_names,
            history_length=128,
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
        rho = corr_dtsr(X_2d, impulse_names, impulse_names_2d, time_mask)
        sys.stderr.write(str(rho) + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():

                fd = {
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.y: y_dv,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                if self.global_step.eval(session=self.sess) == 0:
                    summary_params = self.sess.run(self.summary_params)
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    if self.log_random and len(self.rangf) > 0:
                        summary_random = self.sess.run(self.summary_random)
                        self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))

                while self.global_step.eval(session=self.sess) < n_iter:
                    try:
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
                                plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                cmap=cmap
                            )
                            if type(self).__name__ == 'DTSRBayes':
                                lb = self.sess.run(self.err_dist_lb)
                                ub = self.sess.run(self.err_dist_ub)
                                n_time_units = ub-lb
                                fd_plot = {
                                    self.support_start: lb,
                                    self.n_time_units: n_time_units,
                                    self.n_points_per_time_unit: 500
                                }
                                plot_x = self.sess.run(self.support, feed_dict=fd_plot)
                                plot_y = self.sess.run(self.err_dist_plot, feed_dict=fd_plot)
                                plot_irf(
                                    plot_x,
                                    plot_y,
                                    ['Error Distribution'],
                                    dir=self.outdir,
                                    filename='error_distribution.png',
                                    legend=False,
                                )
                        t1_iter = pytime.time()
                        sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                    except tf.errors.InvalidArgumentError as err:
                        sys.stderr.write('Encountered numerical instability during inference.\nRestarting from the most recent checkpoint.\nError details:\n%s' %err)
                        self.load()

                self.save()

                self.make_plots(
                    irf_name_map=irf_name_map,
                    plot_n_time_units=plot_n_time_units,
                    plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                    plot_x_inches=plot_x_inches,
                    plot_y_inches=plot_y_inches,
                    cmap=cmap
                )

                if type(self).__name__ == 'DTSRBayes':
                    # Generate plots with 95% credible intervals
                    self.make_plots(
                        irf_name_map=irf_name_map,
                        plot_n_time_units=plot_n_time_units,
                        plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                        plot_x_inches=plot_x_inches,
                        plot_y_inches=plot_y_inches,
                        cmap=cmap,
                        mc=True
                    )
                else:
                    # Compute and store the model variance for log likelihood computation
                    loss_total = 0.
                    for j in range(0, len(y), self.minibatch_size):
                        fd_minibatch[self.y] = y_dv[j:j + minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + minibatch_size]
                        fd_minibatch[self.X] = X_2d[j:j + minibatch_size]
                        fd_minibatch[self.time_X] = time_X_2d[j:j + minibatch_size]
                        loss_total += self.sess.run(self.mse_loss, feed_dict=fd_minibatch) * len(fd_minibatch[self.y])
                    loss_total /= len(y)
                    self.sess.run(self.set_y_scale, feed_dict={self.loss_total: math.sqrt(loss_total)})
                sys.stderr.write('%s\n\n' % ('='*50))

    def predict(
            self,
            X,
            y_time,
            y_rangf,
            first_obs,
            last_obs,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None
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
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert len(y_time) == len(y_rangf) == len(first_obs) == len(last_obs), 'y_time, y_rangf, first_obs, and last_obs must be of identical length. Got: len(y_time) = %d, len(y_rangf) = %d, len(first_obs) = %d, len(last_obs) = %d' % (len(y_time), len(y_rangf), len(first_obs), len(last_obs))

        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        sys.stderr.write('Sampling per-datum predictions/errors from posterior predictive distribution...\n')

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_2d, time_X_2d, time_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            history_length=128,
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
                    self.gf_y: gf_y,
                }


                if not np.isfinite(self.eval_minibatch_size):
                    preds = self.run_predict_op(fd, n_samples=n_samples)
                else:
                    preds = np.zeros((len(y_time),))
                    n_eval_minibatch = math.ceil(len(y_time) / self.eval_minibatch_size)
                    for i in range(0, len(y_time), self.eval_minibatch_size):
                        sys.stderr.write('\rMinibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                        sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y
                        }
                        preds[i:i + self.eval_minibatch_size] = self.run_predict_op(fd_minibatch, n_samples=n_samples)

                sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return preds

    def log_lik(
            self,
            X,
            y,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None
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

        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        sys.stderr.write('Sampling per-datum likelihoods from posterior predictive distribution...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_2d, time_X_2d, time_mask = build_DTSR_impulses(
            X,
            y['first_obs'],
            y['last_obs'],
            impulse_names,
            history_length=128,
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
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.y: y_dv
                }


                if not np.isfinite(self.eval_minibatch_size):
                    log_lik = self.run_loglik_op(fd, n_samples=n_samples)
                else:
                    log_lik = np.zeros((len(time_y),))
                    n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                    for i in range(0, len(time_y), self.eval_minibatch_size):
                        sys.stderr.write('\rMinibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                        sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.eval_minibatch_size]
                        }
                        log_lik[i:i+self.eval_minibatch_size] = self.run_loglik_op(fd_minibatch, n_samples=n_samples)

                self.set_predict_mode(False)

                sys.stderr.write('\n\n')

                return log_lik

    def convolve_inputs(
            self,
            X,
            y,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            scaled=False,
            n_samples=None
    ):

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

        X_2d, time_X_2d, time_mask = build_DTSR_impulses(
            X,
            y['first_obs'],
            y['last_obs'],
            impulse_names,
            history_length=128,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        if isinstance(X, pd.DataFrame):
            X_2d, time_X_2d = self.expand_history(X[impulse_names], X.time, y.first_obs, y.last_obs)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.X: X_2d,
                    self.time_X: time_X_2d
                }

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                X_conv = []
                for j in range(0, len(y), self.eval_minibatch_size):
                    fd_minibatch[self.time_y] = time_y[j:j + self.eval_minibatch_size]
                    fd_minibatch[self.gf_y] = gf_y[j:j + self.eval_minibatch_size]
                    fd_minibatch[self.X] = X_2d[j:j + self.eval_minibatch_size]
                    fd_minibatch[self.time_X] = time_X_2d[j:j + self.eval_minibatch_size]
                    X_conv_cur = self.run_conv_op(fd_minibatch, scaled=scaled, n_samples=n_samples)
                    X_conv.append(X_conv_cur)
                names = [sn(''.join(x.split('-')[:-1])) for x in self.terminal_names]
                X_conv = np.concatenate(X_conv, axis=0)
                out = pd.DataFrame(X_conv, columns=names, dtype=self.FLOAT_NP)
                for c in y.columns:
                    if c not in out:
                        out[c] = y[c]

                self.set_predict_mode(False)

                convolution_summary = ''
                corr_conv = out.corr()
                convolution_summary += '=' * 50 + '\n'
                convolution_summary += 'Correlation matrix of convolved predictors:\n\n'
                convolution_summary += str(corr_conv) + '\n\n'

                select = np.where(np.isclose(time_X_2d[:,-1], time_y))[0]

                X_input = X_2d[:,-1,:][select]

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
            plot_n_time_units=2.5,
            plot_n_points_per_time_unit=1000,
            plot_x_inches=7.,
            plot_y_inches=5.,
            cmap=None,
            mc=False,
            n_samples=1000,
            level=95,
            prefix='',
            legend=True,
            xlab=None,
            ylab=None,
            transparent_background=False
    ):
        """
        Generate plots of current state of deconvolution.
        Saves four plots to the output directory:

            * ``irf_atomic_scaled.jpg``: One line for each IRF kernel in the model (ignoring preconvolution in any composite kernels), scaled by the relevant coefficients
            * ``irf_atomic_unscaled.jpg``: One line for each IRF kernel in the model (ignoring preconvolution in any composite kernels), unscaled
            * ``irf_composite_scaled.jpg``: One line for each IRF kernel in the model (including preconvolution in any composite kernels), scaled by the relevant coefficients
            * ``irf_composite_unscaled.jpg``: One line for each IRF kernel in the model (including preconvolution in any composite kernels), unscaled

        If the model contains no composite IRF, corresponding atomic and composite plots will be identical.

        To save space, successive calls to ``make_plots()`` overwrite existing plots.
        Thus, plots only show the most recently plotted state of learning.

        For simplicity, plots for DTSRBayes models use the posterior mean, abstracting away from other characteristics of the posterior distribution (e.g. variance).

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """

        if prefix != '':
            prefix += '_'
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: plot_n_time_units,
                    self.n_points_per_time_unit: plot_n_points_per_time_unit,
                    self.gf_y: np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                }

                plot_x = self.sess.run(self.support, fd)

                switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

                for a in switches[0]:
                    for b in switches[1]:
                        plot_name = 'irf_%s_%s.png' %(a, b)
                        names = self.plots[a][b]['names']
                        if mc:
                            plot_y = []
                            lq = []
                            uq = []
                            for name in names:
                                mean_cur, lq_cur, uq_cur = self.ci_curve(
                                    self.irf_mc[name][a][b],
                                    n_time_units=plot_n_time_units,
                                    n_points_per_time_unit=plot_n_points_per_time_unit,
                                )
                                plot_y.append(mean_cur)
                                lq.append(lq_cur)
                                uq.append(uq_cur)
                            lq = np.stack(lq, axis=1)
                            uq = np.stack(uq, axis=1)
                            plot_y = np.stack(plot_y, axis=1)
                            plot_name = 'mc_' + plot_name
                        else:
                            plot_y = [self.sess.run(x, feed_dict=fd) for x in self.plots[a][b]['plot']]
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
                        for b in switches[1]:
                            if b == 'scaled':
                                plot_name = 'src_irf_%s_%s.png' % (a, b)
                                names = self.src_plot_tensors[a][b]['names']
                                if mc:
                                    plot_y = []
                                    lq = []
                                    uq = []
                                    for name in names:
                                        mean_cur, lq_cur, uq_cur = self.ci_curve(self.src_irf_mc[name][a][b])
                                        plot_y.append(mean_cur)
                                        lq.append(lq_cur)
                                        uq.append(uq_cur)
                                    lq = np.stack(lq, axis=1)
                                    uq = np.stack(uq, axis=1)
                                    plot_y = np.stack(plot_y, axis=1)
                                    plot_name = 'mc_' + plot_name
                                else:
                                    plot_y = [self.sess.run(x, feed_dict=fd) for x in self.src_plot_tensors[a][b]['plot']]
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


