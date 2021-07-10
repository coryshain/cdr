import os
import re
import itertools
import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf

from .kwargs import CDRNN_INITIALIZATION_KWARGS
from .backend import *
from .base import Model
from .util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pd.options.mode.chained_assignment = None


class CDRNN(Model):
    _INITIALIZATION_KWARGS = CDRNN_INITIALIZATION_KWARGS


    _doc_header = """
        Abstract base class for CDRNN. Bayesian (:ref:`cdrnnbayes`) and MLE (:ref:`cdrnnmle`) implementations inherit from ``CDRNN``.
        ``CDRNN`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``CDRNN`` must implement the following instance methods:

            - initialize_input_bias()
            - initialize_feedforward()
            - initialize_rnn()
            - initialize_rnn_h()
            - initialize_rnn_c()
            - initialize_h_bias()
            - initialize_h_normalization()
            - initialize_intercept_l1_weights()
            - initialize_intercept_l1_biases()
            - initialize_intercept_l1_normalization()
            - initialize_irf_l1_weights()
            - initialize_irf_l1_biases()
            - initialize_irf_l1_normalization()
            - initialize_error_params_biases()

        Additionally, if the subclass requires any keyword arguments beyond those provided by ``CDRNN``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.

        Example implementations of each of these methods can be found in the source code for the ``CDRNNMLE`` and ``CDRNNBayes`` classes.

    """
    _doc_args = """
        :param form_str: An R-style string representing the CDRNN model formula.
        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must containcontain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the model ``form_str`` provided at initialization
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
        if cls is CDRNN:
            raise TypeError("CDRNN is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, Y, **kwargs):
        super(CDRNN, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDRNN._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

    def _initialize_metadata(self):
        self.is_cdrnn = True

        self.has_dropout = self.input_projection_dropout_rate or \
                           self.rnn_h_dropout_rate or \
                           self.rnn_c_dropout_rate or \
                           self.h_in_dropout_rate or \
                           self.h_rnn_dropout_rate or \
                           self.rnn_dropout_rate or \
                           self.irf_dropout_rate or \
                           self.ranef_dropout_rate

        super(CDRNN, self)._initialize_metadata()

        self.use_batch_normalization = bool(self.batch_normalization_decay)
        self.use_layer_normalization = bool(self.layer_normalization_type)

        assert not (self.use_batch_normalization and self.use_layer_normalization), 'Cannot batch normalize and layer normalize the same model.'

        self.normalize_activations = self.use_batch_normalization or self.use_layer_normalization

        # self.impulse_vectors_train['rate'] = np.ones((self.n_train,))

        # Initialize tree metadata
        self.t = self.form.t
        t = self.t
        self.node_table = t.node_table()
        self.impulse_names = t.impulse_names(include_interactions=True)
        self.terminal_names = t.terminal_names()

        # CDRNN can't use any of the IRF formula components.
        # To help users, check to make sure formula doesn't contain any IRF stuff, since it will be ignored anyway.
        irf_families = list(t.atomic_irf_by_family().keys())
        assert len(irf_families) < 2, 'CDRNN does not support parametric impulse response functions.'
        if len(irf_families) == 1:
            assert irf_families[0] == 'DiracDelta', 'CDRNN does not support parametric impulse response functions.'

        if self.n_units_input_projection:
            if isinstance(self.n_units_input_projection, str):
                if self.n_units_input_projection.lower() == 'infer':
                    self.n_units_input_projection = [len(self.impulse_names) + len(self.ablated) + 1]
                else:
                    self.n_units_input_projection = [int(x) for x in self.n_units_input_projection.split()]
            elif isinstance(self.n_units_input_projection, int):
                if self.n_layers_input_projection is None:
                    self.n_units_input_projection = [self.n_units_input_projection]
                else:
                    self.n_units_input_projection = [self.n_units_input_projection] * self.n_layers_input_projection
            if self.n_layers_input_projection is None:
                self.n_layers_input_projection = len(self.n_units_input_projection)
            if len(self.n_units_input_projection) == 1 and self.n_layers_input_projection != 1:
                self.n_units_input_projection = [self.n_units_input_projection[0]] * self.n_layers_input_projection
                self.n_layers_input_projection = len(self.n_units_input_projection)
        else:
            self.n_units_input_projection = []
            self.n_layers_input_projection = 0
        assert self.n_layers_input_projection == len(self.n_units_input_projection), 'Inferred n_layers_input_projection and n_units_input_projection must have the same number of layers. Saw %d and %d, respectively.' % (self.n_layers_input_projection, len(self.n_units_input_projection))

        if self.n_units_rnn:
            if isinstance(self.n_units_rnn, str):
                if self.n_units_rnn.lower() == 'infer':
                    self.n_units_rnn = [len(self.impulse_names) + len(self.ablated) + 1]
                elif self.n_units_rnn.lower() == 'inherit':
                    self.n_units_rnn = ['inherit']
                else:
                    self.n_units_rnn = [int(x) for x in self.n_units_rnn.split()]
            elif isinstance(self.n_units_rnn, int):
                if self.n_layers_rnn is None:
                    self.n_units_rnn = [self.n_units_rnn]
                else:
                    self.n_units_rnn = [self.n_units_rnn] * self.n_layers_rnn
            if self.n_layers_rnn is None:
                self.n_layers_rnn = len(self.n_units_rnn)
            if len(self.n_units_rnn) == 1 and self.n_layers_rnn != 1:
                self.n_units_rnn = [self.n_units_rnn[0]] * self.n_layers_rnn
                self.n_layers_rnn = len(self.n_units_rnn)
        else:
            self.n_units_rnn = []
            self.n_layers_rnn = 0
        assert self.n_layers_rnn == len(self.n_units_rnn), 'Inferred n_layers_rnn and n_units_rnn must have the same number of layers. Saw %d and %d, respectively.' % (self.n_layers_rnn, len(self.n_units_rnn))

        if self.n_units_rnn_projection:
            if isinstance(self.n_units_rnn_projection, str):
                self.n_units_rnn_projection = [int(x) for x in self.n_units_rnn_projection.split()]
            elif isinstance(self.n_units_rnn_projection, int):
                if self.n_layers_rnn_projection is None:
                    self.n_units_rnn_projection = [self.n_units_rnn_projection]
                else:
                    self.n_units_rnn_projection = [self.n_units_rnn_projection] * self.n_layers_rnn_projection
            if self.n_layers_rnn_projection is None:
                self.n_layers_rnn_projection = len(self.n_units_rnn_projection)
            if len(self.n_units_rnn_projection) == 1 and self.n_layers_rnn_projection != 1:
                self.n_units_rnn_projection = [self.n_units_rnn_projection[0]] * self.n_layers_rnn_projection
                self.n_layers_rnn_projection = len(self.n_units_rnn_projection)
        else:
            self.n_units_rnn_projection = []
            self.n_layers_rnn_projection = 0
        assert self.n_layers_rnn_projection == len(self.n_units_rnn_projection), 'Inferred n_layers_rnn_projection and n_units_rnn_projection must have the same number of layers. Saw %d and %d, respectively.' % (self.n_layers_rnn_projection, len(self.n_units_rnn_projection))

        if self.n_units_irf:
            if isinstance(self.n_units_irf, str):
                self.n_units_irf = [int(x) for x in self.n_units_irf.split()]
            elif isinstance(self.n_units_irf, int):
                if self.n_layers_irf is None:
                    self.n_units_irf = [self.n_units_irf]
                else:
                    self.n_units_irf = [self.n_units_irf] * self.n_layers_irf
            if self.n_layers_irf is None:
                self.n_layers_irf = len(self.n_units_irf)
            if len(self.n_units_irf) == 1 and self.n_layers_irf != 1:
                self.n_units_irf = [self.n_units_irf[0]] * self.n_layers_irf
                self.n_layers_irf = len(self.n_units_irf)
        else:
            self.n_units_irf = []
            self.n_layers_irf = 0
        assert self.n_layers_irf == len(self.n_units_irf), 'Inferred n_layers_irf and n_units_irf must have the same number of layers. Saw %d and %d, respectively.' % (self.n_layers_irf, len(self.n_units_irf))

        if self.n_units_irf:
            self.n_units_irf_l1 = self.n_units_irf[0]
            self.irf_l1_use_bias = True
        else:
            self.n_units_irf_l1 = 1
            self.irf_l1_use_bias = False

        if self.n_units_hidden_state is None:
            if self.n_units_irf:
                self.n_units_hidden_state = self.n_units_irf[0]
            elif self.n_units_input_projection:
                self.n_units_hidden_state = self.n_units_input_projection[-1]
            elif self.n_units_rnn and self.n_units_rnn[-1] != 'inherit':
                self.n_units_hidden_state = self.n_units_rnn[-1]
            else:
                raise ValueError("Cannot infer size of hidden state. Units are not specified for hidden state, IRF, input projection, or RNN projection.")
        elif isinstance(self.n_units_hidden_state, str):
            if self.n_units_hidden_state.lower() == 'infer':
                self.n_units_hidden_state = len(self.impulse_names) + len(self.ablated) + 1
            else:
                self.n_units_hidden_state = int(self.n_units_hidden_state)

        if self.n_units_rnn and self.n_units_rnn[-1] == 'inherit':
            self.n_units_rnn = [self.n_units_hidden_state]

        self.layers = []

    def _pack_metadata(self):
        md = super(CDRNN, self)._pack_metadata()
        for kwarg in CDRNN._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(CDRNN, self)._unpack_metadata(md)

        for kwarg in CDRNN._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
        super(CDRNN, self)._initialize_inputs()
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.rangf):
                    self.use_rangf = True
                    rangf_1hot = []
                    for i in range(len(self.rangf)):
                        rangf_1hot_cur = tf.one_hot(
                            self.Y_gf[:, i],
                            tf.cast(self.rangf_n_levels[i], dtype=self.INT_TF),
                            dtype=self.FLOAT_TF
                        )[:, :-1]
                        rangf_1hot.append(rangf_1hot_cur)
                    if len(rangf_1hot) > 0:
                        rangf_1hot = tf.concat(rangf_1hot, axis=-1)
                    else:
                        rangf_1hot = tf.zeros(
                            [tf.shape(self.Y_gf)[0], 0],
                            dtype=self.FLOAT_TF
                        )
                    self.rangf_1hot = rangf_1hot
                else:
                    self.use_rangf = False

                self.n_surface_plot_points = tf.placeholder_with_default(
                    tf.cast(self.interp_hz, self.FLOAT_TF),
                    shape=[],
                    name='n_surface_plot_points'
                )

                self.n_surface_plot_points_per_side = tf.cast(
                    tf.ceil(
                        tf.sqrt(
                            tf.cast(
                                self.n_surface_plot_points,
                                dtype=self.FLOAT_TF
                            )
                        )
                    ),
                    dtype=self.INT_TF
                )

                self.t_interaction = tf.placeholder_with_default(
                    tf.cast(0., self.FLOAT_TF),
                    shape=[],
                    name='t_interaction'
                )

                self.n_surface_plot_points_normalized = self.n_surface_plot_points_per_side ** 2

                self.plot_n_sds = tf.placeholder_with_default(
                    tf.cast(2., self.FLOAT_TF),
                    shape=[],
                    name='plot_n_sds'
                )

                self.plot_mean_as_reference_tf = tf.placeholder_with_default(
                    self.default_reference_type == 'mean',
                    shape=[],
                    name='plot_mean_as_reference'
                )

                def get_plot_reference_mean():
                    plot_impulse_base_default = tf.convert_to_tensor(
                        self.reference_arr,
                        dtype=self.FLOAT_TF
                    )

                    return plot_impulse_base_default

                def get_plot_reference_zero():
                    plot_impulse_base_default = tf.zeros(
                        [len(self.impulse_names)],
                        dtype=self.FLOAT_TF
                    )

                    return plot_impulse_base_default

                plot_impulse_base_default = tf.cond(
                    self.plot_mean_as_reference_tf,
                    get_plot_reference_mean,
                    get_plot_reference_zero
                )

                plot_impulse_offset_default = tf.ones(
                    [len(self.impulse_names)],
                    dtype=self.FLOAT_TF
                ) * self.plot_step_arr

                plot_impulse_min_default = tf.ones(
                    [len(self.impulse_names)],
                    dtype=self.FLOAT_TF
                )
                plot_impulse_max_default = tf.ones(
                    [len(self.impulse_names)],
                    dtype=self.FLOAT_TF
                )

                ix = self.PLOT_QUANTILE_IX
                lq = self.impulse_quantiles_arr[ix]
                uq = self.impulse_quantiles_arr[self.N_QUANTILES - ix - 1]
                select = np.isclose(uq - lq, 0)
                while ix > 1 and np.any(select):
                    ix -= 1
                    lq[select] = self.impulse_quantiles_arr[ix][select]
                    uq[select] = self.impulse_quantiles_arr[self.N_QUANTILES - ix - 1][select]
                    select = np.isclose(uq - lq, 0)

                plot_impulse_min_default *= lq
                plot_impulse_max_default *= uq

                self.plot_impulse_base = tf.placeholder_with_default(
                    plot_impulse_base_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_base'
                )
                self.plot_impulse_offset = tf.placeholder_with_default(
                    plot_impulse_offset_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_offset'
                )
                self.plot_impulse_min = tf.placeholder_with_default(
                    plot_impulse_min_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_min'
                )
                self.plot_impulse_max = tf.placeholder_with_default(
                    plot_impulse_max_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_max'
                )
                self.plot_impulse_1hot = tf.placeholder_with_default(
                    tf.zeros([len(self.impulse_names)], dtype=self.FLOAT_TF),
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_1hot'
                )
                self.plot_impulse_1hot_2 = tf.placeholder_with_default(
                    tf.zeros([len(self.impulse_names)], dtype=self.FLOAT_TF),
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_1hot_2'
                )

                self.plot_impulse_base_expanded = self.plot_impulse_base[None, None, ...]
                self.plot_impulse_offset_expanded = self.plot_impulse_offset[None, None, ...]
                self.plot_impulse_min_expanded = self.plot_impulse_min[None, None, ...]
                self.plot_impulse_max_expanded = self.plot_impulse_max[None, None, ...]
                self.plot_impulse_1hot_expanded = self.plot_impulse_1hot[None, None, ...]
                self.plot_impulse_1hot_2_expanded = self.plot_impulse_1hot_2[None, None, ...]

                self.nn_regularizer = self._initialize_regularizer(
                    self.nn_regularizer_name,
                    self.nn_regularizer_scale
                )

                if self.context_regularizer_name is None:
                    self.context_regularizer = None
                elif self.context_regularizer_name == 'inherit':
                    self.context_regularizer = self.regularizer
                else:
                    scale = self.context_regularizer_scale / (self.history_length * max(1, len(self.impulse_indices))) # Average over time
                    if self.scale_regularizer_with_data:
                         scale *= self.minibatch_scale # Sum over batch, multiply by n batches
                    else:
                        scale /= self.minibatch_size # Mean over batch
                    if self.context_regularizer_name == 'l1_l2_regularizer':
                        self.context_regularizer = getattr(tf.contrib.layers, self.context_regularizer_name)(
                            scale,
                            scale
                        )
                    else:
                        self.context_regularizer = getattr(tf.contrib.layers, self.context_regularizer_name)(scale)

                self.regularizable_layers = []

    def _initialize_base_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                super(CDRNN, self)._initialize_base_params()
                
                self.rnn_h_random_base = {}
                self.rnn_h_random_base_summary = {}
                self.rnn_c_random_base = {}
                self.rnn_c_random_base_summary = {}
                self.h_bias_random_base = {}
                self.h_bias_random_base_summary = {}
                self.irf_l1_W_random_base = {}
                self.irf_l1_W_random_base_summary = {}
                self.irf_l1_b_random_base = {}
                self.irf_l1_b_random_base_summary = {}

                for i, gf in enumerate(self.rangf):
                    if self.has_intercept[gf]:
                        if gf not in self.rnn_h_random_base:
                            self.rnn_h_random_base[gf] = []
                        if gf not in self.rnn_h_random_base_summary:
                            self.rnn_h_random_base_summary[gf] = []
                        if gf not in self.rnn_c_random_base:
                            self.rnn_c_random_base[gf] = []
                        if gf not in self.rnn_c_random_base_summary:
                            self.rnn_c_random_base_summary[gf] = []

                        # Random RNN initialization offsets
                        for l in range(self.n_layers_rnn):
                            # RNN hidden state
                            rnn_h_random, rnn_h_random_summary = self.initialize_rnn_h(l, ran_gf=gf)
                            self.rnn_h_random_base[gf].append(rnn_h_random)
                            self.rnn_h_random_base_summary[gf].append(rnn_h_random_summary)
                            
                            # RNN cell state
                            rnn_c_random, rnn_c_random_summary = self.initialize_rnn_c(l, ran_gf=gf)
                            self.rnn_c_random_base[gf].append(rnn_c_random)
                            self.rnn_c_random_base_summary[gf].append(rnn_c_random_summary)

                        # CDRNN hidden state
                        h_bias_random, h_bias_random_summary = self.initialize_h_bias(ran_gf=gf)
                        self.h_bias_random_base[gf] = h_bias_random
                        self.h_bias_random_base_summary[gf] = h_bias_random_summary

                        # IRF L1 weights
                        irf_l1_W_random, irf_l1_W_random_summary, = self.initialize_irf_l1_weights(ran_gf=gf)
                        self.irf_l1_W_random_base[gf] = irf_l1_W_random
                        self.irf_l1_W_random_base_summary[gf] = irf_l1_W_random_summary

                        # IRF L1 biases
                        if self.irf_l1_use_bias:
                            irf_l1_b_random, irf_l1_b_random_summary, = self.initialize_irf_l1_biases(ran_gf=gf)
                            self.irf_l1_b_random_base[gf] = irf_l1_b_random
                            self.irf_l1_b_random_base_summary[gf] = irf_l1_b_random_summary

                return

    def _initialize_nn(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # CDRNN MODEL

                # FEEDFORWARD ENCODER
                if self.input_dropout_rate:
                    self.input_dropout_layer = get_dropout(
                        self.input_dropout_rate,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        rescale=False,
                        name='input_dropout',
                        session=self.sess
                    )
                    self.time_X_dropout_layer = get_dropout(
                        self.input_dropout_rate,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        rescale=False,
                        name='time_X_dropout',
                        session=self.sess
                    )

                input_projection_layers = []
                for l in range(self.n_layers_input_projection + 1):
                    if l < self.n_layers_input_projection:
                        units = self.n_units_input_projection[l]
                        activation = self.input_projection_inner_activation
                        dropout = self.input_projection_dropout_rate
                        if self.normalize_input_projection:
                            bn = self.batch_normalization_decay
                        else:
                            bn = None
                        ln = self.layer_normalization_type
                        use_bias = True
                    else:
                        units = self.n_units_hidden_state
                        activation = self.input_projection_activation
                        dropout = None
                        bn = None
                        ln = None
                        use_bias = False
                    mn = self.maxnorm

                    projection = self.initialize_feedforward(
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        dropout=dropout,
                        maxnorm=mn,
                        batch_normalization_decay=bn,
                        layer_normalization_type=ln,
                        name='input_projection_l%s' % (l + 1)
                    )
                    self.layers.append(projection)

                    self.regularizable_layers.append(projection)
                    input_projection_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                input_projection_fn = compose_lambdas(input_projection_layers)

                self.input_projection_layers = input_projection_layers
                self.input_projection_fn = input_projection_fn
                self.h_in_dropout_layer = get_dropout(
                    self.h_in_dropout_rate,
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    name='h_in_dropout',
                    session=self.sess
                )

                # RNN ENCODER
                rnn_layers = []
                rnn_h_init = []
                rnn_h_ema = []
                rnn_c_init = []
                rnn_c_ema = []
                for l in range(self.n_layers_rnn):
                    units = self.n_units_rnn[l]

                    h_init, h_init_summary = self.initialize_rnn_h(l)
                    rnn_h_init.append(h_init)

                    h_ema_init = tf.Variable(tf.zeros(units), trainable=False, name='rnn_h_ema_l%d' % (l+1))
                    rnn_h_ema.append(h_ema_init)

                    c_init, c_init_summary = self.initialize_rnn_c(l)
                    rnn_c_init.append(c_init)

                    c_ema_init = tf.Variable(tf.zeros(units), trainable=False, name='rnn_c_ema_l%d' % (l+1))
                    rnn_c_ema.append(c_ema_init)

                    layer = self.initialize_rnn(l)
                    self.layers.append(layer)
                    self.regularizable_layers.append(layer)
                    rnn_layers.append(make_lambda(layer, session=self.sess, use_kwargs=True))

                rnn_encoder = compose_lambdas(rnn_layers)

                self.rnn_layers = rnn_layers
                self.rnn_h_init = rnn_h_init
                self.rnn_h_ema = rnn_h_ema
                self.rnn_c_init = rnn_c_init
                self.rnn_c_ema = rnn_c_ema
                self.rnn_encoder = rnn_encoder

                if self.n_layers_rnn:
                    rnn_projection_layers = []
                    for l in range(self.n_layers_rnn_projection + 1):
                        if l < self.n_layers_rnn_projection:
                            units = self.n_units_rnn_projection[l]
                            activation = self.rnn_projection_inner_activation
                            bn = self.batch_normalization_decay
                            ln = self.layer_normalization_type
                            use_bias = True
                        else:
                            units = self.n_units_hidden_state
                            activation = self.rnn_projection_activation
                            bn = None
                            ln = None
                            use_bias = False
                        mn = self.maxnorm

                        projection = self.initialize_feedforward(
                            units=units,
                            use_bias=use_bias,
                            activation=activation,
                            dropout=None,
                            maxnorm=mn,
                            batch_normalization_decay=bn,
                            layer_normalization_type=ln,
                            name='rnn_projection_l%s' % (l + 1)
                        )
                        self.layers.append(projection)

                        self.regularizable_layers.append(projection)
                        rnn_projection_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                    rnn_projection_fn = compose_lambdas(rnn_projection_layers)

                    self.rnn_projection_layers = rnn_projection_layers
                    self.rnn_projection_fn = rnn_projection_fn

                self.h_rnn_dropout_layer = get_dropout(
                    self.h_rnn_dropout_rate,
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    name='h_rnn_dropout',
                    session=self.sess
                )
                self.rnn_dropout_layer = get_dropout(
                    self.rnn_dropout_rate,
                    noise_shape=[None, None, 1],
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    rescale=False,
                    name='rnn_dropout',
                    session=self.sess
                )

                if self.normalize_h and self.normalize_activations:
                    self.h_normalization_layer = self.initialize_h_normalization()
                    self.layers.append(self.h_normalization_layer)
                    # if self.normalize_after_activation and False:
                    if self.normalize_after_activation:
                        h_bias, h_bias_summary = self.initialize_h_bias()
                    else:
                        h_bias = tf.zeros([1, 1, units])
                        h_bias_summary = tf.zeros([1, 1, units])
                else:
                    h_bias, h_bias_summary = self.initialize_h_bias()
                self.h_bias = h_bias

                self.h_dropout_layer = get_dropout(
                    self.h_dropout_rate,
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    name='h_dropout',
                    session=self.sess
                )

                irf_l1_W, irf_l1_W_summary = self.initialize_irf_l1_weights()
                self.irf_l1_W = irf_l1_W
                if self.irf_l1_use_bias:
                    if self.normalize_irf_l1 and self.normalize_activations:
                        self.irf_l1_normalization_layer = self.initialize_irf_l1_normalization()
                        self.layers.append(self.irf_l1_normalization_layer)
                        # if self.normalize_after_activation and False:
                        if self.normalize_after_activation:
                            irf_l1_b, irf_l1_b_summary = self.initialize_irf_l1_biases()
                        else:
                            irf_l1_b = tf.zeros_like(self.irf_l1_W)
                            irf_l1_b_summary = tf.zeros_like(self.irf_l1_W)
                    else:
                        irf_l1_b, irf_l1_b_summary = self.initialize_irf_l1_biases()
                self.irf_l1_b = irf_l1_b

                # Projection from hidden state to first layer (weights and biases) of IRF
                hidden_state_to_irf_l1 = self.initialize_feedforward(
                    units=self.n_units_irf_l1 * 2,
                    use_bias=False,
                    activation=None,
                    dropout=self.irf_dropout_rate,
                    maxnorm=self.maxnorm,
                    name='hidden_state_to_irf_l1'
                )
                self.layers.append(hidden_state_to_irf_l1)

                self.regularizable_layers.append(hidden_state_to_irf_l1)

                self.hidden_state_to_irf_l1 = hidden_state_to_irf_l1

                # IRF
                irf_layers = []
                for l in range(1, self.n_layers_irf + 1):
                    if l < self.n_layers_irf:
                        units = self.n_units_irf[l]
                        activation = self.irf_inner_activation
                        dropout = self.irf_dropout_rate
                        if self.normalize_irf:
                            bn = self.batch_normalization_decay
                            ln = self.layer_normalization_type
                        else:
                            bn = None
                            ln = None
                        use_bias = True
                        final = False
                        mn = self.maxnorm
                    else:
                        units = self.get_irf_output_ndim()
                        activation = self.irf_activation
                        dropout = None
                        bn = None
                        ln = None
                        use_bias = False
                        final = True
                        mn = None

                    projection = self.initialize_feedforward(
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        dropout=dropout,
                        maxnorm=mn,
                        batch_normalization_decay=bn,
                        layer_normalization_type=ln,
                        name='irf_l%s' % (l + 1),
                        final=final
                    )
                    self.layers.append(projection)

                    if l < self.n_layers_irf:
                        self.regularizable_layers.append(projection)
                    irf_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                irf = compose_lambdas(irf_layers)

                self.irf_layers = irf_layers
                self.irf = irf

    def _compile_random_effects(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.rnn_h_random = {}
                self.rnn_h_random_summary = {}
                self.rnn_c_random = {}
                self.rnn_c_random_summary = {}
                self.h_bias_random = {}
                self.h_bias_random_summary = {}
                self.irf_l1_W_random = {}
                self.irf_l1_W_random_summary = {}
                self.irf_l1_b_random = {}
                self.irf_l1_b_random_summary = {}

                Y_gf = self.Y_gf

                if self.ranef_dropout_rate:
                    self.ranef_dropout_layer = get_dropout(
                        self.ranef_dropout_rate,
                        training=self.training,
                        use_MAP_mode=tf.constant(True, dtype=tf.bool),
                        rescale=False,
                        constant=self.gf_defaults,
                        name='ranef_dropout',
                        session=self.sess
                    )

                    Y_gf = self.ranef_dropout_layer(Y_gf)

                for i, gf in enumerate(self.rangf):
                    _Y_gf = Y_gf[:, i]
                    if self.has_intercept[gf]:
                        if gf not in self.rnn_h_random:
                            self.rnn_h_random[gf] = []
                        if gf not in self.rnn_h_random_summary:
                            self.rnn_h_random_summary[gf] = []
                        if gf not in self.rnn_c_random:
                            self.rnn_c_random[gf] = []
                        if gf not in self.rnn_c_random_summary:
                            self.rnn_c_random_summary[gf] = []

                        # Random RNN initialization offsets
                        for l in range(self.n_layers_rnn):
                            # RNN hidden state
                            rnn_h_random = self.rnn_h_random_base[gf][l]
                            rnn_h_random_summary = self.rnn_h_random_base_summary[gf][l]
                            rnn_h_random_summary -= tf.reduce_mean(rnn_h_random_summary, axis=0, keepdims=True)
                            self._regularize(
                                rnn_h_random,
                                regtype='ranef',
                                var_name=reg_name('rnn_h_ran_l%d_by_%s' % (l, sn(gf)))
                            )
                            if self.log_random:
                                tf.summary.histogram(
                                    sn('by_%s/rnn_h_l%d' % (sn(gf), l+1)),
                                    rnn_h_random_summary,
                                    collections=['random']
                                )
                            self.rnn_h_random[gf].append(rnn_h_random)
                            self.rnn_h_random_summary[gf].append(rnn_h_random_summary)
                            rnn_h_random = tf.concat(
                                [
                                    rnn_h_random,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            self.rnn_h_init[l] += tf.gather(rnn_h_random, _Y_gf)

                            # RNN cell state
                            rnn_c_random = self.rnn_c_random_base[gf][l]
                            rnn_c_random_summary = self.rnn_c_random_base_summary[gf][l]
                            rnn_c_random -= tf.reduce_mean(rnn_c_random, axis=0, keepdims=True)
                            rnn_c_random_summary -= tf.reduce_mean(rnn_c_random_summary, axis=0, keepdims=True)
                            self._regularize(
                                rnn_c_random,
                                regtype='ranef',
                                var_name=reg_name('rnn_c_ran_l%d_by_%s' % (l + 1, sn(gf)))
                            )
                            if self.log_random:
                                tf.summary.histogram(
                                    sn('by_%s/rnn_c_l%d' % (sn(gf), l+1)),
                                    rnn_c_random_summary,
                                    collections=['random']
                                )
                            self.rnn_c_random[gf].append(rnn_c_random)
                            self.rnn_c_random_summary[gf].append(rnn_c_random_summary)
                            rnn_c_random = tf.concat(
                                [
                                    rnn_c_random,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            self.rnn_c_init[l] += tf.gather(rnn_c_random, _Y_gf)

                        # CDRNN hidden state
                        h_bias_random = self.h_bias_random_base[gf]
                        h_bias_random_summary = self.h_bias_random_base_summary[gf]
                        h_bias_random -= tf.reduce_mean(h_bias_random, axis=0, keepdims=True)
                        h_bias_random_summary -= tf.reduce_mean(h_bias_random_summary, axis=0, keepdims=True)
                        self._regularize(
                            h_bias_random,
                            regtype='ranef',
                            var_name=reg_name('h_bias_by_%s' % (sn(gf)))
                        )
                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/h' % sn(gf)),
                                h_bias_random_summary,
                                collections=['random']
                            )
                        self.h_bias_random[gf] = h_bias_random
                        self.h_bias_random_summary[gf] = h_bias_random_summary
                        n_units_hidden_state = self.n_units_hidden_state
                        h_bias_random = tf.concat(
                            [
                                h_bias_random,
                                tf.zeros([1, n_units_hidden_state])
                            ],
                            axis=0
                        )
                        self.h_bias += tf.expand_dims(tf.gather(h_bias_random, _Y_gf), axis=-2)

                        # IRF L1 weights
                        irf_l1_W_random = self.irf_l1_W_random_base[gf]
                        irf_l1_W_random_summary = self.irf_l1_W_random_base_summary[gf]
                        irf_l1_W_random -= tf.reduce_mean(irf_l1_W_random, axis=0, keepdims=True)
                        irf_l1_W_random_summary -= tf.reduce_mean(irf_l1_W_random_summary, axis=0, keepdims=True)
                        self._regularize(
                            irf_l1_W_random,
                            regtype='ranef',
                            var_name=reg_name('irf_l1_W_bias_by_%s' % (sn(gf)))
                        )
                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/irf_l1_W' % sn(gf)),
                                irf_l1_W_random_summary,
                                collections=['random']
                            )
                        self.irf_l1_W_random[gf] = irf_l1_W_random
                        self.irf_l1_W_random_summary[gf] = irf_l1_W_random_summary
                        irf_l1_W_random = tf.concat(
                            [
                                irf_l1_W_random,
                                tf.zeros([1, self.n_units_irf[0]])
                            ],
                            axis=0
                        )
                        self.irf_l1_W += tf.expand_dims(tf.gather(irf_l1_W_random, _Y_gf), axis=-2)

                        # IRF L1 biases
                        if self.irf_l1_use_bias:
                            irf_l1_b_random = self.irf_l1_b_random_base[gf]
                            irf_l1_b_random_summary = self.irf_l1_b_random_base_summary[gf]
                            irf_l1_b_random -= tf.reduce_mean(irf_l1_b_random, axis=0, keepdims=True)
                            irf_l1_b_random_summary -= tf.reduce_mean(irf_l1_b_random_summary, axis=0, keepdims=True)
                            self._regularize(
                                irf_l1_b_random,
                                regtype='ranef',
                                var_name=reg_name('irf_l1_b_bias_by_%s' % (sn(gf)))
                            )
                            if self.log_random:
                                tf.summary.histogram(
                                    sn('by_%s/irf_l1_b' % sn(gf)),
                                    irf_l1_b_random_summary,
                                    collections=['random']
                                )
                            self.irf_l1_b_random[gf] = irf_l1_b_random
                            self.irf_l1_b_random_summary[gf] = irf_l1_b_random_summary
                            irf_l1_b_random = tf.concat(
                                [
                                    irf_l1_b_random,
                                    tf.zeros([1, self.n_units_irf[0]])
                                ],
                                axis=0
                            )
                            self.irf_l1_b += tf.expand_dims(tf.gather(irf_l1_b_random, _Y_gf), axis=-2)

    def _rnn_encoder(self, X, plot_mode=False, **kwargs):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                h = [X]
                c = []
                b = tf.shape(X)[0]
                for l in range(len(self.rnn_layers)):
                    if plot_mode:
                        tile_dims = [b, 1]
                        h_init = tf.tile(self.rnn_h_ema[l][None, ...], tile_dims)
                        c_init = tf.tile(self.rnn_c_ema[l][None, ...], tile_dims)
                    else:
                        b = tf.shape(X)[0]
                        tile_dims = [b // tf.shape(self.rnn_h_init[l])[0], 1]
                        h_init = tf.tile(self.rnn_h_init[l], tile_dims)
                        c_init = tf.tile(self.rnn_c_init[l], tile_dims)

                    t_init = tf.zeros([b, 1], dtype=self.FLOAT_TF)
                    initial_state = CDRNNStateTuple(c=c_init, h=h_init, t=t_init)

                    layer = self.rnn_layers[l]
                    h_l, c_l = layer(h[-1], return_state=True, initial_state=initial_state, **kwargs)
                    h.append(h_l)
                    c.append(c_l)

                return h, c

    def compile_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                X = self.X_processed
                t_delta = self.t_delta
                X_time = self.X_time
                X_mask = self.X_mask

                if X_time is None:
                    X_shape = tf.shape(X)
                    time_X_shape = []
                    for j in range(len(X.shape) - 1):
                        s = X.shape[j]
                        try:
                            s = int(s)
                        except TypeError:
                            s = X_shape[j]
                        time_X_shape.append(s)
                    time_X_shape.append(1)
                    X_time = tf.ones(time_X_shape, dtype=self.FLOAT_TF)
                    time_X_mean = self.X_time_mean
                    X_time *= time_X_mean

                if self.center_time_X:
                    X_time -= self.X_time_mean
                if self.center_t_delta:
                    t_delta -= self.t_delta_mean

                if self.rescale_time_X:
                    X_time /= self.X_time_sd
                if self.rescale_t_delta:
                    t_delta /= self.t_delta_sd

                # Handle multiple impulse streams with different timestamps
                # by interleaving the impulses in temporal order
                if len(self.impulse_indices) > 1:
                    X_cdrnn = []
                    t_delta_cdrnn = []
                    time_X_cdrnn = []
                    time_X_mask_cdrnn = []

                    X_shape = tf.shape(X)
                    B = X_shape[0]
                    T = X_shape[1]

                    for i, ix in enumerate(self.impulse_indices):
                        dim_mask = np.zeros(len(self.impulse_names))
                        dim_mask[ix] = 1
                        while len(dim_mask.shape) < len(X.shape):
                            dim_mask = dim_mask[None, ...]
                        X_cur = X * dim_mask

                        if t_delta.shape[-1] > 1:
                            t_delta_cur = t_delta[..., ix[0]:ix[0]+1]
                        else:
                            t_delta_cur = t_delta

                        if X_time.shape[-1] > 1:
                            time_X_cur = X_time[..., ix[0]:ix[0]+1]
                        else:
                            time_X_cur = X_time

                        if X_mask is not None and X_mask.shape[-1] > 1:
                            time_X_mask_cur = X_mask[..., ix[0]]
                        else:
                            time_X_mask_cur = X_mask

                        X_cdrnn.append(X_cur)
                        t_delta_cdrnn.append(t_delta_cur)
                        time_X_cdrnn.append(time_X_cur)
                        if X_mask is not None:
                            time_X_mask_cdrnn.append(time_X_mask_cur)

                    X_cdrnn = tf.concat(X_cdrnn, axis=1)
                    t_delta_cdrnn = tf.concat(t_delta_cdrnn, axis=1)
                    time_X_cdrnn = tf.concat(time_X_cdrnn, axis=1)
                    if X_mask is not None:
                        time_X_mask_cdrnn = tf.concat(time_X_mask_cdrnn, axis=1)

                    sort_ix = tf.contrib.framework.argsort(tf.squeeze(time_X_cdrnn, axis=-1), axis=1)
                    B_ix = tf.tile(
                        tf.range(B)[..., None],
                        [1, T * len(self.impulse_indices)]
                    )
                    gather_ix = tf.stack([B_ix, sort_ix], axis=-1)

                    X = tf.gather_nd(X_cdrnn, gather_ix)
                    t_delta = tf.gather_nd(t_delta_cdrnn, gather_ix)
                    X_time = tf.gather_nd(time_X_cdrnn, gather_ix)
                    if X_mask is not None:
                        X_mask = tf.gather_nd(time_X_mask_cdrnn, gather_ix)
                else:
                    t_delta = t_delta[..., :1]
                    X_time = X_time[..., :1]
                    if X_mask is not None and len(X_mask.shape) == 3:
                        X_mask = X_mask[..., 0]

                if self.input_jitter_level:
                    jitter_sd = self.input_jitter_level
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

                if self.input_dropout_rate:
                    X = self.input_dropout_layer(X)
                    X_time = self.time_X_dropout_layer(X_time)

                X_rate = tf.pad(X, [(0,0), (0,0), (1,0)], constant_values=1.)
                X_rate = X_rate[..., None, None] # Pad out for nparam, ndim of response distribution(s)
                X = tf.concat([X, X_time], axis=-1)

                # Compute hidden state
                h = self.h_bias
                W = self.irf_l1_W
                b = self.irf_l1_b

                h_in = self.input_projection_fn(X)
                if self.h_in_noise_sd:
                    def h_in_train_fn(h_in=h_in):
                        return tf.random_normal(tf.shape(h_in), h_in, stddev=self.h_in_noise_sd)
                    def h_in_eval_fn(h_in=h_in):
                        return h_in
                    h_in = tf.cond(self.training, h_in_train_fn, h_in_eval_fn)
                if self.h_in_dropout_rate:
                    h_in = self.h_in_dropout_layer(h_in)
                h += h_in

                if self.n_layers_rnn:
                    rnn_hidden, rnn_cell = self._rnn_encoder(
                        X,
                        times=X_time,
                        mask=X_mask
                    )
                    h_rnn = self.rnn_projection_fn(rnn_hidden[-1])

                    if self.rnn_dropout_rate:
                        h_rnn = self.rnn_dropout_layer(h_rnn)

                    if self.h_rnn_noise_sd:
                        def h_rnn_train_fn(h_rnn=h_rnn):
                            return tf.random_normal(tf.shape(h_rnn), h_rnn, stddev=self.h_rnn_noise_sd)
                        def h_rnn_eval_fn(h_rnn=h_rnn):
                            return h_rnn
                        h_rnn = tf.cond(self.training, h_rnn_train_fn, h_rnn_eval_fn)
                    if self.h_rnn_dropout_rate:
                        h_rnn = self.h_rnn_dropout_layer(h_rnn)

                    h += h_rnn
                else:
                    h_rnn = rnn_hidden = rnn_cell = None

                if self.h_dropout_rate:
                    h = self.h_dropout_layer(h)

                if self.normalize_after_activation:
                    h = get_activation(self.hidden_state_activation, session=self.sess)(h)
                if self.normalize_h and self.normalize_activations:
                    h = self.h_normalization_layer(h)
                if not self.normalize_after_activation:
                    h = get_activation(self.hidden_state_activation, session=self.sess)(h)

                h_irf_in = h

                # Compute IRF outputs
                Wb_proj = self.hidden_state_to_irf_l1(h_irf_in)
                W_proj = Wb_proj[..., :self.n_units_irf_l1]
                b_proj = Wb_proj[..., self.n_units_irf_l1:]

                W += W_proj
                b += b_proj

                irf_l1 = W * t_delta + b
                if self.normalize_after_activation:
                    irf_l1 = get_activation(self.irf_inner_activation, session=self.sess)(irf_l1)
                if self.normalize_irf_l1 and self.irf_l1_use_bias and self.normalize_activations:
                    irf_l1 = self.irf_l1_normalization_layer(irf_l1)
                if not self.normalize_after_activation:
                    irf_l1 = get_activation(self.irf_inner_activation, session=self.sess)(irf_l1)

                n_impulse = len(self.impulse_names) + 1
                stabilizing_constant = 1. / (self.history_length * n_impulse)
                irf_out = self.irf(irf_l1) * stabilizing_constant

                # Slice and apply IRF outputs
                slices, shapes = self.get_irf_output_slice_and_shape()
                self.X_conv = {}
                self.output = {}
                if X_mask is not None:
                    X_mask = X_mask[..., None, None, None] # Pad out for impulses plus nparam, ndim of response distribution(s)
                for response in self.response_names:
                    _slice = slices[response]
                    _shape = shapes[response]

                    _irf_out = tf.reshape(irf_out[..., _slice], _shape)
                    X_weighted = X_rate * _irf_out
                    if X_mask is not None:
                        X_weighted *= X_mask
                    X_conv = tf.reduce_sum(X_weighted, axis=1) # Reduce along time dimension
                    output = tf.reduce_sum(X_conv, axis=1) # Reduce along impulse dimension

                    self.X_conv[response] = X_conv
                    self.output[response] = output

                ema_rate = self.ema_decay
                if ema_rate is None:
                    ema_rate = 0.

                self.rnn_h_ema_ops = []
                self.rnn_c_ema_ops = []

                mask = X_mask[..., None]
                denom = tf.reduce_sum(mask)

                if h_rnn is not None:
                    h_rnn_masked = h_rnn * mask
                    self._regularize(h_rnn_masked, regtype='context', var_name=reg_name('context'))

                for l in range(self.n_layers_rnn):
                    reduction_axes = list(range(len(rnn_hidden[l].shape)-1))

                    h_sum = tf.reduce_sum(rnn_hidden[l+1] * mask, axis=reduction_axes) # 0th layer is the input, so + 1
                    h_mean = h_sum / (denom + self.epsilon)
                    h_ema = self.rnn_h_ema[l]
                    h_ema_op = tf.assign(
                        h_ema,
                        ema_rate * h_ema + (1. - ema_rate) * h_mean
                    )
                    self.rnn_h_ema_ops.append(h_ema_op)

                    c_sum = tf.reduce_sum(rnn_cell[l] * mask, axis=reduction_axes)
                    c_mean = c_sum / (denom + self.epsilon)
                    c_ema = self.rnn_c_ema[l]
                    c_ema_op = tf.assign(
                        c_ema,
                        ema_rate * c_ema + (1. - ema_rate) * c_mean
                    )
                    self.rnn_c_ema_ops.append(c_ema_op)

                self.batch_norm_ema_ops = []
                if self.input_dropout_rate:
                    self.resample_ops += self.input_dropout_layer.resample_ops() + self.time_X_dropout_layer.resample_ops()
                if self.ranef_dropout_rate:
                    self.resample_ops += self.ranef_dropout_layer.resample_ops()
                if self.rnn_dropout_rate:
                    self.resample_ops += self.rnn_dropout_layer.resample_ops()
                for x in self.layers:
                    self.batch_norm_ema_ops += x.ema_ops()
                    self.resample_ops += x.resample_ops()




    ######################################################
    #
    #  Public utility methods
    #
    ######################################################

    def get_irf_output_ndim(self):
        n = 0
        n_impulse = len(self.impulse_names) + 1
        for response in self.response_names:
            if self.use_distributional_regression:
                nparam = self.get_response_nparam(response)
            else:
                nparam = 1
            ndim = self.get_response_ndim(response)
            n += n_impulse * nparam * ndim

        return n

    def get_irf_output_slice_and_shape(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                slices = {}
                shapes = {}
                n = 0
                n_impulse = len(self.impulse_names) + 1
                for response in self.response_names:
                    if self.use_distributional_regression:
                        nparam = self.get_response_nparam(response)
                    else:
                        nparam = 1
                    ndim = self.get_response_ndim(response)
                    slices[response] = slice(n, n + n_impulse * nparam * ndim)
                    shapes[response] = (self.X_batch_dim, self.X_time_dim, n_impulse, nparam, ndim)
                    n += n_impulse * nparam * ndim

                return slices, shapes




    ######################################################
    #
    #  Internal public network initialization methods.
    #  These must be implemented by all subclasses and
    #  should only be called at initialization.
    #
    ######################################################

    def initialize_feedforward(
            self,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            final=False,
            name=None
    ):
        raise NotImplementedError

    def initialize_rnn(
            self,
            l
    ):
        raise NotImplementedError

    def initialize_rnn_h(
            self,
            l,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_rnn_c(
            self,
            l,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_h_bias(
            self,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_h_normalization(self):
        raise NotImplementedError

    def initialize_intercept_l1_weights(
            self,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_intercept_l1_biases(
            self,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_intercept_l1_normalization(self):
        raise NotImplementedError

    def initialize_irf_l1_weights(
            self,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_irf_l1_biases(
            self,
            ran_gf=None
    ):
        raise NotImplementedError

    def initialize_irf_l1_normalization(self):
        raise NotImplementedError





    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    def initialize_model(self):
        self._initialize_nn()
        self._compile_random_effects()

    def report_settings(self, indent=0):
        out = super(CDRNN, self).report_settings(indent=indent)
        for kwarg in CDRNN_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' % (kwarg.key, "\"%s\"" % val if isinstance(val, str) else val)

        return out

    def run_train_step(self, feed_dict, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                to_run_names = []
                to_run = []
                to_run += [self.train_op, self.ema_op] + self.batch_norm_ema_ops
                to_run += [self.response_params_ema_ops[x] for x in self.response_params_ema_ops]
                if self.n_layers_rnn:
                    to_run += self.rnn_h_ema_ops + self.rnn_c_ema_ops
                if self.loss_filter_n_sds:
                    to_run_names.append('n_dropped')
                    to_run += [self.loss_m1_ema_op, self.loss_m2_ema_op, self.n_dropped]
                to_run_names += ['loss', 'reg_loss']
                to_run += [self.loss_func, self.reg_loss]
                if self.is_bayesian:
                    to_run.append(self.kl_loss)
                    to_run_names.append('kl_loss')
                out = self.sess.run(to_run, feed_dict=feed_dict)

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict
