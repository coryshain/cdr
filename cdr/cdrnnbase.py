import os
import re
import itertools
import numpy as np
import pandas as pd
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

            * TODO

        Additionally, if the subclass requires any keyword arguments beyond those provided by ``CDRNN``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.

        Example implementations of each of these methods can be found in the source code for :ref:`psyrnnmle` and :ref:`psyrnnbayes`.

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

    def __init__(self, form_str, X, y, **kwargs):
        super(CDRNN, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        assert len(X) == 1, 'Because of the recurrence, CDRNN requires synchronously measured predictors and therefore does not support multiple predictor files'

        for kwarg in CDRNN._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

    def _initialize_metadata(self):
        super(CDRNN, self)._initialize_metadata()

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
                self.n_units_input_projection = [int(x) for x in self.n_units_input_projection.split()]
            elif isinstance(self.n_units_input_projection, int):
                if self.n_units_input_projection is None:
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
                self.n_units_rnn = [int(x) for x in self.n_units_rnn.split()]
            elif isinstance(self.n_units_rnn, int):
                if self.n_units_rnn is None:
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
                if self.n_units_rnn_projection is None:
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
                if self.n_units_irf is None:
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

        if self.n_units_error_params_fn:
            if isinstance(self.n_units_error_params_fn, str):
                self.n_units_error_params_fn = [int(x) for x in self.n_units_error_params_fn.split()]
            elif isinstance(self.n_units_error_params_fn, int):
                if self.n_units_error_params_fn is None:
                    self.n_units_error_params_fn = [self.n_units_error_params_fn]
                else:
                    self.n_units_error_params_fn = [self.n_units_error_params_fn] * self.n_layers_error_params_fn
            if self.n_layers_error_params_fn is None:
                self.n_layers_error_params_fn = len(self.n_units_error_params_fn)
            if len(self.n_units_error_params_fn) == 1 and self.n_layers_error_params_fn != 1:
                self.n_units_error_params_fn = [self.n_units_error_params_fn[0]] * self.n_layers_error_params_fn
                self.n_layers_error_params_fn = len(self.n_units_error_params_fn)
        else:
            self.n_units_error_params_fn = []
            self.n_layers_error_params_fn = 0
        assert self.n_layers_error_params_fn == len(self.n_units_error_params_fn), 'Inferred n_layers_error_params_fn and n_units_error_params_fn must have the same number of layers. Saw %d and %d, respectively.' % (self.n_layers_error_params_fn, len(self.n_units_error_params_fn))

        if self.n_units_t_delta_embedding is None:
            if self.n_units_irf and self.n_units_irf[0]:
                self.n_units_t_delta_embedding = self.n_units_irf[0]
            elif self.n_units_rnn and self.n_units_rnn[-1]:
                self.n_units_t_delta_embedding = self.n_units_rnn[-1]
            else:
                raise ValueError('At least one of n_units_rnn, n_units_irf, or n_units_t_delta_embedding must be specified.')

        if self.n_units_hidden_state is None:
            if self.n_units_rnn:
                self.n_units_hidden_state = self.n_units_rnn[-1]
            else:
                self.n_units_hidden_state = self.n_units_t_delta_embedding

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

    def _initialize_cdrnn_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # CDRNN only supports synchronously measured predictors (from a single data frame)
                # so we can safely assume the same mask and timestamps for all impulses
                self.time_X_cdrnn = self.time_X[:, :, 0:1]
                self.t_delta_cdrnn = self.t_delta[:, :, 0:1]
                if self.rescale_time:
                    self.time_X_cdrnn /= self.time_X_sd
                    self.t_delta_cdrnn /= self.t_delta_sd

                time_X_mask_cdrnn = tf.cast(self.time_X_mask[:, :, 0], dtype=self.FLOAT_TF)

                if self.tail_dropout_rate and self.tail_dropout_max:
                    def tail_dropout_train_fn(mask=time_X_mask_cdrnn):
                        b = tf.shape(mask)[0]
                        t = tf.shape(mask)[1]

                        # Sequence mask is the complement of the sampled number of steps
                        tail_dropout_lengths = t - tf.random_uniform(
                            shape=[b],
                            minval=0,
                            maxval=self.tail_dropout_max + 1,
                            dtype=self.INT_TF
                        )
                        tail_dropout_mask = tf.sequence_mask(
                            tail_dropout_lengths,
                            tf.shape(mask)[1],
                        )
                        tail_dropped_mask = mask & tail_dropout_mask

                        drop_indicator = tf.random_uniform([b]) < self.tail_dropout_rate

                        out = tf.where(drop_indicator, tail_dropped_mask, mask)

                        return out

                    def tail_dropout_eval_fn(mask=time_X_mask_cdrnn):
                        return mask

                    time_X_mask_cdrnn = tf.cond(
                        self.training,
                        tail_dropout_train_fn,
                        tail_dropout_eval_fn
                    )

                self.time_X_mask_cdrnn = time_X_mask_cdrnn

                X = self.X

                if len(self.rangf):
                    self.use_rangf = True
                    rangf_1hot = []
                    for i in range(len(self.rangf)):
                        rangf_1hot_cur = tf.one_hot(
                            self.gf_y[:,i],
                            tf.cast(self.rangf_n_levels[i], dtype=self.INT_TF),
                            dtype=self.FLOAT_TF
                        )[:, :-1]
                        rangf_1hot.append(rangf_1hot_cur)
                    if len(rangf_1hot) > 0:
                        rangf_1hot = tf.concat(rangf_1hot, axis=-1)
                        if self.rangf_dropout_rate:
                            rangf_1hot = tf.layers.dropout(
                                rangf_1hot,
                                rate=self.rangf_dropout_rate,
                                training=self.trainings
                            )
                    else:
                        rangf_1hot = tf.zeros(
                            [tf.shape(self.gf_y)[0], 0],
                            dtype=self.FLOAT_TF
                        )
                    self.rangf_1hot = rangf_1hot
                else:
                    self.use_rangf = False

                # self.inputs = tf.concat(
                #     [X, self.t_delta_cdrnn],
                #     axis=-1
                # )
                self.inputs = X

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

                self.PLOT_AT_MEANS = True

                if self.PLOT_AT_MEANS:
                    plot_impulse_base_default = tf.convert_to_tensor(
                        [self.impulse_means[x] for x in self.impulse_names],
                        dtype=self.FLOAT_TF
                    )
                    plot_impulse_center_default = tf.zeros(
                        [len(self.impulse_names)],
                        dtype=self.FLOAT_TF
                    )
                else:
                    plot_impulse_base_default = tf.zeros(
                        [len(self.impulse_names)],
                        dtype=self.FLOAT_TF
                    )
                    plot_impulse_center_default = tf.convert_to_tensor(
                        [self.impulse_means[x] for x in self.impulse_names],
                        dtype=self.FLOAT_TF
                    )
                    # plot_impulse_center_default = tf.zeros(
                    #     [len(self.impulse_names)],
                    #     dtype=self.FLOAT_TF
                    # )
                plot_impulse_offset_default = tf.convert_to_tensor(
                    [self.impulse_sds[x] for x in self.impulse_names],
                    dtype=self.FLOAT_TF
                )
                self.plot_impulse_base = tf.placeholder_with_default(
                    plot_impulse_base_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_base'
                )
                self.plot_impulse_center = tf.placeholder_with_default(
                    plot_impulse_center_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_center'
                )
                self.plot_impulse_offset = tf.placeholder_with_default(
                    plot_impulse_offset_default,
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_offset'
                )
                self.plot_impulse_1hot = tf.placeholder_with_default(
                    tf.zeros([len(self.impulse_names)], dtype=self.FLOAT_TF),
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_1hot'
                )
                self.plot_impulse_1hot_2 = tf.placeholder_with_default(
                    tf.zeros([len(self.impulse_names)], dtype=self.FLOAT_TF),
                    shape=[len(self.impulse_names)],
                    name='plot_impulse_1hot'
                )

                self.plot_impulse_base_expanded = self.plot_impulse_base[None, None, ...]
                self.plot_impulse_center_expanded = self.plot_impulse_center[None, None, ...]
                self.plot_impulse_offset_expanded = self.plot_impulse_offset[None, None, ...]
                self.plot_impulse_1hot_expanded = self.plot_impulse_1hot[None, None, ...]
                self.plot_impulse_1hot_2_expanded = self.plot_impulse_1hot_2[None, None, ...]

                if self.nn_regularizer_name is None:
                    self.nn_regularizer = None
                elif self.nn_regularizer_name == 'inherit':
                    self.nn_regularizer = self.regularizer
                else:
                    if self.nn_regularizer_name == 'l1_l2_regularizer':
                        self.nn_regularizer = getattr(tf.contrib.layers, self.nn_regularizer_name)(
                            self.nn_regularizer_scale,
                            self.nn_regularizer_scale
                        )
                    else:
                        self.nn_regularizer = getattr(tf.contrib.layers, self.nn_regularizer_name)(self.nn_regularizer_scale)

                if self.context_regularizer_name is None:
                    self.context_regularizer = None
                elif self.context_regularizer_name == 'inherit':
                    self.context_regularizer = self.regularizer
                else:
                    self.context_regularizer = getattr(tf.contrib.layers, self.context_regularizer_name)(self.context_regularizer_scale)

                self.regularizable_layers = []

    def _add_rangf(self, x, rangf_embeddings=None):
        if rangf_embeddings is None:
            rangf_embeddings = self.rangf_embeddings
        multiples = tf.shape(x)[0] // tf.shape(rangf_embeddings)[0]
        t = tf.shape(x)[1]
        rangf_embeddings = tf.tile(
            rangf_embeddings,
            tf.convert_to_tensor([multiples, t, 1], dtype=self.INT_TF)
        )
        out = tf.concat([x, rangf_embeddings], axis=-1)

        return out

    def _initialize_encoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                self.input_projection_layers = []
                for l in range(self.n_layers_input_projection + 1):
                    if l < self.n_layers_input_projection:
                        units = self.n_units_input_projection[l]
                        activation = self.input_projection_inner_activation
                        dropout = self.input_projection_dropout_rate
                    else:
                        units = self.n_units_hidden_state
                        activation = self.input_projection_activation
                        dropout = None
                    projection = DenseLayer(
                        training=self.training,
                        units=units,
                        use_bias=True,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer='zeros_initializer',
                        dropout=dropout,
                        batch_normalization_decay=self.batch_normalization_decay,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='input_projection_l%s' % (l + 1)
                    )
                    self.regularizable_layers.append(projection)
                    self.input_projection_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                self.input_projection_fn = compose_lambdas(self.input_projection_layers)

                self.rnn_layers = []
                self.rnn_h_init = []
                self.rnn_h_ema = []
                self.rnn_c_init = []
                self.rnn_c_ema = []
                for l in range(self.n_layers_rnn):
                    if l < self.n_layers_rnn - 1:
                        return_seqs = True
                    else:
                        return_seqs = True
                    units = self.n_units_rnn[l]

                    h_init = tf.Variable(tf.zeros(units), name='rnn_h_%d' % (l+1))
                    self.rnn_h_init.append(h_init)

                    h_ema_init = tf.Variable(tf.zeros(units), trainable=False, name='rnn_h_ema_%d' % (l+1))
                    self.rnn_h_ema.append(h_ema_init)

                    c_init = tf.Variable(tf.zeros(units), name='rnn_c_%d' % (l+1))
                    self.rnn_c_init.append(c_init)
                    
                    c_ema_init = tf.Variable(tf.zeros(units), trainable=False, name='rnn_c_ema_%d' % (l+1))
                    self.rnn_c_ema.append(c_ema_init)

                    layer = CDRNNLayer(
                        training=self.training,
                        units=units,
                        time_projection_depth=self.n_layers_irf+1,
                        activation=self.rnn_activation,
                        recurrent_activation=self.recurrent_activation,
                        time_projection_inner_activation=self.irf_inner_activation,
                        bottomup_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        bottomup_dropout=self.input_projection_dropout_rate,
                        h_dropout=self.rnn_h_dropout_rate,
                        c_dropout=self.rnn_c_dropout_rate,
                        forget_rate=self.forget_rate,
                        bias_initializer='zeros_initializer',
                        return_sequences=return_seqs,
                        batch_normalization_decay=None,
                        name='rnn_l%d' % (l + 1),
                        epsilon=self.epsilon,
                        session=self.sess
                    )
                    self.regularizable_layers.append(layer)
                    self.rnn_layers.append(make_lambda(layer, session=self.sess, use_kwargs=True))

                self.rnn_encoder = compose_lambdas(self.rnn_layers)

                self.rnn_projection_layers = []
                for l in range(self.n_layers_rnn_projection + 1):
                    if l < self.n_layers_rnn_projection:
                        units = self.n_units_rnn_projection[l]
                        activation = self.rnn_projection_inner_activation
                    else:
                        units = self.n_units_hidden_state
                        activation = self.rnn_projection_activation

                    projection = DenseLayer(
                        training=self.training,
                        units=units,
                        use_bias=True,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer='zeros_initializer',
                        dropout=None,
                        batch_normalization_decay=self.batch_normalization_decay,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='rnn_projection_l%s' % (l + 1)
                    )
                    self.regularizable_layers.append(projection)
                    self.rnn_projection_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                self.rnn_projection_fn = compose_lambdas(self.rnn_projection_layers)

                self.t_delta_embedding_W = tf.get_variable(
                    't_delta_embedding_W',
                    shape=[1, 1, self.n_units_t_delta_embedding],
                    initializer=get_initializer(self.kernel_initializer, self.sess)
                )
                self.t_delta_embedding_b = tf.get_variable(
                    't_delta_embedding_b',
                    shape=[1, 1, self.n_units_t_delta_embedding],
                    initializer=tf.zeros_initializer
                )
                self.regularizable_layers += [self.t_delta_embedding_W, self.t_delta_embedding_b]

                self.hidden_state_to_irf_l1 = DenseLayer(
                    training=self.training,
                    units=self.n_units_t_delta_embedding * 2,
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer='zeros_initializer',
                    dropout=self.irf_dropout_rate,
                    batch_normalization_decay=self.batch_normalization_decay,
                    epsilon=self.epsilon,
                    session=self.sess,
                    name='hidden_state_to_irf_l1'
                )
                self.regularizable_layers.append(self.hidden_state_to_irf_l1)

                self.irf_layers = []
                for l in range(self.n_layers_irf + 1):
                    if l < self.n_layers_irf:
                        units = self.n_units_irf[l]
                        activation = self.irf_inner_activation
                        dropout = self.irf_dropout_rate
                        bn = self.batch_normalization_decay
                        use_bias = True
                    else:
                        # units = 1
                        units = 1
                        activation = self.irf_activation
                        dropout = None
                        bn = None
                        use_bias = False

                    projection = DenseLayer(
                        training=self.training,
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer='zeros_initializer',
                        dropout=dropout,
                        batch_normalization_decay=bn,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='irf_l%s' % (l + 1)
                    )
                    if l < self.n_layers_irf:
                        self.regularizable_layers.append(projection)
                    self.irf_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                self.irf = compose_lambdas(self.irf_layers)
                
                self.error_params_fn_layers = []
                for l in range(self.n_layers_error_params_fn + 1):
                    if l < self.n_layers_error_params_fn:
                        units = self.n_units_error_params_fn[l]
                        activation = self.error_params_fn_inner_activation
                        dropout = self.error_params_fn_dropout_rate
                        bn = self.batch_normalization_decay
                        use_bias = True
                    else:
                        units = 1
                        if self.asymmetric_error:
                            units += 2
                        activation = self.error_params_fn_activation
                        dropout = None
                        bn = None
                        use_bias = False

                    projection = DenseLayer(
                        training=self.training,
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer='zeros_initializer',
                        dropout=dropout,
                        batch_normalization_decay=bn,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='error_params_fn_l%s' % (l + 1)
                    )
                    if l < self.n_layers_error_params_fn:
                        self.regularizable_layers.append(projection)
                    self.error_params_fn_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                self.error_params_fn = compose_lambdas(self.error_params_fn_layers)


                # Gates
                self.input_gates = tf.tanh(tf.Variable(
                    tf.zeros([len(self.impulse_names)+1]),
                    name='input_gate_logits'
                ))[None, None, ...]
                # self.t_delta_gate = tf.identity(tf.Variable(
                #     tf.zeros([]),
                #     name='t_delta_gate_logit'
                # ))
                self.y_sd_delta_gate = tf.tanh(tf.Variable(
                    tf.zeros([]), name='y_sd_delta_gate_logit'
                ))
                if self.asymmetric_error:
                    self.y_skewness_delta_gate = tf.tanh(tf.Variable(
                        tf.zeros([]), name='y_skewness_delta_gate_logit'
                    ))
                    self.y_tailweight_delta_gate = tf.tanh(
                        tf.Variable(tf.zeros([]), name='y_tailweight_delta_gate_logit'
                    ))

                def sum_predictions(x, mask=None):
                    if mask is not None:
                        x *= tf.cast(mask, dtype=self.FLOAT_TF)[..., None]

                    return tf.reduce_sum(x, axis=1)

                self.summed_predictions = make_lambda(sum_predictions, session=self.sess, use_kwargs=True)

                # Intercept
                if self.has_intercept[None]:
                    self.intercept_fixed = tf.Variable(0., name='intercept_fixed')
                    self.intercept_fixed_summary = self.intercept_fixed
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_fixed_summary,
                        collections=['params']
                    )
                    self._regularize(self.intercept_fixed, type='intercept', var_name='intercept')
                    if self.convergence_basis.lower() == 'parameters':
                        self._add_convergence_tracker(self.intercept_fixed_summary, 'intercept_fixed')

                else:
                    self.intercept_fixed = self.intercept_fixed_base

                self.intercept = self.intercept_fixed
                self.intercept_summary = self.intercept_fixed_summary
                self.intercept_random = {}
                self.intercept_random_summary = {}
                self.intercept_random_means = {}

                # RANDOM EFFECTS
                self.h_ran_matrix = []
                self.h_ran = []
                self.rnn_h_ran_matrix = [[] for l in range(self.n_layers_rnn)]
                self.rnn_h_ran = [[] for l in range(self.n_layers_rnn)]
                self.rnn_c_ran_matrix = [[] for l in range(self.n_layers_rnn)]
                self.rnn_c_ran = [[] for l in range(self.n_layers_rnn)]
                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    levels_ix = np.arange(self.rangf_n_levels[i] - 1)
                    gf_y = self.gf_y[:, i]
                    if self.ranef_dropout_rate:
                        def gf_train_fn(gf_y=gf_y):
                            dropout_mask = tf.cast(tf.random_uniform(tf.shape(gf_y)) > self.ranef_dropout_rate, tf.bool)
                            alt = tf.zeros_like(gf_y)
                            return tf.where(dropout_mask, gf_y, alt)

                        def gf_eval_fn(gf_y=gf_y):
                            return gf_y

                        gf_y = tf.cond(self.training, gf_train_fn, gf_eval_fn)

                    # Random intercepts
                    if self.has_intercept[gf]:
                        intercept_random = tf.Variable(
                            tf.zeros([len(levels_ix)]),
                            name='intercept_by_%s' % sn(gf)
                        )
                        intercept_random_summary = intercept_random

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

                        # Create record for convergence tracking
                        if self.convergence_basis.lower() == 'parameters':
                            self._add_convergence_tracker(self.intercept_random_summary[gf], 'intercept_by_%s' %gf)

                        self.intercept += tf.gather(intercept_random, gf_y)
                        self.intercept_summary += tf.gather(intercept_random_summary, gf_y)

                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/intercept' % gf),
                                intercept_random_summary,
                                collections=['random']
                            )

                        # Random rnn initialization offsets
                        for l in range(self.n_layers_rnn):
                            rnn_h_ran_matrix = tf.Variable(
                                tf.zeros([len(levels_ix), self.n_units_rnn[l]]),
                                name='rnn_h_ran_%d_by_%s' % (l, sn(gf))
                            )
                            rnn_h_ran_matrix -= tf.reduce_mean(rnn_h_ran_matrix, axis=0, keepdims=True)
                            self._regularize(rnn_h_ran_matrix, type='ranef', var_name='rnn_h_ran_%d_by_%s' % (l, sn(gf)))

                            rnn_h_ran_matrix = tf.concat(
                                [
                                    rnn_h_ran_matrix,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            self.rnn_h_ran_matrix[l].append(rnn_h_ran_matrix)
                            self.rnn_h_ran[l].append(tf.gather(rnn_h_ran_matrix, gf_y))
                            
                            rnn_c_ran_matrix = tf.Variable(
                                tf.zeros([len(levels_ix), self.n_units_rnn[l]]),
                                name='rnn_c_ran_%d_by_%s' % (l, sn(gf))
                            )
                            rnn_c_ran_matrix -= tf.reduce_mean(rnn_c_ran_matrix, axis=0, keepdims=True)
                            self._regularize(rnn_c_ran_matrix, type='ranef', var_name='rnn_c_ran_%d_by_%s' % (l, sn(gf)))

                            rnn_c_ran_matrix = tf.concat(
                                [
                                    rnn_c_ran_matrix,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            self.rnn_c_ran_matrix[l].append(rnn_c_ran_matrix)
                            self.rnn_c_ran[l].append(tf.gather(rnn_c_ran_matrix, gf_y))

                        # Random hidden state offsets
                        h_ran_matrix = tf.Variable(
                            tf.zeros([len(levels_ix), self.n_units_hidden_state]),
                            name='h_ran_by_%s' % sn(gf)
                        )
                        h_ran_matrix -= tf.reduce_mean(h_ran_matrix, axis=0, keepdims=True)
                        self._regularize(h_ran_matrix, type='ranef', var_name='h_ran_by_%s' % sn(gf))

                        h_ran_matrix = tf.concat(
                            [
                                h_ran_matrix,
                                tf.zeros([1, self.n_units_hidden_state])
                            ],
                            axis=0
                        )
                        self.h_ran_matrix.append(h_ran_matrix)
                        self.h_ran.append(tf.gather(h_ran_matrix, gf_y))

    def _rnn_encoder(self, x, zero_rnn_initial_state=False, **kwargs):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                h = [x]
                c = []
                for l in range(len(self.rnn_layers)):
                    if zero_rnn_initial_state:
                        b = tf.shape(x)[0]
                        tile_dims = [b, 1]
                        h_init = tf.tile(self.rnn_h_init[l][None, ...], tile_dims)
                        c_init = tf.tile(self.rnn_c_init[l][None, ...], tile_dims)
                    else:
                        b = tf.shape(x)[0]
                        tile_dims = [b, 1]
                        h_init = tf.tile(self.rnn_h_ema[l][None, ...], tile_dims)
                        c_init = tf.tile(self.rnn_c_ema[l][None, ...], tile_dims)
                        # h_init = tf.tile(self.rnn_h_init[l][None, ...], tile_dims)
                        # c_init = tf.tile(self.rnn_c_init[l][None, ...], tile_dims)

                    if self.use_rangf:
                        h_init += tf.add_n(self.rnn_h_ran[l])
                        c_init += tf.add_n(self.rnn_c_ran[l])

                    t_init = tf.zeros([b, 1], dtype=self.FLOAT_TF)
                    initial_state = CDRNNStateTuple(c=c_init, h=h_init, t=t_init)

                    layer = self.rnn_layers[l]
                    h_cur, c_cur = layer(h[-1], return_state=True, initial_state=initial_state, **kwargs)
                    h.append(h_cur)
                    c.append(c_cur)

                return h, c

    def _apply_model(self, inputs, t_delta, mask=None, times=None, zero_rnn_initial_state=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if times is None:
                    inputs_shape = tf.shape(inputs)
                    times_shape = []
                    for i in range(len(inputs.shape)-1):
                        s = inputs.shape[i]
                        try:
                            s = int(s)
                        except TypeError:
                            s = inputs_shape[i]
                        times_shape.append(s)
                    times_shape.append(1)
                    times = tf.ones(times_shape, dtype=self.FLOAT_TF)
                    time_mean =  self.time_X_mean
                    if self.rescale_time:
                        time_mean /= self.time_X_sd
                    times *= time_mean

                if self.input_jitter_level:
                    jitter_sd = self.input_jitter_level
                    inputs = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(inputs), inputs, jitter_sd),
                        lambda: inputs
                    )
                    t_delta = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(t_delta), t_delta, jitter_sd),
                        lambda: t_delta
                    )
                    times = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(times), times, jitter_sd),
                        lambda: times
                    )

                inputs = tf.concat([inputs, times], axis=-1) * self.input_gates

                if self.predictor_dropout_rate:
                    inputs = tf.layers.dropout(
                        inputs,
                        rate=self.predictor_dropout_rate,
                        training=self.training
                    )

                if self.event_dropout_rate:
                    inputs_shape = tf.shape(inputs)
                    noise_shape = []
                    for i in range(len(inputs.shape)-1):
                        try:
                            s = int(inputs.shape[i])
                        except TypeError:
                            s = inputs_shape[i]
                        noise_shape.append(s)
                    noise_shape.append(1)

                    mask_is_none = mask is None

                    def train_fn(inputs=inputs, noise_shape=noise_shape, mask=mask):
                        dropout_mask = tf.cast(tf.random_uniform(noise_shape) > self.event_dropout_rate, dtype=self.FLOAT_TF)
                        print(dropout_mask)
                        inputs_out = inputs * dropout_mask
                        if mask is not None:
                            mask_out = mask * dropout_mask[..., 0]
                        else:
                            mask_out = tf.zeros(tf.shape(inputs)[:-1])

                        return inputs_out, mask_out

                    def eval_fn(inputs=inputs, mask=mask):
                        if mask is None:
                            mask = tf.zeros(tf.shape(inputs)[:-1])
                        return inputs, mask

                    inputs, mask = tf.cond(self.training, train_fn, eval_fn)

                    if mask_is_none:
                        mask = None

                # Compute hidden state
                h_in = self.input_projection_fn(inputs)
                if self.h_in_noise_sd:
                    def h_in_train_fn(h_in=h_in):
                        return tf.random_normal(tf.shape(h_in), h_in, stddev=self.h_in_noise_sd)
                    def h_in_eval_fn(h_in=h_in):
                        return h_in
                    h_in = tf.cond(self.training, h_in_train_fn, h_in_eval_fn)
                if self.h_in_dropout_rate:
                    h_in = get_dropout(self.h_in_dropout_rate, training=self.training, session=self.sess)(h_in)
                h = h_in

                if self.n_layers_rnn:
                    rnn_hidden, rnn_cell = self._rnn_encoder(inputs, mask=mask, times=times, zero_rnn_initial_state=zero_rnn_initial_state)
                    h_rnn = self.rnn_projection_fn(rnn_hidden[-1])
                    if self.h_rnn_noise_sd:
                        def h_rnn_train_fn(h_rnn=h_rnn):
                            return tf.random_normal(tf.shape(h_rnn), h_rnn, stddev=self.h_rnn_noise_sd)
                        def h_rnn_eval_fn(h_rnn=h_rnn):
                            return h_rnn
                        h_rnn = tf.cond(self.training, h_rnn_train_fn, h_rnn_eval_fn)
                    if self.h_rnn_dropout_rate:
                        h_rnn = get_dropout(self.h_rnn_dropout_rate, training=self.training, session=self.sess)(h_rnn)
                    # h_rnn = tf.Print(h_rnn, [tf.reduce_mean(tf.abs(h_rnn)), tf.reduce_mean(tf.abs(h))])
                    h = h + h_rnn
                    # h = h * self.input_projection_proportion + h_rnn * self.rnn_proportion
                else:
                    h_rnn = rnn_hidden = rnn_cell = None

                if self.use_rangf:
                    h += tf.expand_dims(tf.add_n(self.h_ran), axis=-2)

                h = get_activation(self.hidden_state_activation, session=self.sess)(h)

                # Compute response
                W = self.t_delta_embedding_W
                b = self.t_delta_embedding_b

                Wb_proj = self.hidden_state_to_irf_l1(h)
                W_proj = Wb_proj[..., :self.n_units_t_delta_embedding]
                b_proj = Wb_proj[..., self.n_units_t_delta_embedding:]

                W += W_proj
                b += b_proj

                activation = get_activation(self.irf_inner_activation, session=self.sess)

                # t_delta *= self.t_delta_gate

                t_delta_embedding_preactivations = W * t_delta + b
                t_delta_embeddings = activation(t_delta_embedding_preactivations)

                y = self.summed_predictions(self.irf(t_delta_embeddings))
                # y *= self.y_coef

                error_params = self.error_params_fn(h[..., -1, :])

                y_sd_delta = error_params[..., 0] * self.y_sd_delta_gate
                if self.asymmetric_error:
                    y_skewness_delta = error_params[..., 1] * self.y_skewness_delta_gate
                    y_tailweight_delta = error_params[..., 2] * self.y_tailweight_delta_gate
                else:
                    y_skewness_delta = None
                    y_tailweight_delta = None

                return {
                    'y': y,
                    'y_sd_delta': y_sd_delta,
                    'y_skewness_delta': y_skewness_delta,
                    'y_tailweight_delta': y_tailweight_delta,
                    'rnn_hidden': rnn_hidden,
                    'rnn_cell': rnn_cell,
                    'h_rnn': h_rnn,
                    'h_in': h_in,
                    'h': h
                }

    def _construct_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                model_dict = self._apply_model(
                    self.inputs,
                    self.t_delta_cdrnn,
                    mask=self.time_X_mask_cdrnn,
                    times=self.time_X_cdrnn,
                    zero_rnn_initial_state=True
                )

                y = model_dict['y']
                y_sd_delta = model_dict['y_sd_delta']
                y_skewness_delta = model_dict['y_skewness_delta']
                y_tailweight_delta = model_dict['y_tailweight_delta']

                y = tf.squeeze(y, axis=-1)
                y += self.intercept

                self.out = y
                # Hack needed for MAP evaluation of CDRNNBayes
                self.out_mean = self.out

                ema_rate = self.ema_decay
                if ema_rate is None:
                    ema_rate = 0.

                h_rnn = model_dict['h_rnn']
                if h_rnn is not None:
                    rnn_hidden = model_dict['rnn_hidden']
                    rnn_cell = model_dict['rnn_cell']
                    reduction_axes = list(range(len(h_rnn.shape) - 1))
                    self._regularize(tf.reduce_mean(h_rnn, axis=reduction_axes), type='context', var_name='context')

                    self.rnn_h_ema_ops = []
                    self.rnn_c_ema_ops = []

                    mask = self.time_X_mask_cdrnn[..., None]

                    for l in range(self.n_layers_rnn):
                        reduction_axes = list(range(len(rnn_hidden[l].shape)-1))

                        denom = tf.reduce_sum(mask)

                        h_sum = tf.reduce_sum(rnn_hidden[l+1] * mask, axis=reduction_axes) # 0th layer is the input, so + 1
                        h_mean = h_sum / (denom + self.epsilon)
                        h_ema = self.rnn_h_ema[l]
                        h_ema_op = tf.assign(
                            h_ema,
                            ema_rate * h_ema + (1 - ema_rate) * h_mean
                        )
                        self.rnn_h_ema_ops.append(h_ema_op)

                        c_sum = tf.reduce_sum(rnn_cell[l] * mask, axis=reduction_axes)
                        c_mean = c_sum / (denom + self.epsilon)
                        c_ema = self.rnn_c_ema[l]
                        c_ema_op = tf.assign(
                            c_ema,
                            ema_rate * c_ema + (1 - ema_rate) * c_mean
                        )
                        self.rnn_c_ema_ops.append(c_ema_op)

                self.y_sd_delta = y_sd_delta
                self.y_sd_delta_ema = tf.Variable(0., trainable=False, name='y_sd_delta_ema')
                self.y_sd_delta_ema_op = tf.assign(
                    self.y_sd_delta_ema,
                    ema_rate * self.y_sd_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_sd_delta)
                )

                if self.asymmetric_error:
                    self.y_skewness_delta = y_skewness_delta
                    self.y_skewness_delta_ema = tf.Variable(0., trainable=False, name='y_skewness_delta_ema')
                    self.y_skewness_delta_ema_op = tf.assign(
                        self.y_skewness_delta_ema,
                        ema_rate * self.y_skewness_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_skewness_delta)
                    )

                    self.y_tailweight_delta = y_tailweight_delta
                    self.y_tailweight_delta_ema = tf.Variable(0., trainable=False, name='y_tailweight_delta_ema')
                    self.y_tailweight_delta_ema_op = tf.assign(
                        self.y_tailweight_delta_ema,
                        ema_rate * self.y_tailweight_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_tailweight_delta)
                    )





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _collect_plots(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # IRF 1D PLOTS
                t = tf.shape(self.support)[0]

                t_delta = self.support[..., None]
                if self.rescale_time:
                    t_delta /= self.t_delta_sd

                x = self.plot_impulse_1hot_expanded
                y = self.plot_impulse_1hot_2_expanded
                b = self.plot_impulse_base_expanded
                c = self.plot_impulse_center_expanded
                s = self.plot_impulse_offset_expanded

                X_rate = tf.tile(
                    self.plot_impulse_base_expanded,
                    [t, 1, 1]
                )

                self.irf_1d_rate_support = self.support
                irf_1d_rate_plot = self._apply_model(X_rate, t_delta)['y']
                self.irf_1d_rate_plot = irf_1d_rate_plot[None, ...]

                X = tf.tile(
                     x * (c + s) + b,
                    [t, 1, 1]
                )
                self.irf_1d_support = self.support
                irf_1d_plot = self._apply_model(X, t_delta)['y']
                self.irf_1d_plot = irf_1d_plot[None, ...] - self.irf_1d_rate_plot

                # IRF SURFACE PLOTS
                time_support = tf.linspace(
                    self.support_start,
                    self.n_time_units+self.support_start,
                    self.n_surface_plot_points_per_side
                )

                t_delta_square = tf.tile(
                    time_support[..., None, None],
                    [self.n_surface_plot_points_per_side, 1, 1]
                )
                if self.rescale_time:
                    t_delta_square /= self.t_delta_sd

                u_src = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    self.n_surface_plot_points_per_side,
                )
                u = u_src[..., None, None]

                u_rate = u_src
                X_rate = tf.tile(
                    self.plot_impulse_base_expanded,
                    [self.n_surface_plot_points_normalized, 1, 1]
                )
                irf_surface_rate_plot = self._apply_model(X_rate, t_delta_square)['y']
                self.irf_surface_rate_plot = tf.reshape(
                    irf_surface_rate_plot[None, ...],
                    [self.n_surface_plot_points_per_side, self.n_surface_plot_points_per_side]
                )
                self.irf_surface_rate_meshgrid = tf.meshgrid(time_support, u_rate)

                u = x * (c + u)
                X = tf.reshape(
                    tf.tile(
                        u + b,
                        [1, self.n_surface_plot_points_per_side, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )
                irf_surface_plot = self._apply_model(X, t_delta_square)['y']
                self.irf_surface_plot = tf.reshape(
                    irf_surface_plot[None, ...],
                    [self.n_surface_plot_points_per_side, self.n_surface_plot_points_per_side]
                ) - self.irf_surface_rate_plot
                self.irf_surface_meshgrid = tf.meshgrid(
                    time_support,
                    tf.reduce_prod(
                        u + x * b + (1 - x),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                        axis=[1,2]
                    )
                )

                # CURVATURE PLOTS
                t_interaction = self.t_interaction
                if self.rescale_time:
                    t_interaction /= self.t_delta_sd

                rate_at_t = self._apply_model(
                    self.plot_impulse_base_expanded,
                    tf.ones([1, 1, 1], dtype=self.FLOAT_TF) * t_interaction
                )['y']
                rate_at_t = tf.squeeze(rate_at_t)

                t_delta = tf.ones([t, 1, 1], dtype=self.FLOAT_TF) * t_interaction

                u = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    t,
                )[..., None, None]
                u = x * (c + u)
                X = u + b
                curvature_plot = self._apply_model(X, t_delta)['y']
                self.curvature_plot = curvature_plot - rate_at_t
                self.curvature_support = tf.reduce_prod(
                    u + x * b + (1 - x),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                    axis=[1,2]
                )

                # INTERACTION PLOTS
                t_delta = tf.ones(
                    [self.n_surface_plot_points_normalized, 1, 1],
                    dtype=self.FLOAT_TF
                ) * t_interaction

                v = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    self.n_surface_plot_points_per_side,
                )[..., None, None]

                u_1 = x * (c + v)
                X_1 = tf.reshape(
                    tf.tile(
                        u_1 + b,
                        [self.n_surface_plot_points_per_side, 1, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )

                u_2 = y * (c + v)
                X_2 = tf.reshape(
                    tf.tile(
                        u_2 + b,
                        [1, self.n_surface_plot_points_per_side, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )

                X = X_1 + X_2 + b
                interaction_surface_plot = self._apply_model(X, t_delta)['y']
                self.interaction_surface_plot = interaction_surface_plot - rate_at_t
                self.interaction_surface_support = tf.meshgrid(
                    tf.reduce_prod(
                        u_1 + x * b + (1 - x),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                        axis=[1,2]
                    ),
                    tf.reduce_prod(
                        u_2 + x * b +  (1 - y),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                        axis=[1,2]
                    )
                )



    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    def build(self, outdir=None, restore=True):
        """
        Construct the CDRNN network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Report model details after initialization.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './cdrnn_model/'
        else:
            self.outdir = outdir

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs(len(self.impulse_names))
                self._initialize_cdrnn_inputs()
                self._initialize_encoder()
                self._construct_network()
                self.initialize_objective()
                self._initialize_logging()
                self._initialize_ema()

                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
                self._initialize_saver()
                self.load(restore=restore)

                self._initialize_convergence_checking()
                self._collect_plots()

                self.sess.graph.finalize()

    def get_plot_names(self, composite='composite', scaled='scaled', dirac='dirac', plot_type='irf_1d'):
        if plot_type.lower() in ['irf_1d', 'irf_surface']:
            out = ['rate'] + self.impulse_names[:]
        elif plot_type.lower() == 'curvature':
            out = self.impulse_names[:]
        elif plot_type.lower() == 'interaction_surface':
            out = [x for x in itertools.chain.from_iterable(
                itertools.combinations(self.impulse_names, 2)
            )]
        else:
            raise ValueError('Plot type "%s" not supported.' % plot_type)

        return out

    def plot_impulse_name_to_1hot(self, name):
        names = name.split(':')
        ix = names2ix(names, self.impulse_names)
        one_hot = np.zeros(len(self.impulse_names), self.FLOAT_NP)
        one_hot[ix] = 1

        return one_hot

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
        if rangf_vals is None:
            rangf_keys = [None]
            rangf_vals = [self.gf_defaults[0]]
            if plot_rangf:
                for i in range(len(self.rangf)):
                    if type(self).__name__.startswith('CDRNN') or self.t.has_coefficient(
                            self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                        for k in self.rangf_map[i].keys():
                            rangf_keys.append(str(k))
                            rangf_vals.append(np.concatenate(
                                [self.gf_defaults[0, :i], [self.rangf_map[i][k]], self.gf_defaults[0, i + 1:]], axis=0))
            rangf_vals = np.stack(rangf_vals, axis=0)

        if name == 'rate':
            impulse_one_hot = np.zeros(len(self.impulse_names))
        else:
            impulse_one_hot = self.plot_impulse_name_to_1hot(name)

        fd = {
            self.support_start: 0.,
            self.n_time_units: n_time_units,
            self.n_time_points: n_time_points,
            self.max_tdelta_batch: n_time_points,
            self.gf_y: rangf_vals,
            self.plot_impulse_1hot: impulse_one_hot,
            self.t_interaction: t_interaction,
            self.training: not self.predict_mode,
        }

        if plot_type.lower().startswith('irf_1d'):
            if name == 'rate':
                irf_1d_support = self.irf_1d_rate_support
                irf_1d = self.irf_1d_rate_plot
            else:
                irf_1d_support = self.irf_1d_support
                irf_1d = self.irf_1d_plot
            out = self.sess.run([irf_1d_support, irf_1d], feed_dict=fd)

            return out
        elif plot_type.lower().startswith('irf_surface'):
            if name == 'rate':
                irf_surface_meshgrid = self.irf_surface_rate_meshgrid
                irf_surface = self.irf_surface_rate_plot
            else:
                irf_surface_meshgrid = self.irf_surface_meshgrid
                irf_surface = self.irf_surface_plot
            out = self.sess.run([irf_surface_meshgrid, irf_surface], feed_dict=fd)
        elif plot_type.lower().startswith('curvature'):
            assert not name == 'rate', 'Curvature plots are not available for "rate" (deconvolutional intercept).'
            out = self.sess.run([self.curvature_support, self.curvature_plot], feed_dict=fd)
        elif plot_type.lower().startswith('interaction_surface'):
            names = name.split(':')
            assert len(names) == 2, 'Interaction surface plots require interactions of order 2'
            impulse_one_hot1 = self.plot_impulse_name_to_1hot(names[0])
            impulse_one_hot2 = self.plot_impulse_name_to_1hot(names[1])

            fd[self.plot_impulse_1hot] = impulse_one_hot1
            fd[self.plot_impulse_1hot_2] = impulse_one_hot2

            out = self.sess.run(self.interaction_surface_plot, feed_dict=fd)
        else:
            raise ValueError('Plot type "%s" not supported.' % plot_type)

        return out

    def report_settings(self, indent=0):
        out = super(CDRNN, self).report_settings(indent=indent)
        for kwarg in CDRNN_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' % (kwarg.key, "\"%s\"" % val if isinstance(val, str) else val)

        out += '\n'

        return out

    def summary(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a summary of the fitted model.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; ignored in CDRNN models.
        :param n_samples: ``int`` or ``None``; ignored in CDRNN models.
        :param integral_n_time_units: ``float``; ignored in CDRNN models.
        :return: ``str``; the model summary
        """

        out = '  ' * indent + '*' * 100 + '\n\n'
        out += ' ' * indent + '############################\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '#    CDRNN MODEL SUMMARY    #\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '############################\n\n\n'

        out += self.initialization_summary(indent=indent + 2)
        out += '\n'
        out += self.training_evaluation_summary(indent=indent + 2)
        out += '\n'
        out += self.convergence_summary(indent=indent + 2)
        out += '\n'
        out += '  ' * indent + '*' * 100 + '\n\n'

        return out
