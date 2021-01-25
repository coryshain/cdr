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

        for kwarg in CDRNN._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

    def _initialize_metadata(self):
        super(CDRNN, self)._initialize_metadata()

        self.is_cdrnn = True

        self.use_batch_normalization = bool(self.batch_normalization_decay)
        self.use_layer_normalization = bool(self.layer_normalization_type)

        assert not (self.use_batch_normalization and self.use_layer_normalization), 'Cannot batch normalize and layer normalize the same model.'

        self.normalize_activations = self.use_batch_normalization or self.use_layer_normalization

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

        if self.n_units_error_params_fn:
            if isinstance(self.n_units_error_params_fn, str):
                self.n_units_error_params_fn = [int(x) for x in self.n_units_error_params_fn.split()]
            elif isinstance(self.n_units_error_params_fn, int):
                if self.n_layers_error_params_fn is None:
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

        if self.n_units_hidden_state is None:
            if self.n_units_irf:
                self.n_units_hidden_state = self.n_units_irf[0]
            elif self.n_units_input_projection:
                self.n_units_hidden_state = self.n_units_input_projection[-1]
            elif self.n_units_rnn:
                self.n_units_hidden_state = self.n_units_rnn[-1]
            else:
                raise ValueError("Cannot infer size of hidden state. Units are not specified for hidden state, IRF, input projection, or RNN projection.")
        elif isinstance(self.n_units_hidden_state, str):
            if self.n_units_hidden_state.lower() == 'infer':
                self.n_units_hidden_state = len(self.impulse_names) + len(self.ablated) + 1
            else:
                self.n_units_hidden_state = int(self.n_units_hidden_state)

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

    def _initialize_cdrnn_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
                    else:
                        rangf_1hot = tf.zeros(
                            [tf.shape(self.gf_y)[0], 0],
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

                self.plot_mean_as_reference = tf.placeholder_with_default(
                    True,
                    shape=[],
                    name='plot_mean_as_reference'
                )

                def get_plot_reference_mean():
                    if self.center_inputs:
                        plot_impulse_base_default = tf.zeros(
                            [len(self.impulse_names)],
                            dtype=self.FLOAT_TF
                        )
                    else:
                        plot_impulse_base_default = tf.convert_to_tensor(
                            [self.impulse_means[x] for x in self.impulse_names],
                            dtype=self.FLOAT_TF
                        )
                    plot_impulse_center_default = tf.zeros(
                        [len(self.impulse_names)],
                        dtype=self.FLOAT_TF
                    )

                    return plot_impulse_base_default, plot_impulse_center_default

                def get_plot_reference_zero():
                    plot_impulse_base_default = tf.zeros(
                        [len(self.impulse_names)],
                        dtype=self.FLOAT_TF
                    )
                    if self.center_inputs:
                        plot_impulse_center_default = tf.zeros(
                            [len(self.impulse_names)],
                            dtype=self.FLOAT_TF
                        )
                    else:
                        plot_impulse_center_default = tf.convert_to_tensor(
                            [self.impulse_means[x] for x in self.impulse_names],
                            dtype=self.FLOAT_TF
                        )

                    return plot_impulse_base_default, plot_impulse_center_default

                plot_impulse_base_default, plot_impulse_center_default = tf.cond(
                    self.plot_mean_as_reference,
                    get_plot_reference_mean,
                    get_plot_reference_zero
                )

                plot_impulse_offset_default = tf.ones(
                    [len(self.impulse_names)],
                    dtype=self.FLOAT_TF
                )
                if not self.rescale_inputs:
                    plot_impulse_offset_default *= self.impulse_sds_arr

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
                    name='plot_impulse_1hot_2'
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
                    scale = self.nn_regularizer_scale
                    if self.scale_regularizer_with_data:
                        scale *= self.minibatch_size * self.minibatch_scale
                    if self.nn_regularizer_name == 'l1_l2_regularizer':
                        self.nn_regularizer = getattr(tf.contrib.layers, self.nn_regularizer_name)(
                            scale,
                            scale
                        )
                    else:
                        self.nn_regularizer = getattr(tf.contrib.layers, self.nn_regularizer_name)(scale)

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
                # CDRNN MODEL

                # FEEDFORWARD ENCODER
                input_projection_layers = []
                for l in range(self.n_layers_input_projection + 1):
                    if l < self.n_layers_input_projection:
                        units = self.n_units_input_projection[l]
                        activation = self.input_projection_inner_activation
                        dropout = self.input_projection_dropout_rate
                        bn = self.batch_normalization_decay
                        ln = self.layer_normalization_type
                        use_bias = True
                    else:
                        units = self.n_units_hidden_state
                        activation = self.input_projection_activation
                        dropout = None
                        bn = None
                        ln = None
                        use_bias = False

                    projection = self.initialize_feedforward(
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        dropout=dropout,
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

                        projection = self.initialize_feedforward(
                            units=units,
                            use_bias=use_bias,
                            activation=activation,
                            dropout=None,
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
                    else:
                        units = 1
                        activation = self.irf_activation
                        dropout = None
                        bn = None
                        ln = None
                        use_bias = False
                        final = True

                    projection = self.initialize_feedforward(
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        dropout=dropout,
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

                # ERROR PARAMS FN
                error_params_fn_layers = []
                for l in range(self.n_layers_error_params_fn):
                    units = self.n_units_error_params_fn[l]
                    activation = self.error_params_fn_inner_activation
                    dropout = self.error_params_fn_dropout_rate
                    if self.normalize_error_params_fn:
                        bn = self.batch_normalization_decay
                        ln = self.layer_normalization_type
                    else:
                        bn = None
                        ln = None
                    use_bias = True

                    projection = self.initialize_feedforward(
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        dropout=dropout,
                        batch_normalization_decay=bn,
                        layer_normalization_type=ln,
                        name='error_params_fn_l%s' % (l + 1)
                    )
                    self.layers.append(projection)

                    self.regularizable_layers.append(projection)
                    error_params_fn_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                self.error_params_fn_layers = error_params_fn_layers
                self.error_params_fn = compose_lambdas(error_params_fn_layers)

                units = 1
                if self.asymmetric_error:
                    units += 2
                n = len(self.impulse_indices)
                if n > 1:
                    if n == 2:
                        units += 1
                    else:
                        units += n

                if self.asymmetric_error:
                    denom = self.n_units_hidden_state
                    if self.n_units_error_params_fn:
                        denom = max(denom, self.n_units_error_params_fn[-1])
                    self.y_skewness_coef = tf.Variable(
                        1. / denom,
                        name='skewness_coef'
                    )
                    self.y_tailweight_coef = tf.Variable(
                        1. / denom,
                        name='tailweight_coef'
                    )

                projection = self.initialize_feedforward(
                    units=units,
                    use_bias=False,
                    activation=self.error_params_fn_activation,
                    name='error_params_fn_l%s' % (self.n_layers_error_params_fn + 1),
                    final=final
                )
                self.layers.append(projection)

                self.error_params_final_fn = compose_lambdas([projection])

                # # ERROR PARAM BIASES
                # self.error_params_b, self.error_params_b_summary = self.initialize_error_params_biases(ran_gf=None)

                # INTERCEPT
                if self.has_intercept[None]:
                    self.intercept_fixed, self.intercept_fixed_summary = self.initialize_intercept(ran_gf=None)
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_fixed_summary,
                        collections=['params']
                    )
                    self._regularize(self.intercept_fixed, type='intercept', var_name=reg_name('intercept'))
                    if self.convergence_basis.lower() == 'parameters':
                        self._add_convergence_tracker(self.intercept_fixed_summary, 'intercept_fixed')

                else:
                    self.intercept_fixed = 0.
                self.intercept = self.intercept_fixed
                self.intercept_summary = self.intercept_fixed_summary

                # INTERCEPT DELTA
                if self.nonstationary_intercept:
                    intercept_l1_W, intercept_l1_W_summary = self.initialize_intercept_l1_weights()
                    self.intercept_l1_W = intercept_l1_W
                    if self.irf_l1_use_bias:
                        if self.normalize_irf_l1 and self.normalize_activations:
                            self.intercept_l1_normalization_layer = self.initialize_intercept_l1_normalization()
                            self.layers.append(self.intercept_l1_normalization_layer)
                            # if self.normalize_after_activation and False:
                            if self.normalize_after_activation:
                                intercept_l1_b, intercept_l1_b_summary = self.initialize_intercept_l1_biases()
                            else:
                                intercept_l1_b = tf.zeros_like(self.intercept_l1_W)
                                intercept_l1_b_summary = tf.zeros_like(self.intercept_l1_W)
                        else:
                            intercept_l1_b, intercept_l1_b_summary = self.initialize_intercept_l1_biases()
                    self.intercept_l1_b = intercept_l1_b

                    intercept_delta_layers = []
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
                        else:
                            units = 1
                            activation = self.irf_activation
                            dropout = None
                            bn = None
                            ln = None
                            use_bias = False
                            final = True

                        projection = self.initialize_feedforward(
                            units=units,
                            use_bias=use_bias,
                            activation=activation,
                            dropout=dropout,
                            batch_normalization_decay=bn,
                            layer_normalization_type=ln,
                            name='intercept_l%s' % (l + 1),
                            final=final
                        )
                        self.layers.append(projection)

                        if l < self.n_layers_irf:
                            self.regularizable_layers.append(projection)
                        intercept_delta_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                    intercept_delta = compose_lambdas(intercept_delta_layers)

                    self.intercept_delta_layers = intercept_delta_layers
                    self.intercept_delta_fn = intercept_delta

                # RANDOM EFFECTS
                self.intercept_random = {}
                self.intercept_random_summary = {}
                self.intercept_random_means = {}
                # self.error_params_b_ran = []
                # self.error_params_b_ran_matrix = []
                self.rnn_h_ran_matrix = [[] for l in range(self.n_layers_rnn)]
                self.rnn_h_ran = [[] for l in range(self.n_layers_rnn)]
                self.rnn_c_ran_matrix = [[] for l in range(self.n_layers_rnn)]
                self.rnn_c_ran = [[] for l in range(self.n_layers_rnn)]
                self.h_bias_ran_matrix = []
                self.h_bias_ran = []
                # self.irf_l1_W_ran_matrix = []
                # self.irf_l1_W_ran = []
                # self.irf_l1_b_ran_matrix = []
                # self.irf_l1_b_ran = []
                # if self.nonstationary_intercept:
                #     self.intercept_l1_W_ran_matrix = []
                #     self.intercept_l1_W_ran = []
                #     self.intercept_l1_b_ran_matrix = []
                #     self.intercept_l1_b_ran = []

                gf_y_src = self.gf_y

                if self.ranef_dropout_rate:
                    def gf_train_fn(gf_y_cur=gf_y_src):
                        dropout_mask = tf.cast(tf.random_uniform(tf.shape(gf_y_cur)) > self.ranef_dropout_rate, tf.bool)
                        alt = tf.zeros_like(gf_y_cur)
                        return tf.where(dropout_mask, gf_y_cur, alt)

                    def gf_eval_fn(gf_y_cur=gf_y_src):
                        return gf_y_cur

                    gf_y_src = tf.cond(self.training, gf_train_fn, gf_eval_fn)

                for j in range(len(self.rangf)):
                    gf = self.rangf[j]
                    levels_ix = np.arange(self.rangf_n_levels[j] - 1)
                    gf_y = gf_y_src[:, j]

                    if self.has_intercept[gf]:
                        # Random intercepts
                        intercept_random, intercept_random_summary = self.initialize_intercept(ran_gf=gf)

                        intercept_random_summary = intercept_random

                        intercept_random_means = tf.reduce_mean(intercept_random, axis=0, keepdims=True)
                        intercept_random_summary_means = tf.reduce_mean(intercept_random_summary, axis=0, keepdims=True)

                        intercept_random -= intercept_random_means
                        intercept_random_summary -= intercept_random_summary_means

                        self._regularize(intercept_random, type='ranef', var_name=reg_name('intercept_by_%s' % gf))

                        intercept_random = self._scatter_along_axis(
                            levels_ix,
                            intercept_random,
                            [self.rangf_n_levels[j]]
                        )
                        intercept_random_summary = self._scatter_along_axis(
                            levels_ix,
                            intercept_random_summary,
                            [self.rangf_n_levels[j]]
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
                            
                        # # Random error bias offsets
                        # error_params_b_ran_matrix_cur, error_params_b_ran_matrix_cur_summary = self.initialize_error_params_biases(ran_gf=gf)
                        # error_params_b_ran_matrix_cur -= tf.reduce_mean(error_params_b_ran_matrix_cur, axis=0, keepdims=True)
                        # error_params_b_ran_matrix_cur_summary -= tf.reduce_mean(error_params_b_ran_matrix_cur_summary, axis=0,
                        #                                                 keepdims=True)
                        # self._regularize(error_params_b_ran_matrix_cur, type='ranef',
                        #                  var_name=reg_name('error_params_b_by_%s' % (sn(gf))))
                        #
                        # if self.log_random:
                        #     tf.summary.histogram(
                        #         sn('by_%s/h' % sn(gf)),
                        #         error_params_b_ran_matrix_cur_summary,
                        #         collections=['random']
                        #     )
                        #
                        # error_params_b_ran_matrix_cur = tf.concat(
                        #     [
                        #         error_params_b_ran_matrix_cur,
                        #         tf.zeros([1, self.n_units_hidden_state])
                        #     ],
                        #     axis=0
                        # )
                        #
                        # error_params_b_ran = tf.gather(error_params_b_ran_matrix_cur, gf_y)
                        #
                        # self.error_params_b_ran_matrix.append(error_params_b_ran_matrix_cur)
                        # self.error_params_b_ran.append(error_params_b_ran)
                        #
                        # self.error_params_b += tf.expand_dims(error_params_b_ran, axis=-2)

                        # Random rnn initialization offsets
                        for l in range(self.n_layers_rnn):
                            rnn_h_ran_matrix_cur, rnn_h_ran_matrix_cur_summary = self.initialize_rnn_h(l, ran_gf=gf)
                            rnn_h_ran_matrix_cur -= tf.reduce_mean(rnn_h_ran_matrix_cur, axis=0, keepdims=True)
                            rnn_h_ran_matrix_cur_summary -= tf.reduce_mean(rnn_h_ran_matrix_cur_summary, axis=0, keepdims=True)
                            self._regularize(rnn_h_ran_matrix_cur, type='ranef', var_name=reg_name('rnn_h_ran_l%d_by_%s' % (l, sn(gf))))

                            if self.log_random:
                                tf.summary.histogram(
                                    sn('by_%s/rnn_h_l%d' % (sn(gf), l+1)),
                                    rnn_h_ran_matrix_cur_summary,
                                    collections=['random']
                                )

                            rnn_h_ran_matrix_cur = tf.concat(
                                [
                                    rnn_h_ran_matrix_cur,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            rnn_h_ran = tf.gather(rnn_h_ran_matrix_cur, gf_y)

                            self.rnn_h_ran_matrix[l].append(rnn_h_ran_matrix_cur)
                            self.rnn_h_ran[l].append(rnn_h_ran)

                            self.rnn_h_init[l] += rnn_h_ran

                            rnn_c_ran_matrix_cur, rnn_c_ran_matrix_cur_summary = self.initialize_rnn_c(l, ran_gf=gf)
                            rnn_c_ran_matrix_cur -= tf.reduce_mean(rnn_c_ran_matrix_cur, axis=0, keepdims=True)
                            rnn_c_ran_matrix_cur_summary -= tf.reduce_mean(rnn_c_ran_matrix_cur_summary, axis=0, keepdims=True)
                            self._regularize(rnn_c_ran_matrix_cur, type='ranef', var_name=reg_name('rnn_c_ran_l%d_by_%s' % (l+1, sn(gf))))

                            if self.log_random:
                                tf.summary.histogram(
                                    sn('by_%s/rnn_c_l%d' % (sn(gf), l+1)),
                                    rnn_c_ran_matrix_cur_summary,
                                    collections=['random']
                                )

                            rnn_c_ran_matrix_cur = tf.concat(
                                [
                                    rnn_c_ran_matrix_cur,
                                    tf.zeros([1, self.n_units_rnn[l]])
                                ],
                                axis=0
                            )
                            rnn_c_ran = tf.gather(rnn_c_ran_matrix_cur, gf_y)

                            self.rnn_c_ran_matrix[l].append(rnn_c_ran_matrix_cur)
                            self.rnn_c_ran[l].append(rnn_c_ran)

                            self.rnn_c_init[l] += rnn_c_ran

                        # Random hidden state offsets
                        h_bias_ran_matrix_cur, h_bias_ran_matrix_cur_summary = self.initialize_h_bias(ran_gf=gf)
                        h_bias_ran_matrix_cur -= tf.reduce_mean(h_bias_ran_matrix_cur, axis=0, keepdims=True)
                        h_bias_ran_matrix_cur_summary -= tf.reduce_mean(h_bias_ran_matrix_cur_summary, axis=0, keepdims=True)
                        self._regularize(h_bias_ran_matrix_cur, type='ranef', var_name=reg_name('h_bias_by_%s' % (sn(gf))))

                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/h' % sn(gf)),
                                h_bias_ran_matrix_cur_summary,
                                collections=['random']
                            )

                        h_bias_ran_matrix_cur = tf.concat(
                            [
                                h_bias_ran_matrix_cur,
                                tf.zeros([1, self.n_units_hidden_state])
                            ],
                            axis=0
                        )

                        h_bias_ran = tf.gather(h_bias_ran_matrix_cur, gf_y)

                        self.h_bias_ran_matrix.append(h_bias_ran_matrix_cur)
                        self.h_bias_ran.append(h_bias_ran)

                        self.h_bias += tf.expand_dims(h_bias_ran, axis=-2)

                        # if self.nonstationary_intercept:
                        #     # Random intercept L1 weights
                        #     intercept_l1_W_ran_matrix_cur, intercept_l1_W_ran_matrix_cur_summary, = self.initialize_intercept_l1_weights(ran_gf=gf)
                        #     intercept_l1_W_ran_matrix_cur -= tf.reduce_mean(intercept_l1_W_ran_matrix_cur, axis=0, keepdims=True)
                        #     intercept_l1_W_ran_matrix_cur_summary -= tf.reduce_mean(intercept_l1_W_ran_matrix_cur_summary, axis=0, keepdims=True)
                        #     self._regularize(intercept_l1_W_ran_matrix_cur, type='ranef', var_name=reg_name('intercept_l1_W_bias_by_%s' % (sn(gf))))
                        #
                        #     if self.log_random:
                        #         tf.summary.histogram(
                        #             sn('by_%s/intercept_l1_W' % sn(gf)),
                        #             intercept_l1_W_ran_matrix_cur_summary,
                        #             collections=['random']
                        #         )
                        #
                        #     intercept_l1_W_ran_matrix_cur = tf.concat(
                        #         [
                        #             intercept_l1_W_ran_matrix_cur,
                        #             tf.zeros([1, self.n_units_hidden_state])
                        #         ],
                        #         axis=0
                        #     )
                        #
                        #     intercept_l1_W_bias_ran = tf.gather(intercept_l1_W_ran_matrix_cur, gf_y)
                        #
                        #     self.intercept_l1_W_ran_matrix.append(intercept_l1_W_ran_matrix_cur)
                        #     self.intercept_l1_W_ran.append(intercept_l1_W_bias_ran)
                        #
                        #     self.intercept_l1_W += tf.expand_dims(intercept_l1_W_bias_ran, axis=-2)
                        #
                        #     # Random intercept L1 biases
                        #     if self.irf_l1_use_bias:
                        #         intercept_l1_b_ran_matrix_cur, intercept_l1_b_ran_matrix_cur_summary = self.initialize_intercept_l1_biases(ran_gf=gf)
                        #         intercept_l1_b_ran_matrix_cur -= tf.reduce_mean(intercept_l1_b_ran_matrix_cur, axis=0, keepdims=True)
                        #         intercept_l1_b_ran_matrix_cur_summary -= tf.reduce_mean(intercept_l1_b_ran_matrix_cur_summary, axis=0, keepdims=True)
                        #         self._regularize(intercept_l1_b_ran_matrix_cur, type='ranef', var_name=reg_name('intercept_l1_b_bias_by_%s' % (sn(gf))))
                        #
                        #         if self.log_random:
                        #             tf.summary.histogram(
                        #                 sn('by_%s/intercept_l1_b' % sn(gf)),
                        #                 intercept_l1_b_ran_matrix_cur_summary,
                        #                 collections=['random']
                        #             )
                        #
                        #         intercept_l1_b_ran_matrix_cur = tf.concat(
                        #             [
                        #                 intercept_l1_b_ran_matrix_cur,
                        #                 tf.zeros([1, self.n_units_hidden_state])
                        #             ],
                        #             axis=0
                        #         )
                        #
                        #         intercept_l1_b_bias_ran = tf.gather(intercept_l1_b_ran_matrix_cur, gf_y)
                        #
                        #         self.intercept_l1_b_ran_matrix.append(intercept_l1_b_ran_matrix_cur)
                        #         self.intercept_l1_b_ran.append(intercept_l1_b_bias_ran)
                        #
                        #         self.intercept_l1_b += tf.expand_dims(intercept_l1_b_bias_ran, axis=-2)
                        #
                        # # Random IRF L1 weights
                        # irf_l1_W_ran_matrix_cur, irf_l1_W_ran_matrix_cur_summary, = self.initialize_irf_l1_weights(ran_gf=gf)
                        # irf_l1_W_ran_matrix_cur -= tf.reduce_mean(irf_l1_W_ran_matrix_cur, axis=0, keepdims=True)
                        # irf_l1_W_ran_matrix_cur_summary -= tf.reduce_mean(irf_l1_W_ran_matrix_cur_summary, axis=0, keepdims=True)
                        # self._regularize(irf_l1_W_ran_matrix_cur, type='ranef', var_name=reg_name('irf_l1_W_bias_by_%s' % (sn(gf))))
                        #
                        # if self.log_random:
                        #     tf.summary.histogram(
                        #         sn('by_%s/irf_l1_W' % sn(gf)),
                        #         irf_l1_W_ran_matrix_cur_summary,
                        #         collections=['random']
                        #     )
                        #
                        # irf_l1_W_ran_matrix_cur = tf.concat(
                        #     [
                        #         irf_l1_W_ran_matrix_cur,
                        #         tf.zeros([1, self.n_units_hidden_state])
                        #     ],
                        #     axis=0
                        # )
                        #
                        # irf_l1_W_bias_ran = tf.gather(irf_l1_W_ran_matrix_cur, gf_y)
                        #
                        # self.irf_l1_W_ran_matrix.append(irf_l1_W_ran_matrix_cur)
                        # self.irf_l1_W_ran.append(irf_l1_W_bias_ran)
                        #
                        # self.irf_l1_W += tf.expand_dims(irf_l1_W_bias_ran, axis=-2)
                        #
                        # # Random IRF L1 biases
                        # if self.irf_l1_use_bias:
                        #     irf_l1_b_ran_matrix_cur, irf_l1_b_ran_matrix_cur_summary = self.initialize_irf_l1_biases(ran_gf=gf)
                        #     irf_l1_b_ran_matrix_cur -= tf.reduce_mean(irf_l1_b_ran_matrix_cur, axis=0, keepdims=True)
                        #     irf_l1_b_ran_matrix_cur_summary -= tf.reduce_mean(irf_l1_b_ran_matrix_cur_summary, axis=0, keepdims=True)
                        #     self._regularize(irf_l1_b_ran_matrix_cur, type='ranef', var_name=reg_name('irf_l1_b_bias_by_%s' % (sn(gf))))
                        #
                        #     if self.log_random:
                        #         tf.summary.histogram(
                        #             sn('by_%s/irf_l1_b' % sn(gf)),
                        #             irf_l1_b_ran_matrix_cur_summary,
                        #             collections=['random']
                        #         )
                        #
                        #     irf_l1_b_ran_matrix_cur = tf.concat(
                        #         [
                        #             irf_l1_b_ran_matrix_cur,
                        #             tf.zeros([1, self.n_units_hidden_state])
                        #         ],
                        #         axis=0
                        #     )
                        #
                        #     irf_l1_b_bias_ran = tf.gather(irf_l1_b_ran_matrix_cur, gf_y)
                        #
                        #     self.irf_l1_b_ran_matrix.append(irf_l1_b_ran_matrix_cur)
                        #     self.irf_l1_b_ran.append(irf_l1_b_bias_ran)
                        #
                        #     self.irf_l1_b += tf.expand_dims(irf_l1_b_bias_ran, axis=-2)

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

    def _apply_model(self, X, t_delta, time_X=None, time_X_mask=None, time_y=None, plot_mode=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if time_X is None:
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
                    # time_X = tf.zeros(time_X_shape, dtype=self.FLOAT_TF)
                    time_X = tf.ones(time_X_shape, dtype=self.FLOAT_TF)
                    time_X_mean = self.time_X_mean
                    time_X *= time_X_mean

                if self.nonstationary_intercept:
                    if time_y is None:
                        time_y = tf.ones(tf.shape(X)[0], dtype=self.FLOAT_TF)
                        time_y_mean = self.time_y_mean
                        time_y *= time_y_mean
                    if len(time_y.shape) == 1:
                        time_y = time_y[..., None]

                if self.center_time:
                    time_X -= self.time_X_mean
                    t_delta -= self.t_delta_mean
                    if self.nonstationary_intercept:
                        time_y -= self.time_y_mean

                if self.rescale_time:
                    time_X /= self.time_X_sd
                    t_delta /= self.t_delta_sd
                    if self.nonstationary_intercept:
                        time_y /= self.time_y_sd

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

                        if time_X.shape[-1] > 1:
                            time_X_cur = time_X[..., ix[0]:ix[0]+1]
                        else:
                            time_X_cur = time_X

                        if time_X_mask is not None and time_X_mask.shape[-1] > 1:
                            time_X_mask_cur = time_X_mask[..., ix[0]]
                        else:
                            time_X_mask_cur = time_X_mask

                        X_cdrnn.append(X_cur)
                        t_delta_cdrnn.append(t_delta_cur)
                        time_X_cdrnn.append(time_X_cur)
                        if time_X_mask is not None:
                            time_X_mask_cdrnn.append(time_X_mask_cur)

                    X_cdrnn = tf.concat(X_cdrnn, axis=1)
                    t_delta_cdrnn = tf.concat(t_delta_cdrnn, axis=1)
                    time_X_cdrnn = tf.concat(time_X_cdrnn, axis=1)
                    if time_X_mask is not None:
                        time_X_mask_cdrnn = tf.concat(time_X_mask_cdrnn, axis=1)

                    sort_ix = tf.contrib.framework.argsort(tf.squeeze(time_X_cdrnn, axis=-1), axis=1)
                    B_ix = tf.tile(
                        tf.range(B)[..., None],
                        [1, T * len(self.impulse_indices)]
                    )
                    gather_ix = tf.stack([B_ix, sort_ix], axis=-1)

                    X = tf.gather_nd(X_cdrnn, gather_ix)
                    t_delta = tf.gather_nd(t_delta_cdrnn, gather_ix)
                    time_X = tf.gather_nd(time_X_cdrnn, gather_ix)
                    if time_X_mask is not None:
                        time_X_mask = tf.gather_nd(time_X_mask_cdrnn, gather_ix)
                else:
                    t_delta = t_delta[..., :1]
                    time_X = time_X[..., :1]
                    if time_X_mask is not None and len(time_X_mask.shape) == 3:
                        time_X_mask = time_X_mask[..., 0]

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
                    time_X = tf.cond(
                        self.training,
                        lambda: tf.random_normal(tf.shape(time_X), time_X, jitter_sd),
                        lambda: time_X
                    )

                X = tf.concat([X, time_X], axis=-1)

                if self.predictor_dropout_rate:
                    X = tf.layers.dropout(
                        X,
                        rate=self.predictor_dropout_rate,
                        training=self.training
                    )

                if self.event_dropout_rate:
                    X_shape = tf.shape(X)
                    noise_shape = []
                    for j in range(len(X.shape) - 1):
                        try:
                            s = int(X.shape[j])
                        except TypeError:
                            s = X_shape[j]
                        noise_shape.append(s)
                    noise_shape.append(1)

                    mask_is_none = time_X_mask is None

                    def train_fn(inputs=X, noise_shape=noise_shape, mask=time_X_mask):
                        dropout_mask = tf.cast(tf.random_uniform(noise_shape) > self.event_dropout_rate,
                                               dtype=self.FLOAT_TF)
                        inputs_out = inputs * dropout_mask
                        if mask is not None:
                            mask_out = mask * dropout_mask[..., 0]
                        else:
                            mask_out = tf.zeros(tf.shape(inputs)[:-1])

                        return inputs_out, mask_out

                    def eval_fn(inputs=X, mask=time_X_mask):
                        if mask is None:
                            mask = tf.zeros(tf.shape(inputs)[:-1])
                        return inputs, mask

                    X, time_X_mask = tf.cond(self.training, train_fn, eval_fn)

                    if mask_is_none:
                        time_X_mask = None

                # Compute hidden state
                h = self.h_bias
                W = self.irf_l1_W
                b = self.irf_l1_b
                if self.nonstationary_intercept:
                    W_int = self.intercept_l1_W
                    b_int = self.intercept_l1_b

                # If plotting, tile out random effects
                if plot_mode:
                    R = tf.shape(self.gf_y)[0]
                    B = tf.shape(X)[0]
                    tile_ix = tf.tile(tf.range(R)[..., None], [1, B])
                    tile_ix = tf.reshape(tile_ix, [-1])
                    h = tf.gather(h, tile_ix, axis=0)
                    # W = tf.gather(W, tile_ix, axis=0)
                    # b = tf.gather(b, tile_ix, axis=0)
                    X = tf.tile(X, [R, 1 ,1])
                    time_X = tf.tile(time_X, [R, 1 ,1])
                    t_delta = tf.tile(t_delta, [R, 1 ,1])
                    # if self.nonstationary_intercept:
                    #     tile_ix_int = tf.range(R)
                    #     W_int = tf.gather(W_int, tile_ix_int, axis=0)
                    #     b_int = tf.gather(b_int, tile_ix_int, axis=0)

                if self.nonstationary_intercept:
                    intercept_delta_l1 = W_int * time_y + b_int
                    if self.normalize_after_activation:
                        intercept_delta_l1 = get_activation(self.irf_inner_activation, session=self.sess)(intercept_delta_l1)
                    if self.normalize_irf_l1 and self.irf_l1_use_bias and self.normalize_activations:
                        intercept_delta_l1 = self.intercept_l1_normalization_layer(intercept_delta_l1)
                    if not self.normalize_after_activation:
                        intercept_delta_l1 = get_activation(self.irf_inner_activation, session=self.sess)(intercept_delta_l1)
                    intercept_delta = self.intercept_delta_fn(intercept_delta_l1)

                h_in = self.input_projection_fn(X)
                if self.h_in_noise_sd:
                    def h_in_train_fn(h_in=h_in):
                        return tf.random_normal(tf.shape(h_in), h_in, stddev=self.h_in_noise_sd)
                    def h_in_eval_fn(h_in=h_in):
                        return h_in
                    h_in = tf.cond(self.training, h_in_train_fn, h_in_eval_fn)
                if self.h_in_dropout_rate:
                    h_in = get_dropout(self.h_in_dropout_rate, training=self.training, session=self.sess)(h_in)
                h += h_in

                if self.n_layers_rnn:
                    rnn_hidden, rnn_cell = self._rnn_encoder(
                        X,
                        times=time_X,
                        mask=time_X_mask,
                        plot_mode=plot_mode
                    )
                    h_rnn = self.rnn_projection_fn(rnn_hidden[-1])

                    if self.rnn_dropout_rate:
                        h_rnn_shape = tf.shape(h_rnn)
                        noise_shape = []
                        for j in range(len(h_rnn.shape) - 1):
                            try:
                                s = int(h_rnn.shape[j])
                            except TypeError:
                                s = h_rnn_shape[j]
                            noise_shape.append(s)
                        noise_shape.append(1)

                        def h_rnn_train_fn(inputs=h_rnn, noise_shape=noise_shape):
                            dropout_mask = tf.cast(tf.random_uniform(noise_shape) > self.rnn_dropout_rate, dtype=self.FLOAT_TF)
                            inputs_out = inputs * dropout_mask

                            return inputs_out

                        def h_rnn_eval_fn(inputs=h_rnn):
                            return inputs

                        h_rnn = tf.cond(self.training, h_rnn_train_fn, h_rnn_eval_fn)

                    if self.h_rnn_noise_sd:
                        def h_rnn_train_fn(h_rnn=h_rnn):
                            return tf.random_normal(tf.shape(h_rnn), h_rnn, stddev=self.h_rnn_noise_sd)
                        def h_rnn_eval_fn(h_rnn=h_rnn):
                            return h_rnn
                        h_rnn = tf.cond(self.training, h_rnn_train_fn, h_rnn_eval_fn)
                    if self.h_rnn_dropout_rate:
                        h_rnn = get_dropout(self.h_rnn_dropout_rate, training=self.training, session=self.sess)(h_rnn)

                    h += h_rnn
                else:
                    h_rnn = rnn_hidden = rnn_cell = None

                if self.normalize_after_activation:
                    h = get_activation(self.hidden_state_activation, session=self.sess)(h)
                if self.normalize_h and self.normalize_activations:
                    h = self.h_normalization_layer(h)
                if not self.normalize_after_activation:
                    h = get_activation(self.hidden_state_activation, session=self.sess)(h)

                # Compute response
                Wb_proj = self.hidden_state_to_irf_l1(h)
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

                y = self.irf(irf_l1)
                if time_X_mask is not None:
                    y *= time_X_mask[..., None]
                y = tf.reduce_sum(y, axis=1)

                error_params = self.error_params_fn(h)
                # Max pooling over time
                error_params = tf.reduce_max(error_params, axis=-2)
                error_params = self.error_params_final_fn(error_params)

                y_sd_delta = error_params[..., 0]
                if self.asymmetric_error:
                    y_skewness_delta = error_params[..., 1]
                    y_tailweight_delta = error_params[..., 2]
                else:
                    y_skewness_delta = y_tailweight_delta = None

                out = {
                    'y': y,
                    'y_sd_delta': y_sd_delta,
                    'y_skewness_delta': y_skewness_delta,
                    'y_tailweight_delta': y_tailweight_delta,
                    'rnn_hidden': rnn_hidden,
                    'rnn_cell': rnn_cell,
                    'h_rnn': h_rnn,
                    'h_in': h_in,
                    'h': h,
                    'time_X_mask': time_X_mask
                }
                if self.nonstationary_intercept:
                    out['intercept_delta'] = intercept_delta

                return out

    def _construct_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                model_dict = self._apply_model(
                    self.X_processed,
                    self.t_delta,
                    time_X=self.time_X,
                    time_X_mask=self.time_X_mask,
                    plot_mode=False
                )

                y = model_dict['y']
                y_sd_delta = model_dict['y_sd_delta']
                y_skewness_delta = model_dict['y_skewness_delta']
                y_tailweight_delta = model_dict['y_tailweight_delta']

                y = tf.squeeze(y, axis=-1)
                y += self.intercept
                if self.nonstationary_intercept:
                    intercept_delta = model_dict['intercept_delta']
                    y += intercept_delta

                self.out = y
                # Hack needed for MAP evaluation of CDRNNBayes
                self.out_mean = self.out

                ema_rate = self.ema_decay
                if ema_rate is None:
                    ema_rate = 0.

                h_rnn = model_dict['h_rnn']
                self.rnn_h_ema_ops = []
                self.rnn_c_ema_ops = []

                mask = model_dict['time_X_mask'][..., None]

                rnn_hidden = model_dict['rnn_hidden']
                rnn_cell = model_dict['rnn_cell']

                for l in range(self.n_layers_rnn):
                    h_rnn_masked = h_rnn[l] * mask
                    denom = tf.reduce_sum(mask)

                    self._regularize(h_rnn_masked, type='context', var_name=reg_name('context'))

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

                self.y_sd_delta = y_sd_delta
                self.y_sd_delta_ema = tf.Variable(0., trainable=False, name='y_sd_delta_ema')
                self.y_sd_delta_ema_op = tf.assign(
                    self.y_sd_delta_ema,
                    ema_rate * self.y_sd_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_sd_delta)
                )

                if self.asymmetric_error:
                    y_skewness_delta *= self.y_skewness_coef
                    self.y_skewness_delta = y_skewness_delta
                    self.y_skewness_delta_ema = tf.Variable(0., trainable=False, name='y_skewness_delta_ema')
                    self.y_skewness_delta_ema_op = tf.assign(
                        self.y_skewness_delta_ema,
                        ema_rate * self.y_skewness_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_skewness_delta)
                    )

                    y_tailweight_delta *= self.y_tailweight_coef
                    self.y_tailweight_delta = y_tailweight_delta
                    self.y_tailweight_delta_ema = tf.Variable(0., trainable=False, name='y_tailweight_delta_ema')
                    self.y_tailweight_delta_ema_op = tf.assign(
                        self.y_tailweight_delta_ema,
                        ema_rate * self.y_tailweight_delta_ema + (1 - ema_rate) * tf.reduce_mean(y_tailweight_delta)
                    )

                self.batch_norm_ema_ops = []
                for x in self.layers:
                    self.batch_norm_ema_ops += x.ema_ops()





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _collect_plots(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # IRF 1D PLOTS
                t_delta = self.support[..., None]

                x = self.plot_impulse_1hot_expanded
                y = self.plot_impulse_1hot_2_expanded
                b = self.plot_impulse_base_expanded
                c = self.plot_impulse_center_expanded
                s = self.plot_impulse_offset_expanded

                means = self.impulse_means_arr_expanded
                sds = self.impulse_sds_arr_expanded

                R = tf.shape(self.gf_y)[0]
                T = tf.shape(self.support)[0]

                X_rate = tf.tile(
                    b,
                    [T, 1, 1]
                )

                self.irf_1d_rate_support = self.support
                irf_1d_rate_plot = self._apply_model(X_rate, t_delta, plot_mode=True)['y']
                self.irf_1d_rate_plot = tf.reshape(irf_1d_rate_plot, [R, T, 1])

                X = tf.tile(
                     x * s + b,
                    [T, 1, 1]
                )
                self.irf_1d_support = self.support
                irf_1d_plot = self._apply_model(X, t_delta, plot_mode=True)['y']
                self.irf_1d_plot = tf.reshape(irf_1d_plot, [R, T, 1]) - self.irf_1d_rate_plot

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

                u_src = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    self.n_surface_plot_points_per_side,
                )
                u = u_src[..., None, None]

                u_rate = u_src
                X_rate = tf.tile(
                    b,
                    [self.n_surface_plot_points_normalized, 1, 1]
                )
                irf_surface_rate_plot = self._apply_model(X_rate, t_delta_square, plot_mode=True)['y']
                self.irf_surface_rate_plot = tf.reshape(
                    irf_surface_rate_plot,
                    [R, self.n_surface_plot_points_per_side, self.n_surface_plot_points_per_side, 1]
                )
                self.irf_surface_rate_meshgrid = tf.meshgrid(time_support, u_rate)

                u = x * (c + u * s)
                X = tf.reshape(
                    tf.tile(
                        u + b,
                        [1, self.n_surface_plot_points_per_side, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )
                irf_surface_plot = self._apply_model(X, t_delta_square, plot_mode=True)['y']
                self.irf_surface_plot = tf.reshape(
                    irf_surface_plot,
                    [R, self.n_surface_plot_points_per_side, self.n_surface_plot_points_per_side, 1]
                ) - self.irf_surface_rate_plot
                if self.center_inputs:
                    b_plot = b + means
                else:
                    b_plot = b
                if self.rescale_inputs:
                    u_plot = u * sds
                else:
                    u_plot = u
                irf_surface_support_impulse = tf.reduce_prod(
                    u_plot + x * b_plot + (1 - x),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                    axis=[1,2]
                )
                self.irf_surface_support = tf.meshgrid(
                    time_support,
                    irf_surface_support_impulse
                )

                # CURVATURE PLOTS
                t_interaction = self.t_interaction

                rate_at_t = self._apply_model(
                    b,
                    tf.ones([1, 1, 1], dtype=self.FLOAT_TF) * t_interaction,
                    plot_mode=True
                )['y']
                rate_at_t = tf.squeeze(rate_at_t)

                t_delta = tf.ones([T, 1, 1], dtype=self.FLOAT_TF) * t_interaction

                u = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    T,
                )[..., None, None]
                u = x * (c + u * s)
                X = u + b
                curvature_plot = self._apply_model(X, t_delta, plot_mode=True)['y']
                self.curvature_plot = tf.reshape(curvature_plot, [R, T, 1]) - rate_at_t

                if self.center_inputs:
                    b_plot = b + means
                else:
                    b_plot = b
                if self.rescale_inputs:
                    u_plot = u * sds
                else:
                    u_plot = u
                curvature_support = u_plot + x * b_plot + (1 - x)  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                self.curvature_support = tf.reduce_prod(
                    curvature_support,
                    axis=[1,2]
                )[..., None]

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

                u_1 = x * (c + v * s)
                X_1 = tf.reshape(
                    tf.tile(
                        u_1,
                        [self.n_surface_plot_points_per_side, 1, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )

                u_2 = y * (c + v * s)
                X_2 = tf.reshape(
                    tf.tile(
                        u_2,
                        [1, self.n_surface_plot_points_per_side, 1]
                    ),
                    [-1, 1, len(self.impulse_names)]
                )

                X = X_1 + X_2 + b
                interaction_surface_plot = self._apply_model(X, t_delta, plot_mode=True)['y']
                self.interaction_surface_plot = tf.reshape(
                    interaction_surface_plot[None, ...],
                    [R, self.n_surface_plot_points_per_side, self.n_surface_plot_points_per_side, 1]
                ) - rate_at_t


                if self.center_inputs:
                    b_plot = b + means
                else:
                    b_plot = b
                if self.rescale_inputs:
                    u_1_plot = u_1 * sds
                    u_2_plot = u_2 * sds
                else:
                    u_1_plot = u_1
                    u_2_plot = u_2
                pred1_support = tf.reduce_prod(
                    u_1_plot + x * b_plot + (1 - x),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                    axis=[1, 2]
                )
                pred2_support = tf.reduce_prod(
                    u_2_plot + y * b_plot + (1 - y),  # Fill empty one-hot cols with ones so we only reduce_prod on valid cols
                    axis=[1, 2]
                )

                self.interaction_surface_support = tf.meshgrid(
                    pred1_support,
                    pred2_support
                )



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


    def initialize_error_params_biases(
            self,
            ran_gf=None
    ):
        raise NotImplementedError




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
                self._initialize_parameter_tables()
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

    def get_plot_names(self, composite='composite', scaled='scaled', dirac='dirac', plot_type='irf_1d', interactions=None):
        if interactions is None:
            interactions = []
        interactions_tmp = []
        for x in interactions:  # Make sure all requested inputs are present in the model, otherwise skip
            add = True
            for y in x.split(':'):
                if not y in self.terminal_names:
                    add = False
                    break
            if add:
                interactions_tmp.append(x)
        interactions = interactions_tmp
        if plot_type.lower() in ['irf_1d', 'irf_surface']:
            out = ['rate'] + self.terminal_names[:]
            # out += interactions
        elif plot_type.lower() == 'curvature':
            out = self.terminal_names[:]
            # out = self.impulse_names[:] + interactions
        elif plot_type.lower() == 'interaction_surface':
            if not interactions:
                out = [':'.join(x) for x in itertools.combinations(self.terminal_names, 2)]
            else:
                out = interactions
        else:
            raise ValueError('Plot type "%s" not supported.' % plot_type)

        return out

    def plot_impulse_name_to_1hot(self, name):
        names = name.split(':')
        ix = names2ix(names, self.terminal_names)
        one_hot = np.zeros(len(self.terminal_names), self.FLOAT_NP)
        one_hot[ix] = 1

        return one_hot

    def get_plot_data(
            self,
            name,
            composite='composite',
            scaled='scaled',
            dirac='dirac',
            plot_type='irf_1d',
            level=95,
            n_samples=None,
            support_start=0.,
            n_time_units=2.5,
            n_time_points=1000,
            t_interaction=0.,
            plot_rangf=False,
            rangf_vals=None,
            plot_mean_as_reference=True
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if rangf_vals is None:
                    rangf_keys = [None]
                    rangf_vals = [self.gf_defaults[0]]
                    if plot_rangf:
                        for i in range(len(self.rangf)):
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
                    self.support_start: support_start,
                    self.n_time_units: n_time_units,
                    self.n_time_points: n_time_points,
                    self.max_tdelta_batch: n_time_points,
                    self.gf_y: rangf_vals,
                    self.plot_impulse_1hot: impulse_one_hot,
                    self.t_interaction: t_interaction,
                    self.training: not self.predict_mode,
                    self.plot_mean_as_reference: plot_mean_as_reference
                }
                if n_samples and self.is_bayesian:
                    fd[self.use_MAP_mode] = False

                if plot_type.lower().startswith('irf_1d'):
                    if name == 'rate':
                        irf_1d_support = self.irf_1d_rate_support
                        irf_1d = self.irf_1d_rate_plot
                    else:
                        irf_1d_support = self.irf_1d_support
                        irf_1d = self.irf_1d_plot
                    to_run = [irf_1d_support, irf_1d]
                elif plot_type.lower().startswith('irf_surface'):
                    if name == 'rate':
                        irf_surface_meshgrid = self.irf_surface_rate_meshgrid
                        irf_surface = self.irf_surface_rate_plot
                    else:
                        irf_surface_meshgrid = self.irf_surface_support
                        irf_surface = self.irf_surface_plot
                    to_run = [irf_surface_meshgrid, irf_surface]
                elif plot_type.lower().startswith('curvature'):
                    assert not name == 'rate', 'Curvature plots are not available for "rate" (deconvolutional intercept).'
                    to_run = [self.curvature_support, self.curvature_plot]
                elif plot_type.lower().startswith('interaction_surface'):
                    names = name.split(':')
                    assert len(names) == 2, 'Interaction surface plots require interactions of order 2'
                    impulse_one_hot1 = self.plot_impulse_name_to_1hot(names[0])
                    impulse_one_hot2 = self.plot_impulse_name_to_1hot(names[1])

                    fd[self.plot_impulse_1hot] = impulse_one_hot1
                    fd[self.plot_impulse_1hot_2] = impulse_one_hot2

                    to_run = [self.interaction_surface_support, self.interaction_surface_plot]
                else:
                    raise ValueError('Plot type "%s" not supported.' % plot_type)

                if n_samples and self.is_bayesian:
                    alpha = 100-float(level)
                    support = self.sess.run(to_run[0], feed_dict=fd)
                    samples = [self.sess.run(to_run[1], feed_dict=fd) for _ in range(n_samples)]
                    samples = np.concatenate(samples, axis=-1)
                    mean = samples.mean(axis=-1)
                    lower = np.percentile(samples, alpha / 2, axis=-1)
                    upper = np.percentile(samples, 100 - (alpha / 2), axis=-1)
                    out = (support, mean, lower, upper, samples)
                else:
                    out = self.sess.run(to_run, feed_dict=fd)

                return out

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
                to_run = [self.train_op, self.ema_op, self.y_sd_delta_ema_op] + self.batch_norm_ema_ops
                if self.n_layers_rnn:
                    to_run += self.rnn_h_ema_ops + self.rnn_c_ema_ops
                if self.asymmetric_error:
                    to_run += [self.y_skewness_delta_ema_op, self.y_tailweight_delta_ema_op]
                if self.loss_filter_n_sds:
                    to_run_names.append('n_dropped')
                    to_run += [self.loss_ema_op, self.loss_sd_ema_op, self.n_dropped]
                to_run_names += ['loss', 'reg_loss']
                to_run += [self.loss_func, self.reg_loss]
                if self.is_bayesian:
                    to_run.append(self.kl_loss)
                    to_run_names.append('kl_loss')
                out = self.sess.run(to_run, feed_dict=feed_dict)

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict

