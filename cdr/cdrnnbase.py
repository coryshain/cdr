import os
import re
import itertools
import numpy as np
import pandas as pd

from .kwargs import CDRNN_INITIALIZATION_KWARGS
from .base import Model
from .util import *

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


parse_initializer = re.compile('(.*_initializer)(_(.*))?')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str):
                    if activation.lower() == 'hard_sigmoid':
                        out = hard_sigmoid
                    else:
                        out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                initializer_name, _, initializer_params = parse_initializer.match(initializer).groups()

                kwargs = {}
                if initializer_params:
                    kwarg_list = initializer_params.split('-')
                    for kwarg in kwarg_list:
                        key, val = kwarg.split('=')
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        kwargs[key] = val

                tf.keras.initializers.he_normal()

                if 'identity' in initializer_name:
                    return tf.keras.initializers.Identity
                elif 'he_' in initializer_name:
                    return tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='normal')
                else:
                    out = getattr(tf, initializer_name)
                    if 'glorot' in initializer:
                        out = out()
                    else:
                        out = out(**kwargs)
            else:
                out = initializer

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None and isinstance(init, str) and '_' in init:
                try:
                    init_split = init.split('_')
                    scale = float(init_split[-1])
                    init = '_'.join(init_split[:-1])
                except ValueError:
                    pass

            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                out = init

            return out


def get_dropout(rate, training=True, noise_shape=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if rate:
                def make_dropout(rate):
                    return lambda x: tf.layers.dropout(x, rate=rate, noise_shape=noise_shape, training=training)
                out = make_dropout(rate)
            else:
                out = lambda x: x

            return out


def compose_lambdas(lambdas):
    def composed_lambdas(x, **kwargs):
        out = x
        for l in lambdas:
            out = l(out, **kwargs)
        return out

    return composed_lambdas


def make_lambda(layer, session=None, use_kwargs=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if use_kwargs:
                def apply_layer(x, **kwargs):
                    return layer(x, **kwargs)
            else:
                def apply_layer(x, **kwargs):
                    return layer(x)
            return apply_layer



class MaskedLSTMCell(LayerRNNCell):
    def __init__(
            self,
            units,
            kernel,
            recurrent_kernel,
            bias=None,
            training=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            bottomup_dropout=None,
            recurrent_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self._session = get_session(session)
        with self._session.as_default():
            with self._session.graph.as_default():
                super(MaskedLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self._num_units = units

                self._kernel = kernel
                self._recurrent_kernel = recurrent_kernel
                self._bias = bias

                self._training = training

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._prefinal_activation = get_activation(prefinal_activation, session=self._session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self._session, training=self._training)

                self._bottomup_dropout = get_dropout(bottomup_dropout, training=self._training, session=self._session)
                self._recurrent_dropout = get_dropout(recurrent_dropout, training=self._training, session=self._session)

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._use_bias = use_bias
                self._global_step = global_step

                self._batch_normalization_decay = batch_normalization_decay

                self._epsilon = epsilon

                self._regularizer_map = {}
                self._regularization_initialized = False

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(c=self._num_units,h=self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _add_regularization(self, var, regularizer):
        if regularizer is not None:
            with self._session.as_default():
                with self._session.graph.as_default():
                    self._regularizer_map[var] = regularizer

    def initialize_regularization(self):
        assert self.built, "Cannot initialize regularization before calling the LSTM layer because the weight matrices haven't been built."

        if not self._regularization_initialized:
            for var in tf.trainable_variables(scope=self.name):
                n1, n2 = var.name.split('/')[-2:]
                if 'bias' in n2:
                    self._add_regularization(var, self._bias_regularizer)
                if 'kernel' in n2:
                    if 'bottomup' in n1 or 'revnet' in n1:
                        self._add_regularization(var, self._bottomup_regularizer)
                    elif 'recurrent' in n1:
                        self._add_regularization(var, self._recurrent_regularizer)
            self._regularization_initialized = True

    def get_regularization(self):
        self.initialize_regularization()
        return self._regularizer_map.copy()

    def build(self, inputs_shape):
        with self._session.as_default():
            with self._session.graph.as_default():
                if isinstance(inputs_shape, list): # Has a mask
                    inputs_shape = inputs_shape[0]
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

                self._input_dims = inputs_shape[1].value
                bottomup_dim = self._input_dims
                recurrent_dim = self._num_units
                output_dim = self._num_units * 4 # forget, input, and output gates, plus cell proposal

                # Build LSTM kernels (bottomup and recurrent)
                self.apply_kernel = lambda x: tf.matmul(x, self._kernel)
                self.apply_recurrent_kernel = lambda x: tf.matmul(x, self._recurrent_kernel)

                if not self._layer_normalization and self._use_bias:
                    self.apply_bias = lambda x: x + self._bias
                else:
                    self.apply_bias = lambda x: x

        self.built = True

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                if isinstance(inputs, list):
                    inputs, mask = inputs
                else:
                    mask = None

                units = self._num_units
                c_prev = state.c
                h_prev = state.h

                s_bottomup = self.apply_kernel(inputs)
                s_recurrent = self.apply_recurrent_kernel(h_prev)
                s = s_bottomup + s_recurrent
                if not self._layer_normalization and self._use_bias:
                    s = self.apply_bias(s)

                # Forget gate
                f = s[:, :units]
                if self._layer_normalization:
                    f = self.norm(f, 'f_ln')
                f = self._recurrent_activation(f + self._forget_bias)

                # Input gate
                i = s[:, units:units * 2]
                if self._layer_normalization:
                    i = self.norm(i, 'i_ln')
                i = self._recurrent_activation(i)

                # Output gate
                o = s[:, units * 2:units * 3]
                if self._layer_normalization:
                    o = self.norm(o, 'o_ln')
                o = self._recurrent_activation(o)

                # Cell proposal
                g = s[:, units * 3:units * 4]
                if self._layer_normalization:
                    g = self.norm(g, 'g_ln')
                g = self._activation(g)

                c = f * c_prev + i * g
                h = o * self._activation(c)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                return h, tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)


class MaskedLSTMLayer(object):
    def __init__(
            self,
            units=None,
            training=False,
            kernel_depth=1,
            resnet_n_layers=1,
            prefinal_mode='max',
            forget_bias=1.0,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            bottomup_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            bottomup_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            bottomup_dropout=None,
            recurrent_dropout=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            return_sequences=True,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-8,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.kernel_depth = kernel_depth
        self.resnet_n_layers = resnet_n_layers
        self.prefinal_mode = prefinal_mode
        self.forget_bias = forget_bias
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.prefinal_activation = prefinal_activation
        self.bottomup_initializer = bottomup_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.bottomup_regularizer = bottomup_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.bottomup_dropout = bottomup_dropout
        self.recurrent_dropout = recurrent_dropout
        self.weight_normalization = weight_normalization
        self.layer_normalization = layer_normalization
        self.use_bias = use_bias
        self.global_step = global_step
        self.batch_normalization_decay = batch_normalization_decay
        self.return_sequences = return_sequences
        self.reuse = reuse
        self.name = name
        self.dtype = dtype
        self.epsilon = epsilon

        self.cell = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs.shape[-1]
                    else:
                        units = self.units

                    self.cell = MaskedLSTMCell(
                        units,
                        training=self.training,
                        kernel_depth=self.kernel_depth,
                        resnet_n_layers=self.resnet_n_layers,
                        prefinal_mode=self.prefinal_mode,
                        forget_bias=self.forget_bias,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        prefinal_activation=self.prefinal_activation,
                        bottomup_initializer=self.bottomup_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        bias_initializer=self.bias_initializer,
                        bottomup_regularizer=self.bottomup_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        bottomup_dropout=self.bottomup_dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        batch_normalization_decay=self.batch_normalization_decay,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build(inputs.shape[1:])

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)
                    while len(mask.shape) < 3:
                        mask = mask[..., None]
                    inputs = [inputs, mask]

                H, _ = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                if not self.return_sequences:
                    H = H[:, -1]

                return H


class LSTMCell(LayerRNNCell):
    """
    An LSTM in which weights are externally initialized.
    This makes the module compatible with black-box variational inference, since weights can be initialized as
    random variables.
    """

    def __init__(
            self,
            kernel,
            recurrent_kernel,
            bias=None,
            training=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=None,
            recurrent_dropout=None,
            forget_bias=1.0,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(LSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self.kernel = kernel
                self.recurrent_kernel = recurrent_kernel
                self.bias = bias
                self.training = training
                self.activation = activation
                self.recurrent_activation = recurrent_activation
                self.dropout = dropout
                self.recurrent_dropout = recurrent_dropout
                self.forget_bias = forget_bias

                self.n_units = int(self.kernel.shape[-1]) // 4

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(h=self.n_units, c=self.n_units)

    @property
    def output_size(self):
        return self.n_units

    def build(self, inputs_shape):
        with self.session.as_default():
            with self.session.graph.as_default():
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    def call(self, inputs, state):
        with self.session.as_default():
            with self.session.graph.as_default():
                h_prev = state.h
                c_prev = state.c




class DenseLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            activation=None,
            kernel_initializer='he_normal_initializer',
            bias_initializer='zeros_initializer',
            dropout=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            batch_normalization_decay=None,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.units = units
                self.use_bias = use_bias
                self.activation = get_activation(activation, session=self.session, training=self.training)
                self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
                if bias_initializer is None:
                    bias_initializer = 'zeros_initializer'
                self.bias_initializer = get_initializer(bias_initializer, session=self.session)
                self.dropout = get_dropout(dropout, training=self.training, session=self.session)
                self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
                self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
                self.batch_normalization_decay = batch_normalization_decay
                self.normalize_weights = normalize_weights
                self.reuse = reuse
                self.name = name

                self.dense_layer = None
                self.kernel_lambdas = []
                self.projection = None

                self.initializer = get_initializer(kernel_initializer, self.session)


                self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.layers.Dense(
                        out_dim,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        _reuse=self.reuse,
                        name=self.name
                    )

                    self.kernel_lambdas.append(self.dense_layer)
                    if self.dropout:
                        self.kernel_lambdas.append(make_lambda(self.dropout, use_kwargs=False, session=self.session))
                    self.kernel = compose_lambdas(self.kernel_lambdas)

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.kernel(inputs)

                if self.normalize_weights:
                    self.w = self.dense_layer.kernel
                    self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
                    self.v = tf.norm(self.w, axis=0)
                    self.dense_layer.kernel = self.v

                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
                    )
                if self.activation is not None:
                    H = self.activation(H)

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)


class RNNLayer(object):
    def __init__(
            self,
            rnn_type='LSTM',
            training=True,
            units=None,
            activation=None,
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            recurrent_initializer='orthogonal_initializer',
            bias_initializer='zeros_initializer',
            unit_forget_bias=True,
            dropout=None,
            recurrent_dropout=None,
            refeed_outputs=False,
            return_sequences=True,
            batch_normalization_decay=None,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.rnn_type = rnn_type
        self.training = training
        self.units = units
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self.training)
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        self.recurrent_initializer = get_initializer(recurrent_initializer, session=self.session)
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.unit_forget_bias = unit_forget_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.batch_normalization_decay = batch_normalization_decay
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, input_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    RNN = getattr(tf.keras.layers, self.rnn_type)

                    input_dim = input_shape[-1]

                    if self.units:
                        output_dim = self.units
                    else:
                        output_dim = input_dim

                    if self.rnn_type == 'LSTM':
                        kernel = tf.get_variable(
                            self.name + '/kernel',
                            [input_dim, output_dim * 4],
                            initializer=self.kernel_initializer
                        )
                        print(kernel.shape)
                        recurrent_kernel = tf.get_variable(
                            self.name + '/recurrent_kernel',
                            [output_dim, output_dim * 4],
                            initializer=self.recurrent_initializer
                        )
                        print(recurrent_kernel.shape)
                        if self.unit_forget_bias:
                            bias = tf.concat(
                                [
                                    tf.get_variable(
                                        self.name + '/bias_i',
                                        [output_dim],
                                        initializer=self.bias_initializer
                                    ),
                                    tf.get_variable(
                                        self.name + '/bias_f',
                                        [output_dim],
                                        initializer=tf.ones_initializer
                                    ),
                                    tf.get_variable(
                                        self.name + '/bias_co',
                                        [output_dim * 2],
                                        initializer=self.bias_initializer
                                    )
                                ],
                                axis=0
                            )
                        else:
                            bias = tf.get_variable(
                                self.name + '/bias',
                                [output_dim * 4],
                                initializer=self.bias_initializer
                            )
                        print(bias.shape)
                    #
                    # def kernel_initializer(shape, weights=kernel, **kwargs):
                    #     return weights
                    #
                    # def recurrent_initializer(shape, weights=recurrent_kernel, **kwargs):
                    #     return weights
                    #
                    # def bias_initializer(shape, weights=bias, **kwargs):
                    #     return weights

                    kernel_initializer = self.kernel_initializer
                    recurrent_initializer = self.recurrent_initializer
                    bias_initializer = self.bias_initializer

                    kwargs = {
                        'return_sequences': self.return_sequences,
                        'activation': self.activation,
                        'kernel_initializer': kernel_initializer,
                        'recurrent_initializer': recurrent_initializer,
                        'bias_initializer': bias_initializer,
                        'dropout': self.dropout,
                        'recurrent_dropout': self.recurrent_dropout,
                        'name': self.name
                    }
                    if self.rnn_type != 'SimpleRNN':
                        kwargs['recurrent_activation'] = self.recurrent_activation
                        kwargs['unit_forget_bias'] = False

                    self.rnn_layer = RNN(output_dim, **kwargs)

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.rnn_layer(inputs, mask=mask)
                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                return H


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

        assert not self.n_units is None, 'You must provide a value for **n_units** when initializing a CDRNN model.'
        if isinstance(self.n_units, str):
            self.n_units_cdrnn = [int(x) for x in self.n_units.split()]
        elif isinstance(self.n_units, int):
            if self.n_units is None:
                self.n_units_cdrnn = [self.n_units]
            else:
                self.n_units_cdrnn = [self.n_units] * self.n_layers
        else:
            self.n_units_cdrnn = self.n_units

        if self.n_layers is None:
            self.n_layers_cdrnn = len(self.n_units_cdrnn)
        else:
            self.n_layers_cdrnn = self.n_layers
        if len(self.n_units_cdrnn) == 1:
            self.n_units_cdrnn = [self.n_units_cdrnn[0]] * self.n_layers_cdrnn

        assert len(self.n_units_cdrnn) == self.n_layers_cdrnn, 'Misalignment in number of layers between n_layers and n_units.'


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
                if self.rescale_t_delta:
                    self.t_delta_cdrnn /= self.t_delta_sd

                time_X_mask_cdrnn = self.time_X_mask[:, :, 0]

                if self.event_dropout_rate:
                    def event_dropout_train_fn(mask=time_X_mask_cdrnn):
                        dropout_mask = tf.random_uniform(shape=tf.shape(mask)) > self.event_dropout_rate
                        return mask & dropout_mask

                    def event_dropout_eval_fn(mask=time_X_mask_cdrnn):
                        return mask

                    time_X_mask_cdrnn = tf.cond(
                        self.training,
                        event_dropout_train_fn,
                        event_dropout_eval_fn
                    )

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
                if self.predictor_dropout_rate:
                    X = tf.layers.dropout(
                        X,
                        rate=self.predictor_dropout_rate,
                        training=self.training
                    )

                # if self.all_predictor_dropout_rate:
                #     X = tf.layers.dropout(
                #         X,
                #         rate=self.all_predictor_dropout_rate,
                #         noise_shape=[None, None, 1],
                #         training=self.training
                #     )

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
                            training=self.training
                        )
                else:
                    rangf_1hot = tf.zeros(
                        [tf.shape(self.gf_y)[0], 0],
                        dtype=self.FLOAT_TF
                    )
                self.rangf_1hot = rangf_1hot

                self.inputs = tf.concat(
                    [X, self.t_delta_cdrnn],
                    axis=-1
                )

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

    def _initialize_encoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.encoder_dropout_rate is None:
                    encoder_dropout_rate = 0.
                else:
                    encoder_dropout_rate = self.encoder_dropout_rate
                    
                if self.encoder_recurrent_dropout_rate is None:
                    encoder_recurrent_dropout_rate = 0.
                else:
                    encoder_recurrent_dropout_rate = self.encoder_recurrent_dropout_rate

                if self.encoder_projection_dropout_rate is None:
                    encoder_projection_dropout_rate = 0.
                else:
                    encoder_projection_dropout_rate = self.encoder_projection_dropout_rate
                    
                self.rangf_encoder = DenseLayer(
                    training=self.training,
                    units=self.n_units_cdrnn[-1],
                    use_bias=False,
                    activation=None,
                    kernel_initializer='zeros_initializer',
                    bias_initializer='zeros_initializer',
                    dropout=encoder_projection_dropout_rate,
                    reuse=None,
                    session=self.sess,
                    name='rangf_encoder'
                )
                self.rangf_embeddings = tf.expand_dims(self.rangf_encoder(self.rangf_1hot), axis=1)

                self.encoder_layers = []

                for l in range(self.n_layers_cdrnn):
                    if l < self.n_layers_cdrnn - 1:
                        return_seqs = True
                    else:
                        # return_seqs = False
                        return_seqs = True

                    layer = RNNLayer(
                        rnn_type=self.rnn_type,
                        training=self.training,
                        units=self.n_units_cdrnn[l],
                        activation=self.encoder_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        kernel_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        dropout=encoder_dropout_rate,
                        recurrent_dropout=encoder_recurrent_dropout_rate,
                        bias_initializer='zeros_initializer',
                        refeed_outputs=False,
                        return_sequences=return_seqs,
                        batch_normalization_decay=None,
                        name='rnn_l%d' % (l + 1),
                        session=self.sess
                    )
                    self.encoder_layers.append(make_lambda(layer, session=self.sess, use_kwargs=True))

                def add_rangf(x, rangf_embeddings=self.rangf_embeddings):
                    multiples = tf.shape(x)[0] // tf.shape(self.rangf_embeddings)[0]
                    t = tf.shape(x)[1]
                    rangf_embeddings = tf.tile(
                        rangf_embeddings,
                        tf.convert_to_tensor([multiples, t, 1], dtype=self.INT_TF)
                    )
                    out = tf.concat([x, rangf_embeddings], axis=-1)

                    # out = x + rangf_embeddings

                    return out

                self.encoder_layers.append(make_lambda(add_rangf, session=self.sess, use_kwargs=False))

                assert self.n_layers_projection > 0, 'n_layers_projection must be a positive integer.'

                for l in range(self.n_layers_projection):
                    if l < self.n_layers_projection - 1:
                        units = self.n_units_cdrnn[l]
                        use_bias = True
                        activation = self.projection_activation_inner
                    else:
                        units = 1
                        use_bias = False
                        activation = None

                    projection = DenseLayer(
                        training=self.training,
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer='zeros_initializer',
                        dropout=encoder_projection_dropout_rate,
                        reuse=None,
                        session=self.sess,
                        name='projection_l%s' % (l + 1)
                    )
                    self.encoder_layers.append(make_lambda(projection, session=self.sess, use_kwargs=False))

                def sum_predictions(x, mask=None):
                    if mask is not None:
                        x *= tf.cast(mask, dtype=self.FLOAT_TF)[..., None]

                    return tf.reduce_sum(x, axis=1)

                self.encoder_layers.append(make_lambda(sum_predictions, session=self.sess, use_kwargs=True))

                self.encoder = compose_lambdas(self.encoder_layers)

                # Intercept
                if self.has_intercept[None]:
                    self.intercept_fixed_base, self.intercept_fixed_base_summary = self.initialize_intercept()
                    self.intercept_fixed = self.intercept_fixed_base
                    self.intercept_fixed_summary = self.intercept_fixed_base_summary
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_fixed_summary,
                        collections=['params']
                    )
                    self._regularize(self.intercept_fixed, type='intercept', var_name='intercept')
                    if self.convergence_basis.lower() == 'parameters':
                        self._add_convergence_tracker(self.intercept_fixed_summary, 'intercept_fixed')

                else:
                    self.intercept_fixed_base = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')
                    self.intercept_fixed = self.intercept_fixed_base

                self.intercept = self.intercept_fixed
                self.intercept_summary = self.intercept_fixed_summary
                self.intercept_random_base = {}
                self.intercept_random_base_summary = {}
                self.intercept_random = {}
                self.intercept_random_summary = {}
                self.intercept_random_means = {}


                # RANDOM EFFECTS
                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                    # Random intercepts
                    if self.has_intercept[gf]:
                        self.intercept_random_base[gf], self.intercept_random_base_summary[gf] = self.initialize_intercept(ran_gf=gf)
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

                        # Create record for convergence tracking
                        if self.convergence_basis.lower() == 'parameters':
                            self._add_convergence_tracker(self.intercept_random_summary[gf], 'intercept_by_%s' %gf)

                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])
                        self.intercept_summary += tf.gather(intercept_random_summary, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/intercept' % gf),
                                intercept_random_summary,
                                collections=['random']
                            )

    def _construct_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = tf.squeeze(self.encoder(self.inputs, mask=self.time_X_mask_cdrnn), axis=-1)
                out += self.intercept

                self.out = out
                # Hack needed for MAP evaluation of CDRNNBayes
                self.out_mean = self.out





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
                if self.rescale_t_delta:
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
                inputs = tf.concat([X_rate, t_delta], axis=-1)
                self.irf_1d_rate_support = self.support
                self.irf_1d_rate_plot = self.encoder(inputs)[None, ...]

                X = tf.tile(
                     x * (c + s) + b,
                    [t, 1, 1]
                )
                inputs = tf.concat([X, t_delta], axis=-1)
                self.irf_1d_support = self.support
                self.irf_1d_plot = self.encoder(inputs)[None, ...] - self.irf_1d_rate_plot

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
                if self.rescale_t_delta:
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
                inputs = tf.concat([X_rate, t_delta_square], axis=-1)
                self.irf_surface_rate_plot = tf.reshape(
                    self.encoder(inputs)[None, ...],
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
                inputs = tf.concat([X, t_delta_square], axis=-1)
                self.irf_surface_plot = tf.reshape(
                    self.encoder(inputs)[None, ...],
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
                if self.rescale_t_delta:
                    t_interaction /= self.t_delta_sd

                rate_at_t = tf.squeeze(
                    self.encoder(
                        tf.concat(
                            [
                                self.plot_impulse_base_expanded,
                                tf.ones([1, 1, 1], dtype=self.FLOAT_TF) * t_interaction
                            ],
                            axis=-1
                        )
                    )
                )

                t_delta = tf.ones([t, 1, 1], dtype=self.FLOAT_TF) * t_interaction

                u = tf.linspace(
                    tf.cast(-self.plot_n_sds, dtype=self.FLOAT_TF),
                    tf.cast(self.plot_n_sds, dtype=self.FLOAT_TF),
                    t,
                )[..., None, None]
                u = x * (c + u)
                X = u + b
                inputs = tf.concat([X, t_delta], axis=-1)
                self.curvature_plot = self.encoder(inputs) - rate_at_t
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
                inputs = tf.concat([X, t_delta], axis=-1)
                self.interaction_surface_plot = self.encoder(inputs) - rate_at_t
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
            print(out)
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
            print(names)
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
