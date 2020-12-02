import math
import collections
import tensorflow as tf
from .util import *

from tensorflow.python.ops import rnn_cell_impl

if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell

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
                    elif activation.lower() == 'gelu':
                        out = lambda x: x * tf.nn.sigmoid(1.702*x)
                    elif activation.lower() == 'swish':
                        out = lambda x: x * tf.nn.sigmoid(x)
                    elif activation.lower() == 'shifted_softplus':
                        out = lambda x: tf.nn.softplus(x) - 0.69314718056
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


def make_lambda(layer, session=None, multi_arg=False, use_kwargs=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if multi_arg:
                if use_kwargs:
                    def apply_layer(*x, **kwargs):
                        return layer(*x, **kwargs)
                else:
                    def apply_layer(*x, **kwargs):
                        return layer(*x)
            else:
                if use_kwargs:
                    def apply_layer(x, **kwargs):
                        return layer(x, **kwargs)
                else:
                    def apply_layer(x, **kwargs):
                        return layer(x)
            return apply_layer


CDRNNStateTuple = collections.namedtuple(
    'AttentionalLSTMDecoderStateTuple',
    ' '.join(['c', 'h', 't'])
)


class CDRNNCell(LayerRNNCell):
    def __init__(
            self,
            units,
            training=False,
            kernel_depth=1,
            time_projection_depth=1,
            resnet_n_layers=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='he',
            recurrent_kernel_sd_init='he',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            l2_normalize_states=False,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self._session = get_session(session)
        with self._session.as_default():
            with self._session.graph.as_default():
                super(CDRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self._num_units = units
                self._training = training

                self._kernel_depth = kernel_depth
                self._time_projection_depth = time_projection_depth
                self._resnet_n_layers = resnet_n_layers
                self._prefinal_mode = prefinal_mode
                self._forget_bias = forget_bias
                self._forget_gate_as_irf = forget_gate_as_irf

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self._session, training=self._training)
                self._prefinal_activation = get_activation(prefinal_activation, session=self._session, training=self._training)
                self._time_projection_inner_activation = get_activation(time_projection_inner_activation, session=self._session, training=self._training)

                self._bottomup_kernel_sd_init = bottomup_kernel_sd_init
                self._recurrent_kernel_sd_init = recurrent_kernel_sd_init

                self._bottomup_dropout = get_dropout(bottomup_dropout, training=self._training, session=self._session)
                self._h_dropout = get_dropout(h_dropout, training=self._training, session=self._session)
                self._c_dropout = get_dropout(c_dropout, training=self._training, session=self._session)
                self._forget_rate = forget_rate

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._use_bias = use_bias
                self._global_step = global_step

                self._batch_normalization_decay = batch_normalization_decay

                self._l2_normalize_states = l2_normalize_states

                self._epsilon = epsilon

                self.built = False

    @property
    def state_size(self):
        return CDRNNStateTuple(c=self._num_units, h=self._num_units, t=1)

    @property
    def output_size(self):
        return CDRNNStateTuple(c=self._num_units, h=self._num_units, t=1)

    def initialize_kernel(
            self,
            in_dim,
            out_dim,
            kernel_sd_init,
            depth=None,
            inner_activation=None,
            prefinal_mode=None,
            name=None
    ):
        with self._session.as_default():
            with self._session.graph.as_default():
                units_below = in_dim
                kernel_lambdas = []
                if depth is None:
                    depth = self._kernel_depth
                if prefinal_mode is None:
                    prefinal_mode = self._prefinal_mode

                if prefinal_mode.lower() == 'max':
                    if out_dim > in_dim:
                        prefinal_dim = out_dim
                    else:
                        prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'in':
                    prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'out':
                    prefinal_dim = out_dim
                else:
                    raise ValueError('Unrecognized value for prefinal_mode: %s.' % prefinal_mode)

                layers = []

                for d in range(depth):
                    if d == depth - 1:
                        activation = None
                        units = out_dim
                        use_bias = False
                    else:
                        if inner_activation is None:
                            activation = self._prefinal_activation
                        else:
                            activation = inner_activation
                        units = prefinal_dim
                        use_bias = self._use_bias
                    kernel_initializer = get_initializer(
                        'random_normal_initializer_mean=0-stddev=%s' % kernel_sd_init,
                        session=self._session
                    )
                    if name:
                        name_cur = name + '_d%d' % d
                    else:
                        name_cur = 'd%d' % d

                    if self._resnet_n_layers and self._resnet_n_layers > 1 and units == units_below:
                        kernel_layer = DenseResidualLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            bias_initializer='zeros_initializer',
                            layers_inner=self._resnet_n_layers,
                            activation_inner=self._prefinal_activation,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            project_inputs=False,
                            normalize_weights=self._weight_normalization,
                            reuse=tf.AUTO_REUSE,
                            epsilon=self._epsilon,
                            session=self._session,
                            name=name_cur
                        )
                    else:
                        kernel_layer = DenseLayer(
                            training=self._training,
                            units=units,
                            use_bias=use_bias,
                            kernel_sd_init=kernel_sd_init,
                            activation=activation,
                            batch_normalization_decay=self._batch_normalization_decay,
                            epsilon=self._epsilon,
                            session=self._session,
                            reuse=tf.AUTO_REUSE,
                            name=name_cur
                        )

                    layers.append(kernel_layer)
                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                    units_below = units

                kernel = compose_lambdas(kernel_lambdas)

                return kernel, layers

    @property
    def weights(self):
        weights = [self._bias]
        weights += sum([x.weights for x in self._kernel_bottomup_layers], [])
        weights += sum([x.weights for x in self._kernel_recurrent_layers], [])
        if self._forget_gate_as_irf:
            weights += sum([x.weights for x in self._kernel_time_projection_layers], [])

        return weights[:]

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
                output_dim = self._num_units * 5 # forget (2x), input, and output gates, plus cell proposal

                # Build bias
                if not self._layer_normalization and self._use_bias:
                    self._bias = self.add_variable(
                        'bias',
                        shape=[1, output_dim],
                        initializer=tf.zeros_initializer()
                    )

                # Build LSTM kernels (bottomup and recurrent)
                self._kernel_bottomup, self._kernel_bottomup_layers = self.initialize_kernel(
                    bottomup_dim,
                    output_dim,
                    self._bottomup_kernel_sd_init,
                    name='bottomup'
                )

                self._kernel_recurrent, self._kernel_recurrent_layers = self.initialize_kernel(
                    recurrent_dim,
                    output_dim,
                    self._recurrent_kernel_sd_init,
                    name='recurrent'
                )

                if self._forget_gate_as_irf:
                    self._t_delta_embedding_W = self.add_variable(
                        't_delta_embedding_W',
                        shape=[1, self._num_units],
                        initializer=self._bottomup_initializer
                    )
                    self._t_delta_embedding_b = self.add_variable(
                        't_delta_embedding_b',
                        shape=[1, self._num_units],
                        initializer=tf.zeros_initializer
                    )

                    self._kernel_time_projection, self._kernel_time_projection_layers = self.initialize_kernel(
                        recurrent_dim,
                        recurrent_dim,
                        self._bottomup_kernel_sd_init,
                        depth=self._time_projection_depth,
                        inner_activation=self._time_projection_inner_activation,
                        name='time_projection'
                    )

        self.built = True

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                assert isinstance(inputs, dict), 'Inputs to CDRNNCell must be a dict containing fields ``inputs`` and (optionally) ``times``, and ``mask``.'

                units = self._num_units
                c_prev = state.c
                h_prev = state.h
                t_prev = state.t

                if 'times' in inputs:
                    t_cur = inputs['times']
                else:
                    t_cur = t_prev

                if 'mask' in inputs:
                    mask = inputs['mask']
                else:
                    mask = None

                inputs = inputs['inputs']

                inputs = self._bottomup_dropout(inputs)

                if self._forget_rate:
                    def train_fn_forget(h_prev=h_prev, c_prev=c_prev):
                        dropout_mask = tf.cast(tf.random_uniform(shape=[tf.shape(inputs)[0], 1]) > self._forget_rate, dtype=tf.float32)

                        h_prev_out = h_prev * dropout_mask
                        # c_prev_out = c_prev
                        c_prev_out = c_prev * dropout_mask

                        return h_prev_out, c_prev_out

                    def eval_fn_forget(h_prev=h_prev, c_prev=c_prev):
                        return h_prev, c_prev

                    h_prev, c_prev = tf.cond(self._training, train_fn_forget, eval_fn_forget)

                s_bottomup = self._kernel_bottomup(inputs)
                s_recurrent = self._kernel_recurrent(h_prev)
                s = s_bottomup + s_recurrent
                if not self._layer_normalization and self._use_bias:
                    s += self._bias

                # Input gate
                i = s[:, :units]
                if self._layer_normalization:
                    i = self.norm(i, 'i_ln')
                i = self._recurrent_activation(i)

                # Output gate
                o = s[:, units:units*2]
                if self._layer_normalization:
                    o = self.norm(o, 'o_ln')
                o = self._recurrent_activation(o)

                # Cell proposal
                g = s[:, units*2:units*3]
                if self._layer_normalization:
                    g = self.norm(g, 'g_ln')
                g = self._activation(g)

                # Forget gate
                if self._forget_gate_as_irf:
                    f_W = self._t_delta_embedding_W + s[:, units*3:units*4]
                    f_b = self._t_delta_embedding_b + s[:, units*4:units*5]
                    if self._layer_normalization:
                        f_W = self.norm(f_W, 'f_W_ln')
                        f_b = self.norm(f_b, 'f_b_ln')
                    t = t_cur - t_prev
                    t_embedding = self._time_projection_inner_activation(f_W * t + f_b + self._forget_bias)
                    f = self._kernel_time_projection(t_embedding)
                else:
                    f = s[:, units*3:units*4]
                f = self._recurrent_activation(f)

                c = f * c_prev + i * g
                h = o * self._activation(c)

                if self._l2_normalize_states:
                    h = tf.nn.l2_normalize(h, epsilon=self._epsilon, axis=-1)

                c = self._c_dropout(c)
                h = self._h_dropout(h)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                new_state = CDRNNStateTuple(c=c, h=h, t=t_cur)

                return new_state, new_state


class CDRNNLayer(object):
    def __init__(
            self,
            units=None,
            training=False,
            kernel_depth=1,
            time_projection_depth=1,
            resnet_n_layers=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='he',
            recurrent_kernel_sd_init='he',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            batch_normalization_decay=None,
            l2_normalize_states=False,
            return_sequences=True,
            reuse=None,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.kernel_depth = kernel_depth
        self.time_projection_depth = time_projection_depth
        self.resnet_n_layers = resnet_n_layers
        self.prefinal_mode = prefinal_mode
        self.forget_bias = forget_bias
        self.forget_gate_as_irf = forget_gate_as_irf
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.prefinal_activation = prefinal_activation
        self.time_projection_inner_activation = time_projection_inner_activation
        self.bottomup_kernel_sd_init = bottomup_kernel_sd_init
        self.recurrent_kernel_sd_init = recurrent_kernel_sd_init
        self.bottomup_dropout = bottomup_dropout
        self.h_dropout = h_dropout
        self.c_dropout = c_dropout
        self.forget_rate = forget_rate
        self.weight_normalization = weight_normalization
        self.layer_normalization = layer_normalization
        self.use_bias = use_bias
        self.global_step = global_step
        self.batch_normalization_decay = batch_normalization_decay
        self.l2_normalize_states = l2_normalize_states
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

                    self.cell = CDRNNCell(
                        units,
                        training=self.training,
                        kernel_depth=self.kernel_depth,
                        time_projection_depth=self.time_projection_depth,
                        resnet_n_layers=self.resnet_n_layers,
                        prefinal_mode=self.prefinal_mode,
                        forget_bias=self.forget_bias,
                        forget_gate_as_irf=self.forget_gate_as_irf,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        prefinal_activation=self.prefinal_activation,
                        time_projection_inner_activation=self.time_projection_inner_activation,
                        bottomup_kernel_sd_init=self.bottomup_kernel_sd_init,
                        recurrent_kernel_sd_init=self.recurrent_kernel_sd_init,
                        bottomup_dropout=self.bottomup_dropout,
                        h_dropout=self.h_dropout,
                        c_dropout=self.c_dropout,
                        forget_rate=self.forget_rate,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        batch_normalization_decay=self.batch_normalization_decay,
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build(inputs.shape[1:])

            self.built = True

    @property
    def weights(self):
        return self.cell.weights

    def __call__(self, inputs, times=None, mask=None, return_state=False, initial_state=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                inputs = {'inputs': inputs}

                if times is not None:
                    inputs['times'] = times

                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)
                    while len(mask.shape) < 3:
                        mask = mask[..., None]

                H, _ = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    initial_state=initial_state,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                H, c, _ = H

                if not self.return_sequences:
                    H = H[:, -1]
                    if return_state:
                        c = c[:, -1]

                if return_state:
                    out = (H, c)
                else:
                    out = H

                return out


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


# class DenseLayer(object):
#
#     def __init__(
#             self,
#             training=True,
#             units=None,
#             use_bias=True,
#             activation=None,
#             kernel_initializer='glorot_uniform_initializer',
#             bias_initializer='zeros_initializer',
#             dropout=None,
#             kernel_regularizer=None,
#             bias_regularizer=None,
#             batch_normalization_decay=None,
#             batch_normalization_use_beta=True,
#             batch_normalization_use_gamma=True,
#             normalize_weights=False,
#             reuse=tf.AUTO_REUSE,
#             epsilon=1e-3,
#             session=None,
#             name=None
#     ):
#         self.session = get_session(session)
#         with session.as_default():
#             with session.graph.as_default():
#                 self.training = training
#                 self.units = units
#                 self.use_bias = use_bias
#                 self.activation = get_activation(activation, session=self.session, training=self.training)
#                 self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
#                 if bias_initializer is None:
#                     bias_initializer = 'zeros_initializer'
#                 self.bias_initializer = get_initializer(bias_initializer, session=self.session)
#                 self.dropout = get_dropout(dropout, training=self.training, session=self.session)
#                 self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
#                 self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
#                 self.batch_normalization_decay = batch_normalization_decay
#                 self.batch_normalization_use_beta = batch_normalization_use_beta
#                 self.batch_normalization_use_gamma = batch_normalization_use_gamma
#                 self.normalize_weights = normalize_weights
#                 self.reuse = reuse
#                 self.epsilon = epsilon
#                 self.name = name
#
#                 self.dense_layer = None
#                 self.kernel_lambdas = []
#                 self.projection = None
#
#                 self.initializer = get_initializer(kernel_initializer, self.session)
#
#                 self.built = False
#
#     @property
#     def weights(self):
#         if self.built:
#             return self.dense_layer.weights
#         return []
#
#     def build(self, inputs):
#         if not self.built:
#             if self.units is None:
#                 out_dim = inputs.shape[-1]
#             else:
#                 out_dim = self.units
#
#             with self.session.as_default():
#                 with self.session.graph.as_default():
#                     self.dense_layer = tf.layers.Dense(
#                         out_dim,
#                         use_bias=self.use_bias,
#                         kernel_initializer=self.kernel_initializer,
#                         bias_initializer=self.bias_initializer,
#                         kernel_regularizer=self.kernel_regularizer,
#                         bias_regularizer=self.bias_regularizer,
#                         _reuse=self.reuse,
#                         name=self.name
#                     )
#
#                     self.kernel_lambdas.append(self.dense_layer)
#                     self.kernel_lambdas.append(make_lambda(self.dropout, use_kwargs=False, session=self.session))
#                     self.kernel = compose_lambdas(self.kernel_lambdas)
#
#             self.built = True
#
#     def __call__(self, inputs):
#         if not self.built:
#             self.build(inputs)
#
#         with self.session.as_default():
#             with self.session.graph.as_default():
#
#                 H = self.kernel(inputs)
#
#                 if self.normalize_weights:
#                     self.w = self.dense_layer.kernel
#                     self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
#                     self.v = tf.norm(self.w, axis=0)
#                     self.dense_layer.kernel = self.v
#
#                 if self.batch_normalization_decay:
#                     H = tf.contrib.layers.batch_norm(
#                         H,
#                         decay=self.batch_normalization_decay,
#                         center=self.batch_normalization_use_beta,
#                         scale=self.batch_normalization_use_gamma,
#                         zero_debias_moving_mean=True,
#                         epsilon=self.epsilon,
#                         is_training=self.training,
#                         updates_collections=None,
#                         reuse=self.reuse,
#                         scope=self.name
#                     )
#                 if self.activation is not None:
#                     H = self.activation(H)
#
#                 return H
#
#     def call(self, *args, **kwargs):
#         self.__call__(*args, **kwargs)
#

class DenseLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            activation=None,
            kernel_sd_init='he',
            dropout=None,
            batch_normalization_decay=None,
            batch_normalization_use_beta=True,
            batch_normalization_use_gamma=True,
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
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
                self.kernel_sd_init = kernel_sd_init
                self.dropout = get_dropout(dropout, training=self.training, session=self.session)
                self.batch_normalization_decay = batch_normalization_decay
                self.batch_normalization_use_beta = batch_normalization_use_beta
                self.batch_normalization_use_gamma = batch_normalization_use_gamma
                self.reuse = reuse
                self.epsilon = epsilon
                self.name = name

                self.built = False

    @property
    def weights(self):
        if self.built:
            return [self.kernel]
        return []

    def build(self, inputs):
        if not self.built:
            in_dim = inputs.shape[-1]
            if self.units is None:
                out_dim = in_dim
            else:
                out_dim = self.units

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        if isinstance(self.kernel_sd_init, str):
                            if self.kernel_sd_init.lower() in ['xavier', 'glorot']:
                                sd = math.sqrt(2 / (int(in_dim) + out_dim))
                            elif self.kernel_sd_init.lower() == 'he':
                                sd = math.sqrt(2 / int(in_dim))
                        else:
                            sd = self.kernel_sd_init

                        kernel_init = get_initializer(
                            'random_normal_initializer_mean=0-stddev=%s' % sd,
                            session=self.session
                        )
                        self.kernel = tf.get_variable(
                            name='kernel',
                            initializer=kernel_init,
                            shape=[in_dim, out_dim]
                        )

                        self.bias = tf.get_variable(
                            name='bias',
                            shape=[out_dim],
                            initializer=tf.zeros_initializer(),
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = tf.tensordot(inputs, self.kernel, 1)
                bias = self.bias
                while len(bias.shape) < len(H.shape):
                    bias = bias[None, ...]
                H += bias

                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=self.batch_normalization_use_beta,
                        scale=self.batch_normalization_use_gamma,
                        zero_debias_moving_mean=True,
                        epsilon=self.epsilon,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
                    )
                if self.activation is not None:
                    H = self.activation(H)

                H = self.dropout(H)

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)


class DenseLayerBayes(DenseLayer):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            activation=None,
            dropout=None,
            batch_normalization_decay=None,
            batch_normalization_use_beta=True,
            batch_normalization_use_gamma=True,
            use_MAP_mode=None,
            kernel_prior_sd='he',
            bias_prior_sd=1,
            posterior_to_prior_sd_ratio=1,
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        super(DenseLayerBayes, self).__init__(
            training=training,
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_sd_init=kernel_prior_sd,
            dropout=dropout,
            batch_normalization_decay=batch_normalization_decay,
            batch_normalization_use_beta=batch_normalization_use_beta,
            batch_normalization_use_gamma=batch_normalization_use_gamma,
            reuse=reuse,
            epsilon=epsilon,
            session=session,
            name=name
        )
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.use_MAP_mode = use_MAP_mode
                self.kernel_prior_sd = kernel_prior_sd
                self.bias_prior_sd = bias_prior_sd
                self.bias_sd_init = self.bias_prior_sd
                self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio

                self.built = False

    @property
    def weights(self):
        if self.built:
            return [self.kernel_mean]
        return []

    def build(self, inputs):
        if not self.built:
            in_dim = inputs.shape[-1]
            if self.units is None:
                out_dim = in_dim
            else:
                out_dim = self.units

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name):
                        if isinstance(self.kernel_sd_init, str):
                            if self.kernel_sd_init.lower() in ['xavier', 'glorot']:
                                sd = math.sqrt(2 / (int(in_dim) + out_dim))
                            elif self.kernel_sd_init.lower() == 'he':
                                sd = math.sqrt(2 / int(in_dim))
                        else:
                            sd = self.kernel_sd_init

                        if self.use_MAP_mode is None:
                            self.use_MAP_mode = tf.logical_not(self.training)
                        self.kernel_mean = tf.Variable(
                            tf.zeros([in_dim, out_dim]),
                            name='kernel_mean'
                        )
                        self.kernel_sd_unconstrained = tf.Variable(
                            tf.ones([in_dim, out_dim]) * tf.contrib.distributions.softplus_inverse(sd),
                            name='kernel_sd'
                        )
                        self.kernel_sd = tf.nn.softplus(self.kernel_sd_unconstrained)
                        self.kernel_dist = tf.contrib.distributions.Normal(
                            loc=self.kernel_mean,
                            scale=self.kernel_sd + self.epsilon
                        )
                        self.kernel_prior_dist = tf.contrib.distributions.Normal(
                            loc=0.,
                            scale=self.kernel_prior_sd
                        )
                        self.kernel = tf.cond(
                            self.use_MAP_mode,
                            self.kernel_dist.mean,
                            self.kernel_dist.sample
                        )

                        self.bias_mean = tf.Variable(
                            tf.zeros([out_dim]),
                            name='bias_mean'
                        )
                        self.bias_sd_unconstrained = tf.Variable(
                            tf.ones([out_dim]) * tf.contrib.distributions.softplus_inverse(self.bias_sd_init),
                            name='bias_sd'
                        )
                        self.bias_sd = tf.nn.softplus(self.bias_sd_unconstrained)
                        self.bias_dist = tf.contrib.distributions.Normal(
                            loc=self.bias_mean,
                            scale=self.bias_sd + self.epsilon
                        )
                        self.bias_prior_dist = tf.contrib.distributions.Normal(
                            loc=0.,
                            scale=self.bias_prior_sd
                        )
                        self.bias = tf.cond(
                            self.use_MAP_mode,
                            self.bias_dist.mean,
                            self.bias_dist.sample
                        )

            self.built = True

    def kl_penalties(self):
        return [
            self.kernel_dist.kl_divergence(self.kernel_prior_dist),
            self.bias_dist.kl_divergence(self.bias_prior_dist)
        ]


class DenseResidualLayer(object):
    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            sample_at_train=False,
            sample_at_eval=False,
            batch_normalization_decay=0.9,
            project_inputs=False,
            normalize_weights=False,
            reuse=None,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias

        self.layers_inner = layers_inner
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation_inner = get_activation(
            activation_inner,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
        self.activation = get_activation(
            activation,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.epsilon = epsilon
        self.name = name

        self.dense_layers = None
        self.projection = None

        self.built = False

    @property
    def weights(self):
        out = []
        if self.built:
            for l in self.dense_layers:
                out += l.weights
        return out

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.units is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.units

                    self.dense_layers = []

                    for i in range(self.layers_inner):
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        if self.name:
                            name = self.name + '_projection'
                        else:
                            name = None

                        self.projection = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.dense_layers[i](F)
                    if self.batch_normalization_decay:
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            epsilon=self.epsilon,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        epsilon=self.epsilon,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
                    )

                if self.project_inputs:
                    x = self.projection(inputs)
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


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
            epsilon=1e-5,
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
        self.epsilon = epsilon
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

                    # if self.rnn_type == 'LSTM':
                    #     kernel = tf.get_variable(
                    #         self.name + '/kernel',
                    #         [input_dim, output_dim * 4],
                    #         initializer=self.kernel_initializer
                    #     )
                    #     recurrent_kernel = tf.get_variable(
                    #         self.name + '/recurrent_kernel',
                    #         [output_dim, output_dim * 4],
                    #         initializer=self.recurrent_initializer
                    #     )
                    #     if self.unit_forget_bias:
                    #         bias = tf.concat(
                    #             [
                    #                 tf.get_variable(
                    #                     self.name + '/bias_i',
                    #                     [output_dim],
                    #                     initializer=self.bias_initializer
                    #                 ),
                    #                 tf.get_variable(
                    #                     self.name + '/bias_f',
                    #                     [output_dim],
                    #                     initializer=tf.ones_initializer
                    #                 ),
                    #                 tf.get_variable(
                    #                     self.name + '/bias_co',
                    #                     [output_dim * 2],
                    #                     initializer=self.bias_initializer
                    #                 )
                    #             ],
                    #             axis=0
                    #         )
                    #     else:
                    #         bias = tf.get_variable(
                    #             self.name + '/bias',
                    #             [output_dim * 4],
                    #             initializer=self.bias_initializer
                    #         )
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
                        epsilon=self.epsilon,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                return H

