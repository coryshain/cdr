import math
import collections
import tensorflow as tf
from .util import *

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.distributions import Normal

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
                    elif activation.lower() == 'gelu1p':
                        out = lambda x: (x + 1) * tf.nn.sigmoid(1.702*(x+1))
                    elif activation.lower() == 'l2norm':
                        out = lambda x: tf.nn.l2_normalize(x, axis=-1)
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


def get_dropout(
        rate,
        training=True,
        use_MAP_mode=True,
        rescale=True,
        noise_shape=None,
        name=None,
        constant=None,
        reuse=tf.AUTO_REUSE,
        session=None
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if rate:
                out = DropoutLayer(
                    rate,
                    noise_shape=noise_shape,
                    training=training,
                    use_MAP_mode=use_MAP_mode,
                    rescale=rescale,
                    constant=constant,
                    name=name,
                    reuse=reuse,
                    session=session
                )
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


class RNNCell(LayerRNNCell):
    def __init__(
            self,
            units,
            training=False,
            use_MAP_mode=False,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='glorot_normal',
            recurrent_kernel_sd_init='glorot_normal',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self._session = get_session(session)
        with self._session.as_default():
            with self._session.graph.as_default():
                super(RNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self._num_units = units
                self._training = training
                self.use_MAP_mode = use_MAP_mode

                self._kernel_depth = kernel_depth
                self._time_projection_depth = time_projection_depth
                self._prefinal_mode = prefinal_mode
                self._forget_bias = forget_bias
                self._forget_gate_as_irf = forget_gate_as_irf

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self._session, training=self._training)
                self._prefinal_activation = get_activation(prefinal_activation, session=self._session, training=self._training)
                self._time_projection_inner_activation = get_activation(time_projection_inner_activation, session=self._session, training=self._training)

                self._bottomup_kernel_sd_init = bottomup_kernel_sd_init
                self._recurrent_kernel_sd_init = recurrent_kernel_sd_init

                self.use_dropout = bool(bottomup_dropout or h_dropout or c_dropout)
                self._bottomup_dropout_rate = bottomup_dropout
                self._bottomup_dropout_layer = get_dropout(
                    bottomup_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/bottomup_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._h_dropout_rate = h_dropout
                self._h_dropout_layer = get_dropout(
                    h_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/h_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._c_dropout_rate = c_dropout
                self._c_dropout_layer = get_dropout(
                    c_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/c_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._forget_rate = forget_rate

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._use_bias = use_bias
                self._global_step = global_step

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
            kernel_type='bottomup',
            depth=None,
            inner_activation=None,
            prefinal_mode=None,
            name=None
    ):
        if kernel_type.lower() == 'recurrent':
            kernel_sd_init = self._recurrent_kernel_sd_init
        else:
            kernel_sd_init = self._bottomup_kernel_sd_init
        with self._session.as_default():
            with self._session.graph.as_default():
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

                    if self.name:
                        name_cur = self.name + '/'
                    else:
                        name_cur = ''

                    if name:
                        name_cur += name + '_d%d' % d
                    else:
                        name_cur += 'd%d' % d

                    kernel_layer = DenseLayer(
                        training=self._training,
                        units=units,
                        use_bias=use_bias,
                        kernel_sd_init=kernel_sd_init,
                        activation=activation,
                        epsilon=self._epsilon,
                        session=self._session,
                        reuse=self._reuse,
                        name=name_cur
                    )

                    kernel_layer.build([None, in_dim])

                    layers.append(kernel_layer)
                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel, layers

    @property
    def weights(self):
        weights = [self._bias]
        weights += sum([x.weights for x in self._kernel_bottomup_layers], [])
        weights += sum([x.weights for x in self._kernel_recurrent_layers], [])
        if self._forget_gate_as_irf:
            weights += sum([x.weights for x in self._kernel_time_projection_layers], []) + [self.t_delta_embedding_W]

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
                self.layers = []

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
                    kernel_type='bottomup',
                    name='bottomup'
                )
                self.layers += self._kernel_bottomup_layers
                if self._bottomup_dropout_rate:
                    self._bottomup_dropout_layer.build(inputs_shape)

                self._kernel_recurrent, self._kernel_recurrent_layers = self.initialize_kernel(
                    recurrent_dim,
                    output_dim,
                    kernel_type='recurrent',
                    name='recurrent'
                )
                self.layers += self._kernel_recurrent_layers
                if self._h_dropout_rate:
                    self._h_dropout_layer.build([x for x in inputs_shape[:-1]] + [output_dim])
                if self._c_dropout_rate:
                    self._c_dropout_layer.build([x for x in inputs_shape[:-1]] + [output_dim])

                if self._forget_gate_as_irf:
                    self.initialize_irf_biases()

                    self._kernel_time_projection, self._kernel_time_projection_layers = self.initialize_kernel(
                        recurrent_dim,
                        recurrent_dim,
                        kernel_type='time_projection',
                        depth=self._time_projection_depth,
                        inner_activation=self._time_projection_inner_activation,
                        name='time_projection'
                    )
                    self.layers += self._kernel_time_projection_layers
                else:
                    self._kernel_time_projection = None
                    self._kernel_time_projection_layers = []

        self.built = True

    def initialize_irf_biases(self):
        with self._session.as_default():
            with self._session.graph.as_default():
                self.t_delta_embedding_W = self.add_variable(
                    't_delta_embedding_W',
                    shape=[1, self._num_units],
                    initializer=self._bottomup_initializer
                )
                self.t_delta_embedding_b = self.add_variable(
                    't_delta_embedding_b',
                    shape=[1, self._num_units],
                    initializer=tf.zeros_initializer()
                )

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

                inputs = self._bottomup_dropout_layer(inputs)

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

                c = self._c_dropout_layer(c)
                h = self._h_dropout_layer(h)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                new_state = CDRNNStateTuple(c=c, h=h, t=t_cur)

                return new_state, new_state

    def ema_ops(self):
        return []

    def dropout_resample_ops(self):
        out = []
        if self.built:
            for layer in self._kernel_bottomup_layers + self._kernel_recurrent_layers + self._kernel_time_projection_layers:
                out.append(layer.dropout_resample_ops())

        return out


class RNNLayer(object):
    def __init__(
            self,
            units=None,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='glorot_normal',
            recurrent_kernel_sd_init='glorot_normal',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            return_sequences=True,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.use_MAP_mode = use_MAP_mode
        self.units = units
        self.kernel_depth = kernel_depth
        self.time_projection_depth = time_projection_depth
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
        self.l2_normalize_states = l2_normalize_states
        self.return_sequences = return_sequences
        self.reuse = reuse
        self.name = name
        self.dtype = dtype
        self.epsilon = epsilon

        self.cell = None

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs_shape[-1]
                    else:
                        units = self.units

                    self.cell = RNNCell(
                        units,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_depth=self.kernel_depth,
                        time_projection_depth=self.time_projection_depth,
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
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build((inputs_shape[0], inputs_shape[2]))

            self.built = True

    @property
    def weights(self):
        return self.cell.weights

    def __call__(self, inputs, times=None, mask=None, return_state=False, initial_state=None):
        if not self.built:
            self.build(inputs.shape)

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

    def ema_ops(self):
        return self.cell.ema_ops()

    def dropout_resample_ops(self):
        return self.cell.dropout_resample_ops()


class RNNCellBayes(RNNCell):
    def __init__(
            self,
            units,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init=None,
            recurrent_kernel_sd_init=None,
            declare_priors_weights=True,
            declare_priors_biases=False,
            kernel_sd_prior=1,
            bias_sd_prior=1,
            bias_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            constraint='softplus',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        super(RNNCellBayes, self).__init__(
            units=units,
            training=training,
            use_MAP_mode=use_MAP_mode,
            kernel_depth=kernel_depth,
            time_projection_depth=time_projection_depth,
            prefinal_mode=prefinal_mode,
            forget_bias=forget_bias,
            forget_gate_as_irf=forget_gate_as_irf,
            activation=activation,
            recurrent_activation=recurrent_activation,
            prefinal_activation=prefinal_activation,
            time_projection_inner_activation=time_projection_inner_activation,
            bottomup_kernel_sd_init=bottomup_kernel_sd_init,
            recurrent_kernel_sd_init=recurrent_kernel_sd_init,
            bottomup_dropout=bottomup_dropout,
            h_dropout=h_dropout,
            c_dropout=c_dropout,
            forget_rate=forget_rate,
            weight_normalization=weight_normalization,
            layer_normalization=layer_normalization,
            use_bias=use_bias,
            global_step=global_step,
            l2_normalize_states=l2_normalize_states,
            reuse=reuse,
            name=name,
            dtype=dtype,
            epsilon=epsilon,
            session=session
        )

        self._declare_priors_weights = declare_priors_weights
        self._declare_priors_biases = declare_priors_biases
        self._kernel_sd_prior = kernel_sd_prior
        self._bias_sd_prior = bias_sd_prior
        self._bias_sd_init = bias_sd_init
        self._posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio

        self._constraint = constraint
        if self._constraint.lower() == 'softplus':
            self._constraint_fn = tf.nn.softplus
            self._constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self._constraint.lower() == 'square':
            self._constraint_fn = tf.square
            self._constraint_fn_inv = tf.sqrt
        elif self._constraint.lower() == 'abs':
            self._constraint_fn = self._safe_abs
            self._constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self._constraint)

        self.kl_penalties_base = {}

    def initialize_kernel(
            self,
            in_dim,
            out_dim,
            kernel_type='bottomup',
            depth=None,
            inner_activation=None,
            prefinal_mode=None,
            name=None
    ):
        with self._session.as_default():
            with self._session.graph.as_default():
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

                if kernel_type == 'bottomup':
                    kernel_sd_init = self._bottomup_kernel_sd_init
                elif kernel_type == 'recurrent':
                    kernel_sd_init = self._recurrent_kernel_sd_init
                else:
                    raise ValueError('Unrecognized kernel type: %s.' % kernel_type)

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
                        use_bias = self._use_bia

                    if self.name:
                        name_cur = self.name + '/'
                    else:
                        name_cur = ''

                    if name:
                        name_cur += name + '_d%d' % d
                    else:
                        name_cur += 'd%d' % d

                    kernel_layer = DenseLayerBayes(
                        training=self._training,
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        declare_priors_weights=self._declare_priors_weights,
                        declare_priors_biases=self._declare_priors_biases,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_sd_prior=self._kernel_sd_prior,
                        kernel_sd_init=kernel_sd_init,
                        bias_sd_prior=self._bias_sd_prior,
                        bias_sd_init=self._bias_sd_init,
                        posterior_to_prior_sd_ratio=self._posterior_to_prior_sd_ratio,
                        constraint=self._constraint,
                        epsilon=self._epsilon,
                        session=self._session,
                        reuse=self._reuse,
                        name=name_cur
                    )

                    kernel_layer.build([None, in_dim])

                    layers.append(kernel_layer)
                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel, layers

    def initialize_irf_biases(self):
        with self._session.as_default():
            with self._session.graph.as_default():
                out_dim = self._num_units
                kernel_sd_prior = get_numerical_sd(self._kernel_sd_prior, in_dim=1, out_dim=out_dim)
                kernel_sd_posterior = kernel_sd_prior * self._posterior_to_prior_sd_ratio

                # Posterior distribution
                t_delta_embedding_W_q_loc = self.add_variable(
                    name='t_delta_embedding_W_q_loc',
                    initializer=tf.zeros_initializer(),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_W_q_scale = self.add_variable(
                    name='t_delta_embedding_W_q_scale',
                    initializer=self._constraint_fn_inv(kernel_sd_posterior),
                    shape=[1, self._num_units]
                )
                self.t_delta_embedding_W_q_dist = Normal(
                    loc=t_delta_embedding_W_q_loc,
                    scale=self._constraint_fn(t_delta_embedding_W_q_scale) + self.epsilon,
                    name='t_delta_embedding_W_q'
                )
                if self._declare_priors_weights:
                    # Prior distribution
                    self.t_delta_embedding_W_prior_dist = Normal(
                        loc=0.,
                        scale=kernel_sd_prior,
                        name='t_delta_embedding_W'
                    )
                    self.kl_penalties_base[self.name + '/t_delta_embedding_W'] = {
                        'loc': 0.,
                        'scale': kernel_sd_prior,
                        'val': self.t_delta_embedding_W_q_dist.kl_divergence(self.t_delta_embedding_W_prior_dist)
                    }
                self.t_delta_embedding_W = tf.cond(
                    self.use_MAP_mode,
                    self.t_delta_embedding_W_q_dist.mean,
                    self.t_delta_embedding_W_q_dist.sample
                )

                bias_sd_prior = get_numerical_sd(self._kernel_sd_prior, in_dim=1, out_dim=1)
                bias_sd_posterior = bias_sd_prior * self._posterior_to_prior_sd_ratio

                # Posterior distribution
                t_delta_embedding_b_q_loc = self.add_variable(
                    name='t_delta_embedding_b_q_loc',
                    initializer=tf.zeros_initializer(),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_b_q_scale = self.add_variable(
                    name='t_delta_embedding_b_q_scale',
                    initializer=self._constraint_fn_inv(bias_sd_posterior),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_b_q_dist = Normal(
                    loc=t_delta_embedding_b_q_loc,
                    scale=self._constraint_fn(t_delta_embedding_b_q_scale) + self.epsilon,
                    name='t_delta_embedding_b_q'
                )
                if self._declare_priors_biases:
                    # Prior distribution
                    self.t_delta_embedding_b_prior_dist = Normal(
                        loc=0.,
                        scale=bias_sd_prior,
                        name='t_delta_embedding_b'
                    )
                    self.kl_penalties_base[self.name + '/t_delta_embedding_b'] = {
                        'loc': 0.,
                        'scale': bias_sd_prior,
                        'val': self.t_delta_embedding_b_q_dist.kl_divergence(self.t_delta_embedding_b_prior_dist)
                    }
                self.t_delta_embedding_b = tf.cond(
                    self.use_MAP_mode,
                    t_delta_embedding_b_q_dist.mean,
                    t_delta_embedding_b_q_dist.sample
                )

    def kl_penalties(self):
        out = self.kl_penalties_base
        for layer in self.layers:
            out.update(layer.kl_penalties())

        return out

    def ema_ops(self):
        return []


class RNNLayerBayes(RNNLayer):
    def __init__(
            self,
            units=None,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init=None,
            recurrent_kernel_sd_init=None,
            declare_priors_weights=True,
            declare_priors_biases=False,
            kernel_sd_prior=1,
            bias_sd_prior=1,
            bias_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            constraint='softplus',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            return_sequences=True,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        super(RNNLayerBayes, self).__init__(
            units=units,
            training=training,
            use_MAP_mode=use_MAP_mode,
            kernel_depth=kernel_depth,
            time_projection_depth=time_projection_depth,
            prefinal_mode=prefinal_mode,
            forget_bias=forget_bias,
            forget_gate_as_irf=forget_gate_as_irf,
            activation=activation,
            recurrent_activation=recurrent_activation,
            prefinal_activation=prefinal_activation,
            time_projection_inner_activation=time_projection_inner_activation,
            bottomup_kernel_sd_init=bottomup_kernel_sd_init,
            recurrent_kernel_sd_init=recurrent_kernel_sd_init,
            bottomup_dropout=bottomup_dropout,
            h_dropout=h_dropout,
            c_dropout=c_dropout,
            forget_rate=forget_rate,
            weight_normalization=weight_normalization,
            layer_normalization=layer_normalization,
            use_bias=use_bias,
            global_step=global_step,
            l2_normalize_states=l2_normalize_states,
            return_sequences=return_sequences,
            reuse=reuse,
            name=name,
            dtype=dtype,
            epsilon=epsilon,
            session=session
        )

        self.declare_priors_weights = declare_priors_weights
        self.declare_priors_biases = declare_priors_biases
        self.kernel_sd_prior = kernel_sd_prior
        self.bias_sd_prior = bias_sd_prior
        self.bias_sd_init = bias_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.constraint = constraint

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs_shape[-1]
                    else:
                        units = self.units

                    self.cell = RNNCellBayes(
                        units,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_depth=self.kernel_depth,
                        time_projection_depth=self.time_projection_depth,
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
                        declare_priors_weights=self.declare_priors_weights,
                        declare_priors_biases=self.declare_priors_biases,
                        kernel_sd_prior=self.kernel_sd_prior,
                        bias_sd_prior=self.bias_sd_prior,
                        bias_sd_init=self.bias_sd_init,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build((inputs_shape[0], inputs_shape[2]))

            self.built = True

    def kl_penalties(self):
        return self.cell.kl_penalties()

    def ema_ops(self):
        return self.cell.ema_ops()


class DenseLayer(object):
    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            units=None,
            use_bias=True,
            activation=None,
            kernel_sd_init='he',
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            normalize_after_activation=False,
            normalization_use_gamma=True,
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        self.session = get_session(session)
        self.reuse = reuse
        self.epsilon = epsilon
        self.name = name

        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.use_MAP_mode = use_MAP_mode
                self.units = units
                self.use_bias = use_bias
                self.activation = get_activation(activation, session=self.session, training=self.training)
                self.kernel_sd_init = kernel_sd_init
                self.use_dropout = bool(dropout)
                self.dropout_layer = get_dropout(
                    dropout,
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    name='dropout',
                    reuse=self.reuse,
                    session=self.session
                )
                self.maxnorm = maxnorm

                self.batch_normalization_decay = batch_normalization_decay
                self.use_batch_normalization = bool(self.batch_normalization_decay)

                self.layer_normalization_type = layer_normalization_type
                if layer_normalization_type is None:
                    self.layer_normalization_type = layer_normalization_type
                elif layer_normalization_type.lower() == 'z':
                    self.layer_normalization_type = 'z'
                elif layer_normalization_type.lower() == 'length':
                    self.layer_normalization_type = 'length'
                else:
                    raise ValueError('Unrecognized layer normalization type: %s' % layer_normalization_type)
                self.use_layer_normalization = bool(self.layer_normalization_type)

                assert not (self.use_batch_normalization and self.use_layer_normalization), 'Cannot batch normalize and layer normalize the same layer.'
                self.normalize_activations = self.use_batch_normalization or self.use_layer_normalization

                self.normalize_after_activation = normalize_after_activation
                self.normalization_use_gamma = normalization_use_gamma

                self.normalization_beta = None
                self.normalization_gamma = None

                if batch_normalization_decay and dropout:
                    stderr('WARNING: Batch normalization and dropout are being applied simultaneously in layer %s.\n         This is usually not a good idea.')

                self.built = False

    @property
    def weights(self):
        out = []
        if self.built:
            out.append(self.kernel)
        if self.normalize_activations and self.built:
            out.append(self.normalization_layer.gamma)
        return out

    def build(self, inputs_shape):
        if not self.built:
            in_dim = inputs_shape[-1]
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
                        sd = get_numerical_sd(self.kernel_sd_init, in_dim=in_dim, out_dim=out_dim)

                        kernel_init = get_initializer(
                            'random_normal_initializer_mean=0-stddev=%s' % sd,
                            session=self.session
                        )
                        self.kernel = tf.get_variable(
                            name='kernel',
                            initializer=kernel_init,
                            shape=[in_dim, out_dim]
                        )

                        if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                        # if self.use_bias and not self.normalize_activations:
                            self.bias = tf.get_variable(
                                name='bias',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                            )

                        if self.use_dropout:
                            self.dropout_layer.build([x for x in inputs_shape[:-1]] + [out_dim])

                        if self.use_batch_normalization:
                            self.normalization_layer = BatchNormLayer(
                                decay=self.batch_normalization_decay,
                                shift_activations=self.use_bias,
                                rescale_activations=self.normalization_use_gamma,
                                axis=-1,
                                training=self.training,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )
                        elif self.use_batch_normalization:
                            self.normalization_layer = LayerNormLayer(
                                normalization_type=self.layer_normalization_type,
                                shift_activations=self.use_bias,
                                rescale_activations=self.normalization_use_gamma,
                                axis=-1,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                kernel = self.kernel
                if self.maxnorm:
                    kernel = tf.clip_by_norm(kernel, self.maxnorm, axes=[0])
                H = tf.tensordot(inputs, kernel, 1)
                if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                # if self.use_bias and not self.normalize_activations:
                    bias = self.bias
                    while len(bias.shape) < len(H.shape):
                        bias = bias[None, ...]
                    H += bias

                if self.activation is not None and self.normalize_after_activation:
                    H = self.activation(H)

                if self.normalize_activations:
                    # H = tf.contrib.layers.batch_norm(
                    #     H,
                    #     decay=self.batch_normalization_decay,
                    #     center=self.use_bias,
                    #     scale=self.batch_normalization_use_gamma,
                    #     zero_debias_moving_mean=True,
                    #     epsilon=self.epsilon,
                    #     is_training=self.training,
                    #     updates_collections=None,
                    #     reuse=self.reuse,
                    #     scope=self.name
                    # )
                    H = self.normalization_layer(H)

                if self.activation is not None and not self.normalize_after_activation:
                    H = self.activation(H)

                H = self.dropout_layer(H)

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def ema_ops(self):
        out = []
        if self.use_batch_normalization and self.built:
            out += self.normalization_layer.ema_ops()

        return out

    def dropout_resample_ops(self):
        out = []
        if self.use_dropout and self.built:
            out.append(self.dropout_layer.dropout_resample_ops())

        return out


class DenseLayerBayes(DenseLayer):

    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            units=None,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            normalize_after_activation=False,
            normalization_use_gamma=True,
            declare_priors_weights=True,
            declare_priors_biases=False,
            declare_priors_gamma=False,
            kernel_sd_prior=1,
            kernel_sd_init=None,
            bias_sd_prior=1.,
            bias_sd_init=None,
            gamma_sd_prior=1.,
            gamma_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            constraint='softplus',
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        super(DenseLayerBayes, self).__init__(
            training=training,
            use_MAP_mode=use_MAP_mode,
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_sd_init=kernel_sd_init,
            dropout=dropout,
            maxnorm=maxnorm,
            batch_normalization_decay=batch_normalization_decay,
            layer_normalization_type=layer_normalization_type,
            normalize_after_activation=normalize_after_activation,
            normalization_use_gamma=normalization_use_gamma,
            reuse=reuse,
            epsilon=epsilon,
            session=session,
            name=name
        )
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.declare_priors_weights = declare_priors_weights
                self.declare_priors_biases = declare_priors_biases
                self.declare_priors_gamma = declare_priors_gamma
                self.kernel_sd_prior = kernel_sd_prior
                self.bias_sd_prior = bias_sd_prior
                self.bias_sd_init = bias_sd_init
                self.gamma_sd_prior = gamma_sd_prior
                self.gamma_sd_init = gamma_sd_init
                self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio

                self.constraint = constraint
                if self.constraint.lower() == 'softplus':
                    self.constraint_fn = tf.nn.softplus
                    self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
                elif self.constraint.lower() == 'square':
                    self.constraint_fn = tf.square
                    self.constraint_fn_inv = tf.sqrt
                elif self.constraint.lower() == 'abs':
                    self.constraint_fn = self._safe_abs
                    self.constraint_fn_inv = tf.identity
                else:
                    raise ValueError('Unrecognized constraint function %s' % self.constraint)

                self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            in_dim = inputs_shape[-1]
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
                        kernel_sd_prior = get_numerical_sd(self.kernel_sd_prior, in_dim=in_dim, out_dim=out_dim)
                        if self.kernel_sd_init:
                            kernel_sd_posterior = get_numerical_sd(self.kernel_sd_init, in_dim=in_dim, out_dim=out_dim)
                        else:
                            kernel_sd_posterior = kernel_sd_prior * self.posterior_to_prior_sd_ratio

                        if self.use_MAP_mode is None:
                            self.use_MAP_mode = tf.logical_not(self.training)

                        # Posterior distribution
                        self.kernel_q_loc = tf.get_variable(
                            name='kernel_q_loc',
                            initializer=tf.zeros_initializer(),
                            shape=[in_dim, out_dim]
                        )
                        self.kernel_q_scale = tf.get_variable(
                            name='kernel_q_scale',
                            initializer=lambda: tf.ones([in_dim, out_dim]) * self.constraint_fn_inv(kernel_sd_posterior)
                        )
                        self.kernel_q_dist = Normal(
                            loc=self.kernel_q_loc,
                            scale=self.constraint_fn(self.kernel_q_scale) + self.epsilon,
                            name='kernel_q'
                        )
                        if self.declare_priors_weights:
                            # Prior distribution
                            self.kernel_prior_dist = Normal(
                                loc=0.,
                                scale=kernel_sd_prior,
                                name='kernel'
                            )
                            self.kl_penalties_base[self.name + '/kernel'] = {
                                'loc': 0.,
                                'scale': kernel_sd_prior,
                                'val': self.kernel_q_dist.kl_divergence(self.kernel_prior_dist)
                            }
                        self.kernel = tf.cond(
                            self.use_MAP_mode,
                            self.kernel_q_dist.mean,
                            self.kernel_q_dist.sample
                        )

                        if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                        # if self.use_bias and not self.normalize_activations:
                            bias_sd_prior = get_numerical_sd(self.bias_sd_prior, in_dim=1, out_dim=1)
                            if self.bias_sd_init:
                                bias_sd_posterior = get_numerical_sd(self.bias_sd_init, in_dim=1, out_dim=1)
                            else:
                                bias_sd_posterior = bias_sd_prior * self.posterior_to_prior_sd_ratio

                            # Posterior distribution
                            self.bias_q_loc = tf.get_variable(
                                name='bias_q_loc',
                                initializer=tf.zeros_initializer(),
                                shape=[out_dim]
                            )
                            self.bias_q_scale = tf.get_variable(
                                name='bias_q_scale',
                                initializer=lambda: tf.ones([out_dim]) * self.constraint_fn_inv(bias_sd_posterior)
                            )

                            self.bias_q_dist = Normal(
                                loc=self.bias_q_loc,
                                scale=self.constraint_fn(self.bias_q_scale) + self.epsilon,
                                name='bias_q'
                            )
                            if self.declare_priors_biases:
                                # Prior distribution
                                self.bias_prior_dist = Normal(
                                    loc=0.,
                                    scale=bias_sd_prior,
                                    name='bias'
                                )
                                self.kl_penalties_base[self.name + '/bias'] = {
                                    'loc': 0.,
                                    'scale': bias_sd_prior,
                                    'val': self.bias_q_dist.kl_divergence(self.bias_prior_dist)
                                }
                            self.bias = tf.cond(
                                self.use_MAP_mode,
                                self.bias_q_dist.mean,
                                self.bias_q_dist.sample
                            )

                        if self.use_batch_normalization:
                            self.normalization_layer = BatchNormLayerBayes(
                                decay=self.batch_normalization_decay,
                                shift_activations=self.use_bias,
                                rescale_activations=self.normalization_use_gamma,
                                axis=-1,
                                use_MAP_mode=self.use_MAP_mode,
                                declare_priors_scale=self.declare_priors_gamma,
                                declare_priors_shift=self.declare_priors_biases,
                                scale_sd_prior=self.gamma_sd_prior,
                                scale_sd_init=self.gamma_sd_init,
                                shift_sd_prior=self.bias_sd_prior,
                                shift_sd_init=self.bias_sd_init,
                                posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                                constraint=self.constraint,
                                training=self.training,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )
                        elif self.use_layer_normalization:
                            self.normalization_layer = LayerNormLayerBayes(
                                normalization_type=self.layer_normalization_type,
                                shift_activations=self.use_bias,
                                rescale_activations=self.normalization_use_gamma,
                                axis=-1,
                                use_MAP_mode=self.use_MAP_mode,
                                declare_priors_scale=self.declare_priors_gamma,
                                declare_priors_shift=self.declare_priors_biases,
                                scale_sd_prior=self.gamma_sd_prior,
                                scale_sd_init=self.gamma_sd_init,
                                shift_sd_prior=self.bias_sd_prior,
                                shift_sd_init=self.bias_sd_init,
                                posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                                constraint=self.constraint,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )


            self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.kl_penalties_base.copy()
                if self.batch_normalization_decay:
                    out.update(self.normalization_layer.kl_penalties())

                return out


class BatchNormLayer(object):
    def __init__(
            self,
            decay=0.999,
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        self.session = get_session(session)
        self.decay = decay
        self.shift_activations = shift_activations
        self.rescale_activations = rescale_activations
        self.axis = axis
        self.epsilon = epsilon
        self.training = training
        self.reuse = reuse
        self.name = name

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = len(inputs_shape) - 1
            else:
                axis = self.axis

            self.reduction_axes = sorted(list(set(range(len(inputs_shape))) - {axis}))

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(1)
                else:
                    shape.append(inputs_shape[i])

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.moving_mean = tf.get_variable(
                            name='moving_mean',
                            initializer=tf.zeros_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_mean_op = None

                        self.moving_variance = tf.get_variable(
                            name='moving_variance',
                            initializer=tf.ones_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_variance_op = None

                        if self.shift_activations:
                            self.beta = tf.get_variable(
                                name='beta',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            self.gamma = tf.get_variable(
                                name='gamma',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                name = self.name
                if not name:
                    name = ''
                decay = self.decay

                def train_fn(inputs=inputs):
                    return tf.nn.moments(inputs, self.reduction_axes, keep_dims=True)

                def eval_fn():
                    return self.moving_mean, self.moving_variance

                mean, variance = tf.cond(self.training, train_fn, eval_fn)

                sd = tf.sqrt(variance)
                out = ((inputs - mean) / (sd + self.epsilon)) * self.gamma + self.beta

                if self.moving_mean_op is None:
                    self.moving_mean_op = tf.assign(
                        self.moving_mean,
                        self.moving_mean * decay + mean * (1 - decay),
                        name=name + '_moving_mean_update'
                    )
                if self.moving_variance_op is None:
                    self.moving_variance_op = tf.assign(
                        self.moving_variance,
                        self.moving_variance * decay + variance * (1 - decay),
                        name=name + '_moving_variance_update'
                    )

                return out

    def ema_ops(self):
        return [self.moving_mean_op, self.moving_variance_op]

    def dropout_resample_ops(self):
        return []


class BatchNormLayerBayes(BatchNormLayer):
    def __init__(
            self,
            decay=0.999,
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            use_MAP_mode=None,
            declare_priors_scale=True,
            declare_priors_shift=False,
            scale_sd_prior=1.,
            scale_sd_init=None,
            shift_sd_prior=1.,
            shift_sd_init=None,
            posterior_to_prior_sd_ratio=1.,
            constraint='softplus',
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        super(BatchNormLayerBayes, self).__init__(
            decay=decay,
            shift_activations=shift_activations,
            rescale_activations=rescale_activations,
            axis=axis,
            training=training,
            epsilon=epsilon,
            session=session,
            reuse=reuse,
            name=name
        )

        self.use_MAP_mode = use_MAP_mode
        self.declare_priors_scale = declare_priors_scale
        self.declare_priors_shift = declare_priors_shift
        self.scale_sd_prior = scale_sd_prior
        self.scale_sd_init = scale_sd_init
        self.shift_sd_prior = shift_sd_prior
        self.shift_sd_init = shift_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.constraint = constraint

        if self.constraint.lower() == 'softplus':
            self.constraint_fn = tf.nn.softplus
            self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self.constraint.lower() == 'square':
            self.constraint_fn = tf.square
            self.constraint_fn_inv = tf.sqrt
        elif self.constraint.lower() == 'abs':
            self.constraint_fn = self._safe_abs
            self.constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self.constraint)

        self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = len(inputs_shape) - 1
            else:
                axis = self.axis

            self.reduction_axes = sorted(list(set(range(len(inputs_shape))) - {axis}))

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(1)
                else:
                    shape.append(inputs_shape[i])

            if not self.name:
                name = ''
            else:
                name = self.name

            if self.use_MAP_mode is None:
                self.use_MAP_mode = tf.logical_not(self.training)

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.moving_mean = tf.get_variable(
                            name='moving_mean',
                            initializer=tf.zeros_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_mean_op = None

                        self.moving_variance = tf.get_variable(
                            name='moving_variance',
                            initializer=tf.ones_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_variance_op = None
                        
                        if self.shift_activations:
                            shift_sd_prior = get_numerical_sd(self.shift_sd_prior, in_dim=1, out_dim=1)
                            if self.shift_sd_init:
                                shift_sd_posterior = get_numerical_sd(self.shift_sd_init, in_dim=1, out_dim=1)
                            else:
                                shift_sd_posterior = shift_sd_prior * self.posterior_to_prior_sd_ratio

                            # Posterior distribution
                            self.beta_q_loc = tf.get_variable(
                                name='beta_q_loc',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                            self.beta_q_scale = tf.get_variable(
                                name='beta_q_scale',
                                initializer=lambda: tf.ones(shape) * self.constraint_fn_inv(shift_sd_posterior)
                            )

                            self.beta_q_dist = Normal(
                                loc=self.beta_q_loc,
                                scale=self.constraint_fn(self.beta_q_scale) + self.epsilon,
                                name='beta_q'
                            )
                            if self.declare_priors_shift:
                                # Prior distribution
                                self.beta_prior_dist = Normal(
                                    loc=0.,
                                    scale=shift_sd_prior,
                                    name='beta'
                                )
                                self.kl_penalties_base[self.name + '/beta'] = {
                                    'loc': 0.,
                                    'scale': shift_sd_prior,
                                    'val': self.beta_q_dist.kl_divergence(self.beta_prior_dist)
                                }
                            self.beta = tf.cond(
                                self.use_MAP_mode,
                                self.beta_q_dist.mean,
                                self.beta_q_dist.sample
                            )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            scale_sd_prior = get_numerical_sd(self.scale_sd_prior, in_dim=1, out_dim=1)
                            if self.scale_sd_init:
                                scale_sd_posterior = get_numerical_sd(self.scale_sd_init, in_dim=1, out_dim=1)
                            else:
                                scale_sd_posterior = scale_sd_prior * self.posterior_to_prior_sd_ratio

                            # Posterior distribution
                            self.gamma_q_loc = tf.get_variable(
                                name='gamma_q_loc',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                            self.gamma_q_scale = tf.get_variable(
                                name='gamma_q_scale',
                                initializer=lambda: tf.ones(shape) * self.constraint_fn_inv(scale_sd_posterior)
                            )

                            self.gamma_q_dist = Normal(
                                loc=self.gamma_q_loc,
                                scale=self.constraint_fn(self.gamma_q_scale) + self.epsilon,
                                name='gamma_q'
                            )
                            if self.declare_priors_shift:
                                # Prior distribution
                                self.gamma_prior_dist = Normal(
                                    loc=1.,
                                    scale=scale_sd_prior,
                                    name='gamma'
                                )
                                self.kl_penalties_base[self.name + '/gamma'] = {
                                    'loc': 1.,
                                    'scale': scale_sd_prior,
                                    'val': self.gamma_q_dist.kl_divergence(self.gamma_prior_dist)
                                }
                            self.gamma = tf.cond(
                                self.use_MAP_mode,
                                self.gamma_q_dist.mean,
                                self.gamma_q_dist.sample
                            )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                return self.kl_penalties_base.copy()


class LayerNormLayer(object):
    def __init__(
            self,
            normalization_type='z',
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        self.session = get_session(session)
        self.normalization_type = normalization_type
        assert self.normalization_type in ['z', 'length'], 'Unrecognized normalization type: %s' % self.normalization_type
        self.shift_activations = shift_activations
        self.rescale_activations = rescale_activations
        self.axis = axis
        self.epsilon = epsilon
        self.reuse = reuse
        self.name = name

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = [len(inputs_shape) - 1]
            else:
                axis = [self.axis]
            if isinstance(axis, int):
                axis = [axis]

            self.reduction_axes = axis

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(1)
                else:
                    shape.append(inputs_shape[i])

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        if self.shift_activations:
                            self.beta = tf.get_variable(
                                name='beta',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            self.gamma = tf.get_variable(
                                name='gamma',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                if self.normalization_type == 'z':
                    mean, variance = tf.nn.moments(inputs, self.reduction_axes, keep_dims=True)
                    sd = tf.sqrt(variance)
                    out = ((inputs - mean) / (sd + self.epsilon))
                else: # length normalization
                    out = tf.nn.l2_normalize(inputs, axis=self.reduction_axes, epsilon=self.epsilon)

                out = out * self.gamma + self.beta

                return out

    def ema_ops(self):
        return []

    def dropout_resample_ops(self):
        return []


class LayerNormLayerBayes(LayerNormLayer):
    def __init__(
            self,
            normalization_type='z',
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            use_MAP_mode=None,
            declare_priors_scale=True,
            declare_priors_shift=False,
            scale_sd_prior=1.,
            scale_sd_init=None,
            shift_sd_prior=1.,
            shift_sd_init=None,
            posterior_to_prior_sd_ratio=1.,
            constraint='softplus',
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        super(LayerNormLayerBayes, self).__init__(
            normalization_type=normalization_type,
            shift_activations=shift_activations,
            rescale_activations=rescale_activations,
            axis=axis,
            epsilon=epsilon,
            session=session,
            reuse=reuse,
            name=name
        )

        self.use_MAP_mode = use_MAP_mode
        self.declare_priors_scale = declare_priors_scale
        self.declare_priors_shift = declare_priors_shift
        self.scale_sd_prior = scale_sd_prior
        self.scale_sd_init = scale_sd_init
        self.shift_sd_prior = shift_sd_prior
        self.shift_sd_init = shift_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.constraint = constraint

        if self.constraint.lower() == 'softplus':
            self.constraint_fn = tf.nn.softplus
            self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self.constraint.lower() == 'square':
            self.constraint_fn = tf.square
            self.constraint_fn_inv = tf.sqrt
        elif self.constraint.lower() == 'abs':
            self.constraint_fn = self._safe_abs
            self.constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self.constraint)

        self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = [len(inputs_shape) - 1]
            else:
                axis = [self.axis]
            if isinstance(axis, int):
                axis = [axis]

            self.reduction_axes = axis

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(inputs_shape[i])
                else:
                    shape.append(1)

            if not self.name:
                name = ''
            else:
                name = self.name

            if self.use_MAP_mode is None:
                self.use_MAP_mode = tf.logical_not(self.training)

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.moving_mean = tf.get_variable(
                            name='moving_mean',
                            initializer=tf.zeros_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_mean_op = None

                        self.moving_variance = tf.get_variable(
                            name='moving_variance',
                            initializer=tf.ones_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_variance_op = None

                        if self.shift_activations:
                            shift_sd_prior = get_numerical_sd(self.shift_sd_prior, in_dim=1, out_dim=1)
                            if self.shift_sd_init:
                                shift_sd_posterior = get_numerical_sd(self.shift_sd_init, in_dim=1, out_dim=1)
                            else:
                                shift_sd_posterior = shift_sd_prior * self.posterior_to_prior_sd_ratio

                            # Posterior distribution
                            self.beta_q_loc = tf.get_variable(
                                name='beta_q_loc',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                            self.beta_q_scale = tf.get_variable(
                                name='beta_q_scale',
                                initializer=lambda: tf.ones(shape) * self.constraint_fn_inv(shift_sd_posterior)
                            )

                            self.beta_q_dist = Normal(
                                loc=self.beta_q_loc,
                                scale=self.constraint_fn(self.beta_q_scale) + self.epsilon,
                                name='beta_q'
                            )
                            if self.declare_priors_shift:
                                # Prior distribution
                                self.beta_prior_dist = Normal(
                                    loc=0.,
                                    scale=shift_sd_prior,
                                    name='beta'
                                )
                                self.kl_penalties_base[self.name + '/beta'] = {
                                    'loc': 0.,
                                    'scale': shift_sd_prior,
                                    'val': self.beta_q_dist.kl_divergence(self.beta_prior_dist)
                                }
                            self.beta = tf.cond(
                                self.use_MAP_mode,
                                self.beta_q_dist.mean,
                                self.beta_q_dist.sample
                            )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            scale_sd_prior = get_numerical_sd(self.scale_sd_prior, in_dim=1, out_dim=1)
                            if self.scale_sd_init:
                                scale_sd_posterior = get_numerical_sd(self.scale_sd_init, in_dim=1, out_dim=1)
                            else:
                                scale_sd_posterior = scale_sd_prior * self.posterior_to_prior_sd_ratio

                            # Posterior distribution
                            self.gamma_q_loc = tf.get_variable(
                                name='gamma_q_loc',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                            self.gamma_q_scale = tf.get_variable(
                                name='gamma_q_scale',
                                initializer=lambda: tf.ones(shape) * self.constraint_fn_inv(scale_sd_posterior)
                            )

                            self.gamma_q_dist = Normal(
                                loc=self.gamma_q_loc,
                                scale=self.constraint_fn(self.gamma_q_scale) + self.epsilon,
                                name='gamma_q'
                            )
                            if self.declare_priors_shift:
                                # Prior distribution
                                self.gamma_prior_dist = Normal(
                                    loc=1.,
                                    scale=scale_sd_prior,
                                    name='gamma'
                                )
                                self.kl_penalties_base[self.name + '/gamma'] = {
                                    'loc': 1.,
                                    'scale': scale_sd_prior,
                                    'val': self.gamma_q_dist.kl_divergence(self.gamma_prior_dist)
                                }
                            self.gamma = tf.cond(
                                self.use_MAP_mode,
                                self.gamma_q_dist.mean,
                                self.gamma_q_dist.sample
                            )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                return self.kl_penalties_base.copy()


class DropoutLayer(object):
    def __init__(
            self,
            rate,
            noise_shape=None,
            training=False,
            use_MAP_mode=True,
            rescale=True,
            constant=None,
            name=None,
            reuse=tf.AUTO_REUSE,
            session=None
    ):
        self.rate = rate
        self.noise_shape = noise_shape
        self.training = training
        self.use_MAP_mode = use_MAP_mode
        self.rescale = rescale
        self.constant = constant
        self.name = name
        self.reuse = reuse
        self.session = get_session(session)

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.noise_shape:
                        noise_shape = [inputs_shape[i] if self.noise_shape[i] is None else self.noise_shape[i] for i in
                                       range(len(self.noise_shape))]
                    else:
                        noise_shape = inputs_shape

                    self.noise_shape = noise_shape

                    if self.noise_shape:
                        if self.noise_shape[-1] is None:
                            final_shape = inputs_shape[-1]
                        else:
                            final_shape = self.noise_shape[-1]
                    else:
                        final_shape = inputs_shape[-1]

                    if not self.name:
                        name = ''
                    else:
                        name = self.name

                    with tf.variable_scope(name, reuse=self.reuse):
                        self.noise_shape_eval = [1 for _ in range(len(inputs_shape) - 1)] + [int(final_shape)]
                        self.dropout_mask_eval_sample = tf.random_uniform(self.noise_shape_eval) > self.rate
                        self.dropout_mask_eval = tf.get_variable(
                            name='mask',
                            initializer=tf.ones_initializer(),
                            shape=self.noise_shape_eval,
                            dtype=tf.bool,
                            trainable=False
                        )
                        self.dropout_mask_eval_resample = tf.assign(self.dropout_mask_eval, self.dropout_mask_eval_sample)

                    self.built = True

    def __call__(self, inputs, seed=None):
        if not self.built:
            self.build(inputs.shape)
        with self.session.as_default():
            with self.session.graph.as_default():
                def train_fn(inputs=inputs):
                    inputs_shape = tf.shape(inputs)
                    noise_shape = []
                    for i, x in enumerate(self.noise_shape):
                        try:
                            noise_shape.append(int(x))
                        except TypeError:
                            noise_shape.append(inputs_shape[i])

                    dropout_mask = tf.random_uniform(noise_shape) > self.rate

                    return dropout_mask

                def eval_fn(inputs=inputs):
                    def map_fn(inputs=inputs):
                        return tf.ones(tf.shape(inputs), dtype=tf.bool)

                    def sample_fn():
                        dropout_mask = self.dropout_mask_eval
                        return dropout_mask

                    return tf.cond(self.use_MAP_mode, map_fn, sample_fn)

                dropout_mask = tf.cond(self.training, train_fn, eval_fn)

                if self.constant is None:
                    dropout_mask = tf.cast(dropout_mask, dtype=inputs.dtype)
                    out = inputs * dropout_mask
                else:
                    defaults = tf.ones_like(inputs) * self.constant
                    out = tf.where(dropout_mask, inputs, defaults)
                    # out = tf.Print(out, ['in', inputs, 'mask', dropout_mask, 'defaults', defaults, 'output', out], summarize=100)

                if self.rescale:
                    def rescale(x=out, rate=self.rate):
                        out = x * (1. / (1. - rate))
                        return out

                    out = tf.cond(
                        tf.logical_or(self.training, tf.logical_not(self.use_MAP_mode)),
                        rescale,
                        lambda: out
                    )

                return out

    def dropout_resample_ops(self):
        out = []
        if self.built:
            out.append(self.dropout_mask_eval_resample)

        return out
