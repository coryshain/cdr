import math
import numpy as np
import pandas as pd

from .kwargs import CDRNNMLE_INITIALIZATION_KWARGS
from .backend import get_initializer, DenseLayer, RNNLayer, BatchNormLayer, LayerNormLayer
from .cdrnnbase import CDRNN
from .util import sn, reg_name, stderr

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, SinhArcsinh

pd.options.mode.chained_assignment = None


class CDRNNMLE(CDRNN):
    _INITIALIZATION_KWARGS = CDRNNMLE_INITIALIZATION_KWARGS

    _doc_header = """
        A CDRRNN implementation fitted using maximum likelihood estimation.
    """
    _doc_args = CDRNN._doc_args
    _doc_kwargs = CDRNN._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    def __init__(self, form_str, X, Y, **kwargs):
        super(CDRNNMLE, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDRNNMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(CDRNNMLE, self)._initialize_metadata()

        self.parameter_table_columns = ['Mean', '2.5%', '97.5%']

    def _pack_metadata(self):
        md = super(CDRNNMLE, self)._pack_metadata()
        for kwarg in CDRNNMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(CDRNNMLE, self)._unpack_metadata(md)

        for kwarg in CDRNNMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            stderr('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))





    ######################################################
    #
    #  Network initialization
    #
    ######################################################

    def initialize_intercept(self, response_name, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                init = tf.constant(self.intercept_init[response_name], dtype=self.FLOAT_TF)
                name = sn(response_name)
                if ran_gf is None:
                    intercept = tf.Variable(
                        init,
                        dtype=self.FLOAT_TF,
                        name='intercept_%s' % name
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    shape = [rangf_n_levels] + [int(x) for x in init.shape]
                    intercept = tf.Variable(
                        tf.zeros(shape, dtype=self.FLOAT_TF),
                        name='intercept_%s_by_%s' % (name, sn(ran_gf))
                    )
                intercept_summary = intercept

                return intercept, intercept_summary

    def initialize_input_bias(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = len(self.impulse_names) + 1
                if ran_gf is None:
                    input_bias = tf.zeros([1, 1, units])
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    input_bias = tf.Variable(
                        tf.zeros([rangf_n_levels, units]),
                        name='input_bias_by_%s' % (sn(ran_gf))
                    )
                input_bias_summary = input_bias

                return input_bias, input_bias_summary

    def initialize_feedforward(
            self,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            name=None,
            final=False
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                projection = DenseLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    use_bias=use_bias,
                    activation=activation,
                    dropout=dropout,
                    maxnorm=maxnorm,
                    batch_normalization_decay=batch_normalization_decay,
                    layer_normalization_type=layer_normalization_type,
                    normalize_after_activation=self.normalize_after_activation,
                    normalization_use_gamma=self.normalization_use_gamma,
                    kernel_sd_init=self.weight_sd_init,
                    epsilon=self.epsilon,
                    session=self.sess,
                    name=name
                )

                return projection

    def initialize_rnn(self, l):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                rnn = RNNLayer(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    time_projection_depth=self.n_layers_irf + 1,
                    activation=self.rnn_activation,
                    recurrent_activation=self.recurrent_activation,
                    time_projection_inner_activation=self.irf_inner_activation,
                    bottomup_kernel_sd_init=self.weight_sd_init,
                    recurrent_kernel_sd_init=self.weight_sd_init,
                    bottomup_dropout=self.input_projection_dropout_rate,
                    h_dropout=self.rnn_h_dropout_rate,
                    c_dropout=self.rnn_c_dropout_rate,
                    forget_rate=self.forget_rate,
                    return_sequences=True,
                    name='rnn_l%d' % (l + 1),
                    epsilon=self.epsilon,
                    session=self.sess
                )

                return rnn

    def initialize_rnn_h(self, l, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                if ran_gf is None:
                    rnn_h = tf.Variable(tf.zeros([1, units]), name='rnn_h_l%d' % (l+1))
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    rnn_h = tf.Variable(
                        tf.zeros([rangf_n_levels, self.n_units_rnn[l]]),
                        name='rnn_h_ran_l%d_by_%s' % (l+1, sn(ran_gf))
                    )
                rnn_h_summary = rnn_h

                return rnn_h, rnn_h_summary

    def initialize_rnn_c(self, l, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                if ran_gf is None:
                    rnn_c = tf.Variable(tf.zeros([1, units]), name='rnn_c_l%d' % (l+1))
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    rnn_c = tf.Variable(
                        tf.zeros([rangf_n_levels, units]),
                        name='rnn_c_ran_l%d_by_%s' % (l+1, sn(ran_gf))
                    )
                rnn_c_summary = rnn_c

                return rnn_c, rnn_c_summary

    def initialize_h_bias(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_hidden_state
                if ran_gf is None:
                    h_bias = tf.Variable(tf.zeros([1, 1, units]), name='h_bias')
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    h_bias = tf.Variable(
                        tf.zeros([rangf_n_levels, units]),
                        name='h_bias_by_%s' % (sn(ran_gf))
                    )
                h_bias_summary = h_bias

                return h_bias, h_bias_summary

    def initialize_h_normalization(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.use_batch_normalization:
                    normalization_layer = BatchNormLayer(
                        decay=self.batch_normalization_decay,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        training=self.training,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='h'
                    )
                elif self.use_layer_normalization:
                    normalization_layer = LayerNormLayer(
                        normalization_type=self.layer_normalization_type,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='h'
                    )
                else:
                    normalization_layer = lambda x: x

                return normalization_layer
            
    def initialize_irf_l1_weights(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                if ran_gf is None:
                    if isinstance(self.weight_sd_init, str):
                        if self.weight_sd_init.lower() in ['xavier', 'glorot']:
                            sd = math.sqrt(2 / (1 + self.n_units_irf_l1))
                        elif self.weight_sd_init.lower() == 'he':
                            sd = math.sqrt(2)
                        else:
                            sd = float(self.weight_sd_init)
                    else:
                        sd = self.weight_sd_init

                    kernel_init = get_initializer(
                        'random_normal_initializer_mean=0-stddev=%s' % sd,
                        session=self.sess
                    )
                    irf_l1_W = tf.get_variable(
                        name='irf_l1_W',
                        initializer=kernel_init,
                        shape=[1, 1, units]
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_l1_W = tf.get_variable(
                        name='irf_l1_W_by_%s' % (sn(ran_gf)),
                        initializer=tf.zeros_initializer(),
                        shape=[rangf_n_levels, units],
                    )

                irf_l1_W_summary = irf_l1_W

                return irf_l1_W, irf_l1_W_summary

    def initialize_irf_l1_biases(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                if ran_gf is None:
                    irf_l1_b = tf.get_variable(
                        name='irf_l1_b',
                        initializer=tf.zeros_initializer(),
                        shape=[1, 1, units]
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_l1_b = tf.get_variable(
                        name='irf_l1_b_by_%s' % (sn(ran_gf)),
                        initializer=tf.zeros_initializer(),
                        shape=[rangf_n_levels, units],
                    )

                irf_l1_b_summary = irf_l1_b

                return irf_l1_b, irf_l1_b_summary

    def initialize_error_params_biases(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.asymmetric_error:
                    units = 3
                else:
                    units = 1
                if ran_gf is None:
                    error_params_b = tf.zeros([1, units])
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    error_params_b = tf.get_variable(
                        name='error_params_b_by_%s' % (sn(ran_gf)),
                        initializer=tf.zeros_initializer(),
                        shape=[rangf_n_levels, units],
                    )

                error_params_b_summary = error_params_b

                return error_params_b, error_params_b_summary

    def initialize_irf_l1_normalization(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.use_batch_normalization:
                    normalization_layer = BatchNormLayer(
                        decay=self.batch_normalization_decay,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        training=self.training,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='irf_l1'
                    )
                elif self.use_layer_normalization:
                    normalization_layer = LayerNormLayer(
                        normalization_type=self.layer_normalization_type,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='irf_l1'
                    )
                else:
                    normalization_layer = lambda x: x

                return normalization_layer

    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    out = self.parameter_table_fixed_values.eval(session=self.sess)
                else:
                    out = self.parameter_table_random_values.eval(session=self.sess)

                self.set_predict_mode(False)

            out = np.stack([out, out, out], axis=1)

            return out





    ######################################################
    #
    #  Public methods
    #
    ######################################################


    def report_settings(self, indent=0):
        out = super(CDRNNMLE, self).report_settings(indent=indent)
        for kwarg in CDRNNMLE_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out
