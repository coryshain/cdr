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

    def __init__(self, form_str, X, y, **kwargs):
        super(CDRNNMLE, self).__init__(
            form_str,
            X,
            y,
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

    def initialize_intercept(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    intercept = tf.Variable(
                        self.intercept_init_tf,
                        dtype=self.FLOAT_TF,
                        name='intercept'
                    )
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    intercept = tf.Variable(
                        tf.zeros([rangf_n_levels], dtype=self.FLOAT_TF),
                        name='intercept_by_%s' % sn(ran_gf)
                    )
                intercept_summary = intercept

                return intercept, intercept_summary

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = len(self.impulse_names) + 1
                if ran_gf is None:
                    coefficient = 1.
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    coefficient = tf.Variable(
                        tf.zeros([rangf_n_levels, units]),
                        name='coefficient_by_%s' % (sn(ran_gf))
                    )
                coefficient_summary = coefficient

                return coefficient, coefficient_summary

    def initialize_feedforward(
            self,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
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
                if self.split_h:
                    units *= 2
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
                    error_params_b = tf.get_variable(
                        name='error_params_b',
                        initializer=tf.zeros_initializer(),
                        shape=[1, 1, units]
                    )
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

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.y_sd_base_unconstrained = tf.Variable(
                    self.y_sd_init_unconstrained,
                    dtype=self.FLOAT_TF,
                    name='y_sd_base_unconstrained'
                )
                self.y_sd = self.constraint_fn(self.y_sd_base_unconstrained + self.y_sd_delta) + self.epsilon
                self.y_sd_summary = self.constraint_fn(self.y_sd_base_unconstrained + self.y_sd_delta_ema) + self.epsilon
                tf.summary.scalar(
                    'error/y_sd',
                    self.y_sd_summary,
                    collections=['params']
                )

                if self.asymmetric_error:
                    self.y_skewness_base = tf.Variable(
                        0.,
                        dtype=self.FLOAT_TF,
                        name='y_skewness_base'
                    )
                    self.y_skewness = self.y_skewness_base + self.y_skewness_delta
                    self.y_skewness_summary = self.y_skewness_base + self.y_skewness_delta_ema
                    tf.summary.scalar(
                        'error/y_skewness',
                        self.y_skewness_summary,
                        collections=['params']
                    )

                    self.y_tailweight_base_unconstrained = tf.Variable(
                        self.y_tailweight_init_unconstrained,
                        dtype=self.FLOAT_TF,
                        name='y_tailweight_base_unconstrained'
                    )
                    self.y_tailweight = self.constraint_fn(self.y_tailweight_base_unconstrained + self.y_tailweight_delta) + self.epsilon
                    self.y_tailweight_summary = self.constraint_fn(self.y_tailweight_base_unconstrained + self.y_tailweight_delta_ema) + self.epsilon
                    tf.summary.scalar(
                        'error/y_tailweight',
                        self.y_tailweight_summary,
                        collections=['params']
                    )

                if self.standardize_response:
                    y_standardized = (self.y - self.y_train_mean) / self.y_train_sd

                    if self.asymmetric_error:
                        y_dist_standardized = SinhArcsinh(
                            loc=self.out,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        y_dist = SinhArcsinh(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=self.y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )

                        self.err_dist_standardized = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )

                        self.err_dist_summary_standardized = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                        self.err_dist_summary = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary * self.y_train_sd,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                    else:
                        y_dist_standardized = Normal(
                            loc=self.out,
                            scale=self.y_sd
                        )
                        y_dist = Normal(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=self.y_sd * self.y_train_sd
                        )

                        self.err_dist_standardized = Normal(
                            loc=0.,
                            scale=self.y_sd
                        )
                        self.err_dist = Normal(
                            loc=0.,
                            scale=self.y_sd * self.y_train_sd
                        )

                        self.err_dist_summary_standardized = Normal(
                            loc=0.,
                            scale=self.y_sd_summary
                        )
                        self.err_dist_summary = Normal(
                            loc=0.,
                            scale=self.y_sd_summary * self.y_train_sd
                        )

                    self.ll_standardized = y_dist_standardized.log_prob(y_standardized)
                    self.ll = y_dist.log_prob(self.y)
                    ll_objective = self.ll_standardized
                    # ll_objective = tf.Print(ll_objective, [self.y_sd, ll_objective], summarize=10)
                else:
                    if self.asymmetric_error:
                        y_dist = SinhArcsinh(
                            loc=self.out,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist_summary = SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                    else:
                        y_dist = Normal(
                            loc=self.out,
                            scale=self.y_sd
                        )
                        self.err_dist = Normal(
                            loc=0.,
                            scale=self.y_sd
                        )
                        self.err_dist_summary = Normal(
                            loc=0.,
                            scale=self.y_sd_summary
                        )
                    self.ll = y_dist.log_prob(self.y)
                    ll_objective = self.ll

                self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support[None,...]))
                self.err_dist_plot_summary = tf.exp(self.err_dist_summary.log_prob(self.support[None,...]))
                self.err_dist_lb = self.err_dist_summary.quantile(.025)
                self.err_dist_ub = self.err_dist_summary.quantile(.975)

                empirical_quantiles = tf.linspace(0., 1., self.n_errors)
                if self.standardize_response:
                    self.err_dist_standardized_theoretical_quantiles = self.err_dist_standardized.quantile(empirical_quantiles)
                    self.err_dist_standardized_theoretical_cdf = self.err_dist_standardized.cdf(self.errors)
                    self.err_dist_standardized_summary_theoretical_quantiles = self.err_dist_summary_standardized.quantile(empirical_quantiles)
                    self.err_dist_standardized_summary_theoretical_cdf = self.err_dist_summary_standardized.cdf(self.errors)
                self.err_dist_theoretical_quantiles = self.err_dist.quantile(empirical_quantiles)
                self.err_dist_theoretical_cdf = self.err_dist.cdf(self.errors)
                self.err_dist_summary_theoretical_quantiles = self.err_dist_summary.quantile(empirical_quantiles)
                self.err_dist_summary_theoretical_cdf = self.err_dist_summary.cdf(self.errors)

                loss_func = - ll_objective

                if self.loss_filter_n_sds and self.ema_decay:
                    beta = self.ema_decay
                    ema_warm_up = int(2/(1 - self.ema_decay))
                    n_sds = self.loss_filter_n_sds

                    self.loss_ema = tf.Variable(0., trainable=False, name='loss_ema')
                    self.loss_sd_ema = tf.Variable(0., trainable=False, name='loss_sd_ema')

                    loss_cutoff = self.loss_ema + n_sds * self.loss_sd_ema
                    loss_func_filter = tf.cast(loss_func < loss_cutoff, dtype=self.FLOAT_TF)
                    loss_func_filtered = loss_func * loss_func_filter
                    n_batch = tf.cast(tf.shape(loss_func)[0], dtype=self.FLOAT_TF)
                    n_retained = tf.reduce_sum(loss_func_filter)
                    self.n_dropped = n_batch - n_retained

                    loss_func, n_retained = tf.cond(
                        self.global_batch_step > ema_warm_up,
                        lambda loss_func_filtered=loss_func_filtered, n_retained=n_retained: (loss_func_filtered, n_retained),
                        lambda loss_func=loss_func: (loss_func, n_batch),
                    )

                    # loss_func = tf.Print(loss_func, ['cutoff', loss_cutoff, 'n_retained', n_retained, 'ema', self.loss_ema, 'sd ema', self.loss_sd_ema])

                    loss_mean_cur = tf.reduce_sum(loss_func) / (n_retained + self.epsilon)
                    loss_sd_cur = tf.sqrt(tf.reduce_sum((loss_func - self.loss_ema)**2)) / (n_retained + self.epsilon)

                    loss_ema_update = (beta * self.loss_ema + (1 - beta) * loss_mean_cur)
                    loss_sd_ema_update = beta * self.loss_sd_ema + (1 - beta) * loss_sd_cur

                    self.loss_ema_op = tf.assign(self.loss_ema, loss_ema_update)
                    self.loss_sd_ema_op = tf.assign(self.loss_sd_ema, loss_sd_ema_update)

                if self.scale_loss_with_data:
                    self.loss_func = tf.reduce_sum(loss_func) * self.minibatch_scale
                else:
                    self.loss_func = tf.reduce_mean(loss_func)

                for l in self.regularizable_layers:
                    if hasattr(l, 'weights'):
                        vars = l.weights
                    else:
                        vars = [l]
                    for v in vars:
                        if 'bias' not in v.name:
                            self._regularize(v, type='nn', var_name=reg_name(v.name))

                self.reg_loss = tf.constant(0., dtype=self.FLOAT_TF)
                if len(self.regularizer_losses_varnames) > 0:
                    self.reg_loss += tf.add_n(self.regularizer_losses)
                    self.loss_func += self.reg_loss

                self.optim = self._initialize_optimizer()
                assert self.optim_name is not None, 'An optimizer name must be supplied'

                self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step)


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

    def run_predict_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                preds = self.sess.run(self.out, feed_dict=feed_dict)
                if self.standardize_response and not standardize_response:
                    preds = preds * self.y_train_sd + self.y_train_mean
                return preds

    def run_loglik_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.standardize_response and standardize_response:
                    ll = self.ll_standardized
                else:
                    ll = self.ll
                log_lik = self.sess.run(ll, feed_dict=feed_dict)
                return log_lik

    def run_loss_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss = self.sess.run(self.loss_func, feed_dict=feed_dict)

                return loss

