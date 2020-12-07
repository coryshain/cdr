import math
import pandas as pd

from .kwargs import CDRNNMLE_INITIALIZATION_KWARGS
from .backend import get_initializer, DenseLayer, DenseLayer, CDRNNLayer
from .cdrnnbase import CDRNN
from .util import sn, reg_name, stderr

import tensorflow as tf

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
                    intercept_summary = intercept
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    intercept = tf.Variable(
                        tf.zeros([rangf_n_levels], dtype=self.FLOAT_TF),
                        name='intercept_by_%s' % sn(ran_gf)
                    )
                    intercept_summary = intercept
                return intercept, intercept_summary

    def initialize_feedforward(
            self,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            batch_normalization_decay=None,
            name=None
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                projection = DenseLayer(
                    training=self.training,
                    units=units,
                    use_bias=use_bias,
                    activation=activation,
                    dropout=dropout,
                    batch_normalization_decay=batch_normalization_decay,
                    kernel_sd_init=self.kernel_sd_init,
                    epsilon=self.epsilon,
                    session=self.sess,
                    name=name
                )

                return projection

    def initialize_rnn(self, l):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                rnn = CDRNNLayer(
                    training=self.training,
                    units=units,
                    time_projection_depth=self.n_layers_irf + 1,
                    activation=self.rnn_activation,
                    recurrent_activation=self.recurrent_activation,
                    time_projection_inner_activation=self.irf_inner_activation,
                    bottomup_kernel_sd_init=self.kernel_sd_init,
                    recurrent_kernel_sd_init=self.kernel_sd_init,
                    bottomup_dropout=self.input_projection_dropout_rate,
                    h_dropout=self.rnn_h_dropout_rate,
                    c_dropout=self.rnn_c_dropout_rate,
                    forget_rate=self.forget_rate,
                    return_sequences=True,
                    batch_normalization_decay=None,
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

                return rnn_h

    def initialize_rnn_c(self, l, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                if ran_gf is None:
                    rnn_h = tf.Variable(tf.zeros([1, units]), name='rnn_c_l%d' % (l+1))
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    rnn_h = tf.Variable(
                        tf.zeros([rangf_n_levels, units]),
                        name='rnn_c_ran_l%d_by_%s' % (l+1, sn(ran_gf))
                    )

                return rnn_h

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

                return h_bias

    def initialize_irf_l1_biases(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if isinstance(self.kernel_sd_init, str):
                    if self.kernel_sd_init.lower() in ['xavier', 'glorot']:
                        sd = math.sqrt(2 / (1 + self.n_units_t_delta_embedding))
                    elif self.kernel_sd_init.lower() == 'he':
                        sd = math.sqrt(2)
                else:
                    sd = self.kernel_sd_init

                kernel_init = get_initializer(
                    'random_normal_initializer_mean=0-stddev=%s' % sd,
                    session=self.sess
                )

                irf_l1_W_bias = tf.get_variable(
                    name='irf_l1_W_bias',
                    initializer=kernel_init,
                    shape=[1, 1, self.n_units_t_delta_embedding]
                )

                irf_l1_b_bias = tf.get_variable(
                    name='irf_l1_b_bias',
                    initializer=tf.zeros_initializer(),
                    shape=[1, 1, self.n_units_t_delta_embedding]
                )

                return irf_l1_W_bias, irf_l1_b_bias

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
                        y_dist_standardized = tf.contrib.distributions.SinhArcsinh(
                            loc=self.out,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        y_dist = tf.contrib.distributions.SinhArcsinh(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=self.y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )

                        self.err_dist_standardized = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )

                        self.err_dist_summary_standardized = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                        self.err_dist_summary = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary * self.y_train_sd,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                    else:
                        y_dist_standardized = tf.distributions.Normal(
                            loc=self.out,
                            scale=self.y_sd
                        )
                        y_dist = tf.distributions.Normal(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=self.y_sd * self.y_train_sd
                        )

                        self.err_dist_standardized = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd
                        )
                        self.err_dist = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd * self.y_train_sd
                        )

                        self.err_dist_summary_standardized = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd_summary
                        )
                        self.err_dist_summary = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd_summary * self.y_train_sd
                        )

                    self.ll_standardized = y_dist_standardized.log_prob(y_standardized)
                    self.ll = y_dist.log_prob(self.y)
                    ll_objective = self.ll_standardized
                    # ll_objective = tf.Print(ll_objective, [self.y_sd, ll_objective], summarize=10)
                else:
                    if self.asymmetric_error:
                        y_dist = tf.contrib.distributions.SinhArcsinh(
                            loc=self.out,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.y_tailweight
                        )
                        self.err_dist_summary = tf.contrib.distributions.SinhArcsinh(
                            loc=0.,
                            scale=self.y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.y_tailweight_summary
                        )
                    else:
                        y_dist = tf.distributions.Normal(
                            loc=self.out,
                            scale=self.y_sd
                        )
                        self.err_dist = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd
                        )
                        self.err_dist_summary = tf.distributions.Normal(
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

                self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
                self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)

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

    def run_train_step(self, feed_dict, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                to_run_names = []
                to_run = [self.train_op, self.ema_op, self.y_sd_delta_ema_op]
                if self.n_layers_rnn:
                    to_run += self.rnn_h_ema_ops + self.rnn_c_ema_ops
                if self.asymmetric_error:
                    to_run += [self.y_skewness_delta_ema_op, self.y_tailweight_delta_ema_op]
                if self.loss_filter_n_sds:
                    to_run_names.append('n_dropped')
                    to_run += [self.loss_ema_op, self.loss_sd_ema_op, self.n_dropped]
                to_run_names += ['loss', 'reg_loss']
                to_run += [self.loss_func, self.reg_loss]
                out = self.sess.run(to_run, feed_dict=feed_dict)

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict

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

