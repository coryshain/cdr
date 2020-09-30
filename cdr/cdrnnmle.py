import pandas as pd

from .kwargs import CDRNNMLE_INITIALIZATION_KWARGS
from .cdrnnbase import CDRNN
from .util import sn, stderr

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

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
    #  MLE IMPLEMENTATION OF CDRNN
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
                        # tf.random_normal(
                        #     shape=[rangf_n_levels],
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.zeros([rangf_n_levels], dtype=self.FLOAT_TF),
                        name='intercept_by_%s' % sn(ran_gf)
                    )
                    intercept_summary = intercept
                return intercept, intercept_summary

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.y_sd_base_unconstrained = tf.Variable(
                    self.y_sd_init_unconstrained,
                    dtype=self.FLOAT_TF,
                    name='y_sd_base_unconstrained'
                )
                self.y_sd = self.constraint_fn(self.y_sd_base_unconstrained + self.y_sd_delta) + self.epsilon
                self.y_sd_summary = self.constraint_fn(self.y_sd_base_unconstrained + self.y_sd_delta_ema)
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
                    self.y_tailweight = self.constraint_fn(self.y_tailweight_base_unconstrained + self.y_tailweight_delta + self.epsilon)
                    self.y_tailweight_summary = self.constraint_fn(self.y_tailweight_base_unconstrained + self.y_tailweight_delta_ema)
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

                self.loss_func = -(tf.reduce_sum(ll_objective) * self.minibatch_scale)

                for l in self.regularizable_layers:
                    if hasattr(l, 'weights'):
                        vars = l.weights
                    else:
                        vars = [l]
                    for v in vars:
                        name = v.name.split(':')[0]
                        name = name.replace('/', '_')
                        cap = True
                        var_name = ''
                        for c in name:
                            if c == '_':
                                cap = True
                            else:
                                if cap:
                                    var_name += c.upper()
                                else:
                                    var_name += c
                                cap = False
                        self._regularize(v, type='nn', var_name=var_name)

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
                to_run = [self.train_op, self.ema_op, self.y_sd_delta_ema_op]
                if self.n_layers_rnn:
                    to_run += self.rnn_h_ema_ops + self.rnn_c_ema_ops
                if self.asymmetric_error:
                    to_run += [self.y_skewness_delta_ema_op, self.y_tailweight_delta_ema_op]
                to_run += [self.loss_func, self.reg_loss]
                out = self.sess.run(to_run, feed_dict=feed_dict)
                loss, reg_loss = out[-2:]

                out_dict = {
                    'loss': loss,
                    'reg_loss': reg_loss
                }

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

