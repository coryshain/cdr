import sys
import os
import numpy as np
import pandas as pd

from .cdrbase import CDR
from .kwargs import CDRMLE_INITIALIZATION_KWARGS
from .util import sn, stderr

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None



######################################################
#
#  MLE IMPLEMENTATION OF CDR
#
######################################################


class CDRMLE(CDR):
    _INITIALIZATION_KWARGS = CDRMLE_INITIALIZATION_KWARGS

    _doc_header = """
        A CDR implementation fitted using maximum likelihood estimation.
    """
    _doc_args = CDR._doc_args
    _doc_kwargs = CDR._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    ######################################################
    #
    #  Initialization methods
    #
    ######################################################

    def __init__(
            self,
            form_str,
            X,
            y,
            **kwargs
    ):

        super(CDRMLE, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(CDRMLE, self)._initialize_metadata()

        if self.intercept_init is None:
            self.intercept_init = self.y_train_mean
        if self.intercept_joint_sd is None:
            self.intercept_joint_sd = self.y_train_sd * self.joint_sd_scaling_coefficient
        if self.coef_joint_sd is None:
            self.coef_joint_sd = self.y_train_sd * self.joint_sd_scaling_coefficient

    def _pack_metadata(self):
        md = super(CDRMLE, self)._pack_metadata()
        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        super(CDRMLE, self)._unpack_metadata(md)

        for kwarg in CDRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            stderr('Saved model contained unrecognized attributes %s which are being ignored\n' % sorted(list(md.keys())))

    ######################################################
    #
    #  Network Initialization
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

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    coefficient = tf.Variable(
                        # tf.random_normal(
                        #     shape=[len(coef_ids)],
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.zeros([len(coef_ids)], dtype=self.FLOAT_TF),
                        name='coefficient'
                    )
                    coefficient_summary = coefficient
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    coefficient = tf.Variable(
                        # tf.random_normal(
                        #     shape=[rangf_n_levels, len(coef_ids)],
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.zeros([rangf_n_levels, len(coef_ids)], dtype=self.FLOAT_TF),
                        name='coefficient_by_%s' % sn(ran_gf)
                    )
                    coefficient_summary = coefficient
                return coefficient, coefficient_summary

    def initialize_interaction(self, interaction_ids=None, ran_gf=None):
        if interaction_ids is None:
            interaction_ids = self.interaction_names

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    interaction = tf.Variable(
                        # tf.random_normal(
                        #     shape=[len(interaction_ids)],
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.zeros([len(interaction_ids)], dtype=self.FLOAT_TF),
                        name='interaction'
                    )
                    interaction_summary = interaction
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    interaction = tf.Variable(
                        # tf.random_normal(
                        #     shape=[rangf_n_levels, len(interaction_ids)],
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.zeros([rangf_n_levels, len(interaction_ids)], dtype=self.FLOAT_TF),
                        name='coefficient_by_%s' % sn(ran_gf)
                    )
                    interaction_summary = interaction
                return interaction, interaction_summary

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0., ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    param = tf.Variable(
                        # tf.random_normal(
                        #     shape=[1, len(ids)],
                        #     mean=mean,
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.ones([1, len(ids)], dtype=self.FLOAT_TF) * mean,
                        name=sn('%s_%s' % (param_name, '-'.join(ids)))
                    )
                    param_summary = param
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    param = tf.Variable(
                        # tf.random_normal(
                        #     shape=[rangf_n_levels, len(ids)],
                        #     mean=0.,
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.random_normal(
                            shape=[rangf_n_levels, len(ids)],
                            mean=0.,
                            stddev=self.epsilon,
                            dtype=self.FLOAT_TF
                        ),
                        # tf.zeros([rangf_n_levels, len(ids)], dtype=self.FLOAT_TF),
                        name=sn('%s_%s_by_%s' % (param_name, '-'.join(ids), sn(ran_gf)))
                    )
                    param_summary = param

                return param, param_summary

    def initialize_joint_distribution(self, means, sds, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                dim = int(means.shape[0])

                joint_loc = tf.Variable(
                    # tf.random_normal(
                    #     [dim],
                    #     mean=means,
                    #     stddev=self.init_sd,
                    #     dtype=self.FLOAT_TF
                    # ),
                    tf.ones([dim], dtype=self.FLOAT_TF) * means,
                    name='joint_loc' if ran_gf is None else 'joint_loc_by_%s' % sn(ran_gf)
                )

                # Construct cholesky decomposition of initial covariance using sds, then use for initialization
                n_scale = int(dim * (dim + 1) / 2)
                if ran_gf is not None:
                    sds *= self.ranef_to_fixef_joint_sd_ratio
                cholesky = tf.diag(sds)
                tril_ix = np.ravel_multi_index(
                    np.tril_indices(dim),
                    (dim, dim)
                )
                scale_init = tf.gather(tf.reshape(cholesky, [dim * dim]), tril_ix)

                joint_scale = tf.Variable(
                    # tf.random_normal(
                    #     [n_scale],
                    #     mean=scale_init,
                    #     stddev=self.init_sd,
                    #     dtype=self.FLOAT_TF
                    # ),
                    tf.ones([n_scale], dtype=self.FLOAT_TF) * scale_init,
                    name='joint_scale' if ran_gf is None else 'joint_scale_by_%s' % sn(ran_gf)
                )

                joint_dist = tf.contrib.distributions.MultivariateNormalTriL(
                    loc=joint_loc,
                    scale_tril=tf.contrib.distributions.fill_triangular(joint_scale),
                    name='joint' if ran_gf is None else 'joint_by_%s' % sn(ran_gf)
                )

                joint = tf.reduce_mean(joint_dist.sample(sample_shape=[100]), axis=0)
                joint_summary = joint_dist.mean()

                return joint, joint_summary

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.y_sd = self.constraint_fn(
                    tf.Variable(
                        self.y_sd_init_unconstrained,
                        dtype=self.FLOAT_TF,
                        name='y_sd'
                    )
                )
                self.y_sd_summary = self.y_sd
                tf.summary.scalar(
                    'error/y_sd',
                    self.y_sd_summary,
                    collections=['params']
                )

                if self.asymmetric_error:
                    self.y_skewness = tf.Variable(
                        0.,
                        dtype=self.FLOAT_TF,
                        name='y_skewness'
                    )
                    self.y_skewness_summary = self.y_skewness
                    tf.summary.scalar(
                        'error/y_skewness',
                        self.y_skewness_summary,
                        collections=['params']
                    )

                    self.y_tailweight = self.constraint_fn(
                        tf.Variable(
                            self.y_tailweight_init_unconstrained,
                            dtype=self.FLOAT_TF,
                            name='y_skewness'
                        )
                    )
                    self.y_tailweight_summary = self.y_tailweight
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
                    self.ll_standardized = y_dist_standardized.log_prob(y_standardized)
                    self.ll = y_dist.log_prob(self.y)
                    ll_objective = self.ll_standardized
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
                    else:
                        y_dist = tf.distributions.Normal(
                            loc=self.out,
                            scale=self.y_sd
                        )
                        self.err_dist = tf.distributions.Normal(
                            loc=0.,
                            scale=self.y_sd
                        )
                    self.ll = y_dist.log_prob(self.y)
                    ll_objective = self.ll

                self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support[None,...]))
                self.err_dist_plot_summary = self.err_dist_plot
                self.err_dist_lb = self.err_dist.quantile(.025)
                self.err_dist_ub = self.err_dist.quantile(.975)

                empirical_quantiles = tf.linspace(0., 1., self.n_errors)
                if self.standardize_response:
                    self.err_dist_standardized_theoretical_quantiles = self.err_dist_standardized.quantile(empirical_quantiles)
                    self.err_dist_standardized_theoretical_cdf = self.err_dist_standardized.cdf(self.errors)
                    self.err_dist_standardized_summary_theoretical_quantiles = self.err_dist_standardized_theoretical_quantiles
                    self.err_dist_standardized_summary_theoretical_cdf = self.err_dist_standardized_theoretical_cdf
                self.err_dist_theoretical_quantiles = self.err_dist.quantile(empirical_quantiles)
                self.err_dist_theoretical_cdf = self.err_dist.cdf(self.errors)
                self.err_dist_summary_theoretical_quantiles = self.err_dist_theoretical_quantiles
                self.err_dist_summary_theoretical_cdf = self.err_dist_theoretical_cdf

                self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
                self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)

                self.loss_func = -(tf.reduce_sum(ll_objective) * self.minibatch_scale)
                self.reg_loss = 0.
                if len(self.regularizer_losses_varnames) > 0:
                    self.reg_loss += tf.add_n(self.regularizer_losses)
                    self.loss_func += self.reg_loss

                self.optim = self._initialize_optimizer(self.optim_name)
                assert self.optim_name is not None, 'An optimizer name must be supplied'

                self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step)



    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def report_settings(self, indent=0):
        out = super(CDRMLE, self).report_settings(indent=indent)
        for kwarg in CDRMLE_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out

    def run_train_step(self, feed_dict, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                _, _, loss = self.sess.run(
                    [self.train_op, self.ema_op, self.loss_func],
                    feed_dict=feed_dict
                )

                out_dict = {
                    'loss': loss
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

    def run_conv_op(self, feed_dict, scaled=False, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                X_conv = self.sess.run(self.X_conv_scaled if scaled else self.X_conv, feed_dict=feed_dict)
                if scaled and self.standardize_response and not standardize_response:
                    X_conv = X_conv * self.y_train_sd
                return X_conv

    def extract_irf_integral(self, terminal_name, rangf=None, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if n_time_units is None:
                    n_time_units = self.t_delta_limit

                if self.pc:
                    n_impulse = len(self.src_impulse_names)
                else:
                    n_impulse = len(self.impulse_names)

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: n_time_units,
                    self.n_time_points: n_time_points,
                    self.time_y: [n_time_units],
                    self.time_X: np.zeros((1, self.history_length, n_impulse))
                }

                if rangf is not None:
                    fd[self.gf_y] = rangf

                if terminal_name in self.irf_integral_tensors:
                    irf_integral = self.irf_integral_tensors[terminal_name]
                else:
                    irf_integral = self.src_irf_integral_tensors[terminal_name]

                out = self.sess.run(irf_integral, feed_dict=fd)

                return (out,)
