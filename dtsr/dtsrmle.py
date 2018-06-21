import sys
import os
import numpy as np
import pandas as pd

from .dtsr import Kwarg, DTSR
from .util import names2ix, sn

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None



######################################################
#
#  MLE IMPLEMENTATION OF DTSR
#
######################################################

DTSRMLE_INITIALIZATION_KWARGS = [
    Kwarg(
        'loss_type',
        'mse',
        "``str``",
        "The optimization objective."
    )
]


class DTSRMLE(DTSR):
    _INITIALIZATION_KWARGS = DTSRMLE_INITIALIZATION_KWARGS

    _doc_header = """
        A DTSR implementation fitted using maximum likelihood estimation.
    """
    _doc_args = DTSR._doc_args
    _doc_kwargs = DTSR._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join(
        [' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.type, x.descr]) + ' **Default**: ``%s``.' % x.default_value
         for x in _INITIALIZATION_KWARGS])
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

        super(DTSRMLE, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in DTSR._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' % kwarg_key)

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(DTSRMLE, self)._initialize_metadata()

    def _pack_metadata(self):
        md = super(DTSRMLE, self)._pack_metadata()
        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        super(DTSRMLE, self)._unpack_metadata(md)

        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            sys.stderr.write(
                'Saved model contained unrecognized attributes %s which are being ignored\n' % sorted(list(md.keys())))

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
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                    intercept = tf.Variable(
                        tf.random_normal(
                            shape=[rangf_n_levels],
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='intercept_by_%s' % ran_gf
                    )
                    self._regularize(intercept)
                    intercept_summary = intercept
                return intercept, intercept_summary

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    coefficient = tf.Variable(
                        tf.random_normal(
                            shape=[len(coef_ids)],
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='coefficient'
                    )
                    coefficient_summary = coefficient
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                    coefficient = tf.Variable(
                        tf.random_normal(
                            shape=[rangf_n_levels, len(coef_ids)],
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='coefficient_by_%s' % ran_gf
                    )
                    coefficient_summary = coefficient
                self._regularize(coefficient)
                return coefficient, coefficient_summary

    def initialize_irf_param(self, param_name, ids, trainable=None, mean=0, lb=None, ub=None, irf_by_rangf=None):
        if irf_by_rangf is None:
            irf_by_rangf = []

        with self.sess.as_default():
            with self.sess.graph.as_default():
                param_mean_init, lb, ub = self._process_mean(mean, lb, ub)

                if trainable is None:
                    trainable_ix = np.array(list(range(len(ids))), dtype=self.INT_NP)
                    untrainable_ix = []
                else:
                    trainable_ix = []
                    untrainable_ix = []
                    for i in range(len(ids)):
                        name = ids[i]
                        if param_name in trainable[name]:
                            trainable_ix.append(i)
                        else:
                            untrainable_ix.append(i)
                    trainable_ix = np.array(trainable_ix, dtype=self.INT_NP)
                    untrainable_ix = np.array(untrainable_ix, dtype=self.INT_NP)

                dim = len(trainable_ix)

                # Initialize trainable IRF parameters as trainable variables

                param = tf.Variable(
                    tf.random_normal(
                        shape=[1, dim],
                        mean=tf.expand_dims(tf.gather(param_mean_init, trainable_ix), 0),
                        stddev=self.init_sd,
                        dtype=self.FLOAT_TF
                    ),
                    name=sn('%s_%s' % (param_name, '-'.join(ids)))
                )
                self._regularize(param, param_mean_init)

                if lb is None and ub is None:
                    param_out = param
                elif lb is not None and ub is None:
                    param_out = tf.nn.softplus(param) + lb + self.epsilon
                elif lb is None and ub is not None:
                    param_out = -tf.nn.softplus(param) + ub - self.epsilon
                else:
                    param_out = tf.sigmoid(param) * ((ub - self.epsilon) - (lb + self.epsilon)) + lb + self.epsilon

                for i in range(dim):
                    tf.summary.scalar(
                        sn('%s/%s' % (param_name, ids[i])),
                        param_out[0, i],
                        collections=['params']
                    )

                param_summary = param_out

                # Initialize untrainable IRF parameters as constants

                param_untrainable = tf.expand_dims(tf.gather(param_mean_init, untrainable_ix), 0)

                if lb is None and ub is None:
                    param_untrainable_out = param_untrainable
                elif lb is not None and ub is None:
                    param_untrainable_out = tf.nn.softplus(param_untrainable) + lb + self.epsilon
                elif lb is None and ub is not None:
                    param_untrainable_out = -tf.nn.softplus(param_untrainable) + ub - self.epsilon
                else:
                    param_untrainable_out = tf.sigmoid(param_untrainable) * (
                            (ub - self.epsilon) - (lb + self.epsilon)) + lb + self.epsilon

                # Process any random IRF parameters

                if len(irf_by_rangf) > 0:
                    for gf in irf_by_rangf:
                        i = self.rangf.index(gf)
                        mask_row_np = np.ones(self.rangf_n_levels[i], dtype=getattr(np, self.float_type))
                        mask_row_np[self.rangf_n_levels[i] - 1] = 0
                        mask_row = tf.constant(mask_row_np, dtype=self.FLOAT_TF)
                        col_ix = names2ix(irf_by_rangf[gf], ids)
                        mask_col_np = np.zeros([1, dim])
                        mask_col_np[0, col_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)

                        param_ran = tf.Variable(
                            tf.random_normal(
                                shape=[self.rangf_n_levels[i], dim],
                                mean=0.,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='%s_by_%s' % (param_name, gf)
                        )
                        self._regularize(param_ran)

                        param_ran *= mask_col
                        param_ran *= tf.expand_dims(mask_row, -1)

                        param_ran_mean = tf.reduce_sum(param_ran, axis=0) / tf.reduce_sum(mask_row)
                        param_ran_centering_vector = tf.expand_dims(mask_row, -1) * param_ran_mean
                        param_ran -= param_ran_centering_vector

                        half_interval = None
                        if lb is not None:
                            half_interval = param_out - lb + self.epsilon
                        elif ub is not None:
                            if half_interval is not None:
                                half_interval = tf.minimum(half_interval, ub - self.epsilon - param_out)
                            else:
                                half_interval = ub - self.epsilon - param_out
                        if half_interval is not None:
                            param_ran = tf.tanh(param_ran) * half_interval

                        param_out += tf.gather(param_ran, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(irf_by_rangf[gf])):
                                irf_name = irf_by_rangf[gf][j]
                                ix = col_ix[j]
                                tf.summary.histogram(
                                    'by_%s/%s/%s' % (gf, param_name, irf_name),
                                    param_ran[:, ix],
                                    collections=['random']
                                )

                # Combine trainable and untrainable parameters
                if len(untrainable_ix) > 0:
                    param_out = tf.concat([param_out, param_untrainable_out], axis=1)
                    param_summary = tf.concat([param_summary, param_untrainable_out], axis=1)

                param_out = tf.gather(param_out, np.concatenate([trainable_ix, untrainable_ix]), axis=1)
                param_summary = tf.gather(param_summary, np.concatenate([trainable_ix, untrainable_ix]), axis=1)

                # Since DTSRMLE just learns point estimates, we simply use those
                # estimates for plotting in the 2nd return value
                return (param_out, param_summary)

    def initialize_objective(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
                self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
                if self.loss_type.lower() == 'mae':
                    self.loss_func = self.mae_loss
                else:
                    self.loss_func = self.mse_loss
                if self.regularizer_name is not None:
                    self.loss_func += tf.add_n(self.regularizer_losses)

                self.optim = self._initialize_optimizer(self.optim_name)
                assert self.optim_name is not None, 'An optimizer name must be supplied'

                # self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step,
                #                                     name=sn('optim'))
                # self.gradients = self.optim.compute_gradients(self.loss_func)

                self.gradients, variables = zip(*self.optim.compute_gradients(self.loss_func))
                # ## CLIP GRADIENT NORM
                # self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
                self.train_op = self.optim.apply_gradients(
                    zip(self.gradients, variables),
                    global_step=self.global_batch_step,
                    name=sn('optim')
                )

                ## Likelihood ops
                self.y_scale = tf.Variable(self.y_train_sd, dtype=self.FLOAT_TF)
                self.set_y_scale = tf.assign(self.y_scale, self.loss_total)
                s = self.y_scale
                y_dist = tf.distributions.Normal(loc=self.out, scale=self.y_scale)
                self.ll = y_dist.log_prob(self.y)

    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def run_train_step(self, feed_dict):
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

    def run_predict_op(self, feed_dict, n_samples=None):
        if n_samples is not None:
            sys.stderr.write('Parameter n_samples is irrelevant to predict() from a DTSRMLE model and will be ignored')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                preds = self.sess.run(self.out, feed_dict=feed_dict)
                return preds

    def run_loglik_op(self, feed_dict, n_samples=None):
        if n_samples is not None:
            sys.stderr.write('Parameter n_samples is irrelevant to log_lik() from a DTSRMLE model and will be ignored')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                log_lik = self.sess.run(self.ll, feed_dict=feed_dict)
                return log_lik

    def run_conv_op(self, feed_dict, scaled=False, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                X_conv = self.sess.run(self.X_conv_scaled if scaled else self.X_conv, feed_dict=feed_dict)
                return X_conv

    def summary(self, fixed=True, random=False):
        summary = '=' * 50 + '\n'
        summary += 'DTSR regression\n\n'
        summary += 'Output directory: %s\n\n' % self.outdir
        summary += 'Formula:\n'
        summary += '  ' + self.form_str + '\n\n'

        # TODO: Fill this in

        return (summary)