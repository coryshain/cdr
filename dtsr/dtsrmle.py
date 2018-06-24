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
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.type, x.descr]) + ' **Default**: ``%s``.' %(x.default_value if not isinstance(x.default_value, str) else "'%s'" %x.default_value) for x in _INITIALIZATION_KWARGS])
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

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0., ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    param = tf.Variable(
                        tf.random_normal(
                            shape=[1, len(ids)],
                            mean=mean,
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name=sn('%s_%s' % (param_name, '-'.join(ids)))
                    )
                    self._regularize(param, mean)
                    param_summary = param
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                    param = tf.Variable(
                        tf.random_normal(
                            shape=[rangf_n_levels, len(ids)],
                            mean=0.,
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='%s_by_%s' % (param_name, ran_gf)
                    )
                    self._regularize(param)
                    param_summary = param

                return (param, param_summary)

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