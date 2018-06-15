import os
import time
from collections import defaultdict

import pandas as pd
pd.options.mode.chained_assignment = None

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

from .formula import *
from .util import *
from .dtsr import DTSR
from .data import build_DTSR_impulses, corr_dtsr

class DTSRMLE(DTSR):
    """
    A neural net implementation of DTSR.

    :param form_str: An R-style string representing the DTSR model formula.
    :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:

        * ``time``: Timestamp associated with each observation in ``y``
        * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each observation in ``y``
        * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each observation in ``y``
        * A column with the same name as the DV specified in ``form_str``
        * A column for each random grouping factor in the model specified in ``form_str``

    :param outdir: ``str``; the output directory, where logs and model parameters are saved.
    :param history_length: ``int`` or ``None``; the maximum length of the history window to use (unbounded if ``None``, which requires ``low_memory=True``).
    :param low_memory: ``bool``; whether to use the ``low_memory`` network structure.
        If ``True``, DTSR convolves over history windows for each observation in ``y`` using a TensorFlow control op.
        It can be used with unboundedly long histories and requires less memory, but results in poor GPU utilization.
        If ``False``, DTSR expands the design matrix into a rank 3 tensor in which the 2nd axis contains the history for each independent variable for each observation of the dependent variable.
        This requires more memory in order to store redundant input values and requires a finite history length.
        However, it removes the need for a control op in the feedforward component and therefore generally runs much faster if GPU is available.
        *NOTE*: Support for ``low_memory`` (and consequently unbounded history length) has been temporarily suspended, since the high-memory implementation is generally much faster and for many projects model fit is not greatly affected by history truncation. If your project demands unbounded history modeling, please let the developers know on Github so that they can prioritize re-implementing this feature.
    :param float_type: ``str``; the ``float`` type to use throughout the network.
    :param int_type: ``str``; the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: ``int`` or ``None``; the size of minibatches to use for fitting/prediction (full-batch if ``None``).
    :param n_interp: ``int``; number of interpolation points (ignored unless the model formula specification contains continuous inputs).
    :param log_random: ``bool``; whether to log random effects to Tensorboard.
    :param log_freq: ``int``; the frequency (in iterations) with which to log model params to Tensorboard.
    :param save_freq: ``int``; the frequency (in iterations) with which to save model checkpoints.
    :param optim: ``str``; the name of the optimizer to use. Choose from ``'SGD'``, ``'AdaGrad'``, ``'AdaDelta'``, ``'Adam'``, ``'FTRL'``, ``'RMSProp'``, ``'Nadam'``.
    :param learning_rate: ``float``; the initial value for the learning rate.
    :param learning_rate_decay_factor: ``float``; rate parameter to the learning rate decay schedule (if applicable).
    :param learning_rate_decay_family: ``str``; the functional family for the learning rate decay schedule (if applicable).
        Choose from the following, where :math:`\lambda` is the current learning rate, :math:`\lambda_0` is the initial learning rate, :math:`\delta` is the ``learning_rate_decay_factor``, and :math:`i` is the iteration index.

        * ``'linear'``: :math:`\\lambda_0 \\cdot ( 1 - \\delta \\cdot i )`
        * ``'inverse'``: :math:`\\frac{\\lambda_0}{1 + ( \\delta \\cdot i )}`
        * ``'exponential'``: :math:`\\lambda = \\lambda_0 \\cdot ( 2^{-\\delta \\cdot i} )`
        * ``'stepdownXX'``: where ``XX`` is replaced by an integer representing the stepdown interval :math:`a`: :math:`\\lambda = \\lambda_0 * 2^{\\left \\lfloor \\frac{i}{a} \\right \\rfloor}`

    :param learning_rate_min: ``float``; the minimum value for the learning rate.
        If the decay schedule would take the learning rate below this point, learning rate clipping will occur.
    :param init_sd: ``float``; standard deviation of truncated normal parameter initializer
    :param ema_decay: ``float``; decay factor to use for exponential moving average for parameters (used in prediction)
    :param loss: ``str``; the optimization objective.
        Currently only ``'mse'`` and ``'mae'`` are supported.
    :param regularizer: ``str``; name of regularizer to use (e.g. ``l1``, ``l2``), or if ``None``, no regularization
    :param regularizer: ``float``; scale constant to use for regularization
    :param log_graph: ``bool``; whether to log the network graph to Tensorboard
    """



    ######################################################
    #
    #  Native methods
    #
    ######################################################

    _INITIALIZATION_KWARGS = {
        'loss_type': 'mse'
    }


    def __init__(
            self,
            form_str,
            X,
            y,
            outdir='./dtsr_model/',
            **kwargs
    ):

        super(DTSRMLE, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg, kwargs.pop(kwarg, DTSRMLE._INITIALIZATION_KWARGS[kwarg]))

        for kwarg in kwargs:
            if kwarg not in DTSR._INITIALIZATION_KWARGS:
                raise TypeError('__init__() got an unexpected keyword argument %s' %kwarg)

        self._initialize_metadata()

        self.build(outdir)

    def _initialize_metadata(self):
        super(DTSRMLE, self)._initialize_metadata()

        self.network_type = 'mle'

    def _pack_metadata(self):
        md = super(DTSRMLE, self)._pack_metadata()
        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            md[kwarg] = getattr(self, kwarg)
        return md

    def _unpack_metadata(self, md):
        super(DTSRMLE, self)._unpack_metadata(md)

        for kwarg in DTSRMLE._INITIALIZATION_KWARGS:
            setattr(self, kwarg, md.pop(kwarg, DTSRMLE._INITIALIZATION_KWARGS[kwarg]))

        if len(md) > 0:
            sys.stderr.write('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_intercept(self, ran_gf=None, rangf_n_levels=None):
        f = self.form

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

    def _initialize_coefficient(self, coef_ids=None, ran_gf=None, rangf_n_levels=None):
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

    def _initialize_irf_param(self, param_name, ids, trainable=None, mean=0, lb=None, ub=None, irf_by_rangf=None):
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
                    param_untrainable_out = tf.sigmoid(param_untrainable) * ((ub - self.epsilon) - (lb + self.epsilon)) + lb + self.epsilon

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

    def _initialize_objective(self):
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
                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                tf.summary.scalar('loss/%s' % self.loss_type, self.loss_total, collections=['loss'])

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

    def _initialize_logging(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if os.path.exists(self.outdir + '/tensorboard'):
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard')
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard', self.sess.graph)
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')




    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def expand_history(self, X, X_time, first_obs, last_obs):
        """
        Expand 2D matrix of independent variable values into a 3D tensor of histories of independent variable values and a 1D vector of independent variable timestamps into a 2D matrix of histories of independent variable timestamps.
        This is a necessary preprocessing step for the input data when using ``low_memory=False``.
        However, ``fit``, ``predict``, and ``eval`` all call ``expand_history()`` internally, so users generally should not need to call ``expand_history()`` directly and may pass their data to those methods as is.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain a column for each independent variable in the DTSR ``form_str`` provided at iniialization.
        :param X_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the observations in ``X``, grouped and sorted identically to ``X``.
        :param first_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :return: ``tuple``; two numpy arrays ``(X_2d, time_X_2d)``, the expanded IV and timestamp tensors.
        """
        return super(DTSRMLE, self).expand_history(X, X_time, first_obs, last_obs)

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
        """
        Feedforward a batch of data in feed_dict through the convolutional layer to produce convolved inputs

        :param feed_dict: ``dict``; A dictionary of input variables
        :param scale: ``bool``; Whether to scale the outputs using the latent coefficients
        :return: ``numpy`` array; The convolved inputs
        """

        with self.sess.as_default():
            with self.sess.graph.as_default():
                X_conv = self.sess.run(self.X_conv_scaled if scaled else self.X_conv, feed_dict=feed_dict)
                return X_conv

    def make_plots(self, **kwargs):
        """
        Generate plots of current state of deconvolution.
        Saves four plots to the output directory:

            * ``irf_atomic_scaled.jpg``: One line for each IRF kernel in the model (ignoring preconvolution in any composite kernels), scaled by the relevant coefficients
            * ``irf_atomic_unscaled.jpg``: One line for each IRF kernel in the model (ignoring preconvolution in any composite kernels), unscaled
            * ``irf_composite_scaled.jpg``: One line for each IRF kernel in the model (including preconvolution in any composite kernels), scaled by the relevant coefficients
            * ``irf_composite_unscaled.jpg``: One line for each IRF kernel in the model (including preconvolution in any composite kernels), unscaled

        If the model contains no composite IRF, corresponding atomic and composite plots will be identical.

        To save space successive calls to ``make_plots()`` overwrite existing plots.
        Thus, plots only show the most recently plotted state of learning.

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """
        return super(DTSRMLE, self).make_plots(**kwargs)

    def summary(self, fixed=True, random=False):
        summary = '=' * 50 + '\n'
        summary += 'DTSR regression\n\n'
        summary += 'Output directory: %s\n\n' % self.outdir
        summary += 'Formula:\n'
        summary += '  ' + self.form_str + '\n\n'

        #TODO: Fill this in

        return(summary)
