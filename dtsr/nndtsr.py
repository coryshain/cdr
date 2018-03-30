import os
import time
from collections import defaultdict
from numpy import inf
import pandas as pd

pd.options.mode.chained_assignment = None
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.platform.test import is_gpu_available

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
from .formula import *
from .util import *
from .dtsr import DTSR

class NNDTSR(DTSR):
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
        *Because TensorFlow control ops are not currently supported by Edward, BDTSR currently only works with* ``low_memory=False``.
    :param float_type: ``str``; the ``float`` type to use throughout the network.
    :param int_type: ``str``; the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: ``int`` or ``None``; the size of minibatches to use for fitting/prediction (full-batch if ``None``).
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
    :param loss: ``str``; the optimization objective.
        Currently only ``'mse'`` and ``'mae'`` are supported.
    """



    ######################################################
    #
    #  Native methods
    #
    ######################################################

    def __init__(
            self,
            form_str,
            X,
            y,
            outdir,
            history_length=None,
            low_memory=False,
            pc=False,
            float_type='float32',
            int_type='int32',
            minibatch_size=128,
            eval_minibatch_size=100000,
            log_random=True,
            log_freq=1,
            save_freq=1,
            optim='Adam',
            learning_rate=0.01,
            learning_rate_min=1e-4,
            lr_decay_family=None,
            lr_decay_steps=25,
            lr_decay_rate=0.,
            lr_decay_staircase=False,
            init_sd=1,
            loss='mse',
            regularizer=None,
            regularizer_scale=0.01
    ):

        super(NNDTSR, self).__init__(
            form_str,
            X,
            y,
            outdir,
            history_length=history_length,
            low_memory=low_memory,
            pc=pc,
            float_type=float_type,
            int_type=int_type,
            minibatch_size=minibatch_size,
            eval_minibatch_size=eval_minibatch_size,
            save_freq=save_freq,
            log_freq=log_freq,
            log_random=log_random,
            optim=optim,
            learning_rate=learning_rate,
            learning_rate_min=learning_rate_min,
            lr_decay_family=lr_decay_family,
            lr_decay_steps=lr_decay_steps,
            lr_decay_rate=lr_decay_rate,
            lr_decay_staircase=lr_decay_staircase,
            init_sd=init_sd,
            regularizer=regularizer,
            regularizer_scale=regularizer_scale
        )

        self.loss_name = loss

        self.build()

    def __getstate__(self):

        return (
            self.form_str,
            self.outdir,
            self.history_length,
            self.low_memory,
            self.pc,
            self.eigenvec,
            self.eigenval,
            self.impulse_means,
            self.impulse_sds,
            self.float_type,
            self.int_type,
            self.minibatch_size,
            self.eval_minibatch_size,
            self.log_random,
            self.log_freq,
            self.save_freq,
            self.optim_name,
            self.learning_rate,
            self.learning_rate_min,
            self.lr_decay_family,
            self.lr_decay_steps,
            self.lr_decay_rate,
            self.lr_decay_staircase,
            self.init_sd,
            self.regularizer_name,
            self.regularizer_scale,
            self.n_train,
            self.y_mu_init,
            self.y_scale_init,
            self.rangf_map_base,
            self.rangf_n_levels,
            self.loss_name
        )

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self.form_str, \
        self.outdir, \
        self.history_length, \
        self.low_memory, \
        self.pc, \
        self.eigenvec, \
        self.eigenval, \
        self.impulse_means, \
        self.impulse_sds, \
        self.float_type, \
        self.int_type, \
        self.minibatch_size, \
        self.eval_minibatch_size, \
        self.log_random, \
        self.log_freq, \
        self.save_freq, \
        self.optim_name, \
        self.learning_rate, \
        self.learning_rate_min, \
        self.lr_decay_family, \
        self.lr_decay_steps, \
        self.lr_decay_rate, \
        self.lr_decay_staircase, \
        self.init_sd, \
        self.regularizer_name, \
        self.regularizer_scale, \
        self.n_train, \
        self.y_mu_init, \
        self.y_scale_init, \
        self.rangf_map_base, \
        self.rangf_n_levels, \
        self.loss_name = state

        self.__initialize_metadata__()

        self.rangf_map = []
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(defaultdict(lambda: self.rangf_n_levels[i], self.rangf_map_base[i]))

        with self.sess.as_default():
            with self.g.as_default():
                self.y_mu_init_tf = tf.constant(self.y_mu_init, dtype=self.FLOAT_TF)
                self.y_scale_init_tf = tf.constant(self.y_scale_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(1e-35, dtype=self.FLOAT_TF)

        self.build()




    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def __initialize_intercept__(self, ran_gf=None, rangf_n_levels=None):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    intercept = tf.Variable(
                        self.y_mu_init_tf,
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
                    self.__regularize__(intercept)
                    intercept_summary = intercept
                return intercept, intercept_summary

    def __initialize_coefficient__(self, coef_ids=None, ran_gf=None, rangf_n_levels=None):
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
                self.__regularize__(coefficient)
                return coefficient, coefficient_summary

    def __initialize_irf_param__(self, param_name, ids, mean=0, lb=None, ub=None, irf_by_rangf=None):
        dim = len(ids)
        mean = float(mean)
        if irf_by_rangf is None:
            irf_by_rangf = []

        with self.sess.as_default():
            with self.sess.graph.as_default():
                param_mean_init, lb, ub = self.__process_mean__(mean, lb, ub)

                param = tf.Variable(
                    tf.random_normal(
                        shape=[1, dim],
                        mean=param_mean_init,
                        stddev=self.init_sd,
                        dtype=self.FLOAT_TF
                    ),
                    name=sn('%s_%s' % (param_name, '-'.join(ids)))
                )
                self.__regularize__(param, param_mean_init)

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

                param_mean = param_out

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
                                mean=param_mean_init,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='%s_by_%s' % (param_name, gf)
                        )
                        self.__regularize__(param_ran)

                        half_interval = None
                        if lb is not None:
                            half_interval = param_out - lb + self.epsilon
                        elif ub is not None:
                            if half_interval is not None:
                                half_interval = tf.minimum(half_interval, ub - self.epsilon - param_out)
                            else:
                                half_interval = ub - self.epsilon - param_out
                        if half_interval is not None:
                            param_ran = tf.sigmoid(param_ran) * half_interval
                            if lb is not None:
                                param_ran += lb + self.epsilon
                            else:
                                param_ran = ub - self.epsilon - param_ran

                        param_ran *= mask_col
                        param_ran *= tf.expand_dims(mask_row, -1)
                        param_ran -= tf.reduce_mean(param_ran, axis=0)
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

                # Since NNDTSR just learns point estimates, we simply use those
                # estimates for plotting in the 2nd return value
                return (param_out, param_mean)

    def __initialize_objective__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
                self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
                if self.loss_name.lower() == 'mae':
                    self.loss_func = self.mae_loss
                else:
                    self.loss_func = self.mse_loss
                if self.regularizer_name is not None:
                    self.loss_func += self.regularizer_scale * tf.add_n(self.regularizer_losses)
                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                tf.summary.scalar('loss/%s' % self.loss_name, self.loss_total, collections=['loss'])

                self.optim = self.__initialize_optimizer__(self.optim_name)
                assert self.optim is not None, 'An optimizer name must be supplied'

                # self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step,
                #                                     name=sn('optim'))
                # self.gradients = self.optim.compute_gradients(self.loss_func)

                self.gradients, variables = zip(*self.optim.compute_gradients(self.loss_func))
                # ## CLIP GRADIENT NORM
                # self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
                self.train_op = self.optim.apply_gradients(zip(self.gradients, variables),
                                                           global_step=self.global_batch_step, name=sn('optim'))

    def __initialize_logging__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if os.path.exists(self.outdir + '/tensorboard/fixed'):
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed')
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
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
        :return: ``tuple``; two numpy arrays ``(X_3d, time_X_3d)``, the expanded IV and timestamp tensors.
        """
        return super(NNDTSR, self).expand_history(X, X_time, first_obs, last_obs)

    def fit(self,
            X,
            y,
            n_iter=100,
            irf_name_map=None,
            plot_n_time_units=2.5,
            plot_n_points_per_time_unit=1000,
            plot_x_inches=28,
            plot_y_inches=5,
            cmap='gist_earth'):
        """
        Fit the DTSR model.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

            In general, ``y`` will be identical to the parameter ``y`` provided at model initialization.
            However, this is not necessary.
        :param n_iter: ``int``; the number of training iterations
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_n_time_units: ``float``; number if time units to use in plotting.
        :param plot_n_points_per_time_unit: ``float``; number of plot points to use per time unit.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        usingGPU = is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        sys.stderr.write('Correlation matrix for input variables:\n')
        rho = X[impulse_names].corr()
        sys.stderr.write(str(rho) + '\n\n')

        if not np.isfinite(self.minibatch_size):
            minibatch_size = len(y)
        else:
            minibatch_size = self.minibatch_size
        n_minibatch = math.ceil(float(len(y)) / minibatch_size)

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[impulse_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(y.first_obs, dtype=self.INT_NP)
            last_obs = np.array(y.last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)

        self.assign_training_data(
            X=X_3d,
            time_X=time_X_3d,
            y=y_dv,
            time_y=time_y,
            gf_y=gf_y
        )

        ## Compute and log initial losses and parameters

        fd = {
            self.y: y_dv,
            self.time_y: time_y,
            self.gf_y: gf_y
        }

        if self.low_memory:
            fd[self.X] = X_2d
            fd[self.time_X] = time_X_2d
            fd[self.first_obs] = first_obs
            fd[self.last_obs] = last_obs
        else:
            fd[self.X] = X_3d
            fd[self.time_X] = time_X_3d

        if self.global_step.eval(session=self.sess) == 0:
            summary_params = self.sess.run(self.summary_params, feed_dict=fd)
            loss_total = 0.
            for j in range(0, len(y), self.minibatch_size):
                fd[self.y] = y_dv[j:j + minibatch_size]
                fd[self.time_y] = time_y[j:j + minibatch_size]
                fd[self.gf_y] = gf_y[j:j + minibatch_size]
                if self.low_memory:
                    fd[self.first_obs] = first_obs[j:j + minibatch_size]
                    fd[self.last_obs] = last_obs[j:j + minibatch_size]
                else:
                    fd[self.X] = X_3d[j:j + minibatch_size]
                    fd[self.time_X] = time_X_3d[j:j + minibatch_size]
                loss_total += self.sess.run(self.loss_func, feed_dict=fd) * len(fd[self.y])
            loss_total /= len(y)
            summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
            self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

        ## Begin training

        # QueueRunner setup for parallelism, seems to have negative performance impact, so turned off
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():
        #         fd = {
        #             self.batch_size: minibatch_size
        #         }
        #         self.sess.run(self.iterator_init_train, feed_dict=fd)
        #         coord = tf.train.Coordinator()
        #         enqueue_threads = self.qr.create_threads(self.sess, coord=coord, start=True)
        #
        #         sys.stderr.write('-' * 50 + '\n')
        #         sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
        #         sys.stderr.write('\n')
        #         if self.lr_decay_family is not None:
        #             sys.stderr.write('Learning rate: %s\n' % self.lr.eval(session=self.sess))
        #
        #         pb = tf.contrib.keras.utils.Progbar(n_minibatch)
        #         j = 0
        #         loss_total = 0.
        #         t0_iter = time.time()
        #
        #         while self.global_step.eval(session=self.sess) < n_iter:
        #             try:
        #                 _, loss_minibatch = self.sess.run(
        #                     [self.train_op, self.loss_func],
        #                     feed_dict=fd
        #                 )
        #                 loss_total += loss_minibatch
        #                 pb.update(j + 1, values=[('loss', loss_minibatch)])
        #                 j += 1
        #                 if j >= n_minibatch:
        #                     loss_total /= n_minibatch
        #                     fd[self.loss_total] = loss_total
        #
        #                     self.sess.run(self.incr_global_step)
        #
        #                     if self.global_step.eval(session=self.sess) % self.save_freq == 0:
        #                         self.save()
        #                         self.make_plots(
        #                             irf_name_map=irf_name_map,
        #                             plot_n_time_units=plot_n_time_units,
        #                             plot_n_points_per_time_unit=plot_n_points_per_time_unit,
        #                             plot_x_inches=plot_x_inches,
        #                             plot_y_inches=plot_y_inches,
        #                             cmap=cmap
        #                         )
        #
        #                     if self.global_step.eval(session=self.sess) % self.log_freq == 0:
        #                         summary_params = self.sess.run(self.summary_params, feed_dict=fd)
        #                         summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
        #                         self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
        #                         self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
        #
        #                         if self.log_random and len(self.rangf) > 0:
        #                             summary_random = self.sess.run(self.summary_random)
        #                             self.writer.add_summary(summary_random,
        #                                                     self.global_batch_step.eval(session=self.sess))
        #
        #                     t1_iter = time.time()
        #
        #                     sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))
        #
        #                     t0_iter = time.time()
        #
        #                     sys.stderr.write('-' * 50 + '\n')
        #                     sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
        #                     sys.stderr.write('\n')
        #                     if self.lr_decay_family is not None:
        #                         sys.stderr.write('Learning rate: %s\n' % self.lr.eval(session=self.sess))
        #
        #                     pb = tf.contrib.keras.utils.Progbar(n_minibatch)
        #                     j = 0
        #                     loss_total = 0.
        #             except:
        #                 break
        #
        #         coord.request_stop()
        #         coord.join(enqueue_threads)
        #
        # return

        with self.sess.as_default():
            with self.sess.graph.as_default():

                fd = {
                    self.y: y_dv,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                if self.low_memory:
                    fd[self.X] = X_2d
                    fd[self.time_X] = time_X_2d
                    fd[self.first_obs] = first_obs
                    fd[self.last_obs] = last_obs
                else:
                    fd[self.X] = X_3d
                    fd[self.time_X] = time_X_3d

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                if self.global_step.eval(session=self.sess) == 0:
                    summary_params = self.sess.run(self.summary_params, feed_dict=fd)
                    loss_total = 0.
                    for j in range(0, len(y), self.minibatch_size):
                        fd_minibatch[self.y] = y_dv[j:j + minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + minibatch_size]
                        loss_total += self.sess.run(self.loss_func, feed_dict=fd_minibatch)*len(fd_minibatch[self.y])
                    loss_total /= len(y)
                    summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                while self.global_step.eval(session=self.sess) < n_iter:
                    p, p_inv = getRandomPermutation(len(y))
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    if self.lr_decay_family is not None:
                        sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.
                    max_grad = 0

                    for j in range(0, len(y), minibatch_size):
                        indices = p[j:j + self.minibatch_size]
                        fd_minibatch[self.y] = y_dv[indices]
                        fd_minibatch[self.time_y] = time_y[indices]
                        fd_minibatch[self.gf_y] = gf_y[indices]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[indices]
                            fd_minibatch[self.last_obs] = last_obs[indices]
                        else:
                            fd_minibatch[self.X] = X_3d[indices]
                            fd_minibatch[self.time_X] = time_X_3d[indices]

                        _, loss_minibatch = self.sess.run(
                            [self.train_op, self.loss_func],
                            feed_dict=fd_minibatch
                        )
                        loss_total += loss_minibatch

                        # max_grad_minibatch = np.max([np.max(g) for g in grads_minibatch])
                        # max_grad = max(max_grad, max_grad_minibatch)

                        pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)])

                    loss_total /= n_minibatch
                    fd[self.loss_total] = loss_total
                    # sys.stderr.write('Evaluating gradients...\n')
                    # max_grad = 0
                    # for g in gradients:
                    #     max_grad = max(max_grad, np.max(np.abs(g[0]/n_minibatch)))
                    # sys.stderr.write('  max(grad) = %s\n' % str(max_grad))
                    # sys.stderr.write('  Converged (max(grad) < 0.001) = %s\n' % (max_grad < 0.001))

                    self.sess.run(self.incr_global_step)

                    if self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        self.save()
                        self.make_plots(
                            irf_name_map=irf_name_map,
                            plot_n_time_units=plot_n_time_units,
                            plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap
                        )

                    if self.global_step.eval(session=self.sess) % self.log_freq == 0:
                        summary_params = self.sess.run(self.summary_params, feed_dict=fd)
                        summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                        self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                        self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                        if self.log_random and len(self.rangf) > 0:
                            summary_random = self.sess.run(self.summary_random)
                            self.writer.add_summary(summary_random, self.global_batch_step.eval(session=self.sess))

                    t1_iter = time.time()

                    # sys.stderr.write('Maximum gradient norm: %s\n' %max_grad)
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                self.make_plots(
                    irf_name_map=irf_name_map,
                    plot_n_time_units=plot_n_time_units,
                    plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                    plot_x_inches=plot_x_inches,
                    plot_y_inches=plot_y_inches,
                    cmap=cmap
                )

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        """
        Predict from the pre-trained DTSR model.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the regression targets, grouped by series.
        :param y_rangf: ``pandas`` ``Series`` or 1D ``numpy`` array; random grouping factor values (if applicable). Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param first_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :return: 1D ``numpy`` array; network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert len(y_time) == len(y_rangf) == len(first_obs) == len(last_obs), 'y_time, y_rangf, first_obs, and last_obs must be of identical length. Got: len(y_time) = %d, len(y_rangf) = %d, len(first_obs) = %d, len(last_obs) = %d' % (len(y_time), len(y_rangf), len(first_obs), len(last_obs))

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[impulse_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(first_obs, dtype=self.INT_NP)
            last_obs = np.array(last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, first_obs, last_obs)
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        preds = np.zeros(len(y_time))

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                if self.low_memory:
                    fd[self.X] = X_2d
                    fd[self.time_X] = time_X_2d
                    fd[self.first_obs] = first_obs
                    fd[self.last_obs] = last_obs
                else:
                    fd[self.X] = X_3d
                    fd[self.time_X] = time_X_3d

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                if not np.isfinite(self.eval_minibatch_size):
                    preds = self.sess.run(self.out, feed_dict=fd)
                else:
                    for j in range(0, len(y_time), self.eval_minibatch_size):
                        fd_minibatch[self.time_y] = time_y[j:j + self.eval_minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.eval_minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.eval_minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.eval_minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.eval_minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.eval_minibatch_size]

                        preds[j:j + self.eval_minibatch_size] = self.sess.run(self.out, feed_dict=fd_minibatch)

                return preds

    def eval(self, X, y):
        """
        Evaluate the pre-trained DTSR model.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :return: ``float`` (scalar); the value of the evaluation metric (MSE/MAE) for the evaluation data.
        """
        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[impulse_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(y.first_obs, dtype=self.INT_NP)
            last_obs = np.array(y.last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                if self.low_memory:
                    fd[self.X] = X_2d
                    fd[self.time_X] = time_X_2d
                    fd[self.first_obs] = first_obs
                    fd[self.last_obs] = last_obs
                    fd[self.y] = y_dv
                else:
                    fd[self.X] = X_3d
                    fd[self.time_X] = time_X_3d
                    fd[self.y] = y_dv

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                out = 0.

                if not np.isfinite(self.eval_minibatch_size):
                    out = self.sess.run(self.loss_func, feed_dict=fd)
                else:
                    for j in range(0, len(y), self.eval_minibatch_size):
                        fd_minibatch[self.y] = y_dv[j:j + self.eval_minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + self.eval_minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.eval_minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.eval_minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.eval_minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.eval_minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.eval_minibatch_size]
                        out += self.sess.run(self.loss_func, feed_dict=fd_minibatch)*len(fd_minibatch[self.y])
                    out /= len(y)

                return out

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
        return super(NNDTSR, self).make_plots(**kwargs)

    def run_conv_op(self, feed_dict, scaled=False):
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
