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
from .plot import *
from .dtsr import sn, DTSR

class NNDTSR(DTSR):
    """
    A neural net implementation of DTSR.

    :param form_str: An R-style string representing the DTSR model formula.
    :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:

        * ``time``: Timestamp for each entry in ``y``
        * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with
          each entry in ``y``
        * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series
          associated with each entry in ``y``
        * A column with the same name as the DV specified in ``form_str``
        * A column for each random grouping factor in the model specified in ``form_str``.

    :param outdir: A ``str`` representing the output directory, where logs and model parameters are saved.
    :param history_length: An ``int`` representing the maximum length of the history window to use. If ``None``, history
        length is unbounded and only the low-memory model is permitted.
    :param low_memory: A ``bool`` determining which DTSR memory implementation to use.
        If ``low_memory == True``, DTSR convolves over history windows for each observation of in ``y`` using a TensorFlow control op.
        It can be used with unboundedly long histories and requires less memory, but results in poor GPU utilization.
        If ``low_memory == False``, DTSR expands the design matrix into a rank 3 tensor in which the 2nd axis contains the history for each independent variable for each observation of the dependent variable.
        This requires more memory in order to store redundant input values and requires a finite history length.
        However, it removes the need for a control op in the feedforward component and therefore generally runs much faster if GPU is available.
    :param float_type: A ``str`` representing the ``float`` type to use throughout the network.
    :param int_type: A ``str`` representing the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: An ``int`` representing the size of minibatches to use for fitting/prediction, or the
        string ``inf`` to perform full-batch training.
    :param logging_freq: An ``int`` representing the frequency (in minibatches) with which to write Tensorboard logs.
    :param log_random: A ``bool`` determining whether to log random effects to Tensorboard.
    :param save_freq: An ``int`` representing the frequency (in iterations) with which to save model checkpoints.
    :param optim: A ``str`` representing the name of the optimizer to use. Choose from ``SGD``, ``AdaGrad``, ``AdaDelta``,
        ``Adam``, ``FTRL``, ``RMSProp``, ``Nadam``.
    :param learning_rate: A ``float`` representing the initial value for the learning rate.
    :param learning_rate_decay_factor: A ``float`` used to compute the rate of learning rate decay (if applicable).
    :param learning_rate_decay_family: A ``str`` representing the functional family for the learning rate decay
        schedule (if applicable). Choose from the following, where :math:`\lambda` is the current learning rate,
        :math:`\lambda_0` is the initial learning rate, :math:`\delta` is the ``learning_rate_decay_factor``,
        and :math:`i` is the iteration index.

        * ``linear``: :math:`\\lambda_0 \\cdot ( 1 - \\delta \\cdot i )`
        * ``inverse``: :math:`\\frac{\\lambda_0}{1 + ( \\delta \\cdot i )}`
        * ``exponential``: :math:`\\lambda = \\lambda_0 \\cdot ( 2^{-\\delta \\cdot i} )`
        * ``stepdownXX``: where ``XX`` is replaced by an integer representing the stepdown interval :math:`a`:
          :math:`\\lambda = \\lambda_0 * \\delta^{\\left \\lfloor \\frac{i}{a} \\right \\rfloor}`

    :param learning_rate_min: A ``float`` representing the minimum value for the learning rate. If the decay schedule
        would take the learning rate below this point, learning rate clipping will occur.
    :param loss: A ``str`` representing the optimization objective. Currently only ``MAE`` and ``MSE`` are supported.
    """



    ######################################################
    #
    #  Native methods
    #
    ######################################################

    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 history_length=None,
                 low_memory=True,
                 float_type='float32',
                 int_type='int32',
                 minibatch_size=None,
                 logging_freq=1,
                 log_random=True,
                 save_freq=1,
                 optim='Adam',
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.,
                 learning_rate_decay_family=None,
                 learning_rate_min=1e-4,
                 loss='mse',
                 ):

        super(NNDTSR, self).__init__(
            form_str,
            y,
            outdir,
            history_length=history_length,
            low_memory=low_memory,
            float_type=float_type,
            int_type=int_type,
            minibatch_size=minibatch_size,
            logging_freq=logging_freq,
            log_random=log_random,
            save_freq=save_freq
        )

        self.optim_name = optim
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_rate_decay_family = learning_rate_decay_family
        self.learning_rate_min = learning_rate_min
        self.loss_name = loss

        self.build()

    def __getstate__(self):

        return (
            self.form_str,
            self.outdir,
            self.rangf_map,
            self.rangf_n_levels,
            self.y_mu_init,
            self.float_type,
            self.int_type,
            self.log_random,
            self.optim_name,
            self.learning_rate,
            self.learning_rate_decay_factor,
            self.learning_rate_decay_family,
            self.learning_rate_min,
            self.loss_name,
        )

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)
        self.form_str, \
        self.outdir, \
        self.rangf_map, \
        self.rangf_n_levels, \
        self.y_mu_init, \
        self.float_type, \
        self.int_type, \
        self.log_random, \
        self.optim_name, \
        self.learning_rate, \
        self.learning_rate_decay_factor, \
        self.learning_rate_decay_family, \
        self.learning_rate_min, \
        self.loss_name = state

        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        self.form = Formula(self.form_str)
        self.irf_tree = self.form.irf_tree

        self.build()




    ######################################################
    #
    #  Private methods
    #
    ######################################################

    def __initialize_intercepts_coefficients__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if f.intercept:
                    self.intercept_fixed = tf.Variable(tf.constant(self.y_mu_init, shape=[1]), dtype=self.FLOAT_TF,
                                                       name='intercept')
                else:
                    self.intercept_fixed = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')
                self.intercept = self.intercept_fixed
                tf.summary.scalar('intercept', self.intercept[0], collections=['params'])

                self.coefficient_fixed = tf.Variable(
                    tf.truncated_normal(shape=[len(f.coefficient_names)], mean=0., stddev=0.1, dtype=self.FLOAT_TF),
                    name='coefficient_fixed')
                for i in range(len(f.coefficient_names)):
                    tf.summary.scalar('coefficient' + '/%s' % f.coefficient_names[i], self.coefficient_fixed[i], collections=['params'])
                self.coefficient = self.coefficient_fixed
                fixef_ix = names2ix(f.fixed_coefficient_names, f.coefficient_names)
                coefficient_fixed_mask = np.zeros(len(f.coefficient_names), dtype=self.FLOAT_NP)
                coefficient_fixed_mask[fixef_ix] = 1.
                coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)
                self.coefficient_fixed *= coefficient_fixed_mask
                self.coefficient_fixed_means = self.coefficient_fixed
                self.coefficient = self.coefficient_fixed
                self.coefficient = tf.expand_dims(self.coefficient, 0)
                self.ransl = False
                for i in range(len(f.ran_names)):
                    r = f.random[f.ran_names[i]]
                    mask_row_np = np.ones(self.rangf_n_levels[i], dtype=self.FLOAT_NP)
                    mask_row_np[self.rangf_n_levels[i] - 1] = 0
                    mask_row = tf.constant(mask_row_np)

                    if r.intercept:
                        intercept_random = tf.Variable(
                            tf.truncated_normal(shape=[self.rangf_n_levels[i]], mean=0., stddev=.1, dtype=tf.float32),
                            name='intercept_by_%s' % r.gf)
                        intercept_random *= mask_row
                        intercept_random -= tf.reduce_mean(intercept_random, axis=0)

                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                'by_%s/intercept' % r.gf,
                                intercept_random,
                                collections=['random']
                            )
                    if len(coefs) > 0:
                        coefs = r.coefficient_names
                        coef_ix = names2ix(coefs, f.coefficient_names)
                        mask_col_np = np.zeros(len(f.coefficient_names))
                        mask_col_np[coef_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)
                        self.ransl = True

                        coefficient_random = tf.Variable(
                            tf.truncated_normal(shape=[self.rangf_n_levels[i], len(f.coefficient_names)], mean=0., stddev=.1,
                                                dtype=tf.float32), name='coefficient_by_%s' % (r.gf))

                        coefficient_random *= mask_col
                        coefficient_random *= tf.expand_dims(mask_row, -1)
                        coefficient_random -= tf.reduce_mean(coefficient_random, axis=0)

                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(r.coefficient_names)):
                                coef_name = r.coefficient_names[j]
                                ix = coef_ix[j]
                                tf.summary.histogram(
                                    'by_%s/coefficient/%s' % (r.gf, coef_name),
                                    coefficient_random[:, ix],
                                    collections=['random']
                                )

    def __initialize_irf_params__(self):
        f = self.form
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for x in f.atomic_irf_by_family:
                    self.atomic_irf_by_family[x] = self.__initialize_irf_params_inner__(x, sorted(f.atomic_irf_by_family[x]))
                    self.atomic_irf_means_by_family[x] = self.atomic_irf_by_family[x]

    def __initialize_irf_params_inner__(self, family, ids):
        ## Infinitessimal value to add to bounded parameters
        epsilon = 1e-35
        dim = len(ids)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if family == 'DiracDelta':
                    filler = tf.constant(1., shape=[1, dim])
                    return filler
                if family == 'Exp':
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[1, dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                    return L
                if family == 'ShiftedExp':
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT_TF)
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([L, delta], axis=0)
                if family == 'Gamma':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    return tf.stack([k, theta])
                if family == 'GammaKgt1':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon + 1.
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    return tf.stack([k, theta])
                if family == 'ShiftedGamma':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT_TF)
                    k = tf.exp(log_k, name=sn('k')) + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([k, theta, delta], axis=0)
                if family == 'ShiftedGammaKgt1':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT_TF)
                    k = tf.nn.softplus(log_k, name=sn('k_%s' % '-'.join(ids))) + 1. + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([k, theta, delta], axis=0)
                if family == 'Normal':
                    log_sigma = tf.Variable(
                        tf.truncated_normal([dim], stddev=.1, dtype=self.FLOAT_TF),
                        name=sn('log_sigma_%s' % '-'.join(ids))
                    )
                    mu = tf.Variable(
                        tf.truncated_normal([dim], stddev=.1, dtype=self.FLOAT_TF),
                        name=sn('mu_%s' % '-'.join(ids))
                    )
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                    return tf.stack([mu, sigma], axis=0)
                if family == 'SkewNormal':
                    log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim],
                                         initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    alpha = tf.get_variable(sn('alpha_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                    return tf.stack([mu, sigma, alpha], axis=1)
                if family == 'EMG':
                    log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim],
                                         initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                    return tf.stack([mu, sigma, L], axis=0)
                if family == 'BetaPrime':
                    log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim],
                                               initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                    beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                        tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                    return tf.stack([alpha, beta], axis=0)
                if family == 'ShiftedBetaPrime':
                    log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim],
                                               initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT_TF)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT_TF)
                    alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                    beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                        tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([alpha, beta, delta], axis=0)
                raise ValueError('Impulse response function "%s" is not currently supported.' % family)

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
                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                tf.summary.scalar('loss/%s' % self.loss_name, self.loss_total, collections=['loss'])

                sys.stderr.write('Using optimizer %s\n' % self.optim_name)
                self.lr = tf.constant(self.learning_rate)
                if self.learning_rate_decay_factor > 0:
                    decay_step = tf.cast(self.global_step, dtype=self.FLOAT_TF) * tf.constant(
                        self.learning_rate_decay_factor)
                    if self.learning_rate_decay_family == 'linear':
                        decay_coef = 1 - decay_step
                    elif self.learning_rate_decay_family == 'inverse':
                        decay_coef = 1. / (1. + decay_step)
                    elif self.learning_rate_decay_family == 'exponential':
                        decay_coef = 2 ** (-decay_step)
                    elif self.learning_rate_decay_family.startswith('stepdown'):
                        interval = tf.constant(float(self.learning_rate_decay_family[8:]), dtype=self.FLOAT_TF)
                        decay_coef = tf.constant(self.learning_rate_decay_factor) ** tf.floor(
                            tf.cast(self.global_step, dtype=self.FLOAT_TF) / interval)
                    else:
                        raise ValueError(
                            'Unrecognized learning rate decay schedule: "%s"' % self.learning_rate_decay_family)
                    self.lr = self.lr * decay_coef
                    self.lr = tf.clip_by_value(self.lr, tf.constant(self.learning_rate_min), inf)
                self.optim = self.__optim_init__(self.optim_name, self.lr)

                # self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step,
                #                                     name=sn('optim'))
                # self.gradients = self.optim.compute_gradients(self.loss_func)

                self.gradients, variables = zip(*self.optim.compute_gradients(self.loss_func))
                # ## CLIP GRADIENT NORM
                # self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
                self.train_op = self.optim.apply_gradients(zip(self.gradients, variables),
                                                           global_step=self.global_batch_step, name=sn('optim'))

    def __start_logging__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random and len(f.random) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def __optim_init__(self, name, learning_rate):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return {
                    'SGD': lambda x: tf.train.GradientDescentOptimizer(x),
                    'AdaGrad': lambda x: tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: tf.contrib.opt.NadamOptimizer(x)
                }[name](learning_rate)




    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def build(self, restore=True, verbose=True):
        """
        Construct the DTSR network and initialize/load model parameters.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Show the model tree when called.
        :return: ``None``
        """
        if verbose:
            sys.stderr.write('Constructing network from model tree:\n')
            sys.stdout.write(str(self.irf_tree))
            sys.stdout.write('\n')

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        if self.low_memory:
            self.__initialize_low_memory_inputs__()
        else:
            self.__initialize_inputs__()
        self.__initialize_intercepts_coefficients__()
        self.__initialize_irf_lambdas__()
        self.__initialize_irf_params__()
        self.__initialize_irfs__(self.irf_tree)
        if self.low_memory:
            self.__construct_low_memory_network__()
        else:
            self.__construct_network__()
        self.__initialize_objective__()
        self.__start_logging__()
        self.__initialize_saver__()
        self.load(restore=restore)
        self.__report_n_params__()

    def fit(self,
            X,
            y,
            n_epoch_train=100,
            n_epoch_tune=100,
            irf_name_map=None,
            plot_x_inches=28,
            plot_y_inches=5,
            cmap='gist_earth'):
        """
        Train the DTSR model.

        :param X:
        :param y:
        :param n_epoch_train:
        :param n_epoch_tune:
        :param irf_name_map:
        :param plot_x_inches:
        :param plot_y_inches:
        :param cmap:
        :return:
        """

        usingGPU = is_gpu_available()

        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        f = self.form

        sys.stderr.write('Correlation matrix for input variables:Corr\n')
        rho = X[f.terminal_names].corr()
        sys.stderr.write(str(rho) + '\n\n')

        if not np.isfinite(self.minibatch_size):
            minibatch_size = len(y)
        else:
            minibatch_size = self.minibatch_size
        n_minibatch = math.ceil(float(len(y)) / minibatch_size)

        y_rangf = y[f.rangf]
        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[f.terminal_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(y.first_obs, dtype=self.INT_NP)
            last_obs = np.array(y.last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)
        y_dv = np.array(y[f.dv], dtype=self.FLOAT_NP)

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
                        fd_minibatch[self.y] = y_dv[j:j + self.minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + self.minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.minibatch_size]
                        loss_total += self.sess.run(self.loss_func, feed_dict=fd_minibatch)*len(fd_minibatch[self.y])
                    loss_total /= len(y)
                    summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                while self.global_step.eval(session=self.sess) < n_epoch_train:
                    p, p_inv = getRandomPermutation(len(y))
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    if self.learning_rate_decay_factor > 0:
                        sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.

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

                        # _, gradients_minibatch, loss_minibatch = self.sess.run(
                        #     [self.train_op, self.gradients, self.loss_func],
                        #     feed_dict=fd_minibatch)
                        _, loss_minibatch = self.sess.run(
                            [self.train_op, self.loss_func],
                            feed_dict=fd_minibatch)
                        loss_total += loss_minibatch
                        # if gradients is None:
                        #     gradients = list(gradients_minibatch)
                        # else:
                        #     for i in range(len(gradients)):
                        #         gradients[i] += gradients_minibatch[i]
                        pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

                    loss_total /= n_minibatch
                    fd[self.loss_total] = loss_total
                    # sys.stderr.write('Evaluating gradients...\n')
                    # max_grad = 0
                    # for g in gradients:
                    #     max_grad = max(max_grad, np.max(np.abs(g[0]/n_minibatch)))
                    # sys.stderr.write('  max(grad) = %s\n' % str(max_grad))
                    # sys.stderr.write('  Converged (max(grad) < 0.001) = %s\n' % (max_grad < 0.001))

                    self.sess.run(self.incr_global_step)

                    summary_params = self.sess.run(self.summary_params, feed_dict=fd)
                    loss_total = 0.
                    for j in range(0, len(y), self.minibatch_size):
                        fd_minibatch[self.y] = y_dv[j:j + self.minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + self.minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.minibatch_size]
                        loss_total += self.sess.run(self.loss_func, feed_dict=fd_minibatch)*len(fd_minibatch[self.y])
                    loss_total /= len(y)
                    summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                    if self.log_random and len(f.random) > 0:
                        summary_random = self.sess.run(self.summary_random)
                        self.writer.add_summary(summary_random, self.global_batch_step.eval(session=self.sess))

                    self.save()

                    sys.stderr.write('Number of graph nodes: %d\n' % len(self.sess.graph._nodes_by_name))
                    self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)

                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                X_conv = pd.DataFrame(self.sess.run(self.X_conv, feed_dict=fd), columns=self.preterminals)

                sys.stderr.write('Mean values of convolved predictors\n')
                sys.stderr.write(str(X_conv.mean(axis=0)) + '\n')
                sys.stderr.write('Correlations of convolved predictors')
                sys.stderr.write(str(X_conv.corr()) + '\n')
                sys.stderr.write('\n')

                self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)


    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        """
        Predict from the pre-trained DTSR model.

        :param X:
        :param y_time:
        :param y_rangf:
        :param first_obs:
        :param last_obs:
        :return:
        """
        f = self.form

        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[f.terminal_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(first_obs, dtype=self.INT_NP)
            last_obs = np.array(last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, first_obs, last_obs)
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

                if not np.isfinite(self.minibatch_size):
                    preds = self.sess.run(self.out, feed_dict=fd)
                else:
                    for j in range(0, len(y_time), self.minibatch_size):
                        fd_minibatch[self.time_y] = time_y[j:j + self.minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.minibatch_size]

                        preds[j:j + self.minibatch_size] = self.sess.run(self.out, feed_dict=fd_minibatch)

                return preds

    def eval(self, X, y):
        """
        Evaluate the pre-trained DTSR model

        :param X:
        :param y:
        :return:
        """
        f = self.form

        y_rangf = y[f.rangf]
        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        if self.low_memory:
            X_2d = X[f.terminal_names]
            time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
            first_obs = np.array(y.first_obs, dtype=self.INT_NP)
            last_obs = np.array(y.last_obs, dtype=self.INT_NP)
        else:
            X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)
        y_dv = np.array(y[f.dv], dtype=self.FLOAT_NP)

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

                loss = 0.

                if not np.isfinite(self.minibatch_size):
                    loss = self.sess.run(self.loss_func, feed_dict=fd)
                else:
                    for j in range(0, len(y), self.minibatch_size):
                        fd_minibatch[self.y] = y_dv[j:j + self.minibatch_size]
                        fd_minibatch[self.time_y] = time_y[j:j + self.minibatch_size]
                        fd_minibatch[self.gf_y] = gf_y[j:j + self.minibatch_size]
                        if self.low_memory:
                            fd_minibatch[self.first_obs] = first_obs[j:j + self.minibatch_size]
                            fd_minibatch[self.last_obs] = last_obs[j:j + self.minibatch_size]
                        else:
                            fd_minibatch[self.X] = X_3d[j:j + self.minibatch_size]
                            fd_minibatch[self.time_X] = time_X_3d[j:j + self.minibatch_size]
                        loss += self.sess.run(self.loss_func, feed_dict=fd_minibatch)*len(fd_minibatch[self.y])
                    loss /= len(y)

                return loss
