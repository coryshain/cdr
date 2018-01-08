import os
import math
import pandas as pd
from numpy import inf, nan

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

import edward as ed
from edward.models import Normal, Gamma, Empirical


class BDTSR(DTSR):
    """
    A Bayesian implementation of DTSR.

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
    :param low_memory: A ``bool`` determining which DTSR memory implementation to use. If ``low_memory == True``, DTSR
        convolves over history windows for each observation of in ``y`` using a TensorFlow control op. It can be used
        with unboundedly long histories and uses less memory, but is generally much slower and results in poor GPU
        utilization. If ``low_memory == False``, DTSR expands the design matrix into a rank 3 tensor in which the 2nd
        axis contains the history for each independent variable for each observation of the independent variable.
        This requires more memory in order to store redundant input values and requires a finite history length.
        However, it removes the need for a control op in the feedforward component and therefore generally runs much
        faster if GPU is available.
    :param float_type: A ``str`` representing the ``float`` type to use throughout the network.
    :param int_type: A ``str`` representing the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: An ``int`` representing the size of minibatches to use for fitting/prediction, or the
        string ``inf`` to perform full-batch training.
    :param logging_freq: An ``int`` representing the frequency (in minibatches) with which to write Tensorboard logs.
    :param log_random: A ``bool`` determining whether to log random effects to Tensorboard.
    :param save_freq: An ``int`` representing the frequency (in iterations) with which to save model checkpoints.
    :param inference_name: A ``str`` representing the Edward inference class to use for fitting
    :param n_samples: An ``int`` representing the number of samples to use from the variational posterior if using
        variational inference. If using MCMC, this value is set deterministically as ``n_iter*n_minibatch``, so
        this user-supplied parameter is ignored.
    :param n_samples_eval: An ``int`` representing the number of samples from the predictive posterior to use for
        evaluation/prediction.
    :param n_iter: An ``int`` representing the number of iterations to perform in training. Must be supplied to
        ``__init__()`` because MCMC optimizations hard-code this into the network structure.
    :param conv_prior_sd: A ``float`` representing the standard deviation of the Normal prior on convolution parameters.
        Smaller values concentrate probability mass around the prior mean.
    :param coef_prior_sd: A ``float`` representing the standard deviation of the Normal prior on coefficient parameters.
        Smaller values concentrate probability mass around the prior mean.
    :param y_sigma_scale: A ``float`` representing the scaling coefficient on the standard deviation of the
        distribution of the dependent variable. Specifically, the DV is assumed to have the standard
        deviation ``stddev(y_train)*y_sigma_scale``.
    """

    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 history_length=100,
                 low_memory=False,
                 float_type='float32',
                 int_type='int32',
                 minibatch_size=None,
                 logging_freq = 1,
                 log_random=True,
                 save_freq=1,
                 inference_name='KLqp',
                 n_samples=None,
                 n_samples_eval=100,
                 n_iter=1000,
                 conv_prior_sd=1,
                 coef_prior_sd=1,
                 y_sigma_scale=0.5
                 ):

        super(BDTSR, self).__init__(
            form_str,
            y,
            outdir,
            history_length=history_length,
            low_memory=low_memory,
            float_type=float_type,
            int_type=int_type,
            minibatch_size=minibatch_size,
            logging_freq=logging_freq,
            save_freq=save_freq,
            log_random = log_random
        )

        assert not self.low_memory, 'Because Edward does not support Tensorflow control ops, low_memory is not supported in BDTSR'
        try:
            float(self.history_length)
        except:
            raise ValueError('Because Edward does not support Tensorflow control ops, finite history_length must be specified in BDTSR')

        self.inference_name = inference_name

        self.n_iter = n_iter
        if self.variational():
            self.n_samples = n_samples
        else:
            if n_samples is not None:
                sys.stderr.write('Parameter n_samples being overridden for sampling optimization\n')
            self.n_samples = self.n_iter*self.n_train_minibatch
        self.logging_freq = logging_freq
        self.n_samples_eval = n_samples_eval
        self.conv_prior_sd = float(conv_prior_sd)
        self.coef_prior_sd = float(coef_prior_sd)
        self.y_sigma_scale = float(y_sigma_scale)

        self.inference_map = {}
        if self.inference_name == 'MetropolisHastings':
            self.proposal_map = {}

        self.build()

    def variational(self):
        return self.inference_name in [
            'KLpq',
            'KLqp',
            'ImplicitKLqp',
            'ReparameterizationEntropyKLqp',
            'ReparameterizationKLKLqp',
            'ReparameterizationKLqp',
            'ScoreEntropyKLqp',
            'ScoreKLKLqp',
            'ScoreKLqp',
            'ScoreRBKLqp',
            'WakeSleep'
        ]

    def build(self):
        sys.stderr.write('Constructing network from model tree:\n')
        sys.stdout.write(str(self.irf_tree))
        sys.stdout.write('\n')

        self.initialize_inputs()
        self.initialize_intercepts_coefficients()
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():
        #         self.out = self.X[:,-1,:]
        #         self.out = self.intercept + tf.squeeze(tf.matmul(self.out, tf.expand_dims(self.coefficient[0], -1)), -1)
        #         self.out = Normal(loc=self.out, scale=1., name='output')
        self.initialize_irf_lambdas()
        self.initialize_irf_params()
        self.initialize_irfs(self.irf_tree)
        self.construct_network()
        self.initialize_objective()
        self.start_logging()
        self.initialize_saver()
        self.load()
        self.report_n_params()

    def initialize_intercepts_coefficients(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if f.intercept:
                    self.intercept_fixed = Normal(
                        loc=tf.constant(self.y_mu_init, shape=[1]),
                        scale=self.coef_prior_sd,
                        name='intercept'
                    )
                    if self.variational():
                        self.intercept_fixed_q_loc = tf.Variable(
                            tf.random_normal([1], mean=self.y_mu_init, stddev=self.coef_prior_sd),
                            dtype=self.FLOAT_TF,
                            name='intercept_q_loc'
                        )
                        self.intercept_fixed_q_scale = tf.Variable(
                            tf.random_normal([1], mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd), stddev=self.coef_prior_sd),
                            dtype=self.FLOAT_TF,
                            name='intercept_q_scale'
                        )
                        self.intercept_fixed_q = Normal(
                            loc=self.intercept_fixed_q_loc,
                            scale=tf.nn.softplus(self.intercept_fixed_q_scale),
                            name='intercept_q'
                        )
                        tf.summary.scalar('intercept',
                                          self.intercept_fixed_q.mean()[0],
                                          collections=['params'])
                    else:
                        self.intercept_fixed_q = Empirical(
                            params=tf.Variable(tf.ones((self.n_samples,1))*self.y_mu_init, name='intercept_q'),
                            name='intercept_q'
                        )
                        if self.inference_name == 'MetropolisHastings':
                            self.intercept_fixed_proposal = Normal(
                                loc=self.intercept_fixed,
                                scale=self.coef_prior_sd,
                                name='intercept_proposal'
                            )
                            self.proposal_map[self.intercept_fixed] =  self.intercept_fixed_proposal
                        tf.summary.scalar('intercept',
                                          self.intercept_fixed_q.params[self.global_batch_step-1,0],
                                          collections=['params'])
                    self.inference_map[self.intercept_fixed] = self.intercept_fixed_q
                else:
                    self.intercept_fixed = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')
                self.intercept = self.intercept_fixed

                self.coefficient_fixed = Normal(
                    loc=tf.zeros([len(f.coefficient_names)]),
                    scale=self.coef_prior_sd,
                    name='coefficient_fixed'
                )
                if self.variational():
                    self.coefficient_fixed_q_loc = tf.Variable(
                        tf.random_normal([len(f.coefficient_names)], stddev=self.coef_prior_sd),
                        dtype=self.FLOAT_TF,
                        name='coefficient_fixed_q_loc'
                    )
                    self.coefficient_fixed_q_scale = tf.Variable(
                            tf.random_normal([len(f.coefficient_names)], mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd), stddev=self.coef_prior_sd),
                            dtype=self.FLOAT_TF,
                            name='coefficient_fixed_q_scale'
                        )
                    self.coefficient_fixed_q = Normal(
                        loc=self.coefficient_fixed_q_loc,
                        scale=tf.nn.softplus(self.coefficient_fixed_q_scale),
                        name='coefficient_fixed_q'
                    )
                    for i in range(len(f.coefficient_names)):
                        tf.summary.scalar('coefficient' + '/%s' % f.coefficient_names[i],
                                          self.coefficient_fixed_q.mean()[i],
                                          collections=['params'])
                else:
                    self.coefficient_fixed_q = Empirical(
                        params=tf.Variable(tf.zeros((self.n_samples, len(f.coefficient_names)))),
                        name='coefficient_fixed_q'
                    )
                    if self.inference_name == 'MetropolisHastings':
                        self.coefficient_fixed_proposal = Normal(
                            loc=self.coefficient_fixed,
                            scale=self.coef_prior_sd,
                            name='coefficient_fixed_proposal'
                        )
                        self.proposal_map[self.coefficient_fixed] = self.coefficient_fixed_proposal
                    for i in range(len(f.coefficient_names)):
                        tf.summary.scalar('coefficient' + '/%s' % f.coefficient_names[i],
                                          self.coefficient_fixed_q.params[self.global_batch_step-1,i],
                                          collections=['params'])
                if self.variational():
                    self.coefficient_fixed_means = self.coefficient_fixed_q.mean()
                else:
                    self.coefficient_fixed_means = self.coefficient_fixed_q.params[self.global_batch_step-1]
                self.inference_map[self.coefficient_fixed] = self.coefficient_fixed_q
                fixef_ix = names2ix(f.fixed_coefficient_names, f.coefficient_names)
                coefficient_fixed_mask = np.zeros(len(f.coefficient_names), dtype=getattr(np, self.float_type))
                coefficient_fixed_mask[fixef_ix] = 1.
                coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)
                self.coefficient_fixed *= coefficient_fixed_mask
                self.coefficient_fixed_means *= coefficient_fixed_mask
                self.coefficient = tf.expand_dims(self.coefficient_fixed, 0)
                self.ransl = False

                for i in range(len(f.ran_names)):
                    r = f.random[f.ran_names[i]]
                    mask_row_np = np.ones(self.rangf_n_levels[i], dtype=getattr(np, self.float_type))
                    mask_row_np[self.rangf_n_levels[i] - 1] = 0
                    mask_row = tf.constant(mask_row_np)

                    if r.intercept:
                        intercept_random = Normal(
                            loc=tf.zeros([self.rangf_n_levels[i]], dtype=self.FLOAT_TF),
                            scale=self.coef_prior_sd,
                            name='intercept_by_%s' % r.gf
                        )
                        if self.variational():
                            intercept_random_q = Normal(
                                loc=tf.Variable(
                                    tf.random_normal([self.rangf_n_levels[i]], stddev=self.coef_prior_sd),
                                    dtype=self.FLOAT_TF,
                                    name='intercept_q_loc_by_%s' % r.gf
                                ),
                                scale=tf.nn.softplus(
                                    tf.Variable(
                                        tf.random_normal([self.rangf_n_levels[i]], mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd), stddev=self.coef_prior_sd),
                                        dtype=self.FLOAT_TF,
                                        name='intercept_q_scale_by_%s' % r.gf
                                    )
                                ),
                                name='intercept_q_by_%s' % r.gf
                            )
                            if self.log_random:
                                intercept_random_log = intercept_random_q.mean()
                        else:
                            intercept_random_q = Empirical(
                                params=tf.Variable(tf.zeros((self.n_samples, self.rangf_n_levels[i]))),
                                name='intercept_q_by_%s' % r.gf
                            )
                            if self.inference_name == 'MetropolisHastings':
                                intercept_random_proposal = Normal(
                                    loc=intercept_random,
                                    scale=self.coef_prior_sd,
                                    name='intercept_proposal_by_%s' % r.gf
                                )
                                self.proposal_map[intercept_random] = intercept_random_proposal
                            if self.log_random:
                                intercept_random_log = intercept_random_q.params[self.global_batch_step-1]
                        self.inference_map[intercept_random] = intercept_random_q
                        intercept_random *= mask_row
                        intercept_random -= tf.reduce_mean(intercept_random, axis=0)
                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])

                        if self.log_random:
                            intercept_random_log *= mask_row
                            intercept_random_log -= tf.reduce_mean(intercept_random_log, axis=0)

                            tf.summary.histogram(
                                'by_%s/intercept' % r.gf,
                                intercept_random_log,
                                collections=['random']
                            )

                    if len(r.coefficient_names) > 0:
                        coefs = r.coefficient_names
                        coef_ix = names2ix(coefs, f.coefficient_names)
                        mask_col_np = np.zeros(len(f.coefficient_names))
                        mask_col_np[coef_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)
                        self.ransl = True
                        coefficient_random = Normal(
                            loc=tf.zeros([self.rangf_n_levels[i], len(f.coefficient_names)], dtype=self.FLOAT_TF),
                            scale=self.coef_prior_sd,
                            name='coefficient_by_%s' %r.gf
                        )
                        if self.variational():
                            coefficient_random_q = Normal(
                                loc=tf.Variable(
                                    tf.random_normal([self.rangf_n_levels[i], len(f.coefficient_names)], stddev=self.coef_prior_sd),
                                    dtype=self.FLOAT_TF,
                                    name='coefficient_q_loc_by_%s' %r.gf
                                ),
                                scale = tf.nn.softplus(
                                    tf.Variable(
                                        tf.random_normal([self.rangf_n_levels[i], len(f.coefficient_names)], mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd), stddev=self.coef_prior_sd),
                                        dtype=self.FLOAT_TF,
                                        name='coefficient_q_scale_by_%s' % r.gf
                                    )
                                ),
                                name = 'coefficient_q_by_%s' %r.gf
                            )
                            if self.log_random:
                                coefficient_random_log = tf.gather(coefficient_random_q.mean(), coef_ix, axis=1)
                        else:
                            coefficient_random_q = Empirical(
                                params=tf.Variable(tf.zeros((self.n_samples, self.rangf_n_levels[i], len(f.coefficient_names)))),
                                name='coefficient_q_by_%s' %r.gf
                            )
                            if self.inference_name == 'MetropolisHastings':
                                coefficient_random_proposal = Normal(
                                    loc=coefficient_random,
                                    scale=self.coef_prior_sd,
                                    name='coefficient_proposal_by_%s' % r.gf
                                )
                                self.proposal_map[coefficient_random] = coefficient_random_proposal
                            if self.log_random:
                                coefficient_random_log = tf.gather(coefficient_random_q.params[self.global_batch_step-1], coef_ix, axis=1)
                        self.inference_map[coefficient_random] = coefficient_random_q

                        coefficient_random *= mask_col
                        coefficient_random *= tf.expand_dims(mask_row, -1)
                        coefficient_random -= tf.reduce_mean(coefficient_random, axis=0)
                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            coefficient_random_log *= mask_col
                            coefficient_random_log *= tf.expand_dims(mask_row, -1)
                            coefficient_random_log -= tf.reduce_mean(coefficient_random_log, axis=0)

                            for j in range(len(r.coefficient_names)):
                                coef_name = r.coefficient_names[j]
                                ix = coef_ix[j]
                                tf.summary.histogram(
                                    'by_%s/coefficient/%s' % (r.gf, coef_name),
                                    coefficient_random_log[:,ix],
                                    collections=['random']
                                )

    def initialize_irf_params(self):
        f = self.form
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for x in f.atomic_irf_by_family:
                    p, q = self.initialize_irf_params_inner(x, sorted(f.atomic_irf_by_family[x]))
                    self.atomic_irf_by_family[x] = tf.stack(p, axis=0)
                    self.atomic_irf_means_by_family[x] = tf.stack(q, axis=0)

    def initialize_irf_params_inner(self, family, ids):
        ## Infinitessimal value to add to bounded parameters
        epsilon = 1e-35  # np.nextafter(0, 1, dtype=getattr(np, self.float_type)) * 10
        dim = len(ids)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if family == 'DiracDelta':
                    filler = tf.expand_dims(tf.constant(1., shape=[1, dim]), -1)
                    return (filler,), (filler,)
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
                    mu = Normal(loc=tf.zeros([dim]), scale=self.conv_prior_sd, name=sn('mu_%s' % '-'.join(ids)))
                    sigma = Normal(loc=tf.zeros([dim]), scale=1., name=sn('sigma_%s' % '-'.join(ids)))
                    if self.variational():
                        mu_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], stddev=self.conv_prior_sd),
                                dtype=self.FLOAT_TF,
                                name=sn('mu_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], mean=tf.contrib.distributions.softplus_inverse(self.conv_prior_sd), stddev=self.conv_prior_sd),
                                    dtype=self.FLOAT_TF,
                                    name=sn('mu_q_scale_%s' % '-'.join(ids)))
                            ),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], mean=tf.contrib.distributions.softplus_inverse(1.), stddev=self.conv_prior_sd),
                                dtype=self.FLOAT_TF,
                                name=sn('sigma_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], mean=tf.contrib.distributions.softplus_inverse(self.conv_prior_sd), stddev=self.conv_prior_sd),
                                    dtype=self.FLOAT_TF,
                                    name=sn('sigma_q_scale_%s' % '-'.join(ids))
                                )
                            ),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.mean()[i], collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i], tf.nn.softplus(sigma_q.mean()[i]),
                                              collections=['params'])
                    else:
                        mu_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        if self.inference_name == 'MetropolisHastings':
                            mu_proposal = Normal(
                                loc=mu,
                                scale=self.conv_prior_sd,
                                name=sn('mu_proposal_%s' % '-'.join(ids))
                            )
                            sigma_proposal = Normal(
                                loc=sigma,
                                scale=self.conv_prior_sd,
                                name=sn('sigma_proposal_%s' % '-'.join(ids))
                            )
                            self.proposal_map[mu] = mu_proposal
                            self.proposal_map[sigma] = sigma_proposal
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.params[self.global_batch_step-1,i], collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i], tf.nn.softplus(sigma_q.params[self.global_batch_step-1,i]),
                                              collections=['params'])
                    self.inference_map[mu] = mu_q
                    self.inference_map[sigma] = sigma_q
                    if self.variational():
                        return (mu, tf.nn.softplus(sigma)), (mu_q.mean(), tf.nn.softplus(sigma_q.mean()))
                    return (mu, tf.nn.softplus(sigma)), (mu_q.params[self.global_batch_step-1], tf.nn.softplus(sigma_q.params[self.global_batch_step-1]))
                if family == 'SkewNormal':
                    mu = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('mu_%s' % '-'.join(ids))
                    )
                    sigma = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('sigma_%s' % '-'.join(ids))
                    )
                    alpha = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('alpha_%s' % '-'.join(ids))
                    )
                    if self.variational():
                        mu_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('mu_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('mu_q_scale_%s' % '-'.join(ids)))
                            ),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('sigma_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('sigma_q_scale_%s' % '-'.join(ids))
                                )
                            ),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        alpha_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('alpha_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('alpha_q_scale_%s' % '-'.join(ids))
                                )
                            ),
                            name=sn('alpha_q_%s' % '-'.join(ids))
                        )
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.mean()[i], collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i], tf.nn.softplus(sigma_q.mean()[i]),
                                              collections=['params'])
                            tf.summary.scalar('alpha' + '/%s' % ids[i], alpha_q.mean()[i], collections=['params'])
                    else:
                        mu_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        alpha_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('alpha_q_%s' % '-'.join(ids))
                        )
                        if self.inference_name == 'MetropolisHastings':
                            mu_proposal = Normal(
                                loc=mu,
                                scale=self.conv_prior_sd,
                                name=sn('mu_proposal_%s' % '-'.join(ids))
                            )
                            sigma_proposal = Normal(
                                loc=sigma,
                                scale=self.conv_prior_sd,
                                name=sn('sigma_proposal_%s' % '-'.join(ids))
                            )
                            alpha_proposal = Normal(
                                loc=sigma,
                                scale=self.conv_prior_sd,
                                name=sn('alpha_proposal_%s' % '-'.join(ids))
                            )
                            self.proposal_map[mu] = mu_proposal
                            self.proposal_map[sigma] = sigma_proposal
                            self.proposal_map[alpha] = alpha_proposal
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.params[self.global_batch_step - 1, i],
                                              collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i],
                                              tf.nn.softplus(sigma_q.params[self.global_batch_step - 1, i]),
                                              collections=['params'])
                            tf.summary.scalar('alpha' + '/%s' % ids[i], alpha_q.params[self.global_batch_step - 1, i],
                                              collections=['params'])
                    self.inference_map[mu] = mu_q
                    self.inference_map[sigma] = sigma_q
                    self.inference_map[alpha] = alpha_q
                    if self.variational():
                        return (mu, tf.nn.softplus(sigma), alpha), (mu_q.mean(), tf.nn.softplus(sigma_q.mean()), alpha_q.mean())
                    return (mu, tf.nn.softplus(sigma), alpha), (mu_q.params[self.global_batch_step - 1], tf.nn.softplus(sigma_q.params[self.global_batch_step - 1]), alpha_q.params[self.global_batch_step - 1])
                if family == 'EMG':
                    mu = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('mu_%s' % '-'.join(ids))
                    )
                    sigma = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('sigma_%s' % '-'.join(ids))
                    )
                    L = Normal(
                        loc=tf.zeros([dim]),
                        scale=tf.ones([dim]) * self.conv_prior_sd,
                        name=sn('L_%s' % '-'.join(ids))
                    )
                    if self.variational():
                        mu_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('mu_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('mu_q_scale_%s' % '-'.join(ids)))
                            ),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('sigma_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('sigma_q_scale_%s' % '-'.join(ids))
                                )
                            ),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        L_q = Normal(
                            loc=tf.Variable(
                                tf.random_normal([dim], dtype=self.FLOAT_TF),
                                name=sn('L_q_loc_%s' % '-'.join(ids))
                            ),
                            scale=tf.nn.softplus(
                                tf.Variable(
                                    tf.random_normal([dim], dtype=self.FLOAT_TF),
                                    name=sn('L_q_scale_%s' % '-'.join(ids))
                                )
                            ),
                            name=sn('L_q_%s' % '-'.join(ids))
                        )
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.mean()[i], collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i], tf.nn.softplus(sigma_q.mean()[i]),
                                              collections=['params'])
                            tf.summary.scalar('L' + '/%s' % ids[i], L_q.mean()[i], collections=['params'])
                    else:
                        mu_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('mu_q_%s' % '-'.join(ids))
                        )
                        sigma_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('sigma_q_%s' % '-'.join(ids))
                        )
                        L_q = Empirical(
                            params=tf.Variable(tf.zeros((self.n_samples, dim))),
                            name=sn('L_q_%s' % '-'.join(ids))
                        )
                        if self.inference_name == 'MetropolisHastings':
                            mu_proposal = Normal(
                                loc=mu,
                                scale=self.conv_prior_sd,
                                name=sn('mu_proposal_%s' % '-'.join(ids))
                            )
                            sigma_proposal = Normal(
                                loc=sigma,
                                scale=self.conv_prior_sd,
                                name=sn('sigma_proposal_%s' % '-'.join(ids))
                            )
                            L_proposal = Normal(
                                loc=sigma,
                                scale=self.conv_prior_sd,
                                name=sn('L_proposal_%s' % '-'.join(ids))
                            )
                            self.proposal_map[mu] = mu_proposal
                            self.proposal_map[sigma] = sigma_proposal
                            self.proposal_map[L] = L_proposal
                        for i in range(dim):
                            tf.summary.scalar('mu' + '/%s' % ids[i], mu_q.params[self.global_batch_step - 1, i],
                                              collections=['params'])
                            tf.summary.scalar('sigma' + '/%s' % ids[i],
                                              tf.nn.softplus(sigma_q.params[self.global_batch_step - 1, i]),
                                              collections=['params'])
                            tf.summary.scalar('L' + '/%s' % ids[i], L_q.params[self.global_batch_step - 1, i],
                                              collections=['params'])
                    self.inference_map[mu] = mu_q
                    self.inference_map[sigma] = sigma_q
                    self.inference_map[L] = L_q
                    if self.variational():
                        return (mu, tf.nn.softplus(sigma), tf.nn.softplus(L)), (mu_q.mean(), tf.nn.softplus(sigma_q.mean()), tf.nn.softplus(L_q.mean()))
                    return (mu, tf.nn.softplus(sigma), tf.nn.softplus(L)), (mu_q.params[self.global_batch_step - 1], tf.nn.softplus(sigma_q.params[self.global_batch_step - 1]), tf.nn.softplus(L_q.params[self.global_batch_step - 1]))
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

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.out = Normal(loc=self.out, scale=self.y_sigma_scale*self.y_sigma_init, name='output')
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        n_print=self.logging_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )
                elif self.inference_name == 'MetropolisHastings':
                    self.inference = getattr(ed, self.inference_name)(self.inference_map, self.proposal_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_print=self.logging_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )
                else:
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        step_size=0.0001, # 0.5 / self.minibatch_size,
                        # n_steps=2,
                        n_print=self.logging_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )

    def start_logging(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
                self.summary_params = tf.summary.merge_all(key='params')
                if self.log_random and len(f.random) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def fit(self,
            X,
            y,
            n_epoch_train=100,
            n_epoch_tune=100,
            irf_name_map=None,
            plot_x_inches=28,
            plot_y_inches=5,
            cmap='gist_earth'):

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

        y_rangf = y[f.rangf]
        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[f.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                pb = tf.contrib.keras.utils.Progbar(self.n_iter * self.n_train_minibatch)

                fd = {
                    self.X: X_3d,
                    self.time_X: time_X_3d,
                    self.y: y_dv,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                sys.stderr.write('Running %s inference...\n' % self.inference_name)
                metric_total = 0
                while self.global_step.eval(session=self.sess) < self.n_iter:
                    p, p_inv = getRandomPermutation(len(y))

                    metric_total = 0
                    for j in range(0, len(y), minibatch_size):
                        indices = p[j:j+minibatch_size]
                        fd_minibatch = {
                            self.X: X_3d[indices],
                            self.time_X: time_X_3d[indices],
                            self.y: y_dv[indices],
                            self.time_y: time_y[indices],
                            self.gf_y: gf_y[indices] if len(gf_y > 0) else gf_y
                        }

                        info_dict = self.inference.update(fd_minibatch)
                        metric_cur = info_dict['loss'] if self.variational() else info_dict['accept_rate']
                        if not np.isfinite(metric_cur):
                            metric_cur = 0
                        metric_total += metric_cur

                        if self.global_batch_step.eval(session=self.sess) % self.logging_freq == 0:
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_batch_step.eval(session=self.sess))

                        self.sess.run(self.incr_global_batch_step)
                        pb.update(self.global_batch_step.eval(session=self.sess), values=[('loss' if self.variational() else 'accept_rate', metric_cur)], force=True)


                    if not self.variational():
                        metric_total /= self.n_train_minibatch
                    if self.log_random and len(f.random) > 0:
                        summary_random = self.sess.run(self.summary_random)
                        self.writer.add_summary(summary_random, self.global_batch_step.eval(session=self.sess))

                    self.sess.run(self.incr_global_step)
                    if self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)
                        self.save()


                self.out_post = ed.copy(self.out, self.inference_map)

                self.inference.finalize()

                self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)


    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        sys.stderr.write('Sampling predictions from posterior...\n')

        f = self.form

        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, first_obs, last_obs)
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        preds = np.zeros((len(y_time), self.n_samples_eval))

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.X: X_3d,
                    self.time_X: time_X_3d,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                }

                for i in range(self.n_samples_eval):
                    sys.stderr.write('\r%d/%d'%(i+1, self.n_samples_eval))
                    if not np.isfinite(self.minibatch_size):
                        preds[:,i] = self.sess.run(self.out_post, feed_dict=fd)
                    else:
                        for j in range(0, len(y_time), self.minibatch_size):
                            fd_minibatch = {
                                self.X: X_3d[j:j+self.minibatch_size],
                                self.time_X: time_X_3d[j:j+self.minibatch_size],
                                self.time_y: time_y[j:j+self.minibatch_size],
                                self.gf_y: gf_y[j:j+self.minibatch_size] if len(gf_y) > 0 else gf_y
                            }
                            preds[j:j+self.minibatch_size,i] = self.sess.run(self.out_post, feed_dict=fd_minibatch)

                preds = preds.mean(axis=1)

                sys.stderr.write('\n\n')

                return preds

    def eval(self, X, y):
        f = self.form

        y_rangf = y[f.rangf]
        for i in range(len(f.rangf)):
            c = f.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_3d, time_X_3d = self.expand_history(X[f.terminal_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[f.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.X: X_3d,
                    self.time_X: time_X_3d,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.y: y_dv,
                    self.out_post: self.y
                }

                if not np.isfinite(self.minibatch_size):
                    n_minibatch = 1
                    mse, mae, logLik = ed.evaluate(['mse', 'mae', 'log_lik'], fd, n_samples=self.n_samples_eval)
                else:
                    n_minibatch = math.ceil(len(y)/self.minibatch_size)
                    mse = mae = logLik = 0
                    for j in range(0, len(y), self.minibatch_size):
                        sys.stderr.write('\r%d/%d' % (j + 1, n_minibatch))
                        fd_minibatch = {
                            self.X: X_3d[j:j+self.minibatch_size],
                            self.time_X: time_X_3d[j:j+self.minibatch_size],
                            self.time_y: time_y[j:j+self.minibatch_size],
                            self.gf_y: gf_y[j:j+self.minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[j:j+self.minibatch_size],
                            self.out_post: self.y
                        }
                        mse_cur, mae_cur, logLik_cur = ed.evaluate(['mse', 'mae', 'log_lik'], fd_minibatch, n_samples=self.n_samples_eval)
                        mse += mse_cur*len(fd_minibatch[self.y])
                        mae += mae_cur*len(fd_minibatch[self.y])
                        logLik += logLik_cur

                mse /= n_minibatch
                mae /= n_minibatch

                return mse, mae, logLik

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass
