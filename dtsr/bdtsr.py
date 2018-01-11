import os
import math
import time
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
    :param float_type: ``str``; the ``float`` type to use throughout the network.
    :param int_type: ``str``; the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: ``int`` or ``None``; the size of minibatches to use for fitting/prediction (full-batch if ``None``).
    :param log_random: ``bool``; whether to log random effects to Tensorboard.
    :param log_freq: ``int``; the frequency (in iterations) with which to log model params to Tensorboard.
    :param save_freq: ``int``; the frequency (in iterations) with which to save model checkpoints.
    :param inference_name: ``str``; the Edward inference class to use for fitting.
    :param n_samples: ``int``; the number of samples to draw from the variational posterior if using variational inference.
        If using MCMC, the number of samples is set deterministically as ``n_iter * n_minibatch``, so this user-supplied parameter is ignored.
    :param n_samples_eval: ``int``; the number of samples from the predictive posterior to use for evaluation/prediction.
    :param n_iter: ``int``; the number of iterations to perform in training.
        Must be supplied to ``__init__()`` because MCMC inferences hard-code this into the network structure.
    :param conv_prior_sd: ``float``; the standard deviation of the Normal prior on convolution parameters.
        Smaller values concentrate probability mass around the prior mean.
    :param coef_prior_sd: ``float``; the standard deviation of the Normal prior on coefficient parameters.
        Smaller values concentrate probability mass around the prior mean.
    :param y_sigma_scale: ``float``; the scaling coefficient on the standard deviation of the distribution of the dependent variable.
        Specifically, the DV is assumed to have the standard deviation ``stddev(y_train) * y_sigma_scale``.
        This setup allows the user to configure the output distribution independently of the units of the DV.
    """



    #####################################################
    #
    #  Native methods
    #
    ######################################################

    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 history_length=100,
                 low_memory=False,
                 float_type='float32',
                 int_type='int32',
                 minibatch_size=None,
                 log_random=True,
                 log_freq=1,
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
            save_freq=save_freq,
            log_freq=log_freq,
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
        self.n_samples_eval = n_samples_eval
        self.conv_prior_sd = float(conv_prior_sd)
        self.coef_prior_sd = float(coef_prior_sd)
        self.y_sigma_scale = float(y_sigma_scale)

        self.inference_map = {}
        if self.inference_name == 'MetropolisHastings':
            self.proposal_map = {}

        self.build()

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


    #####################################################
    #
    #  Private methods
    #
    ######################################################

    def __initialize_intercepts_coefficients__(self):
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
                            tf.random_normal(
                                [1],
                                mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd),
                                stddev=self.coef_prior_sd
                            ),
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
                            tf.random_normal(
                                [len(f.coefficient_names)],
                                mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd),
                                stddev=self.coef_prior_sd
                            ),
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
                    mask_row = tf.constant(mask_row_np, dtype=self.FLOAT_TF)

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
                                        tf.random_normal(
                                            [self.rangf_n_levels[i]],
                                            mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd),
                                            stddev=self.coef_prior_sd
                                        ),
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
                                        tf.random_normal(
                                            [self.rangf_n_levels[i], len(f.coefficient_names)],
                                            mean=tf.contrib.distributions.softplus_inverse(self.coef_prior_sd),
                                            stddev=self.coef_prior_sd
                                        ),
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

    def __new_irf_param__(self, param_name, ids, mean=0, lb=None, ub=None, ran_ids=None):
        epsilon = 1e-35
        dim = len(ids)
        mean = float(mean)
        if ran_ids is None:
            ran_ids = []

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if lb is None and ub is None:
                    # Unbounded support
                    param_mean_init = mean
                elif lb is not None and ub is None:
                    # Lower-bounded support only
                    try:
                        float(lb)
                    except:
                        raise ValueError('lb is not a valid number: %s' %lb)
                    param_mean_init = tf.contrib.distributions.softplus_inverse(mean - lb - epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    try:
                        float(ub)
                    except:
                        raise ValueError('ub is not a valid number: %s' %lb)
                    param_mean_init = tf.contrib.distributions.softplus_inverse(-(mean - ub + epsilon))
                else:
                    # Finite-interval bounded support
                    try:
                        float(lb)
                    except:
                        raise ValueError('lb is not a valid number: %s' %lb)
                    try:
                        float(ub)
                    except:
                        raise ValueError('ub is not a valid number: %s' %lb)
                    param_mean_init = tf.contrib.distributions.bijectors.Sigmoid.inverse(
                        (mean- lb - epsilon)  / ((ub-epsilon)-(lb+epsilon))
                    )

                param_mean_init *= tf.ones([dim])

                param = Normal(
                    loc=param_mean_init,
                    scale=self.conv_prior_sd,
                    name=sn('%s_%s' % (param_name, '-'.join(ids)))
                )
                if self.variational():
                    param_q = Normal(
                        loc=tf.Variable(
                            tf.random_normal(
                                [dim],
                                mean=param_mean_init,
                                stddev=self.conv_prior_sd
                            ),
                            dtype=self.FLOAT_TF,
                            name=sn('%s_q_loc_%s' % (param_name, '-'.join(ids)))
                        ),
                        scale=tf.nn.softplus(
                            tf.Variable(
                                tf.random_normal(
                                    [dim],
                                    mean=tf.contrib.distributions.softplus_inverse(self.conv_prior_sd),
                                    stddev=self.conv_prior_sd
                                ),
                                dtype=self.FLOAT_TF,
                                name=sn('%s_q_scale_%s' % (param_name, '-'.join(ids)))
                            )
                        ),
                        name=sn('%s_q_%s' % (param_name, '-'.join(ids)))
                    )

                    if lb is None and ub is None:
                        param_mean = param_q.mean()
                    elif lb is not None and ub is None:
                        param_mean = tf.nn.softplus(param_q.mean()) + lb + epsilon
                    elif lb is None and ub is not None:
                        param_mean = -tf.nn.softplus(param_q.mean()) + ub - epsilon
                    else:
                        param_mean = tf.sigmoid(param_q.mean()) * ((ub-epsilon)-(lb+epsilon)) + lb + epsilon

                    for i in range(dim):
                        tf.summary.scalar(
                            sn('%s/%s' % (param_name, ids[i])),
                            param_mean[i],
                            collections=['params']
                        )
                else:
                    param_q = Empirical(
                        params=tf.Variable(tf.zeros((self.n_samples, dim))),
                        name=sn('%s_q_%s' % (param_name, '-'.join(ids)))
                    )
                    if self.inference_name == 'MetropolisHastings':
                        L_proposal = Normal(
                            loc=param,
                            scale=self.conv_prior_sd,
                            name=sn('%s_proposal_%s' % (param_name, '-'.join(ids)))
                        )
                        self.proposal_map[param] = L_proposal

                    if lb is None and ub is None:
                        param_mean = param_q.params[self.global_batch_step - 1]
                    elif lb is not None and ub is None:
                        param_mean = tf.nn.softplus(param_q.params[self.global_batch_step - 1]) + lb + epsilon
                    elif lb is None and ub is not None:
                        param_mean = -tf.nn.softplus(param_q.params[self.global_batch_step - 1]) + ub - epsilon
                    else:
                        param_mean = tf.sigmoid(param_q.params[self.global_batch_step - 1]) * ((ub-epsilon)-(lb+epsilon)) + lb + epsilon

                    for i in range(dim):
                        tf.summary.scalar(
                            sn('%s/%s' % (param_name, ids[i])),
                            param_mean[i],
                            collections=['params']
                        )

                if lb is None and ub is None:
                    param_out = param
                elif lb is not None and ub is None:
                    param_out = tf.nn.softplus(param) + lb + epsilon
                elif lb is None and ub is not None:
                    param_out = -tf.nn.softplus(param) + ub - epsilon
                else:
                    param_out = tf.sigmoid(param) * ((ub-epsilon) - (lb+epsilon)) + lb + epsilon

                self.inference_map[param] = param_q

                return (param_out, param_mean)

    def __initialize_objective__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.out = Normal(loc=self.out, scale=self.y_sigma_scale*self.y_sigma_init, name='output')
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )
                elif self.inference_name == 'MetropolisHastings':
                    self.inference = getattr(ed, self.inference_name)(self.inference_map, self.proposal_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )
                else:
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        step_size=0.0001, # 0.5 / self.minibatch_size,
                        # n_steps=2,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )

    def __start_logging__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
                self.summary_params = tf.summary.merge_all(key='params')
                if self.log_random and len(f.random) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')




    #####################################################
    #
    #  Public methods
    #
    ######################################################

    def build(self, restore=True, verbose=True):
        """
        Construct the DTSR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

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

        self.__initialize_inputs__()
        self.__initialize_intercepts_coefficients__()
        self.__initialize_irf_lambdas__()
        self.__initialize_irf_params__()
        self.__initialize_irfs__(self.irf_tree)
        self.__construct_network__()
        self.__initialize_objective__()
        self.__start_logging__()
        self.__initialize_saver__()
        self.load(restore=restore)
        self.__report_n_params__()

    def variational(self):
        """
        Check whether the DTSR model uses variational Bayes.

        :return: ``True`` if the model is variational, ``False`` otherwise.
        """
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
        return super(BDTSR, self).expand_history(X, X_time, first_obs, last_obs)

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
            This must hold for MCMC inference, since the number of minibatches is built into the model architecture.
            However, it is not necessary for variational inference.
        :param n_epoch_train: ``int``; the number of training iterations
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
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
                    self.y: y_dv,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }

                if self.global_step.eval(session=self.sess) == 0:
                    summary_params = self.sess.run(self.summary_params)
                    self.writer.add_summary(summary_params, self.global_batch_step.eval(session=self.sess))
                    if self.log_random and len(f.random) > 0:
                        summary_random = self.sess.run(self.summary_random)
                        self.writer.add_summary(summary_random, self.global_batch_step.eval(session=self.sess))

                sys.stderr.write('Running %s inference...\n' % self.inference_name)
                while self.global_step.eval(session=self.sess) < self.n_iter:
                    pb = tf.contrib.keras.utils.Progbar(self.n_train_minibatch)
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    p, p_inv = getRandomPermutation(len(y))

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

                        self.sess.run(self.incr_global_batch_step)
                        pb.update((j/minibatch_size)+1, values=[('loss' if self.variational() else 'accept_rate', metric_cur)], force=True)

                    self.sess.run(self.incr_global_step)
                    if self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        self.save()
                        summary_params = self.sess.run(self.summary_params)
                        self.writer.add_summary(summary_params, self.global_batch_step.eval(session=self.sess))
                        if self.log_random and len(f.random) > 0:
                            summary_random = self.sess.run(self.summary_random)
                            self.writer.add_summary(summary_random, self.global_batch_step.eval(session=self.sess))
                        self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)

                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                self.out_post = ed.copy(self.out, self.inference_map)

                self.inference.finalize()

                self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)


    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        """
        Predict from the pre-trained DTSR model.
        Predictions are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.

        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization

        :param y_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the regression targets, grouped by series.
        :param y_rangf: ``pandas`` ``Series`` or 1D ``numpy`` array; random grouping factor values (if applicable).
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param first_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        assert len(y_time) == len(y_rangf) == len(first_obs) == len(last_obs), 'y_time, y_rangf, first_obs, and last_obs must be of identical length. Got: len(y_time) = %d, len(y_rangf) = %d, len(first_obs) = %d, len(last_obs) = %d' % (len(y_time), len(y_rangf), len(first_obs), len(last_obs))

        sys.stderr.write('Sampling from predictive posterior...\n')

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
        """
        Evaluate the pre-trained DTSR model.
        Metrics are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.

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

        :return: ``tuple``; three floats ``(mse, mae, logLik)`` for the evaluation data.
        """

        sys.stderr.write('Sampling from predictive posterior...\n')

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
                        sys.stderr.write('\r%d/%d' % ((j/self.minibatch_size)+1, n_minibatch))
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

                mse /= len(y)
                mae /= len(y)

                return mse, mae, logLik

    def make_plots(self, irf_name_map=None, plot_x_inches=7., plot_y_inches=5., cmap=None):
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

        For simplicity, plots for BDTSR models use the posterior mean, abstracting away from other characteristics of the posterior distribution (e.g. variance).

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """
        return super(BDTSR, self).make_plots(
            irf_name_map=irf_name_map,
            plot_x_inches=plot_x_inches,
            plot_y_inches=plot_y_inches,
            cmap=cmap
        )