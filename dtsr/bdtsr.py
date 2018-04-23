import os
from collections import defaultdict
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
from .dtsr import DTSR

import edward as ed
from edward.models import Normal, Gamma, Exponential, MultivariateNormalTriL, SinhArcsinh, Empirical


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
        *NOTE*: Support for ``low_memory`` (and consequently unbounded history length) has been temporarily suspended, since the high-memory implementation is generally much faster and for many projects model fit is not greatly affected by history truncation. If your project demands unbounded history modeling, please let the developers know on Github so that they can prioritize re-implementing this feature.
    :param float_type: ``str``; the ``float`` type to use throughout the network.
    :param int_type: ``str``; the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: ``int`` or ``None``; the size of minibatches to use for fitting/prediction (full-batch if ``None``).
    :param log_random: ``bool``; whether to log random effects to Tensorboard.
    :param log_freq: ``int``; the frequency (in iterations) with which to log model params to Tensorboard.
    :param save_freq: ``int``; the frequency (in iterations) with which to save model checkpoints.
    :param optim: ``str``; the name of the optimizer to use. Choose from ``'SGD'``, ``'AdaGrad'``, ``'AdaDelta'``, ``'Adam'``, ``'FTRL'``, ``'RMSProp'``, ``'Nadam'``.
        **Note**: This parameter is only used for variational inference. Other inferences will ignore it.
    :param learning_rate: ``float``; the initial value for the learning rate.
        **Note**: This parameter is only used for variational inference. Other inferences will ignore it.
    :param learning_rate_decay_factor: ``float``; rate parameter to the learning rate decay schedule (if applicable).
        **Note**: This parameter is only used for variational inference. Other inferences will ignore it.
    :param learning_rate_decay_family: ``str``; the functional family for the learning rate decay schedule (if applicable).
        Choose from the following, where :math:`\lambda` is the current learning rate, :math:`\lambda_0` is the initial learning rate, :math:`\delta` is the ``learning_rate_decay_factor``, and :math:`i` is the iteration index.

        * ``'linear'``: :math:`\\lambda_0 \\cdot ( 1 - \\delta \\cdot i )`
        * ``'inverse'``: :math:`\\frac{\\lambda_0}{1 + ( \\delta \\cdot i )}`
        * ``'exponential'``: :math:`\\lambda = \\lambda_0 \\cdot ( 2^{-\\delta \\cdot i} )`
        * ``'stepdownXX'``: where ``XX`` is replaced by an integer representing the stepdown interval :math:`a`: :math:`\\lambda = \\lambda_0 * 2^{\\left \\lfloor \\frac{i}{a} \\right \\rfloor}`

        **Note**: This parameter is only used for variational inference. Other inferences will ignore it.
    :param learning_rate_min: ``float``; the minimum value for the learning rate.
        If the decay schedule would take the learning rate below this point, learning rate clipping will occur.
        **Note**: This parameter is only used for variational inference. Other inferences will ignore it.
    :param init_sd: ``float``; standard deviation of truncated normal parameter initializer
    :param ema_decay: ``float``; decay factor to use for exponential moving average for parameters (used in prediction)
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
    :param asymmetric_error: ``bool``; whether to apply the ``SinhArcsinh`` transform to the normal error, allowing fitting of skewness and tailweight
    :param log_graph: ``bool``; whether to log the network graph to Tensorboard
    """



    #####################################################
    #
    #  Native methods
    #
    #####################################################

    def __init__(
            self,
            form_str,
            X,
            y,
            outdir='./dtsr_model/',
            history_length=100,
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
            ema_decay=0.999,
            mh_proposal_sd=1.,
            inference_name='KLqp',
            n_samples=None,
            n_samples_eval=100,
            n_iter=1000,
            intercept_prior_sd=1,
            coef_prior_sd=1,
            conv_prior_sd=1,
            mv=False,
            mv_ran=False,
            y_scale_fixed=None,
            y_scale_prior_sd=1,
            asymmetric_error=False,
            log_graph=True
    ):

        super(BDTSR, self).__init__(
            form_str,
            X,
            y,
            history_length=history_length,
            low_memory=low_memory,
            pc=pc,
            float_type=float_type,
            int_type=int_type,
            minibatch_size=minibatch_size,
            eval_minibatch_size=eval_minibatch_size,
            save_freq=save_freq,
            log_freq=log_freq,
            log_random = log_random,
            optim=optim,
            learning_rate=learning_rate,
            learning_rate_min=learning_rate_min,
            lr_decay_family=lr_decay_family,
            lr_decay_steps=lr_decay_steps,
            lr_decay_rate=lr_decay_rate,
            lr_decay_staircase=lr_decay_staircase,
            init_sd=init_sd,
            ema_decay=ema_decay,
            log_graph=log_graph
        )

        self.mv = mv
        self.mv_ran = mv_ran
        self.inference_name = inference_name
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.mh_proposal_sd = mh_proposal_sd
        self.coef_prior_sd = coef_prior_sd
        self.conv_prior_sd = conv_prior_sd
        self.intercept_prior_sd = intercept_prior_sd
        self.y_scale_fixed = y_scale_fixed
        self.y_scale_prior_sd = y_scale_prior_sd
        self.asymmetric_error = asymmetric_error

        assert not self.low_memory, 'Because Edward does not support Tensorflow control ops, ' \
                                    'low_memory is not supported in BDTSR'
        try:
            float(self.history_length)
        except:
            raise ValueError('Because Edward does not support Tensorflow control ops, '
                             'finite history_length must be specified in BDTSR')


        if not self.variational():
            if self.n_samples is not None:
                sys.stderr.write('Parameter n_samples being overridden for sampling optimization\n')
            self.n_samples = self.n_iter*self.n_train_minibatch
        self.n_samples_eval = n_samples_eval

        self.inference_map = {}
        if self.inference_name == 'MetropolisHastings':
            self.proposal_map = {}
            if self.mh_proposal_sd is None:
                self.mh_proposal_sd = 1.

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.intercept_prior_sd_tf = tf.constant(float(self.intercept_prior_sd), dtype=self.FLOAT_TF)
                self.coef_prior_sd_tf = tf.constant(float(self.coef_prior_sd), dtype=self.FLOAT_TF)
                self.conv_prior_sd_tf = tf.constant(float(self.conv_prior_sd), dtype=self.FLOAT_TF)
                self.y_scale_prior_sd_tf = tf.constant(float(self.y_scale_prior_sd), dtype=self.FLOAT_TF)
                if self.y_scale_fixed is None:
                    self.y_scale_fixed_tf = self.y_scale_fixed
                else:
                    self.y_scale_fixed_tf = tf.constant(float(self.y_scale_fixed), dtype=self.FLOAT_TF)
                self.intercept_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.intercept_prior_sd_tf)
                self.coef_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.coef_prior_sd_tf)
                self.conv_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.conv_prior_sd_tf)
                self.y_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_init_tf)
                self.y_scale_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_prior_sd_tf)

        if self.mv:
            self.__initialize_full_joint__()

        self.build(outdir)


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
            self.n_train,
            self.n_iter,
            self.y_mu_init,
            self.y_scale_init,
            self.y_scale_fixed,
            self.y_scale_prior_sd,
            self.rangf_map_base,
            self.rangf_n_levels,
            self.mv,
            self.mv_ran,
            self.inference_name,
            self.n_samples,
            self.n_samples_eval,
            self.mh_proposal_sd,
            self.coef_prior_sd,
            self.conv_prior_sd,
            self.intercept_prior_sd,
            self.y_scale_prior_sd,
            self.asymmetric_error,
            self.ema_decay
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
        self.n_train, \
        self.n_iter, \
        self.y_mu_init, \
        self.y_scale_init, \
        self.y_scale_fixed, \
        self.y_scale_prior_sd, \
        self.rangf_map_base, \
        self.rangf_n_levels, \
        self.mv, \
        self.mv_ran, \
        self.inference_name, \
        self.n_samples, \
        self.n_samples_eval, \
        self.mh_proposal_sd, \
        self.coef_prior_sd, \
        self.conv_prior_sd, \
        self.intercept_prior_sd, \
        self.y_scale_prior_sd, \
        self.asymmetric_error, \
        self.ema_decay = state

        self.regularizer_name = None
        self.regularizer_scale = 0
        self.log_graph=False

        self.__initialize_metadata__()

        self.rangf_map = []
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(defaultdict(lambda: self.rangf_n_levels[i], self.rangf_map_base[i]))

        self.inference_map = {}

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.y_mu_init_tf = tf.constant(self.y_mu_init, dtype=self.FLOAT_TF)
                self.y_scale_init_tf = tf.constant(self.y_scale_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(1e-35, dtype=self.FLOAT_TF)
                self.intercept_prior_sd_tf = tf.constant(float(self.intercept_prior_sd), dtype=self.FLOAT_TF)
                self.coef_prior_sd_tf = tf.constant(float(self.coef_prior_sd), dtype=self.FLOAT_TF)
                self.conv_prior_sd_tf = tf.constant(float(self.conv_prior_sd), dtype=self.FLOAT_TF)
                self.y_scale_prior_sd_tf = tf.constant(float(self.y_scale_prior_sd), dtype=self.FLOAT_TF)
                if self.y_scale_fixed is not None:
                    self.y_scale_fixed_tf = tf.constant(float(self.y_scale_fixed), dtype=self.FLOAT_TF)
                else:
                    self.y_scale_fixed_tf = self.y_scale_fixed
                self.intercept_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.intercept_prior_sd_tf)
                self.coef_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.coef_prior_sd_tf)
                self.conv_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.conv_prior_sd_tf)
                self.y_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_init_tf)
                self.y_scale_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_prior_sd_tf)

        if self.mv:
            self.__initialize_full_joint__()


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def __initialize_param_list__(self):
        name = []
        mean_init = []
        sd_init = []
        if self.has_intercept[None]:
            name.append('intercept')
            mean_init.append(self.y_mu_init_tf)
            sd_init.append(self.intercept_scale_mean_init)
        for i in range(len(self.coef_names)):
            coef = self.coef_names[i]
            name.append(coef)
            mean_init.append(0.)
            sd_init.append(self.coef_scale_mean_init)
        if self.mv_ran:
            for i in range(len(self.rangf)):
                gf = self.rangf[i]
                levels = list(range(self.rangf_n_levels[i]))
                if self.has_intercept[gf]:
                    name += ['intercept_by_%s_%s' % (gf, j) for j in levels]
                    mean_init += [0.] * self.rangf_n_levels[i]
                    sd_init += [self.intercept_scale_mean_init] * self.rangf_n_levels[i]
                for coef in self.coef_names:
                    name += ['%s_by_%s_%s' % (coef, gf, j) for j in levels]
                    mean_init += [0.] * self.rangf_n_levels[i]
                    sd_init += [self.coef_scale_mean_init] * self.rangf_n_levels[i]

        name_conv, mean_init_conv, sd_init_conv = self.__initialize_conv_param_list__()

        name += name_conv
        mean_init += mean_init_conv
        sd_init += sd_init_conv

        if self.y_scale_fixed is None:
            name.append('y_scale')
            mean_init.append(self.y_scale_init_tf)
            sd_init.append(self.y_scale_prior_sd_tf)

        assert len(name) == len(mean_init) == len(sd_init), 'Error: lengths of computed lists of parameter names, means, and sds do not match'

        return (name, mean_init, sd_init)

    def __initialize_conv_param_list__(self):
        name = []
        mean_init = []
        sd_init = []

        for family in self.atomic_irf_names_by_family:
            if family == 'DiracDelta':
                continue

            irf_ids = self.atomic_irf_names_by_family[family]

            irf_by_rangf = {}
            for id in irf_ids:
                for gf in self.irf_by_rangf:
                    if id in self.irf_by_rangf[gf]:
                        if gf not in irf_by_rangf:
                            irf_by_rangf[gf] = []
                        irf_by_rangf[gf].append(id)

            if family == 'Exp':
                param_name = ['L']
                L, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [L]
                param_sd = [self.conv_prior_sd_tf]
            elif family == 'SteepExp':
                param_name = ['L']
                L, _, _ = self.__process_mean__(100, lb=0)
                param_mean = [L]
                param_sd = [self.conv_prior_sd_tf]
            elif family in ['Gamma', 'GammaKgt1']:
                param_name = ['k', 'theta']
                k_theta, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [k_theta, k_theta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family in ['ShiftedGamma', 'ShiftedGammaKgt1']:
                param_name = ['k', 'theta', 'delta']
                k_theta, _, _ = self.__process_mean__(1, lb=0)
                delta, _, _ = self.__process_mean__(-1, ub=0)
                param_mean = [k_theta, k_theta, delta]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'Normal':
                param_name = ['mu', 'sigma']
                sigma, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [0., sigma]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'SkewNormal':
                param_name = ['mu', 'sigma', 'alpha']
                sigma, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [0., sigma, 0.]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'EMG':
                param_name = ['mu', 'sigma', 'L']
                sigma_L, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [0., sigma_L, sigma_L]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'BetaPrime':
                param_name = ['alpha', 'beta']
                alpha_beta, _, _ = self.__process_mean__(1, lb=0)
                param_mean = [alpha_beta, alpha_beta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'ShiftedBetaPrime':
                param_name = ['alpha', 'beta', 'delta']
                alpha_beta, _, _ = self.__process_mean__(1, lb=0)
                delta, _, _ = self.__process_mean__(-1, ub=0)
                param_mean = [alpha_beta, alpha_beta, delta]
                param_sd = [self.conv_prior_sd_tf] * 3
            for id in irf_ids:
                name += ['%s_%s' % (p, id) for p in param_name]
                mean_init += param_mean
                sd_init += param_sd
                if self.mv_ran:
                    for i in range(len(self.rangf)):
                        gf = self.rangf[i]
                        if gf in irf_by_rangf:
                            levels = list(range(self.rangf_n_levels[i]))
                            name += ['%s_%s_by_%s_%s' % (p, id, gf, j) for j in levels for p in param_name]
                            mean_init += [0.] * self.rangf_n_levels[i] * len(param_name)
                            sd_init += [self.conv_scale_mean_init] * self.rangf_n_levels[i] * len(param_name)

        return (name, mean_init, sd_init)

    def __initialize_full_joint__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                names, means, sds = self.__initialize_param_list__()
                self.full_joint_names = names
                self.full_joint_mu = tf.stack(means, 0)
                self.full_joint_sigma = tf.stack(sds, 0)

                self.full_joint = MultivariateNormalTriL(
                    loc=self.full_joint_mu,
                    scale_tril=tf.diag(self.full_joint_sigma),
                    name='full_joint'
                )

                if self.variational():
                    full_joint_q_loc = tf.Variable(
                        tf.random_normal(
                            [len(means)],
                            mean=self.full_joint_mu,
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='full_joint_q_loc'
                    )

                    full_joint_q_scale = tf.Variable(
                        tf.random_normal(
                            [len(sds), len(sds)],
                            mean=tf.diag(self.full_joint_sigma),
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='full_joint_q_scale'
                    )

                    self.full_joint_q = MultivariateNormalTriL(
                        loc=full_joint_q_loc,
                        scale_tril=tf.nn.softplus(full_joint_q_scale),
                        name='full_joint_q'
                    )

                else:
                    full_joint_q_samples = tf.Variable(
                        tf.ones((self.n_samples, len(means)), dtype=self.FLOAT_TF),
                        name='full_joint_q_samples'
                    )
                    self.full_joint_q = Empirical(
                        params=full_joint_q_samples,
                        name='full_joint_q'
                    )
                    if self.inference_name == 'MetropolisHastings':
                        self.full_joint_proposal = Normal(
                            loc=self.full_joint,
                            scale=self.mh_proposal_sd,
                            name='full_joint_proposal'
                        )
                        self.proposal_map[self.full_joint] = self.full_joint_proposal

                self.inference_map[self.full_joint] = self.full_joint_q

    def __initialize_intercept__(self, ran_gf=None, rangf_n_levels=None):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.mv:
                        ix = names2ix('intercept', self.full_joint_names)
                        assert len(ix) == 1, 'There should be exactly 1 parameter called "intercept"'
                        ix = ix[0]
                        intercept = self.full_joint[ix]
                        intercept_summary = self.full_joint_q.mean()[ix]
                    else:
                        intercept = Normal(
                            sample_shape=[],
                            loc=self.y_mu_init_tf,
                            scale=self.intercept_prior_sd_tf,
                            name='intercept'
                        )
                        if self.variational():
                            intercept_q_loc = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=self.y_mu_init_tf,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_loc'
                            )

                            intercept_q_scale = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=self.intercept_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_scale'
                            )

                            intercept_q = Normal(
                                loc=intercept_q_loc,
                                scale=tf.nn.softplus(intercept_q_scale),
                                name='intercept_q'
                            )
                            intercept_summary = intercept_q.mean()
                        else:
                            intercept_q_samples = tf.Variable(
                                tf.ones((self.n_samples), dtype=self.FLOAT_TF) * self.y_mu_init_tf,
                                name='intercept_q_samples'
                            )

                            intercept_q = Empirical(
                                params=intercept_q_samples,
                                name='intercept_q'
                            )
                            if self.inference_name == 'MetropolisHastings':
                                intercept_proposal = Normal(
                                    loc=intercept,
                                    scale=self.mh_proposal_sd,
                                    name='intercept_proposal'
                                )
                                self.proposal_map[intercept] = intercept_proposal
                            intercept_summary = intercept_q.params[self.global_batch_step - 1]

                        self.inference_map[intercept] = intercept_q
                else:
                    if self.mv_ran:
                        names = ['intercept_by_%s_%s' %(ran_gf, i) for i in range(self.rangf_n_levels[self.rangf.index(ran_gf)])]
                        ix = names2ix(names, self.full_joint_names)
                        intercept = tf.gather(self.full_joint, ix)
                        intercept_summary = tf.gather(self.full_joint_q.mean(), ix)
                    else:
                        intercept = Normal(
                            sample_shape=[rangf_n_levels],
                            loc=0.,
                            scale=self.intercept_prior_sd_tf,
                            name='intercept_by_%s' % ran_gf
                        )
                        if self.variational():
                            intercept_q_loc = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels],
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_loc_by_%s' % ran_gf
                            )

                            intercept_q_scale = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels],
                                    mean=self.intercept_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_scale_by_%s' % ran_gf
                            )

                            intercept_q = Normal(
                                loc=intercept_q_loc,
                                scale=tf.nn.softplus(intercept_q_scale),
                                name='intercept_q_by_%s' % ran_gf
                            )

                            intercept_summary = intercept_q.mean()
                        else:
                            intercept_q_ran_samples = tf.Variable(
                                tf.zeros((self.n_samples, rangf_n_levels), dtype=self.FLOAT_TF),
                                name='intercept_q_by_%s_samples' % ran_gf
                            )
                            intercept_q = Empirical(
                                params=intercept_q_ran_samples,
                                name='intercept_q_by_%s' % ran_gf
                            )
                            if self.inference_name == 'MetropolisHastings':
                                intercept_proposal = Normal(
                                    loc=intercept,
                                    scale=self.mh_proposal_sd,
                                    name='intercept_proposal_by_%s' % ran_gf
                                )
                                self.proposal_map[intercept] = intercept_proposal
                            intercept_summary = intercept_q.params[self.global_batch_step - 1]

                        self.inference_map[intercept] = intercept_q

                return intercept, intercept_summary

    def __initialize_coefficient__(self, coef_ids=None, ran_gf=None, rangf_n_levels=None):
        f = self.form

        if coef_ids is None:
            coef_ids = f.coefficient_names

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.mv:
                        ix = names2ix(self.coef_names, self.full_joint_names)
                        coefficient = tf.gather(self.full_joint, ix)
                        coefficient_summary = tf.gather(self.full_joint_q.mean(), ix)
                    else:
                        coefficient = Normal(
                            sample_shape=[len(coef_ids)],
                            loc=0.,
                            scale=self.coef_prior_sd_tf,
                            name='coefficient'
                        )
                        if self.variational():
                            coefficient_q_loc = tf.Variable(
                                tf.random_normal(
                                    [len(coef_ids)],
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='coefficient_q_loc'
                            )

                            coefficient_q_scale = tf.Variable(
                                tf.random_normal(
                                    [len(coef_ids)],
                                    mean=self.coef_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='coefficient_q_scale'
                            )

                            coefficient_q = Normal(
                                loc=coefficient_q_loc,
                                scale=tf.nn.softplus(coefficient_q_scale),
                                name='coefficient_q'
                            )
                            coefficient_summary = coefficient_q.mean()
                        else:
                            coefficient_q_samples = tf.Variable(
                                tf.zeros((self.n_samples, len(coef_ids)), dtype=self.FLOAT_TF),
                                name='coefficient_q_samples'
                            )
                            coefficient_q = Empirical(
                                params=coefficient_q_samples,
                                name='coefficient_q'
                            )
                            if self.inference_name == 'MetropolisHastings':
                                coefficient_proposal = Normal(
                                    loc=coefficient,
                                    scale=self.mh_proposal_sd,
                                    name='coefficient_proposal'
                                )
                                self.proposal_map[coefficient] = coefficient_proposal

                            coefficient_summary = coefficient_q.params[self.global_batch_step - 1]

                        self.inference_map[coefficient] = coefficient_q
                else:
                    if self.mv_ran:
                        coefficient = []
                        coefficient_summary = []
                        for coef in self.coef_names:
                            names = ['%s_by_%s_%s' % (coef, ran_gf, i) for i in range(self.rangf_n_levels[self.rangf.index(ran_gf)])]
                            ix = names2ix(names, self.full_joint_names)
                            coefficient.append(tf.gather(self.full_joint, ix))
                            coefficient_summary.append(tf.gather(self.full_joint_q.mean(), ix))
                        coefficient = tf.stack(coefficient, axis=1)
                        coefficient_summary = tf.stack(coefficient_summary, axis=1)
                    else:
                        coefficient = Normal(
                            sample_shape=[rangf_n_levels, len(coef_ids)],
                            loc=0.,
                            scale=self.coef_prior_sd_tf,
                            name='coefficient_by_%s' % ran_gf
                        )
                        if self.variational():
                            coefficient_q_loc = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels, len(coef_ids)],
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='coefficient_q_loc_by_%s' % ran_gf
                            )

                            coefficient_q_scale = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels, len(coef_ids)],
                                    mean=self.coef_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='coefficient_q_scale_by_%s' % ran_gf
                            )

                            coefficient_q = Normal(
                                loc=coefficient_q_loc,
                                scale=tf.nn.softplus(coefficient_q_scale),
                                name='coefficient_q_by_%s' % ran_gf
                            )
                            coefficient_summary = coefficient_q.mean()
                        else:
                            coefficient_q = Empirical(
                                params=tf.Variable(
                                    tf.zeros(
                                        (self.n_samples, rangf_n_levels, len(coef_ids)),
                                        dtype=self.FLOAT_TF
                                    ),
                                    name='coefficient_q_by_%s_samples' % ran_gf
                                ),
                                name='coefficient_q_by_%s' % ran_gf
                            )
                            if self.inference_name == 'MetropolisHastings':
                                coefficient_proposal = Normal(
                                    loc=coefficient,
                                    scale=self.mh_proposal_sd,
                                    name='coefficient_proposal_by_%s' % ran_gf
                                )
                                self.proposal_map[coefficient] = coefficient_proposal
                            coefficient_summary = coefficient_q.params[self.global_batch_step - 1]
                        self.inference_map[coefficient] = coefficient_q

                return coefficient, coefficient_summary

    def __initialize_irf_param__(self, param_name, ids, mean=0, lb=None, ub=None, irf_by_rangf=None):
        dim = len(ids)
        if irf_by_rangf is None:
            irf_by_rangf = []

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.mv:
                    names = ['%s_%s' %(param_name, id) for id in ids]
                    ix = names2ix(names, self.full_joint_names)
                    param_mean_init = self.full_joint_mu[ix[0]]
                    param = tf.expand_dims(tf.gather(self.full_joint, ix), 0)
                    param_summary = tf.expand_dims(tf.gather(self.full_joint_q.mean(), ix), 0)
                else:
                    param_mean_init, lb, ub = self.__process_mean__(mean, lb, ub)

                    param = Normal(
                        sample_shape = [1, dim],
                        loc=param_mean_init,
                        scale=self.conv_prior_sd,
                        name=sn('%s_%s' % (param_name, '-'.join(ids)))
                    )
                    if self.variational():
                        param_q_loc = tf.Variable(
                            tf.random_normal(
                                [1, dim],
                                mean=param_mean_init,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name=sn('%s_q_loc_%s' % (param_name, '-'.join(ids)))
                        )

                        param_q_scale = tf.Variable(
                            tf.random_normal(
                                [1, dim],
                                mean=self.conv_scale_mean_init,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name=sn('%s_q_scale_%s' % (param_name, '-'.join(ids)))
                        )

                        param_q = Normal(
                            loc=param_q_loc,
                            scale=tf.nn.softplus(param_q_scale),
                            name=sn('%s_q_%s' % (param_name, '-'.join(ids)))
                        )

                        param_summary = param_q.mean()
                    else:
                        params_q_samples = tf.Variable(
                            tf.zeros((self.n_samples, 1, dim), dtype=self.FLOAT_TF),
                            name=sn('%s_q_%s_samples' % (param_name, '-'.join(ids)))
                        )
                        param_q = Empirical(
                            params=params_q_samples,
                            name=sn('%s_q_%s_samples' % (param_name, '-'.join(ids)))
                        )
                        if self.inference_name == 'MetropolisHastings':
                            L_proposal = Normal(
                                loc=param,
                                scale=self.mh_proposal_sd,
                                name=sn('%s_proposal_%s' % (param_name, '-'.join(ids)))
                            )
                            self.proposal_map[param] = L_proposal

                        param_summary = param_q.params[self.global_batch_step - 1]

                    self.inference_map[param] = param_q

                if lb is None and ub is None:
                    param_out = param
                elif lb is not None and ub is None:
                    param_out = tf.nn.softplus(param) + lb + self.epsilon
                    param_summary = tf.nn.softplus(param_summary) + lb + self.epsilon
                elif lb is None and ub is not None:
                    param_out = -tf.nn.softplus(param) + ub - self.epsilon
                    param_summary = -tf.nn.softplus(param_summary) + ub - self.epsilon
                else:
                    param_out = tf.sigmoid(param) * ((ub-self.epsilon) - (lb+self.epsilon)) + lb + self.epsilon
                    param_summary = tf.sigmoid(param_summary) * ((ub-self.epsilon) - (lb+self.epsilon)) + lb + self.epsilon


                for i in range(dim):
                    tf.summary.scalar(
                        sn('%s/%s' % (param_name, ids[i])),
                        param_summary[0, i],
                        collections=['params']
                    )

                if len(irf_by_rangf) > 0:
                    for gf in irf_by_rangf:
                        i = self.rangf.index(gf)
                        mask_row_np = np.ones(self.rangf_n_levels[i])
                        mask_row_np[self.rangf_n_levels[i] - 1] = 0
                        mask_row = tf.constant(mask_row_np, dtype=self.FLOAT_TF)
                        col_ix = names2ix(irf_by_rangf[gf], ids)
                        mask_col_np = np.zeros([1, dim])
                        mask_col_np[0, col_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)

                        if self.mv_ran:
                            param_ran = []
                            param_ran_summary = []
                            for id in ids:
                                names = ['%s_%s_by_%s_%s' % (param_name, id, gf, i) for i in range(self.rangf_n_levels[self.rangf.index(gf)])]
                                ix = names2ix(names, self.full_joint_names)
                                param_ran.append(tf.gather(self.full_joint, ix))
                                param_ran_summary.append(tf.gather(self.full_joint_q.mean(), ix))
                            param_ran = tf.stack(param_ran, axis=1)
                            param_ran_summary = tf.stack(param_ran_summary, axis=1)
                        else:
                            param_ran = Normal(
                                sample_shape=[self.rangf_n_levels[i], dim],
                                loc=param_mean_init,
                                scale=self.conv_prior_sd,
                                name='%s_by_%s' % (param_name, gf)
                            )
                            if self.variational():
                                param_ran_q_loc = tf.Variable(
                                    tf.random_normal(
                                        [self.rangf_n_levels[i], dim],
                                        mean=param_mean_init,
                                        stddev=self.init_sd,
                                        dtype=self.FLOAT_TF
                                    ),
                                    name=sn('%s_q_loc_%s_by_%s' % (param_name, '-'.join(ids), gf))
                                )

                                param_ran_q_scale = tf.Variable(
                                    tf.random_normal(
                                        [self.rangf_n_levels[i], dim],
                                        mean=self.conv_scale_mean_init,
                                        stddev=self.init_sd,
                                        dtype=self.FLOAT_TF
                                    ),
                                    name=sn('%s_q_scale_%s_by_%s' % (param_name, '-'.join(ids), gf))
                                )

                                param_ran_q = Normal(
                                    loc=param_ran_q_loc,
                                    scale=tf.nn.softplus(param_ran_q_scale),
                                    name=sn('%s_q_%s_by_%s' % (param_name, '-'.join(ids), gf))
                                )
                                if self.log_random:
                                    param_ran_summary = tf.gather(param_ran_q.mean(), col_ix, axis=1)
                            else:
                                param_ran_q_samples = tf.Variable(
                                    tf.zeros((self.n_samples, self.rangf_n_levels[i], dim), dtype=self.FLOAT_TF),
                                    name=sn('%s_q_%s_by_%s_samples' % (param_name, '-'.join(ids), gf))
                                )
                                param_ran_q = Empirical(
                                    params=param_ran_q_samples,
                                    name=sn('%s_q_%s_by_%s' % (param_name, '-'.join(ids), gf))
                                )
                                if self.inference_name == 'MetropolisHastings':
                                    param_ran_proposal = Normal(
                                        loc=param_ran,
                                        scale=self.mh_proposal_sd,
                                        name=sn('%s_proposal_%s_by_%s' % (param_name, '-'.join(ids), gf))
                                    )
                                    self.proposal_map[param_ran] = param_ran_proposal
                                if self.log_random:
                                    param_ran_summary = tf.gather(
                                        param_ran_q.params[self.global_batch_step - 1],
                                        col_ix,
                                        axis=1
                                    )
                            self.inference_map[param_ran] = param_ran_q

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

                        param_ran_mean = tf.reduce_sum(param_ran, axis=0) / tf.reduce_sum(mask_row)
                        param_ran_centering_vector = tf.expand_dims(mask_row, -1) * param_ran_mean
                        param_ran -= param_ran_centering_vector

                        param_out += tf.gather(param_ran, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            param_ran_summary *= mask_col
                            param_ran_summary *= tf.expand_dims(mask_row, -1)
                            param_ran_summary -= param_ran_centering_vector

                            for j in range(len(irf_by_rangf[gf])):
                                irf_name = irf_by_rangf[gf][j]
                                ix = col_ix[j]
                                tf.summary.histogram(
                                    'by_%s/%s/%s' % (gf, param_name, irf_name),
                                    param_ran_summary[:, ix],
                                    collections=['random']
                                )

                return (param_out, param_summary)

    def __initialize_objective__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.y_scale_fixed is None:
                    if self.mv:
                        ix = names2ix('y_scale', self.full_joint_names)
                        assert len(ix) == 1, 'There should be exactly 1 parameter called "y_scale"'
                        ix = ix[0]
                        y_scale = self.full_joint[ix]
                        y_scale_summary = tf.nn.softplus(self.full_joint_q.mean()[ix])
                    else:
                        y_scale_mean_init = self.y_scale_mean_init
                        y_scale = Normal(
                            loc=y_scale_mean_init,
                            scale=self.y_scale_prior_sd_tf,
                            name='y_scale'
                        )
                        if self.variational():
                            y_scale_loc_q = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=y_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='y_scale_loc_q'
                            )

                            y_scale_scale_q = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=self.y_scale_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='y_scale_scale_q'
                            )
                            y_scale_q = Normal(
                                loc=y_scale_loc_q,
                                scale=tf.nn.softplus(y_scale_scale_q),
                                name='y_scale_q'
                            )
                            y_scale_summary = y_scale_q.mean()
                        else:
                            y_scale_q_samples = tf.Variable(
                                tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                                name=sn('y_scale_q_samples')
                            )
                            y_scale_q = Empirical(
                                params=y_scale_q_samples,
                                name=sn('y_scale_q')
                            )
                            if self.inference_name == 'MetropolisHastings':
                                y_scale_proposal = Normal(
                                    loc=y_scale,
                                    scale=self.mh_proposal_sd,
                                    name=sn('y_scale_proposal')
                                )
                                self.proposal_map[y_scale] = y_scale_proposal
                            y_scale_summary = y_scale_q.params[self.global_batch_step - 1]
                        self.inference_map[y_scale] = y_scale_q

                    tf.summary.scalar(
                        'y_scale',
                        tf.nn.softplus(y_scale_summary),
                        collections=['params']
                    )
                else:
                    sys.stderr.write('Fixed y scale: %s\n' %self.y_scale_fixed)
                    y_scale = self.y_scale_fixed_tf
                    y_scale_summary = y_scale

                if self.asymmetric_error:
                    self.y_skewness_sd_prior = 1.
                    self.y_tailweight_sd_prior = 1.
                    self.y_skewness = Normal(
                        loc = 0.,
                        scale = self.y_skewness_sd_prior,
                        name='y_skewness'
                    )
                    self.y_tailweight = Normal(
                        loc=tf.contrib.distributions.softplus_inverse(1.),
                        scale=self.y_tailweight_sd_prior,
                        name='y_tailweight'
                    )
                    if self.variational():
                        y_skewness_loc_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=0.,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_skewness_loc_q'
                        )
                        y_skewness_scale_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=tf.contrib.distributions.softplus_inverse(self.y_skewness_sd_prior),
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_skewness_loc_q'
                        )
                        self.y_skewness_q = Normal(
                            loc=y_skewness_loc_q,
                            scale=tf.nn.softplus(y_skewness_scale_q),
                            name='y_skewness_q'
                        )
                        self.y_skewness_summary = self.y_skewness_q.mean()
                        tf.summary.scalar(
                            'y_skewness',
                            self.y_skewness_summary,
                            collections=['params']
                        )

                        y_tailweight_loc_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=tf.contrib.distributions.softplus_inverse(1.),
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_tailweight_loc_q'
                        )
                        y_tailweight_scale_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=tf.contrib.distributions.softplus_inverse(self.y_tailweight_sd_prior),
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_tailweight_loc_q'
                        )
                        self.y_tailweight_q = Normal(
                            loc=y_tailweight_loc_q,
                            scale=tf.nn.softplus(y_tailweight_scale_q),
                            name='y_tailweight_q'
                        )
                        self.y_tailweight_summary = self.y_tailweight_q.mean()
                        tf.summary.scalar(
                            'y_tailweight',
                            tf.nn.softplus(self.y_tailweight_summary),
                            collections=['params']
                        )
                    else:
                        y_skewness_q_samples = tf.Variable(
                            tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                            name=sn('y_skewness_q_samples')
                        )
                        self.y_skewness_q = Empirical(
                            params=y_skewness_q_samples,
                            name=sn('y_skewness_q')
                        )
                        if self.inference_name == 'MetropolisHastings':
                            y_skewness_proposal = Normal(
                                loc=self.y_skewness,
                                scale=self.mh_proposal_sd,
                                name=sn('y_skewness_proposal')
                            )
                            self.proposal_map[self.y_skewness] = y_skewness_proposal
                        self.y_skewness_summary = self.y_skewness_q.params[self.global_batch_step - 1]
                        tf.summary.scalar(
                            'y_skewness',
                            self.y_skewness_summary,
                            collections=['params']
                        )

                        y_tailweight_q_samples = tf.Variable(
                            tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                            name=sn('y_tailweight_q_samples')
                        )
                        self.y_tailweight_q = Empirical(
                            params=y_tailweight_q_samples,
                            name=sn('y_tailweight_q')
                        )
                        if self.inference_name == 'MetropolisHastings':
                            y_tailweight_proposal = Normal(
                                loc=self.y_tailweight,
                                scale=self.mh_proposal_sd,
                                name=sn('y_tailweight_proposal')
                            )
                            self.proposal_map[self.y_tailweight] = y_tailweight_proposal
                        self.y_tailweight_summary = self.y_tailweight_q.params[self.global_batch_step - 1]
                        tf.summary.scalar(
                            'y_tailweight',
                            tf.nn.softplus(self.y_tailweight_summary),
                            collections=['params']
                        )

                    self.inference_map[self.y_skewness] = self.y_skewness_q
                    self.inference_map[self.y_tailweight] = self.y_tailweight_q

                    self.out = SinhArcsinh(
                        loc=self.out,
                        scale=tf.nn.softplus(y_scale),
                        skewness=self.y_skewness,
                        tailweight=tf.nn.softplus(self.y_tailweight),
                        name='output'
                    )
                    self.err_dist = SinhArcsinh(
                        loc=0.,
                        scale=tf.nn.softplus(y_scale_summary),
                        skewness=self.y_skewness_summary,
                        tailweight=tf.nn.softplus(self.y_tailweight_summary),
                        name='err_dist'
                    )

                    self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support))
                else:
                    self.out = Normal(
                        loc=self.out,
                        scale=tf.nn.softplus(y_scale),
                        name='output'
                    )
                    self.err_dist = Normal(
                        loc=0.,
                        scale=tf.nn.softplus(y_scale_summary),
                        name='err_dist'
                    )
                    self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support))

                self.err_dist_lb = -2*tf.nn.softplus(y_scale_summary)
                self.err_dist_ub = 2*tf.nn.softplus(y_scale_summary)

                self.optim = self.__initialize_optimizer__(self.optim_name)
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale},
                        optimizer=self.optim
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
                        step_size=self.lr,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/distr',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )

                self.out_post = ed.copy(self.out, self.inference_map)

                self.llprior = self.out.log_prob(self.y)
                self.ll_post = self.out_post.log_prob(self.y)

                ## Set up posteriors for post-hoc MC sampling
                for x in self.irf_mc:
                    for a in self.irf_mc[x]:
                        for b in self.irf_mc[x][a]:
                            self.irf_mc[x][a][b] = ed.copy(self.irf_mc[x][a][b], self.inference_map)
                for x in self.mc_integrals:
                    self.mc_integrals[x] = ed.copy(self.mc_integrals[x], self.inference_map)
                if self.pc:
                    for x in self.src_irf_mc:
                        for a in self.src_irf_mc[x]:
                            for b in self.src_irf_mc[x][a]:
                                self.src_irf_mc[x][a][b] = ed.copy(self.src_irf_mc[x][a][b], self.inference_map)
                    for x in self.src_mc_integrals:
                        self.src_mc_integrals[x] = ed.copy(self.src_mc_integrals[x], self.inference_map)

    def __initialize_logging__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')
                tf.summary.scalar('loss', self.loss_total, collections=['loss'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed')
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')




    #####################################################
    #
    #  Public methods
    #
    ######################################################

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
            This must hold for MCMC inference, since the number of minibatches is built into the model architecture.
            However, it is not necessary for variational inference.
        :param n_epoch_train: ``int``; the number of training iterations
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

        X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
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
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    if self.log_random and len(self.rangf) > 0:
                        summary_random = self.sess.run(self.summary_random)
                        self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))

                sys.stderr.write('Running %s inference...\n' % self.inference_name)
                while self.global_step.eval(session=self.sess) < self.n_iter:
                    p, p_inv = get_random_permutation(len(y))
                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')
                    if self.optim_name is not None and self.lr_decay_family is not None:
                        sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                    pb = tf.contrib.keras.utils.Progbar(self.n_train_minibatch)

                    loss_total = 0.

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
                        self.sess.run(self.ema_op)
                        metric_cur = info_dict['loss'] if self.variational() else info_dict['accept_rate']
                        if not np.isfinite(metric_cur):
                            metric_cur = 0
                        loss_total += metric_cur

                        self.sess.run(self.incr_global_batch_step)
                        pb.update((j/minibatch_size)+1, values=[('loss' if self.variational() else 'accept_rate', metric_cur)])

                    self.sess.run(self.incr_global_step)
                    if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                        loss_total /= n_minibatch
                        summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                        summary_params = self.sess.run(self.summary_params)
                        self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                        self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
                        if self.log_random and len(self.rangf) > 0:
                            summary_random = self.sess.run(self.summary_random)
                            self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                    if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        self.save()
                        self.make_plots(
                            irf_name_map=irf_name_map,
                            plot_n_time_units=plot_n_time_units,
                            plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap
                        )
                        lb = self.sess.run(self.err_dist_lb)
                        ub = self.sess.run(self.err_dist_ub)
                        n_time_units = ub-lb
                        fd_plot = {
                            self.support_start: lb,
                            self.n_time_units: n_time_units,
                            self.n_points_per_time_unit: 1
                        }
                        # plot_x = self.sess.run(self.support, feed_dict=fd_plot)
                        # plot_y = self.sess.run(self.err_dist_plot, feed_dict=fd_plot)
                        # plot_convolutions(
                        #     plot_x,
                        #     plot_y,
                        #     ['Error Distribution'],
                        #     dir=self.outdir,
                        #     filename='error_distribution.png'
                        # )
                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                self.inference.finalize()
                self.save()

                self.make_plots(
                    irf_name_map=irf_name_map,
                    plot_n_time_units=plot_n_time_units,
                    plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                    plot_x_inches=plot_x_inches,
                    plot_y_inches=plot_y_inches,
                    cmap=cmap
                )

                self.make_plots(
                    irf_name_map=irf_name_map,
                    plot_n_time_units=plot_n_time_units,
                    plot_n_points_per_time_unit=plot_n_points_per_time_unit,
                    plot_x_inches=plot_x_inches,
                    plot_y_inches=plot_y_inches,
                    cmap=cmap,
                    mc=True
                )
                sys.stderr.write('%s\n\n' % ('='*50))


    def predict(self, X, y_time, y_rangf, first_obs, last_obs, n_samples=None):
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

        if n_samples is None:
            n_samples = self.n_samples_eval

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        sys.stderr.write('Sampling from predictive posterior...\n')

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, first_obs, last_obs)
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        preds = np.zeros((len(y_time), n_samples))

        with self.sess.as_default():
            with self.sess.graph.as_default():

                self.set_predict_mode(True)

                fd = {
                    self.X: X_3d,
                    self.time_X: time_X_3d,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                }


                if not np.isfinite(self.eval_minibatch_size):
                    pb = tf.contrib.keras.utils.Progbar(n_samples)
                    for j in range(n_samples):
                        preds[:,j] = self.sess.run(self.out_post, feed_dict=fd)
                        pb.update(j+1, force=True)
                else:
                    for i in range(0, len(y_time), self.eval_minibatch_size):
                        sys.stderr.write('Minibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, self.n_eval_minibatch))
                        pb = tf.contrib.keras.utils.Progbar(n_samples)
                        fd_minibatch = {
                            self.X: X_3d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_3d[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y
                        }
                        for j in range(n_samples):
                            preds[i:i + self.eval_minibatch_size, j] = self.sess.run(self.out_post, feed_dict=fd_minibatch)
                            pb.update(j+1, force=True)

                preds = preds.mean(axis=1)

                sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return preds

    def log_lik(self, X, y, n_samples=None):
        """
        Compute log-likelihood of data from predictive posterior.

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

        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if n_samples is None:
            n_samples = self.n_samples_eval

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        sys.stderr.write('Sampling from predictive posterior...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, y.first_obs, y.last_obs)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        log_lik = np.zeros((len(time_y), n_samples))

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.X: X_3d,
                    self.time_X: time_X_3d,
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.y: y_dv
                }


                if not np.isfinite(self.eval_minibatch_size):
                    pb = tf.contrib.keras.utils.Progbar(n_samples)
                    for j in range(n_samples):
                        log_lik[:,j] = self.sess.run(self.ll_post, feed_dict=fd)
                        pb.update(j+1, force=True)
                else:
                    for i in range(0, len(time_y), self.eval_minibatch_size):
                        sys.stderr.write('Minibatch %d/%d\n' %((i/self.eval_minibatch_size)+1, self.n_eval_minibatch))
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                        fd_minibatch = {
                            self.X: X_3d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_3d[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.eval_minibatch_size]
                        }
                        for j in range(n_samples):
                            log_lik[i:i + self.eval_minibatch_size, j] = self.sess.run(self.ll_post, feed_dict=fd_minibatch)
                            pb.update(j+1, force=True)

                log_lik = log_lik.mean(axis=1)

                self.set_predict_mode(False)

                return log_lik

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

        For simplicity, plots for BDTSR models use the posterior mean, abstracting away from other characteristics of the posterior distribution (e.g. variance).

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :return: ``None``
        """
        return super(BDTSR, self).make_plots(**kwargs)

    def run_conv_op(self, feed_dict, scaled=False, n_samples=None):
        """
        Feedforward a batch of data in feed_dict through the convolutional layer to produce convolved inputs

        :param feed_dict: ``dict``; A dictionary of input variables
        :param scale: ``bool``; Whether to scale the outputs using the latent coefficients
        :return: ``pandas`` table; The convolved inputs
        """

        if n_samples is None:
            n_samples = self.n_samples_eval

        X_conv = np.zeros((len(feed_dict[self.X]), self.X_conv.shape[-1], n_samples))

        with self.sess.as_default():
            with self.sess.graph.as_default():
                sys.stderr.write('Convolving input features...\n')
                pb = tf.contrib.keras.utils.Progbar(n_samples)
                for i in range(0, n_samples):
                    X_conv[..., i] = self.sess.run(self.X_conv_scaled if scaled else self.X_conv, feed_dict=feed_dict)
                    pb.update(i + 1, force=True)
                X_conv = X_conv.mean(axis=2)
                return X_conv

