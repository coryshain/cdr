import os
from collections import defaultdict
import time

import pandas as pd
pd.options.mode.chained_assignment = None

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

from .formula import *
from .util import *
from .dtsr import DTSR
from .kwargs import DTSRBAYES_INITIALIZATION_KWARGS

import edward as ed
from edward.models import Empirical, Exponential, Gamma, MultivariateNormalTriL, Normal, SinhArcsinh





######################################################
#
#  BAYESIAN IMPLEMENTATION OF DTSR
#
######################################################

class DTSRBayes(DTSR):

    _INITIALIZATION_KWARGS = DTSRBAYES_INITIALIZATION_KWARGS

    _doc_header = """
        A DTSR implementation fitted using Bayesian inference.
    """
    _doc_args = DTSR._doc_args
    _doc_kwargs = DTSR._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs


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
            **kwargs
    ):

        super(DTSRBayes, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in DTSRBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in DTSR._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' %kwarg_key)

        if not self.variational():
            if self.n_samples is not None:
                sys.stderr.write('Parameter n_samples being overridden for sampling optimization\n')
            self.n_samples = self.n_iter*self.n_train_minibatch

        if not self.declare_priors:
            assert self.variational(), 'Only variational inference can be used to fit parameters without declaring priors'

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(DTSRBayes, self)._initialize_metadata()

        self.inference_map = {}
        if self.intercept_init is None:
            self.intercept_init = self.y_train_mean
        if self.intercept_prior_sd is None:
            self.intercept_prior_sd = self.y_train_sd * self.prior_sd_scaling_coefficient
        if self.coef_prior_sd is None:
            self.coef_prior_sd = self.y_train_sd * self.prior_sd_scaling_coefficient
        if self.y_scale_init is None:
            self.y_scale_init = self.y_train_sd
        if self.y_scale_prior_sd is None:
            self.y_scale_prior_sd = self.y_train_sd * self.y_scale_prior_sd_scaling_coefficient

        if self.inference_name == 'MetropolisHastings':
            self.proposal_map = {}
            if self.mh_proposal_sd is None:
                self.mh_proposal_sd = self.y_train_sd * self.prior_sd_scaling_coefficient

        with self.sess.as_default():
            with self.sess.graph.as_default():

                self.intercept_prior_sd_tf = tf.constant(float(self.intercept_prior_sd), dtype=self.FLOAT_TF)
                self.intercept_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.intercept_prior_sd_tf)
                self.intercept_posterior_init_sd_tf = self.intercept_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.intercept_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.intercept_posterior_init_sd_tf)

                self.coef_prior_sd_tf = tf.constant(float(self.coef_prior_sd), dtype=self.FLOAT_TF)
                self.coef_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.coef_prior_sd_tf)
                self.coef_posterior_init_sd_tf = self.coef_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.coef_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.coef_posterior_init_sd_tf)

                self.conv_prior_sd_tf = tf.constant(float(self.conv_prior_sd), dtype=self.FLOAT_TF)
                self.conv_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.conv_prior_sd_tf)
                self.conv_posterior_init_sd_tf = self.conv_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.conv_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.conv_posterior_init_sd_tf)

                self.y_scale_init_tf = tf.constant(float(self.y_scale_init), dtype=self.FLOAT_TF)
                self.y_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_init_tf)

                self.y_scale_prior_sd_tf = tf.constant(float(self.y_scale_prior_sd), dtype=self.FLOAT_TF)
                self.y_scale_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_prior_sd_tf)
                self.y_scale_posterior_init_sd_tf = self.y_scale_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.y_scale_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_scale_posterior_init_sd_tf)

                self.y_skewness_prior_sd_tf = tf.constant(float(self.y_skewness_prior_sd), dtype=self.FLOAT_TF)
                self.y_skewness_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_skewness_prior_sd_tf)
                self.y_skewness_posterior_init_sd_tf = self.y_skewness_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.y_skewness_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_skewness_posterior_init_sd_tf)

                self.y_tailweight_prior_sd_tf = tf.constant(float(self.y_tailweight_prior_sd), dtype=self.FLOAT_TF)
                self.y_tailweight_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_tailweight_prior_sd_tf)
                self.y_tailweight_posterior_init_sd_tf = self.y_tailweight_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.y_tailweight_posterior_scale_mean_init = tf.contrib.distributions.softplus_inverse(self.y_tailweight_posterior_init_sd_tf)

        if self.mv:
            self._initialize_full_joint()

    def _pack_metadata(self):
        md = super(DTSRBayes, self)._pack_metadata()
        for kwarg in DTSRBayes._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        super(DTSRBayes, self)._unpack_metadata(md)

        for kwarg in DTSRBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            sys.stderr.write('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_param_list(self):
        name = []
        mean_init = []
        sd_init = []
        if self.has_intercept[None]:
            name.append('intercept')
            mean_init.append(self.intercept_init_tf)
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

        name_conv, mean_init_conv, sd_init_conv = self._initialize_conv_param_list()

        name += name_conv
        mean_init += mean_init_conv
        sd_init += sd_init_conv

        if self.y_scale_init is None:
            name.append('y_scale')
            mean_init.append(self.y_scale_init_tf)
            sd_init.append(self.y_scale_prior_sd_tf)

        assert len(name) == len(mean_init) == len(sd_init), 'Error: lengths of computed lists of parameter names, means, and sds do not match'

        return (name, mean_init, sd_init)

    def _initialize_conv_param_list(self):
        name = []
        mean_init = []
        sd_init = []

        for family in self.atomic_irf_names_by_family:
            if family == 'DiracDelta':
                continue

            irf_ids = self.atomic_irf_names_by_family[family]
            irf_param_init = self.atomic_irf_param_init_by_family[family]

            irf_by_rangf = {}
            for id in irf_ids:
                for gf in self.irf_by_rangf:
                    if id in self.irf_by_rangf[gf]:
                        if gf not in irf_by_rangf:
                            irf_by_rangf[gf] = []
                        irf_by_rangf[gf].append(id)

            if family == 'Exp':
                param_name = ['beta']
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [beta]
                param_sd = [self.conv_prior_sd_tf]
            elif family == 'SteepExp':
                param_name = ['beta']
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [beta]
                param_sd = [self.conv_prior_sd_tf]
            elif family == 'Gamma':
                param_name = ['alpha', 'beta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                alpha, _, _ = self._process_mean(alpha_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [alpha, beta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family in ['GammaKgt1', 'GammaShapeGT1']:
                param_name = ['alpha', 'beta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                alpha, _, _ = self._process_mean(alpha_init, lb=1)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [alpha, beta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'SteepGamma':
                param_name = ['alpha', 'beta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=25)
                alpha, _, _ = self._process_mean(alpha_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [alpha, beta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'ShiftedGamma':
                param_name = ['alpha', 'beta', 'delta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                delta_init = self._get_mean_init_vector(irf_ids, 'delta', irf_param_init, default=-0.5)
                alpha, _, _ = self._process_mean(alpha_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                delta, _, _ = self._process_mean(delta_init, ub=0)
                param_mean = [alpha, beta, delta]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family in ['ShiftedGammaKgt1', 'ShiftedGammaShapeGT1']:
                param_name = ['alpha', 'beta', 'delta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=2)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=5)
                delta_init = self._get_mean_init_vector(irf_ids, 'delta', irf_param_init, default=-0.5)
                alpha, _, _ = self._process_mean(alpha_init, lb=1)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                delta, _, _ = self._process_mean(delta_init, ub=0)
                param_mean = [alpha, beta, delta]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'Normal':
                param_name = ['mu', 'sigma']
                mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                mu, _, _ = self._process_mean(mu_init)
                sigma, _, _ = self._process_mean(sigma_init, lb=0)
                param_mean = [mu, sigma]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'SkewNormal':
                param_name = ['mu', 'sigma', 'alpha']
                mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=0)
                mu, _, _ = self._process_mean(mu_init)
                sigma, _, _ = self._process_mean(sigma_init, lb=0)
                alpha, _, _ = self._process_mean(alpha_init)
                param_mean = [mu, sigma, alpha]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'EMG':
                param_name = ['mu', 'sigma', 'beta']
                mu_init = self._get_mean_init_vector(irf_ids, 'mu', irf_param_init, default=0)
                sigma_init = self._get_mean_init_vector(irf_ids, 'sigma', irf_param_init, default=1)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=0)
                mu, _, _ = self._process_mean(mu_init)
                sigma, _, _ = self._process_mean(sigma_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [mu, sigma, beta]
                param_sd = [self.conv_prior_sd_tf] * 3
            elif family == 'BetaPrime':
                param_name = ['alpha', 'beta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=1)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                alpha, _, _ = self._process_mean(alpha_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                param_mean = [alpha, beta]
                param_sd = [self.conv_prior_sd_tf] * 2
            elif family == 'ShiftedBetaPrime':
                param_name = ['alpha', 'beta', 'delta']
                alpha_init = self._get_mean_init_vector(irf_ids, 'alpha', irf_param_init, default=1)
                beta_init = self._get_mean_init_vector(irf_ids, 'beta', irf_param_init, default=1)
                delta_init = self._get_mean_init_vector(irf_ids, 'delta', irf_param_init, default=-1)
                alpha, _, _ = self._process_mean(alpha_init, lb=0)
                beta, _, _ = self._process_mean(beta_init, lb=0)
                delta, _, _ = self._process_mean(delta_init, ub=0)
                param_mean = [alpha, beta, delta]
                param_sd = [self.conv_prior_sd_tf] * 3

            for i in range(len(irf_ids)):
                id = irf_ids[i]
                name += ['%s_%s' % (p, id) for p in param_name]
                mean_init += [x[i] for x in param_mean]
                sd_init += param_sd
                if self.mv_ran:
                    for i in range(len(self.rangf)):
                        gf = self.rangf[i]
                        if gf in irf_by_rangf:
                            levels = list(range(self.rangf_n_levels[i]))
                            name += ['%s_%s_by_%s_%s' % (p, id, gf, j) for j in levels for p in param_name]
                            mean_init += [0.] * self.rangf_n_levels[i] * len(param_name)
                            sd_init += [self.conv_scale_mean_init] * self.rangf_n_levels[i] * len(param_name)

        return(name, mean_init, sd_init)

    def _initialize_full_joint(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                names, means, sds = self._initialize_param_list()
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
                            [len(sds) * (len(sds)) / 2],
                            mean=tf.diag(self.full_joint_sigma),
                            stddev=self.init_sd,
                            dtype=self.FLOAT_TF
                        ),
                        name='full_joint_q_scale'
                    )

                    self.full_joint_q = MultivariateNormalTriL(
                        loc=full_joint_q_loc,
                        scale_tril=tf.contrib.distributions.fill_triangular(tf.nn.softplus(full_joint_q_scale)),
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

    def initialize_intercept(self, ran_gf=None):
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
                        if self.variational():
                            # Posterior distribution
                            intercept_q_loc = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=self.intercept_init_tf,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_loc'
                            )

                            intercept_q_scale = tf.Variable(
                                tf.random_normal(
                                    [],
                                    mean=self.intercept_posterior_scale_mean_init,
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

                            if self.declare_priors:
                                # Prior distribution
                                intercept = Normal(
                                    sample_shape=[],
                                    loc=self.intercept_init_tf,
                                    scale=self.intercept_prior_sd_tf,
                                    name='intercept'
                                )
                                self.inference_map[intercept] = intercept_q
                            else:
                                intercept = intercept_q

                        else:
                            # Prior distribution
                            intercept = Normal(
                                sample_shape=[],
                                loc=self.intercept_init_tf,
                                scale=self.intercept_prior_sd_tf,
                                name='intercept'
                            )

                            # Posterior distribution
                            intercept_q_samples = tf.Variable(
                                tf.ones((self.n_samples), dtype=self.FLOAT_TF) * self.intercept_init_tf,
                                name='intercept_q_samples'
                            )

                            intercept_q = Empirical(
                                params=intercept_q_samples,
                                name='intercept_q'
                            )

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                intercept_proposal = Normal(
                                    loc=intercept,
                                    scale=self.mh_proposal_sd,
                                    name='intercept_proposal'
                                )
                                self.proposal_map[intercept] = intercept_proposal

                            intercept_summary = intercept_q.params[self.global_batch_step - 1]

                            self.inference_map[intercept] = intercept_q

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                    if self.mv_ran:
                        names = ['intercept_by_%s_%s' %(ran_gf, i) for i in range(rangf_n_levels)]
                        ix = names2ix(names, self.full_joint_names)
                        intercept = tf.gather(self.full_joint, ix)
                        intercept_summary = tf.gather(self.full_joint_q.mean(), ix)
                    else:
                        if self.variational():
                            # Posterior distribution
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
                                    mean=self.intercept_posterior_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='intercept_q_scale_by_%s' % ran_gf
                            )

                            intercept_q = Normal(
                                loc=intercept_q_loc,
                                scale=tf.nn.softplus(intercept_q_scale) * self.ranef_to_fixef_prior_sd_ratio,
                                name='intercept_q_by_%s' % ran_gf
                            )

                            intercept_summary = intercept_q.mean()

                            if self.declare_priors:
                                # Prior distribution
                                intercept = Normal(
                                    sample_shape=[rangf_n_levels],
                                    loc=0.,
                                    scale=self.intercept_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio,
                                    name='intercept_by_%s' % ran_gf
                                )
                                self.inference_map[intercept] = intercept_q
                            else:
                                intercept = intercept_q

                        else:
                            # Prior distribution
                            intercept = Normal(
                                sample_shape=[rangf_n_levels],
                                loc=0.,
                                scale=self.intercept_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio,
                                name='intercept_by_%s' % ran_gf
                            )

                            # Posterior distribution
                            intercept_q_ran_samples = tf.Variable(
                                tf.zeros((self.n_samples, rangf_n_levels), dtype=self.FLOAT_TF),
                                name='intercept_q_by_%s_samples' % ran_gf
                            )
                            intercept_q = Empirical(
                                params=intercept_q_ran_samples,
                                name='intercept_q_by_%s' % ran_gf
                            )

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                intercept_proposal = Normal(
                                    loc=intercept,
                                    scale=self.mh_proposal_sd,
                                    name='intercept_proposal_by_%s' % ran_gf
                                )
                                self.proposal_map[intercept] = intercept_proposal

                            intercept_summary = intercept_q.params[self.global_batch_step - 1]

                            self.inference_map[intercept] = intercept_q

                return intercept, intercept_summary

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
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
                        if self.variational():
                            # Posterior distribution
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
                                    mean=self.coef_posterior_scale_mean_init,
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

                            if self.declare_priors:
                                # Prior distribution
                                coefficient = Normal(
                                    sample_shape=[len(coef_ids)],
                                    loc=0.,
                                    scale=self.coef_prior_sd_tf,
                                    name='coefficient'
                                )
                                self.inference_map[coefficient] = coefficient_q
                            else:
                                coefficient = coefficient_q
                        else:
                            # Prior distribution
                            coefficient = Normal(
                                sample_shape=[len(coef_ids)],
                                loc=0.,
                                scale=self.coef_prior_sd_tf,
                                name='coefficient'
                            )

                            # Posterior distribution
                            coefficient_q_samples = tf.Variable(
                                tf.zeros((self.n_samples, len(coef_ids)), dtype=self.FLOAT_TF),
                                name='coefficient_q_samples'
                            )
                            coefficient_q = Empirical(
                                params=coefficient_q_samples,
                                name='coefficient_q'
                            )

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                coefficient_proposal = Normal(
                                    loc=coefficient,
                                    scale=self.mh_proposal_sd,
                                    name='coefficient_proposal'
                                )
                                self.proposal_map[coefficient] = coefficient_proposal

                            coefficient_summary = coefficient_q.params[self.global_batch_step - 1]

                            self.inference_map[coefficient] = coefficient_q
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                    if self.mv_ran:
                        coefficient = []
                        coefficient_summary = []
                        for coef in self.coef_names:
                            names = ['%s_by_%s_%s' % (coef, ran_gf, i) for i in range(rangf_n_levels)]
                            ix = names2ix(names, self.full_joint_names)
                            coefficient.append(tf.gather(self.full_joint, ix))
                            coefficient_summary.append(tf.gather(self.full_joint_q.mean(), ix))
                        coefficient = tf.stack(coefficient, axis=1)
                        coefficient_summary = tf.stack(coefficient_summary, axis=1)
                    else:
                        if self.variational():
                            # Posterior distribution
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
                                    mean=self.coef_posterior_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name='coefficient_q_scale_by_%s' % ran_gf
                            )

                            coefficient_q = Normal(
                                loc=coefficient_q_loc,
                                scale=tf.nn.softplus(coefficient_q_scale) * self.ranef_to_fixef_prior_sd_ratio,
                                name='coefficient_q_by_%s' % ran_gf
                            )
                            coefficient_summary = coefficient_q.mean()

                            if self.declare_priors:
                                # Prior distribution
                                coefficient = Normal(
                                    sample_shape=[rangf_n_levels, len(coef_ids)],
                                    loc=0.,
                                    scale=self.coef_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio,
                                    name='coefficient_by_%s' % ran_gf
                                )
                                self.inference_map[coefficient] = coefficient_q
                            else:
                                coefficient = coefficient_q

                        else:
                            # Prior distribution
                            coefficient = Normal(
                                sample_shape=[rangf_n_levels, len(coef_ids)],
                                loc=0.,
                                scale=self.coef_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio,
                                name='coefficient_by_%s' % ran_gf
                            )

                            # Posterior distribution
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
                                # Proposal distribution
                                coefficient_proposal = Normal(
                                    loc=coefficient,
                                    scale=self.mh_proposal_sd,
                                    name='coefficient_proposal_by_%s' % ran_gf
                                )
                                self.proposal_map[coefficient] = coefficient_proposal

                            coefficient_summary = coefficient_q.params[self.global_batch_step - 1]

                            self.inference_map[coefficient] = coefficient_q

                return coefficient, coefficient_summary

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0., ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.mv:
                        names = ['%s_%s' %(param_name, id) for id in ids]
                        ix = names2ix(names, self.full_joint_names)
                        param_mean_init = self.full_joint_mu[ix[0]]
                        param = tf.expand_dims(tf.gather(self.full_joint, ix), 0)
                        param_summary = tf.expand_dims(tf.gather(self.full_joint_q.mean(), ix), 0)
                    else:
                        if self.variational():
                            # Posterior distribution
                            param_q_loc = tf.Variable(
                                tf.random_normal(
                                    [1, len(ids)],
                                    mean=mean,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name=sn('%s_q_loc_%s' % (param_name, '-'.join(ids)))
                            )

                            param_q_scale = tf.Variable(
                                tf.random_normal(
                                    [1, len(ids)],
                                    mean=self.conv_posterior_scale_mean_init,
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

                            if self.declare_priors:
                                # Prior distribution
                                param = Normal(
                                    loc=mean,
                                    scale=self.conv_prior_sd,
                                    name=sn('%s_%s' % (param_name, '-'.join(ids)))
                                )
                                self.inference_map[param] = param_q
                            else:
                                param = param_q
                        else:
                            # Prior distribution
                            param = Normal(
                                loc=mean,
                                scale=self.conv_prior_sd,
                                name=sn('%s_%s' % (param_name, '-'.join(ids)))
                            )

                            # Posterior distribution
                            params_q_samples = tf.Variable(
                                tf.zeros((self.n_samples, 1, len(ids)), dtype=self.FLOAT_TF),
                                name=sn('%s_q_%s_samples' % (param_name, '-'.join(ids)))
                            )
                            param_q = Empirical(
                                params=params_q_samples,
                                name=sn('%s_q_%s_samples' % (param_name, '-'.join(ids)))
                            )

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                L_proposal = Normal(
                                    loc=param,
                                    scale=self.mh_proposal_sd,
                                    name=sn('%s_proposal_%s' % (param_name, '-'.join(ids)))
                                )
                                self.proposal_map[param] = L_proposal

                            param_summary = param_q.params[self.global_batch_step - 1]

                            self.inference_map[param] = param_q
                else:
                    if self.mv_ran:
                        param = []
                        param_summary = []
                        for ID in ids:
                            names = ['%s_%s_by_%s_%s' % (param_name, ID, ran_gf, i) for i in range(self.rangf_n_levels[self.rangf.index(ran_gf)])]
                            ix = names2ix(names, self.full_joint_names)
                            param.append(tf.gather(self.full_joint, ix))
                            param_summary.append(tf.gather(self.full_joint_q.mean(), ix))
                        param = tf.stack(param, axis=1)
                        param_summary = tf.stack(param_summary, axis=1)
                    else:
                        rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)]
                        if self.variational():
                            # Posterior distribution
                            param_q_loc = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels, len(ids)],
                                    mean=0.,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name=sn('%s_q_loc_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                            )

                            param_q_scale = tf.Variable(
                                tf.random_normal(
                                    [rangf_n_levels, len(ids)],
                                    mean=self.conv_posterior_scale_mean_init,
                                    stddev=self.init_sd,
                                    dtype=self.FLOAT_TF
                                ),
                                name=sn('%s_q_scale_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                            )

                            param_q = Normal(
                                loc=param_q_loc,
                                scale=tf.nn.softplus(param_q_scale) * self.ranef_to_fixef_prior_sd_ratio,
                                name=sn('%s_q_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                            )

                            param_summary = param_q.mean()

                            if self.declare_priors:
                                # Prior distribution
                                param = Normal(
                                    sample_shape=[rangf_n_levels, len(ids)],
                                    loc=0.,
                                    scale=self.conv_prior_sd * self.ranef_to_fixef_prior_sd_ratio,
                                    name='%s_by_%s' % (param_name, ran_gf)
                                )
                                self.inference_map[param] = param_q
                            else:
                                param = param_q

                        else:
                            # Prior distribution
                            param = Normal(
                                sample_shape=[rangf_n_levels, len(ids)],
                                loc=0.,
                                scale=self.conv_prior_sd * self.ranef_to_fixef_prior_sd_ratio,
                                name='%s_by_%s' % (param_name, ran_gf)
                            )

                            # Posterior distribution
                            param_q_samples = tf.Variable(
                                tf.zeros((self.n_samples, rangf_n_levels, len(ids)), dtype=self.FLOAT_TF),
                                name=sn('%s_q_%s_by_%s_samples' % (param_name, '-'.join(ids), ran_gf))
                            )
                            param_q = Empirical(
                                params=param_q_samples,
                                name=sn('%s_q_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                            )

                            param_summary = param_q.params[self.global_batch_step - 1]

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                param_proposal = Normal(
                                    loc=param,
                                    scale=self.mh_proposal_sd,
                                    name=sn('%s_proposal_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                                )
                                self.proposal_map[param] = param_proposal

                            self.inference_map[param] = param_q

                return (param, param_summary)

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.y_scale_trainable:
                    if self.mv:
                        ix = names2ix('y_scale', self.full_joint_names)
                        assert len(ix) == 1, 'There should be exactly 1 parameter called "y_scale"'
                        ix = ix[0]
                        y_scale = self.full_joint[ix]
                        y_scale_summary = tf.nn.softplus(self.full_joint_q.mean()[ix])
                    else:
                        y_scale_mean_init = self.y_scale_mean_init

                        if self.variational():
                            # Posterior distribution
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
                                    mean=self.y_scale_posterior_scale_mean_init,
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

                            if self.declare_priors:
                                # Prior distribution
                                y_scale = Normal(
                                    loc=y_scale_mean_init,
                                    scale=self.y_scale_prior_sd_tf,
                                    name='y_scale'
                                )
                                self.inference_map[y_scale] = y_scale_q
                            else:
                                y_scale = y_scale_q
                        else:
                            # Prior distribution
                            y_scale = Normal(
                                loc=y_scale_mean_init,
                                scale=self.y_scale_prior_sd_tf,
                                name='y_scale'
                            )

                            # Posterior distribution
                            y_scale_q_samples = tf.Variable(
                                tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                                name=sn('y_scale_q_samples')
                            )
                            y_scale_q = Empirical(
                                params=y_scale_q_samples,
                                name=sn('y_scale_q')
                            )

                            if self.inference_name == 'MetropolisHastings':
                                # Proposal distribution
                                y_scale_proposal = Normal(
                                    loc=y_scale,
                                    scale=self.mh_proposal_sd,
                                    name=sn('y_scale_proposal')
                                )
                                self.proposal_map[y_scale] = y_scale_proposal

                            y_scale_summary = y_scale_q.params[self.global_batch_step - 1]

                            self.inference_map[y_scale] = y_scale_q

                    y_scale = tf.nn.softplus(y_scale)
                    y_scale_summary = tf.nn.softplus(y_scale_summary)

                    tf.summary.scalar(
                        'y_scale',
                        y_scale_summary,
                        collections=['params']
                    )
                else:
                    sys.stderr.write('Fixed y scale: %s\n' % self.y_scale_init)
                    y_scale = self.y_scale_init_tf
                    y_scale_summary = y_scale

                self.y_scale = y_scale
                self.y_scale_summary = y_scale_summary

                if self.asymmetric_error:
                    if self.variational():
                        # Posterior distributions
                        y_skewness_loc_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=0.,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_skewness_q_loc'
                        )
                        y_skewness_scale_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=self.y_skewness_posterior_scale_mean_init,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_skewness_q_loc'
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
                            name='y_tailweight_q_loc'
                        )
                        y_tailweight_scale_q = tf.Variable(
                            tf.random_normal(
                                [],
                                mean=self.y_tailweight_posterior_scale_mean_init,
                                stddev=self.init_sd,
                                dtype=self.FLOAT_TF
                            ),
                            name='y_tailweight_q_scale'
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

                        if self.declare_priors:
                            # Prior distributions
                            self.y_skewness = Normal(
                                loc=0.,
                                scale=self.y_skewness_prior_sd,
                                name='y_skewness'
                            )
                            self.y_tailweight = Normal(
                                loc=tf.contrib.distributions.softplus_inverse(1.),
                                scale=self.y_tailweight_prior_sd,
                                name='y_tailweight'
                            )

                            self.inference_map[self.y_skewness] = self.y_skewness_q
                            self.inference_map[self.y_tailweight] = self.y_tailweight_q

                        else:
                            self.y_skewness = self.y_skewness_q
                            self.y_tailweight = self.y_tailweight_q
                    else:
                        # Prior distributions
                        self.y_skewness = Normal(
                            loc=0.,
                            scale=self.y_skewness_prior_sd,
                            name='y_skewness'
                        )
                        self.y_tailweight = Normal(
                            loc=tf.contrib.distributions.softplus_inverse(1.),
                            scale=self.y_tailweight_prior_sd,
                            name='y_tailweight'
                        )

                        # Posterior distributions
                        y_skewness_q_samples = tf.Variable(
                            tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                            name=sn('y_skewness_q_samples')
                        )
                        self.y_skewness_q = Empirical(
                            params=y_skewness_q_samples,
                            name=sn('y_skewness_q')
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
                            # Proposal distributions
                            y_skewness_proposal = Normal(
                                loc=self.y_skewness,
                                scale=self.mh_proposal_sd,
                                name=sn('y_skewness_proposal')
                            )
                            y_tailweight_proposal = Normal(
                                loc=self.y_tailweight,
                                scale=self.mh_proposal_sd,
                                name=sn('y_tailweight_proposal')
                            )
                            self.proposal_map[self.y_skewness] = y_skewness_proposal
                            self.proposal_map[self.y_tailweight] = y_tailweight_proposal

                        self.y_skewness_summary = self.y_skewness_q.params[self.global_batch_step - 1]
                        self.y_tailweight_summary = self.y_tailweight_q.params[self.global_batch_step - 1]

                        tf.summary.scalar(
                            'y_skewness',
                            self.y_skewness_summary,
                            collections=['params']
                        )
                        tf.summary.scalar(
                            'y_tailweight',
                            tf.nn.softplus(self.y_tailweight_summary),
                            collections=['params']
                        )

                        self.inference_map[self.y_skewness] = self.y_skewness_q
                        self.inference_map[self.y_tailweight] = self.y_tailweight_q

                    self.out = SinhArcsinh(
                        loc=self.out,
                        scale=y_scale,
                        skewness=self.y_skewness,
                        tailweight=tf.nn.softplus(self.y_tailweight),
                        name='output'
                    )
                    self.err_dist = SinhArcsinh(
                        loc=0.,
                        scale=y_scale_summary,
                        skewness=self.y_skewness_summary,
                        tailweight=tf.nn.softplus(self.y_tailweight_summary),
                        name='err_dist'
                    )

                    self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support))
                else:
                    self.out = Normal(
                        loc=self.out,
                        scale=self.y_scale,
                        name='output'
                    )
                    self.err_dist = Normal(
                        loc=0.,
                        scale=self.y_scale_summary,
                        name='err_dist'
                    )
                    self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support))

                self.err_dist_lb = -3 * y_scale_summary
                self.err_dist_ub = 3 * y_scale_summary

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_output_model()

                self.opt = self._initialize_optimizer(self.optim_name)
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale},
                        optimizer=self.opt
                    )
                elif self.inference_name == 'MetropolisHastings':
                    self.inference = getattr(ed, self.inference_name)(self.inference_map, self.proposal_map, data={self.out: self.y})
                    self.inference.initialize(
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={self.out: self.minibatch_scale}
                    )
                else:
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={self.out: self.y})
                    self.inference.initialize(
                        step_size=self.lr,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
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

    def report_settings(self, indent=0):
        out = super(DTSRBayes, self).report_settings(indent=indent)
        for kwarg in DTSRBAYES_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out

    def run_train_step(self, feed_dict):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                info_dict = self.inference.update(feed_dict)
                self.sess.run(self.incr_global_batch_step)

                out_dict = {
                    'loss': info_dict['loss'] if self.variational() else info_dict['accept_rate']
                }

                return out_dict

    def run_predict_op(self, feed_dict, n_samples=None, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if n_samples is None:
                    n_samples = self.n_samples_eval
                pb = tf.contrib.keras.utils.Progbar(n_samples)

                preds = np.zeros((len(feed_dict[self.time_y]), n_samples))
                for i in range(n_samples):
                    preds[:, i] = self.sess.run(self.out_post, feed_dict=feed_dict)
                    pb.update(i + 1, force=True)

                preds = preds.mean(axis=1)

                if verbose:
                    sys.stderr.write('\n')

                return preds

    def finalize(self):
        super(DTSRBayes, self).finalize()
        self.inference.finalize()

    def run_loglik_op(self, feed_dict, n_samples=None, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if n_samples is None:
                    n_samples = self.n_samples_eval
                pb = tf.contrib.keras.utils.Progbar(n_samples)

                log_lik = np.zeros((len(feed_dict[self.time_y]), n_samples))
                for i in range(n_samples):
                    log_lik[:, i] = self.sess.run(self.ll_post, feed_dict=feed_dict)
                    pb.update(i + 1, force=True)

                log_lik = log_lik.mean(axis=1)

                if verbose:
                    sys.stderr.write('\n')

                return log_lik

    def run_conv_op(self, feed_dict, scaled=False, n_samples=None):
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

    def summary(self, fixed=True, random=False):
        summary = '=' * 50 + '\n'
        summary += 'DTSR regression\n\n'
        summary += 'Output directory: %s\n\n' % self.outdir
        summary += 'Formula:\n'
        summary += '  ' + self.form_str + '\n\n'

        if fixed:
            if self.pc:
                terminal_names = self.src_terminal_names
            else:
                terminal_names = self.terminal_names
            posterior_summaries = np.zeros((len(terminal_names), 3))
            for i in range(len(terminal_names)):
                terminal = terminal_names[i]
                row = np.array(self.ci_integral(terminal, n_time_units=10))
                posterior_summaries[i] += row
            posterior_summaries = pd.DataFrame(posterior_summaries, index=terminal_names,
                                               columns=['Mean', '2.5%', '97.5%'])

            summary += '\nPosterior integral summaries by predictor:\n'
            summary += posterior_summaries.to_string() + '\n\n'

        #TODO: Fill the rest of this in

        summary += '=' * 50 + '\n'

        return(summary)



