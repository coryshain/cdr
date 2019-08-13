import pandas as pd
pd.options.mode.chained_assignment = None

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

from .util import *
from .cdrbase import CDR
from .kwargs import CDRBAYES_INITIALIZATION_KWARGS

import edward as ed
from edward.models import Empirical, Exponential, Gamma, MultivariateNormalTriL, Normal, SinhArcsinh





######################################################
#
#  BAYESIAN IMPLEMENTATION OF CDR
#
######################################################

class CDRBayes(CDR):

    _INITIALIZATION_KWARGS = CDRBAYES_INITIALIZATION_KWARGS

    _doc_header = """
        A CDR implementation fitted using Bayesian inference.
    """
    _doc_args = CDR._doc_args
    _doc_kwargs = CDR._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs


    #####################################################
    #
    #  Native methods
    #
    #####################################################

    def __init__(self, form_str, X, y, **kwargs):

        super(CDRBayes, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in CDRBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        kwarg_keys = [x.key for x in CDR._INITIALIZATION_KWARGS]
        for kwarg_key in kwargs:
            if kwarg_key not in kwarg_keys:
                raise TypeError('__init__() got an unexpected keyword argument %s' %kwarg_key)

        if not self.variational():
            if self.n_samples is not None:
                stderr('Parameter n_samples being overridden for sampling optimization\n')
            self.n_samples = self.n_iter*self.n_train_minibatch

        if not (self.declare_priors_fixef or self.declare_priors_ranef):
            assert self.variational(), 'Only variational inference can be used to fit parameters without declaring priors'

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(CDRBayes, self)._initialize_metadata()

        self.parameter_table_columns = ['Mean', '2.5%', '97.5%']

        self.inference_map = {}
        self.MAP_map = {}
        if self.intercept_init is None:
            if self.standardize_response:
                self.intercept_init = 0.
            else:
                self.intercept_init = self.y_train_mean
        if self.intercept_prior_sd is None:
            if self.standardize_response:
                self.intercept_prior_sd = self.prior_sd_scaling_coefficient
            else:
                self.intercept_prior_sd = self.y_train_sd * self.prior_sd_scaling_coefficient
        if self.coef_prior_sd is None:
            if self.standardize_response:
                self.coef_prior_sd = self.prior_sd_scaling_coefficient
            else:
                self.coef_prior_sd = self.y_train_sd * self.prior_sd_scaling_coefficient
        if self.y_sd_prior_sd is None:
            if self.standardize_response:
                self.y_sd_prior_sd = self.y_sd_prior_sd_scaling_coefficient
            else:
                self.y_sd_prior_sd = self.y_train_sd * self.y_sd_prior_sd_scaling_coefficient

        if self.inference_name == 'MetropolisHastings':
            self.proposal_map = {}
            if self.mh_proposal_sd is None:
                if self.standardize_response:
                    self.mh_proposal_sd = self.prior_sd_scaling_coefficient
                else:
                    self.mh_proposal_sd = self.y_train_sd * self.prior_sd_scaling_coefficient

        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Alias prior widths for use in multivariate mode
                self.intercept_joint_sd = self.intercept_prior_sd
                self.coef_joint_sd = self.coef_prior_sd
                self.irf_param_joint_sd = self.irf_param_prior_sd
                self.ranef_to_fixef_joint_sd_ratio = self.ranef_to_fixef_prior_sd_ratio

                # Define initialization constants
                self.intercept_prior_sd_tf = tf.constant(float(self.intercept_prior_sd), dtype=self.FLOAT_TF)
                self.intercept_posterior_sd_init = self.intercept_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.intercept_ranef_prior_sd_tf = self.intercept_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio
                self.intercept_ranef_posterior_sd_init = self.intercept_posterior_sd_init * self.ranef_to_fixef_prior_sd_ratio

                self.coef_prior_sd_tf = tf.constant(float(self.coef_prior_sd), dtype=self.FLOAT_TF)
                self.coef_posterior_sd_init = self.coef_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.coef_ranef_prior_sd_tf = self.coef_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio
                self.coef_ranef_posterior_sd_init = self.coef_posterior_sd_init * self.ranef_to_fixef_prior_sd_ratio

                self.irf_param_prior_sd_tf = tf.constant(float(self.irf_param_prior_sd), dtype=self.FLOAT_TF)
                self.irf_param_posterior_sd_init = self.irf_param_prior_sd_tf * self.posterior_to_prior_sd_ratio
                self.irf_param_ranef_prior_sd_tf = self.irf_param_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio
                self.irf_param_ranef_posterior_sd_init = self.irf_param_posterior_sd_init * self.ranef_to_fixef_prior_sd_ratio

                self.y_sd_prior_sd_tf = tf.constant(float(self.y_sd_prior_sd), dtype=self.FLOAT_TF)
                self.y_sd_posterior_sd_init = self.y_sd_prior_sd_tf * self.posterior_to_prior_sd_ratio

                self.y_skewness_prior_sd_tf = tf.constant(float(self.y_skewness_prior_sd), dtype=self.FLOAT_TF)
                self.y_skewness_posterior_sd_init = self.y_skewness_prior_sd_tf * self.posterior_to_prior_sd_ratio

                self.y_tailweight_prior_loc_tf = tf.constant(1., dtype=self.FLOAT_TF)
                self.y_tailweight_posterior_loc_init = tf.constant(1., dtype=self.FLOAT_TF)
                self.y_tailweight_prior_sd_tf = tf.constant(float(self.y_tailweight_prior_sd), dtype=self.FLOAT_TF)
                self.y_tailweight_posterior_sd_init = self.y_tailweight_prior_sd_tf * self.posterior_to_prior_sd_ratio

                if self.constraint.lower() == 'softplus':
                    self.intercept_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.intercept_prior_sd_tf)
                    self.intercept_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.intercept_posterior_sd_init)
                    self.intercept_ranef_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.intercept_ranef_prior_sd_tf)
                    self.intercept_ranef_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.intercept_ranef_posterior_sd_init)

                    self.coef_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.coef_prior_sd_tf)
                    self.coef_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.coef_posterior_sd_init)
                    self.coef_ranef_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.coef_ranef_prior_sd_tf)
                    self.coef_ranef_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.coef_ranef_posterior_sd_init)

                    self.irf_param_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.irf_param_prior_sd_tf)
                    self.irf_param_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.irf_param_posterior_sd_init)
                    self.irf_param_ranef_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.irf_param_ranef_prior_sd_tf)
                    self.irf_param_ranef_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.irf_param_ranef_posterior_sd_init)

                    self.y_sd_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_sd_prior_sd_tf)
                    self.y_sd_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_sd_posterior_sd_init)

                    self.y_skewness_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_skewness_prior_sd_tf)
                    self.y_skewness_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_skewness_posterior_sd_init)

                    self.y_tailweight_prior_loc_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_tailweight_prior_loc_tf)
                    self.y_tailweight_posterior_loc_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_tailweight_posterior_loc_init)
                    self.y_tailweight_prior_sd_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_tailweight_prior_sd_tf)
                    self.y_tailweight_posterior_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_tailweight_posterior_sd_init)

                elif self.constraint.lower() == 'abs':
                    self.intercept_prior_sd_unconstrained = self.intercept_prior_sd_tf
                    self.intercept_posterior_sd_init_unconstrained = self.intercept_posterior_sd_init
                    self.intercept_ranef_prior_sd_unconstrained = self.intercept_ranef_prior_sd_tf
                    self.intercept_ranef_posterior_sd_init_unconstrained = self.intercept_ranef_posterior_sd_init

                    self.coef_prior_sd_unconstrained = self.coef_prior_sd_tf
                    self.coef_posterior_sd_init_unconstrained = self.coef_posterior_sd_init
                    self.coef_ranef_prior_sd_unconstrained = self.coef_ranef_prior_sd_tf
                    self.coef_ranef_posterior_sd_init_unconstrained = self.coef_ranef_posterior_sd_init

                    self.irf_param_prior_sd_unconstrained = self.irf_param_prior_sd_tf
                    self.irf_param_posterior_sd_init_unconstrained = self.irf_param_posterior_sd_init
                    self.irf_param_ranef_prior_sd_unconstrained = self.irf_param_ranef_prior_sd_tf
                    self.irf_param_ranef_posterior_sd_init_unconstrained = self.irf_param_ranef_posterior_sd_init

                    self.y_sd_prior_sd_unconstrained = self.y_sd_prior_sd_tf
                    self.y_sd_posterior_sd_init_unconstrained = self.y_sd_posterior_sd_init

                    self.y_skewness_prior_sd_unconstrained = self.y_skewness_prior_sd_tf
                    self.y_skewness_posterior_sd_init_unconstrained = self.y_skewness_posterior_sd_init

                    self.y_tailweight_prior_loc_unconstrained = self.y_tailweight_prior_loc_tf
                    self.y_tailweight_posterior_loc_init_unconstrained = self.y_tailweight_posterior_loc_init
                    self.y_tailweight_prior_sd_unconstrained = self.y_tailweight_prior_sd_tf
                    self.y_tailweight_posterior_sd_init_unconstrained = self.y_tailweight_posterior_sd_init

                else:
                    raise ValueError('Unrecognized constraint function "%s"' % self.constraint)

    def _pack_metadata(self):
        md = super(CDRBayes, self)._pack_metadata()
        for kwarg in CDRBayes._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(CDRBayes, self)._unpack_metadata(md)

        for kwarg in CDRBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            stderr('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def initialize_intercept(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.variational():
                        # Posterior distribution
                        intercept_q_loc = tf.Variable(
                            self.intercept_init_tf,
                            name='intercept_q_loc'
                        )

                        intercept_q_scale = tf.Variable(
                            self.intercept_posterior_sd_init_unconstrained,
                            name='intercept_q_scale'
                        )

                        intercept_q = Normal(
                            loc=intercept_q_loc,
                            scale=self.constraint_fn(intercept_q_scale),
                            name='intercept_q'
                        )

                        intercept_summary = intercept_q.mean()

                        if self.declare_priors_fixef:
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
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.variational():
                        # Posterior distribution
                        intercept_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels],
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.zeros([rangf_n_levels], dtype=self.FLOAT_TF),
                            name='intercept_q_loc_by_%s' % ran_gf
                        )

                        intercept_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels],
                            #     mean=self.intercept_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([rangf_n_levels], dtype=self.FLOAT_TF) * self.intercept_ranef_posterior_sd_init_unconstrained,
                            name='intercept_q_scale_by_%s' % ran_gf
                        )

                        intercept_q = Normal(
                            loc=intercept_q_loc,
                            scale=self.constraint_fn(intercept_q_scale),
                            name='intercept_q_by_%s' % ran_gf
                        )

                        intercept_summary = intercept_q.mean()

                        if self.declare_priors_ranef:
                            # Prior distribution
                            intercept = Normal(
                                sample_shape=[rangf_n_levels],
                                loc=0.,
                                scale=self.intercept_ranef_prior_sd_tf,
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
                            scale=self.intercept_ranef_prior_sd_tf,
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

                self.MAP_map[intercept] = intercept_summary

                return intercept, intercept_summary

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.variational():
                        # Posterior distribution
                        coefficient_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [len(coef_ids)],
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.zeros([len(coef_ids)], dtype=self.FLOAT_TF),
                            name='coefficient_q_loc'
                        )

                        coefficient_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [len(coef_ids)],
                            #     mean=self.coef_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([len(coef_ids)], dtype=self.FLOAT_TF) * self.coef_posterior_sd_init_unconstrained,
                            name='coefficient_q_scale'
                        )

                        coefficient_q = Normal(
                            loc=coefficient_q_loc,
                            scale=self.constraint_fn(coefficient_q_scale),
                            name='coefficient_q'
                        )
                        coefficient_summary = coefficient_q.mean()

                        if self.declare_priors_fixef:
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
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.variational():
                        # Posterior distribution
                        coefficient_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels, len(coef_ids)],
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.zeros([rangf_n_levels, len(coef_ids)], dtype=self.FLOAT_TF),
                            name='coefficient_q_loc_by_%s' % ran_gf
                        )

                        coefficient_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels, len(coef_ids)],
                            #     mean=self.coef_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([rangf_n_levels, len(coef_ids)], dtype=self.FLOAT_TF) * self.coef_ranef_posterior_sd_init_unconstrained,
                            name='coefficient_q_scale_by_%s' % ran_gf
                        )

                        coefficient_q = Normal(
                            loc=coefficient_q_loc,
                            scale=self.constraint_fn(coefficient_q_scale),
                            name='coefficient_q_by_%s' % ran_gf
                        )
                        coefficient_summary = coefficient_q.mean()

                        if self.declare_priors_ranef:
                            # Prior distribution
                            coefficient = Normal(
                                sample_shape=[rangf_n_levels, len(coef_ids)],
                                loc=0.,
                                scale=self.coef_ranef_prior_sd_tf,
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
                            scale=self.coef_ranef_prior_sd_tf,
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

                self.MAP_map[coefficient] = coefficient_summary

                return coefficient, coefficient_summary

    def initialize_interaction(self, interaction_ids=None, ran_gf=None):
        if interaction_ids is None:
            interaction_ids = self.interaction_names
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.variational():
                        # Posterior distribution
                        interaction_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [len(interaction_ids)],
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.zeros([len(interaction_ids)], dtype=self.FLOAT_TF),
                            name='interaction_q_loc'
                        )

                        interaction_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [len(interaction_ids)],
                            #     mean=self.coef_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([len(interaction_ids)], dtype=self.FLOAT_TF) * self.coef_posterior_sd_init_unconstrained,
                            name='interaction_q_scale'
                        )

                        interaction_q = Normal(
                            loc=interaction_q_loc,
                            scale=self.constraint_fn(interaction_q_scale),
                            name='interaction_q'
                        )
                        interaction_summary = interaction_q.mean()

                        if self.declare_priors_fixef:
                            # Prior distribution
                            interaction = Normal(
                                sample_shape=[len(interaction_ids)],
                                loc=0.,
                                scale=self.coef_prior_sd_tf,
                                name='interaction'
                            )
                            self.inference_map[interaction] = interaction_q
                        else:
                            interaction = interaction_q
                    else:
                        # Prior distribution
                        interaction = Normal(
                            sample_shape=[len(interaction_ids)],
                            loc=0.,
                            scale=self.coef_prior_sd_tf,
                            name='interaction'
                        )

                        # Posterior distribution
                        interaction_q_samples = tf.Variable(
                            tf.zeros((self.n_samples, len(interaction_ids)), dtype=self.FLOAT_TF),
                            name='interaction_q_samples'
                        )
                        interaction_q = Empirical(
                            params=interaction_q_samples,
                            name='interaction_q'
                        )

                        if self.inference_name == 'MetropolisHastings':
                            # Proposal distribution
                            interaction_proposal = Normal(
                                loc=interaction,
                                scale=self.mh_proposal_sd,
                                name='interaction_proposal'
                            )
                            self.proposal_map[interaction] = interaction_proposal

                        interaction_summary = interaction_q.params[self.global_batch_step - 1]

                        self.inference_map[interaction] = interaction_q
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.variational():
                        # Posterior distribution
                        interaction_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels, len(interaction_ids)],
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.zeros([rangf_n_levels, len(interaction_ids)], dtype=self.FLOAT_TF),
                            name='interaction_q_loc_by_%s' % ran_gf
                        )

                        interaction_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels, len(interaction_ids)],
                            #     mean=self.coef_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([rangf_n_levels, len(interaction_ids)], dtype=self.FLOAT_TF) * self.coef_ranef_posterior_sd_init_unconstrained,
                            name='interaction_q_scale_by_%s' % ran_gf
                        )

                        interaction_q = Normal(
                            loc=interaction_q_loc,
                            scale=self.constraint_fn(interaction_q_scale),
                            name='interaction_q_by_%s' % ran_gf
                        )
                        interaction_summary = interaction_q.mean()

                        if self.declare_priors_ranef:
                            # Prior distribution
                            interaction = Normal(
                                sample_shape=[rangf_n_levels, len(interaction_ids)],
                                loc=0.,
                                scale=self.coef_ranef_prior_sd_tf,
                                name='interaction_by_%s' % ran_gf
                            )
                            self.inference_map[interaction] = interaction_q
                        else:
                            interaction = interaction_q

                    else:
                        # Prior distribution
                        interaction = Normal(
                            sample_shape=[rangf_n_levels, len(interaction_ids)],
                            loc=0.,
                            scale=self.coef_ranef_prior_sd_tf,
                            name='interaction_by_%s' % ran_gf
                        )

                        # Posterior distribution
                        interaction_q = Empirical(
                            params=tf.Variable(
                                tf.zeros(
                                    (self.n_samples, rangf_n_levels, len(interaction_ids)),
                                    dtype=self.FLOAT_TF
                                ),
                                name='interaction_q_by_%s_samples' % ran_gf
                            ),
                            name='interaction_q_by_%s' % ran_gf
                        )

                        if self.inference_name == 'MetropolisHastings':
                            # Proposal distribution
                            interaction_proposal = Normal(
                                loc=interaction,
                                scale=self.mh_proposal_sd,
                                name='interaction_proposal_by_%s' % ran_gf
                            )
                            self.proposal_map[interaction] = interaction_proposal

                        interaction_summary = interaction_q.params[self.global_batch_step - 1]

                        self.inference_map[interaction] = interaction_q

                self.MAP_map[interaction] = interaction_summary

                return interaction, interaction_summary

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0., ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    if self.variational():
                        # Posterior distribution
                        param_q_loc = tf.Variable(
                            # tf.random_normal(
                            #     [1, len(ids)],
                            #     mean=mean,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([1, len(ids)], dtype=self.FLOAT_TF) * mean,
                            name=sn('%s_q_loc_%s' % (param_name, '-'.join(ids)))
                        )

                        param_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [1, len(ids)],
                            #     mean=self.irf_param_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([1, len(ids)], dtype=self.FLOAT_TF) * self.irf_param_posterior_sd_init_unconstrained,
                            name=sn('%s_q_scale_%s' % (param_name, '-'.join(ids)))
                        )

                        param_q = Normal(
                            loc=param_q_loc,
                            scale=self.constraint_fn(param_q_scale),
                            name=sn('%s_q_%s' % (param_name, '-'.join(ids)))
                        )

                        param_summary = param_q.mean()

                        if self.declare_priors_fixef:
                            # Prior distribution
                            param = Normal(
                                loc=mean,
                                scale=self.irf_param_prior_sd,
                                name=sn('%s_%s' % (param_name, '-'.join(ids)))
                            )
                            self.inference_map[param] = param_q
                        else:
                            param = param_q
                    else:
                        # Prior distribution
                        param = Normal(
                            loc=mean,
                            scale=self.irf_param_prior_sd,
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
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    if self.variational():
                        # Posterior distribution
                        param_q_loc = tf.Variable(
                            tf.random_normal(
                                [rangf_n_levels, len(ids)],
                                mean=0.,
                                stddev=self.epsilon,
                                dtype=self.FLOAT_TF
                            ),
                            name=sn('%s_q_loc_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                        )

                        param_q_scale = tf.Variable(
                            # tf.random_normal(
                            #     [rangf_n_levels, len(ids)],
                            #     mean=self.irf_param_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            tf.ones([rangf_n_levels, len(ids)], dtype=self.FLOAT_TF) * self.irf_param_ranef_posterior_sd_init_unconstrained,
                            name=sn('%s_q_scale_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                        )

                        param_q = Normal(
                            loc=param_q_loc,
                            scale=self.constraint_fn(param_q_scale),
                            name=sn('%s_q_%s_by_%s' % (param_name, '-'.join(ids), ran_gf))
                        )

                        param_summary = param_q.mean()

                        if self.declare_priors_ranef:
                            # Prior distribution
                            param = Normal(
                                sample_shape=[rangf_n_levels, len(ids)],
                                loc=0.,
                                scale=self.irf_param_ranef_prior_sd_tf,
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
                            scale=self.irf_param_ranef_prior_sd_tf,
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

                self.MAP_map[param] = param_summary

                return param, param_summary

    def initialize_joint_distribution(self, means, sds, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                dim = int(means.shape[0])

                if self.variational():
                    # Posterior distribution
                    joint_q_loc = tf.Variable(
                        # tf.random_normal(
                        #     [dim],
                        #     mean=means,
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.ones([dim], dtype=self.FLOAT_TF) * means,
                        name='joint_q_loc' if ran_gf is None else 'joint_q_loc_by_%s' %ran_gf
                    )

                    # Construct cholesky decomposition of initial covariance using sds, then use for initialization
                    n_scale = int(dim * (dim + 1) / 2)
                    if ran_gf is not None:
                        sds *= self.ranef_to_fixef_prior_sd_ratio
                    cholesky = tf.diag(sds)
                    tril_ix = np.ravel_multi_index(
                        np.tril_indices(dim),
                        (dim, dim)
                    )
                    scale_init = tf.gather(tf.reshape(cholesky, [dim * dim]), tril_ix)

                    scale_posterior_init = scale_init * self.posterior_to_prior_sd_ratio

                    joint_q_scale = tf.Variable(
                        # tf.random_normal(
                        #     [n_scale],
                        #     mean=scale_posterior_init,
                        #     stddev=self.init_sd,
                        #     dtype=self.FLOAT_TF
                        # ),
                        tf.ones([n_scale], dtype=self.FLOAT_TF) * scale_posterior_init,
                        name='joint_q_scale' if ran_gf is None else 'joint_q_scale_by_%s' %ran_gf
                    )

                    joint_q = MultivariateNormalTriL(
                        loc=joint_q_loc,
                        scale_tril=tf.contrib.distributions.fill_triangular(joint_q_scale),
                        name='joint_q' if ran_gf is None else 'joint_q_by_%s' %ran_gf
                    )

                    joint_summary = joint_q.mean()

                    if (ran_gf is None and self.declare_priors_fixef) or (ran_gf is not None and self.declare_priors_ranef):
                        # Prior distribution
                        joint = MultivariateNormalTriL(
                            loc=means,
                            scale_tril=tf.contrib.distributions.fill_triangular(scale_init),
                            name='joint' if ran_gf is None else 'joint_by_%s' %ran_gf
                        )

                        self.inference_map[joint] = joint_q
                    else:
                        joint = joint_q

                else:
                    # Construct cholesky decomposition of initial covariance using sds, then use for initialization
                    cholesky = tf.diag(sds)
                    tril_ix = np.ravel_multi_index(
                        np.tril_indices(dim),
                        (dim, dim)
                    )
                    scale_init = tf.gather(tf.reshape(cholesky, [dim * dim]), tril_ix)

                    # Prior distribution
                    joint = MultivariateNormalTriL(
                        loc=means,
                        scale_tril=tf.contrib.distributions.fill_triangular(scale_init),
                        name='joint' if ran_gf is None else 'joint_by_%s' %ran_gf
                    )

                    # Posterior distribution
                    joint_q_samples = tf.Variable(
                        tf.ones((self.n_samples), dtype=self.FLOAT_TF) * means,
                        name='joint_q_samples'  if ran_gf is None else 'joint_q_samples_by_%s' %ran_gf
                    )

                    joint_q = Empirical(
                        params=joint_q_samples,
                        name='joint_q' if ran_gf is None else 'joint_q_by_%s' %ran_gf
                    )

                    if self.inference_name == 'MetropolisHastings':
                        # Proposal distribution
                        joint_proposal = Normal(
                            loc=joint,
                            scale=self.mh_proposal_sd,
                            name='joint_proposal' if ran_gf is None else 'joint_q_proposal_by_%s' %ran_gf
                        )
                        self.proposal_map[joint] = joint_proposal

                    joint_summary = joint_q.params[self.global_batch_step - 1]

                    self.inference_map[joint] = joint_q

                self.MAP_map[joint] = joint_summary

                return joint, joint_summary

    def _initialize_output_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.y_sd_trainable:
                    y_sd_init_unconstrained = self.y_sd_init_unconstrained

                    if self.variational():
                        # Posterior distribution
                        y_sd_loc_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=y_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            y_sd_init_unconstrained,
                            name='y_sd_loc_q'
                        )

                        y_sd_scale_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=self.y_sd_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            self.y_sd_posterior_sd_init_unconstrained,
                            name='y_sd_scale_q'
                        )
                        y_sd_q = Normal(
                            loc=y_sd_loc_q,
                            scale=self.constraint_fn(y_sd_scale_q),
                            name='y_sd_q'
                        )
                        y_sd_summary = y_sd_q.mean()

                        if self.declare_priors_fixef:
                            # Prior distribution
                            y_sd = Normal(
                                loc=y_sd_init_unconstrained,
                                scale=self.y_sd_prior_sd_tf,
                                name='y_sd'
                            )
                            self.inference_map[y_sd] = y_sd_q
                        else:
                            y_sd = y_sd_q
                    else:
                        # Prior distribution
                        y_sd = Normal(
                            loc=y_sd_init_unconstrained,
                            scale=self.y_sd_prior_sd_tf,
                            name='y_sd'
                        )

                        # Posterior distribution
                        y_sd_q_samples = tf.Variable(
                            tf.zeros([self.n_samples], dtype=self.FLOAT_TF),
                            name=sn('y_sd_q_samples')
                        )
                        y_sd_q = Empirical(
                            params=y_sd_q_samples,
                            name=sn('y_sd_q')
                        )

                        if self.inference_name == 'MetropolisHastings':
                            # Proposal distribution
                            y_sd_proposal = Normal(
                                loc=y_sd,
                                scale=self.mh_proposal_sd,
                                name=sn('y_sd_proposal')
                            )
                            self.proposal_map[y_sd] = y_sd_proposal

                        y_sd_summary = y_sd_q.params[self.global_batch_step - 1]

                        self.inference_map[y_sd] = y_sd_q

                    self.MAP_map[y_sd] = y_sd_summary

                    y_sd = self.constraint_fn(y_sd)
                    y_sd_summary = self.constraint_fn(y_sd_summary)

                    tf.summary.scalar(
                        'error/y_sd',
                        y_sd_summary,
                        collections=['params']
                    )

                else:
                    stderr('Fixed y scale: %s\n' % self.y_sd_init)
                    y_sd = self.y_sd_init_tf
                    y_sd_summary = y_sd

                self.y_sd = y_sd
                self.y_sd_summary = y_sd_summary

                if self.asymmetric_error:
                    if self.variational():
                        # Posterior distributions
                        y_skewness_loc_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=0.,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            0.,
                            name='y_skewness_q_loc'
                        )
                        y_skewness_scale_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=self.y_skewness_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            self.y_skewness_posterior_sd_init_unconstrained,
                            name='y_skewness_q_loc'
                        )

                        self.y_skewness_q = Normal(
                            loc=y_skewness_loc_q,
                            scale=self.constraint_fn(y_skewness_scale_q),
                            name='y_skewness_q'
                        )
                        self.y_skewness_summary = self.y_skewness_q.mean()
                        tf.summary.scalar(
                            'error/y_skewness_summary',
                            self.y_skewness_summary,
                            collections=['params']
                        )

                        y_tailweight_loc_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=self.y_tailweight_posterior_loc_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            self.y_tailweight_posterior_loc_init_unconstrained,
                            name='y_tailweight_q_loc'
                        )
                        y_tailweight_scale_q = tf.Variable(
                            # tf.random_normal(
                            #     [],
                            #     mean=self.y_tailweight_posterior_sd_init_unconstrained,
                            #     stddev=self.init_sd,
                            #     dtype=self.FLOAT_TF
                            # ),
                            self.y_tailweight_posterior_sd_init_unconstrained,
                            name='y_tailweight_q_scale'
                        )
                        self.y_tailweight_q = Normal(
                            loc=y_tailweight_loc_q,
                            scale=self.constraint_fn(y_tailweight_scale_q),
                            name='y_tailweight_q'
                        )
                        self.y_tailweight_summary = self.y_tailweight_q.mean()
                        tf.summary.scalar(
                            'error/y_tailweight',
                            self.constraint_fn(self.y_tailweight_summary),
                            collections=['params']
                        )

                        if self.declare_priors_fixef:
                            # Prior distributions
                            self.y_skewness = Normal(
                                loc=0.,
                                scale=self.y_skewness_prior_sd_tf,
                                name='y_skewness'
                            )
                            self.y_tailweight = Normal(
                                loc=self.y_tailweight_prior_loc_unconstrained,
                                scale=self.y_tailweight_prior_sd_tf,
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
                            scale=self.y_skewness_prior_sd_tf,
                            name='y_skewness'
                        )
                        self.y_tailweight = Normal(
                            loc=self.y_tailweight_prior_loc_unconstrained,
                            scale=self.y_tailweight_prior_sd_tf,
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
                            'error/y_skewness',
                            self.y_skewness_summary,
                            collections=['params']
                        )
                        tf.summary.scalar(
                            'error/y_tailweight',
                            self.constraint_fn(self.y_tailweight_summary),
                            collections=['params']
                        )

                        self.inference_map[self.y_skewness] = self.y_skewness_q
                        self.inference_map[self.y_tailweight] = self.y_tailweight_q

                    self.MAP_map[self.y_skewness] = self.y_skewness_summary
                    self.MAP_map[self.y_tailweight] = self.y_tailweight_summary

                    if self.standardize_response:
                        self.out_standardized = SinhArcsinh(
                            loc=self.out,
                            scale=y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='output_standardized'
                        )
                        self.err_dist_standardized = SinhArcsinh(
                            loc=0.,
                            scale=y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='err_dist_standardized'
                        )
                        self.err_dist_standardized_summary = SinhArcsinh(
                            loc=0.,
                            scale=y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.constraint_fn(self.y_tailweight_summary),
                            name='err_dist_standardized_summary'
                        )
                        self.MAP_map[self.out_standardized] = self.out_mean

                        self.out = SinhArcsinh(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='output'
                        )
                        self.err_dist = SinhArcsinh(
                            loc=0.,
                            scale=y_sd * self.y_train_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='err_dist'
                        )
                        self.err_dist_summary = SinhArcsinh(
                            loc=0.,
                            scale=y_sd_summary * self.y_train_sd,
                            skewness=self.y_skewness_summary,
                            tailweight=self.constraint_fn(self.y_tailweight_summary),
                            name='err_dist_summary'
                        )
                        self.MAP_map[self.out] = self.out_mean * self.y_train_sd + self.y_train_mean
                    else:
                        self.out = SinhArcsinh(
                            loc=self.out,
                            scale=y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='output'
                        )
                        self.err_dist = SinhArcsinh(
                            loc=0.,
                            scale=y_sd,
                            skewness=self.y_skewness,
                            tailweight=self.constraint_fn(self.y_tailweight),
                            name='err_dist'
                        )
                        self.err_dist_summary = SinhArcsinh(
                            loc=0.,
                            scale=y_sd_summary,
                            skewness=self.y_skewness_summary,
                            tailweight=self.constraint_fn(self.y_tailweight_summary),
                            name='err_dist_summary'
                        )
                        self.MAP_map[self.out] = self.out_mean

                else:
                    if self.standardize_response:
                        self.out_standardized = Normal(
                            loc=self.out,
                            scale=self.y_sd,
                            name='output_standardized'
                        )
                        self.err_dist_standardized = Normal(
                            loc=0.,
                            scale=self.y_sd,
                            name='err_dist_standardized'
                        )
                        self.err_dist_standardized_summary = Normal(
                            loc=0.,
                            scale=self.y_sd_summary,
                            name='err_dist_standardized_summary'
                        )

                        self.MAP_map[self.out_standardized] = self.out_mean

                        self.out = Normal(
                            loc=self.out * self.y_train_sd + self.y_train_mean,
                            scale=self.y_sd * self.y_train_sd,
                            name='output'
                        )
                        self.err_dist = Normal(
                            loc=0.,
                            scale=self.y_sd * self.y_train_sd,
                            name='err_dist'
                        )
                        self.err_dist_summary = Normal(
                            loc=0.,
                            scale=self.y_sd_summary * self.y_train_sd,
                            name='err_dist_summary'
                        )
                        self.MAP_map[self.out] = self.out_mean * self.y_train_sd + self.y_train_mean
                    else:
                        self.out = Normal(
                            loc=self.out,
                            scale=self.y_sd,
                            name='output'
                        )
                        self.err_dist = Normal(
                            loc=0.,
                            scale=self.y_sd,
                            name='err_dist'
                        )
                        self.err_dist_summary = Normal(
                            loc=0.,
                            scale=self.y_sd_summary,
                            name='err_dist_summary'
                        )
                        self.MAP_map[self.out] = self.out_mean

                self.err_dist_plot = tf.exp(self.err_dist.log_prob(self.support[None,...]))
                self.err_dist_plot_summary = tf.exp(self.err_dist_summary.log_prob(self.support[None,...]))
                self.err_dist_lb = self.err_dist_summary.quantile(.025)
                self.err_dist_ub = self.err_dist_summary.quantile(.975)

                empirical_quantiles = tf.linspace(0., 1., self.n_errors)
                if self.standardize_response:
                    self.err_dist_standardized_theoretical_quantiles = self.err_dist_standardized.quantile(empirical_quantiles)
                    self.err_dist_standardized_theoretical_cdf = self.err_dist_standardized.cdf(self.errors)
                    self.err_dist_standardized_summary_theoretical_quantiles = self.err_dist_standardized_summary.quantile(empirical_quantiles)
                    self.err_dist_standardized_summary_theoretical_cdf = self.err_dist_standardized_summary.cdf(self.errors)
                self.err_dist_theoretical_quantiles = self.err_dist.quantile(empirical_quantiles)
                self.err_dist_theoretical_cdf = self.err_dist.cdf(self.errors)
                self.err_dist_summary_theoretical_quantiles = self.err_dist_summary.quantile(empirical_quantiles)
                self.err_dist_summary_theoretical_cdf = self.err_dist_summary.cdf(self.errors)

    def initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_output_model()

                if self.standardize_response:
                    out_model = self.out_standardized
                    y = (self.y - self.y_train_mean) / self.y_train_sd
                else:
                    out_model = self.out
                    y = self.y

                self.optim = self._initialize_optimizer(self.optim_name)
                if self.variational():
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={out_model: y})
                    self.inference.initialize(
                        n_samples=self.n_samples,
                        n_iter=self.n_iter,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={out_model: self.minibatch_scale},
                        optimizer=self.optim
                    )
                elif self.inference_name == 'MetropolisHastings':
                    self.inference = getattr(ed, self.inference_name)(self.inference_map, self.proposal_map, data={out_model: y})
                    self.inference.initialize(
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={out_model: self.minibatch_scale}
                    )
                else:
                    self.inference = getattr(ed,self.inference_name)(self.inference_map, data={out_model: y})
                    self.inference.initialize(
                        step_size=self.lr,
                        n_print=self.n_train_minibatch * self.log_freq,
                        logdir=self.outdir + '/tensorboard/edward',
                        log_timestamp=False,
                        scale={out_model: self.minibatch_scale}
                    )

                ## Set up posteriors and MAP estimates
                self.X_conv_prior = self.X_conv
                self.X_conv_post = ed.copy(self.X_conv, self.inference_map)
                self.X_conv_MAP = ed.copy(self.X_conv, self.MAP_map, scope='MAP')

                if self.standardize_response:
                    y_standardized = (self.y - self.y_train_mean) / self.y_train_sd
                    self.X_conv_standardized_scaled_prior = self.X_conv_scaled
                    self.X_conv_standardized_scaled_post = ed.copy(self.X_conv_scaled, self.inference_map)
                    self.X_conv_standardized_scaled_MAP = ed.copy(self.X_conv_scaled, self.MAP_map, scope='MAP')
                    self.X_conv_scaled_prior = self.X_conv_scaled * self.y_train_sd
                    self.X_conv_scaled_post = ed.copy(self.X_conv_scaled * self.y_train_sd, self.inference_map)
                    self.X_conv_scaled_MAP = ed.copy(self.X_conv_scaled * self.y_train_sd, self.MAP_map, scope='MAP')

                    self.out_standardized_prior = self.out_standardized
                    self.out_standardized_post = ed.copy(self.out_standardized, self.inference_map)
                    self.out_standardized_MAP = tf.identity(self.MAP_map[self.out_standardized])
                    self.out_standardized_MAP = ed.copy(self.out_standardized_MAP, self.MAP_map, scope='MAP')
                    self.ll_standardized_prior = self.out_standardized_prior.log_prob(y_standardized)
                    self.ll_standardized_post = self.out_standardized_post.log_prob(y_standardized)
                    self.ll_standardized_MAP = ed.copy(self.out_standardized.log_prob(y_standardized), self.MAP_map, scope='MAP')
                else:
                    self.X_conv_scaled_prior = self.X_conv_scaled
                    self.X_conv_scaled_post = ed.copy(self.X_conv_scaled, self.inference_map)
                    self.X_conv_scaled_MAP = ed.copy(self.X_conv_scaled, self.MAP_map, scope='MAP')

                self.out_prior = self.out
                self.out_post = ed.copy(self.out, self.inference_map)
                self.out_MAP = tf.identity(self.MAP_map[self.out])
                self.out_MAP = ed.copy(self.out_MAP, self.MAP_map, scope='MAP')
                self.ll_prior = self.out_prior.log_prob(self.y)
                self.ll_post = self.out_post.log_prob(self.y)
                self.ll_MAP = ed.copy(self.out.log_prob(self.y), self.MAP_map, scope='MAP')

                for x in self.irf_mc:
                    for a in self.irf_mc[x]:
                        for b in self.irf_mc[x][a]:
                            self.irf_mc[x][a][b] = ed.copy(self.irf_mc[x][a][b], self.inference_map)
                for x in self.irf_integral_tensors:
                    self.irf_integral_tensors[x] = ed.copy(self.irf_integral_tensors[x], self.inference_map)
                if self.pc:
                    for x in self.src_irf_mc:
                        for a in self.src_irf_mc[x]:
                            for b in self.src_irf_mc[x][a]:
                                self.src_irf_mc[x][a][b] = ed.copy(self.src_irf_mc[x][a][b], self.inference_map)
                    for x in self.src_irf_integral_tensors:
                        self.src_irf_integral_tensors[x] = ed.copy(self.src_irf_integral_tensors[x], self.inference_map)

    # Overload this method to perform parameter sampling and compute credible intervals
    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples_eval

        alpha = 100 - float(level)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if fixed:
                    param_vector = self.parameter_table_fixed_values
                else:
                    param_vector = self.parameter_table_random_values

            samples = [param_vector.eval(session=self.sess) for _ in range(n_samples)]
            samples = np.stack(samples, axis=1)

            mean = samples.mean(axis=1)
            lower = np.percentile(samples, alpha / 2, axis=1)
            upper = np.percentile(samples, 100 - (alpha / 2), axis=1)

            out = np.stack([mean, lower, upper], axis=1)

            return out

    def extract_irf_integral(self, terminal_name, rangf=None, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        if n_samples is None:
            n_samples = self.n_samples_eval
        if n_time_units is None:
            n_time_units = self.t_delta_limit
        with self.sess.as_default():
            with self.sess.graph.as_default():
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

                alpha = 100 - float(level)

                if terminal_name in self.irf_integral_tensors:
                    posterior = self.irf_integral_tensors[terminal_name]
                else:
                    posterior = self.src_irf_integral_tensors[terminal_name]

                samples = [self.sess.run(posterior, feed_dict=fd) for _ in range(n_samples)]
                samples = np.stack(samples, axis=1)

                mean = samples.mean(axis=1)
                lower = np.percentile(samples, alpha / 2, axis=1)
                upper = np.percentile(samples, 100 - (alpha / 2), axis=1)

                return (mean, lower, upper)

    # Overload this method to use posterior distribution
    def _initialize_parameter_tables(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                super(CDRBayes, self)._initialize_parameter_tables()

                self.parameter_table_fixed_values = ed.copy(self.parameter_table_fixed_values, self.inference_map)
                if len(self.rangf) > 0:
                    self.parameter_table_random_values = ed.copy(self.parameter_table_random_values, self.inference_map)




    #####################################################
    #
    #  Public methods
    #
    ######################################################

    def variational(self):
        """
        Report whether the CDR model uses variational Bayes.

        :return: ``bool``; ``True`` if the model is variational, ``False`` otherwise.
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

    def ci_curve(
            self,
            posterior,
            rangf=None,
            level=95,
            n_samples=None,
            support_start=0.,
            n_time_units=None,
            n_time_points=1000
    ):
        """
        Extract an IRF with Monte Carlo credible intervals for plotting

        :param posterior: the IRF node in the model's graph
        :param rangf: ``numpy`` array or ``None``; random grouping factor values for which to compute CI curves. If ``None``, only use fixed effects.
        :param level: ``float``; level of the credible interval.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw. If ``None``, use model defaults.
        :param support_start: ``float``; starting point for the support vector.
        :param n_time_units: ``float``; number of time units over which to plot the curve.
        :param n_time_points: ``float``; number of points to use in the plot.
        :return: ``tuple`` of 4 ``numpy`` arrays; mean, lower bound, and upper bound at the desired level for each plot point, plus the full array of samples.
        """

        if n_samples is None:
            n_samples = self.n_samples_eval
        if n_time_units is None:
            n_time_units = self.t_delta_limit
        if self.pc:
            n_impulse = len(self.src_impulse_names)
        else:
            n_impulse = len(self.impulse_names)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.support_start: support_start,
                    self.n_time_units: n_time_units,
                    self.n_time_points: n_time_points,
                    self.time_y: np.ones((1,)) * n_time_units,
                    self.time_X: np.zeros((1, self.history_length, n_impulse))
                }
                
                if rangf is not None:
                    fd[self.gf_y] = rangf

                alpha = 100-float(level)

                samples = [self.sess.run(posterior, feed_dict=fd) for _ in range(n_samples)]
                samples = np.concatenate(samples, axis=2)

                mean = samples.mean(axis=2)
                lower = np.percentile(samples, alpha/2, axis=2)
                upper = np.percentile(samples, 100-(alpha/2), axis=2)

                if rangf is not None:
                    n = len(rangf)
                    if len(mean) == 1 and n > 1:
                        mean = np.repeat(mean, n, axis=0)
                    if len(lower) == 1 and n > 1:
                        lower = np.repeat(lower, n, axis=0)
                    if len(upper) == 1 and n > 1:
                        upper = np.repeat(upper, n, axis=0)

                return (mean, lower, upper, samples)

    def error_quantiles_mc(
            self,
            errors,
            level=95,
            n_samples=None
    ):
        """
        Extract quantiles of errors with Monte Carlo credible intervals for plotting

        :param errors: ``numpy`` 1D array; vector of errors
        :param level: ``float``; level of the credible interval.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw. If ``None``, use model defaults.
        :return: ``tuple`` of 4 ``numpy`` arrays; mean, lower bound, and upper bound at the desired level for each error value, plus the full array of samples.
        """

        assert False, 'Incomplete method. Do not use.'

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.errors: errors
                }

                alpha = 100-float(level)

                samples = [self.sess.run(self.err_dist_cdf, feed_dict=fd) for _ in range(n_samples)]
                samples = np.concatenate(samples, axis=1)

                mean = samples.mean(axis=1)
                lower = np.percentile(samples, alpha/2, axis=1)
                upper = np.percentile(samples, 100-(alpha/2), axis=1)

                return (mean, lower, upper, samples)

    def report_settings(self, indent=0):
        out = super(CDRBayes, self).report_settings(indent=indent)
        for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
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

    def run_predict_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        if algorithm in ['map', 'MAP'] and self.variational():
            MAP = True
        else:
            MAP = False

        if standardize_response and self.standardize_response:
            out_post = self.out_standardized_post
            out_MAP = self.out_standardized_MAP
        else:
            out_post = self.out_post
            out_MAP = self.out_MAP

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if MAP:
                    preds = self.sess.run(out_MAP, feed_dict=feed_dict)
                else:
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    preds = np.zeros((len(feed_dict[self.time_y]), n_samples))

                    for i in range(n_samples):
                        preds[:, i] = self.sess.run(out_post, feed_dict=feed_dict)
                        if verbose:
                            pb.update(i + 1, force=True)

                    preds = preds.mean(axis=1)

                return preds

    def run_loglik_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        if algorithm in ['map', 'MAP'] and self.variational():
            MAP = True
        else:
            MAP = False

        if standardize_response and self.standardize_response:
            ll_post = self.ll_standardized_post
            ll_MAP = self.ll_standardized_MAP
        else:
            ll_post = self.ll_post
            ll_MAP = self.ll_MAP

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if MAP:
                    log_lik = self.sess.run(ll_MAP, feed_dict=feed_dict)
                else:
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    log_lik = np.zeros((len(feed_dict[self.time_y]), n_samples))

                    for i in range(n_samples):
                        log_lik[:, i] = self.sess.run(ll_post, feed_dict=feed_dict)
                        if verbose:
                            pb.update(i + 1, force=True)

                    log_lik = log_lik.mean(axis=1)

                return log_lik

    def run_loss_op(self, feed_dict, n_samples=None, algorithm='MAP', verbose=True):
        if algorithm in ['map', 'MAP'] and self.variational():
            MAP = True
        else:
            MAP = False

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if MAP:
                    loss = self.sess.run(self.inference.loss, feed_dict=feed_dict)
                else:
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    loss = np.zeros((len(feed_dict[self.time_y]), n_samples))

                    for i in range(n_samples):
                        loss[:, i] = self.sess.run(self.inference.loss, feed_dict=feed_dict)
                        if verbose:
                            pb.update(i + 1, force=True)

                    loss = loss.mean()

                return loss

    def run_conv_op(self, feed_dict, scaled=False, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        if algorithm in ['map', 'MAP'] and self.variational():
            MAP = True
        else:
            MAP = False

        if scaled:
            if standardize_response and self.standardize_response:
                X_conv_post = self.X_conv_standardized_scaled_post
                X_conv_MAP = self.X_conv_standardized_scaled_MAP
            else:
                X_conv_post = self.X_conv_scaled_post
                X_conv_MAP = self.X_conv_scaled_MAP
        else:
            X_conv_post = self.X_conv_post
            X_conv_MAP = self.X_conv_MAP

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if MAP:
                    X_conv = self.sess.run(X_conv_MAP, feed_dict=feed_dict)
                else:
                    if n_samples is None:
                        n_samples = self.n_samples_eval
                    if verbose:
                        pb = tf.contrib.keras.utils.Progbar(n_samples)

                    X_conv = np.zeros((len(feed_dict[self.X]), self.X_conv.shape[-1], n_samples))

                    for i in range(0, n_samples):
                        X_conv[..., i] = self.sess.run(X_conv_post, feed_dict=feed_dict)
                        if verbose:
                            pb.update(i + 1, force=True)

                    X_conv = X_conv.mean(axis=2)

                return X_conv

    def finalize(self):
        super(CDRBayes, self).finalize()
        self.inference.finalize()



