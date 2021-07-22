import pandas as pd
pd.options.mode.chained_assignment = None

import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalTriL, Normal, SinhArcsinh

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

from .util import *
from .base import ModelBayes
from .cdrbase import CDR
from .kwargs import CDRBAYES_INITIALIZATION_KWARGS




######################################################
#
#  BAYESIAN IMPLEMENTATION OF CDR
#
######################################################

class CDRBayes(ModelBayes, CDR):

    _INITIALIZATION_KWARGS = CDRBAYES_INITIALIZATION_KWARGS

    _doc_header = """
        A CDR implementation fitted using black box variational Bayes.
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

    def __init__(self, form_str, X, Y, **kwargs):
        super(CDRBayes, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDRBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self._initialize_metadata()

        self.build()

    def _initialize_metadata(self):
        super(CDRBayes, self)._initialize_metadata()
        
        self._coef_prior_sd, \
        self._coef_posterior_sd_init, \
        self._coef_ranef_prior_sd, \
        self._coef_ranef_posterior_sd_init = self._process_prior_sd(self.coef_prior_sd)

        assert isinstance(self.irf_param_prior_sd, str) or isinstance(self.irf_param_prior_sd, float), 'irf_param_prior_sd must either be a string or a float'

        self._irf_param_prior_sd, \
        self._irf_param_posterior_sd_init, \
        self._irf_param_ranef_prior_sd, \
        self._irf_param_ranef_posterior_sd_init = self._process_prior_sd(self.irf_param_prior_sd)

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

    def initialize_coefficient(self, response, coef_ids=None, ran_gf=None):
        if coef_ids is None:
            coef_ids = self.coef_names

        if self.use_distributional_regression:
            nparam = self.get_response_nparam(response)
        else:
            nparam = 1

        ndim = self.get_response_ndim(response)
        ncoef = len(coef_ids)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    prior_sd = self._coef_prior_sd[response]
                    post_sd = self._coef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        prior_sd = prior_sd[:1]
                        post_sd = post_sd[:1]
                    prior_sd = np.ones((ncoef, 1, 1)) * prior_sd[None, ...]
                    post_sd = np.ones((ncoef, 1, 1)) * post_sd[None, ...]

                    # Posterior distribution
                    coefficient_q_loc = tf.Variable(
                        tf.zeros([ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s_q_loc' % sn(response)
                    )

                    coefficient_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='coefficient_%s_q_scale' % sn(response)
                    )

                    coefficient_q_dist = Normal(
                        loc=coefficient_q_loc,
                        scale=self.constraint_fn(coefficient_q_scale) + self.epsilon,
                        name='coefficient_%s_q' % sn(response)
                    )

                    coefficient = tf.cond(self.use_MAP_mode, coefficient_q_dist.mean, coefficient_q_dist.sample)

                    coefficient_summary = coefficient_q_dist.mean()

                    if self.declare_priors_fixef:
                        # Prior distribution
                        coefficient_prior_dist = Normal(
                            loc=0.,
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='coefficient_%s' % sn(response)
                        )
                        self.kl_penalties['coefficient_%s' % sn(response)] = {
                            'loc': 0.,
                            'scale': self.constraint_fn_np(self._coef_prior_sd[response]).flatten(),
                            'val': coefficient_q_dist.kl_divergence(coefficient_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    prior_sd = self._coef_ranef_prior_sd[response]
                    post_sd = self._coef_ranef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        prior_sd = prior_sd[:1]
                        post_sd = post_sd[:1]
                    prior_sd = np.ones((rangf_n_levels, ncoef, 1, 1)) * prior_sd[None, None, ...]
                    post_sd = np.ones((rangf_n_levels, ncoef, 1, 1)) * post_sd[None, None, ...]

                    # Posterior distribution
                    coefficient_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s_by_%s_q_loc' % (sn(response), sn(ran_gf))
                    )

                    coefficient_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='coefficient_%s_by_%s_q_scale' % (sn(response), sn(ran_gf))
                    )

                    coefficient_q_dist = Normal(
                        loc=coefficient_q_loc,
                        scale=self.constraint_fn(coefficient_q_scale) + self.epsilon,
                        name='coefficient_%s_by_%s_q' % (sn(response), sn(ran_gf))
                    )

                    coefficient = tf.cond(self.use_MAP_mode, coefficient_q_dist.mean, coefficient_q_dist.sample)

                    coefficient_summary = coefficient_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        coefficient_prior_dist = Normal(
                            loc=0.,
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='coefficient_%s_by_%s' % (sn(response), sn(ran_gf))
                        )
                        self.kl_penalties['coefficient_%s_by_%s' % (sn(response), sn(ran_gf))] = {
                            'loc': 0.,
                            'scale': self.constraint_fn_np(self._coef_ranef_prior_sd[response]).flatten(),
                            'val': coefficient_q_dist.kl_divergence(coefficient_prior_dist)
                        }

                # shape: (?rangf_n_levels, ncoef, nparam, ndim)

                return coefficient, coefficient_summary

    def initialize_interaction(self, response, interaction_ids=None, ran_gf=None):
        if interaction_ids is None:
            interaction_ids = self.interaction_names

        if self.use_distributional_regression:
            nparam = self.get_response_nparam(response)
        else:
            nparam = 1
        ndim = self.get_response_ndim(response)
        ninter = len(interaction_ids)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    prior_sd = self._coef_prior_sd[response]
                    post_sd = self._coef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        prior_sd = prior_sd[:1]
                        post_sd = post_sd[:1]
                    prior_sd = np.ones((ninter, 1, 1)) * prior_sd[None, ...]
                    post_sd = np.ones((ninter, 1, 1)) * post_sd[None, ...]

                    # Posterior distribution
                    interaction_q_loc = tf.Variable(
                        tf.zeros([ninter, nparam, ndim], dtype=self.FLOAT_TF),
                        name='interaction_%s_q_loc' % sn(response)
                    )

                    interaction_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='interaction_%s_q_scale' % sn(response)
                    )

                    interaction_q_dist = Normal(
                        loc=interaction_q_loc,
                        scale=self.constraint_fn(interaction_q_scale) + self.epsilon,
                        name='interaction_%s_q' % (response)
                    )

                    interaction = tf.cond(self.use_MAP_mode, interaction_q_dist.mean, interaction_q_dist.sample)

                    interaction_summary = interaction_q_dist.mean()

                    if self.declare_priors_fixef:
                        # Prior distribution
                        interaction_prior_dist = Normal(
                            loc=0.,
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='interaction_%s' % sn(response)
                        )
                        self.kl_penalties['interaction_%s' % sn(response)] = {
                            'loc': 0.,
                            'scale': self.constraint_fn_np(self._coef_prior_sd[response]).flatten(),
                            'val': interaction_q_dist.kl_divergence(interaction_prior_dist)
                        }
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    prior_sd = self._coef_ranef_prior_sd[response]
                    post_sd = self._coef_ranef_posterior_sd_init[response]
                    if not self.use_distributional_regression:
                        prior_sd = prior_sd[:1]
                        post_sd = post_sd[:1]
                    prior_sd = np.ones((rangf_n_levels, ninter, 1, 1)) * prior_sd[None, None, ...]
                    post_sd = np.ones((rangf_n_levels, ninter, 1, 1)) * post_sd[None, None, ...]

                    # Posterior distribution
                    interaction_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, ninter], dtype=self.FLOAT_TF),
                        name='interaction_%s_by_%s_q_loc' % (sn(response), sn(ran_gf))
                    )

                    interaction_q_scale = tf.Variable(
                        tf.constant(post_sd, self.FLOAT_TF),
                        name='interaction_%s_by_%s_q_scale' % (sn(response), sn(ran_gf))
                    )

                    interaction_q_dist = Normal(
                        loc=interaction_q_loc,
                        scale=self.constraint_fn(interaction_q_scale) + self.epsilon,
                        name='interaction_%s_by_%s_q' % (sn(response), sn(ran_gf))
                    )

                    interaction = tf.cond(self.use_MAP_mode, interaction_q_dist.mean, interaction_q_dist.sample)

                    interaction_summary = interaction_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        interaction_prior_dist = Normal(
                            loc=0.,
                            scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                            name='interaction_%s_by_%s' % (sn(response), sn(ran_gf))
                        )
                        self.kl_penalties['interaction_%s_by_%s' % (sn(response), sn(ran_gf))] = {
                            'loc': 0.,
                            'scale': self.constraint_fn_np(self._coef_ranef_prior_sd[response]).flatten(),
                            'val': interaction_q_dist.kl_divergence(interaction_prior_dist)
                        }

                # shape: (?rangf_n_levels, ninter, nparam, ndim)

                return interaction, interaction_summary

    def initialize_irf_param(self, response, family, param_name, ran_gf=None):
        param_mean_unconstrained = self.irf_params_means_unconstrained[family][param_name]
        trainable_ix = self.irf_params_trainable_ix[family][param_name]
        mean = param_mean_unconstrained[trainable_ix]
        irf_ids_all = self.atomic_irf_names_by_family[family]
        param_trainable = self.atomic_irf_param_trainable_by_family[family]

        if self.use_distributional_regression:
            response_nparam = self.get_response_nparam(response) # number of params of predictive dist, not IRF
        else:
            response_nparam = 1
        response_ndim = self.get_response_ndim(response)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if ran_gf is None:
                    trainable_ids = [x for x in irf_ids_all if param_name in param_trainable[x]]
                    nirf = len(trainable_ids)

                    if nirf:
                        prior_sd = self._irf_param_prior_sd[response]
                        post_sd = self._irf_param_posterior_sd_init[response]
                        if not self.use_distributional_regression:
                            prior_sd = prior_sd[:1]
                            post_sd = post_sd[:1]
                        prior_sd = np.ones((nirf, 1, 1)) * prior_sd[None, ...]
                        post_sd = np.ones((nirf, 1, 1)) * post_sd[None, ...]

                        # Posterior distribution
                        param_q_loc = tf.Variable(
                            tf.ones([nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF) * tf.constant(mean[..., None, None], dtype=self.FLOAT_TF),
                            name='%s_%s_%s_q_loc' % (param_name, sn(response), sn('-'.join(trainable_ids)))
                        )

                        param_q_scale = tf.Variable(
                            tf.constant(post_sd, self.FLOAT_TF),
                            name='%s_%s_%s_q_scale' % (param_name, sn(response), sn('-'.join(trainable_ids)))
                        )

                        param_q_dist = Normal(
                            loc=param_q_loc,
                            scale=self.constraint_fn(param_q_scale) + self.epsilon,
                            name='%s_%s_%s_q' % (param_name, sn(response), sn('-'.join(trainable_ids)))
                        )

                        param = tf.cond(self.use_MAP_mode, param_q_dist.mean, param_q_dist.sample)

                        param_summary = param_q_dist.mean()

                        if self.declare_priors_fixef:
                            # Prior distribution
                            param_prior_dist = Normal(
                                loc=tf.constant(mean[..., None, None], dtype=self.FLOAT_TF),
                                scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                                name='%s_%s_%s' % (param_name, sn(response), sn('-'.join(trainable_ids)))
                            )
                            self.kl_penalties['%s_%s_%s' % (param_name, sn(response), sn('-'.join(trainable_ids)))] = {
                                'loc': mean.flatten(),
                                'scale': self.constraint_fn_np(self._irf_param_prior_sd[response]).flatten(),
                                'val': param_q_dist.kl_divergence(param_prior_dist)
                            }
                    else:
                        param = param_summary = None
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_ids_gf = self.irf_by_rangf[ran_gf]
                    trainable_ids = [x for x in irf_ids_all if (param_name in param_trainable[x] and x in irf_ids_gf)]
                    nirf = len(trainable_ids)

                    if nirf:
                        prior_sd = self._irf_param_ranef_prior_sd[response]
                        post_sd = self._irf_param_ranef_posterior_sd_init[response]
                        if not self.use_distributional_regression:
                            prior_sd = prior_sd[:1]
                            post_sd = post_sd[:1]
                        prior_sd = np.ones((rangf_n_levels, nirf, 1, 1)) * prior_sd[None, None, ...]
                        post_sd = np.ones((rangf_n_levels, nirf, 1, 1)) * post_sd[None, None, ...]

                        # Posterior distribution
                        param_q_loc = tf.Variable(
                            tf.zeros([rangf_n_levels, nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF),
                            name=sn('%s_%s_%s_by_%s_q_loc' % (param_name, sn(response), '-'.join(trainable_ids), sn(ran_gf)))
                        )

                        param_q_scale = tf.Variable(
                            tf.constant(post_sd, self.FLOAT_TF),
                            name=sn('%s_%s_%s_by_%s_q_scale' % (param_name, sn(response), '-'.join(trainable_ids), sn(ran_gf)))
                        )

                        param_q_dist = Normal(
                            loc=param_q_loc,
                            scale=self.constraint_fn(param_q_scale) + self.epsilon,
                            name=sn('%s_%s_%s_by_%s_q' % (param_name, sn(response), '-'.join(trainable_ids), sn(ran_gf)))
                        )

                        param = tf.cond(self.use_MAP_mode, param_q_dist.mean, param_q_dist.sample)

                        param_summary = param_q_dist.mean()

                        if self.declare_priors_ranef:
                            # Prior distribution
                            param_prior_dist = Normal(
                                loc=0.,
                                scale=self.constraint_fn(tf.constant(prior_sd, self.FLOAT_TF)),
                                name='%s_%s_%s_by_%s' % (param_name, sn(response), sn('-'.join(trainable_ids)), sn(ran_gf))
                            )
                            self.kl_penalties['%s_%s_%s_by_%s' % (param_name, sn(response), sn('-'.join(trainable_ids)), sn(ran_gf))] = {
                                'loc': 0.,
                                'scale': self.constraint_fn_np(self._irf_param_ranef_prior_sd[response]).flatten(),
                                'val': param_q_dist.kl_divergence(param_prior_dist)
                            }
                    else:
                        param = param_summary = None

                # shape: (?rangf_n_levels, nirf, nparam, ndim)

                return param, param_summary




    #####################################################
    #
    #  Public methods
    #
    ######################################################

    def report_settings(self, indent=0):
        out = super(CDRBayes, self).report_settings(indent=indent)
        for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out
