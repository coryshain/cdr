import pandas as pd

from .kwargs import CDRNNBAYES_INITIALIZATION_KWARGS
from .backend import DenseLayerBayes, RNNLayerBayes, BatchNormLayerBayes, LayerNormLayerBayes
from .base import ModelBayes
from .cdrnnbase import CDRNN
from .util import get_numerical_sd, sn, reg_name, stderr

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, SinhArcsinh

pd.options.mode.chained_assignment = None


class CDRNNBayes(ModelBayes, CDRNN):
    _INITIALIZATION_KWARGS = CDRNNBAYES_INITIALIZATION_KWARGS

    _doc_header = """
        A CDRRNN implementation fitted using black box variational Bayes.
    """
    _doc_args = CDRNN._doc_args
    _doc_kwargs = CDRNN._doc_kwargs
    _doc_kwargs += '\n' + '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    def __init__(self, form_str, X, Y, **kwargs):
        super(CDRNNBayes, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDRNNBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self._initialize_metadata()

        self.build()

    def _pack_metadata(self):
        md = super(CDRNNBayes, self)._pack_metadata()
        for kwarg in CDRNNBayes._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)

        return md

    def _unpack_metadata(self, md):
        super(CDRNNBayes, self)._unpack_metadata(md)

        for kwarg in CDRNNBayes._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

        if len(md) > 0:
            stderr('Saved model contained unrecognized attributes %s which are being ignored\n' %sorted(list(md.keys())))





    ######################################################
    #
    #  Network initialization
    #
    ######################################################

    def initialize_intercept(self, response_name, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                init = tf.constant(self.intercept_init[response_name], dtype=self.FLOAT_TF)
                name = sn(response_name)
                if ran_gf is None:
                    # Posterior distribution
                    intercept_q_loc = tf.Variable(
                        init,
                        name='intercept_%s_q_loc' % name
                    )

                    intercept_q_scale = tf.Variable(
                        self.constraint_fn_inv(self._intercept_posterior_sd_init),
                        name='intercept_%s_q_scale' % name
                    )

                    intercept_q_dist = Normal(
                        loc=intercept_q_loc,
                        scale=self.constraint_fn(intercept_q_scale) + self.epsilon,
                        name='intercept_%s_q' % name
                    )

                    intercept = tf.cond(self.use_MAP_mode, intercept_q_dist.mean, intercept_q_dist.sample)

                    intercept_summary = intercept_q_dist.mean()

                    if self.declare_priors_fixef:
                        # Prior distribution
                        intercept_prior_dist = Normal(
                            loc=init,
                            scale=self.intercept_prior_sd_tf,
                            name='intercept_%s' % name
                        )
                        self.kl_penalties_base['intercept_%s' % name] = {
                            'loc': init,
                            'scale': self._intercept_prior_sd,
                            'val': intercept_q_dist.kl_divergence(intercept_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    shape = [rangf_n_levels] + [int(x) for x in init.shape]

                    # Posterior distribution
                    intercept_q_loc = tf.Variable(
                        tf.zeros(shape, dtype=self.FLOAT_TF),
                        name='intercept_%s_by_%s_q_loc' % (name, sn(ran_gf))
                    )

                    intercept_q_scale = tf.Variable(
                        tf.ones(shape, dtype=self.FLOAT_TF) * self.constraint_fn_inv(self._intercept_ranef_posterior_sd_init * self.ranef_to_fixef_prior_sd_ratio),
                        name='intercept_%s_by_%s_q_scale' % (name, sn(ran_gf))
                    )

                    intercept_q_dist = Normal(
                        loc=intercept_q_loc,
                        scale=self.constraint_fn(intercept_q_scale) + self.epsilon,
                        name='intercept_%s_by_%s_q' % (name, sn(ran_gf))
                    )

                    intercept = tf.cond(self.use_MAP_mode, intercept_q_dist.mean, intercept_q_dist.sample)

                    intercept_summary = intercept_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        intercept_prior_dist = Normal(
                            loc=0.,
                            scale=self.intercept_prior_sd_tf * self.ranef_to_fixef_prior_sd_ratio,
                            name='intercept_%s_by_%s' % (name, sn(ran_gf))
                        )
                        self.kl_penalties_base['intercept_%s_by_%s' % (name, sn(ran_gf))] = {
                            'loc': 0.,
                            'scale': self._intercept_prior_sd * self.ranef_to_fixef_prior_sd_ratio,
                            'val': intercept_q_dist.kl_divergence(intercept_prior_dist)
                        }

                return intercept, intercept_summary

    def initialize_feedforward(
            self,
            units,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            name=None,
            final=False
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if final:
                    weight_sd_prior = 1.
                    weight_sd_init = self.weight_sd_init
                    bias_sd_prior = 1.
                    bias_sd_init = self.bias_sd_init
                    gamma_sd_prior = 1.
                    gamma_sd_init = self.gamma_sd_init
                    declare_priors_weights = self.declare_priors_fixef
                else:
                    weight_sd_prior = self.weight_prior_sd
                    weight_sd_init = self.weight_sd_init
                    bias_sd_prior = self.bias_prior_sd
                    bias_sd_init = self.bias_sd_init
                    gamma_sd_prior = self.gamma_prior_sd
                    gamma_sd_init = self.gamma_sd_init
                    declare_priors_weights = self.declare_priors_weights

                projection = DenseLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    use_bias=use_bias,
                    activation=activation,
                    dropout=dropout,
                    maxnorm=maxnorm,
                    batch_normalization_decay=batch_normalization_decay,
                    layer_normalization_type=layer_normalization_type,
                    normalize_after_activation=self.normalize_after_activation,
                    normalization_use_gamma=self.normalization_use_gamma,
                    declare_priors_weights=declare_priors_weights,
                    declare_priors_biases=self.declare_priors_biases,
                    kernel_sd_prior=weight_sd_prior,
                    kernel_sd_init=weight_sd_init,
                    bias_sd_prior=bias_sd_prior,
                    bias_sd_init=bias_sd_init,
                    gamma_sd_prior=gamma_sd_prior,
                    gamma_sd_init=gamma_sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    constraint=self.constraint,
                    epsilon=self.epsilon,
                    session=self.sess,
                    name=name
                )

                return projection

    def initialize_rnn(self, l):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                rnn = RNNLayerBayes(
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    units=units,
                    time_projection_depth=self.n_layers_irf + 1,
                    activation=self.rnn_activation,
                    recurrent_activation=self.recurrent_activation,
                    time_projection_inner_activation=self.irf_inner_activation,
                    bottomup_dropout=self.input_projection_dropout_rate,
                    h_dropout=self.rnn_h_dropout_rate,
                    c_dropout=self.rnn_c_dropout_rate,
                    forget_rate=self.forget_rate,
                    return_sequences=True,
                    declare_priors_weights=self.declare_priors_weights,
                    declare_priors_biases=self.declare_priors_biases,
                    kernel_sd_prior=self.weight_prior_sd,
                    bottomup_kernel_sd_init=self.weight_sd_init,
                    recurrent_kernel_sd_init=self.weight_sd_init,
                    bias_sd_prior=self.bias_prior_sd,
                    bias_sd_init=self.bias_sd_init,
                    posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                    constraint=self.constraint,
                    name='rnn_l%d' % (l + 1),
                    epsilon=self.epsilon,
                    session=self.sess
                )

                return rnn

    def initialize_rnn_h(self, l, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                
                rnn_h_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                rnn_h_sd_posterior = rnn_h_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    rnn_h_q_loc = tf.Variable(
                        tf.zeros([1, units]),
                        name='rnn_h_q_loc'
                    )

                    rnn_h_q_scale = tf.Variable(
                        tf.ones([1, units]) * self.constraint_fn_inv(rnn_h_sd_posterior),
                        name='rnn_h_q_scale'
                    )

                    rnn_h_q_dist = Normal(
                        loc=rnn_h_q_loc,
                        scale=self.constraint_fn(rnn_h_q_scale) + self.epsilon,
                        name='rnn_h_q'
                    )

                    rnn_h = tf.cond(self.use_MAP_mode, rnn_h_q_dist.mean, rnn_h_q_dist.sample)

                    rnn_h_summary = rnn_h_q_dist.mean()

                    if self.declare_priors_biases:
                        # Prior distribution
                        rnn_h_prior_dist = Normal(
                            loc=0.,
                            scale=rnn_h_sd_prior,
                            name='rnn_h'
                        )
                        self.kl_penalties_base['rnn_h'] = {
                            'loc': 0.,
                            'scale': rnn_h_sd_prior,
                            'val': rnn_h_q_dist.kl_divergence(rnn_h_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    rnn_h_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='rnn_h_q_loc_by_%s' % sn(ran_gf)
                    )

                    rnn_h_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(rnn_h_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='rnn_h_q_scale_by_%s' % sn(ran_gf)
                    )

                    rnn_h_q_dist = Normal(
                        loc=rnn_h_q_loc,
                        scale=self.constraint_fn(rnn_h_q_scale) + self.epsilon,
                        name='rnn_h_q_by_%s' % sn(ran_gf)
                    )

                    rnn_h = tf.cond(self.use_MAP_mode, rnn_h_q_dist.mean, rnn_h_q_dist.sample)

                    rnn_h_summary = rnn_h_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        rnn_h_prior_dist = Normal(
                            loc=0.,
                            scale=rnn_h_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='rnn_h_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['rnn_h_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': rnn_h_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': rnn_h_q_dist.kl_divergence(rnn_h_prior_dist)
                        }

                return rnn_h, rnn_h_summary

    def initialize_rnn_c(self, l, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_rnn[l]
                
                rnn_c_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                rnn_c_sd_posterior = rnn_c_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    rnn_c_q_loc = tf.Variable(
                        tf.zeros([1, units]),
                        name='rnn_c_q_loc'
                    )

                    rnn_c_q_scale = tf.Variable(
                        tf.ones([1, units]) * self.constraint_fn_inv(rnn_c_sd_posterior),
                        name='rnn_c_q_scale'
                    )

                    rnn_c_q_dist = Normal(
                        loc=rnn_c_q_loc,
                        scale=self.constraint_fn(rnn_c_q_scale) + self.epsilon,
                        name='rnn_c_q'
                    )

                    rnn_c = tf.cond(self.use_MAP_mode, rnn_c_q_dist.mean, rnn_c_q_dist.sample)

                    rnn_c_summary = rnn_c_q_dist.mean()

                    if self.declare_priors_biases:
                        # Prior distribution
                        rnn_c_prior_dist = Normal(
                            loc=0.,
                            scale=rnn_c_sd_prior,
                            name='rnn_c'
                        )
                        self.kl_penalties_base['rnn_c'] = {
                            'loc': 0.,
                            'scale': rnn_c_sd_prior,
                            'val': rnn_c_q_dist.kl_divergence(rnn_c_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    rnn_c_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='rnn_c_q_loc_by_%s' % sn(ran_gf)
                    )

                    rnn_c_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(rnn_c_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='rnn_c_q_scale_by_%s' % sn(ran_gf)
                    )

                    rnn_c_q_dist = Normal(
                        loc=rnn_c_q_loc,
                        scale=self.constraint_fn(rnn_c_q_scale) + self.epsilon,
                        name='rnn_c_q_by_%s' % sn(ran_gf)
                    )

                    rnn_c = tf.cond(self.use_MAP_mode, rnn_c_q_dist.mean, rnn_c_q_dist.sample)

                    rnn_c_summary = rnn_c_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        rnn_c_prior_dist = Normal(
                            loc=0.,
                            scale=rnn_c_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='rnn_c_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['rnn_c_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': rnn_c_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': rnn_c_q_dist.kl_divergence(rnn_c_prior_dist)
                        }

                return rnn_c, rnn_c_summary

    def initialize_h_bias(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_hidden_state

                h_bias_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                h_bias_sd_posterior = h_bias_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    h_bias_q_loc = tf.Variable(
                        tf.zeros([1, 1, units]),
                        name='h_bias_q_loc'
                    )

                    h_bias_q_scale = tf.Variable(
                        tf.ones([1, 1, units]) * self.constraint_fn_inv(h_bias_sd_posterior),
                        name='h_bias_q_scale'
                    )

                    h_bias_q_dist = Normal(
                        loc=h_bias_q_loc,
                        scale=self.constraint_fn(h_bias_q_scale) + self.epsilon,
                        name='h_bias_q'
                    )

                    h_bias = tf.cond(self.use_MAP_mode, h_bias_q_dist.mean, h_bias_q_dist.sample)

                    h_bias_summary = h_bias_q_dist.mean()

                    if self.declare_priors_biases:
                        # Prior distribution
                        h_bias_prior_dist = Normal(
                            loc=0.,
                            scale=h_bias_sd_prior,
                            name='h_bias'
                        )
                        self.kl_penalties_base['h_bias'] = {
                            'loc': 0.,
                            'scale': h_bias_sd_prior,
                            'val': h_bias_q_dist.kl_divergence(h_bias_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    h_bias_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='h_bias_q_loc_by_%s' % sn(ran_gf)
                    )

                    h_bias_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(h_bias_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='h_bias_q_scale_by_%s' % sn(ran_gf)
                    )

                    h_bias_q_dist = Normal(
                        loc=h_bias_q_loc,
                        scale=self.constraint_fn(h_bias_q_scale) + self.epsilon,
                        name='h_bias_q_by_%s' % sn(ran_gf)
                    )

                    h_bias = tf.cond(self.use_MAP_mode, h_bias_q_dist.mean, h_bias_q_dist.sample)

                    h_bias_summary = h_bias_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        h_bias_prior_dist = Normal(
                            loc=0.,
                            scale=h_bias_sd_prior * h_bias_sd_prior,
                            name='h_bias_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['h_bias_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': h_bias_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': h_bias_q_dist.kl_divergence(h_bias_prior_dist)
                        }

                return h_bias, h_bias_summary

    def initialize_h_normalization(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.use_batch_normalization:
                    normalization_layer = BatchNormLayerBayes(
                        decay=self.batch_normalization_decay,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        training=self.training,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='h'
                    )
                elif self.use_layer_normalization:
                    normalization_layer = LayerNormLayerBayes(
                        normalization_type=self.layer_normalization_type,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='h'
                    )
                else:
                    normalization_layer = lambda x: x

                return normalization_layer

    def initialize_intercept_l1_weights(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                
                intercept_l1_W_sd_prior = get_numerical_sd(self.weight_prior_sd, in_dim=1, out_dim=units)
                intercept_l1_W_sd_posterior = intercept_l1_W_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    intercept_l1_W_q_loc = tf.Variable(
                        tf.zeros([1, units]),
                        name='intercept_l1_W_q_loc'
                    )

                    intercept_l1_W_q_scale = tf.Variable(
                        tf.zeros([1, units]) * self.constraint_fn_inv(intercept_l1_W_sd_posterior),
                        name='intercept_l1_W_q_scale'
                    )

                    intercept_l1_W_q_dist = Normal(
                        loc=intercept_l1_W_q_loc,
                        scale=self.constraint_fn(intercept_l1_W_q_scale) + self.epsilon,
                        name='intercept_l1_W_q'
                    )

                    intercept_l1_W = tf.cond(self.use_MAP_mode, intercept_l1_W_q_dist.mean, intercept_l1_W_q_dist.sample)

                    intercept_l1_W_summary = intercept_l1_W_q_dist.mean()

                    if self.declare_priors_weights:
                        # Prior distribution
                        intercept_l1_W_prior_dist = Normal(
                            loc=0.,
                            scale=intercept_l1_W_sd_prior,
                            name='intercept_l1_W'
                        )
                        self.kl_penalties_base['intercept_l1_W'] = {
                            'loc': 0.,
                            'scale': intercept_l1_W_sd_prior,
                            'val': intercept_l1_W_q_dist.kl_divergence(intercept_l1_W_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    intercept_l1_W_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='intercept_l1_W_q_loc_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_W_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(intercept_l1_W_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='intercept_l1_W_q_scale_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_W_q_dist = Normal(
                        loc=intercept_l1_W_q_loc,
                        scale=self.constraint_fn(intercept_l1_W_q_scale) + self.epsilon,
                        name='intercept_l1_W_q_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_W = tf.cond(self.use_MAP_mode, intercept_l1_W_q_dist.mean, intercept_l1_W_q_dist.sample)

                    intercept_l1_W_summary = intercept_l1_W_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        intercept_l1_W_prior_dist = Normal(
                            loc=0.,
                            scale=intercept_l1_W_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='intercept_l1_W_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['intercept_l1_W_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': intercept_l1_W_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': intercept_l1_W_q_dist.kl_divergence(intercept_l1_W_prior_dist)
                        }

                return intercept_l1_W, intercept_l1_W_summary
            
    def initialize_intercept_l1_biases(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                intercept_l1_b_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                intercept_l1_b_sd_posterior = intercept_l1_b_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    intercept_l1_b_q_loc = tf.Variable(
                        tf.zeros([1, units]),
                        name='intercept_l1_b_q_loc'
                    )

                    intercept_l1_b_q_scale = tf.Variable(
                        tf.zeros([1, units]) * self.constraint_fn_inv(intercept_l1_b_sd_posterior),
                        name='intercept_l1_b_q_scale'
                    )

                    intercept_l1_b_q_dist = Normal(
                        loc=intercept_l1_b_q_loc,
                        scale=self.constraint_fn(intercept_l1_b_q_scale) + self.epsilon,
                        name='intercept_l1_b_q'
                    )

                    intercept_l1_b = tf.cond(self.use_MAP_mode, intercept_l1_b_q_dist.mean, intercept_l1_b_q_dist.sample)

                    intercept_l1_b_summary = intercept_l1_b_q_dist.mean()

                    if self.declare_priors_biases:
                        # Prior distribution
                        intercept_l1_b_prior_dist = Normal(
                            loc=0.,
                            scale=intercept_l1_b_sd_prior,
                            name='intercept_l1_b'
                        )
                        self.kl_penalties_base['intercept_l1_b'] = {
                            'loc': 0.,
                            'scale': intercept_l1_b_sd_prior,
                            'val': intercept_l1_b_q_dist.kl_divergence(intercept_l1_b_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    intercept_l1_b_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='intercept_l1_b_q_loc_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_b_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(intercept_l1_b_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='intercept_l1_b_q_scale_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_b_q_dist = Normal(
                        loc=intercept_l1_b_q_loc,
                        scale=self.constraint_fn(intercept_l1_b_q_scale) + self.epsilon,
                        name='intercept_l1_b_q_by_%s' % sn(ran_gf)
                    )

                    intercept_l1_b = tf.cond(self.use_MAP_mode, intercept_l1_b_q_dist.mean, intercept_l1_b_q_dist.sample)

                    intercept_l1_b_summary = intercept_l1_b_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        intercept_l1_b_prior_dist = Normal(
                            loc=0.,
                            scale=intercept_l1_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='intercept_l1_b_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['intercept_l1_b_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': intercept_l1_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': intercept_l1_b_q_dist.kl_divergence(intercept_l1_b_prior_dist)
                        }

                return intercept_l1_b, intercept_l1_b_summary

    def initialize_intercept_l1_normalization(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.use_batch_normalization:
                    normalization_layer = BatchNormLayerBayes(
                        decay=self.batch_normalization_decay,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        training=self.training,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='intercept_l1'
                    )
                elif self.use_layer_normalization:
                    normalization_layer = LayerNormLayerBayes(
                        normalization_type=self.layer_normalization_type,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='intercept_l1'
                    )
                else:
                    normalization_layer = lambda x: x

                return normalization_layer

    def initialize_irf_l1_weights(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                
                irf_l1_W_sd_prior = get_numerical_sd(self.weight_prior_sd, in_dim=1, out_dim=units)
                irf_l1_W_sd_posterior = irf_l1_W_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    irf_l1_W_q_loc = tf.Variable(
                        tf.zeros([1, 1, units]),
                        name='irf_l1_W_q_loc'
                    )

                    irf_l1_W_q_scale = tf.Variable(
                        tf.zeros([1, 1, units]) * self.constraint_fn_inv(irf_l1_W_sd_posterior),
                        name='irf_l1_W_q_scale'
                    )

                    irf_l1_W_q_dist = Normal(
                        loc=irf_l1_W_q_loc,
                        scale=self.constraint_fn(irf_l1_W_q_scale) + self.epsilon,
                        name='irf_l1_W_q'
                    )

                    irf_l1_W = tf.cond(self.use_MAP_mode, irf_l1_W_q_dist.mean, irf_l1_W_q_dist.sample)

                    irf_l1_W_summary = irf_l1_W_q_dist.mean()

                    if self.declare_priors_weights:
                        # Prior distribution
                        irf_l1_W_prior_dist = Normal(
                            loc=0.,
                            scale=irf_l1_W_sd_prior,
                            name='irf_l1_W'
                        )
                        self.kl_penalties_base['irf_l1_W'] = {
                            'loc': 0.,
                            'scale': irf_l1_W_sd_prior,
                            'val': irf_l1_W_q_dist.kl_divergence(irf_l1_W_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    irf_l1_W_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='irf_l1_W_q_loc_by_%s' % sn(ran_gf)
                    )

                    irf_l1_W_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(irf_l1_W_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='irf_l1_W_q_scale_by_%s' % sn(ran_gf)
                    )

                    irf_l1_W_q_dist = Normal(
                        loc=irf_l1_W_q_loc,
                        scale=self.constraint_fn(irf_l1_W_q_scale) + self.epsilon,
                        name='irf_l1_W_q_by_%s' % sn(ran_gf)
                    )

                    irf_l1_W = tf.cond(self.use_MAP_mode, irf_l1_W_q_dist.mean, irf_l1_W_q_dist.sample)

                    irf_l1_W_summary = irf_l1_W_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        irf_l1_W_prior_dist = Normal(
                            loc=0.,
                            scale=irf_l1_W_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='irf_l1_W_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['irf_l1_W_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': irf_l1_W_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': irf_l1_W_q_dist.kl_divergence(irf_l1_W_prior_dist)
                        }

                return irf_l1_W, irf_l1_W_summary
            
    def initialize_irf_l1_biases(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                units = self.n_units_irf_l1
                irf_l1_b_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                irf_l1_b_sd_posterior = irf_l1_b_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    # Posterior distribution
                    irf_l1_b_q_loc = tf.Variable(
                        tf.zeros([1, 1, units]),
                        name='irf_l1_b_q_loc'
                    )

                    irf_l1_b_q_scale = tf.Variable(
                        tf.zeros([1, 1, units]) * self.constraint_fn_inv(irf_l1_b_sd_posterior),
                        name='irf_l1_b_q_scale'
                    )

                    irf_l1_b_q_dist = Normal(
                        loc=irf_l1_b_q_loc,
                        scale=self.constraint_fn(irf_l1_b_q_scale) + self.epsilon,
                        name='irf_l1_b_q'
                    )

                    irf_l1_b = tf.cond(self.use_MAP_mode, irf_l1_b_q_dist.mean, irf_l1_b_q_dist.sample)

                    irf_l1_b_summary = irf_l1_b_q_dist.mean()

                    if self.declare_priors_biases:
                        # Prior distribution
                        irf_l1_b_prior_dist = Normal(
                            loc=0.,
                            scale=irf_l1_b_sd_prior,
                            name='irf_l1_b'
                        )
                        self.kl_penalties_base['irf_l1_b'] = {
                            'loc': 0.,
                            'scale': irf_l1_b_sd_prior,
                            'val': irf_l1_b_q_dist.kl_divergence(irf_l1_b_prior_dist)
                        }

                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    irf_l1_b_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='irf_l1_b_q_loc_by_%s' % sn(ran_gf)
                    )

                    irf_l1_b_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(irf_l1_b_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='irf_l1_b_q_scale_by_%s' % sn(ran_gf)
                    )

                    irf_l1_b_q_dist = Normal(
                        loc=irf_l1_b_q_loc,
                        scale=self.constraint_fn(irf_l1_b_q_scale) + self.epsilon,
                        name='irf_l1_b_q_by_%s' % sn(ran_gf)
                    )

                    irf_l1_b = tf.cond(self.use_MAP_mode, irf_l1_b_q_dist.mean, irf_l1_b_q_dist.sample)

                    irf_l1_b_summary = irf_l1_b_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        irf_l1_b_prior_dist = Normal(
                            loc=0.,
                            scale=irf_l1_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='irf_l1_b_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['irf_l1_b_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': irf_l1_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': irf_l1_b_q_dist.kl_divergence(irf_l1_b_prior_dist)
                        }

                return irf_l1_b, irf_l1_b_summary

    def initialize_irf_l1_normalization(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.use_batch_normalization:
                    normalization_layer = BatchNormLayerBayes(
                        decay=self.batch_normalization_decay,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        training=self.training,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='irf_l1'
                    )
                elif self.use_layer_normalization:
                    normalization_layer = LayerNormLayerBayes(
                        normalization_type=self.layer_normalization_type,
                        shift_activations=True,
                        rescale_activations=self.normalization_use_gamma,
                        axis=-1,
                        use_MAP_mode=self.use_MAP_mode,
                        declare_priors_scale=self.declare_priors_gamma,
                        declare_priors_shift=self.declare_priors_biases,
                        scale_sd_prior=self.bias_prior_sd,
                        scale_sd_init=self.bias_sd_init,
                        shift_sd_prior=self.bias_prior_sd,
                        shift_sd_init=self.bias_prior_sd,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        epsilon=self.epsilon,
                        session=self.sess,
                        name='irf_l1'
                    )
                else:
                    normalization_layer = lambda x: x

                return normalization_layer

    def initialize_error_params_biases(self, ran_gf=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.asymmetric_error:
                    units = 3
                else:
                    units = 1
                units = self.n_units_irf_l1
                error_params_b_sd_prior = get_numerical_sd(self.bias_prior_sd, in_dim=1, out_dim=1)
                error_params_b_sd_posterior = error_params_b_sd_prior * self.posterior_to_prior_sd_ratio

                if ran_gf is None:
                    error_params_b = tf.zeros([1, units])
                    error_params_b_summary = error_params_b
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1

                    # Posterior distribution
                    error_params_b_q_loc = tf.Variable(
                        tf.zeros([rangf_n_levels, units], dtype=self.FLOAT_TF),
                        name='error_params_q_loc_by_%s' % sn(ran_gf)
                    )

                    error_params_b_q_scale = tf.Variable(
                        tf.ones([rangf_n_levels, units], dtype=self.FLOAT_TF) * self.constraint_fn_inv(
                            error_params_b_sd_posterior * self.ranef_to_fixef_prior_sd_ratio),
                        name='error_params_b_q_scale_by_%s' % sn(ran_gf)
                    )

                    error_params_b_q_dist = Normal(
                        loc=error_params_b_q_loc,
                        scale=self.constraint_fn(error_params_b_q_scale) + self.epsilon,
                        name='error_params_b_q_by_%s' % sn(ran_gf)
                    )

                    error_params_b = tf.cond(self.use_MAP_mode, error_params_b_q_dist.mean, error_params_b_q_dist.sample)

                    error_params_b_summary = error_params_b_q_dist.mean()

                    if self.declare_priors_ranef:
                        # Prior distribution
                        error_params_b_prior_dist = Normal(
                            loc=0.,
                            scale=error_params_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            name='error_params_b_by_%s' % sn(ran_gf)
                        )
                        self.kl_penalties_base['error_params_b_by_%s' % sn(ran_gf)] = {
                            'loc': 0.,
                            'scale': error_params_b_sd_prior * self.ranef_to_fixef_prior_sd_ratio,
                            'val': error_params_b_q_dist.kl_divergence(error_params_b_prior_dist)
                        }

                return error_params_b, error_params_b_summary




    ######################################################
    #
    #  Public methods
    #
    ######################################################


    def report_settings(self, indent=0):
        out = super(CDRNNBayes, self).report_settings(indent=indent)
        for kwarg in CDRNNBAYES_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        out += '\n'

        return out
