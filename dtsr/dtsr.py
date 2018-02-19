import os
from collections import defaultdict
from numpy import inf
import pandas as pd

pd.options.mode.chained_assignment = None
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
from .formula import *
from .util import *
from .plot import *




class DTSR(object):
    """
    Abstract base class for DTSR. Bayesian (BDTSR) and Neural Network (NNDTSR) implementations inherit from DTSR.
    """

    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    def __init__(
            self,
            form_str,
            X,
            y,
            outdir,
            history_length = None,
            low_memory = True,
            pc = False,
            float_type='float32',
            int_type='int32',
            minibatch_size=128,
            eval_minibatch_size=100000,
            log_freq=1,
            log_random=True,
            save_freq=1,
            optim='Adam',
            learning_rate=0.01,
            learning_rate_min=1e-4,
            lr_decay_family=None,
            lr_decay_steps=25,
            lr_decay_rate=0.,
            lr_decay_staircase=False,
            init_sd=1,
            regularizer=None,
            regularizer_scale=0.01
        ):

        ## Save initialization settings
        self.form_str = form_str
        self.outdir = outdir
        self.history_length = history_length
        self.low_memory = low_memory
        if not np.isfinite(self.history_length):
            assert self.low_memory, 'Incompatible DTSR settings: low_memory=False requires finite history_length'
        self.pc = pc
        self.float_type = float_type
        self.minibatch_size = minibatch_size
        self.int_type = int_type
        self.eval_minibatch_size = eval_minibatch_size
        self.log_random = log_random
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.optim_name = optim
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.lr_decay_family = lr_decay_family
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_staircase = lr_decay_staircase
        self.init_sd = init_sd
        self.regularizer_name = regularizer
        self.regularizer_scale = regularizer_scale
        self.n_train = len(y)

        self.__initialize_metadata__()

        ## Set up hash table for random effects lookup
        self.rangf_map_base = []
        self.rangf_map = []
        self.rangf_n_levels = []
        for i in range(len(self.rangf)):
            gf = self.rangf[i]
            keys = np.sort(y[gf].astype('str').unique())
            vals = np.arange(len(keys), dtype=self.INT_NP)
            rangf_map = pd.DataFrame({'id':vals},index=keys).to_dict()['id']
            oov_id = len(keys)+1
            self.rangf_map_base.append(rangf_map)
            self.rangf_n_levels.append(oov_id)
        # Can't pickle defaultdict because it requires a lambda term for the default value,
        # so instead we pickle a normal dictionary (``rangf_map_base``) and compute the defaultdict
        # from it.
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(defaultdict(lambda:self.rangf_n_levels[i], self.rangf_map_base[i]))

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self.y_mu_init = float(y[self.dv].mean())
        self.y_scale_init = float(y[self.dv].std())

        if self.pc:
            _, self.eigenvec, self.eigenval, self.impulse_means, self.impulse_sds = pca(X[self.src_impulse_names_norate])
            self.plot_v()
        else:
            self.eigenvec = self.eigenval = self.impulse_means = self.impulse_sds = None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.y_mu_init_tf = tf.constant(self.y_mu_init, dtype=self.FLOAT_TF)
                self.y_scale_init_tf = tf.constant(self.y_scale_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(1e-35, dtype=self.FLOAT_TF)

    def __initialize_metadata__(self):
        ## Compute secondary data from intialization settings
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        if np.isfinite(self.minibatch_size):
            self.n_train_minibatch = math.ceil(float(self.n_train) / self.minibatch_size)
            self.minibatch_scale = float(self.n_train) / self.minibatch_size
        else:
            self.n_train_minibatch = 1
            self.minibatch_scale = 1
        if np.isfinite(self.eval_minibatch_size):
            self.n_eval_minibatch = math.ceil(float(self.n_train) / self.eval_minibatch_size)
        else:
            self.n_eval_minibatch = 1
            self.eval_minibatch_size = 1
        self.regularizer_losses = []

        # Initialize lookup tables of network objects
        self.irf_lambdas = {}
        self.irf_params = {}
        self.irf_params_summary = {}
        self.irf = {}
        self.irf_plot = {}
        self.irf_mc = {}
        self.mc_integrals = {}
        if self.pc:
            self.src_irf_plot = {}
            self.src_irf_mc = {}
            self.src_mc_integrals = {}
        self.irf_impulses = {}
        self.convolutions = {}

        # Initialize model metadata
        self.form = Formula(self.form_str)
        f = self.form
        self.dv = f.dv
        self.has_intercept = f.has_intercept
        self.rangf = f.rangf

        if self.pc:
            # Initialize source tree metadata
            self.t_src = self.form.t
            t_src = self.t_src
            self.src_node_table = t_src.node_table()
            self.src_coef_names = t_src.coef_names()
            self.src_fixed_coef_names = t_src.fixed_coef_names()
            self.src_impulse_names = t_src.impulse_names()
            self.src_terminal_names = t_src.terminal_names()
            self.src_atomic_irf_names_by_family = t_src.atomic_irf_by_family()
            self.src_coef2impulse = t_src.coef2impulse()
            self.src_impulse2coef = t_src.impulse2coef()
            self.src_coef2terminal = t_src.coef2terminal()
            self.src_terminal2coef = t_src.terminal2coef()
            self.src_impulse2terminal = t_src.impulse2terminal()
            self.src_terminal2impulse = t_src.terminal2impulse()
            self.src_coef_by_rangf = t_src.coef_by_rangf()
            self.src_irf_by_rangf = t_src.irf_by_rangf()

            # Initialize PC tree metadata
            self.n_pc = len(self.src_impulse_names)
            self.has_rate = 'rate' in self.src_impulse_names
            if self.has_rate:
                self.n_pc -= 1
            pointers = {}
            self.t = self.t_src.pc_transform(self.n_pc, pointers)[0]
            self.fw_pointers, self.bw_pointers = IRFNode.pointers2namemmaps(pointers)
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.impulse_names = t.impulse_names()
            self.terminal_names = t.terminal_names()
            self.atomic_irf_names_by_family = t.atomic_irf_by_family()
            self.coef2impulse = t.coef2impulse()
            self.impulse2coef = t.impulse2coef()
            self.coef2terminal = t.coef2terminal()
            self.terminal2coef = t.terminal2coef()
            self.impulse2terminal = t.impulse2terminal()
            self.terminal2impulse = t.terminal2impulse()
            self.coef_by_rangf = t.coef_by_rangf()
            self.irf_by_rangf = t.irf_by_rangf()

            # Compute names and indices of source impulses excluding rate term
            self.src_impulse_names_norate = list(filter(lambda x: x != 'rate', self.src_impulse_names))
            self.src_terminal_ix_norate = names2ix(self.src_impulse_names_norate, self.src_impulse_names)
            self.src_terminal_ix_rate = np.setdiff1d(np.arange(len(self.src_impulse_names)),
                                                     self.src_impulse_names_norate)

            # Compute names and indices of PC impulses excluding rate term
            self.impulse_names_norate = list(filter(lambda x: x != 'rate', self.impulse_names))
            self.terminal_ix_norate = names2ix(self.impulse_names_norate, self.impulse_names)
            self.terminal_ix_rate = np.setdiff1d(np.arange(len(self.impulse_names)), self.impulse_names_norate)

            # Compute names and indices of source coefficients excluding rate term
            self.src_coef_names_norate = list(filter(
                lambda x: not ('rate' in self.src_impulse2coef and x in self.src_impulse2coef['rate']),
                self.src_coef_names
            ))
            self.src_coef_ix_norate = names2ix(self.src_coef_names_norate, self.src_coef_names)
            self.src_coef_names_rate = list(filter(
                lambda x: 'rate' in self.src_impulse2coef and x in self.src_impulse2coef['rate'],
                self.src_coef_names
            ))
            self.src_coef_ix_rate = names2ix(self.src_coef_names_rate, self.src_coef_names)

            # Compute names and indices of PC coefficients excluding rate term
            self.coef_names_norate = list(filter(
                lambda x: not ('rate' in self.impulse2coef and x in self.impulse2coef['rate']),
                self.coef_names
            ))
            self.coef_ix_norate = names2ix(self.src_coef_names_norate, self.src_coef_names)
            self.coef_names_rate = list(filter(
                lambda x: 'rate' in self.impulse2coef and x in self.impulse2coef['rate'],
                self.coef_names
            ))
            self.coef_ix_rate = names2ix(self.coef_names_rate, self.coef_names)
        else:
            # Initialize tree metadata
            self.t = self.form.t
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.impulse_names = t.impulse_names()
            self.terminal_names = t.terminal_names()
            self.atomic_irf_names_by_family = t.atomic_irf_by_family()
            self.coef2impulse = t.coef2impulse()
            self.impulse2coef = t.impulse2coef()
            self.coef2terminal = t.coef2terminal()
            self.terminal2coef = t.terminal2coef()
            self.impulse2terminal = t.impulse2terminal()
            self.terminal2impulse = t.terminal2impulse()
            self.coef_by_rangf = t.coef_by_rangf()
            self.irf_by_rangf = t.irf_by_rangf()

        if self.log_random:
            self.summary_random_writers = {}
            self.summary_random_indexers = {}
            self.summary_random = {}

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError




    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def __initialize_inputs__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                if self.pc:
                    n_impulse = len(self.src_impulse_names)
                else:
                    n_impulse = len(self.impulse_names)

                self.X = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='X'
                )
                self.rate = tf.placeholder(
                    shape=[None],
                    dtype=self.FLOAT_TF,
                    name='rate'
                )
                self.time_X = tf.placeholder(
                    shape=[None, self.history_length],
                    dtype=self.FLOAT_TF,
                    name='time_X'
                )

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)

                # Linspace tensor used for plotting
                self.support_start = tf.placeholder(self.FLOAT_TF, shape=[], name='support_start')
                self.n_time_units = tf.placeholder(self.FLOAT_TF, shape=[], name='n_time_units')
                self.n_points_per_time_unit = tf.placeholder(self.INT_TF, shape=[], name='n_points_per_time_unit')
                self.support = tf.lin_space(
                    self.support_start,
                    self.n_time_units+self.support_start,
                    tf.cast(self.n_time_units * tf.cast(self.n_points_per_time_unit, self.FLOAT_TF), self.INT_TF) + 1,
                    name='support'
                )
                self.support = tf.expand_dims(self.support, -1)
                self.support = tf.cast(self.support, dtype=self.FLOAT_TF)
                self.dd_support = tf.concat(
                    [
                        tf.ones((1, 1), dtype=self.FLOAT_TF),
                        tf.zeros((tf.shape(self.support)[0] - 1, 1), dtype=self.FLOAT_TF)
                    ],
                    axis=0
                )

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)
                    rate_ix = names2ix('rate', self.src_impulse_names)
                    self.X_rate = tf.gather(self.X, rate_ix, axis=-1)

                if self.regularizer_name is not None:
                    self.regularizer = getattr(tf.contrib.layers, '%s_regularizer' %self.regularizer_name)(self.regularizer_scale)

    def __initialize_low_memory_inputs__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)

                self.X = tf.placeholder(shape=[None, len(self.terminal_names)], dtype=self.FLOAT_TF, name=sn('X'))
                self.time_X = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_X'))

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)

                self.first_obs = tf.placeholder(shape=[None], dtype=self.INT_TF, name=sn('first_obs'))
                self.last_obs = tf.placeholder(shape=[None], dtype=self.INT_TF, name=sn('last_obs'))

                # Linspace tensor used for plotting
                self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)

                self.global_step = tf.Variable(0, name=sn('global_step'), trainable=False)
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(0, name=sn('global_batch_step'), trainable=False)
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

    def __intialize_intercept(self, ran_gf=None, rangf_n_levels=None):
        raise NotImplementedError

    def __initialize_coefficient(self, coef_ids=None, ran_gf=None, rangf_n_levels=None):
        raise NotImplementedError

    def __initialize_intercepts_coefficients__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.has_intercept[None]:
                    self.intercept, self.intercept_summary = self.__initialize_intercept__()
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_summary,
                        collections=['params']
                    )
                else:
                    self.intercept = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')

                fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)
                coefficient_fixed_mask = np.zeros(len(self.coef_names), dtype=self.FLOAT_NP)
                coefficient_fixed_mask[fixef_ix] = 1.
                coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)

                coef_ids = self.coef_names

                self.coefficient_fixed, self.coefficient_summary = self.__initialize_coefficient__(coef_ids=coef_ids)

                self.coefficient_fixed *= coefficient_fixed_mask
                self.coefficient_summary *= coefficient_fixed_mask
                self.coefficient = self.coefficient_fixed


                for i in range(len(self.coef_names)):
                    tf.summary.scalar(
                        'coefficient' + '/%s' % self.coef_names[i],
                        self.coefficient_summary[i],
                        collections=['params']
                    )

                # if self.pc:
                #     src_coef = {}
                #     src_coef_summary = {}
                #     for pc_coef in self.coef_names:
                #         for pc_term_name in self.coef2terminal[pc_coef]:
                #             pc_term = self.node_table[pc_term_name]
                #             if pc_term.impulse.name() == 'rate':
                #                 assert len(self.bw_pointers[pc_term.name()]) == 1, 'PC terminal with impulse=rate maps to more than one source terminal'
                #                 term_name = self.bw_pointers[pc_term.name()][0]
                #                 term = self.src_node_table[term_name]
                #                 pc_coef_ix = names2ix(pc_term.coef_id(), self.coef_names)
                #                 computed_coef_summary = tf.gather(self.coefficient_summary, pc_coef_ix)
                #                 if term.coef_id() in src_coef_summary:
                #                     src_coef_summary[term.coef_id()] += computed_coef_summary
                #                 else:
                #                     src_coef_summary[term.coef_id()] = computed_coef_summary
                #             else:
                #                 for term_name in self.bw_pointers[pc_term.name()]:
                #                     term = self.src_node_table[term_name]
                #                     pc_imp_ix = names2ix(pc_term.impulse.name(), self.impulse_names_norate)
                #                     imp_ix = names2ix(term.impulse.name(), self.src_impulse_names_norate)
                #                     computed_coef_summary = self.__apply_pc__(
                #                         self.coefficient_summary,
                #                         src_ix=imp_ix,
                #                         pc_ix=pc_imp_ix,
                #                         inv=True
                #                     )
                #                     if term.coef_id() in src_coef_summary:
                #                         src_coef_summary[term.coef_id()] += computed_coef_summary
                #                     else:
                #                         src_coef_summary[term.coef_id()] = computed_coef_summary
                #
                #     self.src_coefficient_summary = tf.stack([src_coef_summary[x] for x in self.src_coef_names], axis=1)
                #
                #     for i in range(len(self.src_coef_names)):
                #         tf.summary.scalar(
                #             'coefficient_src' + '/%s' % self.src_coef_names[i],
                #             self.src_coefficient_summary[0,i],
                #             collections=['params']
                #         )

                self.coefficient = tf.expand_dims(self.coefficient, 0)

                self.ransl = False
                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    mask_row_np = np.ones(self.rangf_n_levels[i], dtype=self.FLOAT_NP)
                    mask_row_np[self.rangf_n_levels[i] - 1] = 0
                    mask_row = tf.constant(mask_row_np, dtype=self.FLOAT_TF)

                    if self.has_intercept[gf]:
                        intercept_random, intercept_random_summary = self.__initialize_intercept__(
                            ran_gf=gf,
                            rangf_n_levels=self.rangf_n_levels[i]
                        )
                        intercept_random *= mask_row
                        intercept_random_summary *= mask_row
                        intercept_random -= tf.reduce_mean(intercept_random, axis=0)
                        intercept_random_summary -= tf.reduce_mean(intercept_random_summary, axis=0)

                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                'by_%s/intercept' % gf,
                                intercept_random_summary,
                                collections=['random']
                            )

                    coefs = self.coef_by_rangf.get(gf, [])
                    if len(coefs) > 0:
                        coef_ix = names2ix(coefs, self.coef_names)
                        mask_col_np = np.zeros(len(self.coef_names))
                        mask_col_np[coef_ix] = 1.
                        mask_col = tf.constant(mask_col_np, dtype=self.FLOAT_TF)
                        self.ransl = True

                        coefficient_random, coefficient_random_summary = self.__initialize_coefficient__(
                            coef_ids=coef_ids,
                            ran_gf=gf,
                            rangf_n_levels=self.rangf_n_levels[i]
                        )

                        coefficient_random *= mask_col
                        coefficient_random_summary *= mask_col
                        coefficient_random *= tf.expand_dims(mask_row, -1)
                        coefficient_random_summary *= tf.expand_dims(mask_row, -1)

                        coefficient_random -= tf.reduce_mean(coefficient_random, axis=0)
                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(coefs)):
                                coef_name = coefs[j]
                                ix = coef_ix[j]
                                coef_name = coefs[j]
                                ix = coef_ix[j]
                                tf.summary.histogram(
                                    'by_%s/coefficient/%s' % (gf, coef_name),
                                    coefficient_random_summary[:, ix],
                                    collections=['random']
                                )

    def __initialize_irf_lambdas__(self):

        with self.sess.as_default():
            with self.sess.graph.as_default():

                def exponential(params):
                    pdf = tf.contrib.distributions.Exponential(rate=params[:,0:1]).prob
                    return lambda x: pdf(x + self.epsilon)

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['SteepExp'] = exponential

                def gamma(params):
                    pdf = tf.contrib.distributions.Gamma(concentration=params[:,0:1],
                                                         rate=params[:,1:2],
                                                         validate_args=False).prob
                    return lambda x: pdf(x + self.epsilon)

                self.irf_lambdas['Gamma'] = gamma

                self.irf_lambdas['GammaKgt1'] = gamma

                def shifted_gamma(params):
                    pdf = tf.contrib.distributions.Gamma(concentration=params[:,0:1],
                                                         rate=params[:,1:2],
                                                         validate_args=False).prob
                    return lambda x: pdf(x - params[:,2:3] + self.epsilon)

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma

                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma

                def normal(params):
                    pdf = tf.contrib.distributions.Normal(loc=params[:,0:1], scale=params[:,1:2]).prob
                    return lambda x: pdf(x)

                self.irf_lambdas['Normal'] = normal

                def skew_normal(params):
                    mu = params[:,0:1]
                    sigma = params[:,1:2]
                    alpha = params[:,2:3]
                    stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
                    stdnorm_pdf = stdnorm.prob
                    stdnorm_cdf = stdnorm.cdf
                    return lambda x: 2 / sigma * stdnorm_pdf((x - mu) / sigma) * stdnorm_cdf(alpha * (x - mu) / sigma)

                self.irf_lambdas['SkewNormal'] = skew_normal

                def emg(params):
                    mu = params[:,0:1]
                    sigma = params[:,1:2]
                    L = params[:,2:3]
                    return lambda x: L / 2 * tf.exp(0.5 * L * (2. * mu + L * sigma ** 2. - 2. * x)) * tf.erfc(
                        (mu + L * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))

                self.irf_lambdas['EMG'] = emg

                def beta_prime(params):
                    alpha = params[:,1:2]
                    beta = params[:,2:3]
                    return lambda x: (x + self.epsilon) ** (alpha - 1.) * (1. + (x + self.epsilon)) ** (-alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.stack([alpha, beta], axis=0))))

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(params):
                    alpha = params[:,0:1]
                    beta = params[:,1:2]
                    delta = params[:,2:3]
                    return lambda x: (x - delta + self.epsilon) ** (alpha - 1) * (1 + (x - delta + self.epsilon)) ** (
                    -alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.stack([alpha, beta], axis=0))))

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

    def __initialize_irf_params__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
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
                        L, L_mean = self.__initialize_irf_param__('L', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([L], axis=1)
                        params_summary =  tf.stack([L_mean], axis=1)
                    if family == 'SteepExp':
                        L, L_mean = self.__initialize_irf_param__('L', irf_ids, mean=25, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([L], axis=1)
                        params_summary =  tf.stack([L_mean], axis=1)
                    elif family in ['Gamma', 'GammaKgt1']:
                        k, k_mean = self.__initialize_irf_param__('k', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        theta, theta_mean = self.__initialize_irf_param__('theta', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([k, theta], axis=1)
                        params_summary = tf.stack([k_mean, theta_mean], axis=1)
                    elif family in ['ShiftedGamma', 'ShiftedGammaKgt1']:
                        k, k_mean = self.__initialize_irf_param__('k', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        theta, theta_mean = self.__initialize_irf_param__('theta', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        delta, delta_mean = self.__initialize_irf_param__('delta', irf_ids, mean=-1, ub=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([k, theta, delta], axis=1)
                        params_summary = tf.stack([k_mean, theta_mean, delta_mean], axis=1)
                    elif family == 'Normal':
                        mu, mu_mean = self.__initialize_irf_param__('mu', irf_ids, irf_by_rangf=irf_by_rangf)
                        sigma, sigma_mean = self.__initialize_irf_param__('sigma', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([mu, sigma], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean], axis=1)
                    elif family == 'SkewNormal':
                        mu, mu_mean = self.__initialize_irf_param__('mu', irf_ids, irf_by_rangf=irf_by_rangf)
                        sigma, sigma_mean = self.__initialize_irf_param__('sigma', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        alpha, alpha_mean = self.__initialize_irf_param__('alpha', irf_ids, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([mu, sigma, alpha], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean, alpha_mean], axis=1)
                    elif family == 'EMG':
                        mu, mu_mean = self.__initialize_irf_param__('mu', irf_ids, irf_by_rangf=irf_by_rangf)
                        sigma, sigma_mean = self.__initialize_irf_param__('sigma', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        L, L_mean = self.__initialize_irf_param__('L', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([mu, sigma, L], axis=1)
                        params_summary = tf.stack([mu_mean, sigma_mean, L_mean], axis=1)
                    elif family == 'BetaPrime':
                        alpha, alpha_mean = self.__initialize_irf_param__('alpha', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        beta, beta_mean = self.__initialize_irf_param__('beta', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([alpha, beta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean], axis=1)
                    elif family == 'ShiftedBetaPrime':
                        alpha, alpha_mean = self.__initialize_irf_param__('alpha', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        beta, beta_mean = self.__initialize_irf_param__('beta', irf_ids, mean=1, lb=0, irf_by_rangf=irf_by_rangf)
                        delta, delta_mean = self.__initialize_irf_param__('delta', irf_ids, mean=-1, ub=0, irf_by_rangf=irf_by_rangf)
                        params = tf.stack([alpha, beta, delta], axis=1)
                        params_summary = tf.stack([alpha_mean, beta_mean, delta_mean], axis=1)
                    else:
                        raise ValueError('Impulse response function "%s" is not currently supported.' % family)
                    for i in range(len(irf_ids)):
                        id = irf_ids[i]
                        ix = names2ix(id, self.atomic_irf_names_by_family[family])
                        assert id not in self.irf_params, 'Duplicate IRF node name already in self.irf_params'
                        self.irf_params[id] = tf.gather(params, ix, axis=2)
                        self.irf_params_summary[id] = tf.gather(params_summary, ix, axis=2)

    def __initialize_irf_param__(self, param_name, ids, mean=0, lb=None, ub=None, irf_by_rangf=None):
        raise NotImplementedError

    def __initialize_irfs__(self, t):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if t.family is None:
                    self.irf[t.name()] = []
                elif t.family == 'Terminal':
                    coef_name = self.node_table[t.name()].coef_id()
                    coef_ix = names2ix(coef_name, self.coef_names)

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = self.irf[t.p.name()][:]

                    assert not t.name() in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'scaled': self.irf_mc[t.p.name()]['atomic']['unscaled'] * tf.gather(self.coefficient_fixed, coef_ix),
                            'unscaled': self.irf_mc[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_mc[t.p.name()]['composite']['unscaled'] * tf.gather(self.coefficient_fixed, coef_ix),
                            'unscaled': self.irf_mc[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.irf_plot[t.p.name()]['atomic']['unscaled'] * tf.gather(self.coefficient_summary, coef_ix),
                            'unscaled': self.irf_plot[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_plot[t.p.name()]['composite']['unscaled'] * tf.gather(self.coefficient_summary, coef_ix),
                            'unscaled': self.irf_plot[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.mc_integrals, 'Duplicate IRF node name already in self.mc_integrals'
                    self.mc_integrals[t.name()] = self.__reduce_interpolated_sum__(
                        self.irf_mc[t.name()]['composite']['scaled'],
                        self.support[:,0],
                        axis=0
                    )

                elif t.family == 'DiracDelta':
                    assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in DTSR formula strings'
                    assert not t.impulse == 'rate', '"rate" is a reserved keyword in DTSR formula strings and cannot be used under DiracDelta'


                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = self.irf[t.p.name()][:]

                    assert not t.name() in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        },
                        'composite' : {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        },
                        'composite' : {
                            'scaled': self.dd_support,
                            'unscaled': self.dd_support
                        }
                    }
                else:
                    params = self.irf_params[t.irf_id()]
                    params_summary = self.irf_params_summary[t.irf_id()]

                    atomic_irf = self.__new_irf__(self.irf_lambdas[t.family], params)
                    atomic_irf_plot = self.__new_irf__(self.irf_lambdas[t.family], params_summary)
                    if t.p.name() in self.irf:
                        irf = self.irf[t.p.name()][:] + [atomic_irf]
                        irf_plot = self.irf[t.p.name()][:] + [atomic_irf_plot]
                    else:
                        irf = [atomic_irf]
                        irf_plot = [atomic_irf_plot]

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = irf

                    atomic_irf_mc = atomic_irf(self.support)[0]
                    atomic_irf_plot = atomic_irf_plot(self.support)[0]

                    if len(irf_plot) > 1:
                        composite_irf_mc = irf[0]
                        for p_irf in irf[1:]:
                            composite_irf_mc = self.__merge_irf__(composite_irf_mc, p_irf, self.t_delta)
                        composite_irf_mc = composite_irf_mc(self.support)[0]

                        composite_irf_plot = irf_plot[0]
                        for p_irf in irf_plot[1:]:
                            composite_irf_plot = self.__merge_irf__(composite_irf_plot, p_irf, self.t_delta)
                        composite_irf_plot = composite_irf_plot(self.support)[0]

                    else:
                        composite_irf_mc = atomic_irf_mc
                        composite_irf_plot = atomic_irf_plot

                    assert t.name() not in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'unscaled': atomic_irf_mc,
                            'scaled': atomic_irf_mc
                        },
                        'composite': {
                            'unscaled': composite_irf_mc,
                            'scaled': composite_irf_mc
                        }
                    }

                    assert t.name() not in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'unscaled': atomic_irf_plot,
                            'scaled': atomic_irf_plot
                        },
                        'composite': {
                            'unscaled': composite_irf_plot,
                            'scaled': composite_irf_plot
                        }
                    }

                for c in t.children:
                    self.__initialize_irfs__(c)

    def __initialize_backtransformed_irf_plot__(self, t):
        if self.pc:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if t.name() in self.irf_plot:
                        src_irf_names = self.bw_pointers[t.name()]
                        t_impulse_names = t.impulse_names()
                        if t_impulse_names == ['rate'] and len(src_irf_names) == 1 and self.src_node_table[src_irf_names[0]].impulse_names() == ['rate']:
                            self.src_irf_plot[src_irf_names[0]] = self.irf_plot[t.name()]
                            self.src_irf_mc[src_irf_names[0]] = self.irf_mc[t.name()]
                            if t.name() in self.mc_integrals:
                                self.src_mc_integrals[src_irf_names[0]] = self.mc_integrals[t.name()]
                        else:
                            for src_irf_name in src_irf_names:
                                src_irf = self.src_node_table[src_irf_name]
                                src_impulse_names = src_irf.impulse_names()
                                src_impulse_names_norate = list(filter(lambda x: x != 'rate', src_impulse_names))
                                src_ix = names2ix(src_impulse_names_norate, self.src_impulse_names_norate)
                                if len(src_ix) > 0:
                                    impulse_names = t.impulse_names()
                                    impulse_names_norate = list(filter(lambda x: x != 'rate', impulse_names))
                                    pc_ix = names2ix(impulse_names_norate, self.impulse_names_norate)
                                    if len(pc_ix) > 0:
                                        e = self.e
                                        e = tf.gather(e, src_ix, axis=0)
                                        e = tf.gather(e, pc_ix, axis=1)
                                        e = tf.reduce_sum(e, axis=1)

                                        if src_irf_name in self.src_irf_plot:
                                            self.src_irf_plot[src_irf_name]['atomic']['scaled'] += tf.reduce_sum(self.irf_plot[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['atomic']['unscaled'] += tf.reduce_sum(self.irf_plot[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['composite']['scaled'] += tf.reduce_sum(self.irf_plot[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_plot[src_irf_name]['composite']['unscaled'] += tf.reduce_sum(self.irf_plot[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                        else:
                                            self.src_irf_plot[src_irf_name] = {
                                                'atomic': {
                                                    'scaled': tf.reduce_sum(self.irf_plot[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_plot[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                                },
                                                'composite': {
                                                    'scaled': tf.reduce_sum(self.irf_plot[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_plot[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                                }
                                            }
                                        if src_irf_name in self.src_irf_mc:
                                            self.src_irf_mc[src_irf_name]['atomic']['scaled'] += tf.reduce_sum(self.irf_mc[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['atomic']['unscaled'] += tf.reduce_sum(self.irf_mc[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['composite']['scaled'] += tf.reduce_sum(self.irf_mc[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True)
                                            self.src_irf_mc[src_irf_name]['composite']['unscaled'] += tf.reduce_sum(self.irf_mc[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                        else:
                                            self.src_irf_mc[src_irf_name] = {
                                                'atomic': {
                                                    'scaled': tf.reduce_sum(self.irf_mc[t.name()]['atomic']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_mc[t.name()]['atomic']['unscaled'] * e, axis=1, keep_dims=True)
                                                },
                                                'composite': {
                                                    'scaled': tf.reduce_sum(self.irf_mc[t.name()]['composite']['scaled'] * e, axis=1, keep_dims=True),
                                                    'unscaled': tf.reduce_sum(self.irf_mc[t.name()]['composite']['unscaled'] * e, axis=1, keep_dims=True)
                                                }
                                            }
                                        if t.name() in self.mc_integrals:
                                            if src_irf_name in self.src_mc_integrals:
                                                self.src_mc_integrals[src_irf_name] += tf.reduce_sum(self.mc_integrals[t.name()] * e, axis=0, keep_dims=True)
                                            else:
                                                self.src_mc_integrals[src_irf_name] = tf.reduce_sum(self.mc_integrals[t.name()] * e, axis=0, keep_dims=True)

                    for c in t.children:
                        if c.name() in self.irf_plot:
                            self.__initialize_backtransformed_irf_plot__(c)

    def __initialize_impulses__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for name in self.terminal_names:
                    t = self.node_table[name]
                    impulse_name = t.impulse.name()
                    impulse_ix = names2ix(impulse_name, self.impulse_names)

                    if t.p.family == 'DiracDelta':
                        if self.pc:
                            if impulse_name == 'rate':
                                impulse = self.X_rate[:, -1, :]
                            else:
                                src_term_names = self.bw_pointers[t.name()]
                                src_impulse_names = set()
                                for x in src_term_names:
                                    src_impulse_names.add(self.src_node_table[x].impulse.name())
                                src_impulse_names = list(src_impulse_names)
                                src_impulse_ix = names2ix(src_impulse_names, self.src_impulse_names)
                                X = self.X[:, -1, :]
                                impulse = self.__apply_pc__(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)[:, -1, :]
                    else:
                        if self.pc:
                            if impulse_name == 'rate':
                                impulse = self.X_rate
                            else:
                                src_term_names = self.bw_pointers[t.name()]
                                src_impulse_names = set()
                                for x in src_term_names:
                                    src_impulse_names.add(self.src_node_table[x].impulse.name())
                                src_impulse_names = list(src_impulse_names)
                                src_impulse_ix = names2ix(src_impulse_names, self.src_impulse_names)
                                X = self.X
                                impulse = self.__apply_pc__(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)

                    self.irf_impulses[name] = impulse

    def __initialize_convolutions__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for name in self.terminal_names:
                    t = self.node_table[name]

                    if t.p.family == 'DiracDelta':
                        self.convolutions[name] = self.irf_impulses[name]
                    else:
                        if t.cont:
                            impulse = self.irf_impulses[name]
                            irf = self.irf[name]
                            if len(irf) > 1:
                                cur_irf = irf[0]
                                for p_irf in irf[1:]:
                                    cur_irf = self.__merge_irf__(cur_irf, p_irf, self.t_delta)
                                irf = cur_irf(self.support)[0]
                            else:
                                irf = irf[0]

                            irf_seq = irf(self.t_delta)

                            self.convolutions[name] = self.__reduce_interpolated_sum__(impulse * irf_seq, self.time_X, axis=1)
                        else:
                            impulse = self.irf_impulses[name]

                            irf = self.irf[name]
                            if len(irf) > 1:
                                cur_irf = irf[0]
                                for p_irf in irf[1:]:
                                    cur_irf = self.__merge_irf__(cur_irf, p_irf, self.t_delta)
                                irf = cur_irf(self.support)[0]
                            else:
                                irf = irf[0]

                            irf_seq = irf(self.t_delta)

                            self.convolutions[name] = tf.reduce_sum(impulse * irf_seq, axis=1)


    def __initialize_low_memory_convolutional_feedforward__(self, t, inputs, t_delta):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                convolved = []
                for f in t.children:
                    preterminals_discr = []
                    terminals_discr = []
                    preterminals_cont = []
                    terminals_cont = []
                    child_nodes = sorted(t.children[f].keys())
                    if f == 'DiracDelta':
                        for i in range(len(child_nodes)):
                            x = t.children[f][child_nodes[i]]
                            terminals_discr += x.impulse
                        terminals_ix = names2ix(terminals_discr, self.form.terminal_names)
                        new_out = tf.gather(inputs[-1], terminals_ix, axis=0)
                        convolved.append(new_out)
                    else:
                        tensor = t.irfs[f](t.tensor)
                        for i in range(len(child_nodes)):
                            x = t.children[f][child_nodes[i]]
                            if x.is_preterminal():
                                if x.cont:
                                    preterminals_cont.append(x.name())
                                    terminals_cont += x.impulse
                                else:
                                    preterminals_discr.append(x.name())
                                    terminals_discr += x.impulse
                            x.tensor = tf.expand_dims(tensor[:, i], -1)
                            if not x.is_preterminal():
                                convolved_cur = self.__initialize_low_memory_convolutional_feedforward__(x, inputs,
                                                                                                         t_delta)
                                convolved += convolved_cur
                        if len(preterminals_discr) > 0 and len(terminals_discr) > 0:
                            preterminals_ix = names2ix(preterminals_discr, child_nodes)
                            terminals_ix = names2ix(terminals_discr, self.form.terminal_names)
                            new_out = tf.reduce_sum(
                                tf.gather(inputs, terminals_ix, axis=1) * tf.gather(tensor, preterminals_ix, axis=1),
                                axis=0
                            )
                            convolved.append(new_out)
                        if len(preterminals_cont) > 0 and len(terminals_cont) > 0:
                            preterminals_ix = names2ix(preterminals_cont, child_nodes)
                            terminals_ix = names2ix(terminals_cont, self.form.terminal_names)
                            new_out = tf.gather(inputs, terminals_ix, axis=1) * tf.gather(tensor, preterminals_ix,
                                                                                          axis=1)
                            new_out_cur = tf.pad(
                                new_out[1:],
                                tf.constant([[1, 0], [0, 0]]),
                                mode='CONSTANT'
                            )
                            new_out_prev = tf.pad(
                                new_out[:-1],
                                tf.constant([[1, 0], [0, 0]]),
                                mode='CONSTANT'
                            )
                            new_out = (new_out_cur + new_out_prev) / 2
                            t_delta_cur = tf.pad(
                                t_delta[1:],
                                tf.constant([[1, 0]]),
                                mode='CONSTANT'
                            )
                            t_delta_prev = tf.pad(
                                t_delta[:-1],
                                tf.constant([[1, 0]]),
                                mode='CONSTANT'
                            )
                            duration = tf.expand_dims(t_delta_cur - t_delta_prev, -1)
                            new_out = tf.reduce_sum(new_out * duration, axis=0)
                            convolved.append(new_out)
                return convolved

    def __construct_network__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.t_delta = tf.expand_dims(tf.expand_dims(self.time_y, -1) - self.time_X, -1)  # Tensor of temporal offsets with shape (?,history_length)
                self.__initialize_irfs__(self.t)
                self.__initialize_impulses__()
                self.__initialize_convolutions__()
                self.__initialize_backtransformed_irf_plot__(self.t)

                convolutions = [self.convolutions[x] for x in self.terminal_names]
                self.X_conv = tf.concat(convolutions, axis=1)

                coef_names = [self.node_table[x].coef_id() for x in self.terminal_names]
                coef_ix = names2ix(coef_names, self.coef_names)
                coef = tf.gather(self.coefficient, coef_ix, axis=1)

                self.out = self.intercept + tf.reduce_sum(self.X_conv*coef, axis=1)

    def __construct_low_memory_network__(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if np.isfinite(self.history_length):
                    history_length = tf.constant(self.history_length, dtype=self.INT_TF)

                def get_coefs_ix(t):
                    coefficients = []
                    for fam in t.children:
                        child_nodes = sorted(t.children[fam].keys())
                        terminals_discr = []
                        terminals_cont = []
                        for i in range(len(child_nodes)):
                            x = t.children[fam][child_nodes[i]]
                            if x.is_preterminal():
                                if x.cont:
                                    terminals_discr.append(x.coef_id())
                                else:
                                    terminals_cont.append(x.coef_id())
                            else:
                                coefficients_cur = get_coefs_ix(x)
                                coefficients += coefficients_cur
                        if len(terminals_discr) > 0:
                            coefs_ix = names2ix(terminals_discr, self.form.coefficient_names)
                            coefficients.append(coefs_ix)
                        if len(terminals_cont) > 0:
                            coefs_ix = names2ix(terminals_cont, self.form.coefficient_names)
                            coefficients.append(coefs_ix)
                    return coefficients

                def convolve_events(time_target, first_obs, last_obs):
                    if np.isfinite(self.history_length):
                        inputs = self.X[tf.maximum(first_obs, last_obs-history_length):last_obs]
                        input_times = self.time_X[tf.maximum(first_obs, last_obs-history_length):last_obs]
                    else:
                        inputs = self.X[first_obs:last_obs]
                        input_times = self.time_X[first_obs:last_obs]
                    t_delta = time_target - input_times

                    self.t.tensor = tf.expand_dims(t_delta, -1)
                    convolved = self.__initialize_low_memory_convolutional_feedforward__(self.t, inputs, t_delta)
                    convolved = tf.concat(convolved, axis=0)

                    return convolved

                coefs_ix = tf.constant(np.concatenate(get_coefs_ix(self.t)))
                coef_aligned = tf.gather(self.coefficient, coefs_ix, axis=1)

                self.X_conv = tf.map_fn(
                    lambda x: convolve_events(*x),
                    [self.time_y, self.first_obs, self.last_obs],
                    parallel_iterations=10,
                    dtype=self.FLOAT_TF
                )

                self.out = self.intercept + tf.reduce_sum(self.X_conv*coef_aligned, axis=1)

    def __initialize_objective__(self):
        raise NotImplementedError

    def __initialize_optimizer__(self, name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase
                    self.lr = getattr(tf.train, self.lr_decay_family + '_decay')(
                        lr,
                        self.global_step,
                        lr_decay_steps,
                        lr_decay_rate,
                        staircase=lr_decay_staircase,
                        name='learning_rate'
                    )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                return {
                    'SGD': lambda x: tf.train.GradientDescentOptimizer(x),
                    'AdaGrad': lambda x: tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

    def __initialize_logging__(self):
        raise NotImplementedError

    def __initialize_saver__(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()





    ######################################################
    #
    #  Private methods
    #
    ######################################################

    def __new_irf__(self, irf_lambda, params, parent_irf=None):
        irf = irf_lambda(params)
        if parent_irf is None:
            def new_irf(x):
                return irf(x)
        else:
            def new_irf(x):
                return irf(parent_irf(x))
        return new_irf

    def __merge_irf__(self, A, B, t_delta):
        raise ValueError('Hierarchical convolutions are not yet supported.')

        # ode = lambda y, t: A(t) * B(t)
        # A_conv_B = tf.contrib.integrate.odeint(ode, , tf.reverse(self.t_delta))
        return

    def __apply_pc__(self, inputs, src_ix=None, pc_ix=None, inv=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if src_ix is None:
                    src_ix = np.arange(self.n_pc)
                if pc_ix is None:
                    pc_ix = np.arange(self.n_pc)

                e = self.e
                e = tf.gather(e, src_ix, axis=0)
                e = tf.gather(e, pc_ix, axis=1)
                if inv:
                    X = tf.gather(inputs, pc_ix, axis=-1)
                    e = tf.transpose(e, [0,1])
                else:
                    X = tf.gather(inputs, src_ix, axis=-1)
                expansions = 0
                while len(X.shape) < 2:
                    expansions += 1
                    X = tf.expand_dims(X, 0)
                outputs = self.__matmul__(X, e)
                if expansions > 0:
                    outputs = tf.squeeze(outputs, axis=list(range(expansions)))
                return outputs

    def __process_mean__(self, mean, lb=None, ub=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                mean = tf.constant(mean, dtype=self.FLOAT_TF)
                if lb is not None:
                    lb = tf.constant(lb, dtype=self.FLOAT_TF)
                if ub is not None:
                    ub = tf.constant(ub, dtype=self.FLOAT_TF)

                if lb is None and ub is None:
                    # Unbounded support
                    mean = mean
                elif lb is not None and ub is None:
                    # Lower-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(mean - lb - self.epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    mean = tf.contrib.distributions.softplus_inverse(-(mean - ub + self.epsilon))
                else:
                    # Finite-interval bounded support
                    mean = tf.contrib.distributions.bijectors.Sigmoid.inverse(
                        (mean - lb - self.epsilon) / ((ub - self.epsilon) - (lb + self.epsilon))
                    )
        return mean, lb, ub

    def __collect_plots__(self):
        switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.plots = {}
                irf_names = [x for x in self.node_table if x in self.irf_plot and not (len(self.node_table[x].children) == 1 and self.node_table[x].children[0].terminal())]

                for a in switches[0]:
                    if a not in self.plots:
                        self.plots[a] = {}
                    for b in switches[1]:
                        plot_y = []
                        for x in irf_names:
                            plot_y.append(self.irf_plot[x][a][b])

                        self.plots[a][b] = {
                            'names': irf_names,
                            'plot': plot_y
                        }

                if self.pc:
                    self.src_plot_tensors = {}
                    irf_names = [x for x in self.src_node_table if x in self.src_irf_plot and not (len(self.src_node_table[x].children) == 1 and self.src_node_table[x].children[0].terminal())]

                    for a in switches[0]:
                        if a not in self.src_plot_tensors:
                            self.src_plot_tensors[a] = {}
                        for b in switches[1]:
                            plot_y = []
                            for x in irf_names:
                                plot_y.append(self.src_irf_plot[x][a][b])

                            self.src_plot_tensors[a][b] = {
                                'names': irf_names,
                                'plot': plot_y
                            }

    def __matmul__(self, A, B):
        """
        Matmul operation that supports broadcasting of A
        :param A: Left tensor (>= 2D)
        :param B: Right tensor (2D)
        :return: Broadcasted matrix multiplication on the last 2 ranks of A and B
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                A_batch_shape = tf.gather(tf.shape(A), list(range(len(A.shape)-1)))
                A = tf.reshape(A, [-1, A.shape[-1]])
                C = tf.matmul(A, B)
                C_shape = tf.concat([A_batch_shape, [C.shape[-1]]], axis=0)
                C = tf.reshape(C, C_shape)
                return C

    def __regularize__(self, var, center=None):
        if self.regularizer_name is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if center is None:
                        reg = tf.contrib.layers.apply_regularization(self.regularizer, [var])
                    else:
                        reg = tf.contrib.layers.apply_regularization(self.regularizer, [var - center])
                    self.regularizer_losses.append(reg)

    def __reduce_interpolated_sum__(self, X, time, axis=0):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                assert len(X.shape) > 0, 'A scalar cannot be interpolated'

                if axis < 0:
                    axis = len(X.shape) + axis

                X_cur_begin = [0] * (len(X.shape) - 1)
                X_cur_begin.insert(axis, 1)
                X_cur_end = [-1] * len(X.shape)
                X_cur = tf.slice(X, X_cur_begin, X_cur_end)

                time_cur_begin = [0] * axis + [1]
                time_cur_end = [-1] * (axis + 1)
                time_cur = tf.slice(time, time_cur_begin, time_cur_end)

                ub = tf.shape(X)[axis] - 1
                X_prev_begin = [0] * len(X.shape)
                X_prev_end = [-1] * (len(X.shape) - 1)
                X_prev_end.insert(axis, ub)
                X_prev = tf.slice(X, X_prev_begin, X_prev_end)

                time_prev_begin = [0] * (axis + 1)
                time_prev_end = [-1] * axis + [ub]
                time_prev = tf.slice(time, time_prev_begin, time_prev_end)

                time_diff = time_cur-time_prev

                for _ in range(axis+1, len(X.shape)):
                    time_diff = tf.expand_dims(time_diff, -1)

                out = tf.reduce_sum((X_prev + X_cur) / 2 * time_diff, axis=axis)

                return out





    ######################################################
    #
    #  Public methods
    #
    ######################################################

    def report_n_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                sys.stderr.write('Trainable variables:\n')
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = np.prod(np.array(v_val).shape)
                    n_params += cur_params
                    sys.stderr.write('  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params))
                sys.stderr.write('Network contains %d total trainable parameters.\n' % n_params)
                sys.stderr.write('\n')

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
            sys.stdout.write(str(self.t))
            sys.stdout.write('\n\n')

            if self.pc:
                sys.stderr.write('Source model tree:\n')
                sys.stdout.write(str(self.t_src))
                sys.stdout.write('\n\n')

        if self.low_memory:
            self.__initialize_low_memory_inputs__()
        else:
            self.__initialize_inputs__()
        self.__initialize_intercepts_coefficients__()
        self.__initialize_irf_lambdas__()
        self.__initialize_irf_params__()
        if self.low_memory:
            self.__construct_low_memory_network__()
        else:
            self.__construct_network__()
        self.__initialize_objective__()
        self.__initialize_logging__()
        self.__initialize_saver__()
        self.load(restore=restore)
        self.__collect_plots__()
        self.report_n_params()

    def save(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver.save(self.sess, self.outdir + '/model.ckpt')

    def load(self, restore=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if restore and os.path.exists(self.outdir + '/checkpoint'):
                    self.saver.restore(self.sess, self.outdir + '/model.ckpt')
                else:
                    self.sess.run(tf.global_variables_initializer())

    def expand_history(self, X, X_time, first_obs, last_obs):
        last_obs = np.array(last_obs, dtype=self.INT_NP)
        first_obs = np.maximum(np.array(first_obs, dtype=self.INT_NP), last_obs - self.history_length)
        X_time = np.array(X_time, dtype=self.FLOAT_NP)
        X = np.array(X, dtype=self.FLOAT_NP)

        X_history = np.zeros((first_obs.shape[0], self.history_length, X.shape[1]))
        time_X_history = np.zeros((first_obs.shape[0], self.history_length))

        for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
            sX = X[first:last]
            sXt = X_time[first:last]
            X_history[i, -sX.shape[0]:] += sX
            time_X_history[i][-len(sXt):] += sXt

        return X_history, time_X_history

    def convolve_inputs(self, X, time_y, gf_y, first_obs, last_obs):

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if not np.isfinite(self.minibatch_size):
            minibatch_size = len(time_y)
        else:
            minibatch_size = self.minibatch_size

        if isinstance(gf_y, pd.Series):
            for i in range(len(self.rangf)):
                c = self.rangf[i]
                gf_y[c] = pd.Series(gf_y[c].astype(str)).map(self.rangf_map[i])
            gf_y = np.array(gf_y, dtype=self.INT_NP)

        if isinstance(X, pd.DataFrame):
            if self.low_memory:
                X_2d = X[impulse_names]
                time_X_2d = np.array(X.time, dtype=self.FLOAT_NP)
                first_obs = np.array(first_obs, dtype=self.INT_NP)
                last_obs = np.array(last_obs, dtype=self.INT_NP)
            else:
                X_3d, time_X_3d = self.expand_history(X[impulse_names], X.time, first_obs, last_obs)

        time_y = np.array(time_y)

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

                X_conv = []
                for j in range(0, len(time_y), minibatch_size):
                    fd_minibatch[self.time_y] = time_y[j:j + minibatch_size]
                    fd_minibatch[self.gf_y] = gf_y[j:j + minibatch_size]
                    if self.low_memory:
                        fd_minibatch[self.first_obs] = first_obs[j:j + minibatch_size]
                        fd_minibatch[self.last_obs] = last_obs[j:j + minibatch_size]
                    else:
                        fd_minibatch[self.X] = X_3d[j:j + minibatch_size]
                        fd_minibatch[self.time_X] = time_X_3d[j:j + minibatch_size]
                    X_conv_cur = self.sess.run(self.X_conv, feed_dict=fd_minibatch)
                    X_conv.append(X_conv_cur)
                X_conv = pd.DataFrame(np.concatenate(X_conv), columns=self.terminal_names)
                return X_conv

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        raise NotImplementedError

    def eval(self, X, y):
        raise NotImplementedError

    def ci_curve(
            self,
            posterior,
            level=95,
            n_samples=1000,
            n_time_units=2.5,
            n_points_per_time_unit=1000
    ):
        fd = {
            self.support_start: 0.,
            self.n_time_units: n_time_units,
            self.n_points_per_time_unit: n_points_per_time_unit
        }

        alpha = 100-float(level)

        samples = [self.sess.run(posterior, feed_dict=fd) for _ in range(n_samples)]
        samples = np.concatenate(samples, axis=1)

        mean = samples.mean(axis=1)
        lower = np.percentile(samples, alpha/2, axis=1)
        upper = np.percentile(samples, 100-(alpha/2), axis=1)

        return (mean, lower, upper)

    def ci_integral(
            self,
            terminal_name,
            level=95,
            n_samples=1000,
            n_time_units=2.5,
            n_points_per_time_unit=1000
    ):
        fd = {
            self.support_start: 0.,
            self.n_time_units: n_time_units,
            self.n_points_per_time_unit: n_points_per_time_unit
        }

        alpha = 100 - float(level)

        if terminal_name in self.mc_integrals:
            posterior = self.mc_integrals[terminal_name]
        else:
            posterior = self.src_mc_integrals[terminal_name]

        samples = [np.squeeze(self.sess.run(posterior, feed_dict=fd)[0]) for _ in range(n_samples)]
        samples = np.stack(samples, axis=0)

        mean = samples.mean(axis=0)
        lower = np.percentile(samples, alpha / 2, axis=0)
        upper = np.percentile(samples, 100 - (alpha / 2), axis=0)

        return (mean, lower, upper)

    def make_plots(
            self,
            total_seconds=2.5,
            points_per_second=1000,
            irf_name_map=None,
            plot_x_inches=7.,
            plot_y_inches=5.,
            cmap=None,
            mc=False,
            n_samples=1000,
            level=95
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.support_start: 0.,
                    self.n_time_units: total_seconds,
                    self.n_points_per_time_unit: points_per_second
                }

                plot_x = self.sess.run(self.support, fd)

                switches = [['atomic', 'composite'], ['scaled', 'unscaled']]

                for a in switches[0]:
                    for b in switches[1]:
                        plot_name = 'irf_%s_%s.png' %(a, b)
                        names = self.plots[a][b]['names']
                        if mc:
                            plot_y = []
                            lq = []
                            uq = []
                            for name in names:
                                mean_cur, lq_cur, uq_cur = self.ci_curve(self.irf_mc[name][a][b])
                                plot_y.append(mean_cur)
                                lq.append(lq_cur)
                                uq.append(uq_cur)
                            lq = np.stack(lq, axis=1)
                            uq = np.stack(uq, axis=1)
                            plot_y = np.stack(plot_y, axis=1)
                            plot_name = 'mc_' + plot_name
                        else:
                            plot_y = [self.sess.run(x, feed_dict=fd) for x in self.plots[a][b]['plot']]
                            lq = None
                            uq = None
                            plot_y = np.concatenate(plot_y, axis=1)

                        plot_convolutions(
                            plot_x,
                            plot_y,
                            names,
                            lq=lq,
                            uq=uq,
                            dir=self.outdir,
                            filename=plot_name,
                            irf_name_map=irf_name_map,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap
                        )

                if self.pc:
                    for a in switches[0]:
                        for b in switches[1]:
                            if b == 'scaled':
                                plot_name = 'src_irf_%s_%s.png' % (a, b)
                                names = self.src_plot_tensors[a][b]['names']
                                if mc:
                                    plot_y = []
                                    lq = []
                                    uq = []
                                    for name in names:
                                        mean_cur, lq_cur, uq_cur = self.ci_curve(self.src_irf_mc[name][a][b])
                                        plot_y.append(mean_cur)
                                        lq.append(lq_cur)
                                        uq.append(uq_cur)
                                    lq = np.stack(lq, axis=1)
                                    uq = np.stack(uq, axis=1)
                                    plot_y = np.stack(plot_y, axis=1)
                                    plot_name = 'mc_' + plot_name
                                else:
                                    plot_y = [self.sess.run(x, feed_dict=fd) for x in self.src_plot_tensors[a][b]['plot']]
                                    lq = None
                                    uq = None
                                    plot_y = np.concatenate(plot_y, axis=1)

                                plot_convolutions(
                                    plot_x,
                                    plot_y,
                                    names,
                                    lq=lq,
                                    uq=uq,
                                    dir=self.outdir,
                                    filename=plot_name,
                                    irf_name_map=irf_name_map,
                                    plot_x_inches=plot_x_inches,
                                    plot_y_inches=plot_y_inches,
                                    cmap=cmap
                                )


    def plot_v(self):
        plot_heatmap(self.eigenvec, self.src_impulse_names_norate, self.impulse_names_norate, dir=self.outdir)
