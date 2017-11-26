import sys
import re
import os
import math
import time
import numpy as np
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
from matplotlib import pyplot as plt


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def sn(string):
    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)


class DTSR(object):
    """Deconvolutional Time Series Regression (DTSR) class
    
    # Arguments
        bform: String or Formula. An R-style linear mixed-effects model formula string or an instance of the Formula class
        irf: String. One of `"exp"` or `"gamma"` (other functions coming soon...)
        optim: String. Optimizer. One of `"Adam"`.
        learning_rate: Float. Defaults to 0.01
        
    """

    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 irf='gamma',
                 optim='Adam',
                 learning_rate=0.01,
                 loss='mse',
                 log_convolution_plots=False,
                 float_type='float32',
                 int_type='int32',
                 log_random=False
                 ):

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self.form_str = form_str
        self.form = Formula(form_str)
        f = self.form

        self.outdir = outdir
        self.irf_str = irf
        self.optim_name = optim
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.log_convolution_plots = log_convolution_plots
        self.rangf_keys = []
        self.rangf_values = []
        self.rangf_n_levels = []
        for i in range(len(f.random)):
            gf = f.random[sorted(f.random.keys())[i]].gf
            self.rangf_keys.append(y[gf].unique().astype('str'))
            self.rangf_values.append(np.arange(len(self.rangf_keys[i]), dtype=np.int32))
            self.rangf_n_levels.append(len(self.rangf_keys[i]) + 1)
        self.intercept_init = float(y[f.dv].mean())
        self.float_type = float_type
        self.FLOAT = getattr(tf, self.float_type)
        self.int_type = int_type
        self.INT = getattr(tf, self.int_type)
        self.log_random = log_random
        self.irf_tree = self.form.irf_tree
        self.preterminals = []

        self.construct_network()

    def construct_network(self):
        f = self.form

        sys.stderr.write('Constructing network from model tree:\n')
        sys.stdout.write(str(f.irf_tree))
        sys.stdout.write('\n')

        self.irf_lambdas = {}
        self.irf_params = {}
        self.atomic_irfs = {}
        self.composite_irfs = {}
        self.atomic_irf_by_family = {}
        self.initialize_irf_lambdas()
        self.build_irf_params()
        if self.log_random:
            self.summary_random_writers = {}
            self.summary_random_indexers = {}
            self.summary_random = {}

        with self.sess.graph.as_default():
            if self.optim_name != 'LBFGS':
                self.optim = self.optim_init(self.optim_name, self.learning_rate)

            self.X = tf.placeholder(shape=[None, len(f.terminals)], dtype=self.FLOAT, name=sn('X_raw'))
            self.time_X = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('time_X'))

            self.y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('y'))
            self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('time_y'))

            ## Build random effects lookup tensor
            ## Using strings for indexing guarantees correct random effects behavior on unseen data
            self.gf_y_raw = tf.placeholder(shape=[None, len(f.random)], dtype=tf.string, name=sn('gf_y'))
            self.gf_table = []
            self.gf_y = []
            for i in range(len(f.random)):
                self.gf_table.append(tf.contrib.lookup.HashTable(
                    tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(np.asarray(self.rangf_keys[i])),
                                                                tf.constant(np.asarray(self.rangf_values[i]))),
                    self.rangf_n_levels[i] - 1))
                self.gf_table[i].init.run(session=self.sess)
                self.gf_y.append(self.gf_table[i].lookup(self.gf_y_raw[:, i]))
            if len(self.gf_y) > 0:
                self.gf_y = tf.stack(self.gf_y, axis=1)

            self.first_obs = tf.placeholder(shape=[None], dtype=self.INT, name=sn('first_obs'))
            self.last_obs = tf.placeholder(shape=[None], dtype=self.INT, name=sn('last_obs'))

            if f.intercept:
                self.intercept_fixed = tf.Variable(tf.constant(self.intercept_init, shape=[1]), dtype=self.FLOAT,
                                                   name='intercept')
            else:
                self.intercept_fixed = tf.constant(0., dtype=self.FLOAT, name='intercept')
            self.intercept = self.intercept_fixed
            tf.summary.scalar('intercept', self.intercept[0], collections=['params'])

            self.coefficient_fixed = tf.Variable(
                tf.truncated_normal(shape=[1, len(f.coefficients)], mean=0., stddev=0.1, dtype=self.FLOAT),
                name='coefficient_fixed')
            for i in range(len(f.coefficient_names)):
                tf.summary.scalar('coefficient' + '/%s' % f.coefficient_names[i], self.coefficient_fixed[0, i], collections=['params'])
            self.coefficient = self.coefficient_fixed
            fixef_ix = names2ix(f.fixed_coefficient_names, f.coefficient_names)
            coefficient_fixed_mask = np.zeros(len(f.coefficients), dtype=getattr(np, self.float_type))
            coefficient_fixed_mask[fixef_ix] = 1.
            coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)
            self.coefficient_fixed *= coefficient_fixed_mask
            self.coefficient = self.coefficient_fixed
            self.ransl = False
            if self.log_random:
                writers = {}
            for i in range(len(f.ran_names)):
                r = f.random[f.ran_names[i]]
                coefs = r.coefficients
                mask_col_indices = np.zeros((self.rangf_n_levels[i], len(f.coefficient_names)))
                for j in range(len(f.coefficient_names)):
                    if f.coefficient_names[j] in coefs:
                        mask_col_indices[:, j] = 1
                mask_col_indices = tf.constant(mask_col_indices, dtype=self.FLOAT)
                mask = np.ones(self.rangf_n_levels[i], dtype=getattr(np, self.float_type))
                mask[self.rangf_n_levels[i] - 1] = 0
                mask = tf.constant(mask)

                if r.intercept:
                    intercept_random = tf.Variable(
                        tf.truncated_normal(shape=[self.rangf_n_levels[i]], mean=0., stddev=.1, dtype=tf.float32),
                        name='intercept_by_%s' % r.gf)
                    intercept_random *= mask
                    intercept_random -= tf.reduce_mean(intercept_random, axis=0)
                    self.intercept += tf.gather(intercept_random, self.gf_y[:, i])
                    if self.log_random:
                        self.summary_random_writers[r.name] = [tf.summary.FileWriter(self.outdir + '/by_' + r.gf + '/%d' % j) for j in range(min(10, self.rangf_n_levels[i]))]
                        self.summary_random_indexers[r.name] = tf.placeholder(dtype=tf.int32)
                        tf.summary.scalar('by_' + r.gf + '/intercept', intercept_random[self.summary_random_indexers[r.name]], collections=['by_' + r.gf])
                if len(coefs) > 0:
                    self.ransl = True
                    coefficient_random = tf.Variable(
                        tf.truncated_normal(shape=[self.rangf_n_levels[i], len(f.coefficient_names)], mean=0., stddev=.1,
                                            dtype=tf.float32), name='coefficient_by_%s' % (r.gf))
                    coefficient_random *= mask_col_indices
                    coefficient_random *= tf.expand_dims(mask, -1)
                    coefficient_random -= tf.reduce_mean(coefficient_random, axis=0)
                    if self.log_random:
                        coef_names = sorted(coefs.keys())
                        coef_ix = names2ix(coef_names, f.coefficient_names)
                        for k in coef_ix:
                            tf.summary.scalar('by_' + r.gf + '/' + coef_names[k], coefficient_random[self.summary_random_indexers[r.name],k], collections=['by_' + r.gf])

                    self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

            self.irf_names = []
            self.build_irfs(self.irf_tree)

            def convolve_events(time_target, first_obs, last_obs, coefficient):
                inputs = self.X[first_obs:last_obs]
                input_times = self.time_X[first_obs:last_obs]
                t_delta = time_target - input_times

                if not self.ransl:
                    coefficient = self.coefficient[0]

                self.irf_tree.tensor = tf.expand_dims(t_delta, -1)
                out = self.build_convolutional_feedforward(self.irf_tree, inputs, coefficient)

                out = tf.concat(out, axis=1)

                return tf.reduce_sum(out, 0)

            self.X_conv = tf.map_fn(lambda x: convolve_events(*x),
                                    [self.time_y, self.first_obs, self.last_obs, self.coefficient], dtype=self.FLOAT)

            self.out = self.intercept + tf.reduce_sum(self.X_conv, axis=1)
            self.out = self.intercept + tf.reduce_sum(self.X_conv, axis=1)

            self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
            self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
            if self.loss_name.lower() == 'mae':
                self.loss_func = self.mae_loss
            else:
                self.loss_func = self.mse_loss
            self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT, name='loss_total')
            tf.summary.scalar('loss/%s' % self.loss_name, self.loss_total, collections=['loss'])

            self.global_step = tf.Variable(0, name=sn('global_step'), trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
            self.global_batch_step = tf.Variable(0, name=sn('global_batch_step'), trainable=False)
            if self.optim_name != 'LBFGS':
                self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step,
                                                    name=sn('optim'))
                self.gradients = self.optim.compute_gradients(self.loss_func)

                # ## CLIP GRADIENT NORM
                # gradients, variables = zip(*self.optim.compute_gradients(self.loss_func))
                # self.gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                # self.max_grad = tf.reduce_max(tf.stack([tf.reduce_max(g) for g in self.gradients]))
                # self.train_op = self.optim.apply_gradients(zip(gradients, variables), global_step=self.global_batch_step, name=sn('optim'))
            else:
                self.train_op = tf.contrib.opt.ScipyOptimizerInterface(self.loss_func, method='LBFGS',
                                                                       options={'maxiter': 50000})

            self.writer = tf.summary.FileWriter(self.outdir + '/train', self.sess.graph)
            self.summary_params = tf.summary.merge_all(key='params')
            self.summary_losses = tf.summary.merge_all(key='loss')
            if self.log_random:
                for r in f.ran_names:
                    self.summary_random[r] = tf.summary.merge_all(key='by_' + f.random[r].gf)

            self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)

            self.saver = tf.train.Saver()
            self.load()

            n_params = 0
            var_names = [v.name for v in tf.trainable_variables()]
            var_vals = self.sess.run(tf.trainable_variables())
            sys.stderr.write('Trainable variables:\n')
            for i in range(len(var_names)):
                v_name = var_names[i]
                v_val = var_vals[i]
                cur_params = np.prod(np.array(v_val).shape)
                n_params += cur_params
                sys.stderr.write('  ' + v_name.split(':')[0] + ': %s\n' %str(cur_params))
            sys.stderr.write('Network contains %d total trainable parameters.\n' % n_params)
            sys.stderr.write('\n')

    def initialize_irf_lambdas(self):
        epsilon = 1e-35  # np.nextafter(0, 1, dtype=getattr(np, self.float_type)) * 10

        with self.sess.graph.as_default():
            def dirac_delta(params):
                return lambda x: tf.cast(tf.equal(x, 0.), dtype=self.FLOAT)*params
            self.irf_lambdas['DiracDelta'] = dirac_delta

            def exponential(params):
                return lambda x: tf.contrib.distributions.Exponential(rate=params[0]).prob(x + epsilon)
            self.irf_lambdas['Exp'] = exponential

            def shifted_exp(params):
                return lambda x: tf.contrib.distributions.Exponential(rate=params[0]).prob(x - params[1] + epsilon)
            self.irf_lambdas['ShiftedExp'] = shifted_exp

            def gamma(params):
                return lambda x: tf.contrib.distributions.Gamma(concentration=params[0],
                                                      rate=params[1],
                                                      validate_args=False).prob(x + epsilon)
            self.irf_lambdas['Gamma'] = gamma

            self.irf_lambdas['GammaKgt1'] = gamma

            def shifted_gamma(params):
                return lambda x: tf.contrib.distributions.Gamma(concentration=params[0],
                                                      rate=params[1],
                                                      validate_args=False).prob(x - params[2] + epsilon)
            self.irf_lambdas['ShiftedGamma'] = shifted_gamma

            self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma

            def normal(params):
                return lambda x: tf.contrib.distributions.Normal(loc=params[0], scale=params[1]).prob(x)
            self.irf_lambdas['Normal'] = normal

            def skew_normal(params):
                mu = params[0]
                sigma = params[1]
                alpha = params[2]
                stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
                stdnorm_pdf = stdnorm.prob
                stdnorm_cdf = stdnorm.cdf
                return lambda x: 2 / sigma * stdnorm_pdf((x - mu) / sigma) * stdnorm_cdf(alpha * (x - mu) / sigma)
            self.irf_lambdas['SkewNormal'] = skew_normal

            def emg(params):
                mu = params[0]
                sigma = params[1]
                L = params[2]
                return lambda x: L / 2 * tf.exp(0.5 * L * (2. * mu + L * sigma ** 2. - 2. * x)) * tf.erfc(
                    (mu + L * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))
            self.irf_lambdas['EMG'] = emg

            def beta_prime(params):
                alpha = params[0]
                beta = params[1]
                return lambda x: (x+epsilon) ** (alpha - 1.) * (1. + (x+epsilon)) ** (-alpha - beta) / tf.exp(
                    tf.lbeta(tf.transpose(tf.concat([alpha, beta], axis=0))))
            self.irf_lambdas['BetaPrime'] = beta_prime

            def shifted_beta_prime(params):
                alpha = params[0]
                beta = params[1]
                delta = params[3]
                return lambda x: (x-delta+epsilon) ** (alpha - 1) * (1 + (x-delta+epsilon)) ** (-alpha - beta) / tf.exp(
                    tf.lbeta(tf.transpose(tf.concat([alpha, beta], axis=0))))
            self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

    def initialize_irf_params(self, family, ids):
        ## Infinitessimal value to add to bounded parameters
        epsilon = 1e-35  # np.nextafter(0, 1, dtype=getattr(np, self.float_type)) * 10
        dim = len(ids)

        with self.sess.graph.as_default():
            if family == 'DiracDelta':
                filler = tf.constant(1., shape=[1, dim])
                return filler
            if family == 'Exp':
                log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[1, dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                return L
            if family == 'ShiftedExp':
                log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                for i in range(dim):
                    tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                    tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                return tf.stack([L, delta], axis=0)
            if family == 'Gamma':
                log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon
                theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                    tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                return tf.stack([k, theta])
            if family == 'GammaKgt1':
                log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon + 1.
                theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                    tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                return tf.stack([k, theta])
            if family == 'ShiftedGamma':
                log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                k = tf.exp(log_k, name=sn('k')) + epsilon
                theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                for i in range(dim):
                    tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                    tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                return tf.stack([k, theta, delta], axis=0)
            if family == 'ShiftedGammaKgt1':
                log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                k = tf.nn.softplus(log_k, name=sn('k_%s' % '-'.join(ids))) + 1. + epsilon
                theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                for i in range(dim):
                    tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                    tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                return tf.stack([k, theta, delta], axis=0)
            if family == 'Normal':
                log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids)))
                for i in range(dim):
                    tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                    tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                return tf.stack([mu, sigma], axis=0)
            elif family == 'SkewNormal':
                log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                alpha = tf.get_variable(sn('alpha_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                    tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                    tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                return tf.stack([mu, sigma, alpha], axis=1)
            elif family == 'EMG':
                log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                    tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                    tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                return tf.stack([mu, sigma, L], axis=0)
            elif family == 'BetaPrime':
                log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                for i in range(dim):
                    tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                    tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                return tf.stack([alpha, beta], axis=0)
            elif family == 'ShiftedBetaPrime':
                log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim], initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                for i in range(dim):
                    tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                    tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                    tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                return tf.stack([alpha, beta, delta], axis=0)
            else:
                raise ValueError('Impulse response function "%s" is not currently supported.' % family)

    def build_irf_params(self):
        f = self.form
        with self.sess.graph.as_default():
            for x in f.atomic_irf_by_family:
                self.atomic_irf_by_family[x] = self.initialize_irf_params(x, f.atomic_irf_by_family[x])

    def build_irfs(self, t):
        with self.sess.graph.as_default():
            t.irfs = {}
            t.params = {}
            for f in t.children:
                child_nodes = sorted(t.children[f].keys())
                child_irfs = [t.children[f][x].irf_id for x in child_nodes]
                params_ix = names2ix(child_irfs, self.form.atomic_irf_by_family[f])
                t.params[f] = tf.gather(self.atomic_irf_by_family[f], params_ix, axis=1)
                def main_irf(irf, params):
                    def new_irf(x):
                        return irf(params)(x)
                    return new_irf
                def child_irf(irf_lambda, params):
                    def new_irf(x):
                        return irf_lambda(params)(x)
                    return new_irf
                composite = False
                parent_irf = getattr(t, 'composite_irf', None)
                if parent_irf is not None:
                    composite = True
                    def child_composite_irf(irf_lambda, parent_irf, params):
                        def new_irf(x):
                            return irf_lambda(params)(parent_irf(x))
                        return new_irf
                t.irfs[f] = main_irf(self.irf_lambdas[f], t.params[f])
                for i in range(len(child_nodes)):
                    child = t.children[f][child_nodes[i]]
                    child.atomic_irf = child_irf(self.irf_lambdas[f], t.params[f][:,i])
                    self.atomic_irfs[child.name] = child.atomic_irf
                    if composite:
                        child.composite_irf = child_composite_irf(self.irf_lambdas[f], parent_irf, t.params[f][:,i])
                    else:
                        child.composite_irf = child.atomic_irf
                    self.composite_irfs[child.name] = child.composite_irf
                    if len(child.children) > 0:
                        self.build_irfs(child)
                    self.irf_names.append(child.name)

    def build_convolutional_feedforward(self, t, inputs, coef):
        with self.sess.graph.as_default():
            out = []
            for f in t.children:
                preterminals = []
                terminals = []
                child_nodes = sorted(t.children[f].keys())
                child_coefs = [t.children[f][x].coef_id for x in child_nodes]
                coefs_ix = names2ix(child_coefs, self.form.coefficient_names)
                tensor = t.irfs[f](t.tensor) * tf.gather(coef, coefs_ix)
                for i in range(len(child_nodes)):
                    x = t.children[f][child_nodes[i]]
                    if x.terminal is not None:
                        preterminals.append(x.name)
                        terminals.append(x.terminal)
                    x.tensor = tf.expand_dims(tensor[:, i], -1)
                    if len(x.children) > 0:
                        out += self.build_convolutional_feedforward(x, inputs, coef)
                if len(preterminals) > 0 and len(terminals) > 0:
                    preterminals_ix = names2ix(preterminals, child_nodes)
                    terminals_ix = names2ix(terminals, self.form.terminal_names)
                    out.append(tf.gather(inputs, terminals_ix, axis=1) * tf.gather(tensor, preterminals_ix, axis=1))
            return out

    def apply_op(self, op, input):
        if op in ['c', 'c.']:
            out = input - tf.reduce_mean(input, axis=0)
        elif op in ['z', 'z.']:
            out = (input - tf.reduce_mean(input, axis=0)) / reduce_std(input, axis=0)
        elif op in ['s', 's.']:
            out = input / reduce_std(input, axis=0)
        elif op == 'log':
            out = tf.log(input)
        elif op == 'log1p':
            out = tf.log(input + 1)
        else:
            raise ValueError('DTSR graph op "%s" not recognized.' % op)
        return out

    def train(self,
              X,
              y,
              n_epoch_train=100,
              n_epoch_tune=100,
              minibatch_size=128,
              irf_name_map=None,
              plot_x_inches=7,
              plot_y_inches=5,
              cmap='gist_earth'):

        usingGPU = is_gpu_available()

        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        f = self.form

        with self.sess.graph.as_default():
            y_rangf = y[f.rangf]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            y_range = np.arange(len(y))

            if self.optim_name == 'LBFGS':
                fd = {}
                fd[self.X] = X[f.terminal_names]
                fd[self.y] = y[f.dv]
                fd[self.time_X] = X.time
                fd[self.time_y] = y.time
                fd[self.gf_y_raw] = y_rangf
                fd[self.first_obs] = y.first_obs
                fd[self.last_obs] = y.last_obs

                def step_callback(x):
                    sys.stderr.write('\rCurrent loss: %s' % x[-1])

                self.train_op.minimize(session=self.sess,
                                       feed_dict=fd,
                                       fetches=[self.loss_func],
                                       loss_callback=lambda x: '\rCurrent loss: %s' % x)

                self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)

                self.save()

            else:
                fd_minibatch = {}
                fd_minibatch[self.X] = X[f.terminal_names]
                fd_minibatch[self.time_X] = X.time

                fd = {}
                fd[self.X] = fd_minibatch[self.X]
                fd[self.time_X] = fd_minibatch[self.time_X]
                fd[self.y] = y[f.dv]
                fd[self.time_y] = y.time
                fd[self.gf_y_raw] = y_rangf
                fd[self.first_obs] = y.first_obs
                fd[self.last_obs] = y.last_obs

                if self.global_step.eval(session=self.sess) == 0:
                    summary_params, loss_total = self.sess.run(
                        [self.summary_params, self.loss_func],
                        feed_dict=fd)
                    summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                if self.log_random:
                    for i in range(len(f.ran_names)):
                        r = f.ran_names[i]
                        for j in range(min(10, self.rangf_n_levels[i])):
                            summary_by_subject_batch = self.sess.run(self.summary_random[r],
                                                                     feed_dict={self.summary_random_indexers[r]: j})
                            self.summary_random_writers[r][j].add_summary(summary_by_subject_batch,
                                                                          self.global_batch_step.eval(
                                                                              session=self.sess))

                while self.global_step.eval(session=self.sess) < n_epoch_train + n_epoch_tune:
                    if self.global_step.eval(session=self.sess) < n_epoch_train:
                        p, p_inv = getRandomPermutation(len(y))
                    else:
                        minibatch_size = len(y)
                        p = y_range
                    n_minibatch = math.ceil(float(len(y)) / minibatch_size)

                    t0_iter = time.time()
                    sys.stderr.write('-' * 50 + '\n')
                    sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                    sys.stderr.write('\n')

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0

                    for j in range(0, len(y), minibatch_size):
                        fd_minibatch[self.y] = y[f.dv].iloc[p[j:j + minibatch_size]]
                        fd_minibatch[self.time_y] = y.time.iloc[p[j:j + minibatch_size]]
                        fd_minibatch[self.gf_y_raw] = y_rangf.iloc[p[j:j + minibatch_size]]
                        fd_minibatch[self.first_obs] = y.first_obs.iloc[p[j:j + minibatch_size]]
                        fd_minibatch[self.last_obs] = y.last_obs.iloc[p[j:j + minibatch_size]]

                        _, loss_minibatch = self.sess.run(
                            [self.train_op, self.loss_func],
                            feed_dict=fd_minibatch)
                        loss_total += loss_minibatch
                        pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

                    loss_total /= n_minibatch
                    fd[self.loss_total] = loss_total
                    if (self.global_step.eval(session=self.sess) + 1) % 10 == 0:
                        sys.stderr.write('Evaluating gradients...\n')
                        gradients = self.sess.run(self.gradients, feed_dict=fd)
                        max_grad = -inf
                        for g in gradients:
                            max_grad = max(max_grad, np.max(g[0]))
                        sys.stderr.write('  max(grad) = %s\n' % str(max_grad))
                        sys.stderr.write('  Converged (max(grad) < 0.001) = %s\n' % (max_grad < 0.001))

                    self.sess.run(self.incr_global_step)

                    summary_params, summary_train_loss = self.sess.run(
                        [self.summary_params, self.summary_losses],
                        feed_dict=fd)
                    self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))

                    if self.log_random:
                        for i in range(len(f.ran_names)):
                            r = f.ran_names[i]
                            for j in range(min(10, self.rangf_n_levels[i])):
                                summary_by_subject_batch = self.sess.run(self.summary_random[r], feed_dict={self.summary_random_indexers[r]: j})
                                self.summary_random_writers[r][j].add_summary(summary_by_subject_batch, self.global_step.eval(session=self.sess))

                    self.save()

                    self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)

                    # X_conv = pd.DataFrame(self.sess.run(self.X_conv, feed_dict=fd), columns=sorted(self.irf.keys()))
                    # print('Mean values of convolved predictors')
                    # print(X_conv.mean(axis=0))
                    # print('Correlations of convolved predictors')
                    # print(X_conv.corr())

                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

            fd = {}
            fd[self.X] = X[f.terminal_names]
            fd[self.y] = y[f.dv]
            fd[self.time_X] = X.time
            fd[self.time_y] = y.time
            fd[self.gf_y_raw] = y_rangf
            fd[self.first_obs] = y.first_obs
            fd[self.last_obs] = y.last_obs

            X_conv = pd.DataFrame(self.sess.run(self.X_conv, feed_dict=fd), columns=sorted(self.irf_lambdas.keys()))

            self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)

            sys.stderr.write('Mean values of convolved predictors\n')
            sys.stderr.write(str(X_conv.mean(axis=0)) + '\n')
            sys.stderr.write('Correlations of convolved predictors')
            sys.stderr.write(str(X_conv.corr()) + '\n')

            sys.stderr.write('\n')

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        f = self.form

        with self.sess.graph.as_default():
            y_rangf = y_rangf[y_rangf.columns]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            fd = {}
            fd[self.X] = X[f.terminal_names]
            fd[self.time_X] = X.time
            fd[self.time_y] = y_time
            fd[self.gf_y_raw] = y_rangf
            fd[self.first_obs] = first_obs
            fd[self.last_obs] = last_obs

            return self.sess.run(self.out, feed_dict=fd)

    def eval(self, X, y):
        f = self.form

        with self.sess.graph.as_default():
            y_rangf = y[f.rangf]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            fd = {}
            fd[self.X] = X[f.terminal_names]
            fd[self.time_X] = X.time
            fd[self.y] = y[f.dv]
            fd[self.time_y] = y.time
            fd[self.gf_y_raw] = y_rangf
            fd[self.first_obs] = y.first_obs
            fd[self.last_obs] = y.last_obs

            return self.sess.run(self.loss_func, feed_dict=fd)

    def optim_init(self, name, learning_rate):
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

    def save(self):
        with self.sess.graph.as_default():
            self.saver.save(self.sess, self.outdir + '/model.ckpt')

    def load(self):
        with self.sess.graph.as_default():
            if os.path.exists(self.outdir + '/checkpoint'):
                self.saver.restore(self.sess, self.outdir + '/model.ckpt')
            else:
                self.sess.run(tf.global_variables_initializer())

    def make_plots(self, irf_name_map, plot_x_inches, plot_y_inches, cmap):
        with self.sess.graph.as_default():
            plot_x = self.support.eval(session=self.sess)

            plot_y = []
            for x in self.irf_names:
                plot_y.append(self.atomic_irfs[x](self.support).eval(session=self.sess))
            plot_y = np.concatenate(plot_y, axis=1)

            plot_convolutions(plot_x,
                              plot_y,
                              self.irf_names,
                              dir=self.outdir,
                              filename='irf_atomic_unscaled.jpg',
                              irf_name_map=irf_name_map,
                              plot_x_inches=plot_x_inches,
                              plot_y_inches=plot_y_inches,
                              cmap=cmap)

            plot_y = []
            for x in self.irf_names:
                coef_ix = self.form.coefficient_names.index(x)
                plot_y.append(
                    (self.atomic_irfs[x](self.support) * self.coefficient_fixed[0, coef_ix]).eval(session=self.sess))
            plot_y = np.concatenate(plot_y, axis=1)

            plot_convolutions(plot_x,
                              plot_y,
                              self.irf_names,
                              dir=self.outdir,
                              filename='irf_atomic_scaled.jpg',
                              irf_name_map=irf_name_map,
                              plot_x_inches=plot_x_inches,
                              plot_y_inches=plot_y_inches,
                              cmap=cmap)

            # plot_y = []
            # for x in self.irf_names:
            #     plot_y.append(self.composite_irfs[x](self.support).eval(session=self.sess))
            # plot_y = np.concatenate(plot_y, axis=1)
            #
            # plot_convolutions(plot_x,
            #                   plot_y,
            #                   self.irf_names,
            #                   dir=self.outdir,
            #                   filename='irf_composite_unscaled.jpg',
            #                   irf_name_map=irf_name_map,
            #                   plot_x_inches=plot_x_inches,
            #                   plot_y_inches=plot_y_inches,
            #                   cmap=cmap)
            #
            # plot_y = []
            # for x in self.irf_names:
            #     coef_ix = self.form.coefficient_names.index(x)
            #     plot_y.append((self.composite_irfs[x](self.support) * self.coefficient_fixed[0, coef_ix]).eval(
            #         session=self.sess))
            # plot_y = np.concatenate(plot_y, axis=1)
            #
            # plot_convolutions(plot_x,
            #                   plot_y,
            #                   self.irf_names,
            #                   dir=self.outdir,
            #                   filename='irf_composite_scaled.jpg',
            #                   irf_name_map=irf_name_map,
            #                   plot_x_inches=plot_x_inches,
            #                   plot_y_inches=plot_y_inches,
            #                   cmap=cmap)

    def __getstate__(self):
        return (
            self.form_str,
            self.outdir,
            self.irf_str,
            self.optim_name,
            self.learning_rate,
            self.loss_name,
            self.log_convolution_plots,
            self.rangf_keys,
            self.rangf_values,
            self.rangf_n_levels,
            self.intercept_init,
            self.float_type,
            self.FLOAT,
            self.int_type,
            self.INT,
            self.log_random
        )

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)
        self.form_str, \
        self.outdir, \
        self.irf_str, \
        self.optim_name, \
        self.learning_rate, \
        self.loss_name, \
        self.log_convolution_plots, \
        self.rangf_keys, \
        self.rangf_values, \
        self.rangf_n_levels, \
        self.intercept_init, \
        self.float_type, \
        self.FLOAT, \
        self.int_type, \
        self.INT,\
        self.log_random = state

        self.form = Formula(self.form_str)
        self.irf_tree = self.form.irf_tree
        self.preterminals = []

        self.construct_network()
        self.load()
