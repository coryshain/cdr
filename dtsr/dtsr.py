import sys
import os
import math
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.platform.test import is_gpu_available
usingGPU = is_gpu_available()
sys.stderr.write('Using GPU: %s\n' % usingGPU)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
from .formula import *
from .util import *
from .plot import *

def compute_history_intervals(y, X, series_ids, cutoff=100):
    pb = tf.contrib.keras.utils.Progbar(len(y))

    id_vectors_X = np.zeros((len(X), len(series_ids))).astype('int32')
    id_vectors_y = np.zeros((len(y), len(series_ids))).astype('int32')

    time_X = np.array(X.time)
    time_y = np.array(y.time)

    for i in range(len(series_ids)):
        col = series_ids[i]
        id_vectors_X[:, i] = np.array(X[col].cat.codes)
        id_vectors_y[:, i] = np.array(y[col].cat.codes)
    cur_ids = id_vectors_y[0]

    first_obs = np.zeros(len(y)).astype('int32')
    last_obs = np.zeros(len(y)).astype('int32')

    i = j = 0
    start = 0
    end = 0
    while i < len(y) and j < len(X):
        if (id_vectors_y[i] != cur_ids).any():
            start = end = j
            cur_ids = id_vectors_y[i]
        while j < len(X) and not (id_vectors_X[j] == cur_ids).all():
            start += 1
            end += 1
            j += 1
        while j < len(X) and time_X[j] <= time_y[i] and (id_vectors_X[j] == cur_ids).all():
            end += 1
            j += 1
        first_obs[i] = start
        last_obs[i] = end
        pb.update(i, force=True)

        i += 1
    return first_obs, last_obs

class DTSR(object):
    """Deconvolutional Time Series Regression (DTSR) class
    
    # Arguments
        bform: String or Formula. An R-style linear mixed-effects model formula string or an instance of the Formula class
        conv_func: String. One of `"exp"` or `"gamma"` (other functions coming soon...)
        optim: String. Optimizer. One of `"Adam"`.
        learning_rate: Float. Defaults to 0.01
        
    """

    def __init__(self,
                 form,
                 X,
                 y,
                 outdir,
                 conv_func='gamma',
                 optim='Adam',
                 learning_rate=0.01,
                 loss='mse',
                 log_convolution_plots=False
                 ):

        self.sess = tf.Session(config=tf_config)

        if isinstance(form, Formula):
            self.form = form
        elif isinstance(form, str):
            self.form = Formula(form)
        else:
            raise ValueError('Invalid type for argument bform. Must be str or Formula.')
        f = self.form

        self.outdir = outdir
        self.conv_func_str = conv_func
        self.optim_name = optim
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.log_convolution_plots = log_convolution_plots
        self.rangf_unique = []
        self.n_levels = []
        for i in range(len(f.rangf)):
            self.rangf_unique.append(y[f.rangf[i]].unique())
            self.n_levels.append(len(self.rangf_unique[i]) + 1)
        self.intercept_init = float(y[f.dv].mean())

        self.construct_network()

    def construct_network(self):
        f = self.form
        with self.sess.graph.as_default():
            self.optim = self.optim_init(self.optim_name, self.learning_rate)

            self.X_iv = tf.placeholder(shape=[None, len(f.allsl)], dtype=tf.float32, name='X_raw')
            self.time_X = tf.placeholder(shape=[None], dtype=tf.float32, name='time_X')

            self.y = tf.placeholder(shape=[None], dtype=tf.float32, name='y_')
            self.time_y = tf.placeholder(shape=[None], dtype=tf.float32, name='time_y')

            ## Build random effects lookup tensor
            ## Using strings for indexing guarantees correct random effects behavior on unseen data
            self.gf_y_raw = tf.placeholder(shape=[None, len(f.rangf)], dtype=tf.string, name='y_gf')
            self.gf_table = []
            self.gf_y = []
            for i in range(len(f.rangf)):
                unique = self.rangf_unique[i]
                keys = tf.constant(np.asarray(unique, dtype=np.str))
                values = tf.constant(np.arange(len(unique), dtype=np.int32))
                self.gf_table.append(tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), self.n_levels[i]-1))
                self.gf_table[i].init.run(session=self.sess)
                self.gf_y.append(self.gf_table[i].lookup(self.gf_y_raw[:,i]))
            self.gf_y = tf.stack(self.gf_y, axis=1)

            self.first_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='first_obs')
            self.last_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='last_obs')

            if self.conv_func_str == 'exp':
                self.L_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.allsl)], mean=0., stddev=.1, dtype=tf.float32),
                                       name='lambda')
            elif self.conv_func_str == 'gamma':
                self.log_k_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_k')
                self.log_theta_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_theta')
                self.log_neg_delta_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_neg_delta')

            intercept_global = tf.Variable(tf.constant(self.intercept_init, shape=[1]), name='intercept')
            self.out = intercept_global

            ## Exponential convolution function
            if self.conv_func_str == 'exp':
                self.conv_global = lambda x: tf.exp(self.L_global) * tf.exp(-tf.exp(self.L_global) * x)
                conv = self.conv_global
            ## Gamma convolution function
            elif self.conv_func_str == 'gamma':
                conv_global_delta_zero = tf.contrib.distributions.Gamma(concentration=tf.exp(self.log_k_global[0]),
                                                                        rate=tf.exp(self.log_theta_global[0]),
                                                                        validate_args=True).prob
                self.conv_global = lambda x: conv_global_delta_zero(x + tf.exp(self.log_neg_delta_global)[0])
                conv_delta_zero = conv_global_delta_zero
                conv = self.conv_global

            def convolve_events(time_target, first_obs, last_obs):
                col_ix = names2ix(f.allsl, f.allsl)
                input_rows = tf.gather(self.X_iv[first_obs:last_obs], col_ix, axis=1)
                input_times = self.time_X[first_obs:last_obs]
                t_delta = time_target - input_times
                conv_coef = conv(tf.expand_dims(t_delta, -1))

                return tf.reduce_sum(conv_coef * input_rows, 0)

            self.X_conv = tf.map_fn(lambda x: convolve_events(*x), [self.time_y, self.first_obs, self.last_obs], dtype=tf.float32)

            self.beta_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.fixed)], mean=0., stddev=0.1, dtype=tf.float32), name='beta')
            fixef_ix = names2ix(f.fixed, f.allsl)
            fixef_cols = tf.gather(self.X_conv, fixef_ix, axis=1)
            if len(fixef_cols.shape) == 1:
                fixef_cols = tf.expand_dims(fixef_cols, -1)
            self.out += tf.squeeze(tf.matmul(fixef_cols, tf.transpose(self.beta_global)), axis=-1)
            self.zeta = []
            self.Z = []
            for i in range(len(f.rangf)):
                r = f.random[i]
                vars = r.vars
                n_var = len(vars)
                mask = np.ones(self.n_levels[i], dtype=np.float32)
                mask[self.n_levels[i]-1] = 0
                mask = tf.constant(mask)

                if r.intercept:
                    random_intercept = tf.Variable(tf.truncated_normal(shape=[self.n_levels[i]], mean=0., stddev=.1, dtype=tf.float32), name='intercept_by_%s' % r.grouping_factor)
                    random_intercept *= mask
                    random_intercept -= tf.reduce_mean(random_intercept, axis=0)
                    self.zeta.append(random_intercept)
                    self.out += tf.gather(random_intercept, self.gf_y[:, i])

                if len(vars) > 0:
                    random_effects_matrix = tf.Variable(tf.truncated_normal(shape=[self.n_levels[i], n_var], mean=0., stddev=.1, dtype=tf.float32), name='random_slopes_by_%s' % (r.grouping_factor))
                    random_effects_matrix *= tf.expand_dims(mask, -1)
                    random_effects_matrix -= tf.reduce_mean(random_effects_matrix, axis=0)
                    self.Z.append(random_effects_matrix)

                    ransl_ix = names2ix(vars, f.allsl)

                    self.out += tf.reduce_sum(tf.gather(self.X_conv, ransl_ix, axis=1) * tf.gather(random_effects_matrix, tf.clip_by_value(self.gf_y[:, i], 0, self.n_levels[i] - 1)), axis=1)

            self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
            self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
            if self.loss_name.lower() == 'mae':
                self.loss_func = self.mae_loss
            else:
                self.loss_func = self.mse_loss

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
            self.global_batch_step = tf.Variable(0, name='global_batch_step', trainable=False)
            self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step, name='optim')

            self.writer = tf.summary.FileWriter(self.outdir + '/train')
            tf.summary.scalar('intercept', intercept_global[0], collections=['params'])
            self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)
            for i in range(len(f.fixed)):
                tf.summary.scalar('beta/%s' % f.fixed[i], self.beta_global[0, i], collections=['params'])
                if self.conv_func_str == 'exp':
                    tf.summary.scalar('log_lambda/%s' % f.fixed[i], self.L_global[0, i], collections=['params'])
                elif self.conv_func_str == 'gamma':
                    tf.summary.scalar('log_k/%s' % f.fixed[i], self.log_k_global[0, i], collections=['params'])
                    tf.summary.scalar('log_theta/%s' % f.fixed[i], self.log_theta_global[0, i], collections=['params'])
                    tf.summary.scalar('log_neg_delta/%s' % f.fixed[i], self.log_neg_delta_global[0, i], collections=['params'])
                if self.log_convolution_plots:
                    tf.summary.histogram('conv/%s' % f.fixed[i], self.conv_global(self.support)[:, i], collections=['params'])

            tf.summary.scalar('loss/%s' % self.loss_name, self.loss_func, collections=['loss'])
            self.summary_params = tf.summary.merge_all(key='params')
            self.summary_losses = tf.summary.merge_all(key='loss')

            self.saver = tf.train.Saver()
            self.load()

            n_params = 0
            for v in self.sess.run(tf.trainable_variables()):
                n_params += np.prod(np.array(v).shape)
            sys.stderr.write('Network contains %d trainable parameters\n' % n_params)

    def train(self, X, y, n_epoch_train=100, n_epoch_tune=100, minibatch_size=128, fixef_name_map=None):

            f = self.form

            y_rangf = y[f.rangf]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            y_range = np.arange(len(y))

            fd_minibatch = {}
            fd_minibatch[self.X_iv] = X[f.allsl]
            fd_minibatch[self.time_X] = X.time

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

                for j in range(0, len(y), minibatch_size):
                    fd_minibatch[self.y] = y[f.dv].iloc[p[j:j + minibatch_size]]
                    fd_minibatch[self.time_y] = y.time.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[self.gf_y_raw] = y_rangf.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[self.first_obs] = y.first_obs.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[self.last_obs] = y.last_obs.iloc[p[j:j + minibatch_size]]

                    summary_params_batch, summary_train_losses_batch, _, loss_minibatch = self.sess.run([self.summary_params, self.summary_losses, self.train_op, self.loss_func], feed_dict=fd_minibatch)
                    self.writer.add_summary(summary_params_batch, self.global_batch_step.eval(session=self.sess))
                    self.writer.add_summary(summary_train_losses_batch, self.global_batch_step.eval(session=self.sess))

                    pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

                self.sess.run(self.incr_global_step)

                plot_x = self.support.eval(session=self.sess)
                plot_y = (self.beta_global * self.conv_global(self.support)).eval(session=self.sess)

                plot_convolutions(plot_x, plot_y, f.fixed, dir=self.outdir, filename='convolution_plot.jpg', fixef_name_map=fixef_name_map)

                self.save()
                t1_iter = time.time()
                sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

            sys.stderr.write('\n')

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        f = self.form

        y_rangf = y_rangf[y_rangf.columns]
        for c in f.rangf:
            y_rangf[c] = y_rangf[c].astype(str)

        fd = {}
        fd[self.X_iv] = X[f.allsl]
        fd[self.time_X] = X.time
        fd[self.time_y] = y_time
        fd[self.gf_y_raw] = y_rangf
        fd[self.first_obs] = first_obs
        fd[self.last_obs] = last_obs

        return self.sess.run(self.out, feed_dict=fd)

    def eval(self, X, y):
        f = self.form

        y_rangf = y[f.rangf]
        for c in f.rangf:
            y_rangf[c] = y_rangf[c].astype(str)

        fd = {}
        fd[self.X_iv] = X[f.allsl]
        fd[self.time_X] = X.time
        fd[self.y] = y[f.dv]
        fd[self.time_y] = y.time
        fd[self.gf_y_raw] = y_rangf
        fd[self.first_obs] = y.first_obs
        fd[self.last_obs] = y.last_obs

        return self.sess.run(self.loss_func, feed_dict=fd)

    def optim_init(self, name, learning_rate):
        return {
            'Adagrad': lambda x: tf.train.AdagradOptimizer(x),
            'Adadelta': lambda x: tf.train.AdadeltaOptimizer(x),
            'Adam': lambda x: tf.train.AdamOptimizer(x),
            'FTRL': lambda x: tf.train.FtrlOptimizer(x),
            'RMSProp': lambda x: tf.train.RMSPropOptimizer(x),
            'Nadam': lambda x: tf.contrib.opt.NadamOptimizer()
        }[name](learning_rate)

    def save(self):
        self.saver.save(self.sess, self.outdir + '/model.ckpt')

    def load(self):
        if os.path.exists(self.outdir + '/checkpoint'):
            self.saver.restore(self.sess, self.outdir + '/model.ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())

    def __getstate__(self):
        return (
            self.form,
            self.outdir,
            self.conv_func_str,
            self.optim_name,
            self.learning_rate,
            self.loss_name,
            self.log_convolution_plots,
            self.rangf_unique,
            self.n_levels,
            self.intercept_init
        )

    def __setstate__(self, state):
        self.sess = tf.Session(config=tf_config)
        self.form, \
        self.outdir, \
        self.conv_func_str, \
        self.optim_name, \
        self.learning_rate, \
        self.loss_name, \
        self.log_convolution_plots, \
        self.rangf_unique, \
        self.n_levels, \
        self.intercept_init = state

        self.construct_network()
        self.load()
