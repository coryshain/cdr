import sys
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

class DTSR(object):
    """Deconvolutional Time Series Regression (DTSR) class
    
    # Arguments
        bform: String or Formula. An R-style linear mixed-effects model formula string or an instance of the Formula class
        irf: String. One of `"exp"` or `"gamma"` (other functions coming soon...)
        optim: String. Optimizer. One of `"Adam"`.
        learning_rate: Float. Defaults to 0.01
        
    """

    def __init__(self,
                 form,
                 y,
                 outdir,
                 irf='gamma',
                 optim='Adam',
                 learning_rate=0.01,
                 loss='mse',
                 log_convolution_plots=False,
                 float_type='float32',
                 int_type='int32'
                 ):

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        if isinstance(form, Formula):
            self.form = form
        elif isinstance(form, str):
            self.form = Formula(form)
        else:
            raise ValueError('Invalid type for argument bform. Must be str or Formula.')
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
        for i in range(len(f.rangf)):
            self.rangf_keys.append(y[f.rangf[i]].unique().astype('str'))
            self.rangf_values.append(np.arange(len(self.rangf_keys[i]), dtype=np.int32))
            self.rangf_n_levels.append(len(self.rangf_keys[i]) + 1)
        self.intercept_init = float(y[f.dv].mean())
        self.float_type = float_type
        self.FLOAT = getattr(tf, self.float_type)
        self.int_type = int_type
        self.INT = getattr(tf, self.int_type)

        self.construct_network()

    def construct_network(self):
        f = self.form

        ## Infinitessimal value (smallest positive permitted by float type) to add to bounded parameters
        epsilon = np.nextafter(0, 1, dtype=getattr(np, self.float_type))

        with self.sess.graph.as_default():
            if self.optim_name != 'LBFGS':
                self.optim = self.optim_init(self.optim_name, self.learning_rate)

            self.X = tf.placeholder(shape=[None, len(f.allsl)], dtype=self.FLOAT, name='X_raw')
            self.time_X = tf.placeholder(shape=[None], dtype=self.FLOAT, name='time_X')

            self.y = tf.placeholder(shape=[None], dtype=self.FLOAT, name='y')
            self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT, name='time_y')

            ## Build random effects lookup tensor
            ## Using strings for indexing guarantees correct random effects behavior on unseen data
            self.gf_y_raw = tf.placeholder(shape=[None, len(f.rangf)], dtype=tf.string, name='gf_y')
            self.gf_table = []
            self.gf_y = []
            for i in range(len(f.rangf)):
                self.gf_table.append(tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(np.asarray(self.rangf_keys[i])), tf.constant(np.asarray(self.rangf_values[i]))), self.rangf_n_levels[i] - 1))
                self.gf_table[i].init.run(session=self.sess)
                self.gf_y.append(self.gf_table[i].lookup(self.gf_y_raw[:,i]))
            if len(self.gf_y) > 0:
                self.gf_y = tf.stack(self.gf_y, axis=1)

            self.first_obs = tf.placeholder(shape=[None], dtype=self.INT, name='first_obs')
            self.last_obs = tf.placeholder(shape=[None], dtype=self.INT, name='last_obs')

            intercept_global = tf.Variable(tf.constant(self.intercept_init, shape=[1]), dtype=self.FLOAT, name='intercept')
            self.out = intercept_global

            dim = len(f.allsl)
            if self.irf_str == 'exp':
                self.conv_param_names = ['L']
                log_L = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_L')
                self.L = tf.exp(log_L, name='L') + epsilon
                irf = tf.contrib.distributions.Exponential(rate=self.L).prob
                self.irf_global = lambda x: irf(x + epsilon)
            elif self.irf_str == 'shifted_exp':
                self.conv_param_names = ['L', 'delta']
                log_L = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_L')
                log_neg_delta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_neg_delta')
                self.L = tf.exp(log_L, name='L') + epsilon
                self.delta = -tf.exp(log_neg_delta, name='delta')
                irf = tf.contrib.distributions.Exponential(rate=self.L).prob
                self.irf_global = lambda x: irf(x - self.delta + epsilon)
            elif self.irf_str == 'gamma':
                self.conv_param_names = ['k', 'theta']
                log_k = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_k')
                log_theta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_theta')
                self.k = tf.exp(log_k, name='k') + epsilon
                self.theta = tf.exp(log_theta, name='theta') + epsilon
                irf = tf.contrib.distributions.Gamma(concentration=self.k,
                                                                        rate=self.theta,
                                                                        validate_args=False).prob
                self.irf_global = lambda x: irf(x + epsilon)
            elif self.irf_str == 'gamma_kgt1':
                self.conv_param_names = ['k', 'theta']
                log_k = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_k')
                log_theta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_theta')
                self.k = tf.exp(log_k, name='k') + epsilon + 1.
                self.theta = tf.exp(log_theta, name='theta') + epsilon
                irf = tf.contrib.distributions.Gamma(concentration=self.k,
                                                                        rate=self.theta,
                                                                        validate_args=False).prob
                self.irf_global = lambda x: irf(x + epsilon)
            elif self.irf_str == 'shifted_gamma':
                self.conv_param_names = ['k', 'theta', 'delta']
                log_k = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_k')
                log_theta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_theta')
                log_neg_delta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_neg_delta')
                self.k = tf.exp(log_k, name='k') + epsilon
                self.theta = tf.exp(log_theta, name='theta') + epsilon
                self.delta = -tf.exp(log_neg_delta, name='delta')
                irf = tf.contrib.distributions.Gamma(concentration=self.k,
                                                                        rate=self.theta,
                                                                        validate_args=False).prob
                self.irf_global = lambda x: irf(x - self.delta + epsilon)
            elif self.irf_str == 'shifted_gamma_kgt1':
                self.conv_param_names = ['k', 'theta', 'delta']
                log_k = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_k')
                log_theta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_theta')
                log_neg_delta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_neg_delta')
                self.k = tf.exp(log_k, name='k') + epsilon + 1.
                self.theta = tf.exp(log_theta, name='theta') + epsilon
                self.delta = -tf.exp(log_neg_delta, name='delta')
                irf = tf.contrib.distributions.Gamma(concentration=self.k,
                                                                        rate=self.theta,
                                                                        validate_args=False).prob
                self.irf_global = lambda x: irf(x - self.delta + epsilon)
            elif self.irf_str == 'normal':
                self.conv_param_names = ['mu', 'sigma']
                log_sigma = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_sigma'))
                self.mu = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='mu'))
                self.sigma = tf.exp(log_sigma, name='sigma')
                irf = tf.contrib.distributions.Normal(loc=self.mu,
                                                              scale=self.sigma).prob
                self.irf_global = lambda x: irf(x)
            elif self.irf_str == 'skew_normal':
                self.conv_param_names = ['mu', 'sigma', 'alpha']
                log_sigma = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_sigma'))
                self.mu = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='mu'))
                self.sigma = tf.exp(log_sigma, name='sigma') + epsilon
                self.alpha = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='alpha'))
                stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
                stdnorm_pdf = stdnorm.prob
                stdnorm_cdf = stdnorm.cdf
                Z = lambda x: (x-self.mu)/self.sigma
                irf = lambda x: 2/self.sigma * stdnorm_pdf(Z(x)) * stdnorm_cdf(self.alpha * Z(x))
                self.irf_global = lambda x: irf(x)
            elif self.irf_str == 'emg':
                self.conv_param_names = ['mu', 'sigma', 'L']
                log_sigma = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_sigma'))
                log_L = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_L'))
                self.mu = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='mu'))
                self.sigma = tf.exp(log_sigma, name='sigma') + epsilon
                self.L = tf.exp(log_L, name='L') + epsilon
                irf = lambda x: self.L/2 * tf.exp(0.5*self.L*(2.*self.mu + self.L*self.sigma**2. - 2.*x)) * tf.erfc((self.mu + self.L*self.sigma**2 - x)/(tf.sqrt(2.)*self.sigma))
                self.irf_global = lambda x: irf(x)
            elif self.irf_str == 'beta_prime':
                self.conv_param_names = ['alpha', 'beta']
                log_alpha = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_alpha'))
                log_beta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_beta'))
                self.alpha = tf.exp(log_alpha, name='alpha') + epsilon
                self.beta = tf.exp(log_beta, name='beta') + epsilon
                irf = lambda x: x**(self.alpha-1.)*(1.+x)**(-self.alpha - self.beta) / tf.exp(tf.lbeta(tf.transpose(tf.concat([self.alpha, self.beta], axis=0))))
                self.irf_global = lambda x: irf(x + epsilon)
            elif self.irf_str == 'shifted_beta_prime':
                self.conv_param_names = ['alpha', 'beta', 'delta']
                log_alpha = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_alpha'))
                log_beta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT, name='log_beta'))
                log_neg_delta = tf.Variable(tf.truncated_normal(shape=[1, dim], mean=0., stddev=.1, dtype=self.FLOAT), name='log_neg_delta')
                self.alpha = tf.exp(log_alpha, name='alpha') + epsilon
                self.beta = tf.exp(log_beta, name='beta') + epsilon
                self.delta = -tf.exp(log_neg_delta, name='delta')
                irf = lambda x: x**(self.alpha-1)*(1+x)**(-self.alpha - self.beta) / tf.exp(tf.lbeta(tf.transpose(tf.concat([self.alpha, self.beta], axis=0))))
                self.irf_global = lambda x: irf(x - self.delta + epsilon)
            else:
                raise ValueError('Impulse response function "%s" is not currently supported.' %self.irf_str)

            def convolve_events(time_target, first_obs, last_obs):
                col_ix = names2ix(f.allsl, f.allsl)
                input_rows = tf.gather(self.X[first_obs:last_obs], col_ix, axis=1)
                input_times = self.time_X[first_obs:last_obs]
                t_delta = time_target - input_times
                conv_coef = self.irf_global(tf.expand_dims(t_delta, -1))
                out = conv_coef*input_rows
                return tf.reduce_sum(out, 0)

            self.X_conv = tf.map_fn(lambda x: convolve_events(*x), [self.time_y, self.first_obs, self.last_obs], dtype=self.FLOAT)

            self.coefficient_global = tf.Variable(tf.truncated_normal(shape=[1, len(f.allsl)], mean=0., stddev=0.1, dtype=self.FLOAT), name='coefficient_global')
            fixef_ix = names2ix(f.fixed, f.allsl)
            coefficient_global_mask = np.zeros(len(f.allsl), dtype=getattr(np, self.float_type))
            coefficient_global_mask[fixef_ix] = 1.
            coefficient_global_mask = tf.constant(coefficient_global_mask)
            self.coefficient_global *= coefficient_global_mask
            self.out += tf.squeeze(tf.matmul(self.X_conv, tf.transpose(self.coefficient_global)), axis=-1)
            self.zeta = []
            self.Z = []
            for i in range(len(f.rangf)):
                r = f.random[i]
                vars = r.vars
                n_var = len(vars)
                mask = np.ones(self.rangf_n_levels[i], dtype=getattr(np, self.float_type))
                mask[self.rangf_n_levels[i] - 1] = 0
                mask = tf.constant(mask)

                if r.intercept:
                    random_intercept = tf.Variable(tf.truncated_normal(shape=[self.rangf_n_levels[i]], mean=0., stddev=.1, dtype=tf.float32), name='intercept_by_%s' % r.grouping_factor)
                    random_intercept *= mask
                    random_intercept -= tf.reduce_mean(random_intercept, axis=0)
                    self.zeta.append(random_intercept)
                    self.out += tf.gather(random_intercept, self.gf_y[:, i])

                if len(vars) > 0:
                    random_effects_matrix = tf.Variable(tf.truncated_normal(shape=[self.rangf_n_levels[i], n_var], mean=0., stddev=.1, dtype=tf.float32), name='random_slopes_by_%s' % (r.grouping_factor))
                    random_effects_matrix *= tf.expand_dims(mask, -1)
                    random_effects_matrix -= tf.reduce_mean(random_effects_matrix, axis=0)
                    self.Z.append(random_effects_matrix)

                    ransl_ix = names2ix(vars, f.allsl)

                    self.out += tf.reduce_sum(tf.gather(self.X_conv, ransl_ix, axis=1) * tf.gather(random_effects_matrix, self.gf_y[:, i]), axis=1)

            self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
            self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
            if self.loss_name.lower() == 'mae':
                self.loss_func = self.mae_loss
            else:
                self.loss_func = self.mse_loss

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
            self.global_batch_step = tf.Variable(0, name='global_batch_step', trainable=False)
            if self.optim_name != 'LBFGS':
                self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step, name='optim')
            else:
                self.train_op = tf.contrib.opt.ScipyOptimizerInterface(self.loss_func, method='LBFGS', options={'maxiter': 50000})

            self.writer = tf.summary.FileWriter(self.outdir + '/train')
            tf.summary.scalar('intercept', intercept_global[0], collections=['params'])
            self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)
            for i in range(len(f.allsl)):
                tf.summary.scalar('coefficient/%s' % f.allsl[i], self.coefficient_global[0, i], collections=['params'])
                for param_name in self.conv_param_names:
                    tf.summary.scalar(param_name+'/%s' % f.allsl[i], getattr(self, param_name)[0, i], collections=['params'])
                if self.log_convolution_plots:
                    tf.summary.histogram('conv/%s' % f.allsl[i], self.irf_global(self.support)[:, i], collections=['params'])

            tf.summary.scalar('loss/%s' % self.loss_name, self.loss_func, collections=['loss'])
            self.summary_params = tf.summary.merge_all(key='params')
            self.summary_losses = tf.summary.merge_all(key='loss')

            self.saver = tf.train.Saver()
            self.load()

            n_params = 0
            for v in self.sess.run(tf.trainable_variables()):
                n_params += np.prod(np.array(v).shape)
            sys.stderr.write('Network contains %d trainable parameters\n' % n_params)

    def train(self,
              X,
              y,
              n_epoch_train=100,
              n_epoch_tune=100,
              minibatch_size=128,
              fixef_name_map=None,
              plot_x_inches=7,
              plot_y_inches=5,
              cmap='gist_earth'):

        usingGPU = is_gpu_available()

        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        f = self.form
        fixef_ix = names2ix(f.fixed, f.allsl)

        with self.sess.graph.as_default():
            y_rangf = y[f.rangf]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            y_range = np.arange(len(y))

            if self.optim_name == 'LBFGS':
                fd = {}
                fd[self.X] = X[f.allsl]
                fd[self.y] = y[f.dv]
                fd[self.time_X] = X.time
                fd[self.time_y] = y.time
                fd[self.gf_y_raw] = y_rangf
                fd[self.first_obs] = y.first_obs
                fd[self.last_obs] = y.last_obs

                def step_callback(x):
                    sys.stderr.write('\rCurrent loss: %s' %x[-1])

                self.train_op.minimize(session=self.sess,
                                       feed_dict=fd,
                                       fetches=[self.loss_func],
                                       loss_callback=lambda x: '\rCurrent loss: %s' %x)

                plot_x = self.support.eval(session=self.sess)
                plot_y = (self.coefficient_global * self.irf_global(self.support)).eval(session=self.sess)
                plot_y = np.swapaxes(np.swapaxes(plot_y, 0, 1)[fixef_ix], 0, 1)

                plot_convolutions(plot_x,
                                  plot_y,
                                  f.fixed,
                                  dir=self.outdir,
                                  filename='convolution_plot.jpg',
                                  fixef_name_map=fixef_name_map,
                                  plot_x_inches=plot_x_inches,
                                  plot_y_inches=plot_y_inches,
                                  cmap=cmap)

                self.save()

            else:
                fd_minibatch = {}
                fd_minibatch[self.X] = X[f.allsl]
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
                    plot_y = (self.coefficient_global * self.irf_global(self.support)).eval(session=self.sess)
                    plot_y = np.swapaxes(np.swapaxes(plot_y, 0, 1)[fixef_ix], 0 ,1)

                    plot_convolutions(plot_x,
                                      plot_y,
                                      f.fixed,
                                      dir=self.outdir,
                                      filename='convolution_plot.jpg',
                                      fixef_name_map=fixef_name_map,
                                      plot_x_inches=plot_x_inches,
                                      plot_y_inches=plot_y_inches,
                                      cmap = cmap)

                    self.save()
                    t1_iter = time.time()
                    sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))


            sys.stderr.write('\n')

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        f = self.form

        with self.sess.graph.as_default():
            y_rangf = y_rangf[y_rangf.columns]
            for c in f.rangf:
                y_rangf[c] = y_rangf[c].astype(str)

            fd = {}
            fd[self.X] = X[f.allsl]
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
            fd[self.X] = X[f.allsl]
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

    def __getstate__(self):
        return (
            self.form,
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
            self.INT
        )

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)
        self.form, \
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
        self.INT = state

        self.construct_network()
        self.load()
