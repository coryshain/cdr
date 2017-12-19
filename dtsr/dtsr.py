import os
import time
from collections import defaultdict
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


class DTSR_kernel(object):
    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 history_length = None,
                 low_memory = True,
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
        self.history_length = history_length
        self.low_memory = low_memory
        if self.history_length is None:
            assert self.low_memory, 'Incompatible DTSR settings: history_length=None and low_memory=False'
        self.rangf_map = []
        self.rangf_n_levels = []
        for i in range(len(f.random)):
            gf = f.random[sorted(f.random.keys())[i]].gf
            keys = np.sort(y[gf].astype('str').unique())
            vals = np.arange(len(keys), dtype=np.int32)
            rangf_map = pd.DataFrame({'id':vals},index=keys).to_dict()['id']
            oov_id = len(keys)+1
            rangf_map = defaultdict(lambda:oov_id, rangf_map)
            self.rangf_map.append(rangf_map)
            self.rangf_n_levels.append(oov_id)
        self.y_mu_init = float(y[f.dv].mean())
        self.y_sigma_init = float(y[f.dv].std())
        self.float_type = float_type
        self.FLOAT = getattr(tf, self.float_type)
        self.int_type = int_type
        self.INT = getattr(tf, self.int_type)
        self.log_random = log_random
        self.irf_tree = self.form.irf_tree

        self.preterminals = []
        self.irf_lambdas = {}
        self.irf_names = []
        self.plot_tensors_atomic_unscaled = {}
        self.plot_tensors_atomic_scaled = {}
        self.plot_tensors_composite_unscaled = {}
        self.plot_tensors_composite_scaled = {}
        self.composite_irfs = {}
        self.atomic_irf_by_family = {}
        self.atomic_irf_means_by_family = {}

        if self.log_random:
            self.summary_random_writers = {}
            self.summary_random_indexers = {}
            self.summary_random = {}

    def initialize_inputs(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.X = tf.placeholder(shape=[None, self.history_length, len(f.terminal_names)], dtype=self.FLOAT, name=sn('X'))
                self.time_X = tf.placeholder(shape=[None, self.history_length], dtype=self.FLOAT, name=sn('time_X'))

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(f.random)], dtype=tf.int32)

                # Linspace tensor used for plotting
                self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)

                self.global_step = tf.Variable(0, name=sn('global_step'), trainable=False)
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(0, name=sn('global_batch_step'), trainable=False)
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

    def initialize_low_memory_inputs(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.X = tf.placeholder(shape=[None, len(f.terminal_names)], dtype=self.FLOAT, name=sn('X'))
                self.time_X = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('time_X'))

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(f.random)], dtype=tf.int32)

                self.first_obs = tf.placeholder(shape=[None], dtype=self.INT, name=sn('first_obs'))
                self.last_obs = tf.placeholder(shape=[None], dtype=self.INT, name=sn('last_obs'))

                # Linspace tensor used for plotting
                self.support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)

                self.global_step = tf.Variable(0, name=sn('global_step'), trainable=False)
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(0, name=sn('global_batch_step'), trainable=False)
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

    def build(self):
        raise NotImplementedError

    def initialize_intercepts_coefficients(self):
        raise NotImplementedError


    def initialize_irf_lambdas(self):
        epsilon = 1e-35  # np.nextafter(0, 1, dtype=getattr(np, self.float_type)) * 10

        with self.sess.as_default():
            with self.sess.graph.as_default():

                def exponential(params):
                    pdf = tf.contrib.distributions.Exponential(rate=params[0]).prob
                    return lambda x: pdf(x + epsilon)

                self.irf_lambdas['Exp'] = exponential

                def shifted_exp(params):
                    pdf = tf.contrib.distributions.Exponential(rate=params[0]).prob
                    return lambda x: pdf(x - params[1] + epsilon)

                self.irf_lambdas['ShiftedExp'] = shifted_exp

                def gamma(params):
                    pdf = tf.contrib.distributions.Gamma(concentration=params[0],
                                                         rate=params[1],
                                                         validate_args=False).prob
                    return lambda x: pdf(x + epsilon)

                self.irf_lambdas['Gamma'] = gamma

                self.irf_lambdas['GammaKgt1'] = gamma

                def shifted_gamma(params):
                    pdf = tf.contrib.distributions.Gamma(concentration=params[0],
                                                         rate=params[1],
                                                         validate_args=False).prob
                    return lambda x: pdf(x - params[2] + epsilon)

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma

                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma

                def normal(params):
                    pdf = tf.contrib.distributions.Normal(loc=params[0], scale=params[1]).prob
                    return lambda x: pdf(x)

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
                    return lambda x: (x + epsilon) ** (alpha - 1.) * (1. + (x + epsilon)) ** (-alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.concat([alpha, beta], axis=0))))

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(params):
                    alpha = params[0]
                    beta = params[1]
                    delta = params[3]
                    return lambda x: (x - delta + epsilon) ** (alpha - 1) * (1 + (x - delta + epsilon)) ** (
                    -alpha - beta) / tf.exp(
                        tf.lbeta(tf.transpose(tf.concat([alpha, beta], axis=0))))

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

    def initialize_irf_params(self):
        raise NotImplementedError

    def initialize_irf_params_inner(self, family, ids):
        raise NotImplementedError

    def initialize_irfs(self, t):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                t.irfs = {}
                t.params = {}
                for f in t.children:
                    child_nodes = sorted(t.children[f].keys())
                    child_irfs = [t.children[f][x].irf_id() for x in child_nodes]
                    if f == 'DiracDelta':
                        assert t.name() == 'ROOT', 'DiracDelta may be embedded under other IRF in DTSR formula strings'
                        plot_base = tf.concat([tf.ones((1,1)), tf.zeros((self.support.shape[0]-1,1))], axis=0)
                        for i in range(len(child_nodes)):
                            child = t.children[f][child_nodes[i]]
                            assert child.terminal(), 'DiracDelta may not dominate other IRF in DTSR formula strings'
                            coefs_ix = names2ix(child.coef_id(), self.form.coefficient_names)
                            child.plot_tensor_atomic_unscaled = plot_base
                            child.plot_tensor_atomic_scaled = plot_base*tf.gather(self.coefficient_fixed_means, coefs_ix)
                            self.plot_tensors_atomic_unscaled[child.name()] = child.plot_tensor_atomic_unscaled
                            self.plot_tensors_atomic_scaled[child.name()] = child.plot_tensor_atomic_scaled
                            child.plot_tensor_composite_unscaled = child.plot_tensor_atomic_unscaled
                            child.plot_tensor_composite_scaled = child.plot_tensor_atomic_scaled
                            self.plot_tensors_composite_unscaled[child.name()] = child.plot_tensor_composite_unscaled
                            self.plot_tensors_composite_scaled[child.name()] = child.plot_tensor_composite_scaled
                            self.preterminals.append(child.name())
                            self.irf_names.append(child.name())
                    else:
                        params_ix = names2ix(child_irfs, sorted(self.form.atomic_irf_by_family[f]))
                        t.params[f] = tf.gather(self.atomic_irf_by_family[f], params_ix, axis=1)
                        composite = False
                        parent_unscaled = getattr(t, 'plot_tensor_composite_unscaled', None)
                        parent_scaled = getattr(t, 'plot_tensor_composite_scaled', None)
                        if parent_unscaled is not None:
                            composite = True
                        t.irfs[f] = self.new_irf(self.irf_lambdas[f], t.params[f])

                        for i in range(len(child_nodes)):
                            child = t.children[f][child_nodes[i]]
                            coefs_ix = names2ix(child.coef_id(), self.form.coefficient_names)
                            child.plot_tensor_atomic_unscaled = self.new_irf(self.irf_lambdas[f], self.atomic_irf_means_by_family[f][:, i])(self.support)
                            child.plot_tensor_atomic_scaled = self.new_irf(self.irf_lambdas[f], self.atomic_irf_means_by_family[f][:, i])(self.support)*tf.gather(self.coefficient_fixed_means, coefs_ix)
                            self.plot_tensors_atomic_unscaled[child.name()] = child.plot_tensor_atomic_unscaled
                            self.plot_tensors_atomic_scaled[child.name()] = child.plot_tensor_atomic_scaled
                            if composite:
                                child.plot_tensor_composite_unscaled = self.new_irf(self.irf_lambdas[f], self.atomic_irf_means_by_family[:, i])(parent_unscaled)
                                child.plot_tensor_composite_scaled = self.new_irf(self.irf_lambdas[f], self.atomic_irf_means_by_family[:, i])(parent_scaled)*tf.gather(self.coefficient_fixed_means, coefs_ix)
                            else:
                                child.plot_tensor_composite_unscaled = child.plot_tensor_atomic_unscaled
                                child.plot_tensor_composite_scaled = child.plot_tensor_atomic_scaled
                            self.plot_tensors_composite_unscaled[child.name()] = child.plot_tensor_composite_unscaled
                            self.plot_tensors_composite_scaled[child.name()] = child.plot_tensor_composite_scaled
                            if child.terminal:
                                self.preterminals.append(child.name())
                            else:
                                self.initialize_irfs(child)
                            self.irf_names.append(child.name())

    def initialize_convolutional_feedforward(self, t):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = []
                for f in t.children:
                    preterminals = []
                    terminals = []
                    child_nodes = sorted(t.children[f].keys())
                    child_coefs = [t.children[f][x].coef_id() for x in child_nodes]
                    coefs_ix = names2ix(child_coefs, self.form.coefficient_names)
                    if f == 'DiracDelta':
                        for i in range(len(child_nodes)):
                            x = t.children[f][child_nodes[i]]
                            terminals.append(x.impulse.name())
                        terminals_ix = names2ix(terminals, self.form.terminal_names)
                        new_out = tf.gather(self.X, terminals_ix, axis=2)[:,-1,:] * tf.gather(self.coefficient, coefs_ix, axis=1)
                        out.append(new_out)
                    else:
                        tensor = t.irfs[f](t.tensor) * tf.expand_dims(tf.gather(self.coefficient, coefs_ix, axis=1), 1)
                        for i in range(len(child_nodes)):
                            x = t.children[f][child_nodes[i]]
                            if x.terminal():
                                preterminals.append(x.name())
                                terminals.append(x.impulse.name())
                            x.tensor = tf.expand_dims(tensor[:, :, i],-1)
                            if not x.terminal:
                                out += self.initialize_convolutional_feedforward(x)
                        if len(preterminals) > 0 and len(terminals) > 0:
                            preterminals_ix = names2ix(preterminals, child_nodes)
                            terminals_ix = names2ix(terminals, self.form.terminal_names)
                            new_out = tf.reduce_sum(tf.gather(self.X, terminals_ix, axis=2) * tf.gather(tensor, preterminals_ix, axis=2), 1)
                            out.append(new_out)
                return out

    def initialize_low_memory_convolutional_feedforward(self, t, inputs, coef):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = []
                for f in t.children:
                    preterminals = []
                    terminals = []
                    child_nodes = sorted(t.children[f].keys())
                    child_coefs = [t.children[f][x].coef_id() for x in child_nodes]
                    coefs_ix = names2ix(child_coefs, self.form.coefficient_names)
                    tensor = t.irfs[f](t.tensor) * tf.gather(coef, coefs_ix)
                    for i in range(len(child_nodes)):
                        x = t.children[f][child_nodes[i]]
                        if x.terminal():
                            preterminals.append(x.name())
                            terminals.append(x.impulse.name())
                        x.tensor = tf.expand_dims(tensor[:, i], -1)
                        if not x.terminal:
                            out += self.initialize_low_memory_convolutional_feedforward(x, inputs, coef)
                    if len(preterminals) > 0 and len(terminals) > 0:
                        preterminals_ix = names2ix(preterminals, child_nodes)
                        terminals_ix = names2ix(terminals, self.form.terminal_names)
                        out.append(tf.gather(inputs, terminals_ix, axis=1) * tf.gather(tensor, preterminals_ix, axis=1))
                return out

    def construct_network(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.irf_tree.tensor = tf.expand_dims(tf.expand_dims(self.time_y, -1) - self.time_X, -1)  # Tensor of temporal offsets with shape (?,history_length)
                self.X_conv = self.initialize_convolutional_feedforward(self.irf_tree) # num_terminals-length array of convolved IV with shape (?)
                self.X_conv = tf.concat(self.X_conv, axis=1)
                self.out = self.intercept + tf.reduce_sum(self.X_conv, axis=1)

    def construct_low_memory_network(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                def convolve_events(time_target, first_obs, last_obs, coefficient):
                    inputs = self.X[first_obs:last_obs]
                    input_times = self.time_X[first_obs:last_obs]
                    t_delta = time_target - input_times

                    if not self.ransl:
                        coefficient = self.coefficient[0]

                    self.irf_tree.tensor = tf.expand_dims(t_delta, -1)
                    out = self.initialize_low_memory_convolutional_feedforward(self.irf_tree, inputs, coefficient)

                    out = tf.concat(out, axis=1)

                    return tf.reduce_sum(out, 0)

                self.X_conv = tf.map_fn(lambda x: convolve_events(*x),
                                        [self.time_y, self.first_obs, self.last_obs, self.coefficient],
                                        parallel_iterations=10,
                                        dtype=self.FLOAT)

                self.out = self.intercept + tf.reduce_sum(self.X_conv, axis=1)


    def initialize_objective(self):
        raise NotImplementedError

    def start_logging(self):
        raise NotImplementedError

    def initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

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

    ######################################################
    #
    #  Inner functions for network construction
    #
    ######################################################

    def optim_init(self, name, learning_rate):
        with self.sess.as_default():
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

    def new_irf(self, irf_lambda, params, parent_irf=None):
        irf = irf_lambda(params)
        if parent_irf is None:
            def new_irf(x):
                return irf(x)
        else:
            def new_irf(x):
                return irf(parent_irf(x))
        return new_irf

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




    ######################################################
    #
    #  Handles for external methods
    #
    ######################################################

    def save(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver.save(self.sess, self.outdir + '/model.ckpt')

    def load(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if os.path.exists(self.outdir + '/checkpoint'):
                    self.saver.restore(self.sess, self.outdir + '/model.ckpt')
                else:
                    self.sess.run(tf.global_variables_initializer())

    def expand_history(self, X, X_time, first_obs, last_obs):
        last_obs = np.array(last_obs, dtype='int32')
        first_obs = np.maximum(np.array(first_obs, dtype='int32'), last_obs - self.history_length)
        X_time = np.array(X_time, dtype=getattr(np, self.float_type))
        X = np.array(X, dtype=getattr(np, self.float_type))

        X_history = np.zeros((first_obs.shape[0], self.history_length, X.shape[1]))
        time_X_history = np.zeros((first_obs.shape[0], self.history_length))

        for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
            sX = X[first:last]
            sXt = X_time[first:last]
            X_history[i, -sX.shape[0]:] += sX
            time_X_history[i][-len(sXt):] += sXt

        return X_history, time_X_history

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, y_time, y_rangf, first_obs, last_obs):
        raise NotImplementedError

    def eval(self, X, y):
        raise NotImplementedError

    def make_plots(self, irf_name_map=None, plot_x_inches=7., plot_y_inches=5., cmap=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                plot_x = self.support.eval(session=self.sess)

                plot_y = []
                for x in self.irf_names:
                    plot_y.append(self.plot_tensors_atomic_unscaled[x].eval(session=self.sess))
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
                    plot_y.append(self.plot_tensors_atomic_scaled[x].eval(session=self.sess))
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

                plot_y = []
                for x in self.irf_names:
                    plot_y.append(self.plot_tensors_composite_unscaled[x].eval(session=self.sess))
                plot_y = np.concatenate(plot_y, axis=1)

                plot_convolutions(plot_x,
                                  plot_y,
                                  self.irf_names,
                                  dir=self.outdir,
                                  filename='irf_composite_unscaled.jpg',
                                  irf_name_map=irf_name_map,
                                  plot_x_inches=plot_x_inches,
                                  plot_y_inches=plot_y_inches,
                                  cmap=cmap)

                plot_y = []
                for x in self.irf_names:
                    coef_ix = self.form.coefficient_names.index(x)
                    plot_y.append(self.plot_tensors_composite_scaled[x].eval(session=self.sess))
                plot_y = np.concatenate(plot_y, axis=1)

                plot_convolutions(plot_x,
                                  plot_y,
                                  self.irf_names,
                                  dir=self.outdir,
                                  filename='irf_composite_scaled.jpg',
                                  irf_name_map=irf_name_map,
                                  plot_x_inches=plot_x_inches,
                                  plot_y_inches=plot_y_inches,
                                  cmap=cmap)

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError




class DTSR(DTSR_kernel):

    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 float_type='float32',
                 int_type='int32',
                 log_random=False,
                 optim='Adam',
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.,
                 learning_rate_decay_family='exponential',
                 learning_rate_min=1e-4,
                 loss='mse',
                 ):

        super(DTSR, self).__init__(
            form_str,
            y,
            outdir,
            float_type=float_type,
            int_type=int_type,
            log_random=log_random
        )

        self.optim_name = optim
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_rate_decay_family = learning_rate_decay_family
        self.learning_rate_min = learning_rate_min
        self.loss_name = loss

        self.build()

    def build(self):
        sys.stderr.write('Constructing network from model tree:\n')
        sys.stdout.write(str(self.irf_tree))
        sys.stdout.write('\n')

        self.initialize_low_memory_inputs()
        self.initialize_intercepts_coefficients()
        self.initialize_irf_lambdas()
        self.initialize_irf_params()
        self.initialize_irfs(self.irf_tree, bayesian=False)
        self.construct_low_memory_network()
        self.initialize_objective()
        self.start_logging()
        self.initialize_saver()
        self.load()
        self.report_n_params()

    def initialize_intercepts_coefficients(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if f.intercept:
                    self.intercept_fixed = tf.Variable(tf.constant(self.intercept_init, shape=[1]), dtype=self.FLOAT,
                                                       name='intercept')
                else:
                    self.intercept_fixed = tf.constant(0., dtype=self.FLOAT, name='intercept')
                self.intercept = self.intercept_fixed
                tf.summary.scalar('intercept', self.intercept[0], collections=['params'])

                self.coefficient_fixed = tf.Variable(
                    tf.truncated_normal(shape=[len(f.coefficient_names)], mean=0., stddev=0.1, dtype=self.FLOAT),
                    name='coefficient_fixed')
                for i in range(len(f.coefficient_names)):
                    tf.summary.scalar('coefficient' + '/%s' % f.coefficient_names[i], self.coefficient_fixed[i], collections=['params'])
                self.coefficient = self.coefficient_fixed
                fixef_ix = names2ix(f.fixed_coefficient_names, f.coefficient_names)
                coefficient_fixed_mask = np.zeros(len(f.coefficient_names), dtype=getattr(np, self.float_type))
                coefficient_fixed_mask[fixef_ix] = 1.
                coefficient_fixed_mask = tf.constant(coefficient_fixed_mask)
                self.coefficient_fixed *= coefficient_fixed_mask
                self.coefficient_fixed_means = self.coefficient_fixed
                self.coefficient = self.coefficient_fixed
                self.coefficient = tf.expand_dims(tf.coefficient, -1)
                self.ransl = False
                if self.log_random:
                    writers = {}
                for i in range(len(f.ran_names)):
                    r = f.random[f.ran_names[i]]
                    coefs = r.coefficient_names
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
                            self.summary_random_writers[r.name()] = [tf.summary.FileWriter(self.outdir + '/tensorboard/by_' + r.gf + '/%d' % j) for j in range(min(10, self.rangf_n_levels[i]))]
                            self.summary_random_indexers[r.name()] = tf.placeholder(dtype=tf.int32)
                            tf.summary.scalar('by_' + r.gf + '/intercept', intercept_random[self.summary_random_indexers[r.name()]], collections=['by_' + r.gf])
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
                                tf.summary.scalar('by_' + r.gf + '/' + coef_names[k], coefficient_random[self.summary_random_indexers[r.name()],k], collections=['by_' + r.gf])

                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)

    def initialize_irf_params(self):
        f = self.form
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for x in f.atomic_irf_by_family:
                    self.atomic_irf_by_family[x] = self.initialize_irf_params_inner(x, sorted(f.atomic_irf_by_family[x]))
                    self.atomic_irf_means_by_family[x] = self.atomic_irf_by_family[x]

    def initialize_irf_params_inner(self, family, ids):
        ## Infinitessimal value to add to bounded parameters
        epsilon = 1e-35  # np.nextafter(0, 1, dtype=getattr(np, self.float_type)) * 10
        dim = len(ids)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if family == 'DiracDelta':
                    filler = tf.constant(1., shape=[1, dim])
                    return filler
                if family == 'Exp':
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[1, dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                    return L
                if family == 'ShiftedExp':
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT)
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([L, delta], axis=0)
                if family == 'Gamma':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    return tf.stack([k, theta])
                if family == 'GammaKgt1':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    k = tf.exp(log_k, name=sn('k_%s' % '-'.join(ids))) + epsilon + 1.
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                    return tf.stack([k, theta])
                if family == 'ShiftedGamma':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT)
                    k = tf.exp(log_k, name=sn('k')) + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([k, theta, delta], axis=0)
                if family == 'ShiftedGammaKgt1':
                    log_k = tf.get_variable(sn('log_k_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_theta = tf.get_variable(sn('log_theta_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT)
                    k = tf.nn.softplus(log_k, name=sn('k_%s' % '-'.join(ids))) + 1. + epsilon
                    theta = tf.exp(log_theta, name=sn('theta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('k' + '/%s' % ids[i], k[i], collections=['params'])
                        tf.summary.scalar('theta' + '/%s' % ids[i], theta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([k, theta, delta], axis=0)
                if family == 'Normal':
                    log_sigma = tf.Variable(
                        tf.truncated_normal([dim], stddev=.1, dtype=self.FLOAT),
                        name=sn('log_sigma_%s' % '-'.join(ids))
                    )
                    mu = tf.Variable(
                        tf.truncated_normal([dim], stddev=.1, dtype=self.FLOAT),
                        name=sn('mu_%s' % '-'.join(ids))
                    )
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                    return tf.stack([mu, sigma], axis=0)
                if family == 'SkewNormal':
                    log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim],
                                         initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    alpha = tf.get_variable(sn('alpha_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                    return tf.stack([mu, sigma, alpha], axis=1)
                if family == 'EMG':
                    log_sigma = tf.get_variable(sn('log_sigma_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    mu = tf.get_variable(sn('mu_%s' % '-'.join(ids)), shape=[dim],
                                         initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_L = tf.get_variable(sn('log_L_%s' % '-'.join(ids)), shape=[dim],
                                            initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    sigma = tf.exp(log_sigma, name=sn('sigma_%s' % '-'.join(ids))) + epsilon
                    L = tf.exp(log_L, name=sn('L_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('mu' + '/%s' % ids[i], mu[i], collections=['params'])
                        tf.summary.scalar('sigma' + '/%s' % ids[i], sigma[i], collections=['params'])
                        tf.summary.scalar('L' + '/%s' % ids[i], L[i], collections=['params'])
                    return tf.stack([mu, sigma, L], axis=0)
                if family == 'BetaPrime':
                    log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim],
                                               initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                    beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                    for i in range(dim):
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                        tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                    return tf.stack([alpha, beta], axis=0)
                if family == 'ShiftedBetaPrime':
                    log_alpha = tf.get_variable(sn('log_alpha_%s' % '-'.join(ids)), shape=[dim],
                                                initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_beta = tf.get_variable(sn('log_beta_%s' % '-'.join(ids)), shape=[dim],
                                               initializer=tf.truncated_normal_initializer(stddev=.1), dtype=self.FLOAT)
                    log_neg_delta = tf.get_variable(sn('log_neg_delta_%s' % '-'.join(ids)), shape=[dim],
                                                    initializer=tf.truncated_normal_initializer(stddev=.1),
                                                    dtype=self.FLOAT)
                    alpha = tf.exp(log_alpha, name=sn('alpha_%s' % '-'.join(ids))) + epsilon
                    beta = tf.exp(log_beta, name=sn('beta_%s' % '-'.join(ids))) + epsilon
                    delta = -tf.exp(log_neg_delta, name=sn('delta_%s' % '-'.join(ids)))
                    for i in range(dim):
                        tf.summary.scalar('alpha' + '/%s' % ids[i], alpha[i], collections=['params'])
                        tf.summary.scalar('beta' + '/%s' % ids[i], beta[i], collections=['params'])
                        tf.summary.scalar('delta' + '/%s' % ids[i], delta[i], collections=['params'])
                    return tf.stack([alpha, beta, delta], axis=0)
                raise ValueError('Impulse response function "%s" is not currently supported.' % family)

    def initialize_objective(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.mae_loss = tf.losses.absolute_difference(self.y, self.out)
                self.mse_loss = tf.losses.mean_squared_error(self.y, self.out)
                if self.loss_name.lower() == 'mae':
                    self.loss_func = self.mae_loss
                else:
                    self.loss_func = self.mse_loss
                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT, name='loss_total')
                tf.summary.scalar('loss/%s' % self.loss_name, self.loss_total, collections=['loss'])

                sys.stderr.write('Using optimizer %s\n' % self.optim_name)
                if self.optim_name != 'LBFGS':
                    self.lr = tf.constant(self.learning_rate)
                    if self.learning_rate_decay_factor > 0:
                        decay_step = tf.cast(self.global_step, dtype=self.FLOAT) * tf.constant(
                            self.learning_rate_decay_factor)
                        if self.learning_rate_decay_family == 'linear':
                            decay_coef = 1 - decay_step
                        elif self.learning_rate_decay_family == 'inverse':
                            decay_coef = 1. / (1. + decay_step)
                        elif self.learning_rate_decay_family == 'exponential':
                            decay_coef = 2 ** (-decay_step)
                        elif self.learning_rate_decay_family.startswith('stepdown'):
                            interval = tf.constant(float(self.learning_rate_decay_family[8:]), dtype=self.FLOAT)
                            decay_coef = tf.constant(self.learning_rate_decay_factor) ** tf.floor(
                                tf.cast(self.global_step, dtype=self.FLOAT) / interval)
                        else:
                            raise ValueError(
                                'Unrecognized learning rate decay schedule: "%s"' % self.learning_rate_decay_family)
                        self.lr = self.lr * decay_coef
                        self.lr = tf.clip_by_value(self.lr, tf.constant(self.learning_rate_min), inf)
                    self.optim = self.optim_init(self.optim_name, self.lr)

                    # self.train_op = self.optim.minimize(self.loss_func, global_step=self.global_batch_step,
                    #                                     name=sn('optim'))
                    # self.gradients = self.optim.compute_gradients(self.loss_func)

                    self.gradients, variables = zip(*self.optim.compute_gradients(self.loss_func))
                    # ## CLIP GRADIENT NORM
                    # self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
                    self.train_op = self.optim.apply_gradients(zip(self.gradients, variables),
                                                               global_step=self.global_batch_step, name=sn('optim'))
                else:
                    self.train_op = tf.contrib.opt.ScipyOptimizerInterface(self.loss_func, method='LBFGS',
                                                                           options={'maxiter': 50000})

    def start_logging(self):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/fixed', self.sess.graph)
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random:
                    for r in f.ran_names:
                        self.summary_random[r] = tf.summary.merge_all(key='by_' + f.random[r].gf)

    def fit(self,
            X,
            y,
            n_epoch_train=100,
            n_epoch_tune=100,
            minibatch_size=128,
            irf_name_map=None,
            plot_x_inches=28,
            plot_y_inches=5,
            cmap='gist_earth'):

        usingGPU = is_gpu_available()

        sys.stderr.write('Using GPU: %s\n' % usingGPU)

        f = self.form

        sys.stderr.write('Correlation matrix for input variables:Corr\n')
        rho = X[f.terminal_names].corr()
        sys.stderr.write(str(rho) + '\n\n')

        with self.sess.as_default():
            with self.sess.graph.as_default():
                y_rangf = y[f.rangf]
                for i in range(len(f.rangf)):
                    c = f.rangf[i]
                    y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

                y_range = np.arange(len(y))

                if self.optim_name == 'LBFGS':
                    fd = {}
                    fd[self.X] = X[f.terminal_names]
                    fd[self.y] = y[f.dv]
                    fd[self.time_X] = X.time
                    fd[self.time_y] = y.time
                    fd[self.gf_y] = y_rangf
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
                    fd[self.gf_y] = y_rangf
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
                        if minibatch_size == inf:
                            minibatch_size = len(y)
                        n_minibatch = math.ceil(float(len(y)) / minibatch_size)

                        t0_iter = time.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.learning_rate_decay_factor > 0:
                            sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                        loss_total = 0

                        for j in range(0, len(y), minibatch_size):
                            fd_minibatch[self.y] = y[f.dv].iloc[p[j:j + minibatch_size]]
                            fd_minibatch[self.time_y] = y.time.iloc[p[j:j + minibatch_size]]
                            fd_minibatch[self.gf_y] = y_rangf.iloc[p[j:j + minibatch_size]]
                            fd_minibatch[self.first_obs] = y.first_obs.iloc[p[j:j + minibatch_size]]
                            fd_minibatch[self.last_obs] = y.last_obs.iloc[p[j:j + minibatch_size]]

                            # _, gradients_minibatch, loss_minibatch = self.sess.run(
                            #     [self.train_op, self.gradients, self.loss_func],
                            #     feed_dict=fd_minibatch)
                            _, loss_minibatch = self.sess.run(
                                [self.train_op, self.loss_func],
                                feed_dict=fd_minibatch)
                            loss_total += loss_minibatch
                            # if gradients is None:
                            #     gradients = list(gradients_minibatch)
                            # else:
                            #     for i in range(len(gradients)):
                            #         gradients[i] += gradients_minibatch[i]
                            pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

                        loss_total /= n_minibatch
                        fd[self.loss_total] = loss_total
                        # sys.stderr.write('Evaluating gradients...\n')
                        # max_grad = 0
                        # for g in gradients:
                        #     max_grad = max(max_grad, np.max(np.abs(g[0]/n_minibatch)))
                        # sys.stderr.write('  max(grad) = %s\n' % str(max_grad))
                        # sys.stderr.write('  Converged (max(grad) < 0.001) = %s\n' % (max_grad < 0.001))

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

                        sys.stderr.write('Number of graph nodes: %d\n' % len(self.sess.graph._nodes_by_name))
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
                fd[self.gf_y] = y_rangf
                fd[self.first_obs] = y.first_obs
                fd[self.last_obs] = y.last_obs

                X_conv = pd.DataFrame(self.sess.run(self.X_conv, feed_dict=fd), columns=self.preterminals)

                sys.stderr.write('Mean values of convolved predictors\n')
                sys.stderr.write(str(X_conv.mean(axis=0)) + '\n')
                sys.stderr.write('Correlations of convolved predictors')
                sys.stderr.write(str(X_conv.corr()) + '\n')
                sys.stderr.write('\n')

                self.make_plots(irf_name_map, plot_x_inches, plot_y_inches, cmap)


    def predict(self, X, y_time, y_rangf, first_obs, last_obs, minibatch_size=inf):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                y_rangf = y_rangf[y_rangf.columns]
                for i in range(len(f.rangf)):
                    c = f.rangf[i]
                    y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

                fd = {}
                fd[self.X] = X[f.terminal_names]
                fd[self.time_X] = X.time
                fd[self.time_y] = y_time
                fd[self.gf_y] = y_rangf
                fd[self.first_obs] = first_obs
                fd[self.last_obs] = last_obs

                return self.sess.run(self.out, feed_dict=fd)

    def eval(self, X, y):
        f = self.form

        with self.sess.as_default():
            with self.sess.graph.as_default():
                y_rangf = y[f.rangf]
                for i in range(len(f.rangf)):
                    c = f.rangf[i]
                    y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

                fd = {}
                fd[self.X] = X[f.terminal_names]
                fd[self.time_X] = X.time
                fd[self.y] = y[f.dv]
                fd[self.time_y] = y.time
                fd[self.gf_y] = y_rangf
                fd[self.first_obs] = y.first_obs
                fd[self.last_obs] = y.last_obs

                return self.sess.run(self.loss_func, feed_dict=fd)

    def __getstate__(self):

        return (
            self.form_str,
            self.outdir,
            self.rangf_map,
            self.rangf_n_levels,
            self.intercept_init,
            self.float_type,
            self.FLOAT,
            self.int_type,
            self.INT,
            self.log_random,
            self.optim_name,
            self.learning_rate,
            self.learning_rate_decay_factor,
            self.learning_rate_decay_family,
            self.learning_rate_min,
            self.loss_name,
        )

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)
        self.form_str, \
        self.outdir, \
        self.rangf_map, \
        self.rangf_n_levels, \
        self.intercept_init, \
        self.float_type, \
        self.FLOAT, \
        self.int_type, \
        self.INT, \
        self.log_random, \
        self.optim_name, \
        self.learning_rate, \
        self.learning_rate_decay_factor, \
        self.learning_rate_decay_family, \
        self.learning_rate_min, \
        self.loss_name = state

        self.form = Formula(self.form_str)
        self.irf_tree = self.form.irf_tree

        self.build()

