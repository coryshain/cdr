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
    """
    Variance of a tensor, along the specified axis.

    :param x: A tensor or variable.
    :param axis:  An integer, the axis along which to compute the variance.
    :param keepdims: A boolean, whether to keep the dimensions or not.
        If ``keepdims`` is ``False``, the rank of the tensor is reduced
        by 1. If ``keepdims`` is ``True``,
        the reduced dimension is retained with length 1.
    :return: A tensor with the variance of elements of ``x``.
    """

    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """
    Standard deviation of a tensor, along the specified axis.

    :param x: A tensor or variable.
    :param axis:  An integer, the axis along which to compute the standard deviation.
    :param keepdims: A boolean, whether to keep the dimensions or not.
        If ``keepdims`` is ``False``, the rank of the tensor is reduced
        by 1. If ``keepdims`` is ``True``,
        the reduced dimension is retained with length 1.
    :return: A tensor with the variance of elements of ``x``.
    """

    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def sn(string):
    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)


class DTSR(object):
    """
    Abstract base class for DTSR. Bayesian (BDTSR) and Neural Network (NNDTSR) implementations inherit from DTSR.

    To build a new DTSR implementation, you must (at minimum) implement the following methods:

    - ``build``
    - ``initialize_intercepts_coefficients``
    - ``initialize_irf_params``
    - ``initialize_irf_params_inner``
    - ``initialize_objective``
    - ``start_logging``
    - ``fit``
    - ``predict``
    - ``eval``
    - ``__getstate__``
    - ``__setstate__``

    You will also likely need to override the ``__init__`` method.

    Arguments
    =========
    :param form_str: An R-style string representing the DTSR model formula.
    :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:
        - ``time``: Timestamp for each entry in ``y``
        - ``first_obs``: Index in the design matrix X of the first observation in the time series associated with
            each entry in ``y``
        - ``last_obs``: Index in the design matrix :math:``X`` of the immediately preceding observation in the time series
            associated with each entry in ``y``
        - A column with the same name as the DV specified in ``form_str``
        - A column for each random grouping factor in the model specified in ``form_str``.
    :param outdir: A ``str`` representing the output directory, where logs and model parameters are saved.
    :param history_length: An ``int`` representing the maximum length of the history window to use. If ``None``, history
        length is unbounded and only the low-memory model is permitted.
    :param low_memory: A ``bool`` determining which DTSR memory implementation to use. If ``low_memory == True``, DTSR
        convolves over history windows for each observation of in ``y`` using a TensorFlow control op. It can be used
        with unboundedly long histories and uses less memory, but is generally much slower and results in poor GPU
        utilization. If ``low_memory == False``, DTSR expands the design matrix into a rank 3 tensor in which the 2nd
        axis contains the history for each independent variable for each observation of the independent variable.
        This requires more memory in order to store redundant input values and requires a finite history length.
        However, it removes the need for a control op in the feedforward component and therefore generally runs much
        faster if GPU is available.
    :param float_type: A ``str`` representing the ``float`` type to use throughout the network.
    :param int_type: A ``str`` representing the ``int`` type to use throughout the network (used for tensor slicing).
    :param minibatch_size: An ``int`` representing the size of minibatches to use for fitting/prediction, or the
        string ``inf`` to perform full-batch training.
    :param logging_freq: An ``int`` representing the frequency (in minibatches) with which to write Tensorboard logs.
    :param log_random: A ``bool`` determining whether to log random effects to Tensorboard.
    :param save_freq: An ``int`` representing the frequency (in iterations) with which to save model checkpoints.
    """
    def __init__(self,
                 form_str,
                 y,
                 outdir,
                 history_length = None,
                 low_memory = True,
                 float_type='float32',
                 int_type='int32',
                 minibatch_size=None,
                 logging_freq=1,
                 log_random=True,
                 save_freq=1
                 ):

        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self.form_str = form_str
        self.form = Formula(form_str)
        f = self.form

        self.float_type = float_type
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.int_type = int_type
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        self.outdir = outdir
        if history_length is None or history_length == 'inf':
            self.history_length = inf
        else:
            self.history_length = history_length
        self.low_memory = low_memory
        if self.history_length is None:
            assert self.low_memory, 'Incompatible DTSR settings: history_length=None and low_memory=False'
        self.rangf_map = []
        self.rangf_n_levels = []
        for i in range(len(f.random)):
            gf = f.random[sorted(f.random.keys())[i]].gf
            keys = np.sort(y[gf].astype('str').unique())
            vals = np.arange(len(keys), dtype=self.INT_NP)
            rangf_map = pd.DataFrame({'id':vals},index=keys).to_dict()['id']
            oov_id = len(keys)+1
            rangf_map = defaultdict(lambda:oov_id, rangf_map)
            self.rangf_map.append(rangf_map)
            self.rangf_n_levels.append(oov_id)
        self.y_mu_init = float(y[f.dv].mean())
        self.y_sigma_init = float(y[f.dv].std())
        if minibatch_size is None or minibatch_size == 'inf':
            self.minibatch_size = inf
        else:
            self.minibatch_size = minibatch_size
        self.n_train_minibatch = math.ceil(float(len(y)) / self.minibatch_size)
        self.minibatch_scale = float(len(y)) / self.minibatch_size
        self.log_random = log_random
        self.logging_freq = logging_freq
        self.save_freq = save_freq
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
                self.X = tf.placeholder(shape=[None, self.history_length, len(f.terminal_names)], dtype=self.FLOAT_TF, name=sn('X'))
                self.time_X = tf.placeholder(shape=[None, self.history_length], dtype=self.FLOAT_TF, name=sn('time_X'))

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(f.random)], dtype=self.INT_TF)

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
                self.X = tf.placeholder(shape=[None, len(f.terminal_names)], dtype=self.FLOAT_TF, name=sn('X'))
                self.time_X = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_X'))

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))

                self.gf_y = tf.placeholder(shape=[None, len(f.random)], dtype=tf.INT)

                self.first_obs = tf.placeholder(shape=[None], dtype=self.INT_TF, name=sn('first_obs'))
                self.last_obs = tf.placeholder(shape=[None], dtype=self.INT_TF, name=sn('last_obs'))

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
        epsilon = 1e-35

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
                    if f == 'DiracDelta':
                        for i in range(len(child_nodes)):
                            x = t.children[f][child_nodes[i]]
                            terminals.append(x.impulse.name())
                        terminals_ix = names2ix(terminals, self.form.terminal_names)
                        new_out = tf.gather(inputs[-1], terminals_ix, axis=0) * tf.gather(self.coefficient[-1], coefs_ix, axis=0)
                        out.append(new_out)
                    else:
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
                            out.append(tf.reduce_sum(tf.gather(inputs, terminals_ix, axis=1) * tf.gather(tensor, preterminals_ix, axis=1), 0))
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
                if np.isfinite(self.history_length):
                    history_length = tf.constant(self.history_length, dtype=self.INT_TF)

                def convolve_events(time_target, first_obs, last_obs, coefficient):
                    if np.isfinite(self.history_length):
                        inputs = self.X[tf.maximum(first_obs, last_obs-history_length):last_obs]
                        input_times = self.time_X[tf.maximum(first_obs, last_obs-history_length):last_obs]
                    else:
                        inputs = self.X[first_obs:last_obs]
                        input_times = self.time_X[first_obs:last_obs]
                    t_delta = time_target - input_times

                    if not self.ransl:
                        coefficient = self.coefficient[0]

                    self.irf_tree.tensor = tf.expand_dims(t_delta, -1)
                    out = self.initialize_low_memory_convolutional_feedforward(self.irf_tree, inputs, coefficient)
                    out = tf.concat(out, axis=0)

                    return out

                self.X_conv = tf.map_fn(lambda x: convolve_events(*x),
                                        [self.time_y, self.first_obs, self.last_obs, self.coefficient],
                                        parallel_iterations=10,
                                        dtype=self.FLOAT_TF)

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
                                  filename='irf_atomic_unscaled.png',
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
                                  filename='irf_atomic_scaled.png',
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
                                  filename='irf_composite_unscaled.png',
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
                                  filename='irf_composite_scaled.png',
                                  irf_name_map=irf_name_map,
                                  plot_x_inches=plot_x_inches,
                                  plot_y_inches=plot_y_inches,
                                  cmap=cmap)

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError



