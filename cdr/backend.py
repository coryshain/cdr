import collections
import tensorflow as tf

from .util import *

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.distributions import Normal

if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell

parse_initializer = re.compile('(.*_initializer)(_(.*))?')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str):
                    if activation.lower() == 'hard_sigmoid':
                        out = hard_sigmoid
                    elif activation.lower() == 'gelu':
                        out = lambda x: x * tf.nn.sigmoid(1.702*x)
                    elif activation.lower() == 'gelu1p':
                        out = lambda x: (x + 1) * tf.nn.sigmoid(1.702*(x+1))
                    elif activation.lower() == 'l2norm':
                        out = lambda x: tf.nn.l2_normalize(x, axis=-1)
                    elif activation.lower() == 'swish':
                        out = lambda x: x * tf.nn.sigmoid(x)
                    elif activation.lower() == 'shifted_softplus':
                        out = lambda x: tf.nn.softplus(x) - 0.69314718056
                    elif activation.lower() == 'nlrelu':
                        out = lambda x: tf.log1p(tf.nn.relu(x))
                    elif activation.lower() in ('log', 'logmod', 'log-modulus', 'logmodulus'):
                        # def out(x):
                        #     # This is mathematically equivalent to tf.sign(x) * tf.log1p(tf.abs(x))
                        #     # but ensures that the gradient is well defined (and equal to 1) at 0
                        #     pos = x > 0.
                        #     one = tf.ones_like(x)
                        #     return tf.where(pos, one, -one) * tf.log1p(tf.where(pos, x, -x))
                        out = lambda x: tf.sign(x) * tf.log1p(tf.abs(x))
                    else:
                        out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_constraint(name):
    if name.lower() == 'softplus':
        constraint_fn = tf.nn.softplus
        constraint_fn_np = lambda x: np.log(np.exp(x) + 1.)
        constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        constraint_fn_inv_np = lambda x: np.log(np.exp(x) - 1.)
    elif name.lower() == 'square':
        constraint_fn = tf.square
        constraint_fn_np = np.square
        constraint_fn_inv = tf.sqrt
        constraint_fn_inv_np = np.sqrt
    elif name.lower() == 'abs':
        constraint_fn = lambda x: tf.where(x > 0., x, -x)
        constraint_fn_np = np.abs
        constraint_fn_inv = tf.identity
        constraint_fn_inv_np = lambda x: x
    else:
        raise ValueError('Unrecognized constraint function %s' % name)

    return constraint_fn, constraint_fn_np, constraint_fn_inv, constraint_fn_inv_np


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                initializer_name, _, initializer_params = parse_initializer.match(initializer).groups()

                kwargs = {}
                if initializer_params:
                    kwarg_list = initializer_params.split('-')
                    for kwarg in kwarg_list:
                        key, val = kwarg.split('=')
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        kwargs[key] = val

                if 'identity' in initializer_name:
                    return tf.keras.initializers.Identity
                elif 'he_' in initializer_name:
                    return tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='normal')
                else:
                    out = getattr(tf, initializer_name)
                    if 'glorot' in initializer:
                        out = out()
                    else:
                        out = out(**kwargs)
            else:
                out = initializer

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None and isinstance(init, str) and '_' in init:
                try:
                    init_split = init.split('_')
                    scale = float(init_split[-1])
                    init = '_'.join(init_split[:-1])
                except ValueError:
                    pass

            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                out = init

            return out


def get_dropout(
        rate,
        training=True,
        use_MAP_mode=True,
        rescale=True,
        noise_shape=None,
        name=None,
        constant=None,
        reuse=tf.AUTO_REUSE,
        session=None
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if rate:
                out = DropoutLayer(
                    rate,
                    noise_shape=noise_shape,
                    training=training,
                    use_MAP_mode=use_MAP_mode,
                    rescale=rescale,
                    constant=constant,
                    name=name,
                    reuse=reuse,
                    session=session
                )
            else:
                out = lambda x: x

            return out


def get_random_variable(
        name,
        shape,
        sd_posterior,
        init=None,
        constraint=None,
        sd_prior=None,
        training=None,
        use_MAP_mode=None,
        epsilon=1e-8,
        session=None
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():

            if constraint is None:
                constraint = 'softplus'
            constraint_fn, \
            constraint_fn_np, \
            constraint_fn_inv, \
            constraint_fn_inv_np = get_constraint(constraint)

            if init is None:
                init = 0.
                loc_initializer = tf.zeros_initializer()
            else:
                loc_initializer = tf.constant_initializer(init, dtype=tf.float32)
            scale_initializer = tf.constant_initializer(
                np.ones(shape) * constraint_fn_inv_np(sd_posterior),
                dtype=tf.float32
            )
            init_np = init
            init = tf.constant(init, dtype=tf.float32)

            if sd_prior is None:
                sd_prior_np = None
            else:
                sd_prior_np = sd_prior
                sd_prior = tf.constant(sd_prior, dtype=tf.float32)

            if use_MAP_mode is None:
                use_MAP_mode = tf.logical_not(training)

            kl_penalties = {}

            # Posterior distribution
            v_q_loc = tf.get_variable(
                name='%s_q_loc' % name,
                initializer=loc_initializer,
                shape=shape
            )
            v_q_scale = tf.get_variable(
                name='%s_q_scale' % name,
                initializer=scale_initializer,
                shape=shape
            )
            v_q_dist = Normal(
                loc=v_q_loc,
                scale=constraint_fn(v_q_scale) + epsilon,
                name='%s_q' % name
            )

            # Prior distribution
            if sd_prior is None:
                v_prior_dist = None
            else:
                v_prior_dist = Normal(
                    loc=init,
                    scale=sd_prior,
                    name=name
                )
                kl_penalties[v_prior_dist.name] = {
                    'loc': np.array(init_np).flatten(),
                    'scale': np.array(sd_prior_np).flatten(),
                    'val': v_q_dist.kl_divergence(v_prior_dist)
                }

            v_eval_sample = v_q_dist.sample()
            v_eval = tf.get_variable(
                name='%s_sample' % name,
                initializer=tf.zeros_initializer(),
                shape=v_eval_sample.shape,
                dtype=tf.float32,
                trainable=False
            )
            v_eval_resample = tf.assign(v_eval, v_eval_sample)
            v = tf.cond(
                training,
                v_q_dist.sample,
                lambda: tf.cond(
                    use_MAP_mode,
                    v_q_dist.mean,
                    lambda: v_eval
                )
            )
            
            assert v_q_loc.shape == shape, 'v_q_loc.shape should be %s, saw %s.' % (shape, v_q_loc.shape)
            assert v_q_scale.shape == shape, 'v_q_scale.shape should be %s, saw %s.' % (shape, v_q_scale.shape)
            assert v_q_dist.batch_shape == shape, 'v_q_dist.shape should be %s, saw %s.' % (shape, v_q_dist.batch_shape)
            if v_prior_dist is not None:
                assert v_prior_dist.batch_shape == shape, 'v_prior_dist.shape should be %s, saw %s.' % (shape, v_prior_dist.batch_shape)
            for k in kl_penalties:
                assert kl_penalties[k]['val'].shape == shape, "kl_penalties['%s']['val'].shape should be %s, saw %s." % (k, shape, kl_penalties[k]['val'].shape)
            assert v_eval_sample.shape == shape, 'v_eval_sample.shape should be %s, saw %s.' % (shape, v_eval_sample.shape)
            assert v_eval.shape == shape, 'v_eval.shape should be %s, saw %s.' % (shape, v_eval.shape)
            assert v.shape == shape, 'v.shape should be %s, saw %s.' % (shape, v.shape)

            return {
                'v_q_loc': v_q_loc,
                'v_q_scale': v_q_scale,
                'v_q_dist': v_q_dist,
                'v_prior_dist': v_prior_dist,
                'v_eval_sample': v_eval_sample,
                'v_eval': v_eval,
                'v_eval_resample': v_eval_resample,
                'v': v,
                'kl_penalties': kl_penalties,
            }


def compose_lambdas(lambdas):
    def composed_lambdas(x, **kwargs):
        out = x
        for l in lambdas:
            out = l(out, **kwargs)
        return out

    return composed_lambdas


def make_lambda(layer, session=None, multi_arg=False, use_kwargs=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if multi_arg:
                if use_kwargs:
                    def apply_layer(*x, **kwargs):
                        return layer(*x, **kwargs)
                else:
                    def apply_layer(*x, **kwargs):
                        return layer(*x)
            else:
                if use_kwargs:
                    def apply_layer(x, **kwargs):
                        return layer(x, **kwargs)
                else:
                    def apply_layer(x, **kwargs):
                        return layer(x)
            return apply_layer


def matmul(A, B, session=None):
    """
    Matmul operation that supports broadcasting

    :param A: Left tensor
    :param B: Right tensor
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: Broadcasted matrix multiplication
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            assert len(A.shape) == len(B.shape), 'A and B must have the same rank. Got %s and %s.' % (len(A.shape), len(B.shape))
            A_shape = tf.shape(A)[:-2]
            B_shape = tf.shape(B)[:-2]
            A_tile = tf.concat([tf.maximum(B_shape // A_shape, 1), [1, 1]], 0)
            B_tile = tf.concat([tf.maximum(A_shape // B_shape, 1), [1, 1]], 0)

            A = tf.tile(A, A_tile)
            B = tf.tile(B, B_tile)

            C = tf.matmul(A, B)

            return C


def interpolated_integral(loc, val, mask=None, axis=0, session=None):
    assert isinstance(axis, int), 'axis must be a scalar integer'
    assert len(loc.shape) == len(val.shape), 'loc and val must be compatibly shaped. Got loc=%s, val=%s' % (loc.shape, val.shape)
    if mask is not None:
        assert len(loc.shape) == len(mask.shape), 'loc and mask must be compatibly shaped. Got loc=%s, mask=%s' % (loc.shape, mask.shape)
    ndim = len(loc.shape)
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if axis < 0:
                _axis = ndim + axis
            else:
                _axis = axis
            assert 0 <= _axis < ndim, 'Attempted to integrate over axis %d, but array only has %d dimensions.' % (
            axis, ndim)
            axis = _axis

            ntime = tf.shape(loc)[axis]

            def fnA(val=val, mask=mask, axis=axis):
                return tf.reduce_sum(val * mask, axis=axis)

            def fnB(loc=loc, val=val, mask=mask, axis=axis):
                sliceA = [slice(None, None) for _ in range(axis)] + [slice(None, -1)]
                sliceB = [slice(None, None) for _ in range(axis)] + [slice(1, None)]

                locA = loc[sliceA]
                locB = loc[sliceB]
                stepsize = locB - locA

                valA = val[sliceA]
                valB = val[sliceB]
                valInterp = (valA + valB) / 2

                integral = valInterp * stepsize
                if mask is not None:
                    mask = mask[sliceB]
                    integral = integral * mask
                integral = tf.reduce_sum(integral, axis=axis)

                return integral

            return tf.cond(ntime > 1, fnB, fnA)


# IRF Lambdas (response dimension is always last)

def empirical_integral(irf, session=None):
    def f(x, irf=irf, n_time_points=1000, session=session):
        session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                out = irf(x)
                support = tf.linspace(0., x, n_time_points)
                step_size = x / n_time_points
                while len(support.shape) < len(out.shape):
                    support = support[None, ...]
                irf_samples = irf(support)
                out = (tf.reduce_sum(irf_samples, axis=-2, keepdims=True)) * step_size

                return out

    return f


def exponential_irf_factory(
        beta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate an exponential impulse response function (IRF).

    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            beta = tf.convert_to_tensor(beta)

            def pdf(x, beta=beta):
                return beta * tf.exp(-beta * x)

            def cdf(x, beta=beta):
                return 1 - tf.exp(-beta * x)

            if support_ub is None:
                def irf(x, pdf=pdf):
                    return pdf(x)
            else:
                norm_const = cdf(support_ub)

                def irf(x, pdf=pdf, norm_const=norm_const, epsilon=epsilon):
                    # Ensure proper broadcasting
                    while len(x.shape) > len(norm_const.shape):
                        norm_const = norm_const[None, ...]
                    return pdf(x) / (norm_const + epsilon)

            return irf


def gamma_irf_factory(
        alpha,
        beta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a gamma impulse response function (IRF).

    :param alpha: TF tensor; alpha (shape) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :param validate_irf_args: ``bool``; whether to validate any constraints on the IRF parameters.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = tf.convert_to_tensor(alpha)
            beta = tf.convert_to_tensor(beta)

            dist = tf.contrib.distributions.Gamma(
                concentration=alpha,
                rate=beta,
                validate_args=validate_irf_args
            )
            pdf = dist.prob
            cdf = dist.cdf

            if support_ub is None:
                def irf(x, pdf=pdf, epsilon=epsilon):
                    return pdf(x + epsilon)

            else:
                norm_const = cdf(support_ub)

                def irf(x, pdf=pdf, norm_const=norm_const, epsilon=epsilon):
                    # Ensure proper broadcasting
                    while len(x.shape) > len(norm_const.shape):
                        norm_const = norm_const[None, ...]
                    return pdf(x + epsilon) / (norm_const + epsilon)

            return irf


def shifted_gamma_irf_factory(
        alpha,
        beta,
        delta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a gamma impulse response function (IRF) with an additional shift parameter.

    :param alpha: TF tensor; alpha (shape) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param delta: TF tensor; delta (shift) parameter (delta < 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :param validate_irf_args: ``bool``; whether to validate any constraints on the IRF parameters.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = tf.convert_to_tensor(alpha)
            beta = tf.convert_to_tensor(beta)
            delta = tf.convert_to_tensor(delta)

            dist = tf.contrib.distributions.Gamma(
                concentration=alpha,
                rate=beta,
                validate_args=validate_irf_args
            )

            pdf = dist.prob
            cdf = dist.cdf
            cdf_0 = cdf(-delta)

            if support_ub is None:
                ub = 1.
            else:
                ub = cdf(support_ub - delta)

            norm_const = ub - cdf_0

            def irf(x, pdf=pdf, delta=delta, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(delta.shape):
                    delta = delta[None, ...]
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return pdf(x - delta) / (norm_const + epsilon)

            return irf


def normal_irf_factory(
        mu,
        sigma,
        support_ub=None,
        support_lb=0.,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate a normal impulse response function (IRF).

    :param mu: TF tensor; mu (location) parameter.
    :param sigma: TF tensor; sigma (scale) parameter (sigma > 0).
    :param causal: ``bool``; whether to assume causality (x >= 0) in the normalization.
    :param support_lb: ``tensor``, ``float`` or ``None``; lower bound on the IRF's support. If ``None``, lower bound set to 0.
    :param support_lb: ``tensor``, ``float`` or ``None``; lower bound on the IRF's support. If ``None``, no lower bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = tf.convert_to_tensor(mu)
            sigma = tf.convert_to_tensor(sigma)

            dist = tf.contrib.distributions.Normal(
                mu,
                sigma
            )

            pdf = dist.prob
            cdf = dist.cdf

            if support_lb is None:
                lb = 0.
            else:
                lb = cdf(support_lb)
            if support_ub is None:
                ub = 1.
            else:
                ub = cdf(support_ub)

            norm_const = ub - lb

            def irf(x, pdf=pdf, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return pdf(x) / (norm_const + epsilon)

            return irf


def skew_normal_irf_factory(
        mu,
        sigma,
        alpha,
        support_lb=0.,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate a skew-normal impulse response function (IRF).

    :param mu: TF tensor; mu (location) parameter.
    :param sigma: TF tensor; sigma (scale) parameter (sigma > 0).
    :param alpha: TF tensor; alpha (skew) parameter.
    :param causal: ``bool``; whether to assume causality (x >= 0) in the normalization.
    :param support_lb: ``tensor``, ``float`` or ``None``; lower bound on the IRF's support. If ``None``, lower bound set to 0.
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = tf.convert_to_tensor(mu)
            sigma = tf.convert_to_tensor(sigma)
            alpha = tf.convert_to_tensor(alpha)

            stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
            stdnorm_pdf = stdnorm.prob
            stdnorm_cdf = stdnorm.cdf

            def irf_base(x, mu=mu, sigma=sigma, alpha=alpha, pdf=stdnorm_pdf, cdf=stdnorm_cdf):
                # Ensure proper broadcasting
                while len(x.shape) > len(mu.shape):
                    mu = mu[None, ...]
                while len(x.shape) > len(sigma.shape):
                    sigma = sigma[None, ...]
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                return (pdf((x - mu) / (sigma)) * cdf(alpha * (x - mu) / (sigma)))

            cdf = empirical_integral(irf_base, session=session)

            if support_lb is None:
                lb = 0.
            else:
                lb = cdf(support_lb)
            if support_ub is None:
                ub = 1.
            else:
                ub = cdf(support_ub)

            norm_const = ub - lb

            def irf(x, irf_base=irf_base, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return irf_base(x) / (norm_const + epsilon)

            return irf


def emg_irf_factory(
        mu,
        sigma,
        beta,
        support_lb=0.,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate an exponentially-modified Gaussian (EMG) impulse response function (IRF).

    :param mu: TF tensor; mu (location) parameter.
    :param sigma: TF tensor; sigma (scale) parameter (sigma > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_lb: ``tensor``, ``float`` or ``None``; lower bound on the IRF's support. If ``None``, lower bound set to 0.
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = tf.convert_to_tensor(mu)
            sigma = tf.convert_to_tensor(sigma)
            beta = tf.convert_to_tensor(beta)

            def cdf(x):
                return tf.contrib.distributions.Normal(
                    loc=0.,
                    scale=beta * sigma
                )(beta * (x - mu))

            if support_lb is None:
                lb = 0.
            else:
                lb = cdf(support_lb)
            if support_ub is None:
                ub = 1.
            else:
                ub = cdf(support_ub)

            norm_const = ub - lb

            def irf(x, mu=mu, sigma=sigma, beta=beta, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(mu.shape):
                    mu = mu[None, ...]
                while len(x.shape) > len(sigma.shape):
                    sigma = sigma[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return (beta / 2 * tf.exp(0.5 * beta * (2. * mu + beta * sigma ** 2. - 2. * x)) *
                        tf.erfc((mu + beta * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))) / (norm_const + epsilon)

            return irf


def beta_prime_irf_factory(
        alpha,
        beta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate a beta-prime impulse response function (IRF).

    :param alpha: TF tensor; alpha (rate) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = tf.convert_to_tensor(alpha)
            beta = tf.convert_to_tensor(beta)

            def cdf(x, alpha=alpha, beta=beta):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                return tf.betainc(alpha, beta, x / (1 + x)) * tf.exp(tf.lbeta(alpha, beta))

            if support_ub is None:
                ub = 1
            else:
                ub = cdf(support_ub)

            cdf_0 = cdf(epsilon)

            norm_const = ub - cdf_0

            def irf(x, alpha=alpha, beta=beta, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return ((x + epsilon) ** (alpha - 1.) * (1. + (x + epsilon)) ** (-alpha - beta)) / (
                            norm_const + epsilon)

            return irf


def shifted_beta_prime_irf_factory(
        alpha,
        beta,
        delta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None
):
    """
    Instantiate a beta-prime impulse response function (IRF) with an additional shift parameter.

    :param alpha: TF tensor; alpha (rate) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param delta: TF tensor; delta (shift) parameter (delta < 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = tf.convert_to_tensor(alpha)
            beta = tf.convert_to_tensor(beta)
            delta = tf.convert_to_tensor(delta)

            def cdf(x, alpha=alpha, beta=beta, delta=delta):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                while len(x.shape) > len(delta.shape):
                    delta = delta[None, ...]
                return tf.betainc(alpha, beta, (x - delta) / (1 + x - delta)) * tf.exp(tf.lbeta(alpha, beta))

            if support_ub is None:
                ub = 1
            else:
                ub = cdf(support_ub - delta)

            cdf_0 = cdf(-delta)

            norm_const = ub - cdf_0

            def irf(x, alpha=alpha, beta=beta, delta=delta, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                while len(x.shape) > len(delta.shape):
                    delta = delta[None, ...]
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return ((x - delta) ** (alpha - 1) * (1 + (x - delta)) ** (-alpha - beta)) / (norm_const + epsilon)

            return irf


def double_gamma_1_irf_factory(
        beta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a double-gamma hemodynamic response function (HRF) for fMRI with one trainable parameter.

    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = 6.
            alpha_undershoot = 16.
            beta_main = beta
            beta_undershoot = beta
            c = 1. / 6.

            return double_gamma_5_irf_factory(
                alpha_main=alpha_main,
                alpha_undershoot=alpha_undershoot,
                beta_main=beta_main,
                beta_undershoot=beta_undershoot,
                c=c,
                support_ub=support_ub,
                epsilon=epsilon,
                session=session,
                validate_irf_args=validate_irf_args
            )


def double_gamma_2_irf_factory(
        alpha,
        beta,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a double-gamma hemodynamic response function (HRF) for fMRI with two trainable parameters.

    :param alpha: TF tensor; alpha (shape) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = alpha
            alpha_undershoot = alpha + 10.
            beta_main = beta
            beta_undershoot = beta
            c = 1. / 6.

            return double_gamma_5_irf_factory(
                alpha_main=alpha_main,
                alpha_undershoot=alpha_undershoot,
                beta_main=beta_main,
                beta_undershoot=beta_undershoot,
                c=c,
                support_ub=support_ub,
                epsilon=epsilon,
                session=session,
                validate_irf_args=validate_irf_args
            )


def double_gamma_3_irf_factory(
        alpha,
        beta,
        c,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a double-gamma hemodynamic response function (HRF) for fMRI with three trainable parameters.

    :param alpha: TF tensor; alpha (shape) parameter (alpha > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param c: TF tensor; c parameter (amplitude of undershoot).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = alpha
            alpha_undershoot = alpha + 10.
            beta_main = beta
            beta_undershoot = beta

            return double_gamma_5_irf_factory(
                alpha_main=alpha_main,
                alpha_undershoot=alpha_undershoot,
                beta_main=beta_main,
                beta_undershoot=beta_undershoot,
                c=c,
                support_ub=support_ub,
                epsilon=epsilon,
                session=session,
                validate_irf_args=validate_irf_args
            )


def double_gamma_4_irf_factory(
        alpha_main,
        alpha_undershoot,
        beta,
        c,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a double-gamma hemodynamic response function (HRF) for fMRI with four trainable parameters.

    :param alpha_main: TF tensor; alpha (shape) parameter of peak response (alpha_main > 0).
    :param alpha_undershoot: TF tensor; alpha (shape) parameter of undershoot component (alpha_undershoot > 0).
    :param beta: TF tensor; beta (rate) parameter (beta > 0).
    :param c: TF tensor; c parameter (amplitude of undershoot).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            beta_main = beta
            beta_undershoot = beta

            return double_gamma_5_irf_factory(
                alpha_main=alpha_main,
                alpha_undershoot=alpha_undershoot,
                beta_main=beta_main,
                beta_undershoot=beta_undershoot,
                c=c,
                support_ub=support_ub,
                epsilon=epsilon,
                session=session,
                validate_irf_args=validate_irf_args
            )


def double_gamma_5_irf_factory(
        alpha_main,
        alpha_undershoot,
        beta_main,
        beta_undershoot,
        c,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        validate_irf_args=False
):
    """
    Instantiate a double-gamma hemodynamic response function (HRF) for fMRI with five trainable parameters.

    :param alpha_main: TF tensor; alpha (shape) parameter of peak response (alpha_main > 0).
    :param alpha_undershoot: TF tensor; alpha (shape) parameter of undershoot component (alpha_undershoot > 0).
    :param beta_main: TF tensor; beta (rate) parameter of peak response (beta_main > 0).
    :param beta_undershoot: TF tensor; beta (rate) parameter of undershoot component (beta_undershoot > 0).
    :param c: TF tensor; c parameter (amplitude of undershoot).
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = tf.convert_to_tensor(alpha_main)
            alpha_undershoot = tf.convert_to_tensor(alpha_undershoot)
            beta_main = tf.convert_to_tensor(beta_main)
            beta_undershoot = tf.convert_to_tensor(beta_undershoot)
            c = tf.convert_to_tensor(c)

            dist_main = tf.contrib.distributions.Gamma(
                concentration=alpha_main,
                rate=beta_main,
                validate_args=validate_irf_args
            )
            pdf_main = dist_main.prob
            cdf_main = dist_main.cdf
            dist_undershoot = tf.contrib.distributions.Gamma(
                concentration=alpha_undershoot,
                rate=beta_undershoot,
                validate_args=validate_irf_args
            )
            pdf_undershoot = dist_undershoot.prob
            cdf_undershoot = dist_undershoot.cdf

            if support_ub is None:
                norm_const = 1 - c
            else:
                norm_const = cdf_main(support_ub) - c * cdf_undershoot(support_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, c=c, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                while len(x.shape) > len(c.shape):
                    c = c[None, ...]
                while len(x.shape) > len(norm_const.shape):
                    norm_const = norm_const[None, ...]
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / (norm_const + epsilon)

            return irf


def LCG_irf_factory(
        bases,
        support_lb=0.,
        support_ub=None,
        epsilon=4 * np.finfo('float32').eps,
        session=None,
        **params
):
    """
    Instantiate a linear combination of Gaussians (LCG) impulse response function (IRF).

    :param bases: ``int``; number of basis kernels in LCG.
    :param support_lb: ``tensor``, ``float`` or ``None``; lower bound on the IRF's support. If ``None``, lower bound set to 0.
    :param support_ub: ``tensor``, ``float`` or ``None``; upper bound on the IRF's support. If ``None``, no upper bound.
    :param epsilon: ``float``; additive constant for numerical stability in the normalization term.
    :param session: TF ``session`` object; the graph's TensorFlow session.
    :param **params: ``tensors``; the LCG parameters. Must have ``3*bases`` total parameters with the following keyword names: x1, ..., xN, y1, ... yN, s1, ..., sN, where N stands for the value of **bases** and x, y, and s parameters respectively encode location, amplitude, and scale of the corresponding basis kernel.
    :return: ``function``; the IRF
    """

    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            # Build kernel locations
            params = {x: tf.convert_to_tensor(params[x]) for x in params}
            c = tf.stack([params['x%s' % i] for i in range(1, bases + 1)], -1)

            # Build kernel amplitudes
            v = tf.stack([params['y%s' % i] for i in range(1, bases + 1)], -1)

            # Build kernel widths
            b = tf.stack([params['s%s' % i] for i in range(1, bases + 1)], -1)

            dist = tf.contrib.distributions.Normal(
                loc=c,
                scale=b + epsilon,
            )
            pdf = dist.prob
            cdf = dist.cdf

            if support_lb is None:
                lb = 0.
            else:
                lb = cdf(support_lb)
            if support_ub is None:
                ub = 1.
            else:
                ub = cdf(support_ub)

            norm_const = tf.reduce_sum((ub - lb) * v, axis=-1)

            def irf(x, v=v, pdf=pdf, norm_const=norm_const, epsilon=epsilon):
                # Ensure proper broadcasting
                x = x[..., None]  # Add a summation axis
                while len(x.shape) > len(v.shape):
                    v = v[None, ...]
                while len(x.shape) - 1 > len(norm_const.shape):
                    norm_const = norm_const[None, ...]

                return tf.reduce_sum(pdf(x) * v, axis=-1) / (norm_const + epsilon)

            return irf


CDRNNStateTuple = collections.namedtuple(
    'AttentionalLSTMDecoderStateTuple',
    ' '.join(['c', 'h', 't'])
)


class BiasLayer(object):
    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        self.session = get_session(session)
        self.reuse = reuse
        self.epsilon = epsilon
        self.name = name

        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.use_MAP_mode = use_MAP_mode
                if not bool(rangf_map):
                    rangf_map = {}
                self.rangf_map = rangf_map

                self.built = False

    @property
    def regularizable_weights(self):
        return []

    def build(self, inputs_shape):
        if not self.built:
            units = inputs_shape[-1]

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.bias = tf.get_variable(
                            name='bias',
                            shape=[units],
                            initializer=tf.zeros_initializer(),
                        )
                        self.bias_ran = {}
                        for gf in self.rangf_map:
                            n_levels = self.rangf_map[gf][0] - 1
                            self.bias_ran[gf] = tf.get_variable(
                                name='bias_by_%s' % sn(gf),
                                initializer=tf.zeros_initializer(),
                                shape=[n_levels, units]
                            )

            self.built = True

    def __call__(self, inputs, kernel_offsets=None, bias_offsets=None):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                H = inputs
                bias = self.bias[None, ...] # Expand batch dim
                for gf in self.bias_ran:
                    Y_gf = self.rangf_map[gf][1]
                    bias_ran = self.bias_ran[gf]
                    bias_ran -= tf.reduce_mean(bias_ran, axis=0, keepdims=True)
                    bias_ran = tf.pad(bias_ran, [[0,1], [0,0]])
                    bias += tf.gather(bias_ran, Y_gf)
                while len(bias.shape) < len(H.shape):
                    bias = tf.expand_dims(bias, axis=1)
                if bias_offsets is not None:
                    bias = bias + bias_offsets
                H += bias

                return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return []

    def resample_ops(self):
        return []


class BiasLayerBayes(BiasLayer):
    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            rangf_map=None,
            declare_priors=False,
            sd_prior=1.,
            sd_init=None,
            posterior_to_prior_sd_ratio=1,
            ranef_to_fixef_prior_sd_ratio=0.1,
            constraint='softplus',
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        super(BiasLayerBayes, self).__init__(
            training=training,
            use_MAP_mode=use_MAP_mode,
            rangf_map=rangf_map,
            reuse=reuse,
            epsilon=epsilon,
            session=session,
            name=name
        )
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.declare_priors = declare_priors
                self.sd_prior = sd_prior
                self.sd_init = sd_init
                self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
                self.ranef_to_fixef_prior_sd_ratio = ranef_to_fixef_prior_sd_ratio

                self.constraint = constraint
                self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            units = inputs_shape[-1]
            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):

                        bias_sd_prior = get_numerical_sd(self.sd_prior, in_dim=1, out_dim=1)
                        if self.sd_init:
                            bias_sd_posterior = get_numerical_sd(self.sd_init, in_dim=1, out_dim=1)
                        else:
                            bias_sd_posterior = bias_sd_prior * self.posterior_to_prior_sd_ratio
                        _bias_sd_prior = np.ones([units]) * bias_sd_prior
                        _bias_sd_posterior = np.ones([units]) * bias_sd_posterior

                        rv_dict = get_random_variable(
                            'bias',
                            [units],
                            _bias_sd_posterior,
                            constraint=self.constraint,
                            sd_prior=_bias_sd_prior,
                            training=self.training,
                            use_MAP_mode=self.use_MAP_mode,
                            epsilon=self.epsilon,
                            session=self.session
                        )
                        if self.declare_priors:
                            self.kl_penalties_base.update(rv_dict['kl_penalties'])
                        self.bias_eval_resample = rv_dict['v_eval_resample']
                        self.bias = rv_dict['v']
                        self.bias_eval_resample_ran = {}
                        self.bias_ran = {}
                        for gf in self.rangf_map:
                            n_levels = self.rangf_map[gf][0] - 1
                            _bias_sd_prior = np.ones([n_levels, units]) * bias_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                            _bias_sd_posterior = np.ones([n_levels, units]) * bias_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                            rv_dict = get_random_variable(
                                'bias_by_%s' % sn(gf),
                                [n_levels, units],
                                _bias_sd_posterior,
                                constraint=self.constraint,
                                sd_prior=_bias_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.bias_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                            self.bias_ran[gf] = rv_dict['v']

            self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                return self.kl_penalties_base.copy()

    def resample_ops(self):
        out = super(BiasLayerBayes, self).resample_ops()
        if self.built:
            out.append(self.bias_eval_resample)
            for gf in self.rangf_map:
                out.append(self.bias_eval_resample_ran[gf])

        return out


class DenseLayer(object):
    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            units=None,
            use_bias=True,
            activation=None,
            kernel_sd_init='he',
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            normalize_after_activation=False,
            shift_normalized_activations=True,
            rescale_normalized_activations=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        self.session = get_session(session)
        self.reuse = reuse
        self.epsilon = epsilon
        self.name = name

        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.use_MAP_mode = use_MAP_mode
                self.units = units
                self.use_bias = use_bias
                self.activation = get_activation(activation, session=self.session, training=self.training)
                self.kernel_sd_init = kernel_sd_init
                self.use_dropout = bool(dropout)
                self.dropout_layer = get_dropout(
                    dropout,
                    training=self.training,
                    use_MAP_mode=self.use_MAP_mode,
                    name='dropout',
                    reuse=self.reuse,
                    session=self.session
                )
                self.maxnorm = maxnorm

                self.batch_normalization_decay = batch_normalization_decay
                self.use_batch_normalization = bool(self.batch_normalization_decay)

                self.layer_normalization_type = layer_normalization_type
                if layer_normalization_type is None:
                    self.layer_normalization_type = layer_normalization_type
                elif layer_normalization_type.lower() == 'z':
                    self.layer_normalization_type = 'z'
                elif layer_normalization_type.lower() == 'length':
                    self.layer_normalization_type = 'length'
                else:
                    raise ValueError('Unrecognized layer normalization type: %s' % layer_normalization_type)
                self.use_layer_normalization = bool(self.layer_normalization_type)

                assert not (self.use_batch_normalization and self.use_layer_normalization), 'Cannot batch normalize and layer normalize the same layer.'
                self.normalize_activations = self.use_batch_normalization or self.use_layer_normalization

                self.normalize_after_activation = normalize_after_activation
                self.shift_normalized_activations = shift_normalized_activations
                self.rescale_normalized_activations = rescale_normalized_activations

                if not bool(rangf_map):
                    rangf_map = {}
                self.rangf_map = rangf_map

                self.normalization_beta = None
                self.normalization_gamma = None

                if batch_normalization_decay and dropout:
                    stderr('WARNING: Batch normalization and dropout are being applied simultaneously in layer %s.\n         This is usually not a good idea.')

                self.built = False

    @property
    def regularizable_weights(self):
        out = []
        if self.built:
            out.append(self.kernel)
            for gf in self.rangf_map:
                out.append(self.kernel_ran[gf])
                if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                    out.append(self.bias_ran[gf])

        return out

    def build(self, inputs_shape):
        if not self.built:
            in_dim = inputs_shape[-1]
            if self.units is None:
                out_dim = in_dim
            else:
                out_dim = self.units

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        sd = get_numerical_sd(self.kernel_sd_init, in_dim=in_dim, out_dim=out_dim)

                        kernel_init = get_initializer(
                            'random_normal_initializer_mean=0-stddev=%s' % sd,
                            session=self.session
                        )
                        self.kernel = tf.get_variable(
                            name='kernel',
                            initializer=kernel_init,
                            shape=[in_dim, out_dim]
                        )
                        self.kernel_ran = {}
                        for gf in self.rangf_map:
                            n_levels = self.rangf_map[gf][0] - 1
                            self.kernel_ran[gf] = tf.get_variable(
                                name='kernel_by_%s' % sn(gf),
                                initializer=tf.zeros_initializer(),
                                shape=[n_levels, in_dim, out_dim]
                            )

                        if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                            self.bias = tf.get_variable(
                                name='bias',
                                shape=[out_dim],
                                initializer=tf.zeros_initializer(),
                            )
                            self.bias_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                self.bias_ran[gf] = tf.get_variable(
                                    name='bias_by_%s' % sn(gf),
                                    initializer=tf.zeros_initializer(),
                                    shape=[n_levels, out_dim]
                                )
                        #
                        # if self.use_dropout:
                        #     self.dropout_layer.build([x for x in inputs_shape[:-1]] + [out_dim])

                        if self.use_batch_normalization:
                            self.normalization_layer = BatchNormLayer(
                                decay=self.batch_normalization_decay,
                                shift_activations=self.shift_normalized_activations,
                                rescale_activations=self.rescale_normalized_activations,
                                axis=-1,
                                training=self.training,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )
                        elif self.use_layer_normalization:
                            self.normalization_layer = LayerNormLayer(
                                normalization_type=self.layer_normalization_type,
                                shift_activations=self.shift_normalized_activations,
                                rescale_activations=self.rescale_normalized_activations,
                                axis=-1,
                                training=self.training,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )

            self.built = True

    def __call__(self, inputs, kernel_offsets=None, bias_offsets=None):
        assert len(inputs.shape) > 1, 'inputs must have a batch dim'
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                if not self.name:
                    name = ''
                else:
                    name = self.name
                with tf.variable_scope(name, reuse=self.reuse):
                    if len(inputs.shape) == 2:
                        inputs = tf.expand_dims(inputs, axis=1) # Ensure batch dim
                        squeeze = True
                    else:
                        squeeze = False

                    kernel = self.kernel[None, ...] # Expand batch dim
                    for gf in self.kernel_ran:
                        Y_gf = self.rangf_map[gf][1]
                        kernel_ran = self.kernel_ran[gf]
                        kernel_ran -= tf.reduce_mean(kernel_ran, axis=0, keepdims=True)
                        kernel_ran = tf.pad(kernel_ran, [[0,1], [0,0], [0,0]])
                        kernel += tf.gather(kernel_ran, Y_gf)
                    while len(kernel.shape) < len(inputs.shape):
                        kernel = tf.expand_dims(kernel, axis=1)
                    if kernel_offsets is not None:
                        while len(kernel.shape) < len(kernel_offsets.shape):
                            kernel = tf.expand_dims(kernel, axis=1)
                        while len(kernel_offsets.shape) < len(kernel.shape):
                            kernel_offsets = tf.expand_dims(kernel_offsets, axis=1)
                        kernel = kernel + kernel_offsets
                    if self.maxnorm:
                        kernel = tf.clip_by_norm(kernel, self.maxnorm, axes=[0])
                    if (len(kernel.shape) - len(inputs.shape)) == 1:
                        inputs = tf.expand_dims(inputs, axis=-2)
                        squeeze = True
                    H = matmul(inputs, kernel)
                    if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                        bias = self.bias[None, ...] # Expand batch dim
                        for gf in self.bias_ran:
                            Y_gf = self.rangf_map[gf][1]
                            bias_ran = self.bias_ran[gf]
                            bias_ran -= tf.reduce_mean(bias_ran, axis=0, keepdims=True)
                            bias_ran = tf.pad(bias_ran, [[0,1], [0,0]])
                            bias += tf.gather(bias_ran, Y_gf)
                        while len(bias.shape) < len(H.shape):
                            bias = tf.expand_dims(bias, axis=1)
                        if bias_offsets is not None:
                            bias = bias + bias_offsets
                        H += bias

                    if self.activation is not None and self.normalize_after_activation:
                        H = self.activation(H)

                    if self.normalize_activations:
                        H = self.normalization_layer(H)

                    if self.activation is not None and not self.normalize_after_activation:
                        H = self.activation(H)

                    H = self.dropout_layer(H)

                    if squeeze:
                        H = tf.squeeze(H, axis=-2)

                    return H

    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        out = []
        if self.use_batch_normalization and self.built:
            out += self.normalization_layer.ema_ops()

        return out

    def resample_ops(self):
        out = []
        if self.use_dropout and self.built:
            out.append(self.dropout_layer.resample_ops())

        return out


class DenseLayerBayes(DenseLayer):
    def __init__(
            self,
            training=False,
            use_MAP_mode=True,
            units=None,
            use_bias=True,
            activation=None,
            dropout=None,
            maxnorm=None,
            batch_normalization_decay=None,
            layer_normalization_type=None,
            normalize_after_activation=False,
            shift_normalized_activations=True,
            rescale_normalized_activations=True,
            rangf_map=None,
            declare_priors_weights=True,
            declare_priors_biases=False,
            declare_priors_gamma=False,
            kernel_sd_prior=1,
            kernel_sd_init=None,
            bias_sd_prior=1.,
            bias_sd_init=None,
            gamma_sd_prior=1.,
            gamma_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            ranef_to_fixef_prior_sd_ratio=0.1,
            constraint='softplus',
            reuse=tf.AUTO_REUSE,
            epsilon=1e-5,
            session=None,
            name=None
    ):
        super(DenseLayerBayes, self).__init__(
            training=training,
            use_MAP_mode=use_MAP_mode,
            units=units,
            use_bias=use_bias,
            activation=activation,
            kernel_sd_init=kernel_sd_init,
            dropout=dropout,
            maxnorm=maxnorm,
            batch_normalization_decay=batch_normalization_decay,
            layer_normalization_type=layer_normalization_type,
            normalize_after_activation=normalize_after_activation,
            shift_normalized_activations=shift_normalized_activations,
            rescale_normalized_activations=rescale_normalized_activations,
            rangf_map=rangf_map,
            reuse=reuse,
            epsilon=epsilon,
            session=session,
            name=name
        )
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.declare_priors_weights = declare_priors_weights
                self.declare_priors_biases = declare_priors_biases
                self.declare_priors_gamma = declare_priors_gamma
                self.kernel_sd_prior = kernel_sd_prior
                self.bias_sd_prior = bias_sd_prior
                self.bias_sd_init = bias_sd_init
                self.gamma_sd_prior = gamma_sd_prior
                self.gamma_sd_init = gamma_sd_init
                self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
                self.ranef_to_fixef_prior_sd_ratio = ranef_to_fixef_prior_sd_ratio

                self.constraint = constraint
                self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            in_dim = inputs_shape[-1]
            if self.units is None:
                out_dim = in_dim
            else:
                out_dim = self.units

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        kernel_sd_prior = get_numerical_sd(self.kernel_sd_prior, in_dim=in_dim, out_dim=out_dim)
                        if self.kernel_sd_init:
                            kernel_sd_posterior = get_numerical_sd(self.kernel_sd_init, in_dim=in_dim, out_dim=out_dim)
                        else:
                            kernel_sd_posterior = kernel_sd_prior * self.posterior_to_prior_sd_ratio
                        _kernel_sd_prior = np.ones([in_dim, out_dim]) * kernel_sd_prior
                        _kernel_sd_posterior = np.ones([in_dim, out_dim]) * kernel_sd_posterior

                        if self.use_MAP_mode is None:
                            self.use_MAP_mode = tf.logical_not(self.training)

                        rv_dict = get_random_variable(
                            'kernel',
                            [in_dim, out_dim],
                            _kernel_sd_posterior,
                            constraint=self.constraint,
                            sd_prior=_kernel_sd_prior,
                            training=self.training,
                            use_MAP_mode=self.use_MAP_mode,
                            epsilon=self.epsilon,
                            session=self.session
                        )
                        if self.declare_priors_weights:
                            self.kl_penalties_base.update(rv_dict['kl_penalties'])
                        self.kernel_eval_resample = rv_dict['v_eval_resample']
                        self.kernel = rv_dict['v']
                        self.kernel_eval_resample_ran = {}
                        self.kernel_ran = {}
                        for gf in self.rangf_map:
                            n_levels = self.rangf_map[gf][0] - 1
                            _kernel_sd_prior = np.ones([n_levels, in_dim, out_dim]) * kernel_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                            _kernel_sd_posterior = np.ones([n_levels, in_dim, out_dim]) * kernel_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                            rv_dict = get_random_variable(
                                'kernel_by_%s' % sn(gf),
                                [n_levels, in_dim, out_dim],
                                _kernel_sd_posterior,
                                constraint=self.constraint,
                                sd_prior=_kernel_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_weights:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.kernel_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                            self.kernel_ran[gf] = rv_dict['v']

                        if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                            bias_sd_prior = get_numerical_sd(self.bias_sd_prior, in_dim=1, out_dim=1)
                            if self.bias_sd_init:
                                bias_sd_posterior = get_numerical_sd(self.bias_sd_init, in_dim=1, out_dim=1)
                            else:
                                bias_sd_posterior = bias_sd_prior * self.posterior_to_prior_sd_ratio
                            _bias_sd_prior = np.ones([out_dim]) * bias_sd_prior
                            _bias_sd_posterior = np.ones([out_dim]) * bias_sd_posterior

                            # Posterior distribution
                            rv_dict = get_random_variable(
                                'bias',
                                [out_dim],
                                _bias_sd_posterior,
                                constraint=self.constraint,
                                sd_prior=_bias_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_biases:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.bias_eval_resample = rv_dict['v_eval_resample']
                            self.bias = rv_dict['v']
                            self.bias_eval_resample_ran = {}
                            self.bias_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                _bias_sd_prior = np.ones([n_levels, out_dim]) * bias_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                                _bias_sd_posterior = np.ones([n_levels, out_dim]) * bias_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                                rv_dict = get_random_variable(
                                    'bias_by_%s' % sn(gf),
                                    [n_levels, out_dim],
                                    _bias_sd_posterior,
                                    constraint=self.constraint,
                                    sd_prior=_bias_sd_prior,
                                    training=self.training,
                                    use_MAP_mode=self.use_MAP_mode,
                                    epsilon=self.epsilon,
                                    session=self.session
                                )
                                if self.declare_priors_weights:
                                    self.kl_penalties_base.update(rv_dict['kl_penalties'])
                                self.bias_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                                self.bias_ran[gf] = rv_dict['v']

                        # if self.use_dropout:
                        #     self.dropout_layer.build([x for x in inputs_shape[:-1]] + [out_dim])

                        if self.use_batch_normalization:
                            self.normalization_layer = BatchNormLayerBayes(
                                decay=self.batch_normalization_decay,
                                shift_activations=self.shift_normalized_activations,
                                rescale_activations=self.rescale_normalized_activations,
                                axis=-1,
                                training=self.training,
                                rangf_map=self.rangf_map,
                                use_MAP_mode=self.use_MAP_mode,
                                declare_priors_scale=self.declare_priors_gamma,
                                declare_priors_shift=self.declare_priors_biases,
                                scale_sd_prior=self.gamma_sd_prior,
                                scale_sd_init=self.gamma_sd_init,
                                shift_sd_prior=self.bias_sd_prior,
                                shift_sd_init=self.bias_sd_init,
                                posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                                ranef_to_fixef_prior_sd_ratio=self.ranef_to_fixef_prior_sd_ratio,
                                constraint=self.constraint,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )
                        elif self.use_layer_normalization:
                            self.normalization_layer = LayerNormLayerBayes(
                                normalization_type=self.layer_normalization_type,
                                shift_activations=self.shift_normalized_activations,
                                rescale_activations=self.rescale_normalized_activations,
                                axis=-1,
                                training=self.training,
                                rangf_map=self.rangf_map,
                                use_MAP_mode=self.use_MAP_mode,
                                declare_priors_scale=self.declare_priors_gamma,
                                declare_priors_shift=self.declare_priors_biases,
                                scale_sd_prior=self.gamma_sd_prior,
                                scale_sd_init=self.gamma_sd_init,
                                shift_sd_prior=self.bias_sd_prior,
                                shift_sd_init=self.bias_sd_init,
                                posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                                constraint=self.constraint,
                                epsilon=self.epsilon,
                                session=self.session,
                                reuse=self.reuse,
                                name=self.name
                            )

            self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                out = self.kl_penalties_base.copy()
                if self.batch_normalization_decay:
                    out.update(self.normalization_layer.kl_penalties())

                return out

    def resample_ops(self):
        out = super(DenseLayerBayes, self).resample_ops()
        if self.built:
            out.append(self.kernel_eval_resample)
            for gf in self.rangf_map:
                out.append(self.kernel_eval_resample_ran[gf])
            if self.use_bias and (not self.normalize_activations or self.normalize_after_activation):
                out.append(self.bias_eval_resample)
                for gf in self.rangf_map:
                    out.append(self.bias_eval_resample_ran[gf])
            if self.use_batch_normalization or self.use_layer_normalization:
                out += self.normalization_layer.resample_ops()

        return out


class RNNCell(LayerRNNCell):
    def __init__(
            self,
            units,
            training=False,
            use_MAP_mode=False,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='glorot_normal',
            recurrent_kernel_sd_init='glorot_normal',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self._session = get_session(session)
        with self._session.as_default():
            with self._session.graph.as_default():
                super(RNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                self._num_units = units
                self._training = training
                self.use_MAP_mode = use_MAP_mode

                self._kernel_depth = kernel_depth
                self._time_projection_depth = time_projection_depth
                self._prefinal_mode = prefinal_mode
                self._forget_bias = forget_bias
                self._forget_gate_as_irf = forget_gate_as_irf

                self._activation = get_activation(activation, session=self._session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self._session,
                                                            training=self._training)
                self._prefinal_activation = get_activation(prefinal_activation, session=self._session,
                                                           training=self._training)
                self._time_projection_inner_activation = get_activation(time_projection_inner_activation,
                                                                        session=self._session, training=self._training)

                self._bottomup_kernel_sd_init = bottomup_kernel_sd_init
                self._recurrent_kernel_sd_init = recurrent_kernel_sd_init

                self.use_dropout = bool(bottomup_dropout or h_dropout or c_dropout)
                self._bottomup_dropout_rate = bottomup_dropout
                self._bottomup_dropout_layer = get_dropout(
                    bottomup_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/bottomup_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._h_dropout_rate = h_dropout
                self._h_dropout_layer = get_dropout(
                    h_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/h_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._c_dropout_rate = c_dropout
                self._c_dropout_layer = get_dropout(
                    c_dropout,
                    training=self._training,
                    use_MAP_mode=self.use_MAP_mode,
                    name=self.name + '/c_dropout',
                    reuse=self._reuse,
                    session=self._session
                )
                self._forget_rate = forget_rate

                self._weight_normalization = weight_normalization
                self._layer_normalization = layer_normalization
                self._rangf_map = rangf_map
                self._use_bias = use_bias
                self._global_step = global_step

                self._l2_normalize_states = l2_normalize_states

                self._epsilon = epsilon

                self.built = False

    @property
    def state_size(self):
        return CDRNNStateTuple(c=self._num_units, h=self._num_units, t=1)

    @property
    def output_size(self):
        return CDRNNStateTuple(c=self._num_units, h=self._num_units, t=1)

    def initialize_kernel(
            self,
            in_dim,
            out_dim,
            kernel_type='bottomup',
            depth=None,
            inner_activation=None,
            prefinal_mode=None,
            name=None
    ):
        if kernel_type.lower() == 'recurrent':
            kernel_sd_init = self._recurrent_kernel_sd_init
        else:
            kernel_sd_init = self._bottomup_kernel_sd_init
        with self._session.as_default():
            with self._session.graph.as_default():
                kernel_lambdas = []
                if depth is None:
                    depth = self._kernel_depth
                if prefinal_mode is None:
                    prefinal_mode = self._prefinal_mode

                if prefinal_mode.lower() == 'max':
                    if out_dim > in_dim:
                        prefinal_dim = out_dim
                    else:
                        prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'in':
                    prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'out':
                    prefinal_dim = out_dim
                else:
                    raise ValueError('Unrecognized value for prefinal_mode: %s.' % prefinal_mode)

                layers = []

                for d in range(depth):
                    if d == depth - 1:
                        activation = None
                        units = out_dim
                        use_bias = False
                    else:
                        if inner_activation is None:
                            activation = self._prefinal_activation
                        else:
                            activation = inner_activation
                        units = prefinal_dim
                        use_bias = self._use_bias

                    if self.name:
                        name_cur = self.name + '/'
                    else:
                        name_cur = ''

                    if name:
                        name_cur += name + '_d%d' % d
                    else:
                        name_cur += 'd%d' % d

                    kernel_layer = DenseLayer(
                        training=self._training,
                        units=units,
                        rangf_map=self._rangf_map,
                        use_bias=use_bias,
                        kernel_sd_init=kernel_sd_init,
                        activation=activation,
                        epsilon=self._epsilon,
                        session=self._session,
                        reuse=self._reuse,
                        name=name_cur
                    )

                    kernel_layer.build([None, in_dim])

                    layers.append(kernel_layer)
                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel, layers

    @property
    def regularizable_weights(self):
        weights = [self._bias]
        weights += sum([x.regularizable_weights for x in self._kernel_bottomup_layers], [])
        weights += sum([x.regularizable_weights for x in self._kernel_recurrent_layers], [])
        if self._forget_gate_as_irf:
            weights += sum([x.regularizable_weights for x in self._kernel_time_projection_layers], []) + [
                self.t_delta_embedding_W]

        return weights[:]

    def build(self, inputs_shape):
        with self._session.as_default():
            with self._session.graph.as_default():
                if isinstance(inputs_shape, list):  # Has a mask
                    inputs_shape = inputs_shape[0]
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

                self._input_dims = inputs_shape[1].value
                bottomup_dim = self._input_dims
                recurrent_dim = self._num_units
                output_dim = self._num_units * 5  # forget (2x), input, and output gates, plus cell proposal
                self.layers = []

                # Build bias
                if not self._layer_normalization and self._use_bias:
                    self._bias = self.add_variable(
                        'bias',
                        shape=[1, output_dim],
                        initializer=tf.zeros_initializer()
                    )

                # Build LSTM kernels (bottomup and recurrent)
                self._kernel_bottomup, self._kernel_bottomup_layers = self.initialize_kernel(
                    bottomup_dim,
                    output_dim,
                    kernel_type='bottomup',
                    name='bottomup'
                )
                self.layers += self._kernel_bottomup_layers
                if self._bottomup_dropout_rate:
                    self._bottomup_dropout_layer.build(inputs_shape)

                self._kernel_recurrent, self._kernel_recurrent_layers = self.initialize_kernel(
                    recurrent_dim,
                    output_dim,
                    kernel_type='recurrent',
                    name='recurrent'
                )
                self.layers += self._kernel_recurrent_layers
                if self._h_dropout_rate:
                    self._h_dropout_layer.build([x for x in inputs_shape[:-1]] + [output_dim])
                if self._c_dropout_rate:
                    self._c_dropout_layer.build([x for x in inputs_shape[:-1]] + [output_dim])

                if self._forget_gate_as_irf:
                    self.initialize_irf_biases()

                    self._kernel_time_projection, self._kernel_time_projection_layers = self.initialize_kernel(
                        recurrent_dim,
                        recurrent_dim,
                        kernel_type='time_projection',
                        depth=self._time_projection_depth,
                        inner_activation=self._time_projection_inner_activation,
                        name='time_projection'
                    )
                    self.layers += self._kernel_time_projection_layers
                else:
                    self._kernel_time_projection = None
                    self._kernel_time_projection_layers = []

        self.built = True

    def initialize_irf_biases(self):
        with self._session.as_default():
            with self._session.graph.as_default():
                self.t_delta_embedding_W = self.add_variable(
                    't_delta_embedding_W',
                    shape=[1, self._num_units],
                    initializer=self._bottomup_initializer
                )
                self.t_delta_embedding_b = self.add_variable(
                    't_delta_embedding_b',
                    shape=[1, self._num_units],
                    initializer=tf.zeros_initializer()
                )

    def call(self, inputs, state):
        with self._session.as_default():
            with self._session.graph.as_default():
                assert isinstance(inputs,
                                  dict), 'Inputs to CDRNNCell must be a dict containing fields ``inputs`` and (optionally) ``times``, and ``mask``.'

                units = self._num_units
                c_prev = state.c
                h_prev = state.h
                t_prev = state.t

                if 'times' in inputs:
                    t_cur = inputs['times']
                else:
                    t_cur = t_prev

                if 'mask' in inputs:
                    mask = inputs['mask']
                else:
                    mask = None

                inputs = inputs['inputs']

                inputs = self._bottomup_dropout_layer(inputs)

                if self._forget_rate:
                    def train_fn_forget(h_prev=h_prev, c_prev=c_prev):
                        dropout_mask = tf.cast(tf.random_uniform(shape=[tf.shape(inputs)[0], 1]) > self._forget_rate,
                                               dtype=tf.float32)

                        h_prev_out = h_prev * dropout_mask
                        # c_prev_out = c_prev
                        c_prev_out = c_prev * dropout_mask

                        return h_prev_out, c_prev_out

                    def eval_fn_forget(h_prev=h_prev, c_prev=c_prev):
                        return h_prev, c_prev

                    h_prev, c_prev = tf.cond(self._training, train_fn_forget, eval_fn_forget)

                s_bottomup = self._kernel_bottomup(inputs)
                s_recurrent = self._kernel_recurrent(h_prev)
                s = s_bottomup + s_recurrent
                if not self._layer_normalization and self._use_bias:
                    s += self._bias

                # Input gate
                i = s[:, :units]
                if self._layer_normalization:
                    i = self.norm(i, 'i_ln')
                i = self._recurrent_activation(i)

                # Output gate
                o = s[:, units:units * 2]
                if self._layer_normalization:
                    o = self.norm(o, 'o_ln')
                o = self._recurrent_activation(o)

                # Cell proposal
                g = s[:, units * 2:units * 3]
                if self._layer_normalization:
                    g = self.norm(g, 'g_ln')
                g = self._activation(g)

                # Forget gate
                if self._forget_gate_as_irf:
                    f_W = self._t_delta_embedding_W + s[:, units * 3:units * 4]
                    f_b = self._t_delta_embedding_b + s[:, units * 4:units * 5]
                    if self._layer_normalization:
                        f_W = self.norm(f_W, 'f_W_ln')
                        f_b = self.norm(f_b, 'f_b_ln')
                    t = t_cur - t_prev
                    t_embedding = self._time_projection_inner_activation(f_W * t + f_b + self._forget_bias)
                    f = self._kernel_time_projection(t_embedding)
                else:
                    f = s[:, units * 3:units * 4]
                f = self._recurrent_activation(f)

                c = f * c_prev + i * g
                h = o * self._activation(c)

                if self._l2_normalize_states:
                    h = tf.nn.l2_normalize(h, epsilon=self._epsilon, axis=-1)

                c = self._c_dropout_layer(c)
                h = self._h_dropout_layer(h)

                if mask is not None:
                    c = c * mask + c_prev * (1 - mask)
                    h = h * mask + h_prev * (1 - mask)

                new_state = CDRNNStateTuple(c=c, h=h, t=t_cur)

                return new_state, new_state

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return []

    def resample_ops(self):
        out = []
        if self.built:
            for layer in self._kernel_bottomup_layers + self._kernel_recurrent_layers + self._kernel_time_projection_layers:
                out.append(layer.resample_ops())

        return out


class RNNLayer(object):
    def __init__(
            self,
            units=None,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init='glorot_normal',
            recurrent_kernel_sd_init='glorot_normal',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            return_sequences=True,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.use_MAP_mode = use_MAP_mode
        self.units = units
        self.kernel_depth = kernel_depth
        self.time_projection_depth = time_projection_depth
        self.prefinal_mode = prefinal_mode
        self.forget_bias = forget_bias
        self.forget_gate_as_irf = forget_gate_as_irf
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.prefinal_activation = prefinal_activation
        self.time_projection_inner_activation = time_projection_inner_activation
        self.bottomup_kernel_sd_init = bottomup_kernel_sd_init
        self.recurrent_kernel_sd_init = recurrent_kernel_sd_init
        self.bottomup_dropout = bottomup_dropout
        self.h_dropout = h_dropout
        self.c_dropout = c_dropout
        self.forget_rate = forget_rate
        self.weight_normalization = weight_normalization
        self.layer_normalization = layer_normalization
        if not bool(rangf_map):
            rangf_map = {}
        self.rangf_map = rangf_map
        self.use_bias = use_bias
        self.global_step = global_step
        self.l2_normalize_states = l2_normalize_states
        self.return_sequences = return_sequences
        self.reuse = reuse
        self.name = name
        self.dtype = dtype
        self.epsilon = epsilon

        self.cell = None

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs_shape[-1]
                    else:
                        units = self.units

                    self.cell = RNNCell(
                        units,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_depth=self.kernel_depth,
                        time_projection_depth=self.time_projection_depth,
                        prefinal_mode=self.prefinal_mode,
                        forget_bias=self.forget_bias,
                        forget_gate_as_irf=self.forget_gate_as_irf,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        prefinal_activation=self.prefinal_activation,
                        time_projection_inner_activation=self.time_projection_inner_activation,
                        bottomup_kernel_sd_init=self.bottomup_kernel_sd_init,
                        recurrent_kernel_sd_init=self.recurrent_kernel_sd_init,
                        bottomup_dropout=self.bottomup_dropout,
                        h_dropout=self.h_dropout,
                        c_dropout=self.c_dropout,
                        forget_rate=self.forget_rate,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        rangf_map=self.rangf_map,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build((inputs_shape[0], inputs_shape[2]))

            self.built = True

    @property
    def regularizable_weights(self):
        return self.cell.regularizable_weights

    def __call__(self, inputs, times=None, mask=None, return_state=False, initial_state=None):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                inputs = {'inputs': inputs}

                if times is not None:
                    inputs['times'] = times

                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)
                    while len(mask.shape) < 3:
                        mask = mask[..., None]

                H, _ = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs,
                    initial_state=initial_state,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                H, c, _ = H

                if not self.return_sequences:
                    H = H[:, -1]
                    if return_state:
                        c = c[:, -1]

                if return_state:
                    out = (H, c)
                else:
                    out = H

                return out

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return self.cell.ema_ops()

    def resample_ops(self):
        return self.cell.resample_ops()


class RNNCellBayes(RNNCell):
    def __init__(
            self,
            units,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init=None,
            recurrent_kernel_sd_init=None,
            declare_priors_weights=True,
            declare_priors_biases=False,
            kernel_sd_prior=1,
            bias_sd_prior=1,
            bias_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            constraint='softplus',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        super(RNNCellBayes, self).__init__(
            units=units,
            training=training,
            use_MAP_mode=use_MAP_mode,
            kernel_depth=kernel_depth,
            time_projection_depth=time_projection_depth,
            prefinal_mode=prefinal_mode,
            forget_bias=forget_bias,
            forget_gate_as_irf=forget_gate_as_irf,
            activation=activation,
            recurrent_activation=recurrent_activation,
            prefinal_activation=prefinal_activation,
            time_projection_inner_activation=time_projection_inner_activation,
            bottomup_kernel_sd_init=bottomup_kernel_sd_init,
            recurrent_kernel_sd_init=recurrent_kernel_sd_init,
            bottomup_dropout=bottomup_dropout,
            h_dropout=h_dropout,
            c_dropout=c_dropout,
            forget_rate=forget_rate,
            weight_normalization=weight_normalization,
            layer_normalization=layer_normalization,
            rangf_map=rangf_map,
            use_bias=use_bias,
            global_step=global_step,
            l2_normalize_states=l2_normalize_states,
            reuse=reuse,
            name=name,
            dtype=dtype,
            epsilon=epsilon,
            session=session
        )

        self._declare_priors_weights = declare_priors_weights
        self._declare_priors_biases = declare_priors_biases
        self._kernel_sd_prior = kernel_sd_prior
        self._bias_sd_prior = bias_sd_prior
        self._bias_sd_init = bias_sd_init
        self._posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio

        self._constraint = constraint
        if self._constraint.lower() == 'softplus':
            self._constraint_fn = tf.nn.softplus
            self._constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self._constraint.lower() == 'square':
            self._constraint_fn = tf.square
            self._constraint_fn_inv = tf.sqrt
        elif self._constraint.lower() == 'abs':
            self._constraint_fn = self._safe_abs
            self._constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self._constraint)

        self.kl_penalties_base = {}

    def initialize_kernel(
            self,
            in_dim,
            out_dim,
            kernel_type='bottomup',
            depth=None,
            inner_activation=None,
            prefinal_mode=None,
            name=None
    ):
        with self._session.as_default():
            with self._session.graph.as_default():
                kernel_lambdas = []
                if depth is None:
                    depth = self._kernel_depth
                if prefinal_mode is None:
                    prefinal_mode = self._prefinal_mode

                if prefinal_mode.lower() == 'max':
                    if out_dim > in_dim:
                        prefinal_dim = out_dim
                    else:
                        prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'in':
                    prefinal_dim = in_dim
                elif prefinal_mode.lower() == 'out':
                    prefinal_dim = out_dim
                else:
                    raise ValueError('Unrecognized value for prefinal_mode: %s.' % prefinal_mode)

                layers = []

                if kernel_type == 'bottomup':
                    kernel_sd_init = self._bottomup_kernel_sd_init
                elif kernel_type == 'recurrent':
                    kernel_sd_init = self._recurrent_kernel_sd_init
                else:
                    raise ValueError('Unrecognized kernel type: %s.' % kernel_type)

                for d in range(depth):
                    if d == depth - 1:
                        activation = None
                        units = out_dim
                        use_bias = False
                    else:
                        if inner_activation is None:
                            activation = self._prefinal_activation
                        else:
                            activation = inner_activation
                        units = prefinal_dim
                        use_bias = self._use_bia

                    if self.name:
                        name_cur = self.name + '/'
                    else:
                        name_cur = ''

                    if name:
                        name_cur += name + '_d%d' % d
                    else:
                        name_cur += 'd%d' % d

                    kernel_layer = DenseLayerBayes(
                        training=self._training,
                        units=units,
                        use_bias=use_bias,
                        activation=activation,
                        rangf_map=self._rangf_map,
                        declare_priors_weights=self._declare_priors_weights,
                        declare_priors_biases=self._declare_priors_biases,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_sd_prior=self._kernel_sd_prior,
                        kernel_sd_init=kernel_sd_init,
                        bias_sd_prior=self._bias_sd_prior,
                        bias_sd_init=self._bias_sd_init,
                        posterior_to_prior_sd_ratio=self._posterior_to_prior_sd_ratio,
                        constraint=self._constraint,
                        epsilon=self._epsilon,
                        session=self._session,
                        reuse=self._reuse,
                        name=name_cur
                    )

                    kernel_layer.build([None, in_dim])

                    layers.append(kernel_layer)
                    kernel_lambdas.append(make_lambda(kernel_layer, session=self._session))

                kernel = compose_lambdas(kernel_lambdas)

                return kernel, layers

    def initialize_irf_biases(self):
        with self._session.as_default():
            with self._session.graph.as_default():
                out_dim = self._num_units
                kernel_sd_prior = get_numerical_sd(self._kernel_sd_prior, in_dim=1, out_dim=out_dim)
                kernel_sd_posterior = kernel_sd_prior * self._posterior_to_prior_sd_ratio

                if not self.name:
                    name = ''
                else:
                    name = self.name

                # Posterior distribution
                t_delta_embedding_W_q_loc = self.add_variable(
                    name='t_delta_embedding_W_q_loc',
                    initializer=tf.zeros_initializer(),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_W_q_scale = self.add_variable(
                    name='t_delta_embedding_W_q_scale',
                    initializer=self._constraint_fn_inv(kernel_sd_posterior),
                    shape=[1, self._num_units]
                )
                self.t_delta_embedding_W_q_dist = Normal(
                    loc=t_delta_embedding_W_q_loc,
                    scale=self._constraint_fn(t_delta_embedding_W_q_scale) + self.epsilon,
                    name='t_delta_embedding_W_q'
                )
                if self._declare_priors_weights:
                    # Prior distribution
                    self.t_delta_embedding_W_prior_dist = Normal(
                        loc=0.,
                        scale=kernel_sd_prior,
                        name='t_delta_embedding_W'
                    )
                    self.kl_penalties_base[self.name + '/t_delta_embedding_W'] = {
                        'loc': 0.,
                        'scale': kernel_sd_prior,
                        'val': self.t_delta_embedding_W_q_dist.kl_divergence(self.t_delta_embedding_W_prior_dist)
                    }
                with tf.variable_scope(name, reuse=self.reuse):
                    self.t_delta_embedding_W_eval_sample = self.t_delta_embedding_W_q_dist.sample()
                    self.t_delta_embedding_W_eval = tf.get_variable(
                        name='t_delta_embedding_W_sample',
                        initializer=tf.zeros_initializer(),
                        shape=self.t_delta_embedding_W_eval_sample.shape,
                        dtype=tf.float32,
                        trainable=False
                    )
                    self.t_delta_embedding_W_eval_resample = tf.assign(self.t_delta_embedding_W_eval,
                                                                       self.t_delta_embedding_W_eval_sample)
                self.t_delta_embedding_W = tf.cond(
                    self.training,
                    self.t_delta_embedding_W_q_dist.sample,
                    tf.cond(
                        self.use_MAP_mode,
                        self.t_delta_embedding_W_q_dist.mean,
                        lambda: self.t_delta_embedding_W_eval,
                    )
                )

                bias_sd_prior = get_numerical_sd(self._kernel_sd_prior, in_dim=1, out_dim=1)
                bias_sd_posterior = bias_sd_prior * self._posterior_to_prior_sd_ratio

                # Posterior distribution
                t_delta_embedding_b_q_loc = self.add_variable(
                    name='t_delta_embedding_b_q_loc',
                    initializer=tf.zeros_initializer(),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_b_q_scale = self.add_variable(
                    name='t_delta_embedding_b_q_scale',
                    initializer=self._constraint_fn_inv(bias_sd_posterior),
                    shape=[1, self._num_units]
                )
                t_delta_embedding_b_q_dist = Normal(
                    loc=t_delta_embedding_b_q_loc,
                    scale=self._constraint_fn(t_delta_embedding_b_q_scale) + self.epsilon,
                    name='t_delta_embedding_b_q'
                )
                if self._declare_priors_biases:
                    # Prior distribution
                    self.t_delta_embedding_b_prior_dist = Normal(
                        loc=0.,
                        scale=bias_sd_prior,
                        name='t_delta_embedding_b'
                    )
                    self.kl_penalties_base[self.name + '/t_delta_embedding_b'] = {
                        'loc': 0.,
                        'scale': bias_sd_prior,
                        'val': t_delta_embedding_b_q_dist.kl_divergence(self.t_delta_embedding_b_prior_dist)
                    }
                with tf.variable_scope(name, reuse=self.reuse):
                    self.t_delta_embedding_b_eval_sample = self.t_delta_embedding_b_q_dist.sample()
                    self.t_delta_embedding_b_eval = tf.get_variable(
                        name='t_delta_embedding_b_sample',
                        initializer=tf.zeros_initializer(),
                        shape=self.t_delta_embedding_b_eval_sample.shape,
                        dtype=tf.float32,
                        trainable=False
                    )
                    self.t_delta_embedding_b_eval_resample = tf.assign(self.t_delta_embedding_b_eval,
                                                                       self.t_delta_embedding_b_eval_sample)
                self.t_delta_embedding_b = tf.cond(
                    self.training,
                    self.t_delta_embedding_b_q_dist.sample,
                    tf.cond(
                        self.use_MAP_mode,
                        self.t_delta_embedding_b_q_dist.mean,
                        lambda: self.t_delta_embedding_b_eval
                    )
                )

    def kl_penalties(self):
        out = self.kl_penalties_base
        for layer in self.layers:
            out.update(layer.kl_penalties())

        return out

    def ema_ops(self):
        return []

    def resample_ops(self):
        out = super(RNNCellBayes, self).resample_ops()
        if self._forget_gate_as_irf:
            out += [
                self.t_delta_embedding_W_eval_resample,
                self.t_delta_embedding_b_eval_resample,
            ]
        return out


class RNNLayerBayes(RNNLayer):
    def __init__(
            self,
            units=None,
            training=False,
            use_MAP_mode=True,
            kernel_depth=1,
            time_projection_depth=1,
            prefinal_mode='max',
            forget_bias=1.0,
            forget_gate_as_irf=False,
            activation=None,
            recurrent_activation='sigmoid',
            prefinal_activation='tanh',
            time_projection_inner_activation='tanh',
            bottomup_kernel_sd_init=None,
            recurrent_kernel_sd_init=None,
            declare_priors_weights=True,
            declare_priors_biases=False,
            kernel_sd_prior=1,
            bias_sd_prior=1,
            bias_sd_init=None,
            posterior_to_prior_sd_ratio=1,
            constraint='softplus',
            bottomup_dropout=None,
            h_dropout=None,
            c_dropout=None,
            forget_rate=None,
            weight_normalization=False,
            layer_normalization=False,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_bias=True,
            global_step=None,
            l2_normalize_states=False,
            return_sequences=True,
            reuse=tf.AUTO_REUSE,
            name=None,
            dtype=None,
            epsilon=1e-5,
            session=None
    ):
        super(RNNLayerBayes, self).__init__(
            units=units,
            training=training,
            use_MAP_mode=use_MAP_mode,
            kernel_depth=kernel_depth,
            time_projection_depth=time_projection_depth,
            prefinal_mode=prefinal_mode,
            forget_bias=forget_bias,
            forget_gate_as_irf=forget_gate_as_irf,
            activation=activation,
            recurrent_activation=recurrent_activation,
            prefinal_activation=prefinal_activation,
            time_projection_inner_activation=time_projection_inner_activation,
            bottomup_kernel_sd_init=bottomup_kernel_sd_init,
            recurrent_kernel_sd_init=recurrent_kernel_sd_init,
            bottomup_dropout=bottomup_dropout,
            h_dropout=h_dropout,
            c_dropout=c_dropout,
            forget_rate=forget_rate,
            weight_normalization=weight_normalization,
            layer_normalization=layer_normalization,
            rangf_map=rangf_map,
            use_bias=use_bias,
            global_step=global_step,
            l2_normalize_states=l2_normalize_states,
            return_sequences=return_sequences,
            reuse=reuse,
            name=name,
            dtype=dtype,
            epsilon=epsilon,
            session=session
        )

        self.declare_priors_weights = declare_priors_weights
        self.declare_priors_biases = declare_priors_biases
        self.kernel_sd_prior = kernel_sd_prior
        self.bias_sd_prior = bias_sd_prior
        self.bias_sd_init = bias_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.constraint = constraint

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.units is None:
                        units = inputs_shape[-1]
                    else:
                        units = self.units

                    self.cell = RNNCellBayes(
                        units,
                        training=self.training,
                        use_MAP_mode=self.use_MAP_mode,
                        kernel_depth=self.kernel_depth,
                        time_projection_depth=self.time_projection_depth,
                        prefinal_mode=self.prefinal_mode,
                        forget_bias=self.forget_bias,
                        forget_gate_as_irf=self.forget_gate_as_irf,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        prefinal_activation=self.prefinal_activation,
                        time_projection_inner_activation=self.time_projection_inner_activation,
                        bottomup_kernel_sd_init=self.bottomup_kernel_sd_init,
                        recurrent_kernel_sd_init=self.recurrent_kernel_sd_init,
                        bottomup_dropout=self.bottomup_dropout,
                        h_dropout=self.h_dropout,
                        c_dropout=self.c_dropout,
                        forget_rate=self.forget_rate,
                        declare_priors_weights=self.declare_priors_weights,
                        declare_priors_biases=self.declare_priors_biases,
                        kernel_sd_prior=self.kernel_sd_prior,
                        bias_sd_prior=self.bias_sd_prior,
                        bias_sd_init=self.bias_sd_init,
                        posterior_to_prior_sd_ratio=self.posterior_to_prior_sd_ratio,
                        constraint=self.constraint,
                        weight_normalization=self.weight_normalization,
                        layer_normalization=self.layer_normalization,
                        rangf_map=self.rangf_map,
                        use_bias=self.use_bias,
                        global_step=self.global_step,
                        l2_normalize_states=self.l2_normalize_states,
                        reuse=self.reuse,
                        name=self.name,
                        dtype=self.dtype,
                        epsilon=self.epsilon,
                    )

                    self.cell.build((inputs_shape[0], inputs_shape[2]))

            self.built = True

    def kl_penalties(self):
        return self.cell.kl_penalties()

    def ema_ops(self):
        return self.cell.ema_ops()


class BatchNormLayer(object):
    def __init__(
            self,
            decay=0.999,
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        assert axis != 0, 'Cannot target the batch dimension for normalization'
        self.session = get_session(session)
        self.decay = decay
        self.shift_activations = shift_activations
        self.rescale_activations = rescale_activations
        self.axis = axis
        self.epsilon = epsilon
        self.training = training
        if not bool(rangf_map):
            rangf_map = {}
        self.rangf_map = rangf_map
        self.reuse = reuse
        self.name = name

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = len(inputs_shape) - 1
            else:
                axis = self.axis

            self.reduction_axes = sorted(list(set(range(len(inputs_shape))) - {axis}))

            # Train gains and biases along axis/axes not being reduced
            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(1)
                else:
                    shape.append(int(inputs_shape[i]))

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.moving_mean = tf.get_variable(
                            name='moving_mean',
                            initializer=tf.zeros_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_mean_op = None

                        self.moving_variance = tf.get_variable(
                            name='moving_variance',
                            initializer=tf.ones_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_variance_op = None

                        if self.shift_activations:
                            self.beta = tf.get_variable(
                                name='beta',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                            self.beta_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                self.beta_ran[gf] = tf.get_variable(
                                    name='beta_by_%s' % sn(gf),
                                    initializer=tf.zeros_initializer(),
                                    shape=[n_levels] + shape[1:]
                                )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            self.gamma = tf.get_variable(
                                name='gamma',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                            self.gamma_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                self.gamma_ran[gf] = tf.get_variable(
                                    name='gamma_by_%s' % sn(gf),
                                    initializer=tf.zeros_initializer(),
                                    shape=[n_levels] + shape[1:]
                                )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                name = self.name
                if not name:
                    name = ''
                decay = self.decay

                def train_fn(inputs=inputs):
                    return tf.nn.moments(inputs, self.reduction_axes, keep_dims=True)

                def eval_fn():
                    return self.moving_mean, self.moving_variance

                mean, variance = tf.cond(self.training, train_fn, eval_fn)
                sd = tf.sqrt(variance + self.epsilon)
                out = ((inputs - mean) / sd)

                beta = self.beta
                gamma = self.gamma
                for gf in self.rangf_map:
                    Y_gf = self.rangf_map[gf][1]

                    if self.shift_activations:
                        beta_ran = self.beta_ran[gf]
                        beta_ran -= tf.reduce_mean(beta_ran, axis=0, keepdims=True)
                        beta_ran = tf.pad(beta_ran, [[0,1]] + [[0,0]] * (len(beta_ran.shape) - 1))
                        beta += tf.gather(beta_ran, Y_gf)

                    if self.rescale_activations:
                        gamma_ran = self.gamma_ran[gf]
                        gamma_ran -= tf.reduce_mean(gamma_ran, axis=0, keepdims=True)
                        gamma_ran = tf.pad(gamma_ran, [[0,1]] + [[0,0]] * (len(beta_ran.shape) - 1))
                        gamma += tf.gather(gamma_ran, Y_gf)

                while len(beta.shape) < len(out.shape):
                    beta = tf.expand_dims(beta, axis=1)
                while len(gamma.shape) < len(out.shape):
                    gamma = tf.expand_dims(gamma, axis=1)
                out = out * gamma + beta

                if self.moving_mean_op is None:
                    self.moving_mean_op = tf.assign(
                        self.moving_mean,
                        self.moving_mean * decay + mean * (1 - decay),
                        name=name + '_moving_mean_update'
                    )
                if self.moving_variance_op is None:
                    self.moving_variance_op = tf.assign(
                        self.moving_variance,
                        self.moving_variance * decay + variance * (1 - decay),
                        name=name + '_moving_variance_update'
                    )

                return out

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return [self.moving_mean_op, self.moving_variance_op]

    def resample_ops(self):
        return []


class BatchNormLayerBayes(BatchNormLayer):
    def __init__(
            self,
            decay=0.999,
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_MAP_mode=None,
            declare_priors_scale=True,
            declare_priors_shift=False,
            scale_sd_prior=1.,
            scale_sd_init=None,
            shift_sd_prior=1.,
            shift_sd_init=None,
            posterior_to_prior_sd_ratio=1.,
            ranef_to_fixef_prior_sd_ratio=0.1,
            constraint='softplus',
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        super(BatchNormLayerBayes, self).__init__(
            decay=decay,
            shift_activations=shift_activations,
            rescale_activations=rescale_activations,
            axis=axis,
            training=training,
            rangf_map=rangf_map,
            epsilon=epsilon,
            session=session,
            reuse=reuse,
            name=name
        )

        self.use_MAP_mode = use_MAP_mode
        self.declare_priors_scale = declare_priors_scale
        self.declare_priors_shift = declare_priors_shift
        self.scale_sd_prior = scale_sd_prior
        self.scale_sd_init = scale_sd_init
        self.shift_sd_prior = shift_sd_prior
        self.shift_sd_init = shift_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.ranef_to_fixef_prior_sd_ratio = ranef_to_fixef_prior_sd_ratio
        self.constraint = constraint

        if self.constraint.lower() == 'softplus':
            self.constraint_fn = tf.nn.softplus
            self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self.constraint.lower() == 'square':
            self.constraint_fn = tf.square
            self.constraint_fn_inv = tf.sqrt
        elif self.constraint.lower() == 'abs':
            self.constraint_fn = self._safe_abs
            self.constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self.constraint)

        self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = len(inputs_shape) - 1
            else:
                axis = self.axis

            self.reduction_axes = sorted(list(set(range(len(inputs_shape))) - {axis}))

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(1)
                else:
                    shape.append(inputs_shape[i])
            shape = tf.convert_to_tensor(shape)

            if not self.name:
                name = ''
            else:
                name = self.name

            if self.use_MAP_mode is None:
                self.use_MAP_mode = tf.logical_not(self.training)

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        self.moving_mean = tf.get_variable(
                            name='moving_mean',
                            initializer=tf.zeros_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_mean_op = None

                        self.moving_variance = tf.get_variable(
                            name='moving_variance',
                            initializer=tf.ones_initializer(),
                            shape=shape,
                            trainable=False
                        )
                        self.moving_variance_op = None
                        
                        if self.shift_activations:
                            shift_sd_prior = get_numerical_sd(self.shift_sd_prior, in_dim=1, out_dim=1)
                            if self.shift_sd_init:
                                shift_sd_posterior = get_numerical_sd(self.shift_sd_init, in_dim=1, out_dim=1)
                            else:
                                shift_sd_posterior = shift_sd_prior * self.posterior_to_prior_sd_ratio
                            _shift_sd_prior = np.ones(shape) * shift_sd_prior
                            _shift_sd_posterior = np.ones(shape) * shift_sd_posterior

                            rv_dict = get_random_variable(
                                'beta',
                                shape,
                                _shift_sd_posterior,
                                constraint=self.constraint,
                                sd_prior=_shift_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_shift:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.beta_eval_resample = rv_dict['v_eval_resample']
                            self.beta = rv_dict['v']
                            self.beta_eval_resample_ran = {}
                            self.beta_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                _shift_sd_prior = np.ones([n_levels] + shape) * shift_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                                _shift_sd_posterior = np.ones([n_levels] + shape) * shift_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                                rv_dict = get_random_variable(
                                    'beta_by_%s' % sn(gf),
                                    [n_levels] + shape[1:],
                                    _shift_sd_posterior,
                                    constraint=self.constraint,
                                    sd_prior=_shift_sd_prior,
                                    training=self.training,
                                    use_MAP_mode=self.use_MAP_mode,
                                    epsilon=self.epsilon,
                                    session=self.session
                                )
                                if self.declare_priors_shift:
                                    self.kl_penalties_base.update(rv_dict['kl_penalties'])
                                self.beta_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                                self.beta_ran[gf] = rv_dict['v']

                        if self.rescale_activations:
                            scale_sd_prior = get_numerical_sd(self.scale_sd_prior, in_dim=1, out_dim=1)
                            if self.scale_sd_init:
                                scale_sd_posterior = get_numerical_sd(self.scale_sd_init, in_dim=1, out_dim=1)
                            else:
                                scale_sd_posterior = scale_sd_prior * self.posterior_to_prior_sd_ratio
                            init = np.ones(shape)
                            _scale_sd_prior = init * scale_sd_prior
                            _scale_sd_posterior = init * scale_sd_posterior

                            rv_dict = get_random_variable(
                                'gamma',
                                shape,
                                _scale_sd_posterior,
                                init=init,
                                constraint=self.constraint,
                                sd_prior=_scale_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_scale:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.gamma_eval_resample = rv_dict['v_eval_resample']
                            self.gamma = rv_dict['v']
                            self.gamma_eval_resample_ran = {}
                            self.gamma_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                _scale_sd_prior = np.ones([n_levels] + shape) * scale_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                                _scale_sd_posterior = np.ones([n_levels] + shape) * scale_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                                rv_dict = get_random_variable(
                                    'gamma_by_%s' % sn(gf),
                                    [n_levels] + shape[1:],
                                    _scale_sd_posterior,
                                    constraint=self.constraint,
                                    sd_prior=_scale_sd_prior,
                                    training=self.training,
                                    use_MAP_mode=self.use_MAP_mode,
                                    epsilon=self.epsilon,
                                    session=self.session
                                )
                                if self.declare_priors_scale:
                                    self.kl_penalties_base.update(rv_dict['kl_penalties'])
                                self.gamma_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                                self.gamma_ran[gf] = rv_dict['v']

        self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                return self.kl_penalties_base.copy()

    def resample_ops(self):
        out = super(BatchNormLayerBayes, self).resample_ops()
        if self.built:
            if self.shift_activations:
                out.append(self.beta_eval_resample)
                for gf in self.rangf_map:
                    out.append(self.beta_eval_resample_ran[gf])
            if self.rescale_activations:
                out.append(self.gamma_eval_resample)
                for gf in self.rangf_map:
                    out.append(self.gamma_eval_resample_ran[gf])

        return out


class LayerNormLayer(object):
    def __init__(
            self,
            normalization_type='z',
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        assert axis != 0, 'Cannot target the batch dimension for normalization'
        self.session = get_session(session)
        self.training = training
        self.normalization_type = normalization_type
        assert self.normalization_type in ['z', 'length'], 'Unrecognized normalization type: %s' % self.normalization_type
        self.shift_activations = shift_activations
        self.rescale_activations = rescale_activations
        self.axis = axis
        if not bool(rangf_map):
            rangf_map = {}
        self.rangf_map = rangf_map
        self.epsilon = epsilon
        self.reuse = reuse
        self.name = name

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = [len(inputs_shape) - 1]
            else:
                axis = [self.axis]
            if isinstance(axis, int):
                axis = [axis]

            self.reduction_axes = axis

            # Train gains and biases along axis/axes being reduced
            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(int(inputs_shape[i]))
                else:
                    shape.append(1)

            if not self.name:
                name = ''
            else:
                name = self.name

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        if self.shift_activations:
                            self.beta = tf.get_variable(
                                name='beta',
                                initializer=tf.zeros_initializer(),
                                shape=shape
                            )
                            self.beta_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                self.beta_ran[gf] = tf.get_variable(
                                    name='beta_by_%s' % sn(gf),
                                    initializer=tf.zeros_initializer(),
                                    shape=[n_levels] + shape[1:]
                                )
                        else:
                            self.beta = tf.Variable(0., name='beta', trainable=False)

                        if self.rescale_activations:
                            self.gamma = tf.get_variable(
                                name='gamma',
                                initializer=tf.ones_initializer(),
                                shape=shape
                            )
                            self.gamma_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                self.gamma_ran[gf] = tf.get_variable(
                                    name='gamma_by_%s' % sn(gf),
                                    initializer=tf.zeros_initializer(),
                                    shape=[n_levels] + shape[1:]
                                )
                        else:
                            self.gamma = tf.Variable(1., name='beta', trainable=False)

        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        with self.session.as_default():
            with self.session.graph.as_default():
                if self.normalization_type == 'z': # ordinary layer normalization
                    mean, variance = tf.nn.moments(inputs, self.reduction_axes, keep_dims=True)
                    sd = tf.sqrt(variance + self.epsilon)
                    out = (inputs - mean) / sd
                else: # length normalization
                    out = tf.nn.l2_normalize(inputs, axis=self.reduction_axes, epsilon=self.epsilon)

                beta = self.beta
                gamma = self.gamma

                for gf in self.rangf_map:
                    Y_gf = self.rangf_map[gf][1]

                    if self.shift_activations:
                        beta_ran = self.beta_ran[gf]
                        beta_ran -= tf.reduce_mean(beta_ran, axis=0, keepdims=True)
                        beta_ran = tf.pad(beta_ran, [[0,1]] + [[0,0]] * (len(beta_ran.shape) - 1))
                        beta += tf.gather(beta_ran, Y_gf)

                    if self.rescale_activations:
                        gamma_ran = self.gamma_ran[gf]
                        gamma_ran -= tf.reduce_mean(gamma_ran, axis=0, keepdims=True)
                        gamma_ran = tf.pad(gamma_ran, [[0,1]] + [[0,0]] * (len(beta_ran.shape) - 1))
                        gamma += tf.gather(gamma_ran, Y_gf)

                while len(beta.shape) < len(out.shape):
                    beta = tf.expand_dims(beta, axis=1)
                while len(gamma.shape) < len(out.shape):
                    gamma = tf.expand_dims(gamma, axis=1)

                out = out * gamma + beta

                return out

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return []

    def resample_ops(self):
        return []


class LayerNormLayerBayes(LayerNormLayer):
    def __init__(
            self,
            normalization_type='z',
            shift_activations=True,
            rescale_activations=True,
            axis=-1,
            training=True,
            rangf_map=None,  # Dict mapping <rangfid>: (n_levels, tensor)
            use_MAP_mode=None,
            declare_priors_scale=True,
            declare_priors_shift=False,
            scale_sd_prior=1.,
            scale_sd_init=None,
            shift_sd_prior=1.,
            shift_sd_init=None,
            posterior_to_prior_sd_ratio=1.,
            ranef_to_fixef_prior_sd_ratio=0.1,
            constraint='softplus',
            epsilon=1e-5,
            session=None,
            reuse=tf.AUTO_REUSE,
            name=None
    ):
        super(LayerNormLayerBayes, self).__init__(
            training=training,
            normalization_type=normalization_type,
            shift_activations=shift_activations,
            rescale_activations=rescale_activations,
            axis=axis,
            rangf_map=rangf_map,
            epsilon=epsilon,
            session=session,
            reuse=reuse,
            name=name
        )

        self.use_MAP_mode = use_MAP_mode
        self.declare_priors_scale = declare_priors_scale
        self.declare_priors_shift = declare_priors_shift
        self.scale_sd_prior = scale_sd_prior
        self.scale_sd_init = scale_sd_init
        self.shift_sd_prior = shift_sd_prior
        self.shift_sd_init = shift_sd_init
        self.posterior_to_prior_sd_ratio = posterior_to_prior_sd_ratio
        self.ranef_to_fixef_prior_sd_ratio = ranef_to_fixef_prior_sd_ratio
        self.constraint = constraint

        if self.constraint.lower() == 'softplus':
            self.constraint_fn = tf.nn.softplus
            self.constraint_fn_inv = tf.contrib.distributions.softplus_inverse
        elif self.constraint.lower() == 'square':
            self.constraint_fn = tf.square
            self.constraint_fn_inv = tf.sqrt
        elif self.constraint.lower() == 'abs':
            self.constraint_fn = self._safe_abs
            self.constraint_fn_inv = tf.identity
        else:
            raise ValueError('Unrecognized constraint function %s' % self.constraint)

        self.kl_penalties_base = {}

    def build(self, inputs_shape):
        if not self.built:
            if self.axis is None or self.axis == -1:
                axis = [len(inputs_shape) - 1]
            else:
                axis = [self.axis]
            if isinstance(axis, int):
                axis = [axis]

            self.reduction_axes = axis

            shape = []
            for i in range(len(inputs_shape)):
                if i in self.reduction_axes:
                    shape.append(int(inputs_shape[i]))
                else:
                    shape.append(1)
            # shape = tf.convert_to_tensor(shape)

            if not self.name:
                name = ''
            else:
                name = self.name

            if self.use_MAP_mode is None:
                self.use_MAP_mode = tf.logical_not(self.training)

            with self.session.as_default():
                with self.session.graph.as_default():
                    with tf.variable_scope(name, reuse=self.reuse):
                        if self.shift_activations:
                            shift_sd_prior = get_numerical_sd(self.shift_sd_prior, in_dim=1, out_dim=1)
                            if self.shift_sd_init:
                                shift_sd_posterior = get_numerical_sd(self.shift_sd_init, in_dim=1, out_dim=1)
                            else:
                                shift_sd_posterior = shift_sd_prior * self.posterior_to_prior_sd_ratio
                            _shift_sd_prior = np.ones(shape) * shift_sd_prior
                            _shift_sd_posterior = np.ones(shape) * shift_sd_posterior

                            rv_dict = get_random_variable(
                                'beta',
                                shape,
                                _shift_sd_posterior,
                                constraint=self.constraint,
                                sd_prior=_shift_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_shift:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.beta_eval_resample = rv_dict['v_eval_resample']
                            self.beta = rv_dict['v']
                            self.beta_eval_resample_ran = {}
                            self.beta_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                _shift_sd_prior = np.ones([n_levels] + shape) * shift_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                                _shift_sd_posterior = np.ones([n_levels] + shape) * shift_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                                rv_dict = get_random_variable(
                                    'beta_by_%s' % sn(gf),
                                    [n_levels] + shape[1:],
                                    _shift_sd_posterior,
                                    constraint=self.constraint,
                                    sd_prior=_shift_sd_prior,
                                    training=self.training,
                                    use_MAP_mode=self.use_MAP_mode,
                                    epsilon=self.epsilon,
                                    session=self.session
                                )
                                if self.declare_priors_shift:
                                    self.kl_penalties_base.update(rv_dict['kl_penalties'])
                                self.beta_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                                self.beta_ran[gf] = rv_dict['v']

                        if self.rescale_activations:
                            scale_sd_prior = get_numerical_sd(self.scale_sd_prior, in_dim=1, out_dim=1)
                            if self.scale_sd_init:
                                scale_sd_posterior = get_numerical_sd(self.scale_sd_init, in_dim=1, out_dim=1)
                            else:
                                scale_sd_posterior = scale_sd_prior * self.posterior_to_prior_sd_ratio
                            init = np.ones(shape)
                            _scale_sd_prior = init * scale_sd_prior
                            _scale_sd_posterior = init * scale_sd_posterior

                            rv_dict = get_random_variable(
                                'gamma',
                                shape,
                                _scale_sd_posterior,
                                init=init,
                                constraint=self.constraint,
                                sd_prior=_scale_sd_prior,
                                training=self.training,
                                use_MAP_mode=self.use_MAP_mode,
                                epsilon=self.epsilon,
                                session=self.session
                            )
                            if self.declare_priors_scale:
                                self.kl_penalties_base.update(rv_dict['kl_penalties'])
                            self.gamma_eval_resample = rv_dict['v_eval_resample']
                            self.gamma = rv_dict['v']
                            self.gamma_eval_resample_ran = {}
                            self.gamma_ran = {}
                            for gf in self.rangf_map:
                                n_levels = self.rangf_map[gf][0] - 1
                                _scale_sd_prior = np.ones([n_levels] + shape) * scale_sd_prior * self.ranef_to_fixef_prior_sd_ratio
                                _scale_sd_posterior = np.ones([n_levels] + shape) * scale_sd_posterior * self.ranef_to_fixef_prior_sd_ratio
                                rv_dict = get_random_variable(
                                    'gamma_by_%s' % sn(gf),
                                    [n_levels] + shape[1:],
                                    _scale_sd_posterior,
                                    constraint=self.constraint,
                                    sd_prior=_scale_sd_prior,
                                    training=self.training,
                                    use_MAP_mode=self.use_MAP_mode,
                                    epsilon=self.epsilon,
                                    session=self.session
                                )
                                if self.declare_priors_scale:
                                    self.kl_penalties_base.update(rv_dict['kl_penalties'])
                                self.gamma_eval_resample_ran[gf] = rv_dict['v_eval_resample']
                                self.gamma_ran[gf] = rv_dict['v']

        self.built = True

    def kl_penalties(self):
        with self.session.as_default():
            with self.session.graph.as_default():
                return self.kl_penalties_base.copy()

    def resample_ops(self):
        out = super(LayerNormLayerBayes, self).resample_ops()
        if self.built:
            if self.shift_activations:
                out.append(self.beta_eval_resample)
                for gf in self.rangf_map:
                    out.append(self.beta_eval_resample_ran[gf])
            if self.rescale_activations:
                out.append(self.gamma_eval_resample)
                for gf in self.rangf_map:
                    out.append(self.gamma_eval_resample_ran[gf])

        return out


class DropoutLayer(object):
    def __init__(
            self,
            rate,
            noise_shape=None,
            training=False,
            use_MAP_mode=True,
            rescale=True,
            constant=None,
            name=None,
            reuse=tf.AUTO_REUSE,
            session=None
    ):
        self.rate = rate
        self.noise_shape = noise_shape
        self.training = training
        self.use_MAP_mode = use_MAP_mode
        self.rescale = rescale
        self.constant = constant
        self.name = name
        self.reuse = reuse
        self.session = get_session(session)

        self.built = False

    def build(self, inputs_shape):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.noise_shape:
                        noise_shape = [inputs_shape[i] if self.noise_shape[i] is None else self.noise_shape[i] for i in
                                       range(len(self.noise_shape))]
                    else:
                        noise_shape = inputs_shape

                    self.noise_shape = noise_shape

                    if self.noise_shape:
                        if self.noise_shape[-1] is None:
                            final_shape = inputs_shape[-1]
                        else:
                            final_shape = self.noise_shape[-1]
                    else:
                        final_shape = inputs_shape[-1]

                    if not self.name:
                        name = ''
                    else:
                        name = self.name

                    with tf.variable_scope(name, reuse=self.reuse):
                        self.noise_shape_eval = [1 for _ in range(len(inputs_shape) - 1)] + [int(final_shape)]
                        self.dropout_mask_eval_sample = tf.random_uniform(self.noise_shape_eval) > self.rate
                        self.dropout_mask_eval = tf.get_variable(
                            name='mask',
                            initializer=tf.ones_initializer(),
                            shape=self.noise_shape_eval,
                            dtype=tf.bool,
                            trainable=False
                        )
                        self.dropout_mask_eval_resample = tf.assign(self.dropout_mask_eval, self.dropout_mask_eval_sample)

                    self.built = True

    def __call__(self, inputs, seed=None):
        if not self.built:
            self.build(inputs.shape)
        with self.session.as_default():
            with self.session.graph.as_default():
                def train_fn(inputs=inputs):
                    inputs_shape = tf.shape(inputs)
                    noise_shape = []
                    for i, x in enumerate(self.noise_shape):
                        try:
                            noise_shape.append(int(x))
                        except TypeError:
                            noise_shape.append(inputs_shape[i])

                    dropout_mask = tf.random_uniform(noise_shape) > self.rate

                    return dropout_mask

                def eval_fn(inputs=inputs):
                    def map_fn(inputs=inputs):
                        return tf.ones(tf.shape(inputs), dtype=tf.bool)

                    def sample_fn():
                        dropout_mask = self.dropout_mask_eval
                        return dropout_mask

                    return tf.cond(self.use_MAP_mode, map_fn, sample_fn)

                dropout_mask = tf.cond(self.training, train_fn, eval_fn)

                if self.constant is None:
                    dropout_mask = tf.cast(dropout_mask, dtype=inputs.dtype)
                    out = inputs * dropout_mask
                else:
                    defaults = tf.ones_like(inputs) * self.constant
                    out = tf.where(dropout_mask, inputs, defaults)

                if self.rescale:
                    def rescale(x=out, rate=self.rate):
                        out = x * (1. / (1. - rate))
                        return out

                    out = tf.cond(
                        tf.logical_or(self.training, tf.logical_not(self.use_MAP_mode)),
                        rescale,
                        lambda: out
                    )

                return out

    def kl_penalties(self):
        return {}

    def ema_ops(self):
        return []

    def resample_ops(self):
        out = []
        if self.built:
            out.append(self.dropout_mask_eval_resample)

        return out
