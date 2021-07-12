import os

from .formula import *
from .kwargs import CDR_INITIALIZATION_KWARGS
from .util import *
from .base import Model

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True




######################################################
#
#  ABSTRACT CDR CLASS
#
######################################################

def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess

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
            stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
            stdnorm_pdf = stdnorm.prob
            stdnorm_cdf = stdnorm.cdf
            
            def irf_base(x,  mu=mu, sigma=sigma, alpha=alpha, pdf=stdnorm_pdf, cdf=stdnorm_cdf):
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
            def cdf(x, alpha=alpha, beta=beta):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                return tf.betainc(alpha, beta, x / (1+x)) * tf.exp(tf.lbeta(alpha, beta))
            
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
                return ((x + epsilon) ** (alpha - 1.) * (1. + (x + epsilon)) ** (-alpha - beta)) / (norm_const + epsilon)

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
            def cdf(x, alpha=alpha, beta=beta, delta=delta):
                # Ensure proper broadcasting
                while len(x.shape) > len(alpha.shape):
                    alpha = alpha[None, ...]
                while len(x.shape) > len(beta.shape):
                    beta = beta[None, ...]
                while len(x.shape) > len(delta.shape):
                    delta = delta[None, ...]
                return tf.betainc(alpha, beta, (x-delta) / (1+x-delta)) * tf.exp(tf.lbeta(alpha, beta))
            
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
                x = x[..., None] # Add a summation axis
                while len(x.shape) > len(v.shape):
                    v = v[None, ...]
                while len(x.shape) - 1 > len(norm_const.shape):
                    norm_const = norm_const[None, ...]

                return tf.reduce_sum(pdf(x) * v, axis=-1) / (norm_const + epsilon)

            return irf


class CDR(Model):

    _INITIALIZATION_KWARGS = CDR_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for CDR. Bayesian (:ref:`cdrbayes`) and MLE (:ref:`cdrmle`) implementations inherit from ``CDR``.
        ``CDR`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``CDR`` must implement the following instance methods:
        
            * ``initialize_objective()``
            * ``run_conv_op()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            
        Additionally, if the subclass requires any keyword arguments beyond those provided by ``CDR``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
        
        Example implementations of each of these methods can be found in the source code for :ref:`cdrmle` and :ref:`cdrbayes`.
        
    """
    _doc_args = """
        :param form_str: An R-style string representing the CDR model formula.
        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the CDR ``form_str`` provided at initialization
        :param y: A 2D pandas tensor representing the dependent variable. Must contain the following columns:
    
            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each observation in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each observation in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join([x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value) for x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs




    ######################################################
    #
    #  Initialization Methods
    #
    ######################################################

    IRF_KERNELS = {
        'DiracDelta': [],
        'Exp': [
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ExpRateGT1': [
            ('beta', {'lb': 1., 'default': 2.})
        ],
        'Gamma': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'GammaShapeGT1': [
            ('alpha', {'lb': 1., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ShiftedGamma': [
            ('alpha', {'lb': 0., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'ShiftedGammaShapeGT1': [
            ('alpha', {'lb': 1., 'default': 2.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'Normal': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.})
        ],
        'SkewNormal': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.}),
            ('alpha', {'default': 0.})
        ],
        'EMG': [
            ('mu', {'default': 0.}),
            ('sigma', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'BetaPrime': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'ShiftedBetaPrime': [
            ('alpha', {'lb': 0., 'default': 1.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('delta', {'ub': 0., 'default': -1.})
        ],
        'HRFSingleGamma': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma1': [
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma2': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.})
        ],
        'HRFDoubleGamma3': [
            ('alpha', {'lb': 1., 'default': 6.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
        'HRFDoubleGamma4': [
            ('alpha_main', {'lb': 1., 'default': 6.}),
            ('alpha_undershoot', {'lb': 1., 'default': 16.}),
            ('beta', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
        'HRFDoubleGamma5': [
            ('alpha_main', {'lb': 1., 'default': 6.}),
            ('alpha_undershoot', {'lb': 1., 'default': 16.}),
            ('beta_main', {'lb': 0., 'default': 1.}),
            ('beta_undershoot', {'lb': 0., 'default': 1.}),
            ('c', {'default': 1./6.})
        ],
    }

    def __new__(cls, *args, **kwargs):
        if cls is CDR:
            raise TypeError("CDR is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, Y, **kwargs):
        super(CDR, self).__init__(
            form_str,
            X,
            Y,
            **kwargs
        )

        for kwarg in CDR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

    def _initialize_metadata(self):
        super(CDR, self)._initialize_metadata()

        # Initialize model metadata
        self.t = self.form.t
        t = self.t
        self.node_table = t.node_table()
        self.coef_names = t.coef_names()
        self.fixed_coef_names = t.fixed_coef_names()
        self.unary_nonparametric_coef_names = t.unary_nonparametric_coef_names()
        self.interaction_list = t.interactions()
        self.interaction_names = t.interaction_names()
        self.fixed_interaction_names = t.fixed_interaction_names()
        self.impulse_names = t.impulse_names(include_interactions=True)
        self.terminal_names = t.terminal_names()
        self.atomic_irf_names_by_family = t.atomic_irf_by_family()
        self.atomic_irf_family_by_name = {}
        for family in self.atomic_irf_names_by_family:
            for id in self.atomic_irf_names_by_family[family]:
                assert id not in self.atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' % id
                self.atomic_irf_family_by_name[id] = family
        self.atomic_irf_param_init_by_family = t.atomic_irf_param_init_by_family()
        self.atomic_irf_param_trainable_by_family = t.atomic_irf_param_trainable_by_family()
        self.coef2impulse = t.coef2impulse()
        self.impulse2coef = t.impulse2coef()
        self.coef2terminal = t.coef2terminal()
        self.terminal2coef = t.terminal2coef()
        self.impulse2terminal = t.impulse2terminal()
        self.terminal2impulse = t.terminal2impulse()
        self.interaction2inputs = t.interactions2inputs()
        self.coef_by_rangf = t.coef_by_rangf()
        self.interaction_by_rangf = t.interaction_by_rangf()
        self.irf_by_rangf = t.irf_by_rangf()
        self.interactions_list = t.interactions()

        self.irf = {}

    def __getstate__(self):
        md = self._pack_metadata()
        return md

    def __setstate__(self, state):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

        self._unpack_metadata(state)
        self._initialize_metadata()

        self.log_graph = False

    def _pack_metadata(self):
        md = super(CDR, self)._pack_metadata()
        for kwarg in CDR._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        super(CDR, self)._unpack_metadata(md)

        for kwarg in CDR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
        super(CDR, self)._initialize_inputs()
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.is_response_aligned = tf.cast(
                    tf.logical_not(
                        tf.cast(self.t_delta[:, -1, :], dtype=tf.bool)
                    ),
                    self.FLOAT_TF
                )

                # Initialize regularizers
                self.irf_regularizer = self._initialize_regularizer(
                    self.irf_regularizer_name,
                    self.irf_regularizer_scale
                )

    def _initialize_base_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                super(CDR, self)._initialize_base_params()

                # Coefficients
                # Key order: response, ?(ran_gf)
                self.coefficient_fixed_base = {}
                self.coefficient_fixed_base_summary = {}
                self.coefficient_random_base = {}
                self.coefficient_random_base_summary = {}
                for response in self.response_names:
                    # Fixed
                    coef_ids = self.fixed_coef_names
                    if len(coef_ids) > 0:
                        _coefficient_fixed_base, _coefficient_fixed_base_summary = self.initialize_coefficient(
                            response,
                            coef_ids=coef_ids
                        )
                    else:
                        _coefficient_fixed_base = []
                        _coefficient_fixed_base_summary = []
                    self.coefficient_fixed_base[response] = _coefficient_fixed_base
                    self.coefficient_fixed_base_summary[response] = _coefficient_fixed_base_summary

                    # Random
                    for gf in self.rangf:
                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids):
                            _coefficient_random_base, _coefficient_random_base_summary = self.initialize_coefficient(
                                response,
                                coef_ids=coef_ids,
                                ran_gf=gf
                            )
                            if response not in self.coefficient_random_base:
                                self.coefficient_random_base[response] = {}
                            if response not in self.coefficient_random_base_summary:
                                self.coefficient_random_base_summary[response] = {}
                            self.coefficient_random_base[response][gf] = _coefficient_random_base
                            self.coefficient_random_base_summary[response][gf] = _coefficient_random_base_summary

                # Interactions
                # Key order: response, ?(ran_gf)
                self.interaction_fixed_base = {}
                self.interaction_fixed_base_summary = {}
                self.interaction_random_base = {}
                self.interaction_random_base_summary = {}
                for response in self.response_names:
                    if len(self.interaction_names):
                        interaction_ids = self.fixed_interaction_names
                        if len(interaction_ids):
                            # Fixed
                            _interaction_fixed_base, _interaction_fixed_base_summary = self.initialize_interaction(
                                response,
                                interaction_ids=interaction_ids
                            )
                            self.interaction_fixed_base[response] = _interaction_fixed_base
                            self.interaction_fixed_base_summary[response] = _interaction_fixed_base_summary

                        # Random
                        for gf in self.rangf:
                            interaction_ids = self.interaction_by_rangf.get(gf, [])
                            if len(interaction_ids):
                                _interaction_random_base, _interaction_random_base_summary = self.initialize_interaction(
                                    response,
                                    interaction_ids=interaction_ids,
                                    ran_gf=gf
                                )
                                if response not in self.interaction_random_base:
                                    self.interaction_random_base[response] = {}
                                if response not in self.interaction_random_base_summary:
                                    self.interaction_random_base_summary[response] = {}
                                self.interaction_random_base[response][gf] = _interaction_random_base
                                self.interaction_random_base_summary[response][gf] = _interaction_random_base_summary

                # IRF parameters
                # Key order: family, param
                self.irf_params_means = {}
                self.irf_params_means_unconstrained = {}
                self.irf_params_lb = {}
                self.irf_params_ub = {}
                self.irf_params_trainable_ix = {}
                self.irf_params_untrainable_ix = {}
                # Key order: response, ?(ran_gf,) family, param
                self.irf_params_fixed_base = {}
                self.irf_params_fixed_base_summary = {}
                self.irf_params_random_base = {}
                self.irf_params_random_base_summary = {}
                for family in self.atomic_irf_names_by_family:
                    # Collect metadata for IRF params
                    self.irf_params_means[family] = {}
                    self.irf_params_means_unconstrained[family] = {}
                    self.irf_params_lb[family] = {}
                    self.irf_params_ub[family] = {}
                    self.irf_params_trainable_ix[family] = {}
                    self.irf_params_untrainable_ix[family] = {}

                    param_names = []
                    param_kwargs = []
                    if family in self.IRF_KERNELS:
                        for x in self.IRF_KERNELS[family]:
                            param_names.append(x[0])
                            param_kwargs.append(x[1])
                    elif Formula.is_LCG(family):
                        _param_names, _param_kwargs = self._get_nonparametric_irf_params(family)
                        param_names += _param_names
                        param_kwargs += _param_kwargs
                    else:
                        raise ValueError('Unrecognized IRF kernel family "%s".' % family)

                    # Process and store metadata for IRF params
                    for _param_name, _param_kwargs in zip(param_names, param_kwargs):
                        param_mean, param_mean_unconstrained, param_lb, param_ub, trainable_ix, \
                            untrainable_ix = self._get_irf_param_metadata(_param_name, family, **_param_kwargs)

                        self.irf_params_means[family][_param_name] = param_mean
                        self.irf_params_means_unconstrained[family][_param_name] = param_mean_unconstrained
                        self.irf_params_lb[family][_param_name] = param_lb
                        self.irf_params_ub[family][_param_name] = param_ub
                        self.irf_params_trainable_ix[family][_param_name] = trainable_ix
                        self.irf_params_untrainable_ix[family][_param_name] = untrainable_ix

                    # Initialize IRF params
                    for response in self.response_names:
                        for _param_name in param_names:
                            # Fixed
                            _param, _param_summary = self.initialize_irf_param(
                                response,
                                family,
                                _param_name
                            )
                            if _param is not None:
                                if response not in self.irf_params_fixed_base:
                                    self.irf_params_fixed_base[response] = {}
                                if family not in self.irf_params_fixed_base[response]:
                                    self.irf_params_fixed_base[response][family] = {}
                                if response not in self.irf_params_fixed_base_summary:
                                    self.irf_params_fixed_base_summary[response] = {}
                                if family not in self.irf_params_fixed_base_summary[response]:
                                    self.irf_params_fixed_base_summary[response][family] = {}
                                self.irf_params_fixed_base[response][family][_param_name] = _param
                                self.irf_params_fixed_base_summary[response][family][_param_name] = _param

                            # Random
                            for gf in self.irf_by_rangf:
                                _param, _param_summary = self.initialize_irf_param(
                                    response,
                                    family,
                                    _param_name,
                                    ran_gf=gf
                                )
                                if _param is not None:
                                    if response not in self.irf_params_random_base:
                                        self.irf_params_random_base[response] = {}
                                    if gf not in self.irf_params_random_base[response]:
                                        self.irf_params_random_base[response][gf] = {}
                                    if family not in self.irf_params_random_base[response][gf]:
                                        self.irf_params_random_base[response][gf][family] = {}
                                    if response not in self.irf_params_random_base_summary:
                                        self.irf_params_random_base_summary[response] = {}
                                    if gf not in self.irf_params_random_base_summary[response]:
                                        self.irf_params_random_base_summary[response][gf] = {}
                                    if family not in self.irf_params_random_base_summary[response][gf]:
                                        self.irf_params_random_base_summary[response][gf][family] = {}
                                    self.irf_params_random_base[response][gf][family][_param_name] = _param
                                    self.irf_params_random_base_summary[response][gf][family][_param_name] = _param_summary

    def _get_nonparametric_irf_params(self, family):
        param_names = []
        param_kwargs = []
        bases = Formula.bases(family)
        x_init = np.zeros(bases)

        for _param_name in Formula.irf_params(family):
            _param_kwargs = {}
            if _param_name.startswith('x'):
                n = int(_param_name[1:])
                _param_kwargs['default'] = x_init[n - 1]
                _param_kwargs['lb'] = None
            elif _param_name.startswith('y'):
                n = int(_param_name[1:])
                if n == 1:
                    _param_kwargs['default'] = 1.
                else:
                    _param_kwargs['default'] = 0.
                _param_kwargs['lb'] = None
            else:
                n = int(_param_name[1:])
                _param_kwargs['default'] = n
                _param_kwargs['lb'] = 0.
            param_names.append(_param_name)
            param_kwargs.append(_param_kwargs)

        return param_names, param_kwargs

    def _get_irf_param_metadata(self, param_name, family, lb=None, ub=None, default=0.):
        irf_ids = self.atomic_irf_names_by_family[family]
        param_init = self.atomic_irf_param_init_by_family[family]
        param_trainable = self.atomic_irf_param_trainable_by_family[family]

        # Process and store initial/prior means
        param_mean = self._get_mean_init_vector(irf_ids, param_name, param_init, default=default)
        param_mean_unconstrained, param_lb, param_ub = self._process_mean(param_mean, lb=lb, ub=ub)

        # Select out irf IDs for which this param is trainable
        trainable_ix, untrainable_ix = self._get_trainable_untrainable_ix(
            param_name,
            irf_ids,
            trainable=param_trainable
        )

        return param_mean, param_mean_unconstrained, param_lb, param_ub, trainable_ix, untrainable_ix

    def _compile_coefficients(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.coefficient = {}
                self.coefficient_summary = {}
                self.coefficient_fixed = {}
                self.coefficient_fixed_summary = {}
                self.coefficient_random = {}
                self.coefficient_random_summary = {}
                for response in self.response_names:
                    self.coefficient_fixed[response] = {}
                    self.coefficient_fixed_summary[response] = {}

                    response_params = self.get_response_params(response)
                    if not self.use_distributional_regression:
                        response_params = response_params[:1]
                    nparam = len(response_params)
                    ndim = self.get_response_ndim(response)
                    fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)
                    coef_ids = self.coef_names

                    coefficient_fixed = self._scatter_along_axis(
                        fixef_ix,
                        self.coefficient_fixed_base[response],
                        [len(coef_ids), nparam, ndim]
                    )
                    coefficient_fixed_summary = self._scatter_along_axis(
                        fixef_ix,
                        self.coefficient_fixed_base_summary[response],
                        [len(coef_ids), nparam, ndim]
                    )
                    self._regularize(
                        self.coefficient_fixed_base[response],
                        regtype='coefficient',
                        var_name='coefficient_%s' % response
                    )

                    coefficient = coefficient_fixed
                    coefficient_summary = coefficient_fixed_summary

                    for i, coef_name in enumerate(self.coef_names):
                        self.coefficient_fixed[response][coef_name] = {}
                        self.coefficient_fixed_summary[response][coef_name] = {}
                        for j, response_param in enumerate(response_params):
                            _p = coefficient_fixed[:, j]
                            _p_summary = coefficient_fixed_summary[:, j]
                            if self.standardize_response and \
                                    self.is_real(response) and \
                                    response_param in ['mu', 'sigma']:
                                _p = _p * self.Y_train_sds[response]
                                _p_summary = _p_summary * self.Y_train_sds[response]
                            dim_names = self._expand_param_name_by_dim(response, response_param)
                            for k, dim_name in enumerate(dim_names):
                                val = _p[i, k]
                                val_summary = _p_summary[i, k]
                                tf.summary.scalar(
                                    'coefficient' + '/%s/%s_%s' % (
                                        sn(coef_name),
                                        sn(response),
                                        sn(dim_name)
                                    ),
                                    val_summary,
                                    collections=['params']
                                )
                                self.coefficient_fixed[response][coef_name][dim_name] = val
                                self.coefficient_fixed_summary[response][coef_name][dim_name] = val_summary

                    self.coefficient_random[response] = {}
                    self.coefficient_random_summary[response] = {}
                    for i in range(len(self.rangf)):
                        gf = self.rangf[i]
                        levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                        coefs = self.coef_by_rangf.get(gf, [])
                        if len(coefs) > 0:
                            self.coefficient_random[response][gf] = {}
                            self.coefficient_random_summary[response][gf] = {}

                            nonzero_coef_ix = names2ix(coefs, self.coef_names)

                            coefficient_random = self.coefficient_random_base[response][gf]
                            coefficient_random_summary = self.coefficient_random_base_summary[response][gf]

                            coefficient_random_means = tf.reduce_mean(coefficient_random, axis=0, keepdims=True)
                            coefficient_random_summary_means = tf.reduce_mean(coefficient_random_summary, axis=0, keepdims=True)

                            coefficient_random -= coefficient_random_means
                            coefficient_random_summary -= coefficient_random_summary_means

                            self._regularize(
                                coefficient_random,
                                regtype='ranef',
                                var_name='coefficient_%s_by_%s' % (sn(response),sn(gf))
                            )

                            for j, coef_name in enumerate(coefs):
                                self.coefficient_random[response][gf][coef_name] = {}
                                self.coefficient_random_summary[response][gf][coef_name] = {}
                                for k, response_param in enumerate(response_params):
                                    _p = coefficient_random[:, :, k]
                                    _p_summary = coefficient_random_summary[:, :, k]
                                    if self.standardize_response and \
                                            self.is_real(response) and \
                                            response_param in ['mu', 'sigma']:
                                        _p = _p * self.Y_train_sds[response]
                                        _p_summary = _p_summary * self.Y_train_sds[response]
                                    dim_names = self._expand_param_name_by_dim(response, response_param)
                                    for l, dim_name in enumerate(dim_names):
                                        val = _p[:, j, l]
                                        val_summary = _p_summary[:, j, l]
                                        tf.summary.histogram(
                                            'by_%s/coefficient/%s/%s_%s' % (
                                                sn(gf),
                                                sn(coef_name),
                                                sn(response),
                                                sn(dim_name)
                                            ),
                                            val_summary,
                                            collections=['random']
                                        )
                                        self.coefficient_random[response][gf][coef_name][dim_name] = val
                                        self.coefficient_random_summary[response][gf][coef_name][dim_name] = val_summary

                            coefficient_random = self._scatter_along_axis(
                                nonzero_coef_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    coefficient_random,
                                    [self.rangf_n_levels[i], len(coefs), nparam, ndim]
                                ),
                                [self.rangf_n_levels[i], len(self.coef_names), nparam, ndim],
                                axis=1
                            )
                            coefficient_random_summary = self._scatter_along_axis(
                                nonzero_coef_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    coefficient_random_summary,
                                    [self.rangf_n_levels[i], len(coefs), nparam, ndim]
                                ),
                                [self.rangf_n_levels[i], len(self.coef_names), nparam, ndim],
                                axis=1
                            )

                            coefficient = coefficient[None, ...] + tf.gather(coefficient_random, self.Y_gf[:, i], axis=0)
                            coefficient_summary = coefficient_summary[None, ...] + tf.gather(coefficient_random_summary, self.Y_gf[:, i], axis=0)

                    self.coefficient[response] = coefficient
                    self.coefficient_summary[response] = coefficient_summary

    def _compile_interactions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.interaction = {}
                self.interaction_summary = {}
                self.interaction_fixed = {}
                self.interaction_fixed_summary = {}
                self.interaction_random = {}
                self.interaction_random_summary = {}
                fixef_ix = names2ix(self.fixed_interaction_names, self.interaction_names)
                if len(self.interaction_names) > 0:
                    for response in self.response_names:
                        self.interaction_fixed[response] = {}
                        self.interaction_fixed_summary[response] = {}

                        response_params = self.get_response_params(response)
                        if not self.use_distributional_regression:
                            response_params = response_params[:1]
                        nparam = len(response_params)
                        ndim = self.get_response_ndim(response)
                        interaction_ids = self.interaction_names

                        interaction_fixed = self._scatter_along_axis(
                            fixef_ix,
                            self.interaction_fixed_base[response],
                            [len(interaction_ids), nparam, ndim]
                        )
                        interaction_fixed_summary = self._scatter_along_axis(
                            fixef_ix,
                            self.interaction_fixed_base_summary[response],
                            [len(interaction_ids), nparam, ndim]
                        )
                        self._regularize(
                            self.interaction_fixed_base[response],
                            regtype='interaction',
                            var_name='interaction_%s' % response
                        )

                        interaction = interaction_fixed
                        interaction_summary = interaction_fixed_summary

                        for i, interaction_name in enumerate(self.interaction_names):
                            self.interaction_fixed[response][interaction_name] = {}
                            self.interaction_fixed_summary[response][interaction_name] = {}
                            for j, response_param in enumerate(response_params):
                                _p = interaction_fixed[:, j]
                                _p_summary = interaction_fixed_summary[:, j]
                                if self.standardize_response and \
                                        self.is_real(response) and \
                                        response_param in ['mu', 'sigma']:
                                    _p = _p * self.Y_train_sds[response]
                                    _p_summary = _p_summary * self.Y_train_sds[response]
                                dim_names = self._expand_param_name_by_dim(response, response_param)
                                for k, dim_name in enumerate(dim_names):
                                    val = _p[i, k]
                                    val_summary = _p_summary[i, k]
                                    tf.summary.scalar(
                                        'interaction' + '/%s/%s_%s' % (
                                            sn(interaction_name),
                                            sn(response),
                                            sn(dim_name)
                                        ),
                                        val_summary,
                                        collections=['params']
                                    )
                                    self.interaction_fixed[response][interaction_name][dim_name] = val
                                    self.interaction_fixed_summary[response][interaction_name][dim_name] = val_summary

                        self.interaction_random[response] = {}
                        self.interaction_random_summary[response] = {}
                        for i in range(len(self.rangf)):
                            gf = self.rangf[i]
                            levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                            interactions = self.interaction_by_rangf.get(gf, [])
                            if len(interactions) > 0:
                                interaction_ix = names2ix(interactions, self.interaction_names)

                                interaction_random = self.interaction_random_base[response][gf]
                                interaction_random_summary = self.interaction_random_base_summary[response][gf]

                                interaction_random_means = tf.reduce_mean(interaction_random, axis=0, keepdims=True)
                                interaction_random_summary_means = tf.reduce_mean(interaction_random_summary, axis=0, keepdims=True)

                                interaction_random -= interaction_random_means
                                interaction_random_summary -= interaction_random_summary_means

                                self._regularize(
                                    interaction_random,
                                    regtype='ranef',
                                    var_name='interaction_%s_by_%s' % (sn(response), sn(gf))
                                )

                                for j, interaction_name in enumerate(interactions):
                                    self.interaction_random[response][gf][interaction_name] = {}
                                    self.interaction_random_summary[response][gf][interaction_name] = {}
                                    for k, response_param in enumerate(response_params):
                                        _p = interaction_random[:, :, k]
                                        _p_summary = interaction_random_summary[:, :, k]
                                        if self.standardize_response and \
                                                self.is_real(response) and \
                                                response_param in ['mu', 'sigma']:
                                            _p = _p * self.Y_train_sds[response]
                                            _p_summary = _p_summary * self.Y_train_sds[response]
                                        dim_names = self._expand_param_name_by_dim(response, response_param)
                                        for l, dim_name in enumerate(dim_names):
                                            val = _p[:, j, l]
                                            val_summary = _p_summary[:, j, l]
                                            tf.summary.histogram(
                                                'by_%s/interaction/%s/%s_%s' % (
                                                    sn(gf),
                                                    sn(interaction_name),
                                                    sn(response),
                                                    sn(dim_name)
                                                ),
                                                val_summary,
                                                collections=['random']
                                            )
                                            self.interaction_random[response][gf][interaction_name][dim_name] = val
                                            self.interaction_random_summary[response][gf][interaction_name][dim_name] = val_summary

                                interaction_random = self._scatter_along_axis(
                                    interaction_ix,
                                    self._scatter_along_axis(
                                        levels_ix,
                                        interaction_random,
                                        [self.rangf_n_levels[i], len(interactions), nparam, ndim]
                                    ),
                                    [self.rangf_n_levels[i], len(self.interaction_names), nparam, ndim],
                                    axis=1
                                )
                                interaction_random_summary = self._scatter_along_axis(
                                    interaction_ix,
                                    self._scatter_along_axis(
                                        levels_ix,
                                        interaction_random_summary,
                                        [self.rangf_n_levels[i], len(interactions), nparam, ndim]
                                    ),
                                    [self.rangf_n_levels[i], len(self.interaction_names), nparam, ndim],
                                    axis=1
                                )

                                interaction = interaction[None, ...] + tf.gather(interaction_random, self.Y_gf[:, i], axis=0)
                                interaction_summary = interaction_summary[None, ...] + tf.gather(interaction_random_summary, self.Y_gf[:, i], axis=0)

                        self.interaction[response] = interaction
                        self.interaction_summary[response] = interaction_summary

    def _compile_irf_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Base IRF params are saved as tensors with shape (nid, npredparam, npreddim),
                # one for each irf_param of each IRF kernel family.
                # Here fixed and random IRF params are summed, constraints are applied, and new tensors are stored
                # with shape (batch, 1, npredparam, npreddim), one for each parameter of each IRF ID for each response variable.
                # The 1 in the 2nd dim supports broadcasting over the time dimension.

                # Key order: response, ?(ran_gf), irf_id, irf_param
                self.irf_params = {}
                self.irf_params_summary = {}
                self.irf_params_fixed = {}
                self.irf_params_fixed_summary = {}
                self.irf_params_random = {}
                self.irf_params_random_summary = {}
                for response in self.response_names:
                    self.irf_params[response] = {}
                    self.irf_params_summary[response] = {}
                    self.irf_params_fixed[response] = {}
                    self.irf_params_fixed_summary[response] = {}
                    for family in self.atomic_irf_names_by_family:
                        if family == 'DiracDelta':
                            continue

                        irf_ids = self.atomic_irf_names_by_family[family]
                        trainable = self.atomic_irf_param_trainable_by_family[family]

                        for irf_param_name in Formula.irf_params(family):
                            response_params = self.get_response_params(response)
                            if not self.use_distributional_regression:
                                response_params = response_params[:1]
                            nparam_response = len(response_params)  # number of params of predictive dist, not IRF
                            ndim = self.get_response_ndim(response)
                            irf_param_lb = self.irf_params_lb[family][irf_param_name]
                            if irf_param_lb is not None:
                                irf_param_lb = tf.constant(irf_param_lb, dtype=self.FLOAT_TF)
                            irf_param_ub = self.irf_params_ub[family][irf_param_name]
                            if irf_param_ub is not None:
                                irf_param_ub = tf.constant(irf_param_ub, dtype=self.FLOAT_TF)
                            trainable_ix = self.irf_params_trainable_ix[family][irf_param_name]
                            untrainable_ix = self.irf_params_untrainable_ix[family][irf_param_name]

                            irf_param_means = self.irf_params_means_unconstrained[family][irf_param_name]
                            irf_param_trainable_means = tf.constant(
                                irf_param_means[trainable_ix][..., None, None],
                                dtype=self.FLOAT_TF
                            )

                            self._regularize(
                                self.irf_params_fixed_base[response][family][irf_param_name],
                                irf_param_trainable_means,
                                regtype='irf', var_name='%s_%s' % (irf_param_name, sn(response))
                            )

                            irf_param_untrainable_means = tf.constant(
                                irf_param_means[untrainable_ix][..., None, None],
                                dtype=self.FLOAT_TF
                            )
                            irf_param_untrainable_means = tf.broadcast_to(
                                irf_param_untrainable_means,
                                [len(untrainable_ix), nparam_response, ndim]
                            )

                            irf_param_trainable = self._scatter_along_axis(
                                trainable_ix,
                                self.irf_params_fixed_base[response][family][irf_param_name],
                                [len(irf_ids), nparam_response, ndim]
                            )
                            irf_param_trainable_summary = self._scatter_along_axis(
                                trainable_ix,
                                self.irf_params_fixed_base_summary[response][family][irf_param_name],
                                [len(irf_ids), nparam_response, ndim]
                            )
                            irf_param_untrainable = self._scatter_along_axis(
                                untrainable_ix,
                                irf_param_untrainable_means,
                                [len(irf_ids), nparam_response, ndim]
                            )
                            is_trainable = np.zeros(len(irf_ids), dtype=bool)
                            is_trainable[trainable_ix] = True

                            irf_param_fixed = tf.where(
                                is_trainable,
                                irf_param_trainable,
                                irf_param_untrainable
                            )
                            irf_param_fixed_summary = tf.where(
                                is_trainable,
                                irf_param_trainable_summary,
                                irf_param_untrainable
                            )

                            # Add batch dimension
                            irf_param = irf_param_fixed[None, ...]
                            irf_param_summary = irf_param_fixed_summary[None, ...]

                            for i, irf_id in enumerate(irf_ids):
                                if irf_id not in self.irf_params_fixed[response]:
                                    self.irf_params_fixed[response][irf_id] = {}
                                if irf_param_name not in self.irf_params_fixed[response][irf_id]:
                                    self.irf_params_fixed[response][irf_id][irf_param_name] = {}
                                if irf_id not in self.irf_params_fixed_summary[response]:
                                    self.irf_params_fixed_summary[response][irf_id] = {}
                                if irf_param_name not in self.irf_params_fixed_summary[response][irf_id]:
                                    self.irf_params_fixed_summary[response][irf_id][irf_param_name] = {}

                                _p = irf_param_fixed[i]
                                _p_summary = irf_param_fixed_summary[i]
                                if irf_param_lb is not None and irf_param_ub is None:
                                    _p = irf_param_lb + self.constraint_fn(_p) + self.epsilon
                                    _p_summary = irf_param_lb + self.constraint_fn(_p_summary) + self.epsilon
                                elif irf_param_lb is None and irf_param_ub is not None:
                                    _p = irf_param_ub - self.constraint_fn(_p) - self.epsilon
                                    _p_summary = irf_param_ub - self.constraint_fn(_p_summary) - self.epsilon
                                elif irf_param_lb is not None and irf_param_ub is not None:
                                    _p = self._sigmoid(_p, a=irf_param_lb, b=irf_param_ub) * (1 - 2 * self.epsilon) + self.epsilon
                                    _p_summary = self._sigmoid(_p_summary, a=irf_param_lb, b=irf_param_ub) * (1 - 2 * self.epsilon) + self.epsilon

                                for j, response_param in enumerate(response_params):
                                    dim_names = self._expand_param_name_by_dim(response, response_param)
                                    for k, dim_name in enumerate(dim_names):
                                        val = _p[j, k]
                                        val_summary = _p_summary[j, k]
                                        tf.summary.scalar(
                                            '%s/%s/%s_%s' % (
                                                irf_param_name,
                                                sn(irf_id),
                                                sn(response),
                                                sn(dim_name)
                                            ),
                                            val_summary,
                                            collections=['params']
                                        )
                                        self.irf_params_fixed[response][irf_id][irf_param_name][dim_name] = val
                                        self.irf_params_fixed_summary[response][irf_id][irf_param_name][dim_name] = val_summary

                            for i, gf in enumerate(self.rangf):
                                if gf in self.irf_by_rangf:
                                    irf_ids_ran = [x for x in self.irf_by_rangf[gf] if irf_param_name in trainable[x]]
                                    if len(irf_ids_ran):
                                        irfs_ix = names2ix(self.irf_by_rangf[gf], irf_ids)
                                        levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                                        irf_param_random = self.irf_params_random_base[response][gf][family][irf_param_name]
                                        irf_param_random_mean = tf.reduce_mean(irf_param_random, axis=0, keepdims=True)
                                        irf_param_random -= irf_param_random_mean
                                        irf_param_random_summary = self.irf_params_random_base_summary[response][gf][family][irf_param_name]
                                        irf_param_random_summary_mean = tf.reduce_mean(irf_param_random_summary, axis=0, keepdims=True)
                                        irf_param_random_summary -= irf_param_random_summary_mean

                                        self._regularize(
                                            irf_param_random,
                                            regtype='ranef',
                                            var_name='%s_%s_by_%s' % (irf_param_name, sn(response), sn(gf))
                                        )

                                        for j, irf_id in enumerate(irf_ids_ran):
                                            if irf_id in irf_ids_ran:
                                                if response not in self.irf_params_random:
                                                    self.irf_params_random[response] = {}
                                                if gf not in self.irf_params_random[response]:
                                                    self.irf_params_random[response][gf] = {}
                                                if irf_id not in self.irf_params_random[response][gf]:
                                                    self.irf_params_random[response][gf][irf_id] = {}
                                                if irf_param_name not in self.irf_params_random[response][gf][irf_id]:
                                                    self.irf_params_random[response][gf][irf_id][irf_param_name] = {}
                                                if response not in self.irf_params_random_summary:
                                                    self.irf_params_random_summary[response] = {}
                                                if gf not in self.irf_params_random_summary[response]:
                                                    self.irf_params_random_summary[response][gf] = {}
                                                if irf_id not in self.irf_params_random_summary[response][gf]:
                                                    self.irf_params_random_summary[response][gf][irf_id] = {}
                                                if irf_param_name not in self.irf_params_random_summary[response][gf][irf_id]:
                                                    self.irf_params_random_summary[response][gf][irf_id][irf_param_name] = {}

                                                for k, response_param in enumerate(response_params):
                                                    dim_names = self._expand_param_name_by_dim(response, response_param)
                                                    for l, dim_name in enumerate(dim_names):
                                                        val = irf_param_random[:, j, k, l]
                                                        val_summary = irf_param_random_summary[:, j, k, l]
                                                        tf.summary.histogram(
                                                            'by_%s/%s/%s/%s_%s' % (
                                                                sn(gf),
                                                                sn(irf_id),
                                                                irf_param_name,
                                                                sn(dim_name),
                                                                sn(response)
                                                            ),
                                                            val_summary,
                                                            collections=['random']
                                                        )
                                                        self.irf_params_random[response][gf][irf_id][irf_param_name][dim_name] = val
                                                        self.irf_params_random_summary[response][gf][irf_id][irf_param_name][dim_name] = val_summary

                                        irf_param_random = self._scatter_along_axis(
                                            irfs_ix,
                                            self._scatter_along_axis(
                                                levels_ix,
                                                irf_param_random,
                                                [self.rangf_n_levels[i], len(irfs_ix), nparam_response, ndim]
                                            ),
                                            [self.rangf_n_levels[i], len(irf_ids), nparam_response, ndim],
                                            axis=1
                                        )
                                        irf_param_random_summary = self._scatter_along_axis(
                                            irfs_ix,
                                            self._scatter_along_axis(
                                                levels_ix,
                                                irf_param_random_summary,
                                                [self.rangf_n_levels[i], len(irfs_ix), nparam_response, ndim]
                                            ),
                                            [self.rangf_n_levels[i], len(irf_ids), nparam_response, ndim],
                                            axis=1
                                        )

                                        irf_param = irf_param + tf.gather(irf_param_random, self.Y_gf[:, i], axis=0)
                                        irf_param_summary = irf_param_summary + tf.gather(irf_param_random_summary, self.Y_gf[:, i], axis=0)

                            if irf_param_lb is not None and irf_param_ub is None:
                                irf_param = irf_param_lb + self.constraint_fn(irf_param) + self.epsilon
                                irf_param_summary = irf_param_lb + self.constraint_fn(irf_param_summary) + self.epsilon
                            elif irf_param_lb is None and irf_param_ub is not None:
                                irf_param = irf_param_ub - self.constraint_fn(irf_param) - self.epsilon
                                irf_param_summary = irf_param_ub - self.constraint_fn(irf_param_summary) - self.epsilon
                            elif irf_param_lb is not None and irf_param_ub is not None:
                                irf_param = self._sigmoid(irf_param, a=irf_param_lb, b=irf_param_ub) * (1 - 2 * self.epsilon) + self.epsilon
                                irf_param_summary = self._sigmoid(irf_param_summary, a=irf_param_lb, b=irf_param_ub) * (1 - 2 * self.epsilon) + self.epsilon

                            for j, irf_id in enumerate(irf_ids):
                                if irf_param_name in trainable[irf_id]:
                                    if irf_id not in self.irf_params[response]:
                                        self.irf_params[response][irf_id] = {}
                                    if irf_id not in self.irf_params_summary[response]:
                                        self.irf_params_summary[response][irf_id] = {}
                                    # id is -3 dimension
                                    self.irf_params[response][irf_id][irf_param_name] = irf_param[..., j, :, :]
                                    self.irf_params_summary[response][irf_id][irf_param_name] = irf_param_summary[..., j, :, :]

    def _initialize_parameter_tables(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                super(CDR, self)._initialize_parameter_tables()

                # Fixed
                for response in self.coefficient_fixed:
                    for coef_name in self.coefficient_fixed[response]:
                        coef_name_str = 'coefficient_' + coef_name
                        for dim_name in self.coefficient_fixed[response][coef_name]:
                            self.parameter_table_fixed_types.append(coef_name_str)
                            self.parameter_table_fixed_responses.append(response)
                            self.parameter_table_fixed_response_params.append(dim_name)
                            self.parameter_table_fixed_values.append(
                                self.coefficient_fixed[response][coef_name][dim_name]
                            )
                for response in self.interaction_fixed:
                    for interaction_name in self.interaction_fixed[response]:
                        interaction_name_str = 'interaction_' + interaction_name
                        for dim_name in self.interaction_fixed[response][interaction_name]:
                            self.parameter_table_fixed_types.append(interaction_name_str)
                            self.parameter_table_fixed_responses.append(response)
                            self.parameter_table_fixed_response_params.append(dim_name)
                            self.parameter_table_fixed_values.append(
                                self.interaction_fixed[response][interaction_name][dim_name]
                            )
                for response in self.irf_params_fixed:
                    for irf_id in self.irf_params_fixed[response]:
                        for irf_param in self.irf_params_fixed[response][irf_id]:
                            irf_str = irf_param + '_' + irf_id
                            for dim_name in self.irf_params_fixed[response][irf_id][irf_param]:
                                self.parameter_table_fixed_types.append(irf_str)
                                self.parameter_table_fixed_responses.append(response)
                                self.parameter_table_fixed_response_params.append(dim_name)
                                self.parameter_table_fixed_values.append(
                                    self.irf_params_fixed[response][irf_id][irf_param][dim_name]
                                )

                # Random
                for response in self.coefficient_random:
                    for r, gf in enumerate(self.rangf):
                        if gf in self.coefficient_random[response]:
                            levels = sorted(self.rangf_map_ix_2_levelname[r][:-1])
                            for coef_name in self.coefficient_random[response][gf]:
                                coef_name_str = 'coefficient_' + coef_name
                                for dim_name in self.coefficient_random[response][gf][coef_name]:
                                    for l, level in enumerate(levels):
                                        self.parameter_table_random_types.append(coef_name_str)
                                        self.parameter_table_random_responses.append(response)
                                        self.parameter_table_random_response_params.append(dim_name)
                                        self.parameter_table_random_rangf.append(gf)
                                        self.parameter_table_random_rangf_levels.append(level)
                                        self.parameter_table_random_values.append(
                                            self.coefficient_random[response][gf][coef_name][dim_name][l]
                                        )
                for response in self.interaction_random:
                    for r, gf in enumerate(self.rangf):
                        if gf in self.interaction_random[response]:
                            levels = sorted(self.rangf_map_ix_2_levelname[r][:-1])
                            for interaction_name in self.interaction_random[response][gf]:
                                interaction_name_str = 'interaction_' + interaction_name
                                for dim_name in self.interaction_random[response][gf][interaction_name]:
                                    for l, level in enumerate(levels):
                                        self.parameter_table_random_types.append(interaction_name_str)
                                        self.parameter_table_random_responses.append(response)
                                        self.parameter_table_random_response_params.append(dim_name)
                                        self.parameter_table_random_rangf.append(gf)
                                        self.parameter_table_random_rangf_levels.append(level)
                                        self.parameter_table_random_values.append(
                                            self.interaction_random[response][gf][interaction_name][dim_name][l]
                                        )
                for response in self.irf_params_fixed:
                    for r, gf in enumerate(self.rangf):
                        if gf in self.irf_params_fixed[response]:
                            levels = sorted(self.rangf_map_ix_2_levelname[r][:-1])
                            for irf_id in self.irf_params_fixed[response]:
                                for irf_param in self.irf_params_fixed[response][irf_id]:
                                    irf_str = irf_param + '_' + irf_id
                                    for dim_name in self.irf_params_fixed[response][irf_id][irf_param]:
                                        for l, level in enumerate(levels):
                                            self.parameter_table_fixed_types.append(irf_str)
                                            self.parameter_table_fixed_responses.append(response)
                                            self.parameter_table_fixed_response_params.append(dim_name)
                                            self.parameter_table_fixed_values.append(
                                                self.irf_params_random[response][gf][irf_id][irf_param][dim_name][l]
                                            )

    def _initialize_impulses(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.irf_impulses = {}
                for name in self.terminal_names:
                    t = self.node_table[name]
                    impulse_name = t.impulse.name()
                    impulse_ix = names2ix(impulse_name, self.impulse_names)

                    if t.p.family == 'DiracDelta':
                        impulse = tf.gather(self.X_processed, impulse_ix, axis=2)[:, -1, :]

                        # Zero-out impulses to DiracDelta that are not response-aligned
                        impulse *= self.is_response_aligned[:, impulse_ix[0]:impulse_ix[0]+1]
                    else:
                        impulse = tf.gather(self.X_processed, impulse_ix, axis=2)

                    self.irf_impulses[name] = impulse

    def _initialize_irf_lambdas(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.irf_lambdas = {}
                if self.future_length: # Non-causal
                    support_lb = None
                else: # Causal
                    support_lb = 0.
                support_ub = None

                def exponential(**params):
                    return exponential_irf_factory(
                        **params,
                        session=self.sess
                    )

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['ExpRateGT1'] = exponential

                def gamma(**params):
                    return gamma_irf_factory(
                        **params,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['SteepGamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['GammaKgt1'] = gamma
                self.irf_lambdas['HRFSingleGamma'] = gamma

                def shifted_gamma_lambdas(**params):
                    return shifted_gamma_irf_factory(
                        **params,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma_lambdas
                self.irf_lambdas['ShiftedGammaShapeGT1'] = shifted_gamma_lambdas
                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma_lambdas

                def normal(**params):
                    return normal_irf_factory(
                        **params,
                        support_lb=support_lb,
                        support_ub=support_ub,
                        session=self.sess
                    )

                self.irf_lambdas['Normal'] = normal

                def skew_normal(**params):
                    return skew_normal_irf_factory(
                        **params,
                        support_lb=support_lb,
                        support_ub=self.t_delta_limit.astype(dtype=self.FLOAT_NP) if support_ub is None else support_ub,
                        session=self.sess
                    )

                self.irf_lambdas['SkewNormal'] = skew_normal

                def emg(**kwargs):
                    return emg_irf_factory(
                        **kwargs,
                        support_lb=support_lb,
                        support_ub=support_ub,
                        session=self.sess
                    )

                self.irf_lambdas['EMG'] = emg

                def beta_prime(**kwargs):
                    return beta_prime_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess
                    )

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(**kwargs):
                    return shifted_beta_prime_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess
                    )

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

                def double_gamma_1(**kwargs):
                    return double_gamma_1_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma1'] = double_gamma_1

                def double_gamma_2(**kwargs):
                    return double_gamma_2_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma2'] = double_gamma_2

                def double_gamma_3(**kwargs):
                    return double_gamma_3_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma3'] = double_gamma_3

                def double_gamma_4(**kwargs):
                    return double_gamma_4_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma4'] = double_gamma_4

                def double_gamma_5(**kwargs):
                    return double_gamma_5_irf_factory(
                        **kwargs,
                        support_ub=support_ub,
                        session=self.sess,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma'] = double_gamma_5
                self.irf_lambdas['HRFDoubleGamma5'] = double_gamma_5

    def _initialize_LCG_irf(
            self,
            bases
    ):
        if self.future_length: # Non-causal
            support_lb = None
        else: # Causal
            support_lb = 0.
        support_ub = None

        def f(
                bases=bases,
                int_type=self.INT_TF,
                float_type=self.FLOAT_TF,
                session=self.sess,
                support_lb=support_lb,
                support_ub=support_ub,
                **params
        ):
            return LCG_irf_factory(
                bases,
                int_type=int_type,
                float_type=float_type,
                session=session,
                support_lb=support_lb,
                support_ub=support_ub,
                **params
            )

        return f

    def _get_irf_lambdas(self, family):
        if family in self.irf_lambdas:
            return self.irf_lambdas[family]
        elif Formula.is_LCG(family):
            bases = Formula.bases(family)
            return self._initialize_LCG_irf(
                bases
            )
        else:
            raise ValueError('No IRF lamdba found for family "%s"' % family)

    def _initialize_irfs(self, t, response):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if response not in self.irf:
                    self.irf[response] = {}
                if t.family is None:
                    self.irf[response][t.name()] = []
                elif t.family == 'Terminal':
                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[response][t.name()] = self.irf[response][t.p.name()][:]
                elif t.family == 'DiracDelta':
                    assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in CDR formula strings'
                    assert not t.impulse == 'rate', '"rate" is a reserved keyword in CDR formula strings and cannot be used under DiracDelta'
                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[response][t.name()] = self.irf[response][t.p.name()][:]
                else:
                    params = self.irf_params[response][t.irf_id()]
                    atomic_irf = self._get_irf_lambdas(t.family)(**params)
                    if t.p.name() in self.irf:
                        irf = self.irf[response][t.p.name()][:] + [atomic_irf]
                    else:
                        irf = [atomic_irf]
                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[response][t.name()] = irf

                for c in t.children:
                    self._initialize_irfs(c, response)

    def _initialize_convolutions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.convolutions = {}
                for i, response in enumerate(self.response_names):
                    self.convolutions[response] = {}
                    for name in self.terminal_names:
                        t = self.node_table[name]
                        impulse_name = self.terminal2impulse[name]
                        impulse_ix = names2ix(impulse_name, self.impulse_names)[0]

                        if t.p.family == 'DiracDelta':
                            if self.use_distributional_regression:
                                nparam = self.get_response_nparam(response)
                            else:
                                nparam = 1
                            ndim = self.get_response_ndim(response)
                            impulse = self.irf_impulses[name][..., None]
                            impulse = tf.tile(impulse, [1, nparam, ndim])
                            out = impulse
                        else:
                            if t.cont:
                                # Create a continuous piecewise linear function
                                # that interpolates between points in the impulse history.
                                # Reverse because the history contains time offsets in descending order
                                knot_location = tf.reverse(
                                    tf.transpose(self.t_delta[:,:,impulse_ix:impulse_ix+1], [0, 2, 1]),
                                    axis=[-1]
                                )
                                knot_amplitude = tf.reverse(
                                    tf.transpose(self.irf_impulses[name], [0, 2, 1]),
                                    axis=[-1]
                                )
                                impulse_resampler = self._piecewise_linear_interpolant(knot_location, knot_amplitude)
                                t_delta = tf.linspace(
                                    self.interp_step * (self.history_length + self.future_length -1),
                                    0,
                                    self.history_length + self.future_length
                                )[None, ...]
                                t_delta = tf.tile(t_delta, [tf.shape(self.t_delta)[0], 1])
                                impulse = impulse_resampler(t_delta)
                                impulse *= self.interp_step
                            else:
                                impulse = self.irf_impulses[name]
                                t_delta = self.t_delta[:,:,impulse_ix]

                            irf = self.irf[response][name]
                            if len(irf) > 1:
                                irf = self._compose_irf(irf)
                            else:
                                irf = irf[0]

                            # Put batch dim last
                            t_delta = tf.transpose(t_delta, [1, 0])
                            # Add broadcasting for response nparam, ndim
                            t_delta = t_delta[..., None, None]
                            # Impulse already has an extra dim
                            impulse = impulse[..., None]
                            # Apply IRF
                            irf_seq = irf(t_delta)
                            # Put batch dim first
                            irf_seq = tf.transpose(irf_seq, [1, 0, 2, 3])

                            out = tf.reduce_sum(impulse * irf_seq, axis=1)

                        self.convolutions[response][name] = out

    def _sum_interactions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.interaction_names) > 0:
                    self.summed_interactions = {}
                    for response in self.response_names:
                        interaction_ix = np.arange(len(self.interaction_names))
                        interaction_coefs = tf.gather(self.interaction[response], interaction_ix, axis=1)
                        interaction_inputs = []
                        for i, interaction in enumerate(self.interaction_list):
                            assert interaction.name() == self.interaction_names[i], 'Mismatched sort order between self.interaction_names and self.interaction_list. This should not have happened, so please report it in issue tracker on Github.'
                            irf_input_names = [x.name() for x in interaction.irf_responses()]

                            inputs_cur = None

                            if len(irf_input_names) > 0:
                                irf_input_ix = names2ix(irf_input_names, self.terminal_names)
                                irf_inputs = tf.gather(self.X_conv[response], irf_input_ix, axis=1)
                                inputs_cur = tf.reduce_prod(irf_inputs, axis=1)

                            non_irf_input_names = [x.name() for x in interaction.non_irf_responses()]
                            if len(non_irf_input_names):
                                non_irf_input_ix = names2ix(non_irf_input_names, self.impulse_names)
                                non_irf_inputs = tf.gather(self.X_processed[:,-1,:], non_irf_input_ix, axis=1)
                                non_irf_inputs = tf.reduce_prod(non_irf_inputs, axis=1)
                                if inputs_cur is not None:
                                    inputs_cur *= non_irf_inputs
                                else:
                                    inputs_cur = non_irf_inputs

                            interaction_inputs.append(inputs_cur)
                        interaction_inputs = tf.stack(interaction_inputs, axis=1)
                        self.summed_interactions[response] = tf.reduce_sum(interaction_coefs * interaction_inputs, axis=1)




    ######################################################
    #
    #  Public network initialization methods.
    #
    ######################################################

    def initialize_coefficient(self, response, coef_ids=None, ran_gf=None):
        """
        Add coefficients for a given response variable.
        Must be called for each response variable.
        This method should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param response: ``str``: name of response variable
        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of ``len(coef_ids)`` trainable weights. For random coefficients, should return batch-length matrix of trainable weights with ``len(coef_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

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
                    coefficient = tf.Variable(
                        tf.zeros([ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s' % sn(response)
                    )
                    coefficient_summary = coefficient
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    coefficient = tf.Variable(
                        tf.zeros([rangf_n_levels, ncoef, nparam, ndim], dtype=self.FLOAT_TF),
                        name='coefficient_%s_by_%s' % (sn(response), sn(ran_gf))
                    )
                    coefficient_summary = coefficient

                # shape: (?rangf_n_levels, ncoef, nparam, ndim)

                return coefficient, coefficient_summary

    def initialize_interaction(self, response, interaction_ids=None, ran_gf=None):
        """
        Add (response-level) interactions for a given response variable.
        Must be called for each response variable.
        This method should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param response: ``str``: name of response variable
        :param coef_ids: ``list`` of ``str``: List of interaction IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random interaction (if ``None``, constructs a fixed interaction)
        :return: 2-tuple of ``Tensor`` ``(interaction, interaction_summary)``; ``interaction`` is the interaction for use by the model. ``interaction_summary`` is an identically-shaped representation of the current interaction values for logging and plotting (can be identical to ``interaction``). For fixed interactions, should return a vector of ``len(interaction_ids)`` trainable weights. For random interactions, should return batch-length matrix of trainable weights with ``len(interaction_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

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
                    interaction = tf.Variable(
                        tf.zeros([ninter, nparam, ndim], dtype=self.FLOAT_TF),
                        name='interaction_%s' % sn(response)
                    )
                    interaction_summary = interaction
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    interaction = tf.Variable(
                        tf.zeros([rangf_n_levels, ninter, nparam, ndim], dtype=self.FLOAT_TF),
                        name='interaction_%s_by_%s' % (sn(response), sn(ran_gf))
                    )
                    interaction_summary = interaction

                # shape: (?rangf_n_levels, ninter, nparam, ndim)

                return interaction, interaction_summary

    def initialize_irf_param(self, response, family, param_name, ran_gf=None):
        """
        Add IRF parameters in the unconstrained space for a given response variable.
        CDR will apply appropriate constraint transformations as needed.
        This method should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param response: ``str``: name of response variable
        :param family: ``str``; Name of IRF kernel family (e.g. ``Normal``)
        :param param_name: ``str``; Name of parameter (e.g. ``"alpha"``)
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(param, param_summary)``; ``param`` is the parameter for use by the model. ``param_summary`` is an identically-shaped representation of the current param values for logging and plotting (can be identical to ``param``). For fixed params, should return a vector of ``len(trainable_ids)`` trainable weights. For random params, should return batch-length matrix of trainable weights with ``len(trainable_ids)``. Weights should be initialized around **mean** (if fixed) or ``0`` (if random).
        """

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
                        param = tf.Variable(
                            tf.ones([nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF) * tf.constant(mean[..., None, None], dtype=self.FLOAT_TF),
                            name=sn('%s_%s_%s' % (param_name, '-'.join(trainable_ids), sn(response)))
                        )
                        param_summary = param
                    else:
                        param = param_summary = None
                else:
                    rangf_n_levels = self.rangf_n_levels[self.rangf.index(ran_gf)] - 1
                    irf_ids_gf = self.irf_by_rangf[ran_gf]
                    trainable_ids = [x for x in irf_ids_all if (param_name in param_trainable[x] and x in irf_ids_gf)]
                    nirf = len(trainable_ids)

                    if nirf:
                        param = tf.Variable(
                            tf.zeros([rangf_n_levels, nirf, response_nparam, response_ndim], dtype=self.FLOAT_TF),
                            name=sn('%s_%s_%s_by_%s' % (param_name, '-'.join(trainable_ids), sn(response), sn(ran_gf)))
                        )
                        param_summary = param
                    else:
                        param = param_summary = None

                # shape: (?rangf_n_levels, nirf, nparam, ndim)

                return param, param_summary

    def initialize_model(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._compile_coefficients()
                self._compile_interactions()
                self._compile_irf_params()
                self._initialize_irf_lambdas()
                for response in self.response_names:
                    self._initialize_irfs(self.t, response)
                self._initialize_impulses()
                self._initialize_convolutions()

    def compile_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.X_conv = {}
                for response in self.response_names:
                    convolutions = [self.convolutions[response][x] for x in self.terminal_names]
                    if len(convolutions) > 0:
                        X_conv = tf.stack(convolutions, axis=1)
                    else:
                        X_conv = tf.zeros((1, 1), dtype=self.FLOAT_TF)


                    coef_names = [self.node_table[x].coef_id() for x in self.terminal_names]
                    coef_ix = names2ix(coef_names, self.coef_names)
                    coef = tf.gather(self.coefficient[response], coef_ix, axis=1)
                    X_conv = X_conv * coef
                    self.X_conv[response] = X_conv

                    output = tf.reduce_sum(X_conv, axis=1)
                    self.output[response] = output

                    if len(self.interaction_names) > 0:
                        self._sum_interactions()





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _new_irf(self, irf_lambda, params):
        irf = irf_lambda(params)
        def new_irf(x):
            return irf(x)
        return new_irf

    def _compose_irf(self, f_list):
        if not isinstance(f_list, list):
            f_list = [f_list]
        with self.sess.as_default():
            with self.sess.graph.as_default():
                f = f_list[0](self.interpolation_support)[..., 0]
                for g in f_list[1:]:
                    _f = tf.spectral.rfft(f)
                    _g = tf.spectral.rfft(g(self.interpolation_support)[..., 0])
                    f = tf.spectral.irfft(
                        _f * _g
                    ) * self.max_tdelta_batch / tf.cast(self.n_interp, dtype=self.FLOAT_TF)

                def make_composed_irf(seq):
                    def composed_irf(t):
                        squeezed = 0
                        while t.shape[-1] == 1:
                            t = tf.squeeze(t, axis=-1)
                            squeezed += 1
                        ix = tf.cast(tf.round(t * tf.cast(self.n_interp - 1, self.FLOAT_TF) / self.max_tdelta_batch), dtype=self.INT_TF)
                        row_ix = tf.tile(tf.range(tf.shape(t)[0])[..., None], [1, tf.shape(t)[1]])
                        ix = tf.stack([row_ix, ix], axis=-1)
                        out = tf.gather_nd(seq, ix)

                        for _ in range(squeezed):
                            out = out[..., None]

                        return out

                    return composed_irf

                return make_composed_irf(f)

    def _get_mean_init_vector(self, irf_ids, param_name, irf_param_init, default=0.):
        mean = np.zeros(len(irf_ids))
        for i in range(len(irf_ids)):
            mean[i] = irf_param_init[irf_ids[i]].get(param_name, default)
        return mean

    def _process_mean(self, mean, lb=None, ub=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if lb is not None and ub is None:
                    # Lower-bounded support only
                    mean = self.constraint_fn_inv_np(mean - lb - self.epsilon)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    mean = self.constraint_fn_inv_np(-(mean - ub + self.epsilon))
                elif lb is not None and ub is not None:
                    # Finite-interval bounded support
                    mean = self._logit_np(mean, lb, ub)

        return mean, lb, ub

    def _get_trainable_untrainable_ix(self, param_name, ids, trainable=None):
        if trainable is None:
            trainable_ix = np.array(list(range(len(ids))), dtype=self.INT_NP)
            untrainable_ix = []
        else:
            trainable_ix = []
            untrainable_ix = []
            for i in range(len(ids)):
                name = ids[i]
                if param_name in trainable[name]:
                    trainable_ix.append(i)
                else:
                    untrainable_ix.append(i)
            trainable_ix = np.array(trainable_ix, dtype=self.INT_NP)
            untrainable_ix = np.array(untrainable_ix, dtype=self.INT_NP)

        return trainable_ix, untrainable_ix





    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    def report_settings(self, indent=0):
        out = super(CDR, self).report_settings(indent=indent)
        for kwarg in CDR_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out



    ######################################################
    #
    #  High-level methods for training, prediction,
    #  and plotting
    #
    ######################################################


    def run_train_step(self, feed_dict, verbose=True):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                to_run = [self.train_op, self.ema_op]
                to_run += [self.response_params_ema_ops[x] for x in self.response_params_ema_ops]

                to_run += [self.loss_func, self.reg_loss]
                to_run_names = ['loss', 'reg_loss']
                if self.is_bayesian:
                    to_run.append(self.kl_loss)
                    to_run_names.append('kl_loss')

                out = self.sess.run(
                    to_run,
                    feed_dict=feed_dict
                )

                out_dict = {x: y for x, y in zip(to_run_names, out[-len(to_run_names):])}

                return out_dict
