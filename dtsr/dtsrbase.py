import os
from collections import defaultdict
import textwrap
from numpy import inf
import pandas as pd
import scipy.stats
import time as pytime

from .formula import *
from .kwargs import DTSR_INITIALIZATION_KWARGS
from .util import *
from .data import build_DTSR_impulses, corr_dtsr, get_first_last_obs_lists
from .plot import *
from .interpolate_spline import interpolate_spline

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

pd.options.mode.chained_assignment = None


######################################################
#
#  ABSTRACT DTSR CLASS
#
######################################################

def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def normalize_irf(irf, support, session=None, epsilon=4*np.finfo('float32').eps):
    def f(x, support=support, session=session, epsilon=epsilon):
        session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                out = irf(x)
                _support = support
                while len(_support.shape) < len(out.shape):
                    _support = _support[None, ...]
                n_time_units = tf.squeeze(tf.cast(_support[...,-1,0], dtype=tf.float32))
                n_time_points = tf.cast(tf.shape(_support)[-2], dtype=tf.float32)
                irf_samples = irf(_support)
                normalization_constant = (tf.reduce_sum(irf_samples, axis=-2, keepdims=True)) * (n_time_units / n_time_points)
                e = tf.ones_like(normalization_constant) * epsilon
                tf.where(tf.not_equal(normalization_constant, 0.), normalization_constant, e)
                out /= normalization_constant
                return out

    return f


def unnormalized_gamma(alpha, beta):
    return lambda x: x ** (alpha - 1) * tf.exp(-beta * x)


def unnormalized_gaussian(mu, sigma2):
    return lambda x: tf.exp(-(x - mu) ** 2 / sigma2)


def exponential_irf(params, integral_ub=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            beta = params[:, 0:1]

            dist = tf.contrib.distributions.Exponential(beta)
            pdf = dist.prob
            cdf = dist.cdf

            if integral_ub is None:
                ub = None
            else:
                ub = cdf(integral_ub)

            if integral_ub is None:
                return lambda x, pdf=pdf: pdf(x)
            else:
                return lambda x, pdf=pdf, cdf=cdf, ub=ub: pdf(x) / ub


def gamma_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = params[:, 0:1]
            beta = params[:, 1:2]
            
            dist = tf.contrib.distributions.Gamma(
                concentration=alpha,
                rate=beta,
                validate_args=validate_irf_args
            )
            pdf = dist.prob
            cdf = dist.cdf

            if integral_ub is None:
                ub = None
            else:
                ub = cdf(integral_ub)

            if integral_ub is None:
                return lambda x, pdf=pdf, epsilon=epsilon: pdf(x + epsilon)
            else:
                return lambda x, pdf=pdf, cdf=cdf, ub=ub, epsilon=epsilon: pdf(x + epsilon) / ub


def shifted_gamma_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = params[:, 0:1]
            beta = params[:, 1:2]
            delta = params[:, 2:3]
            
            dist = tf.contrib.distributions.Gamma(
                concentration=alpha,
                rate=beta,
                validate_args=validate_irf_args
            )
            pdf = dist.prob
            cdf = dist.cdf

            print(integral_ub)
            print(np.dtype(integral_ub))

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub)

            return lambda x, pdf=pdf, cdf=cdf, delta=delta, ub=ub, epsilon=epsilon: pdf(x - delta + epsilon) / (ub - cdf(- delta + epsilon) + epsilon)


def normal_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = params[:, 0:1]
            sigma = params[:, 1:2]

            dist = tf.contrib.distributions.Normal(
                mu,
                sigma
            )
            pdf = dist.prob
            cdf = dist.cdf

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub)
                
            return lambda x, pdf=pdf, cdf=cdf, ub=ub: pdf(x) / (ub - cdf(0.) + epsilon)


def skew_normal_irf(params, session=None, epsilon=4*np.finfo('float32').eps):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = params[:, 0:1]
            sigma = params[:, 1:2]
            alpha = params[:, 2:3]
            
            stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
            stdnorm_pdf = stdnorm.prob
            stdnorm_cdf = stdnorm.cdf

            return lambda x, mu=mu, sigma=sigma, alpha=alpha, pdf=stdnorm_pdf, cdf=stdnorm_cdf: (stdnorm_pdf((x - mu) / sigma) * stdnorm_cdf(alpha * (x - mu) / sigma))


def emg_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = params[:, 0:1]
            sigma = params[:, 1:2]
            L = params[:, 2:3]

            def cdf(x):
                return tf.contrib.distributions.Normal(
                    loc=0.,
                    scale=L*sigma
                )(L * (x - mu))

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub)

            return lambda x, L=L, mu=mu, sigma=sigma, ub=ub, cdf=cdf, epsilon=epsilon: (L / 2 * tf.exp(0.5 * L * (2. * mu + L * sigma ** 2. - 2. * x)) * tf.erfc(
                (mu + L * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))) / (ub - cdf(0.) + epsilon)


def beta_prime_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = params[:, 1:2]
            beta = params[:, 2:3]

            def cdf(x):
                return tf.betainc(alpha, beta, x / (1+x)) * tf.exp(tf.lbeta(alpha, beta))
            
            if integral_ub is None:
                ub = 1
            else:
                ub = cdf(integral_ub)

            return lambda x, alpha=alpha, beta=beta, ub=ub, epsilon=epsilon: ((x + epsilon) ** (alpha - 1.) * (1. + (x + epsilon)) ** (-alpha - beta)) / (ub - cdf(epsilon) + epsilon)


def shifted_beta_prime_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha = params[:, 0:1]
            beta = params[:, 1:2]
            delta = params[:, 2:3]

            def cdf(x):
                return tf.betainc(alpha, beta, (x-delta) / (1+x-delta)) * tf.exp(tf.lbeta(alpha, beta))
            
            if integral_ub is None:
                ub = 1
            else:
                ub = cdf(integral_ub)

            return lambda x, alpha=alpha, beta=beta, delta=delta, ub=ub, epsilon=epsilon: ((x - delta + epsilon) ** (alpha - 1) * (1 + (x - delta + epsilon)) ** (-alpha - beta)) / (ub - cdf(-delta + epsilon) + epsilon)


def double_gamma_1_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = 6.
            alpha_undershoot = 16.
            beta_main = params[:, 0:1]
            beta_undershoot = beta_main
            c = 1. / 6.

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

            if integral_ub is None:
                denom = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)
            else:
                denom = 1 - c

            return lambda x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, denom=denom, epsilon=epsilon: (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / denom


def double_gamma_2_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = params[:, 0:1]
            alpha_undershoot = alpha_main + 10.
            beta_main = params[:, 1:2]
            beta_undershoot = beta_main
            c = 1. / 6.

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

            if integral_ub is None:
                denom = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)
            else:
                denom = 1 - c

            return lambda x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, denom=denom, epsilon=epsilon: (pdf_main(
                x + epsilon) - c * pdf_undershoot(x + epsilon)) / denom


def double_gamma_3_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = params[:, 0:1]
            alpha_undershoot = alpha_main + 10.
            beta_main = params[:, 1:2]
            beta_undershoot = beta_main
            c = params[:, 2:3]

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

            if integral_ub is None:
                denom = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)
            else:
                denom = 1 - c

            return lambda x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, denom=denom, epsilon=epsilon: (pdf_main(
                x + epsilon) - c * pdf_undershoot(x + epsilon)) / denom


def double_gamma_4_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = params[:, 0:1]
            alpha_undershoot = params[:, 1:2]
            beta_main = params[:, 2:3]
            beta_undershoot = beta_main
            c = params[:, 3:4]

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

            if integral_ub is None:
                denom = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)
            else:
                denom = 1 - c

            return lambda x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, denom=denom, epsilon=epsilon: (pdf_main(
                x + epsilon) - c * pdf_undershoot(x + epsilon)) / denom


def double_gamma_5_irf(params, integral_ub=None, session=None, epsilon=4*np.finfo('float32').eps, validate_irf_args=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            alpha_main = params[:, 0:1]
            alpha_undershoot = params[:, 1:2]
            beta_main = params[:, 2:3]
            beta_undershoot = params[:, 3:4]
            c = params[:, 4:5]

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

            if integral_ub is None:
                denom = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)
            else:
                denom = 1 - c

            return lambda x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, denom=denom, epsilon=epsilon: (pdf_main(
                x + epsilon) - c * pdf_undershoot(x + epsilon)) / denom


def piecewise_linear_interpolant(c, v, session=None):
    # c: knot locations, shape=[B, Q, K], B = batch, Q = query points or 1, K = n knots
    # v: knot values, shape identical to c
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if len(c.shape) == 1:
                # No batch or query dim
                c = c[None, None, ...]
            elif len(c.shape) == 2:
                # No query dim
                c = tf.expand_dims(c, axis=-2)
            elif len(c.shape) > 3:
                # Too many dims
                raise ValueError('Rank of knot location tensor c to piecewise resampler must be >= 1 and <= 3. Saw "%d"' % len(c.shape))
            if len(v.shape) == 1:
                # No batch or query dim
                v = v[None, None, ...]
            elif len(v.shape) == 2:
                # No query dim
                v = tf.expand_dims(v, axis=-2)
            elif len(v.shape) > 3:
                # Too many dims
                raise ValueError('Rank of knot amplitude tensor c to piecewise resampler must be >= 1 and <= 3. Saw "%d"' % len(v.shape))

            c_t = c[..., 1:]
            c_tm1 = c[..., :-1]
            y_t = v[..., 1:]
            y_tm1 = v[..., :-1]

            # Compute intercepts a_ and slopes b_ of line segments
            a_ = (y_t - y_tm1) / (c_t - c_tm1)
            valid = c_t > c_tm1
            a_ = tf.where(valid, a_, tf.zeros_like(a_))
            b_ = y_t - a_ * c_t

            # Handle points beyond final knot location (0 response)
            a_ = tf.concat([a_, tf.zeros_like(a_[..., -1:])], axis=-1)
            b_ = tf.concat([b_, tf.zeros_like(b_[..., -1:])], axis=-1)
            c_ = tf.concat([c, tf.ones_like(c[..., -1:]) * np.inf], axis=-1)

            def make_piecewise(a, b, c):
                def select_segment(x, c):
                    c_t = c[..., 1:]
                    c_tm1 = c[..., :-1]
                    select = tf.cast(tf.logical_and(x >= c_tm1, x < c_t), dtype=self.FLOAT_TF)
                    return select

                def piecewise(x):
                    select = select_segment(x, c)
                    # a_select = tf.reduce_sum(a * select, axis=-1, keepdims=True)
                    # b_select = tf.reduce_sum(b * select, axis=-1, keepdims=True)
                    # response = a_select * x + b_select
                    response = tf.reduce_sum((a * x + b) * select, axis=-1, keepdims=True)
                    return response

                return piecewise

            out = make_piecewise(a_, b_, c_)

            return out


def spline(c, v, dynamic_batch_dim, order, roughness_penalty=0., int_type=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if int_type is None:
                INT_TF = getattr(tf, 'int32')
            elif isinstance(int_type, str):
                INT_TF = getattr(tf, int_type)
            else:
                INT_TF = int_type

            def apply_spline(x):
                x_ = x
                if len(x_.shape) == 1:
                    x_ = x_[None, :, None]
                elif len(x_.shape) == 2:
                    x_ = x_[None, ...]
                if len(x_.shape) != 3:
                    raise ValueError('Query to spline IRF must be exactly rank 3')

                batch_max = tf.reduce_max(tf.stack([tf.shape(x_)[0], dynamic_batch_dim], axis=0))

                x_batch = tf.shape(x_)[0]
                x_tile = tf.cast(batch_max / x_batch, dtype=INT_TF)
                x_ = tf.cond(x_tile > 1, lambda: tf.tile(x_, [x_tile, 1, 1]), lambda: x_)

                splines = []

                for i in range(len(c)):
                    if order == 1:
                        interp = piecewise_linear_interpolant(c[i], v[i], session=session)(x_)
                    else:
                        c_ = c[i]
                        c_batch = tf.shape(c_)[0]
                        c_tile = tf.cast(batch_max / c_batch, dtype=INT_TF)
                        c_ = tf.tile(c_[..., None], [c_tile, 1, 1])

                        v_ = v[i]
                        y_batch = tf.shape(v_)[0]
                        y_tile = tf.cast(batch_max / y_batch, dtype=INT_TF)
                        v_ = tf.tile(v_[..., None], [y_tile, 1, 1])

                        interp = tf.where(
                            x_ <= c_[:,-1:],
                            interpolate_spline(
                                c_,
                                v_,
                                x_,
                                order,
                                regularization_weight=roughness_penalty
                            ),
                            tf.zeros_like(x_)
                        )

                        # interp = interpolate_spline(
                        #     c_,
                        #     y_,
                        #     x_,
                        #     order,
                        #     regularization_weight=roughness_penalty
                        # )

                    splines.append(interp)

                out = tf.concat(splines, axis=2)

                return out

            out = apply_spline

            return out


def kernel_smooth(c, v, b, epsilon=4 * np.finfo('float32').eps, session=None):
    def f(x, c=c, v=v, b=b, epsilon=epsilon, session=session):
        session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                _x = x
                if len(x.shape) == 1:
                    _x = _x[None, :, None]
                elif len(x.shape) == 2:
                    _x = _x[None, ...]
                if len(_x.shape) != 3:
                    raise ValueError('Query to kernel smooth IRF must be exactly rank 3')
                _x = tf.expand_dims(_x, axis=-2)

                _c = c
                _b = b
                _v = v

                while len(_c.shape) < len(x.shape):
                    _c = _c[None, ...]
                while len(_b.shape) < len(x.shape):
                    _b = _b[None, ...]
                while len(_v.shape) < len(x.shape):
                    _v = _v[None, ...]

                _c = tf.expand_dims(_c, axis=-3)
                _b = tf.expand_dims(_b, axis=-3)
                _v = tf.expand_dims(_v, axis=-3)

                dist = tf.contrib.distributions.Normal(
                    loc=_c,
                    scale=_b
                )

                r = dist.prob(_x - _c)

                num = tf.reduce_sum(r * _v, axis=-2)
                denom = tf.reduce_sum(r, axis=-2) + epsilon

                out = num / denom

                return out

    return f


def summed_gaussians(c, v, b, integral_ub=None, session=None):
    def f(x, c=c, v=v, b=b, session=session):
        session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                _x = x
                if len(x.shape) == 1:
                    _x = _x[None, :, None]
                elif len(x.shape) == 2:
                    _x = _x[None, ...]
                if len(_x.shape) != 3:
                    raise ValueError('Query to summed gaussians IRF must be exactly rank 3')
                _x = _x[..., None]
                _c = c
                _b = b
                _v = v

                while len(_c.shape) < len(x.shape):
                    _c = _c[None, ...]
                while len(_b.shape) < len(x.shape):
                    _b = _b[None, ...]
                while len(_v.shape) < len(x.shape):
                    _v = _v[None, ...]

                _c = tf.expand_dims(_c, axis=-3)
                _b = tf.expand_dims(_b, axis=-3)
                _v = tf.expand_dims(_v, axis=-3)

                _v *= _b * np.sqrt(2 * np.pi) # Rescale by Gaussian normalization constant

                dist = tf.contrib.distributions.Normal(
                    loc=_c,
                    scale=_b,
                )

                if integral_ub is None:
                    ub = 1.
                else:
                    ub = dist.cdf(integral_ub)

                unnormalized = tf.reduce_sum(dist.prob(_x) * _v, axis=-2)
                normalization_constant = tf.reduce_sum((ub - dist.cdf(0.)) * _v, axis=-2)
                normalized = unnormalized / normalization_constant

                return normalized

    return f


def nonparametric_smooth(
        method,
        params,
        bases,
        integral_ub=None,
        support=None,
        epsilon=4 * np.finfo('float32').eps,
        int_type=None,
        float_type=None,
        session=None,
        **kwargs
):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if int_type is None:
                INT_TF = getattr(tf, 'int32')
            elif isinstance(float_type, str):
                INT_TF = getattr(tf, int_type)
            else:
                INT_TF = int_type

            if float_type is None:
                FLOAT_TF = getattr(tf, 'float32')
            elif isinstance(float_type, str):
                FLOAT_TF = getattr(tf, float_type)
            else:
                FLOAT_TF = float_type

            # Build control point locations
            c = params[:, 0:bases - 1]

            # Build values at control points
            v = params[:, bases - 1:2 * (bases - 1)]

            if method.lower() == 'spline': # Pad appropriately
                c_endpoint_shape = [tf.shape(c)[0], 1, tf.shape(params)[2]]
                zero = tf.zeros(c_endpoint_shape, dtype=FLOAT_TF)
                c = tf.concat([zero, c], axis=1)
                # c = tf.cumsum(c, axis=1)

                c = tf.unstack(c, axis=2)

                if not kwargs['instantaneous']:
                    v = [zero] + v[:, :-1]
                v = tf.concat(v, axis=1)
                v = tf.unstack(v, axis=2)

                assert len(c) == len(
                    v), 'c and y coordinates of spline unpacked into lists of different lengths (%s and %s, respectively)' % (
                    len(c), len(v))

                dynamic_batch_dim = tf.shape(params)[0]

                f = spline(
                    c,
                    v,
                    dynamic_batch_dim,
                    kwargs['order'],
                    roughness_penalty=kwargs['roughness_penalty'],
                    int_type=INT_TF,
                    session=session
                )

                assert support is not None, 'Argument ``support`` must be provided for spline IRFs'
                f = normalize_irf(f, support, session=session, epsilon=epsilon)

            else:
                # Build scales at control points
                b = params[:, 2 * (bases - 1):]
                if method.lower() == 'kernel_smooth':
                    f = kernel_smooth(c, v, b, epsilon=epsilon, session=session)
                    assert support is not None, 'Argument ``support`` must be provided for kernel smooth IRFS'
                    f = normalize_irf(f, support, session=session, epsilon=epsilon)

                elif method.lower() == 'summed_gaussians':
                    f = summed_gaussians(c, v, b, integral_ub=integral_ub, session=session)

                else:
                    raise ValueError('Unrecognized non-parametric IRF type: %s' % method)

            return f


def corr(A, B):
    # Assumes A and B are n x a and n x b matrices and computes a x b pairwise correlations
    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)

    A_ss = (A_centered ** 2).sum(axis=0)
    B_ss = (B_centered ** 2).sum(axis=0)

    rho = np.dot(A_centered.T, B_centered) / np.sqrt(np.dot(A_ss[..., None], B_ss[None, ...]))
    rho = np.clip(rho, -1, 1)
    return rho


class DTSR(object):

    _INITIALIZATION_KWARGS = DTSR_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for DTSR. Bayesian (:ref:`dtsrbayes`) and MLE (:ref:`dtsrmle`) implementations inherit from ``DTSR``.
        ``DTSR`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``DTSR`` must implement the following instance methods:
        
            * ``initialize_intercept()``
            * ``initialize_coefficient()``
            * ``initialize_irf_param_unconstrained()``
            * ``initialize_joint_distribution()``
            * ``initialize_objective()``
            * ``run_conv_op()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            * ``run_train_step()``
            
        Additionally, if the subclass requires any keyword arguments beyond those provided by ``DTSR``, it must also implement ``__init__()``, ``_pack_metadata()`` and ``_unpack_metadata()`` to support model initialization, saving, and resumption, respectively.
        
        Example implementations of each of these methods can be found in the source code for :ref:`dtsrmle` and :ref:`dtsrbayes`.
        
    """
    _doc_args = """
        :param form_str: An R-style string representing the DTSR model formula.
        :param X: ``pandas`` table; matrix of independent variables, grouped by series and temporally sorted.
            ``X`` must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in ``X``
            * A column for each independent variable in the DTSR ``form_str`` provided at iniialization
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

    def __new__(cls, *args, **kwargs):
        if cls is DTSR:
            raise TypeError("DTSR is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, y, **kwargs):

        ## Store initialization settings
        self.form_str = form_str
        for kwarg in DTSR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        # Parse and store model data from formula
        form = Formula(self.form_str)
        form = form.categorical_transform(X)
        form = form.categorical_transform(y)
        self.form = form
        dv = form.dv
        rangf = form.rangf

        # Compute from training data
        self.n_train = len(y)
        self.y_train_mean = float(y[dv].mean())
        self.y_train_sd = float(y[dv].std())

        t_deltas = []
        first_obs, last_obs = get_first_last_obs_lists(y)
        for i, cols in enumerate(zip(first_obs, last_obs)):
            first_obs_cur, last_obs_cur = cols
            X_time = np.array(X[i].time, dtype=getattr(np, self.float_type))
            last_obs_cur = np.array(last_obs_cur, dtype=getattr(np, self.int_type))
            first_obs_cur = np.maximum(np.array(first_obs_cur, dtype=getattr(np, self.int_type)), last_obs_cur - self.history_length + 1)
            t_delta = (y.time - X_time[first_obs_cur])
            t_deltas.append(t_delta)
        t_deltas = np.concatenate(t_deltas, axis=0)
        self.t_delta_limit = np.percentile(t_deltas, 75)
        self.max_tdelta = t_deltas.max()

        if self.pc:
            self.src_impulse_names_norate = list(filter(lambda x: x != 'rate', self.form.t.impulse_names(include_interactions=True)))
            _, self.eigenvec, self.eigenval, self.impulse_means, self.impulse_sds = pca(X[self.src_impulse_names_norate])
        else:
            self.eigenvec = self.eigenval = self.impulse_means = self.impulse_sds = None

        ## Set up hash table for random effects lookup
        self.rangf_map_base = []
        self.rangf_n_levels = []
        for i in range(len(rangf)):
            gf = rangf[i]
            keys = np.sort(y[gf].astype('str').unique())
            vals, counts = np.unique(y[gf].astype('str'), return_counts=True)
            sd = counts.std()
            if np.isfinite(sd):
                mu = counts.mean()
                lb = mu - 2 * sd
                too_few = []
                for v, c in zip(vals, counts):
                    if c < lb:
                        too_few.append((v, c))
                if len(too_few) > 0:
                    report = '\nWARNING: Some random effects levels had fewer than 2 standard deviations (%.2f)\nbelow the mean number of data points per level (%.2f):\n' % (sd*2, mu)
                    for t in too_few:
                        report += ' ' * 4 + str(t[0]) + ': %d\n' % t[1]
                    report += 'Having too few instances for some levels can lead to degenerate random effects estimates.\nConsider filtering out these levels.\n\n'
                    sys.stderr.write(report)
            vals = np.arange(len(keys), dtype=getattr(np, self.int_type))
            rangf_map = pd.DataFrame({'id':vals},index=keys).to_dict()['id']
            self.rangf_map_base.append(rangf_map)
            self.rangf_n_levels.append(len(keys) + 1)

        self._initialize_session()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        ## Compute secondary data from intialization settings
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)

        assert not self.pc, 'The use of ``pc=True`` is not currently supported.'

        f = self.form
        self.dv = f.dv
        self.has_intercept = f.has_intercept
        self.rangf = f.rangf

        if np.isfinite(self.minibatch_size):
            self.n_train_minibatch = math.ceil(float(self.n_train) / self.minibatch_size)
            self.minibatch_scale = float(self.n_train) / self.minibatch_size
        else:
            self.n_train_minibatch = 1
            self.minibatch_scale = 1
        self.regularizer_losses = []
        self.regularizer_losses_names = []
        self.regularizer_losses_scales = []
        self.regularizer_losses_varnames = []

        # Initialize lookup tables of network objects
        self.irf_lambdas = {}
        self.irf_params_means = {} # {family: {param_name: mean_vector}}
        self.irf_params_means_unconstrained = {} # {family: {param_name: mean_init_vector}}
        self.irf_params_random_means = {} # {rangf: {family: {param_name: mean_vector}}}
        self.irf_params_lb = {} # {family: {param_name: value}}
        self.irf_params_ub = {} # {family: {param_name: value}}
        self.irf_params = {} # {irf_id: param_vector}
        self.irf_params_summary = {} # {irf_id: param_summary_vector}
        self.irf_params_fixed = {} # {irf_id: param_vector}
        self.irf_params_fixed_summary = {} # {irf_id: param_summary_vector}
        self.irf_params_fixed_base = {}  # {family: {param_name: param_vector}}
        self.irf_params_fixed_base_summary = {}  # {family: {param_name: param_summary_vector}}
        self.irf_params_random = {} # {rangf: {irf_id: param_matrix}}
        self.irf_params_random_summary = {} # {rangf: {irf_id: param_matrix}}
        self.irf_params_random_base = {}  # {rangf: {family: {irf_id: param_matrix}}
        self.irf_params_random_base_summary = {}  # {rangf: {family: {irf_id: param_summary_matrix}}
        self.irf = {}
        self.irf_plot = {}
        self.irf_mc = {}
        self.irf_integral_tensors = {}
        if self.pc:
            self.src_irf_plot = {}
            self.src_irf_mc = {}
            self.src_irf_integral_tensors = {}
        self.irf_impulses = {}
        self.convolutions = {}

        # Initialize model metadata

        if self.pc:
            # Initialize source tree metadata
            self.t_src = self.form.t
            t_src = self.t_src
            self.src_node_table = t_src.node_table()
            self.src_coef_names = t_src.coef_names()
            self.src_fixed_coef_names = t_src.fixed_coef_names()
            self.src_spline_coef_names = t_src.spline_coef_names()
            self.src_interaction_list = t_src.interactions()
            self.src_interaction_names = t_src.interaction_names()
            self.src_fixed_interaction_names = t_src.fixed_interaction_names()
            self.src_impulse_names = t_src.impulse_names(include_interactions=True)
            self.src_terminal_names = t_src.terminal_names()
            self.src_atomic_irf_names_by_family = t_src.atomic_irf_by_family()
            self.src_atomic_irf_family_by_name = {}
            for family in self.src_atomic_irf_names_by_family:
                for id in self.src_atomic_irf_names_by_family[family]:
                    assert id not in self.src_atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' % id
                    self.src_atomic_irf_family_by_name[id] = family
            self.src_param_init_by_family = t_src.atomic_irf_param_init_by_family()
            self.src_param_trainable_by_family = t_src.atomic_irf_param_trainable_by_family()
            self.src_coef2impulse = t_src.coef2impulse()
            self.src_impulse2coef = t_src.impulse2coef()
            self.src_coef2terminal = t_src.coef2terminal()
            self.src_terminal2coef = t_src.terminal2coef()
            self.src_impulse2terminal = t_src.impulse2terminal()
            self.src_terminal2impulse = t_src.terminal2impulse()
            self.src_interaction2inputs = t_src.interactions2inputs()
            self.src_coef_by_rangf = t_src.coef_by_rangf()
            self.src_interaction_by_rangf = t_src.interaction_by_rangf()
            self.src_irf_by_rangf = t_src.irf_by_rangf()
            self.src_interactions_list = t_src.interactions()

            # Initialize PC tree metadata
            self.n_pc = len(self.src_impulse_names)
            self.has_rate = 'rate' in self.src_impulse_names
            if self.has_rate:
                self.n_pc -= 1
            pointers = {}
            self.form_pc = self.form.pc_transform(self.n_pc, pointers)
            self.t = self.form_pc.t
            self.fw_pointers, self.bw_pointers = IRFNode.pointers2namemmaps(pointers)
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.unary_spline_coef_names = t.unary_spline_coef_names()
            self.interaction_list = t.interactions()
            self.interaction_names = t.interaction_names()
            self.fixed_interaction_names = t.fixed_interaction_names()
            self.impulse_names = t.impulse_names(include_interactions=True)
            self.terminal_names = t.terminal_names()
            self.atomic_irf_names_by_family = t.atomic_irf_by_family()
            self.atomic_irf_family_by_name = {}
            for family in self.atomic_irf_names_by_family:
                for id in self.atomic_irf_names_by_family[family]:
                    assert id not in self.atomic_irf_family_by_name, 'Duplicate IRF ID found for multiple families: %s' %id
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

            # Compute names and indices of source impulses excluding rate term
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

            self.plot_eigenvectors()

        else:
            # Initialize tree metadata
            self.t = self.form.t
            t = self.t
            self.node_table = t.node_table()
            self.coef_names = t.coef_names()
            self.fixed_coef_names = t.fixed_coef_names()
            self.unary_spline_coef_names = t.unary_spline_coef_names()
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

        if self.log_random:
            self.summary_random_writers = {}
            self.summary_random_indexers = {}
            self.summary_random = {}

        # Can't pickle defaultdict because it requires a lambda term for the default value,
        # so instead we pickle a normal dictionary (``rangf_map_base``) and compute the defaultdict
        # from it.
        self.rangf_map = []
        for i in range(len(self.rangf_map_base)):
            self.rangf_map.append(
                defaultdict((lambda x: lambda: x)(self.rangf_n_levels[i] - 1), self.rangf_map_base[i]))

        self.rangf_map_ix_2_levelname = []
        for i in range(len(self.rangf_map_base)):
            ix_2_levelname = [None] * self.rangf_n_levels[i]
            for level in self.rangf_map_base[i]:
                ix_2_levelname[self.rangf_map_base[i][level]] = level
            assert ix_2_levelname[-1] is None, 'Non-null value found in rangf map for unknown level'
            ix_2_levelname[-1] = 'UNK'
            self.rangf_map_ix_2_levelname.append(ix_2_levelname)

        if self.intercept_init is None:
            if self.standardize_response:
                self.intercept_init = 0.
            else:
                self.intercept_init = self.y_train_mean
        if self.y_sd_init is None:
            if self.standardize_response:
                self.y_sd_init = 1.
            else:
                self.y_sd_init = self.y_train_sd

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.intercept_init_tf = tf.constant(self.intercept_init, dtype=self.FLOAT_TF)
                self.epsilon = tf.constant(4 * np.finfo(self.FLOAT_NP).eps, dtype=self.FLOAT_TF)

                self.y_sd_init_tf = tf.constant(float(self.y_sd_init), dtype=self.FLOAT_TF)
                if self.constraint.lower() == 'softplus':
                    self.y_sd_init_unconstrained = tf.contrib.distributions.softplus_inverse(self.y_sd_init_tf)
                    if self.asymmetric_error:
                        self.y_tailweight_init_unconstrained = tf.contrib.distributions.softplus_inverse(1.)
                    self.constraint_fn = tf.nn.softplus
                else:
                    self.y_sd_init_unconstrained = self.y_sd_init_tf
                    if self.asymmetric_error:
                        self.y_tailweight_init_unconstrained = 1.
                    self.constraint_fn = self._safe_abs

                if self.convergence_n_iterates and self.convergence_alpha is not None:
                    self.d0 = []
                    self.d0_names = []
                    self.d0_saved = []
                    self.d0_saved_update = []
                    self.d0_assign = []

                    self.convergence_history = tf.Variable(tf.zeros([int(self.convergence_n_iterates / self.convergence_stride), 1]), trainable=False, dtype=self.FLOAT_NP, name='convergence_history')
                    self.convergence_history_update = tf.placeholder(self.FLOAT_TF, shape=[int(self.convergence_n_iterates / self.convergence_stride), 1], name='convergence_history_update')
                    self.convergence_history_assign = tf.assign(self.convergence_history, self.convergence_history_update)
                    self.proportion_converged = tf.reduce_mean(self.convergence_history)

                    self.last_convergence_check = tf.Variable(0, trainable=False, dtype=self.INT_NP, name='last_convergence_check')
                    self.last_convergence_check_update = tf.placeholder(self.INT_NP, shape=[], name='last_convergence_check_update')
                    self.last_convergence_check_assign = tf.assign(self.last_convergence_check, self.last_convergence_check_update)
                    self.check_convergence = True

        self.parameter_table_columns = ['Estimate']
        self.predict_mode = False

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
        md = {
            'form_str': self.form_str,
            'form': self.form,
            'n_train': self.n_train,
            'y_train_mean': self.y_train_mean,
            'y_train_sd': self.y_train_sd,
            'max_tdelta': self.max_tdelta,
            't_delta_limit': self.t_delta_limit,
            'rangf_map_base': self.rangf_map_base,
            'rangf_n_levels': self.rangf_n_levels,
            'outdir': self.outdir,
        }
        for kwarg in DTSR._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.form_str = md.pop('form_str')
        self.form = md.pop('form', Formula(self.form_str))
        self.n_train = md.pop('n_train')
        self.y_train_mean = md.pop('y_train_mean')
        self.y_train_sd = md.pop('y_train_sd')
        self.max_tdelta = md.pop('max_tdelta')
        self.t_delta_limit = md.pop('t_delta_limit', self.max_tdelta)
        self.rangf_map_base = md.pop('rangf_map_base')
        self.rangf_n_levels = md.pop('rangf_n_levels')
        self.outdir = md.pop('outdir', './dtsr_model/')

        for kwarg in DTSR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))


    ######################################################
    #
    #  Network Initialization
    #
    ######################################################

    def _initialize_inputs(self):
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
                self.time_X = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=self.FLOAT_TF,
                    name='time_X'
                )

                self.time_X_mask = tf.placeholder(
                    shape=[None, self.history_length, n_impulse],
                    dtype=tf.bool,
                    name='time_X_mask'
                )

                self.y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('y'))
                self.time_y = tf.placeholder(shape=[None], dtype=self.FLOAT_TF, name=sn('time_y'))
                # Tensor of temporal offsets with shape (?, history_length, 1)
                self.t_delta = self.time_y[..., None, None] - self.time_X
                # Mask on variables that are not response-aligned, used for implementation of DiracDelta IRF
                self.is_response_aligned = tf.cast(
                    tf.logical_not(
                        tf.cast(self.t_delta[:, -1, :], dtype=tf.bool)
                    ),
                    self.FLOAT_TF
                )
                # self.gf_y = tf.placeholder(shape=[None, len(self.rangf)], dtype=self.INT_TF)
                self.gf_defaults = np.expand_dims(np.array(self.rangf_n_levels, dtype=self.INT_NP), 0) - 1
                self.gf_y = tf.placeholder_with_default(
                    tf.cast(self.gf_defaults, dtype=self.INT_TF),
                    shape=[None, len(self.rangf)],
                    name='gf_y'
                )

                # Tensors used for interpolated IRF composition
                self.max_tdelta_batch = tf.reduce_max(self.t_delta)
                self.interpolation_support = tf.linspace(0., self.max_tdelta_batch, self.n_interp)[..., None]

                # Linspace tensor used for plotting
                self.support_start = tf.placeholder_with_default(
                    tf.cast(0., self.FLOAT_TF),
                    shape=[],
                    name='support_start'
                )
                self.n_time_units = tf.placeholder_with_default(
                    tf.cast(self.t_delta_limit, self.FLOAT_TF),
                    shape=[],
                    name='n_time_units'
                )
                self.n_time_points = tf.placeholder_with_default(
                    tf.cast(self.interp_hz, self.FLOAT_TF),
                    shape=[],
                    name='n_time_points'
                )
                self.support = tf.lin_space(
                    self.support_start,
                    self.n_time_units+self.support_start,
                    tf.cast(self.n_time_points, self.INT_TF) + 1,
                    name='support'
                )[..., None]
                self.support = tf.cast(self.support, dtype=self.FLOAT_TF)
                self.dd_support = tf.concat(
                    [
                        tf.ones((1, 1), dtype=self.FLOAT_TF),
                        tf.zeros((tf.shape(self.support)[0] - 1, 1), dtype=self.FLOAT_TF)
                    ],
                    axis=0
                )

                # Error vector for probability plotting
                self.errors = tf.placeholder(self.FLOAT_TF, shape=[None], name='errors')
                self.n_errors = tf.placeholder(self.INT_TF, shape=[], name='n_errors')

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

                self.training_complete = tf.Variable(
                    False,
                    trainable=False,
                    dtype=tf.bool,
                    name='training_complete'
                )
                self.training_complete_true = tf.assign(self.training_complete, True)
                self.training_complete_false = tf.assign(self.training_complete, False)

                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)
                    rate_ix = names2ix('rate', self.src_impulse_names)
                    self.X_rate = tf.gather(self.X, rate_ix, axis=-1)

                # Initialize regularizers
                if self.regularizer_name is None:
                    self.regularizer = None
                else:
                    self.regularizer = getattr(tf.contrib.layers, self.regularizer_name)(self.regularizer_scale)

                if self.intercept_regularizer_name is None:
                    self.intercept_regularizer = None
                elif self.intercept_regularizer_name == 'inherit':
                    self.intercept_regularizer = self.regularizer
                else:
                    self.intercept_regularizer = getattr(tf.contrib.layers, self.intercept_regularizer_name)(self.intercept_regularizer_scale)
                    
                if self.coefficient_regularizer_name is None:
                    self.coefficient_regularizer = None
                elif self.coefficient_regularizer_name == 'inherit':
                    self.coefficient_regularizer = self.regularizer
                else:
                    self.coefficient_regularizer = getattr(tf.contrib.layers, self.coefficient_regularizer_name)(self.coefficient_regularizer_scale)
                    
                if self.irf_regularizer_name is None:
                    self.irf_regularizer = None
                elif self.irf_regularizer_name == 'inherit':
                    self.irf_regularizer = self.regularizer
                else:
                    self.irf_regularizer = getattr(tf.contrib.layers, self.irf_regularizer_name)(self.irf_regularizer_scale)
                    
                if self.ranef_regularizer_name is None:
                    self.ranef_regularizer = None
                elif self.ranef_regularizer_name == 'inherit':
                    self.ranef_regularizer = self.regularizer
                else:
                    self.ranef_regularizer = getattr(tf.contrib.layers, self.ranef_regularizer_name)(self.ranef_regularizer_scale)

                self.loss_total = tf.placeholder(shape=[], dtype=self.FLOAT_TF, name='loss_total')

                self.training_mse_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mse_in')
                self.training_mse = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mse')
                self.set_training_mse = tf.assign(self.training_mse, self.training_mse_in)
                if self.standardize_response:
                    max_sd = 1
                else:
                    max_sd = self.y_train_sd
                self.training_percent_variance_explained = tf.maximum(0., (1. - self.training_mse / (max_sd ** 2)) * 100.)

                self.training_mae_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_mae_in')
                self.training_mae = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_mae')
                self.set_training_mae = tf.assign(self.training_mae, self.training_mae_in)

                self.training_loglik_in = tf.placeholder(self.FLOAT_TF, shape=[], name='training_loglik_in')
                self.training_loglik = tf.Variable(np.nan, dtype=self.FLOAT_TF, trainable=False, name='training_loglik')
                self.set_training_loglik = tf.assign(self.training_loglik, self.training_loglik_in)

                if self.convergence_basis.lower() == 'loss':
                    self._add_convergence_tracker(self.loss_total, 'loss_total')
                self.converged_in = tf.placeholder(tf.bool, shape=[], name='converged_in')
                self.converged = tf.Variable(False, trainable=False, dtype=tf.bool, name='converged')
                self.set_converged = tf.assign(self.converged, self.converged_in)

    def _initialize_base_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # FIXED EFFECTS

                # Intercept
                if self.has_intercept[None]:
                    if not self.covarying_fixef:
                        self.intercept_fixed_base, self.intercept_fixed_base_summary = self.initialize_intercept()
                else:
                    self.intercept_fixed_base = tf.constant(0., dtype=self.FLOAT_TF, name='intercept')
                self.intercept_random_base = {}
                self.intercept_random_base_summary = {}

                # Coefficients
                coef_ids = self.fixed_coef_names
                if len(coef_ids) > 0 and not self.covarying_fixef:
                    self.coefficient_fixed_base, self.coefficient_fixed_base_summary = self.initialize_coefficient(coef_ids=coef_ids)
                else:
                    self.coefficient_fixed_base = []
                    self.coefficient_fixed_base_summary = []
                self.coefficient_random_base = {}
                self.coefficient_random_base_summary = {}

                # Interactions
                if len(self.interaction_names) > 0:
                    interaction_ids = self.fixed_interaction_names
                    if len(interaction_ids) > 0 and not self.covarying_fixef:
                        self.interaction_fixed_base, self.interaction_fixed_base_summary = self.initialize_interaction(interaction_ids=interaction_ids)
                        self.interaction_random_base = {}
                        self.interaction_random_base_summary = {}

                # RANDOM EFFECTS

                for i in range(len(self.rangf)):
                    gf = self.rangf[i]

                    # Intercept
                    if self.has_intercept[gf]:
                        if not self.covarying_ranef:
                            self.intercept_random_base[gf], self.intercept_random_base_summary[gf] = self.initialize_intercept(ran_gf=gf)

                    # Coefficients
                    coef_ids = self.coef_by_rangf.get(gf, [])
                    if len(coef_ids) > 0 and not self.covarying_ranef:
                        self.coefficient_random_base[gf], self.coefficient_random_base_summary[gf] = self.initialize_coefficient(
                            coef_ids=coef_ids,
                            ran_gf=gf,
                        )

                    # Interactions
                    interaction_ids = self.interaction_by_rangf.get(gf, [])
                    if len(interaction_ids) > 0 and not self.covarying_ranef:
                        self.interaction_random_base[gf], self.interaction_random_base_summary[
                            gf] = self.initialize_interaction(
                            interaction_ids=interaction_ids,
                            ran_gf=gf,
                        )

                # All IRF parameters
                for family in self.atomic_irf_names_by_family:
                    if family == 'DiracDelta':
                        continue

                    elif family == 'Exp':
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ExpRateGT1':
                        self._initialize_base_irf_param('beta', family, lb=1., default=2.)

                    elif family == 'Gamma':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family in ['GammaKgt1', 'GammaShapeGT1']:
                        self._initialize_base_irf_param('alpha', family, lb=1., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ShiftedGamma':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family in ['ShiftedGammaKgt1', 'ShiftedGammaShapeGT1']:
                        self._initialize_base_irf_param('alpha', family, lb=1., default=2.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family == 'Normal':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma2', family, lb=0., default=1.)

                    elif family == 'SkewNormal':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha', family, default=1.)

                    elif family == 'EMG':
                        self._initialize_base_irf_param('mu', family, default=0.)
                        self._initialize_base_irf_param('sigma', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'BetaPrime':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'ShiftedBetaPrime':
                        self._initialize_base_irf_param('alpha', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('delta', family, ub=0., default=-1.)

                    elif family == 'HRFSingleGamma':
                        self._initialize_base_irf_param('alpha', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'HRFDoubleGamma':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha_undershoot_offset', family, lb=0., default=10.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGamma1':
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'HRFDoubleGamma2':
                        self._initialize_base_irf_param('alpha', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)

                    elif family == 'HRFDoubleGamma3':
                        self._initialize_base_irf_param('alpha', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGamma4':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=1., default=16.)
                        self._initialize_base_irf_param('beta', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGamma5':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=1., default=16.)
                        self._initialize_base_irf_param('beta_main', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta_undershoot', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGammaUnconstrained':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('beta_main', family, lb=0., default=1.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=0., default=16.)
                        self._initialize_base_irf_param('beta_undershoot', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif Formula.is_nonparametric(family):
                        bases = Formula.bases(family)
                        spacing_power = Formula.spacing_power(family)
                        # x_init = np.cumsum(np.ones(bases-1))
                        x_init = np.concatenate([[0.], np.cumsum(np.ones(bases-2)) ** spacing_power], axis=0)

                        time_limit = Formula.time_limit(family)
                        if time_limit is None:
                            time_limit = self.t_delta_limit
                        x_init *= time_limit / x_init[-1]
                        # x_init[1:] -= x_init[:-1]

                        for param_name in Formula.irf_params(family):
                            if param_name.startswith('x'):
                                n = int(param_name[1:])
                                # default = x_init[n-2]
                                default = 0.
                                # lb = 0
                                lb = None
                            elif param_name.startswith('y'):
                                n = int(param_name[1:])
                                if n == 1:
                                    default = 1
                                # default = np.sqrt(2 * np.pi)
                                else:
                                    default = 0
                                lb = None
                            else:
                                n = int(param_name[1:])
                                default = n
                                # default = 1
                                lb = 0
                            self._initialize_base_irf_param(param_name, family, default=default, lb=lb)

    def _initialize_intercepts_coefficients_interactions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # FIXED EFFECTS

                # Intercept
                if self.has_intercept[None]:
                    self.intercept_fixed = self.intercept_fixed_base
                    self.intercept_fixed_summary = self.intercept_fixed_base_summary
                    tf.summary.scalar(
                        'intercept',
                        self.intercept_fixed_summary,
                        collections=['params']
                    )
                    self._regularize(self.intercept_fixed, type='intercept', var_name='intercept')
                    if self.convergence_basis.lower() == 'parameters':
                        self._add_convergence_tracker(self.intercept_fixed_summary, 'intercept_fixed')

                else:
                    self.intercept_fixed = self.intercept_fixed_base

                self.intercept = self.intercept_fixed
                self.intercept_summary = self.intercept_fixed_summary
                self.intercept_random = {}
                self.intercept_random_summary = {}
                self.intercept_random_means = {}


                # COEFFICIENTS
                fixef_ix = names2ix(self.fixed_coef_names, self.coef_names)
                coef_ids = self.coef_names
                # if len(self.unary_spline_coef_names) > 0:
                #     nonzero_coefficients = tf.concat(
                #         [
                #             self.coefficient_fixed_base,
                #             tf.ones([len(self.unary_spline_coef_names)], dtype=self.FLOAT_TF)
                #         ],
                #         axis=0
                #     )
                #     nonzero_coefficients_summary = tf.concat(
                #         [
                #             self.coefficient_fixed_base_summary,
                #             tf.ones([len(self.unary_spline_coef_names)], dtype=self.FLOAT_TF)
                #         ],
                #         axis=0
                #     )
                #     nonzero_coef_ix = names2ix(self.fixed_coef_names + self.unary_spline_coef_names, self.coef_names)
                # else:
                nonzero_coefficients = self.coefficient_fixed_base
                nonzero_coefficients_summary = self.coefficient_fixed_base_summary
                nonzero_coef_ix = fixef_ix
                self.coefficient_fixed = self._scatter_along_axis(
                    nonzero_coef_ix,
                    nonzero_coefficients,
                    [len(coef_ids)]
                )
                self.coefficient_fixed_summary = self._scatter_along_axis(
                    nonzero_coef_ix,
                    nonzero_coefficients_summary,
                    [len(coef_ids)]
                )
                self._regularize(self.coefficient_fixed_base, type='coefficient', var_name='coefficient')
                if self.convergence_basis.lower() == 'parameters':
                    self._add_convergence_tracker(self.coefficient_fixed_base_summary, 'coefficient_fixed')

                for i in range(len(self.fixed_coef_names)):
                    tf.summary.scalar(
                        'coefficient' + '/%s' % self.fixed_coef_names[i],
                        self.coefficient_fixed_base_summary[i],
                        collections=['params']
                    )

                self.coefficient = self.coefficient_fixed
                self.coefficient_summary = self.coefficient_fixed_summary
                self.coefficient_random = {}
                self.coefficient_random_summary = {}
                self.coefficient_random_means = {}
                self.coefficient = tf.expand_dims(self.coefficient, 0)
                self.coefficient_summary = tf.expand_dims(self.coefficient_summary, 0)

                # INTERACTIONS
                fixef_ix = names2ix(self.fixed_interaction_names, self.interaction_names)
                if len(self.interaction_names) > 0:
                    interaction_ids = self.interaction_names
                    self.interaction_fixed = self._scatter_along_axis(
                        fixef_ix,
                        self.interaction_fixed_base,
                        [len(interaction_ids)]
                    )
                    self.interaction_fixed_summary = self._scatter_along_axis(
                        fixef_ix,
                        self.interaction_fixed_base_summary,
                        [len(interaction_ids)]
                    )
                    self._regularize(self.interaction_fixed, type='coefficient', var_name='interaction')
                    if self.convergence_basis.lower() == 'parameters':
                        self._add_convergence_tracker(self.interaction_fixed_summary, 'interaction_fixed')

                    for i in range(len(self.interaction_names)):
                        tf.summary.scalar(
                            sn('interaction' + '/%s' % self.interaction_names[i]),
                            self.interaction_fixed_summary[i],
                            collections=['params']
                        )
                    self.interaction = self.interaction_fixed
                    self.interaction_summary = self.interaction_fixed_summary
                    self.interaction_random = {}
                    self.interaction_random_summary = {}
                    self.interaction_random_means = {}
                    self.interaction = tf.expand_dims(self.interaction, 0)

                # RANDOM EFFECTS
                for i in range(len(self.rangf)):
                    gf = self.rangf[i]
                    levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                    # Random intercepts
                    if self.has_intercept[gf]:
                        intercept_random = self.intercept_random_base[gf]
                        intercept_random_summary = self.intercept_random_base_summary[gf]

                        intercept_random_means = tf.reduce_mean(intercept_random, axis=0, keepdims=True)
                        intercept_random_summary_means = tf.reduce_mean(intercept_random_summary, axis=0, keepdims=True)

                        intercept_random -= intercept_random_means
                        intercept_random_summary -= intercept_random_summary_means

                        self._regularize(intercept_random, type='ranef', var_name='intercept_by_%s' % gf)

                        intercept_random = self._scatter_along_axis(
                            levels_ix,
                            intercept_random,
                            [self.rangf_n_levels[i]]
                        )
                        intercept_random_summary = self._scatter_along_axis(
                            levels_ix,
                            intercept_random_summary,
                            [self.rangf_n_levels[i]]
                        )

                        self.intercept_random[gf] = intercept_random
                        self.intercept_random_summary[gf] = intercept_random_summary
                        self.intercept_random_means[gf] = tf.reduce_mean(intercept_random_summary, axis=0)

                        # Create record for convergence tracking
                        if self.convergence_basis.lower() == 'parameters':
                            self._add_convergence_tracker(self.intercept_random_summary[gf], 'intercept_by_%s' %gf)

                        self.intercept += tf.gather(intercept_random, self.gf_y[:, i])
                        self.intercept_summary += tf.gather(intercept_random_summary, self.gf_y[:, i])

                        if self.log_random:
                            tf.summary.histogram(
                                sn('by_%s/intercept' % gf),
                                intercept_random_summary,
                                collections=['random']
                            )

                    # Random coefficients
                    coefs = self.coef_by_rangf.get(gf, [])
                    if len(coefs) > 0:
                        nonzero_coef_ix = names2ix(coefs, self.coef_names)

                        coefficient_random = self.coefficient_random_base[gf]
                        coefficient_random_summary = self.coefficient_random_base_summary[gf]

                        coefficient_random_means = tf.reduce_mean(coefficient_random, axis=0, keepdims=True)
                        coefficient_random_summary_means = tf.reduce_mean(coefficient_random_summary, axis=0, keepdims=True)

                        coefficient_random -= coefficient_random_means
                        coefficient_random_summary -= coefficient_random_summary_means
                        self._regularize(coefficient_random, type='ranef', var_name='coefficient_by_%s' % gf)

                        coefficient_random = self._scatter_along_axis(
                            nonzero_coef_ix,
                            self._scatter_along_axis(
                                levels_ix,
                                coefficient_random,
                                [self.rangf_n_levels[i], len(coefs)]
                            ),
                            [self.rangf_n_levels[i], len(self.coef_names)],
                            axis=1
                        )
                        coefficient_random_summary = self._scatter_along_axis(
                            nonzero_coef_ix,
                            self._scatter_along_axis(
                                levels_ix,
                                coefficient_random_summary,
                                [self.rangf_n_levels[i], len(coefs)]
                            ),
                            [self.rangf_n_levels[i], len(self.coef_names)],
                            axis=1
                        )

                        self.coefficient_random[gf] = coefficient_random
                        self.coefficient_random_summary[gf] = coefficient_random_summary
                        self.coefficient_random_means[gf] = tf.reduce_mean(coefficient_random_summary, axis=0)

                        if self.convergence_basis.lower() == 'parameters':
                            self._add_convergence_tracker(self.coefficient_random_summary[gf], 'coefficient_by_%s' %gf)

                        self.coefficient += tf.gather(coefficient_random, self.gf_y[:, i], axis=0)
                        self.coefficient_summary += tf.gather(coefficient_random_summary, self.gf_y[:, i], axis=0)

                        if self.log_random:
                            for j in range(len(coefs)):
                                coef_name = coefs[j]
                                ix = nonzero_coef_ix[j]
                                tf.summary.histogram(
                                    sn('by_%s/coefficient/%s' % (gf, coef_name)),
                                    coefficient_random_summary[:, ix],
                                    collections=['random']
                                )
                                
                    # Random interactions
                    if len(self.interaction_names) > 0:
                        interactions = self.interaction_by_rangf.get(gf, [])
                        if len(interactions) > 0:
                            interaction_ix = names2ix(interactions, self.interaction_names)

                            interaction_random = self.interaction_random_base[gf]
                            interaction_random_summary = self.interaction_random_base_summary[gf]

                            interaction_random_means = tf.reduce_mean(interaction_random, axis=0, keepdims=True)
                            interaction_random_summary_means = tf.reduce_mean(interaction_random_summary, axis=0, keepdims=True)

                            interaction_random -= interaction_random_means
                            interaction_random_summary -= interaction_random_summary_means
                            self._regularize(interaction_random, type='ranef', var_name='interaction_by_%s' % gf)

                            interaction_random = self._scatter_along_axis(
                                interaction_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    interaction_random,
                                    [self.rangf_n_levels[i], len(interactions)]
                                ),
                                [self.rangf_n_levels[i], len(self.interaction_names)],
                                axis=1
                            )
                            interaction_random_summary = self._scatter_along_axis(
                                interaction_ix,
                                self._scatter_along_axis(
                                    levels_ix,
                                    interaction_random_summary,
                                    [self.rangf_n_levels[i], len(interactions)]
                                ),
                                [self.rangf_n_levels[i], len(self.interaction_names)],
                                axis=1
                            )

                            self.interaction_random[gf] = interaction_random
                            self.interaction_random_summary[gf] = interaction_random_summary
                            self.interaction_random_means[gf] = tf.reduce_mean(interaction_random_summary, axis=0)

                            if self.convergence_basis.lower() == 'parameters':
                                self._add_convergence_tracker(self.interaction_random_summary[gf], 'interaction_by_%s' % gf)

                            self.interaction += tf.gather(interaction_random, self.gf_y[:, i], axis=0)
                            self.interaction_summary += tf.gather(interaction_random_summary, self.gf_y[:, i], axis=0)

                            if self.log_random:
                                for j in range(len(interactions)):
                                    interaction_name = interactions[j]
                                    ix = interaction_ix[j]
                                    tf.summary.histogram(
                                        sn('by_%s/interaction/%s' % (gf, interaction_name)),
                                        interaction_random_summary[:, ix],
                                        collections=['random']
                                    )

    def _initialize_irf_lambdas(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                integral_ub = self.t_delta_limit.astype(dtype=self.FLOAT_NP)

                def exponential(params):
                    return lambda x: exponential_irf(
                        params,
                        session=self.sess
                    )(x)

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['ExpRateGT1'] = exponential

                def gamma(params):
                    return lambda x: gamma_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['SteepGamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['GammaKgt1'] = gamma
                self.irf_lambdas['HRFSingleGamma'] = gamma

                def shifted_gamma(params):
                    return lambda x: shifted_gamma_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma
                self.irf_lambdas['ShiftedGammaShapeGT1'] = shifted_gamma
                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma

                def normal(params):
                    return lambda x: normal_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )(x)

                self.irf_lambdas['Normal'] = normal

                def skew_normal(params):
                    return lambda x: skew_normal_irf(
                        params,
                        session=self.sess,
                        epsilon=self.epsilon
                    )(x)

                self.irf_lambdas['SkewNormal'] = normalize_irf(skew_normal, self.support, session=self.sess, epsilon=self.epsilon)

                def emg(params):
                    return lambda x: emg_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )(x)

                self.irf_lambdas['EMG'] = emg

                def beta_prime(params):
                    return lambda x: beta_prime_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )(x)

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(params):
                    return lambda x: shifted_beta_prime_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )(x)

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

                def double_gamma_1(params):
                    return lambda x: double_gamma_1_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['HRFDoubleGamma1'] = double_gamma_1

                def double_gamma_2(params):
                    return lambda x: double_gamma_2_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['HRFDoubleGamma2'] = double_gamma_2

                def double_gamma_3(params):
                    return lambda x: double_gamma_3_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['HRFDoubleGamma3'] = double_gamma_3

                def double_gamma_4(params):
                    return lambda x: double_gamma_4_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['HRFDoubleGamma4'] = double_gamma_4

                def double_gamma_5(params):
                    return lambda x: double_gamma_5_irf(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )(x)

                self.irf_lambdas['HRFDoubleGamma5'] = double_gamma_5

    def _initialize_nonparametric_irf(self, order, bases, method='summed_gaussians', instantaneous=True, roughness_penalty=0.):
        def f(
                params,
                order=order,
                bases=bases,
                method=method,
                instantaneous=instantaneous,
                roughness_penalty=roughness_penalty,
                support=self.support,
                epsilon=self.epsilon,
                int_type=self.INT_TF,
                float_type=self.FLOAT_TF,
                session=self.sess
        ):
            return nonparametric_smooth(
                method,
                params,
                bases,
                integral_ub=self.t_delta_limit.astype(dtype=self.FLOAT_NP),
                order=order,
                instantaneous=instantaneous,
                roughness_penalty=roughness_penalty,
                support=support,
                epsilon=epsilon,
                int_type=int_type,
                float_type=float_type,
                session=session
            )

        return f

    def _get_irf_lambda(self, family):
        if family in self.irf_lambdas:
            return self.irf_lambdas[family]
        elif Formula.is_nonparametric(family):
            order = Formula.order(family)
            bases = Formula.bases(family)
            instantaneous = Formula.instantaneous(family)
            roughness_penalty = Formula.roughness_penalty(family)
            return self._initialize_nonparametric_irf(
                order,
                bases,
                instantaneous=instantaneous,
                roughness_penalty=roughness_penalty
            )
        else:
            raise ValueError('No IRF lamdba found for family "%s"' % family)

    def _initialize_irf_params(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for family in self.atomic_irf_names_by_family:
                    if family == 'DiracDelta':
                        continue

                    irf_ids = self.atomic_irf_names_by_family[family]

                    params = []
                    params_summary = []
                    params_fixed = []
                    params_fixed_summary = []
                    params_random = {}
                    params_random_summary = {}

                    for param_name in Formula.irf_params(family):
                        param_vals = self._initialize_irf_param(param_name, family)
                        params.append(param_vals[0])

                        params_summary.append(param_vals[1])
                        if param_vals[2] is not None:
                            params_fixed.append(param_vals[2])
                        if param_vals[3] is not None:
                            params_fixed_summary.append(param_vals[3])
                            if self.convergence_basis.lower() == 'parameters':
                                self._add_convergence_tracker(param_vals[3], 'irf_%s_%s' % (family, param_name))

                        if param_vals[4] is not None and param_vals[5] is not None:
                            assert(set(param_vals[4].keys()) == set(param_vals[5].keys()))

                            for gf in param_vals[4].keys():
                                if gf not in params_random:
                                    params_random[gf] = []
                                params_random[gf].append(param_vals[4][gf])

                                if gf not in params_random_summary:
                                    params_random_summary[gf] = []
                                params_random_summary[gf].append(param_vals[5][gf])
                                if self.convergence_basis.lower() == 'parameters':
                                    self._add_convergence_tracker(param_vals[5][gf], 'irf_%s_%s_by_%s' % (family, param_name, gf))

                    has_random_irf = False
                    for param in params:
                        if not param.shape.is_fully_defined():
                            has_random_irf = True
                            break
                    if has_random_irf:
                        for i in range(len(params)):
                            param = params[i]
                            if param.shape.is_fully_defined():
                                assert param.shape[0] == 1, 'Parameter with shape %s not broadcastable to batch length' %param.shape
                                params[i] =  tf.tile(param, [tf.shape(self.time_y)[0], 1])

                    params = tf.stack(params, axis=1)
                    params_summary = tf.stack(params_summary, axis=1)
                    params_fixed = tf.stack(params_fixed, axis=1)
                    params_fixed_summary = tf.stack(params_fixed_summary, axis=1)
                    for gf in params_random:
                        params_random[gf] = tf.stack(params_random[gf], axis=1)

                    for i in range(len(irf_ids)):
                        id = irf_ids[i]
                        ix = names2ix(id, self.atomic_irf_names_by_family[family])
                        assert id not in self.irf_params, 'Duplicate IRF node name already in self.irf_params'
                        self.irf_params[id] = tf.gather(params, ix, axis=2)
                        self.irf_params_summary[id] = tf.gather(params_summary, ix, axis=2)
                        trainable_param_ix = names2ix(self.atomic_irf_param_trainable_by_family[family][id], Formula.irf_params(family))
                        if len(trainable_param_ix) > 0:
                            self.irf_params_fixed[id] = tf.gather(tf.gather(params_fixed, ix, axis=2), trainable_param_ix, axis=1)
                            self.irf_params_fixed_summary[id] = tf.gather(tf.gather(params_fixed_summary, ix, axis=2), trainable_param_ix, axis=1)
                            for gf in params_random:
                                if gf not in self.irf_params_random:
                                    self.irf_params_random[gf] = {}
                                self.irf_params_random[gf][id] = tf.gather(tf.gather(params_random[gf], ix, axis=2), trainable_param_ix, axis=1)
                                if gf not in self.irf_params_random_summary:
                                    self.irf_params_random_summary[gf] = {}
                                self.irf_params_random_summary[gf][id] = tf.gather(tf.gather(params_random_summary[gf], ix, axis=2), trainable_param_ix, axis=1)

    def _initialize_irf_param(self, param_name, family):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                irf_ids = self.atomic_irf_names_by_family[family]
                param_mean = self.irf_params_means[family][param_name]
                param_lb = self.irf_params_lb[family][param_name]
                param_ub = self.irf_params_ub[family][param_name]
                param_mean_unconstrained = self.irf_params_means_unconstrained[family][param_name]
                trainable = self.atomic_irf_param_trainable_by_family[family]

                trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                    param_name,
                    irf_ids,
                    trainable=trainable
                )

                dim = len(trainable_ix)
                trainable_means = tf.expand_dims(tf.gather(param_mean_unconstrained, trainable_ix), 0)

                # Initialize trainable IRF parameters as trainable variables
                if dim > 0:
                    param_fixed = self.irf_params_fixed_base[family][param_name]
                    param_fixed_summary = self.irf_params_fixed_base_summary[family][param_name]

                    if param_lb is not None and param_ub is None:
                        param_fixed = param_lb + self.epsilon + self.constraint_fn(param_fixed)
                        param_fixed_summary = param_lb + self.epsilon + self.constraint_fn(param_fixed_summary)
                    elif param_lb is None and param_ub is not None:
                        param_fixed = param_ub - self.epsilon - self.constraint_fn(param_fixed)
                        param_fixed_summary = param_ub - self.epsilon - self.constraint_fn(param_fixed_summary)
                    elif param_lb is not None and param_ub is not None:
                        param_fixed = self._softplus_sigmoid(param_fixed, a=param_lb, b=param_ub)
                        param_fixed_summary = self._softplus_sigmoid(param_fixed_summary, a=param_lb, b=param_ub)

                    self._regularize(param_fixed, trainable_means, type='irf', var_name=param_name)
                else:
                    param_fixed = param_fixed_summary = None

                for i in range(dim):
                    tf.summary.scalar(
                        sn('%s/%s' % (param_name, irf_ids[i])),
                        param_fixed_summary[0, i],
                        collections=['params']
                    )

                # Initialize untrainable IRF parameters as constants
                param_untrainable = tf.expand_dims(
                    tf.gather(
                        tf.ones((len(irf_ids),), dtype=self.FLOAT_TF) * param_mean,
                        untrainable_ix)
                    ,
                    axis=0
                )

                param = param_fixed
                param_summary = param_fixed_summary

                # Process any random IRF parameters
                irf_by_rangf = {}
                for id in irf_ids:
                    for gf in self.irf_by_rangf:
                        if id in self.irf_by_rangf[gf]:
                            if gf not in irf_by_rangf:
                                irf_by_rangf[gf] = []
                            irf_by_rangf[gf].append(id)

                param_random_by_rangf = {}
                param_random_summary_by_rangf = {}

                if len(irf_by_rangf) > 0:
                   for i, gf in enumerate(self.rangf):
                        if gf in irf_by_rangf:
                            irf_ids_ran = [x for x in irf_by_rangf[gf] if param_name in trainable[x]]
                            if len(irf_ids_ran):
                                irfs_ix = names2ix(irf_by_rangf[gf], irf_ids)
                                levels_ix = np.arange(self.rangf_n_levels[i] - 1)

                                param_random = self._center_and_constrain(
                                    self.irf_params_random_base[gf][family][param_name],
                                    tf.gather(param_fixed, irfs_ix, axis=1),
                                    lb=param_lb,
                                    ub=param_ub
                                )
                                param_random_summary = self._center_and_constrain(
                                    self.irf_params_random_base_summary[gf][family][param_name],
                                    tf.gather(param_summary, irfs_ix, axis=1),
                                    lb=param_lb,
                                    ub=param_ub
                                )

                                self._regularize(param_random, type='ranef', var_name='%s_by_%s' % (param_name, gf))

                                param_random = self._scatter_along_axis(
                                    irfs_ix,
                                    self._scatter_along_axis(
                                        levels_ix,
                                        param_random,
                                        [self.rangf_n_levels[i], len(irfs_ix)]
                                    ),
                                    [self.rangf_n_levels[i], len(irfs_ix)],
                                    axis=1
                                )
                                param_random_summary = self._scatter_along_axis(
                                    irfs_ix,
                                    self._scatter_along_axis(
                                        levels_ix,
                                        param_random_summary,
                                        [self.rangf_n_levels[i], len(irfs_ix)]
                                    ),
                                    [self.rangf_n_levels[i], len(irfs_ix)],
                                    axis=1
                                )

                                param_random_by_rangf[gf] = param_random
                                param_random_summary_by_rangf[gf] = param_random_summary
                                if gf not in self.irf_params_random_means:
                                    self.irf_params_random_means[gf] = {}
                                if family not in self.irf_params_random_means[gf]:
                                    self.irf_params_random_means[gf][family] = {}
                                self.irf_params_random_means[gf][family][param_name] = tf.reduce_mean(param_random_summary, axis=0)

                                param += tf.gather(param_random, self.gf_y[:, i], axis=0)
                                param_summary += tf.gather(param_random_summary, self.gf_y[:, i], axis=0)

                                if self.log_random:
                                    for j in range(len(irf_by_rangf[gf])):
                                        irf_name = irf_by_rangf[gf][j]
                                        ix = irfs_ix[j]
                                        tf.summary.histogram(
                                            'by_%s/%s/%s' % (gf, param_name, irf_name),
                                            param_random_summary[:, ix],
                                            collections=['random']
                                        )
                            else:
                                param_random = param_random_summary = None
                        else:
                            param_random = param_random_summary = None

                # Combine trainable and untrainable parameters
                if len(untrainable_ix) > 0:
                    if len(trainable_ix) > 0:
                        param = tf.concat([param, param_untrainable], axis=1)
                        param_summary = tf.concat([param_summary, param_untrainable], axis=1)
                    else:
                        param = param_untrainable
                        param_summary = param_untrainable

                    param = tf.gather(param, np.concatenate([trainable_ix, untrainable_ix]), axis=1)
                param_summary = tf.gather(param_summary, np.concatenate([trainable_ix, untrainable_ix]), axis=1)

                return param, param_summary, param_fixed, param_fixed_summary, param_random_by_rangf, param_random_summary_by_rangf

    def _initialize_base_irf_param(self, param_name, family, lb=None, ub=None, default=0.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                irf_ids = self.atomic_irf_names_by_family[family]
                param_init = self.atomic_irf_param_init_by_family[family]
                param_trainable = self.atomic_irf_param_trainable_by_family[family]

                # Process and store initial/prior means
                param_mean = self._get_mean_init_vector(irf_ids, param_name, param_init, default=default)
                param_mean_unconstrained, param_lb, param_ub = self._process_mean(param_mean, lb=lb, ub=ub)

                if family not in self.irf_params_means:
                    self.irf_params_means[family] = {}
                self.irf_params_means[family][param_name] = param_mean

                if family not in self.irf_params_means_unconstrained:
                    self.irf_params_means_unconstrained[family] = {}
                self.irf_params_means_unconstrained[family][param_name] = param_mean_unconstrained

                if family not in self.irf_params_lb:
                    self.irf_params_lb[family] = {}
                self.irf_params_lb[family][param_name] = param_lb

                if family not in self.irf_params_ub:
                    self.irf_params_ub[family] = {}
                self.irf_params_ub[family][param_name] = param_ub


                # Select out irf IDs for which this param is trainable
                trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                    param_name,
                    irf_ids,
                    trainable=param_trainable
                )
                trainable_means = tf.expand_dims(tf.gather(param_mean_unconstrained, trainable_ix), axis=0)


                if not self.covarying_fixef:
                    # Initialize and store fixed params on the unconstrained space
                    if len(trainable_ix) > 0:
                        param_fixed_base, param_fixed_base_summary = self.initialize_irf_param_unconstrained(
                            param_name,
                            [x for x in irf_ids if param_name in param_trainable[x]],
                            mean=trainable_means
                        )

                        if family not in self.irf_params_fixed_base:
                            self.irf_params_fixed_base[family] = {}
                        self.irf_params_fixed_base[family][param_name] = param_fixed_base

                        if family not in self.irf_params_fixed_base_summary:
                            self.irf_params_fixed_base_summary[family] = {}
                        self.irf_params_fixed_base_summary[family][param_name] = param_fixed_base_summary


                if not self.covarying_ranef:
                    # Initialize and store random params on the unconstrained space
                    irf_by_rangf = {}
                    for id in irf_ids:
                        for gf in self.irf_by_rangf:
                            if id in self.irf_by_rangf[gf]:
                                if gf not in irf_by_rangf:
                                    irf_by_rangf[gf] = []
                                irf_by_rangf[gf].append(id)

                    for gf in irf_by_rangf:
                        irf_ids_ran = [x for x in irf_by_rangf[gf] if param_name in param_trainable[x]]
                        if len(irf_ids_ran) > 0:
                            param_random_base, param_random_base_summary = self.initialize_irf_param_unconstrained(
                                param_name,
                                [x for x in irf_by_rangf[gf] if param_name in param_trainable[x]],
                                mean=0.,
                                ran_gf=gf
                            )

                            if gf not in self.irf_params_random_base:
                                self.irf_params_random_base[gf] = {}
                            if family not in self.irf_params_random_base[gf]:
                                self.irf_params_random_base[gf][family] = {}
                            self.irf_params_random_base[gf][family][param_name] = param_random_base

                            if gf not in self.irf_params_random_base_summary:
                                self.irf_params_random_base_summary[gf] = {}
                            if family not in self.irf_params_random_base_summary[gf]:
                                self.irf_params_random_base_summary[gf][family] = {}
                            self.irf_params_random_base_summary[gf][family][param_name] = param_random_base_summary

    def _initialize_joint_distributions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # FIXED EFFECTS

                if self.covarying_fixef:
                    joint_fixed_means = []
                    joint_fixed_sds = []
                    joint_fixed_ix = {}

                    i = 0

                    # Intercept
                    if self.has_intercept[None]:
                        joint_fixed_means.append(tf.expand_dims(self.intercept_init_tf, axis=0))
                        joint_fixed_sds.append(tf.expand_dims(self.intercept_joint_sd, axis=0))
                        joint_fixed_ix['intercept'] = (i, i + 1)
                        i += 1

                    # Coefficients
                    coef_ids = self.fixed_coef_names
                    if len(coef_ids) > 0:
                        joint_fixed_means.append(tf.zeros((len(coef_ids), ), dtype=self.FLOAT_TF))
                        joint_fixed_sds.append(tf.ones((len(coef_ids), ), dtype=self.FLOAT_TF) * self.coef_joint_sd)
                        joint_fixed_ix['coefficient'] = (i, i + len(coef_ids))
                        i += len(coef_ids)

                    # Interactions
                    interaction_ids = self.fixed_interaction_names
                    if len(interaction_ids) > 0:
                        joint_fixed_means.append(tf.zeros((len(interaction_ids),), dtype=self.FLOAT_TF))
                        joint_fixed_sds.append(tf.ones((len(interaction_ids),), dtype=self.FLOAT_TF) * self.coef_joint_sd)
                        joint_fixed_ix['interaction'] = (i, i + len(interaction_ids))
                        i += len(interaction_ids)

                    # IRF Parameters
                    for family in sorted(list(self.atomic_irf_names_by_family.keys())):
                        if family not in joint_fixed_ix:
                            joint_fixed_ix[family] = {}
                        for param_name in Formula.irf_params(family):
                            irf_ids = self.atomic_irf_names_by_family[family]
                            param_trainable = self.atomic_irf_param_trainable_by_family[family]

                            trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                                param_name,
                                irf_ids,
                                trainable=param_trainable
                            )

                            if len(trainable_ix) > 0:
                                joint_fixed_means.append(
                                    tf.gather(
                                        self.irf_params_means_unconstrained[family][param_name],
                                        trainable_ix
                                    )
                                )
                                joint_fixed_sds.append(
                                    tf.ones((len(trainable_ix),), dtype=self.FLOAT_TF) * self.irf_param_joint_sd
                                )
                                joint_fixed_ix[family][param_name] = (i,  i + len(trainable_ix))
                                i += len(trainable_ix)
                            else:
                                joint_fixed_ix[family][param_name] = None

                    joint_fixed_means = tf.concat(joint_fixed_means, axis=0)
                    joint_fixed_sds = tf.concat(joint_fixed_sds, axis=0)

                    self.joint_fixed, self.joint_fixed_summary = self.initialize_joint_distribution(
                        joint_fixed_means,
                        joint_fixed_sds,
                    )

                    self.joint_fixed_ix = joint_fixed_ix

                # RANDOM EFFECTS

                if self.covarying_ranef:
                    joint_random = {}
                    joint_random_summary = {}
                    joint_random_means = {}
                    joint_random_sds = {}
                    joint_random_ix = {}

                    for i, gf in enumerate(self.rangf):
                        joint_random_means[gf] = []
                        joint_random_sds[gf] = []
                        joint_random_ix[gf] = {}
                        n_levels = self.rangf_n_levels[i] - 1

                        i = 0

                        # Intercepts
                        if self.has_intercept[gf]:
                            joint_random_means[gf].append(tf.zeros([n_levels,], dtype=self.FLOAT_TF))
                            joint_random_sds[gf].append(tf.ones([n_levels,], dtype=self.FLOAT_TF) * self.intercept_joint_sd)
                            joint_random_ix[gf]['intercept'] = (i, i + n_levels)
                            i += n_levels

                        # Coefficients
                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids) > 0:
                            joint_random_means[gf].append(tf.zeros([n_levels * len(coef_ids)], dtype=self.FLOAT_TF))
                            joint_random_sds[gf].append(tf.ones([n_levels * len(coef_ids)], dtype=self.FLOAT_TF) * self.coef_joint_sd)
                            joint_random_ix[gf]['coefficient'] = (i, i + n_levels * len(coef_ids))
                            i += n_levels * len(coef_ids)

                        # Interactions
                        interaction_ids = self.interaction_by_rangf.get(gf, [])
                        if len(interaction_ids) > 0:
                            joint_random_means[gf].append(tf.zeros([n_levels * len(interaction_ids)], dtype=self.FLOAT_TF))
                            joint_random_sds[gf].append(
                                tf.ones([n_levels * len(interaction_ids)], dtype=self.FLOAT_TF) * self.coef_joint_sd)
                            joint_random_ix[gf]['interaction'] = (i, i + n_levels * len(interaction_ids))
                            i += n_levels * len(interaction_ids)

                        for family in sorted(list(self.atomic_irf_names_by_family.keys())):
                            for param_name in Formula.irf_params(family):
                                irf_ids_src = self.atomic_irf_names_by_family[family]

                                irf_ids = []
                                for id in irf_ids_src:
                                    if gf in self.irf_by_rangf and id in self.irf_by_rangf[gf]:
                                        irf_ids.append(id)

                                if len(irf_ids) > 0:
                                    if family not in joint_random_ix[gf]:
                                        joint_random_ix[gf][family] = {}
                                    param_trainable = self.atomic_irf_param_trainable_by_family[family]

                                    trainable_ix, untrainable_ix = self._compute_trainable_untrainable_ix(
                                        param_name,
                                        irf_ids,
                                        trainable=param_trainable
                                    )

                                    if len(trainable_ix) > 0:
                                        joint_random_means[gf].append(
                                            tf.zeros([n_levels * len(trainable_ix)], dtype=self.FLOAT_TF)
                                        )
                                        joint_random_sds[gf].append(
                                            tf.ones([n_levels * len(trainable_ix)], dtype=self.FLOAT_TF) * self.irf_param_joint_sd
                                        )
                                        joint_random_ix[gf][family][param_name] = (i, i + n_levels * len(trainable_ix))
                                        i += n_levels * len(trainable_ix)
                                    else:
                                        joint_random_ix[gf][family][param_name] = None

                        joint_random_means[gf] = tf.concat(joint_random_means[gf], axis=0)
                        joint_random_sds[gf] = tf.concat(joint_random_sds[gf], axis=0)

                        joint_random[gf], joint_random_summary[gf] = self.initialize_joint_distribution(
                            joint_random_means[gf],
                            joint_random_sds[gf],
                            ran_gf=gf
                        )

                    self.joint_random = joint_random
                    self.joint_random_summary = joint_random_summary
                    self.joint_random_ix = joint_random_ix

    def _initialize_joint_distribution_slices(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.covarying_fixef:
                    if self.has_intercept[None]:
                        s, e = self.joint_fixed_ix['intercept']
                        self.intercept_fixed_base = self.joint_fixed[s]
                        self.intercept_fixed_base_summary = self.joint_fixed_summary[s]

                    s, e = self.joint_fixed_ix['coefficient']
                    self.coefficient_fixed_base = self.joint_fixed[s:e]
                    self.coefficient_fixed_base_summary = self.joint_fixed_summary[s:e]

                    s, e = self.joint_fixed_ix['interaction']
                    self.interaction_fixed_base = self.joint_fixed[s:e]
                    self.interaction_fixed_base_summary = self.joint_fixed_summary[s:e]

                    for family in sorted(list(self.atomic_irf_names_by_family.keys())):
                        if family not in self.irf_params_fixed_base:
                            self.irf_params_fixed_base[family] = {}
                        if family not in self.irf_params_fixed_base_summary:
                            self.irf_params_fixed_base_summary[family] = {}
                        for param_name in Formula.irf_params(family):
                            bounds = self.joint_fixed_ix[family][param_name]
                            if bounds is not None:
                                s, e = bounds
                                self.irf_params_fixed_base[family][param_name] = tf.expand_dims(self.joint_fixed[s:e], axis=0)
                                self.irf_params_fixed_base_summary[family][param_name] = tf.expand_dims(self.joint_fixed_summary[s:e], axis=0)

                if self.covarying_ranef:
                    for i, gf in enumerate(self.rangf):
                        rangf_n_levels = self.rangf_n_levels[i] - 1

                        if self.has_intercept[gf]:
                            if gf not in self.intercept_random_base:
                                self.intercept_random_base[gf] = {}
                            if gf not in self.intercept_random_base_summary:
                                self.intercept_random_base_summary[gf] = {}
                            s, e = self.joint_random_ix[gf]['intercept']
                            self.intercept_random_base[gf] = self.joint_random[gf][s:e]
                            self.intercept_random_base_summary[gf] = self.joint_random_summary[gf][s:e]

                        coef_ids = self.coef_by_rangf.get(gf, [])
                        if len(coef_ids) > 0:
                            if gf not in self.coefficient_random_base:
                                self.coefficient_random_base[gf] = {}
                            if gf not in self.coefficient_random_base_summary:
                                self.coefficient_random_base_summary[gf] = {}
                            s, e = self.joint_random_ix[gf]['coefficient']
                            self.coefficient_random_base[gf] = tf.reshape(
                                self.joint_random[gf][s:e],
                                [rangf_n_levels, len(coef_ids)]
                            )
                            self.coefficient_random_base_summary[gf] = tf.reshape(
                                self.joint_random_summary[gf][s:e],
                                [rangf_n_levels, len(coef_ids)]
                            )

                        interaction_ids = self.interaction_by_rangf.get(gf, [])
                        if len(interaction_ids) > 0:
                            if gf not in self.interaction_random_base:
                                self.interaction_random_base[gf] = {}
                            if gf not in self.interaction_random_base_summary:
                                self.interaction_random_base_summary[gf] = {}
                            s, e = self.joint_random_ix[gf]['interaction']
                            self.interaction_random_base[gf] = tf.reshape(
                                self.joint_random[gf][s:e],
                                [rangf_n_levels, len(interaction_ids)]
                            )
                            self.interaction_random_base_summary[gf] = tf.reshape(
                                self.joint_random_summary[gf][s:e],
                                [rangf_n_levels, len(interaction_ids)]
                            )

                        if gf not in self.irf_params_random_base:
                            self.irf_params_random_base[gf] = {}
                        if gf not in self.irf_params_random_base_summary:
                            self.irf_params_random_base_summary[gf] = {}

                        for family in sorted(list(self.atomic_irf_names_by_family.keys())):
                            irf_ids_src = self.atomic_irf_names_by_family[family]

                            irf_ids = []
                            for id in irf_ids_src:
                                if gf in self.irf_by_rangf and id in self.irf_by_rangf[gf]:
                                    irf_ids.append(id)

                            if len(irf_ids) > 0:
                                if family not in self.irf_params_random_base[gf]:
                                    self.irf_params_random_base[gf][family] = {}
                                if family not in self.irf_params_random_base_summary[gf]:
                                    self.irf_params_random_base_summary[gf][family] = {}
                                for param_name in Formula.irf_params(family):
                                    if gf in self.irf_by_rangf:
                                        if param_name not in self.irf_params_random_base[gf][family]:
                                            self.irf_params_random_base[gf][family][param_name] = {}
                                        if param_name not in self.irf_params_random_base_summary[gf][family]:
                                            self.irf_params_random_base_summary[gf][family][param_name] = {}

                                        bounds = self.joint_random_ix[gf][family][param_name]
                                        if bounds is not None:
                                            s, e = bounds
                                            self.irf_params_random_base[gf][family][param_name] = tf.reshape(
                                                self.joint_random[gf][s:e],
                                                [rangf_n_levels, len(irf_ids)]
                                            )
                                            self.irf_params_random_base_summary[gf][family][param_name] = tf.reshape(
                                                self.joint_random_summary[gf][s:e],
                                                [rangf_n_levels, len(irf_ids)]
                                            )

    def _initialize_parameter_tables(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                parameter_table_fixed_keys = []
                parameter_table_fixed_values = []
                if self.has_intercept[None]:
                    parameter_table_fixed_keys.append('intercept')
                    parameter_table_fixed_values.append(
                        tf.expand_dims(self.intercept_fixed, axis=0)
                    )
                for coef_name in self.fixed_coef_names:
                    coef_name_str = 'coefficient_' + coef_name
                    parameter_table_fixed_keys.append(coef_name_str)
                parameter_table_fixed_values.append(
                    tf.gather(self.coefficient_fixed, names2ix(self.fixed_coef_names, self.coef_names))
                )
                if len(self.fixed_interaction_names) > 0:
                    for interaction_name in self.fixed_interaction_names:
                        interaction_name_str = 'interaction_' + interaction_name
                        parameter_table_fixed_keys.append(interaction_name_str)
                    parameter_table_fixed_values.append(
                        tf.gather(self.interaction_fixed, names2ix(self.fixed_interaction_names, self.interaction_names))
                    )
                for irf_id in self.irf_params_fixed:
                    family = self.atomic_irf_family_by_name[irf_id]
                    for param in self.atomic_irf_param_trainable_by_family[family][irf_id]:
                        param_ix = names2ix(param, Formula.irf_params(family))
                        parameter_table_fixed_keys.append(param + '_' + irf_id)
                        parameter_table_fixed_values.append(
                            tf.squeeze(
                                tf.gather(self.irf_params_fixed[irf_id], param_ix, axis=1),
                                axis=(0, 2)
                            )
                        )

                self.parameter_table_fixed_keys = parameter_table_fixed_keys
                self.parameter_table_fixed_values = tf.concat(parameter_table_fixed_values, 0)

                parameter_table_random_keys = []
                parameter_table_random_rangf = []
                parameter_table_random_rangf_levels = []
                parameter_table_random_values = []

                if len(self.rangf) > 0:
                    for i in range(len(self.rangf)):
                        gf = self.rangf[i]
                        levels = sorted(self.rangf_map_ix_2_levelname[i][:-1])
                        levels_ix = names2ix([self.rangf_map[i][level] for level in levels], range(self.rangf_n_levels[i]))
                        if self.has_intercept[gf]:
                            for level in levels:
                                parameter_table_random_keys.append('intercept')
                                parameter_table_random_rangf.append(gf)
                                parameter_table_random_rangf_levels.append(level)
                            parameter_table_random_values.append(
                                tf.gather(self.intercept_random[gf], levels_ix)
                            )
                        if gf in self.coefficient_random:
                            coef_names = self.coef_by_rangf.get(gf, [])
                            for coef_name in coef_names:
                                coef_ix = names2ix(coef_name, self.coef_names)
                                coef_name_str = 'coefficient_' + coef_name
                                for level in levels:
                                    parameter_table_random_keys.append(coef_name_str)
                                    parameter_table_random_rangf.append(gf)
                                    parameter_table_random_rangf_levels.append(level)
                                parameter_table_random_values.append(
                                    tf.squeeze(
                                        tf.gather(
                                            tf.gather(self.coefficient_random[gf], coef_ix, axis=1),
                                            levels_ix
                                        )
                                    )
                                )
                        if len(self.interaction_names) > 0:
                            if gf in self.interaction_random:
                                interaction_names = self.interaction_by_rangf.get(gf, [])
                                for interaction_name in interaction_names:
                                    interaction_ix = names2ix(interaction_name, self.interaction_names)
                                    interaction_name_str = 'interaction_' + interaction_name
                                    for level in levels:
                                        parameter_table_random_keys.append(interaction_name_str)
                                        parameter_table_random_rangf.append(gf)
                                        parameter_table_random_rangf_levels.append(level)
                                    parameter_table_random_values.append(
                                        tf.squeeze(
                                            tf.gather(
                                                tf.gather(self.interaction_random[gf], interaction_ix, axis=1),
                                                levels_ix
                                            )
                                        )
                                    )
                        if gf in self.irf_params_random:
                            for irf_id in self.irf_params_random[gf]:
                                family = self.atomic_irf_family_by_name[irf_id]
                                for param in self.atomic_irf_param_trainable_by_family[family][irf_id]:
                                    param_ix = names2ix(param, Formula.irf_params(family))
                                    for level in levels:
                                        parameter_table_random_keys.append(param + '_' + irf_id)
                                        parameter_table_random_rangf.append(gf)
                                        parameter_table_random_rangf_levels.append(level)
                                    parameter_table_random_values.append(
                                        tf.squeeze(
                                            tf.gather(
                                                tf.gather(self.irf_params_random[gf][irf_id], param_ix, axis=1),
                                                levels_ix,
                                            )
                                        )
                                    )

                    self.parameter_table_random_keys = parameter_table_random_keys
                    self.parameter_table_random_rangf = parameter_table_random_rangf
                    self.parameter_table_random_rangf_levels = parameter_table_random_rangf_levels
                    self.parameter_table_random_values = tf.concat(parameter_table_random_values, 0)

    def _initialize_random_mean_vector(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.rangf) > 0:
                    means = []
                    for gf in sorted(list(self.intercept_random_means.keys())):
                        means.append(tf.expand_dims(self.intercept_random_means[gf], 0))
                    for gf in sorted(list(self.coefficient_random_means.keys())):
                        means.append(self.coefficient_random_means[gf])
                    if len(self.interaction_names) > 0:
                        for gf in sorted(list(self.interaction_random_means.keys())):
                            means.append(self.interaction_random_means[gf])
                    for gf in sorted(list(self.irf_params_random_means.keys())):
                        for family in sorted(list(self.irf_params_random_means[gf].keys())):
                            for param_name in sorted(list(self.irf_params_random_means[gf][family].keys())):
                                means.append(self.irf_params_random_means[gf][family][param_name])

                    self.random_means = tf.concat(means, axis=0)

    def _initialize_irfs(self, t):
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
                            'scaled': self.irf_mc[t.p.name()]['atomic']['unscaled'] * tf.expand_dims(tf.gather(self.coefficient, coef_ix, axis=1), axis=-1),
                            'unscaled': self.irf_mc[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_mc[t.p.name()]['composite']['unscaled'] * tf.expand_dims(tf.gather(self.coefficient, coef_ix, axis=1), axis=-1),
                            'unscaled': self.irf_mc[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.irf_plot[t.p.name()]['atomic']['unscaled'] * tf.expand_dims(tf.gather(self.coefficient_summary, coef_ix, axis=1), axis=-1),
                            'unscaled': self.irf_plot[t.p.name()]['atomic']['unscaled']
                        },
                        'composite': {
                            'scaled': self.irf_plot[t.p.name()]['composite']['unscaled'] * tf.expand_dims(tf.gather(self.coefficient_summary, coef_ix, axis=1), axis=-1),
                            'unscaled': self.irf_plot[t.p.name()]['composite']['unscaled']
                        }
                    }

                    assert not t.name() in self.irf_integral_tensors, 'Duplicate IRF node name already in self.mc_integrals'
                    if t.p.family == 'DiracDelta':
                        integral_tensor = tf.gather(self.coefficient_summary, coef_ix, axis=1)[:,0]
                    else:
                        integral_tensor = tf.reduce_sum(self.irf_mc[t.name()]['composite']['scaled'], axis=(1,2)) * self.n_time_units / tf.cast(self.n_time_points, dtype=self.FLOAT_TF)
                    self.irf_integral_tensors[t.name()] = integral_tensor

                elif t.family == 'DiracDelta':
                    assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in DTSR formula strings'
                    assert not t.impulse == 'rate', '"rate" is a reserved keyword in DTSR formula strings and cannot be used under DiracDelta'


                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = self.irf[t.p.name()][:]

                    assert not t.name() in self.irf_mc, 'Duplicate IRF node name already in self.irf_mc'
                    self.irf_mc[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support[None, ...],
                            'unscaled': self.dd_support[None, ...]
                        },
                        'composite' : {
                            'scaled': self.dd_support[None, ...],
                            'unscaled': self.dd_support[None, ...]
                        }
                    }

                    assert not t.name() in self.irf_plot, 'Duplicate IRF node name already in self.irf_plot'
                    self.irf_plot[t.name()] = {
                        'atomic': {
                            'scaled': self.dd_support[None, ...],
                            'unscaled': self.dd_support[None, ...]
                        },
                        'composite' : {
                            'scaled': self.dd_support[None, ...],
                            'unscaled': self.dd_support[None, ...]
                        }
                    }
                else:
                    params = self.irf_params[t.irf_id()]
                    params_summary = self.irf_params_summary[t.irf_id()]

                    atomic_irf = self._new_irf(self._get_irf_lambda(t.family), params)
                    atomic_irf_plot = self._new_irf(self._get_irf_lambda(t.family), params_summary)

                    if t.p.name() in self.irf:
                        irf = self.irf[t.p.name()][:] + [atomic_irf]
                        irf_plot = self.irf[t.p.name()][:] + [atomic_irf_plot]
                    else:
                        irf = [atomic_irf]
                        irf_plot = [atomic_irf_plot]

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = irf

                    atomic_irf_mc = atomic_irf(self.support)
                    atomic_irf_plot = atomic_irf_plot(self.support)

                    if len(irf) > 1:
                        composite_irf_mc = self._compose_irf(irf)(self.support[None, ...])
                    else:
                        composite_irf_mc = atomic_irf_mc
                    if len(irf_plot) > 1:
                        composite_irf_plot = self._compose_irf(irf_plot)(self.support[None, ...])
                    else:
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
                    self._initialize_irfs(c)

    def _initialize_backtransformed_irf_plot(self, t):
        if self.pc:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if t.name() in self.irf_plot:
                        src_irf_names = self.bw_pointers[t.name()]
                        t_impulse_names = t.impulse_names()
                        if t_impulse_names == ['rate'] and len(src_irf_names) == 1 and self.src_node_table[src_irf_names[0]].impulse_names() == ['rate']:
                            self.src_irf_plot[src_irf_names[0]] = self.irf_plot[t.name()]
                            self.src_irf_mc[src_irf_names[0]] = self.irf_mc[t.name()]
                            if t.name() in self.irf_integral_tensors:
                                self.src_irf_integral_tensors[src_irf_names[0]] = self.irf_integral_tensors[t.name()]
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
                                        if t.name() in self.irf_integral_tensors:
                                            if src_irf_name in self.src_irf_integral_tensors:
                                                self.src_irf_integral_tensors[src_irf_name] += tf.reduce_sum(self.irf_integral_tensors[t.name()] * e, axis=0, keep_dims=True)
                                            else:
                                                self.src_irf_integral_tensors[src_irf_name] = tf.reduce_sum(self.irf_integral_tensors[t.name()] * e, axis=0, keep_dims=True)

                    for c in t.children:
                        if c.name() in self.irf_plot:
                            self._initialize_backtransformed_irf_plot(c)

    def _initialize_impulses(self):
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
                                impulse = self._apply_pc(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)[:, -1, :]

                        # Zero-out impulses to DiracDelta that are not response-aligned
                        impulse *= self.is_response_aligned[:, impulse_ix[0]:impulse_ix[0]+1]
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
                                impulse = self._apply_pc(X, src_ix=src_impulse_ix, pc_ix=impulse_ix)
                        else:
                            impulse = tf.gather(self.X, impulse_ix, axis=2)

                    self.irf_impulses[name] = impulse

    def _initialize_convolutions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for name in self.terminal_names:
                    t = self.node_table[name]
                    impulse_name = self.terminal2impulse[name]
                    impulse_ix = names2ix(impulse_name, self.impulse_names)[0]

                    if t.p.family == 'DiracDelta':
                        self.convolutions[name] = self.irf_impulses[name]
                    else:
                        if t.cont:
                            # Create a continuous piecewise linear function
                            # that interpolates between points in the impulse history.
                            # Reverse because the history contains time offsets in descen   ding order
                            knot_location = tf.reverse(
                                tf.transpose(self.t_delta[:,:,impulse_ix:impulse_ix+1], [0, 2, 1]),
                                axis=[-1]
                            )
                            knot_amplitude = tf.reverse(
                                tf.transpose(self.irf_impulses[name], [0, 2, 1]),
                                axis=[-1]
                            )
                            impulse_resampler = piecewise_linear_interpolant(knot_location, knot_amplitude, session=self.sess)
                            t_delta = tf.linspace(self.interp_step * (self.history_length-1), 0, self.history_length)[None, ..., None]
                            t_delta = tf.tile(t_delta, [tf.shape(self.t_delta)[0], 1, 1])
                            impulse = impulse_resampler(t_delta)
                            impulse *= self.interp_step
                        else:
                            impulse = self.irf_impulses[name]
                            t_delta = self.t_delta[:,:,impulse_ix:impulse_ix+1]

                        irf = self.irf[name]
                        if len(irf) > 1:
                            irf = self._compose_irf(irf)
                        else:
                            irf = irf[0]

                        irf_seq = irf(t_delta)

                        self.convolutions[name] = tf.reduce_sum(impulse * irf_seq, axis=1)

    def _initialize_interactions(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.interaction_names) > 0:
                    interaction_ix = np.arange(len(self.interaction_names))
                    interaction_coefs = tf.gather(self.interaction, interaction_ix, axis=1)
                    interaction_inputs = []
                    for i, interaction in enumerate(self.interaction_list):
                        assert interaction.name() == self.interaction_names[i], 'Mismatched sort order between self.interaction_names and self.interaction_list. This should not have happened, so please report it in issue tracker on Github.'
                        irf_input_names = [x.name() for x in interaction.irf_responses()]

                        inputs_cur = None

                        if len(irf_input_names) > 0:
                            irf_input_ix = names2ix(irf_input_names, self.terminal_names)
                            irf_inputs = tf.gather(self.X_conv, irf_input_ix, axis=1)
                            inputs_cur = tf.reduce_prod(irf_inputs, axis=1)

                        non_irf_input_names = [x.name() for x in interaction.non_irf_responses()]
                        if len(non_irf_input_names):
                            non_irf_input_ix = names2ix(non_irf_input_names, self.impulse_names)
                            non_irf_inputs = tf.gather(self.X[:,-1,:], non_irf_input_ix, axis=1)
                            non_irf_inputs = tf.reduce_prod(non_irf_inputs, axis=1)
                            if inputs_cur is not None:
                                inputs_cur *= non_irf_inputs
                            else:
                                inputs_cur = non_irf_inputs

                        interaction_inputs.append(inputs_cur)
                    interaction_inputs = tf.stack(interaction_inputs, axis=1)
                    self.summed_interactions = tf.reduce_sum(interaction_coefs * interaction_inputs, axis=1)

    def _initialize_interaction_plots(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for i, interaction in enumerate(self.interaction_names):
                    interaction_ix = [i]
                    estimate = self.dd_support * tf.gather(self.interaction_fixed, interaction_ix)
                    self.irf_mc[interaction] = {
                        'atomic': {
                            'scaled': estimate,
                            'unscaled': estimate
                        },
                        'composite': {
                            'scaled': estimate,
                            'unscaled': estimate
                        }
                    }

                    estimate = self.dd_support * tf.gather(self.interaction_fixed_summary, interaction_ix)
                    self.irf_plot[interaction] = {
                        'atomic': {
                            'scaled': estimate,
                            'unscaled': estimate
                        },
                        'composite': {
                            'scaled': estimate,
                            'unscaled': estimate
                        }
                    }

    def _construct_network(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_irfs(self.t)
                self._initialize_impulses()
                self._initialize_convolutions()
                self._initialize_backtransformed_irf_plot(self.t)

                convolutions = [self.convolutions[x] for x in self.terminal_names]
                if len(convolutions) > 0:
                    self.X_conv = tf.concat(convolutions, axis=1)
                else:
                    self.X_conv = tf.zeros((1, 1), dtype=self.FLOAT_TF)

                coef_names = [self.node_table[x].coef_id() for x in self.terminal_names]
                coef_ix = names2ix(coef_names, self.coef_names)
                coef = tf.gather(self.coefficient, coef_ix, axis=1)
                self.X_conv_scaled = self.X_conv*coef

                out = self.intercept + tf.reduce_sum(self.X_conv_scaled, axis=1)

                if len(self.interaction_names) > 0:
                    self._initialize_interactions()
                    #self._initialize_interaction_plots()
                    out += self.summed_interactions

                self.out = out
                # Hack needed for MAP evaluation of DTSRBayes
                self.out_mean = self.out

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__(*args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer

    def _initialize_optimizer(self, name):
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

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.global_step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.global_step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            t,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)
    #
    # def _initialize_optimizer(self, name):
    #     with self.sess.as_default():
    #         with self.sess.graph.as_default():
    #             lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
    #             if name is None:
    #                 self.lr = lr
    #                 return None
    #             if self.lr_decay_family is not None:
    #                 lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
    #                 lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
    #                 lr_decay_staircase = self.lr_decay_staircase
    #                 if self.lr_decay_iteration_power != 1:
    #                     t = tf.cast(self.global_step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
    #                 else:
    #                     t = self.global_step
    #
    #                 if 'cosine' in self.lr_decay_family:
    #                     self.lr = getattr(tf.train, self.lr_decay_family)(
    #                         lr,
    #                         t,
    #                         lr_decay_steps,
    #                         name='learning_rate'
    #                     )
    #                 else:
    #                     self.lr = getattr(tf.train, self.lr_decay_family)(
    #                         lr,
    #                         t,
    #                         lr_decay_steps,
    #                         lr_decay_rate,
    #                         staircase=lr_decay_staircase,
    #                         name='learning_rate'
    #                     )
    #                 if np.isfinite(self.learning_rate_min):
    #                     lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
    #                     INF_TF = tf.constant(inf, dtype=self.FLOAT_TF)
    #                     self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
    #             else:
    #                 self.lr = lr
    #
    #             return {
    #                 'SGD': lambda x: tf.train.GradientDescentOptimizer(x),
    #                 'Momentum': lambda x: tf.train.MomentumOptimizer(x, 0.9),
    #                 'AdaGrad': lambda x: tf.train.AdagradOptimizer(x),
    #                 'AdaDelta': lambda x: tf.train.AdadeltaOptimizer(x),
    #                 'Adam': lambda x: tf.train.AdamOptimizer(x, epsilon=self.optim_epsilon),
    #                 'FTRL': lambda x: tf.train.FtrlOptimizer(x),
    #                 'RMSProp': lambda x: tf.train.RMSPropOptimizer(x),
    #                 'Nadam': lambda x: tf.contrib.opt.NadamOptimizer(x, epsilon=self.optim_epsilon)
    #             }[name](self.lr)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('loss_by_iter', self.loss_total, collections=['loss'])
                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dtsr', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/dtsr')
                self.summary_params = tf.summary.merge_all(key='params')
                self.summary_losses = tf.summary.merge_all(key='loss')
                if self.log_random and len(self.rangf) > 0:
                    self.summary_random = tf.summary.merge_all(key='random')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                    self.saver = tf.train.Saver()

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.ema_vars = tf.get_collection('trainable_variables')
                self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                self.ema_op = self.ema.apply(self.ema_vars)
                self.ema_map = {}
                for v in self.ema_vars:
                    self.ema_map[self.ema.average_name(v)] = v
                self.ema_saver = tf.train.Saver(self.ema_map)

    def _initialize_convergence_checking(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.rho_t = tf.placeholder(self.FLOAT_TF, name='rho_t_in')
                self.p_rho_t = tf.placeholder(self.FLOAT_TF, name='p_rho_t_in')
                if self.convergence_basis.lower() == 'parameters':
                    self.rho_a = tf.placeholder(self.FLOAT_TF, name='rho_a_in')
                    self.p_rho_a = tf.placeholder(self.FLOAT_TF, name='p_rho_a_in')

                tf.summary.scalar('convergence/rho_t', self.rho_t, collections=['convergence'])
                tf.summary.scalar('convergence/p_rho_t', self.p_rho_t, collections=['convergence'])
                if self.convergence_basis.lower() == 'parameters':
                    tf.summary.scalar('convergence/rho_a', self.rho_a, collections=['convergence'])
                    tf.summary.scalar('convergence/p_rho_a', self.p_rho_a, collections=['convergence'])
                tf.summary.scalar('convergence/proportion_converged', self.proportion_converged, collections=['convergence'])
                self.summary_convergence = tf.summary.merge_all(key='convergence')







    ######################################################
    #
    #  Internal public network initialization methods.
    #  These must be implemented by all subclasses and
    #  should only be called at initialization.
    #
    ######################################################

    def initialize_intercept(self, ran_gf=None):
        """
        Add an intercept.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param ran_gf: ``str`` or ``None``; Name of random grouping factor for random intercept (if ``None``, constructs a fixed intercept)
        :return: 2-tuple of ``Tensor`` ``(intercept, intercept_summary)``; ``intercept`` is the intercept for use by the model. ``intercept_summary`` is an identically-shaped representation of the current intercept value for logging and plotting (can be identical to ``intercept``). For fixed intercepts, should return a trainable scalar. For random intercepts, should return batch-length vector of trainable weights. Weights should be initialized around 0.
        """
        raise NotImplementedError

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        """
        Add coefficients.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of ``len(coef_ids)`` trainable weights. For random coefficients, should return batch-length matrix of trainable weights with ``len(coef_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_interaction(self, interaction_ids=None, ran_gf=None):
        """
        Add (response-level) interactions.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of interaction IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random interaction (if ``None``, constructs a fixed interaction)
        :return: 2-tuple of ``Tensor`` ``(interaction, interaction_summary)``; ``interaction`` is the interaction for use by the model. ``interaction_summary`` is an identically-shaped representation of the current interaction values for logging and plotting (can be identical to ``interaction``). For fixed interactions, should return a vector of ``len(interaction_ids)`` trainable weights. For random interactions, should return batch-length matrix of trainable weights with ``len(interaction_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0, ran_gf=None):
        """
        Add IRF parameters in the unconstrained space.
        DTSR will apply appropriate constraint transformations as needed.
        This method must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param param_name: ``str``; Name of parameter (e.g. ``"alpha"``)
        :param ids: ``list`` of ``str``; Names of IRF nodes to which this parameter applies
        :param mean: ``float`` or ``Tensor``; scalar (broadcasted) or 1D tensor (shape = ``(len(ids),)``) of parameter means on the transformed space.
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(param, param_summary)``; ``param`` is the parameter for use by the model. ``param_summary`` is an identically-shaped representation of the current param values for logging and plotting (can be identical to ``param``). For fixed params, should return a vector of ``len(ids)`` trainable weights. For random params, should return batch-length matrix of trainable weights with ``len(ids)``. Weights should be initialized around **mean** (if fixed) or ``0`` (if random).
        """

        raise NotImplementedError

    def initialize_joint_distribution(self, means, sds, ran_gf=None):
        """
        Add a multivariate joint distribution over the parameters represented by **means**, where **means** are on the unconstrained space for bounded params.
        The variance-covariance matrix is initialized using the square of **sds** as the diagonal.
        This method is required for multivariate mode and must be implemented by subclasses of ``DTSR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param means: ``Tensor``; 1-D tensor as MVN mean parameter.
        :param sds: ``Tensor``; 1-D tensor used to construct diagonal of MVN variance-covariance parameter.
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(joint, join_summary)``; ``joint`` is the random variable for use by the model. ``joint_summary`` is an identically-shaped representation of the current joint for logging and plotting (can be identical to ``joint``). Returns a multivariate normal distribution of dimension len(means) in all cases.
        """

        raise NotImplementedError

    def initialize_objective(self):
        """
        Add an objective function to the DTSR model.

        :return: ``None``
        """

        raise NotImplementedError





    ######################################################
    #
    #  Model construction subroutines
    #
    ######################################################

    def _new_irf(self, irf_lambda, params, parent_irf=None):
        irf = irf_lambda(params)
        if parent_irf is None:
            def new_irf(x):
                return irf(x)
        else:
            def new_irf(x):
                return irf(parent_irf(x))
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

    def _apply_pc(self, inputs, src_ix=None, pc_ix=None, inv=False):
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
                outputs = self._matmul(X, e)
                if expansions > 0:
                    outputs = tf.squeeze(outputs, axis=list(range(expansions)))
                return outputs

    def _get_mean_init_vector(self, irf_ids, param_name, irf_param_init, default=0.):
        mean = np.zeros(len(irf_ids))
        for i in range(len(irf_ids)):
            mean[i] = irf_param_init[irf_ids[i]].get(param_name, default)
        return mean

    def _process_mean(self, mean, lb=None, ub=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                mean = tf.constant(mean, dtype=self.FLOAT_TF)
                if lb is not None:
                    lb = tf.constant(lb, dtype=self.FLOAT_TF)
                if ub is not None:
                    ub = tf.constant(ub, dtype=self.FLOAT_TF)

                if lb is not None and ub is None:
                    # Lower-bounded support only
                    if self.constraint.lower() == 'softplus':
                        mean = tf.contrib.distributions.softplus_inverse(mean - lb - self.epsilon)
                    elif self.constraint.lower() == 'abs':
                        mean = mean - lb - self.epsilon
                    else:
                        raise ValueError('Unrecognized constraint function "%s"' % self.constraint)
                elif lb is None and ub is not None:
                    # Upper-bounded support only
                    if self.constraint.lower() == 'softplus':
                        mean = tf.contrib.distributions.softplus_inverse(-(mean - ub + self.epsilon))
                    elif self.constraint.lower() == 'abs':
                        mean = -(mean - ub + self.epsilon)
                    else:
                        raise ValueError('Unrecognized constraint function "%s"' % self.constraint)
                elif lb is not None and ub is not None:
                    # Finite-interval bounded support
                    mean = self._softplus_sigmoid_inverse(mean, lb, ub)

        return mean, lb, ub

    def _compute_trainable_untrainable_ix(self, param_name, ids, trainable=None):
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

    def _add_convergence_tracker(self, var, name, alpha=0.9):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.convergence_n_iterates:
                    # Flatten the variable for easy argmax
                    var = tf.reshape(var, [-1])
                    self.d0.append(var)

                    self.d0_names.append(name)

                    # Initialize tracker of parameter iterates
                    var_d0_iterates = tf.Variable(
                        tf.zeros([int(self.convergence_n_iterates / self.convergence_stride)] + list(var.shape), dtype=self.FLOAT_TF),
                        name=name + '_d0',
                        trainable=False
                    )

                    var_d0_iterates_update = tf.placeholder(self.FLOAT_TF, shape=var_d0_iterates.shape)
                    self.d0_saved.append(var_d0_iterates)
                    self.d0_saved_update.append(var_d0_iterates_update)
                    self.d0_assign.append(tf.assign(var_d0_iterates, var_d0_iterates_update))

    def _compute_and_test_corr(self, iterates):
        x = np.arange(0, len(iterates)*self.convergence_stride, self.convergence_stride).astype('float')[..., None]
        y = iterates

        n_iterates = int(self.convergence_n_iterates / self.convergence_stride)

        rt = corr(x, y)[0]
        tt = rt * np.sqrt((n_iterates - 2) / (1 - rt ** 2))
        p_tt = 1 - (scipy.stats.t.cdf(np.fabs(tt), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(tt), n_iterates - 2))
        p_tt = np.where(np.isfinite(p_tt), p_tt, np.zeros_like(p_tt))

        ra = corr(y[1:], y[:-1])[0]
        ta = ra * np.sqrt((n_iterates - 2) / (1 - ra ** 2))
        p_ta = 1 - (scipy.stats.t.cdf(np.fabs(ta), n_iterates - 2) - scipy.stats.t.cdf(-np.fabs(ta), n_iterates - 2))
        p_ta = np.where(np.isfinite(p_ta), p_ta, np.zeros_like(p_ta))

        return rt, p_tt, ra, p_ta

    def run_convergence_check(self, verbose=True, feed_dict=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.check_convergence:
                    min_p = 1.
                    min_p_ix = 0
                    rt_at_min_p = 0
                    ra_at_min_p = 0
                    p_ta_at_min_p = 0
                    fd_assign = {}

                    cur_step = self.global_step.eval(session=self.sess)
                    last_check = self.last_convergence_check.eval(session=self.sess)
                    offset = cur_step % self.convergence_stride
                    update = last_check < cur_step and self.convergence_stride > 0
                    if update and self.convergence_basis == 'loss' and feed_dict is None:
                        update = False
                        sys.stderr.write('Skipping convergence history update because no feed_dict provided.\n')

                    push = update and offset == 0
                    # End of stride if next step is a push
                    end_of_stride = last_check < (cur_step+1) and self.convergence_stride > 0 and ((cur_step+1) % self.convergence_stride == 0)

                    if self.check_convergence:
                        if update:
                            var_d0, var_d0_iterates = self.sess.run([self.d0, self.d0_saved], feed_dict=feed_dict)
                        else:
                            var_d0_iterates = self.sess.run(self.d0_saved)

                        start_ix = int(self.convergence_n_iterates / self.convergence_stride) - int(cur_step / self.convergence_stride)
                        start_ix = max(0, start_ix)

                        for i in range(len(var_d0_iterates)):
                            if update:
                                new_d0 = var_d0[i]
                                iterates_d0 = var_d0_iterates[i]
                                if push:
                                    iterates_d0[:-1] = iterates_d0[1:]
                                    iterates_d0[-1] = new_d0
                                else:
                                    new_d0 = (new_d0 + offset * iterates_d0[-1]) / (offset + 1)
                                    iterates_d0[-1] = new_d0
                                fd_assign[self.d0_saved_update[i]] = iterates_d0

                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(iterates_d0[start_ix:])
                            else:
                                rt, p_tt, ra, p_ta = self._compute_and_test_corr(var_d0_iterates[i][start_ix:])

                            new_min_p_ix = p_tt.argmin()
                            new_min_p = p_tt[new_min_p_ix]
                            if new_min_p < min_p:
                                min_p = new_min_p
                                min_p_ix = i
                                rt_at_min_p = rt[new_min_p_ix]
                                ra_at_min_p = ra[new_min_p_ix]
                                p_ta_at_min_p = p_ta[new_min_p_ix]

                        if update:
                            fd_assign[self.last_convergence_check_update] = self.global_step.eval(session=self.sess)
                            to_run = [self.d0_assign, self.last_convergence_check_assign]
                            self.sess.run(to_run, feed_dict=fd_assign)

                    if end_of_stride:
                        locally_converged = cur_step > self.convergence_n_iterates and \
                                    (min_p > self.convergence_alpha)
                        if self.convergence_basis.lower() == 'parameters':
                            locally_converged &= p_ta_at_min_p > self.convergence_alpha
                        convergence_history = self.convergence_history.eval(session=self.sess)
                        convergence_history[:-1] = convergence_history[1:]
                        convergence_history[-1] = locally_converged
                        self.sess.run(self.convergence_history_assign, {self.convergence_history_update: convergence_history})

                    if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                        fd_convergence = {
                                self.rho_t: rt_at_min_p,
                                self.p_rho_t: min_p
                            }
                        if self.convergence_basis.lower() == 'parameters':
                            fd_convergence[self.rho_a] = ra_at_min_p
                            fd_convergence[self.p_rho_a] =  p_ta_at_min_p
                        summary_convergence = self.sess.run(
                            self.summary_convergence,
                            feed_dict=fd_convergence
                        )
                        self.writer.add_summary(summary_convergence, self.global_step.eval(session=self.sess))

                    proportion_converged = self.proportion_converged.eval(session=self.sess)
                    converged = cur_step > self.convergence_n_iterates and \
                                (min_p > self.convergence_alpha) and \
                                (p_ta_at_min_p > self.convergence_alpha) and \
                                (proportion_converged > self.convergence_alpha)

                    if verbose:
                        sys.stderr.write('rho_t: %s.\n' % rt_at_min_p)
                        sys.stderr.write('p of rho_t: %s.\n' % min_p)
                        if self.convergence_basis.lower() == 'parameters':
                            sys.stderr.write('rho_a: %s.\n' % ra_at_min_p)
                            sys.stderr.write('p of rho_a: %s.\n' % p_ta_at_min_p)
                        sys.stderr.write('Location: %s.\n\n' % self.d0_names[min_p_ix])
                        sys.stderr.write('Iterate meets convergence criteria: %s.\n\n' % converged)
                        sys.stderr.write('Proportion of recent iterates converged: %s.\n' % proportion_converged)

                else:
                    min_p_ix = min_p = rt_at_min_p = ra_at_min_p = p_ta_at_min_p = None
                    proportion_converged = 0
                    converged = False
                    if verbose:
                        sys.stderr.write('Convergence checking off.\n')

                self.sess.run(self.set_converged, feed_dict={self.converged_in: converged})

                return min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged

    def _collect_plots(self):
        switches = [['atomic', 'composite'], ['scaled', 'unscaled'], ['dirac', 'nodirac']]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.plots = {}
                irf_names = [x for x in self.node_table if x in self.irf_plot and not (len(self.node_table[x].children) == 1 and self.node_table[x].children[0].terminal())]
                irf_names_nodirac = [x for x in irf_names if not x.startswith('DiracDelta')]
                irf_names_terminal = [x for x in self.node_table if x in self.irf_plot and self.node_table[x].terminal()]
                irf_names_terminal_nodirac = [x for x in irf_names_terminal if not x.startswith('DiracDelta')]


                for a in switches[0]:
                    if a not in self.plots:
                        self.plots[a] = {}
                    for b in switches[1]:
                        self.plots[a][b] = {}
                        for c in switches[2]:
                            plot_y = []

                            if b == 'unscaled':
                                if c == 'dirac':
                                    names = irf_names
                                else:
                                    names = irf_names_nodirac
                            else:
                                if c == 'dirac':
                                    names = irf_names_terminal
                                else:
                                    names = irf_names_terminal_nodirac

                            for x in names:
                                plot_y.append(self.irf_plot[x][a][b])

                            self.plots[a][b][c] = {
                                'names': names,
                                'plot': plot_y
                            }

                if self.pc:
                    self.src_plot_tensors = {}
                    irf_names = [x for x in self.src_node_table if x in self.src_irf_plot and not (len(self.src_node_table[x].children) == 1 and self.src_node_table[x].children[0].terminal())]
                    irf_names_terminal = [x for x in self.src_node_table if x in self.src_irf_plot and self.src_node_table[x].terminal()]

                    for a in switches[0]:
                        if a not in self.src_plot_tensors:
                            self.src_plot_tensors[a] = {}
                        for b in switches[1]:
                            self.plots[a][b] = {}
                            for c in switches[2]:
                                plot_y = []

                                if b == 'unscaled':
                                    if c == 'dirac':
                                        names = irf_names
                                    else:
                                        names = irf_names_nodirac
                                else:
                                    if c == 'dirac':
                                        names = irf_names_terminal
                                    else:
                                        names = irf_names_terminal_nodirac

                                for x in names:
                                    plot_y.append(self.src_irf_plot[x][a][b])

                                self.src_plot_tensors[a][b][c] = {
                                    'names': names,
                                    'plot': plot_y
                                }

    def _regularize(self, var, center=None, type=None, var_name=None):
        assert type in [None, 'intercept', 'coefficient', 'irf', 'ranef']
        if type is None:
            regularizer = self.regularizer
        else:
            regularizer = getattr(self, '%s_regularizer' %type)

        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    if center is None:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    else:
                        reg = tf.contrib.layers.apply_regularization(regularizer, [var - center])
                    self.regularizer_losses.append(reg)
                    self.regularizer_losses_varnames.append(str(var_name))
                    if type is None:
                        reg_name = self.regularizer_name
                        reg_scale = self.regularizer_scale
                    else:
                        reg_name = getattr(self, '%s_regularizer_name' %type)
                        reg_scale = getattr(self, '%s_regularizer_scale' %type)
                    if reg_name == 'inherit':
                        reg_name = self.regularizer_name
                    if reg_scale == 'inherit':
                        reg_scale = self.regularizer_scale
                    self.regularizer_losses_names.append(reg_name)
                    self.regularizer_losses_scales.append(reg_scale)

    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if fixed:
                    out = self.parameter_table_fixed_values.eval(session=self.sess)
                else:
                    out = self.parameter_table_random_values.eval(session=self.sess)

            return out

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    sys.stderr.write('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            sys.stderr.write(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            sys.stderr.write(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err




    ######################################################
    #
    #  Math subroutines
    #
    ######################################################

    def _matmul(self, A, B):
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

    def _tril_diag_ix(self, n):
        return (np.arange(1, n+1).cumsum() - 1).astype(self.INT_NP)

    def _scatter_along_axis(self, axis_indices, updates, shape, axis=0):
        # Except for axis, updates and shape must be identically shaped
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if axis != 0:
                    transpose_axes = [axis] + list(range(axis)) + list(range(axis + 1, len(updates.shape)))
                    inverse_transpose_axes = list(range(1, axis + 1)) + [0] + list(range(axis + 1, len(updates.shape)))
                    updates_transposed = tf.transpose(updates, transpose_axes)
                    scatter_shape = [shape[axis]] + shape[:axis] + shape[axis + 1:]
                else:
                    updates_transposed = updates
                    scatter_shape = shape

                out = tf.scatter_nd(
                    tf.expand_dims(axis_indices, -1),
                    updates_transposed,
                    scatter_shape
                )

                if axis != 0:
                    out = tf.transpose(out, inverse_transpose_axes)

                return out

    def _reduce_interpolated_sum(self, X, time, axis=0):
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

    def _softplus_sigmoid(self, x, a=-1., b=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                f = tf.nn.softplus
                c = b - a

                g = (-f(-f(x - a) + c) + f(c)) * c / f(c) + a
                return g

    def _softplus_sigmoid_inverse(self, x, a=-1., b=1.):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                f = tf.nn.softplus
                ln = tf.log
                exp = tf.exp
                c = b - a

                g = ln(exp(c) / ( (exp(c) + 1) * exp( -f(c) * (x - a) / c ) - 1) - 1) + a
                return g

    def _center_and_constrain(self, param_random, param_fixed, lb=None, ub=None, use_softplus_sigmoid=True):
        with self.sess.as_default():
            # If the parameter is constrained, passes random effects matrix first through a non-linearity, then
            # mean-centers its columns. However, centering after constraint can cause the constraint to be violated,
            # so the recentered weights must be renormalized proportionally to their pre-centering mean in order
            # to guarantee that the constraint is obeyed. How this renormalization is done depends on whether
            # the parameter is upper-bounded, lower-bounded, or both.

            with self.sess.graph.as_default():

                if lb is None and ub is None:
                    param_random -= tf.reduce_mean(param_random, axis=0, keepdims=True)

                    return param_random

                elif lb is not None and ub is None:
                    max_lower_offset = param_fixed - (lb + self.epsilon)

                    # Enforce constraint
                    param_random = self.constraint_fn(param_random)

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_lower_offset / (param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

                elif lb is None and ub is not None:
                    max_upper_offset = ub - param_fixed - self.epsilon

                    # Enforce constraint
                    param_random = -self.constraint_fn(param_random)

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_upper_offset / (-param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

                else:
                    max_lower_offset = param_fixed - (lb + self.epsilon)

                    # Enforce constraint
                    if use_softplus_sigmoid:
                        param_random = self._softplus_sigmoid(param_random, a=lb, b=ub)
                    else:
                        max_range_param = ub - lb
                        param_random = tf.sigmoid(param_random) * max_range_param

                    # Center
                    param_random_mean = tf.reduce_mean(param_random, axis=0, keepdims=True)
                    param_random -= param_random_mean

                    # Rescale to re-enforce constraint
                    correction_factor = max_lower_offset / (param_random_mean + self.epsilon) # Add epsilon in case of underflow in softplus
                    param_random *= correction_factor

                    return param_random

    def _linspace_nd(self, B, A=None, axis=0, n_interp=None):
        if n_interp is None:
            n_interp = self.n_interp
        if axis < 0:
            axis = len(B.shape) + axis
        with self.sess.as_default():
            with self.sess.graph.as_default():
                linspace_support = tf.cast(tf.range(n_interp), dtype=self.FLOAT_TF)
                B = tf.expand_dims(B, axis)
                rank = len(B.shape)
                assert axis < rank, 'Tried to perform linspace_nd on axis %s, which exceeds rank %s of tensor' %(axis, rank)
                expansion = ([None] * axis) + [slice(None)] + ([None] * (rank - axis - 1))
                linspace_support = linspace_support[expansion]

                if A is None:
                    out = B * linspace_support / n_interp
                else:
                    A = tf.expand_dims(A, axis)
                    assert A.shape == B.shape, 'A and B must have the same shape, got %s and %s' %(A.shape, B.shape)
                    out = A + ((B-A) * linspace_support / n_interp)
                return out

    def _lininterp_fixed_n_points(self, x, n):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_input = tf.shape(x)[1]
                n_output = n_input * (n+1)
                interp = tf.image.resize_bilinear(tf.expand_dims(tf.expand_dims(x, -1), -1), [n_output, 1])
                interp = tf.squeeze(tf.squeeze(interp, -1), -1)[..., :-n]
                return interp

    def _lininterp_fixed_frequency(self, x, time, time_mask, hz=1000):
        # Performs a linear interpolation at a fixed frequency between impulses that are variably spaced in time.
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Reverse arrays so that max time is left-aligned
                x = tf.reverse(x, axis=[1])
                x_deltas = x[:, 1:, :] - x[:, :-1, :]
                time = tf.reverse(time, axis=[1])
                time_mask = tf.reverse(time_mask, axis=[1])
                time_mask_int = tf.cast(time_mask, dtype=self.INT_TF)

                # Round timestamps to integers so they can be used to index the interpolated array
                time_int = tf.cast(tf.round(time * hz), dtype=self.INT_TF)

                # Compute intervals by subtracting the lower bound
                end_ix = tf.reduce_sum(time_mask_int, axis=1) - 1
                time_minima = tf.gather_nd(time_int, tf.stack([tf.range(tf.shape(x)[0]), end_ix], axis=1))
                time_delta_int = time_int - time_minima[..., None]

                # Compute the minimum number of interpolation points needed for each data point
                impulse_ix_max_batch = tf.reduce_max(time_delta_int, axis=1, keepdims=True)

                # Compute the largest number of interpolation points in the batch, which will be used to compute the length of the interpolation
                impulse_ix_max = tf.reduce_max(impulse_ix_max_batch)

                # Since events are now temporally reversed, reverse indices by subtracting the max and negating
                impulse_ix = -(time_delta_int - impulse_ix_max_batch)

                # Rescale the deltas by the number of timesteps over which they will be interpolated
                n_steps = impulse_ix[:, 1:] - impulse_ix[:, :-1]
                x_deltas = tf.where(
                    tf.tile(tf.not_equal(n_steps, 0)[..., None], [1, 1, tf.shape(x)[2]]),
                    x_deltas / tf.cast(n_steps, dtype=self.FLOAT_TF)[..., None],
                    tf.zeros_like(x_deltas)
                )

                # Pad x_deltas
                x_deltas = tf.concat([x_deltas, tf.zeros([tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)

                # Compute a mask for the interpolated output
                time_mask_interp = tf.cast(tf.range(impulse_ix_max + 1)[None, ...] <= impulse_ix_max_batch, dtype=self.FLOAT_TF)

                # Compute an array of indices for scattering impulses into the interpolation array
                row_ix = tf.tile(tf.range(tf.shape(x)[0])[..., None], [1, tf.shape(x)[1]])
                scatter_ix = tf.stack([row_ix, impulse_ix], axis=2)

                # Create an array for use by gather_nd by taking the cumsum of an array with ones at indices with impulses, zeros otherwise
                gather_ix = tf.cumsum(
                    tf.scatter_nd(
                        scatter_ix,
                        tf.ones_like(impulse_ix, dtype=self.INT_TF) * time_mask_int,
                        [tf.shape(x)[0], impulse_ix_max + 1]
                    ),
                    axis=1
                ) - 1
                row_ix = tf.tile(tf.range(tf.shape(x)[0])[..., None], [1, impulse_ix_max + 1])
                gather_ix = tf.stack([row_ix, gather_ix], axis=2)

                x_interp_base = tf.gather_nd(
                    x,
                    gather_ix
                )

                interp_factor = tf.cast(
                    tf.range(impulse_ix_max + 1)[None, ...] - tf.gather_nd(
                        impulse_ix,
                        gather_ix
                    ),
                    dtype=self.FLOAT_TF
                )


                interp_delta = tf.cast(interp_factor, dtype=self.FLOAT_TF)[..., None] * tf.gather_nd(
                    x_deltas,
                    gather_ix
                )

                x_interp = (x_interp_base + interp_delta) * time_mask_interp[..., None]

                x_interp = tf.reverse(x_interp, axis=[1])
                time_interp = tf.cast(
                    tf.reverse(
                        tf.maximum(
                            (tf.range(0, -impulse_ix_max - 1, delta=-1)[None, ...] + impulse_ix_max_batch),
                            tf.zeros([impulse_ix_max + 1], dtype=self.INT_TF)
                        ) + time_minima[..., None],
                        axis=[1]
                    ),
                    dtype=self.FLOAT_TF
                ) * tf.reverse(time_mask_interp, axis=[1]) / hz

                return x_interp, time_interp

    def _safe_abs(self, x):
        out = tf.where(
            tf.equal(x, 0.),
            x + 1e-8,
            tf.abs(x)
        )

        return out







    ######################################################
    #
    #  Public methods that must be implemented by
    #  subclasses
    #
    ######################################################

    def run_train_step(self, feed_dict):
        """
        Update the model from a batch of training data.
        **All subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """

        raise NotImplementedError

    def run_predict_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Generate predictions from a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor values.
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Predicted responses, one for each training sample
        """
        raise NotImplementedError

    def run_loglik_op(self, feed_dict, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Compute the log-likelihoods of a batch of data.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor and response values
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; Pointwise log-likelihoods, one for each training sample
        """

        raise NotImplementedError

    def run_conv_op(self, feed_dict, scaled=False, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.
        **All DTSR subclasses must implement this method.**

        :param feed_dict: ``dict``; A dictionary of predictor variables
        :param scaled: ``bool``; Whether to scale the outputs using the model's coefficients
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless ``scaled==True`` and model was fitted using ``standardize_response==True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; Algorithm (``MAP`` or ``sampling``) to use for extracting predictions. Only relevant for variational Bayesian models. If ``MAP``, uses posterior means as point estimates for the parameters (no sampling). If ``sampling``, draws **n_samples** from the posterior.
        :param verbose: ``bool``; Send progress reports to standard error.
        :return: ``numpy`` array; The convolved inputs
        """

        raise NotImplementedError

    def extract_irf_integral(self, terminal_name, rangf=None, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        """
        Extract integrals of IRF defined at **terminal_name**.

        :param terminal_name: ``str``; ID of terminal IRF to extract
        :param rangf: ``numpy`` array or ``None``; random grouping factor values for which to compute IRF integral. If ``None``, only use fixed effects.
        :param level: ``float``; level of credible interval (used for ``DTSRBayes`` only)
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw (used for ``DTSRBayes`` only). If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``float`` or 3-element ``numpy`` vector; either integral or mean, upper quantile, and lower quantile of integral (depending on whether model is instance of ``DTSRBayes``).
        """
        raise NotImplementedError




    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def verify_random_centering(self):
        """
        Assert that all random effects are properly centered (means sufficiently close to zero).

        :return: ``None``
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if len(self.rangf) > 0:
                    means = self.random_means.eval(session=self.sess)
                    centered = np.allclose(means, 0., rtol=1e-3, atol=1e-3)
                    assert centered, 'Some random parameters are not properly centered\n. Current random parameter means:\n %s' %means

    def build(self, outdir=None, restore=True):
        """
        Construct the DTSR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Report model details after initialization.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './dtsr_model/'
        else:
            self.outdir = outdir

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self._initialize_inputs()
                self._initialize_base_params()
                self._initialize_joint_distributions()
                self._initialize_joint_distribution_slices()
                self._initialize_intercepts_coefficients_interactions()
                self._initialize_irf_lambdas()
                self._initialize_irf_params()
                self._initialize_parameter_tables()
                self._initialize_random_mean_vector()
                self._construct_network()
                self.initialize_objective()
                self._initialize_logging()
                self._initialize_ema()

                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
                self._initialize_saver()
                self.load(restore=restore)

                self._initialize_convergence_checking()
                self._collect_plots()

    def save(self, dir=None):
        """
        Save the DTSR model.

        :param dir: ``str``; output directory. If ``None``, use model default.
        :return: ``None``
        """

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except:
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        pytime.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)
    
    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DTSR checkpoint and/or initialize the DTSR model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def finalize(self):
        """
        Close the DTSR instance to prevent memory leaks.

        :return: ``None``
        """
        self.sess.close()

    def irf_integrals(self, level=95, random=False, n_samples=None, n_time_units=None, n_time_points=1000):
        """
        Generate effect size estimates by computing the area under each IRF curve in the model via discrete approximation.

        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``pandas`` DataFrame; IRF integrals, one IRF per row. If Bayesian, array also contains credible interval bounds.
        """
        if self.pc:
            terminal_names = self.src_terminal_names
        else:
            terminal_names = self.terminal_names
        irf_integrals = []
        
        rangf_keys = ['']
        rangf_groups = ['']
        rangf_vals = [self.gf_defaults[0]]
        if random:
            for i in range(len(self.rangf)):
                if self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                    for k in self.rangf_map[i].keys():
                        rangf_keys.append(str(k))
                        rangf_groups.append(self.rangf[i])
                        rangf_vals.append(np.concatenate([self.gf_defaults[0, :i], [self.rangf_map[i][k]], self.gf_defaults[0, i + 1:]], axis=0))
        rangf_vals = np.stack(rangf_vals, axis=0)
        
        for i in range(len(terminal_names)):
            terminal = terminal_names[i]
            integral = np.stack(
                self.irf_integral(
                    terminal,
                    rangf=rangf_vals,
                    level=level,
                    n_samples=n_samples,
                    n_time_units=n_time_units,
                    n_time_points=n_time_points
                ),
                axis=0
            )
            irf_integrals.append(integral)
        irf_integrals = np.stack(irf_integrals, axis=0)
        if self.standardize_response:
            irf_integrals *= self.y_train_sd
        irf_integrals = np.split(irf_integrals, irf_integrals.shape[2], axis=2)
        
        if self.pc:
            terminal_names = self.src_terminal_names
        else:
            terminal_names = self.terminal_names
        for i, x in enumerate(irf_integrals):
            if rangf_keys[i]:
                terminal_names_cur = [y + '_' + rangf_keys[i] for y in terminal_names]
            else:
                terminal_names_cur = terminal_names
            x = pd.DataFrame(x[..., 0], columns=self.parameter_table_columns)
            x['IRF'] = terminal_names
            cols = ['IRF']
            if random:
                x['Group'] = rangf_groups[i]
                x['Level'] = rangf_keys[i]
                cols += ['Group', 'Level']
            cols += self.parameter_table_columns
            x = x[cols]
            irf_integrals[i] = x
        irf_integrals = pd.concat(irf_integrals, axis=0)

        return irf_integrals

    def irf_integral(self, terminal_name, rangf=None, level=95, n_samples=None, n_time_units=None, n_time_points=1000):
        """
        Generate effect size estimates by computing the area under a specific IRF curve via discrete approximation.

        :param terminal_name: ``str``; string ID of IRF to extract.
        :param rangf: ``numpy`` array or ``None``; random grouping factor values for which to compute IRF integral. If ``None``, only use fixed effects.
        :param level: ``float``; level of the credible interval if Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``numpy`` array; IRF integral (scalar), or (if Bayesian) IRF 3x1 vector with mean, lower bound, and upper bound of credible interval.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return self.extract_irf_integral(
                    terminal_name,
                    rangf=rangf,
                    level=level,
                    n_samples=n_samples,
                    n_time_units=n_time_units,
                    n_time_points=n_time_points
                )

    def set_predict_mode(self, mode):
        """
        Set predict mode.
        If set to ``True``, the model enters predict mode and replaces parameters with the exponential moving average of their training iterates.
        If set to ``False``, the model exits predict mode and replaces parameters with their most recently saved values.
        To avoid data loss, always save the model before entering predict mode.

        :param mode: ``bool``; if ``True``, enter predict mode. If ``False``, exit predict mode.
        :return: ``None``
        """

        if mode != self.predict_mode:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.load(predict=mode)

            self.predict_mode = mode

    def has_converged(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.check_convergence:
                    return self.sess.run(self.converged)
                else:
                    return False

    def set_training_complete(self, status):
        """
        Change internal record of whether training is complete.
        Training is recorded as complete when fit() terminates.
        If fit() is called again with a larger number of iterations, training is recorded as incomplete and will not change back to complete until either fit() is called or set_training_complete() is called and the model is saved.

        :param status: ``bool``; Target state (``True`` if training is complete, ``False`` otherwise).
        :return: ``None``
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if status:
                    self.sess.run(self.training_complete_true)
                else:
                    self.sess.run(self.training_complete_false)

    def report_formula_string(self, indent=0):
        """
        Generate a string representation of the model formula.

        :param indent: ``int``; indentation level
        :return: ``str``; the formula report
        """
        out = ' ' * indent + 'MODEL FORMULA:\n'
        form_str = textwrap.wrap(str(self.form), 150)
        for line in form_str:
            out += ' ' * indent + '  ' + line + '\n'

        out += '\n'

        return out

    def report_settings(self, indent=0):
        """
        Generate a string representation of the model settings.

        :param indent: ``int``; indentation level
        :return: ``str``; the settings report
        """
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        for kwarg in DTSR_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def report_irf_tree(self, indent=0):
        """
        Generate a string representation of the model's IRF tree structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the IRF tree report
        """

        out = ''

        out += ' ' * indent + 'IRF TREE:\n'
        tree_str = str(self.t)
        new_tree_str = ''
        for line in tree_str.splitlines():
            new_tree_str += ' ' * (indent + 2) + line + '\n'
        out += new_tree_str + '\n'

        if self.pc:
            out += ' ' * indent + 'SOURCE IRF TREE:\n'
            tree_str = str(self.t_src)
            new_tree_str = ''
            for line in tree_str.splitlines():
                new_tree_str += ' ' * (indent + 2) + line + '\n'
            out += new_tree_str + '\n'

        return out

    def report_impulse_types(self, indent=0):
        """
        Generate a string representation of types of impulses (transient or continuous) in the model.

        :param indent: ``int``; indentation level
        :return: ``str``; the impulse type report
        """

        out = ''
        out += ' ' * indent + 'IMPULSE TYPES:\n'

        if self.pc:
            t = self.t_src
        else:
            t = self.t
        for x in t.terminals():
            out += ' ' * (indent + 2) + x.name() + ': ' + ('continuous' if x.cont else 'transient') + '\n'

        out += '\n'

        return out

    def report_n_params(self, indent=0):
        """
        Generate a string representation of the number of trainable model parameters

        :param indent: ``int``; indentation level
        :return: ``str``; the num. parameters report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = int(np.prod(np.array(v_val).shape))
                    n_params += cur_params
                    out += ' ' * indent + '  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params)
                out +=  ' ' * indent + '  TOTAL: %d\n\n' % n_params

                return out

    def report_regularized_variables(self, indent=0):
        """
        Generate a string representation of the model's regularization structure.

        :param indent: ``int``; indentation level
        :return: ``str``; the regularization report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                assert len(self.regularizer_losses) == len(self.regularizer_losses_names) == len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 'Different numbers of regularized variables found in different places'

                out = ' ' * indent + 'REGULARIZATION:\n'

                if len(self.regularizer_losses_names) == 0:
                    out +=  ' ' * indent + '  No regularized variables.\n\n'
                else:
                    for i, name in enumerate(self.regularizer_losses_varnames):
                        out += ' ' * indent + '  %s:\n' %name
                        out += ' ' * indent + '    Regularizer: %s\n' %self.regularizer_losses_names[i]
                        out += ' ' * indent + '    Scale: %s\n' %self.regularizer_losses_scales[i]

                    out += '\n'

                return out

    def report_training_mse(self, indent=0):
        """
        Generate a string representation of the model's training MSE.

        :param indent: ``int``; indentation level
        :return: ``str``; the training MSE report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_mse = self.training_mse.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING MSE:\n'
                out += ' ' * (indent + 2) + str(training_mse)
                out += '\n\n'

                return out

    def report_training_mae(self, indent=0):
        """
        Generate a string representation of the model's training MAE.

        :param indent: ``int``; indentation level
        :return: ``str``; the training MAE report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_mae = self.training_mae.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING MAE:\n'
                out += ' ' * (indent + 2) + str(training_mae)
                out += '\n\n'

                return out

    def report_training_loglik(self, indent=0):
        """
        Generate a string representation of the model's training log likelihood.

        :param indent: ``int``; indentation level
        :return: ``str``; the training log likelihood report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loglik_train = self.training_loglik.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING LOG LIKELIHOOD:\n'
                out += ' ' * (indent + 2) + str(loglik_train)
                out += '\n\n'

                return out

    def report_training_percent_variance_explained(self, indent=0):
        """
        Generate a string representation of the percent variance explained by the model on training data.

        :param indent: ``int``; indentation level
        :return: ``str``; the training percent variance explained report
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                training_percent_variance_explained = self.training_percent_variance_explained.eval(session=self.sess)

                out = ' ' * indent + 'TRAINING PERCENT VARIANCE EXPLAINED:\n'
                out += ' ' * (indent + 2) + '%.2f' %training_percent_variance_explained + '%'
                out += '\n\n'

                return out

    def report_evaluation(
            self,
            mse=None,
            mae=None,
            loglik=None,
            loss=None,
            percent_variance_explained=None,
            true_variance=None,
            ks_results=None,
            indent=0
    ):
        """
        Generate a string representation of pre-comupted evaluation metrics.

        :param mse: ``float`` or ``None``; mean squared error, skipped if ``None``.
        :param mae: ``float`` or ``None``; mean absolute error, skipped if ``None``.
        :param loglik: ``float`` or ``None``; log likelihood, skipped if ``None``.
        :param loss: ``float`` or ``None``; loss per training objective, skipped if ``None``.
        :param true_variance: ``float`` or ``None``; variance of targets, skipped if ``None``.
        :param percent_variance_explained: ``float`` or ``None``; percent variance explained, skipped if ``None``.
        :param true_variance: ``float`` or ``None``; true variance, skipped if ``None``.
        :param ks_results: pair of ``float`` or ``None``; if non-null, pair containing ``(D, p_value)`` from Kolmogorov-Smirnov test of errors againts fitted error distribution ; skipped if ``None``.
        :param indent: ``int``; indentation level
        :return: ``str``; the evaluation report
        """
        out = ' ' * indent + 'MODEL EVALUATION STATISTICS:\n'
        if mse is not None:
            out += ' ' * (indent+2) + 'MSE: %s\n' %mse
        if mae is not None:
            out += ' ' * (indent+2) + 'MAE: %s\n' %mae
        if loglik is not None:
            out += ' ' * (indent+2) + 'Log likelihood: %s\n' %loglik
        if loss is not None:
            out += ' ' * (indent+2) + 'Loss per training objective: %s\n' %loss
        if true_variance is not None:
            out += ' ' * (indent+2) + 'True variance: %s\n' %true_variance
        if percent_variance_explained is not None:
            out += ' ' * (indent+2) + 'Percent variance explained: %.2f%%\n' %percent_variance_explained
        if ks_results is not None:
            out += ' ' * (indent+2) + 'Kolmogorov-Smirnov test of goodness of fit of modeled to true error:\n'
            out += ' ' * (indent+4) + 'D value: %s\n' % ks_results[0]
            out += ' ' * (indent+4) + 'p value: %s\n' % ks_results[1]
            if ks_results[1] < 0.05:
                out += '\n'
                out += ' ' * (indent+4) + 'NOTE: KS tests will likely reject on large datasets.\n'
                out += ' ' * (indent+4) + 'This does not entail that the model is fatally flawed.\n'
                out += ' ' * (indent+4) + "Check the Q-Q plot in the model's output directory.\n"
                if not self.asymmetric_error:
                    out += ' ' * (indent+4) + 'Poor error fit can usually be improved without transforming\n'
                    out += ' ' * (indent+4) + 'the response by optimizing using ``asymmetric_error=True``.\n'
                    out += ' ' * (indent+4) + 'Consult the documentation for details.\n'

        out += '\n'

        return out

    def report_parameter_values(self, random=False, level=95, n_samples=None, indent=0):
        """
        Generate a string representation of the model's parameter table.

        :param random: ``bool``; report random effects estimates.
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param indent: ``int``; indentation level.
        :return: ``str``; the parameter table report
        """
        left_justified_formatter = lambda df, col: '{{:<{}s}}'.format(df[col].str.len().max()).format

        pd.set_option("display.max_colwidth", 10000)
        out = ' ' * indent + 'FITTED PARAMETER VALUES:\n'
        parameter_table = self.parameter_table(
            fixed=True,
            level=level,
            n_samples=n_samples
        )
        formatters = {
            'Parameter': left_justified_formatter(parameter_table, 'Parameter')
        }
        parameter_table_str = parameter_table.to_string(
            index=False,
            justify='left',
            formatters = formatters
        )

        out += ' ' * (indent + 2) + 'Fixed:\n'
        for line in parameter_table_str.splitlines():
            out += ' ' * (indent + 4) + line + '\n'
        out += '\n'

        if random:
            parameter_table = self.parameter_table(
                fixed=False,
                level=level,
                n_samples=n_samples
            )
            parameter_table = pd.concat(
                [
                    pd.DataFrame({'Parameter': parameter_table['Parameter'] + ' | ' + parameter_table['Group'] + ' | ' + parameter_table['Level']}),
                    parameter_table[self.parameter_table_columns]
                ],
                axis=1
            )
            formatters = {
                'Parameter': left_justified_formatter(parameter_table, 'Parameter')
            }
            parameter_table_str = parameter_table.to_string(
                index=False,
                justify='left',
                formatters = formatters
            )

            out += ' ' * (indent + 2) + 'Random:\n'
            for line in parameter_table_str.splitlines():
                out += ' ' * (indent + 4) + line + '\n'
            out += '\n'

        pd.set_option("display.max_colwidth", 50)

        return out

    def report_irf_integrals(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's IRF integrals (effect sizes)

        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the IRF integrals report
        """
        
        pd.set_option("display.max_colwidth", 10000)
        left_justified_formatter = lambda df, col: '{{:<{}s}}'.format(df[col].str.len().max()).format

        if integral_n_time_units is None:
            integral_n_time_units = self.t_delta_limit

        irf_integrals = self.irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            n_time_units=integral_n_time_units,
            n_time_points=1000
        )

        formatters = {
            'IRF': left_justified_formatter(irf_integrals, 'IRF')
        }

        out = ' ' * indent + 'IRF INTEGRALS (EFFECT SIZES):\n'
        out += ' ' * (indent + 2) + 'Integral upper bound (time): %s\n\n' % integral_n_time_units

        ci_str = irf_integrals.to_string(
            index=False,
            justify='left',
            formatters=formatters
        )

        for line in ci_str.splitlines():
            out += ' ' * (indent + 2) + line + '\n'

        out += '\n'

        return out

    def initialization_summary(self, indent=0):
        """
        Generate a string representation of the model's initialization details

        :param indent: ``int``; indentation level.
        :return: ``str``; the initialization summary
        """

        out = ' ' * indent + '----------------------\n'
        out += ' ' * indent + 'INITIALIZATION SUMMARY\n'
        out += ' ' * indent + '----------------------\n\n'

        out += self.report_formula_string(indent=indent+2)
        out += self.report_settings(indent=indent+2)
        out += '\n' + ' ' * (indent + 2) + 'Training iterations completed: %d\n\n' %self.global_step.eval(session=self.sess)
        out += self.report_irf_tree(indent=indent+2)
        out += self.report_impulse_types(indent=indent+2)
        out += self.report_n_params(indent=indent+2)
        out += self.report_regularized_variables(indent=indent+2)

        return out

    def training_evaluation_summary(self, indent=0):
        """
        Generate a string representation of the model's training metrics.
        Correctness is not guaranteed until fit() has successfully exited.

        :param indent: ``int``; indentation level.
        :return: ``str``; the training evaluation summary
        """

        out = ' ' * indent + '---------------------------\n'
        out += ' ' * indent + 'TRAINING EVALUATION SUMMARY\n'
        out += ' ' * indent + '---------------------------\n\n'

        out += self.report_training_mse(indent=indent+2)
        out += self.report_training_mae(indent=indent+2)
        out += self.report_training_loglik(indent=indent+2)
        out += self.report_training_percent_variance_explained(indent=indent+2)

        return out

    def convergence_summary(self, indent=0):
        """
        Generate a string representation of model's convergence status.

        :param indent: ``int``; indentation level
        :return: ``str``; the convergence report
        """

        out = ' ' * indent + '-------------------\n'
        out += ' ' * indent + 'CONVERGENCE SUMMARY\n'
        out += ' ' * indent + '-------------------\n\n'

        if self.check_convergence:
            n_iter = self.global_step.eval(session=self.sess)
            min_p_ix, min_p, rt_at_min_p, ra_at_min_p, p_ta_at_min_p, proportion_converged, converged = self.run_convergence_check(verbose=False)
            location = self.d0_names[min_p_ix]

            out += ' ' * (indent * 2) + 'Converged: %s\n' % converged
            out += ' ' * (indent * 2) + 'Convergence basis: %s\n' % self.convergence_basis.lower()
            out += ' ' * (indent * 2) + 'Convergence n iterates: %s\n' % self.convergence_n_iterates
            out += ' ' * (indent * 2) + 'Convergence stride: %s\n' % self.convergence_stride
            out += ' ' * (indent * 2) + 'Convergence alpha: %s\n' % self.convergence_alpha
            out += ' ' * (indent * 2) + 'Convergence min p of rho_t: %s\n' % min_p
            out += ' ' * (indent * 2) + 'Convergence rho_t at min p: %s\n' % rt_at_min_p
            if self.convergence_basis.lower() == 'parameters':
                out += ' ' * (indent * 2) + 'Convergence rho_a at min p: %s\n' % ra_at_min_p
                out += ' ' * (indent * 2) + 'Convergence p of rho_a at min p: %s\n' % p_ta_at_min_p
            out += ' ' * (indent * 2) + 'Proportion converged: %s\n' % proportion_converged

            if converged:
                out += ' ' * (indent + 2) + 'NOTE:\n'
                out += ' ' * (indent + 4) + 'Programmatic diagnosis of convergence in DTSR is error-prone because of stochastic optimization.\n'
                out += ' ' * (indent + 4) + 'It is possible that the convergence diagnostics used are too permissive given the stochastic dynamics of the model.\n'
                out += ' ' * (indent + 4) + 'Consider visually checking the learning curves in Tensorboard to see whether the losses have flatlined:\n'
                out += ' ' * (indent + 6) + 'python -m tensorboard.main --logdir=<path_to_model_directory>\n'
                out += ' ' * (indent + 4) + 'If not, consider raising **convergence_alpha** and resuming training.\n'

            else:
                out += ' ' * (indent + 2) + 'Model did not reach convergence criteria in %s epochs.\n' % n_iter
                out += ' ' * (indent + 2) + 'NOTE:\n'
                out += ' ' * (indent + 4) + 'Programmatic diagnosis of convergence in DTSR is error-prone because of stochastic optimization.\n'
                out += ' ' * (indent + 4) + 'It is possible that the convergence diagnostics used are too conservative given the stochastic dynamics of the model.\n'
                out += ' ' * (indent + 4) + 'Consider visually checking the learning curves in Tensorboard to see whether thelosses have flatlined:\n'
                out += ' ' * (indent + 6) + 'python -m tensorboard.main --logdir=<path_to_model_directory>\n'
                out += ' ' * (indent + 4) + 'If so, consider the model converged.\n'

        else:
            out += ' ' * (indent + 2) + 'Convergence checking is turned off.\n'

        return out

    def parameter_summary(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a string representation of the model's effect sizes and parameter values.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param indent: ``int``; indentation level.
        :return: ``str``; the parameter summary
        """

        out = ' ' * indent + '-----------------\n'
        out += ' ' * indent + 'PARAMETER SUMMARY\n'
        out += ' ' * indent + '-----------------\n\n'

        out += self.report_irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            integral_n_time_units=integral_n_time_units,
            indent=indent+2
        )

        out += self.report_parameter_values(
            random=random,
            level=level,
            n_samples=n_samples,
            indent=indent+2
        )

        return out

    def summary(self, random=False, level=95, n_samples=None, integral_n_time_units=None, indent=0):
        """
        Generate a summary of the fitted model.

        :param random: ``bool``; report random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :return: ``str``; the model summary
        """

        out = '  ' * indent + '*' * 100 + '\n\n'
        out += ' ' * indent + '############################\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '#    DTSR MODEL SUMMARY    #\n'
        out += ' ' * indent + '#                          #\n'
        out += ' ' * indent + '############################\n\n\n'

        out += self.initialization_summary(indent=indent + 2)
        out += '\n'
        out += self.training_evaluation_summary(indent=indent + 2)
        out += '\n'
        out += self.convergence_summary(indent=indent + 2)
        out += '\n'
        out += self.parameter_summary(
            random=random,
            level=level,
            n_samples=n_samples,
            integral_n_time_units=integral_n_time_units,
            indent=indent + 2
        )
        out += '\n'
        out += '  ' * indent + '*' * 100 + '\n\n'

        return out


    ######################################################
    #
    #  High-level methods for training, prediction,
    #  and plotting
    #
    ######################################################

    def fit(self,
            X,
            y,
            n_iter=100,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            force_training_evaluation=True,
            irf_name_map=None,
            plot_n_time_units=2.5,
            plot_n_time_points=1000,
            plot_x_inches=7,
            plot_y_inches=5,
            cmap='gist_rainbow',
            dpi=300
            ):
        """
        Fit the DTSR model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the DTSR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in **y**
            * ``first_obs``:  Index in the design matrix **X** of the first observation in the time series associated with each entry in **y**
            * ``last_obs``:  Index in the design matrix **X** of the immediately preceding observation in the time series associated with each entry in **y**
            * A column with the same name as the dependent variable specified in the model formula
            * A column for each random grouping factor in the model formula

            In general, **y** will be identical to the parameter **y** provided at model initialization.
            This must hold for MCMC inference, since the number of minibatches is built into the model architecture.
            However, it is not necessary for variational inference.
        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param force_training_evaluation: ``bool``; (Re-)run post-fitting evaluation, even if resuming a model whose training is already complete.
        :param n_iter: ``int``; the number of training iterations
        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``float``; number of points to use for plotting.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param dpi: ``int``; dots per inch.
        :return: ``None``
        """

        sys.stderr.write('*' * 100 + '\n')
        sys.stderr.write(self.initialization_summary())
        sys.stderr.write('*' * 100 + '\n\n')

        usingGPU = tf.test.is_gpu_available()
        sys.stderr.write('Using GPU: %s\n' % usingGPU)
        sys.stderr.write('Number of training samples: %d\n\n' % len(y))

        if self.pc:
            impulse_names = self.src_impulse_names
            assert X_2d_predictors is None, 'Principal components regression not supported for models with 2d predictors'
        else:
            impulse_names  = self.impulse_names

        if not np.isfinite(self.minibatch_size):
            minibatch_size = len(y)
        else:
            minibatch_size = self.minibatch_size
        n_minibatch = math.ceil(float(len(y)) / minibatch_size)

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        sys.stderr.write('Correlation matrix for input variables:\n')
        impulse_names_2d = [x for x in impulse_names if x in X_2d_predictor_names]
        rho = corr_dtsr(X_2d, impulse_names, impulse_names_2d, time_X_2d, time_X_mask)
        sys.stderr.write(str(rho) + '\n\n')

        # self.make_plots(
        #     irf_name_map=irf_name_map,
        #     plot_n_time_units=plot_n_time_units,
        #     plot_n_time_points=plot_n_time_points,
        #     plot_x_inches=plot_x_inches,
        #     plot_y_inches=plot_y_inches,
        #     cmap=cmap,
        #     dpi=dpi,
        #     keep_plot_history=self.keep_plot_history
        # )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.run_convergence_check(verbose=False)

                if (self.global_step.eval(session=self.sess) < n_iter) and not self.has_converged():
                    self.set_training_complete(False)

                if self.training_complete.eval(session=self.sess):
                    sys.stderr.write('Model training is already complete; no additional updates to perform. To train for additional iterations, re-run fit() with a larger n_iter.\n\n')
                else:
                    if self.global_step.eval(session=self.sess) == 0:
                        summary_params = self.sess.run(self.summary_params)
                        self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                        if self.log_random and len(self.rangf) > 0:
                            summary_random = self.sess.run(self.summary_random)
                            self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))
                    else:
                        sys.stderr.write('Resuming training from most recent checkpoint...\n\n')

                    while not self.has_converged() and self.global_step.eval(session=self.sess) < n_iter:
                        p, p_inv = get_random_permutation(len(y))
                        t0_iter = pytime.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            sys.stderr.write('Learning rate: %s\n' %self.lr.eval(session=self.sess))

                        pb = tf.contrib.keras.utils.Progbar(self.n_train_minibatch)

                        loss_total = 0.

                        for j in range(0, len(y), minibatch_size):
                            indices = p[j:j+minibatch_size]
                            fd_minibatch = {
                                self.X: X_2d[indices],
                                self.time_X: time_X_2d[indices],
                                self.time_X_mask: time_X_mask[indices],
                                self.y: y_dv[indices],
                                self.time_y: time_y[indices],
                                self.gf_y: gf_y[indices] if len(gf_y > 0) else gf_y
                            }

                            info_dict = self.run_train_step(fd_minibatch)
                            loss_cur = info_dict['loss']
                            self.sess.run(self.ema_op)
                            if not np.isfinite(loss_cur):
                                loss_cur = 0
                            loss_total += loss_cur

                            pb.update((j/minibatch_size)+1, values=[('loss', loss_cur)])

                        self.sess.run(self.incr_global_step)

                        self.verify_random_centering()

                        if self.check_convergence:
                            self.run_convergence_check(verbose=False, feed_dict={self.loss_total: loss_total/n_minibatch})

                        if self.log_freq > 0 and self.global_step.eval(session=self.sess) % self.log_freq == 0:
                            loss_total /= n_minibatch
                            summary_train_loss = self.sess.run(self.summary_losses, {self.loss_total: loss_total})
                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))
                            self.writer.add_summary(summary_train_loss, self.global_step.eval(session=self.sess))
                            if self.log_random and len(self.rangf) > 0:
                                summary_random = self.sess.run(self.summary_random)
                                self.writer.add_summary(summary_random, self.global_step.eval(session=self.sess))

                        if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                            self.save()
                            self.make_plots(
                                irf_name_map=irf_name_map,
                                plot_n_time_units=plot_n_time_units,
                                plot_n_time_points=plot_n_time_points,
                                plot_x_inches=plot_x_inches,
                                plot_y_inches=plot_y_inches,
                                cmap=cmap,
                                dpi=dpi,
                                keep_plot_history=self.keep_plot_history
                            )

                            self.verify_random_centering()

                        t1_iter = pytime.time()
                        sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                    self.save()

                    # End of training plotting and evaluation.
                    # For DTSRMLE, this is a crucial step in the model definition because it provides the
                    # variance of the output distribution for computing log likelihood.

                    self.make_plots(
                        irf_name_map=irf_name_map,
                        plot_n_time_units=plot_n_time_units,
                        plot_n_time_points=plot_n_time_points,
                        plot_x_inches=plot_x_inches,
                        plot_y_inches=plot_y_inches,
                        cmap=cmap,
                        dpi=dpi,
                        keep_plot_history=self.keep_plot_history
                    )

                    if type(self).__name__ == 'DTSRBayes':
                        # Generate plots with 95% credible intervals
                        self.make_plots(
                            irf_name_map=irf_name_map,
                            plot_n_time_units=plot_n_time_units,
                            plot_n_time_points=plot_n_time_points,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            cmap=cmap,
                            dpi=dpi,
                            mc=True,
                            keep_plot_history=self.keep_plot_history
                        )


                if not self.training_complete.eval(session=self.sess) or force_training_evaluation:
                    # Extract and save predictions
                    preds = self.predict(
                        X,
                        y.time,
                        y[self.form.rangf],
                        first_obs,
                        last_obs,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors
                    )

                    with open(self.outdir + '/preds_train.txt', 'w') as p_file:
                        for i in range(len(preds)):
                            p_file.write(str(preds[i]) + '\n')

                    # Extract and save losses
                    training_se = np.array((y[self.dv] - preds) ** 2)
                    training_mse = training_se.mean()
                    training_percent_variance_explained = percent_variance_explained(y[self.dv], preds)

                    training_ae = np.array(np.abs(y[self.dv] - preds))
                    training_mae = training_ae.mean()

                    # Extract and save log likelihoods
                    training_logliks = self.log_lik(
                        X,
                        y,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors,
                    )
                    with open(self.outdir + '/loglik_train.txt','w') as l_file:
                        for i in range(len(training_logliks)):
                            l_file.write(str(training_logliks[i]) + '\n')
                    training_loglik = training_logliks.sum()

                    # Store training evaluation statistics in the graph
                    self.sess.run(
                        [self.set_training_mse, self.set_training_mae],
                        feed_dict={
                            self.training_mse_in: training_mse,
                            self.training_mae_in: training_mae,
                        }
                    )

                    self.sess.run(
                        [self.set_training_loglik],
                        feed_dict={
                            self.training_loglik_in: training_loglik
                        }
                    )

                    self.save()

                    with open(self.outdir + '/eval_train.txt', 'w') as e_file:
                        eval_train = '------------------------\n'
                        eval_train += 'DTSR TRAINING EVALUATION\n'
                        eval_train += '------------------------\n\n'
                        eval_train += self.report_formula_string(indent=2)
                        eval_train += self.report_evaluation(
                            mse=training_mse,
                            mae=training_mae,
                            loglik=training_loglik,
                            percent_variance_explained=training_percent_variance_explained,
                            indent=2
                        )

                        e_file.write(eval_train)

                    self.save_parameter_table()
                    self.save_integral_table()

                    self.set_training_complete(True)

                    self.save()

    def predict(
            self,
            X,
            y_time,
            y_rangf,
            first_obs,
            last_obs,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            standardize_response=False,
            verbose=True
    ):
        """
        Predict from the pre-trained DTSR model.
        Predictions are averaged over ``self.n_samples_eval`` samples from the predictive posterior for each regression target.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the DTSR ``form_str`` provided at initialization.

        :param y_time: ``pandas`` ``Series`` or 1D ``numpy`` array; timestamps for the regression targets, grouped by series.
        :param y_rangf: ``pandas`` ``Series`` or 1D ``numpy`` array; random grouping factor values (if applicable).
            Can be of type ``str`` or ``int``.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param first_obs: list of ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param last_obs: list of ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with the current regression target.
            Sort order and number of observations must be identical to that of ``y_time``.
        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: 1D ``numpy`` array; mean network predictions for regression targets (same length and sort order as ``y_time``).
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            sys.stderr.write('Computing predictions...\n')

        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])
        time_y = np.array(y_time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_y: time_y,
                    self.gf_y: gf_y
                }


                if not np.isfinite(self.eval_minibatch_size):
                    preds = self.run_predict_op(
                        fd,
                        standardize_response=standardize_response,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    preds = np.zeros((len(y_time),))
                    n_eval_minibatch = math.ceil(len(y_time) / self.eval_minibatch_size)
                    for i in range(0, len(y_time), self.eval_minibatch_size):
                        if verbose:
                            sys.stderr.write('\rMinibatch %d/%d' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                            sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y
                        }
                        preds[i:i + self.eval_minibatch_size] = self.run_predict_op(
                            fd_minibatch,
                            standardize_response=standardize_response,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )

                if verbose:
                    sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return preds

    def error_theoretical_quantiles(
            self,
            n_errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.n_errors: n_errors
                }
                err_q = self.sess.run(self.err_dist_summary_theoretical_quantiles, feed_dict=fd)

                return err_q

    def error_theoretical_cdf(
            self,
            errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd = {
                    self.errors: errors
                }
                err_cdf = self.sess.run(self.err_dist_summary_theoretical_cdf, feed_dict=fd)

                return err_cdf

    def error_ks_test(
            self,
            errors
    ):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                err_cdf = self.error_theoretical_cdf(errors)

                D, p_value = scipy.stats.kstest(errors, lambda x: err_cdf)

                return D, p_value

    def log_lik(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            standardize_response=False,
            verbose=True
    ):
        """
        Compute log-likelihood of data from predictive posterior.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the DTSR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            sys.stderr.write('Computing likelihoods...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if not np.isfinite(self.eval_minibatch_size):
                    fd = {
                        self.X: X_2d,
                        self.time_X: time_X_2d,
                        self.time_X_mask: time_X_mask,
                        self.time_y: time_y,
                        self.gf_y: gf_y,
                        self.y: y_dv
                    }
                    log_lik = self.run_loglik_op(
                        fd,
                        standardize_response=standardize_response,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    log_lik = np.zeros((len(time_y),))
                    n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                    for i in range(0, len(time_y), self.eval_minibatch_size):
                        if verbose:
                            sys.stderr.write('\rMinibatch %d/%d' %((i/self.eval_minibatch_size)+1, n_eval_minibatch))
                            sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.eval_minibatch_size],
                            self.time_X: time_X_2d[i:i + self.eval_minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.eval_minibatch_size],
                            self.time_y: time_y[i:i + self.eval_minibatch_size],
                            self.gf_y: gf_y[i:i + self.eval_minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.eval_minibatch_size]
                        }
                        log_lik[i:i+self.eval_minibatch_size] = self.run_loglik_op(
                            fd_minibatch,
                            standardize_response=standardize_response,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )

                if verbose:
                    sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return log_lik

    def loss(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            n_samples=None,
            algorithm='MAP',
            verbose=True
    ):
        """
        Compute compute the loss over a dataset using the model's optimization objective.
        Useful for checking divergence between optimization objective and other evaluation metrics.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the DTSR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        if verbose:
            sys.stderr.write('Computing loss using objective function...\n')

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        y_dv = np.array(y[self.dv], dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if not np.isfinite(self.minibatch_size):
                    fd = {
                        self.X: X_2d,
                        self.time_X: time_X_2d,
                        self.time_X_mask: time_X_mask,
                        self.time_y: time_y,
                        self.gf_y: gf_y,
                        self.y: y_dv
                    }
                    loss = self.run_loss_op(
                        fd,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                else:
                    n_minibatch = math.ceil(len(y) / self.minibatch_size)
                    loss = np.zeros((n_minibatch,))
                    for i in range(0, n_minibatch):
                        if verbose:
                            sys.stderr.write('\rMinibatch %d/%d' %(i+1, n_minibatch))
                            sys.stderr.flush()
                        fd_minibatch = {
                            self.X: X_2d[i:i + self.minibatch_size],
                            self.time_X: time_X_2d[i:i + self.minibatch_size],
                            self.time_X_mask: time_X_mask[i:i + self.minibatch_size],
                            self.time_y: time_y[i:i + self.minibatch_size],
                            self.gf_y: gf_y[i:i + self.minibatch_size] if len(gf_y) > 0 else gf_y,
                            self.y: y_dv[i:i+self.minibatch_size]
                        }
                        loss[i] = self.run_loss_op(
                            fd_minibatch,
                            n_samples=n_samples,
                            algorithm=algorithm,
                            verbose=verbose
                        )
                    loss = loss.mean()

                if verbose:
                    sys.stderr.write('\n\n')

                self.set_predict_mode(False)

                return loss

    def convolve_inputs(
            self,
            X,
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            scaled=False,
            n_samples=None,
            algorithm='MAP',
            standardize_response=False,
            verbose=True
    ):
        """
        Convolve input data using the fitted DTSR model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the DTSR ``form_str`` provided at initialization.

        :param y: ``pandas`` table; the dependent variable. Must contain the following columns:

            * ``time``: Timestamp associated with each observation in ``y``
            * ``first_obs``:  Index in the design matrix `X` of the first observation in the time series associated with each entry in ``y``
            * ``last_obs``:  Index in the design matrix `X` of the immediately preceding observation in the time series associated with each entry in ``y``
            * A column with the same name as the DV specified in ``form_str``
            * A column for each random grouping factor in the model specified in ``form_str``.

        :param X_response_aligned_predictor_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param X_response_aligned_predictors: ``pandas`` table; Response-aligned predictors if applicable, ``None`` otherwise.
        :param X_2d_predictor_names: ``list`` or ``None``; List of column names 2D predictors (predictors whose value depends on properties of the most recent impulse) if applicable, ``None`` otherwise.
        :param X_2d_predictors: ``pandas`` table; 2D predictors if applicable, ``None`` otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting predictions, one of [``MAP``, ``sampling``].
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless ``scaled==True`` and model was fitted using ``standardize_response==True``.
        :param verbose: ``bool``; Report progress and metrics to standard error.
        :return: ``numpy`` array of shape [len(X)], log likelihood of each data point.
        """

        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        if self.pc:
            impulse_names = self.src_impulse_names
        else:
            impulse_names  = self.impulse_names

        y_rangf = y[self.rangf]
        for i in range(len(self.rangf)):
            c = self.rangf[i]
            y_rangf[c] = pd.Series(y_rangf[c].astype(str)).map(self.rangf_map[i])

        first_obs, last_obs = get_first_last_obs_lists(y)
        time_y = np.array(y.time, dtype=self.FLOAT_NP)
        gf_y = np.array(y_rangf, dtype=self.INT_NP)

        X_2d, time_X_2d, time_X_mask = build_DTSR_impulses(
            X,
            first_obs,
            last_obs,
            impulse_names,
            time_y=time_y,
            history_length=self.history_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=X_response_aligned_predictors,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=self.int_type,
            float_type=self.float_type,
        )

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                fd = {
                    self.time_y: time_y,
                    self.gf_y: gf_y,
                    self.X: X_2d,
                    self.time_X: time_X_2d,
                    self.time_X_mask: time_X_mask
                }

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X]
                }

                X_conv = []
                n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                for i in range(0, len(y), self.eval_minibatch_size):
                    if verbose:
                        sys.stderr.write('\rMinibatch %d/%d' % ((i / self.eval_minibatch_size) + 1, n_eval_minibatch))
                        sys.stderr.flush()
                    fd_minibatch[self.time_y] = time_y[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.gf_y] = gf_y[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.X] = X_2d[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.time_X] = time_X_2d[i:i + self.eval_minibatch_size]
                    fd_minibatch[self.time_X_mask] = time_X_mask[i:i + self.eval_minibatch_size]
                    X_conv_cur = self.run_conv_op(
                        fd_minibatch,
                        scaled=scaled,
                        standardize_response=standardize_response,
                        n_samples=n_samples,
                        algorithm=algorithm,
                        verbose=verbose
                    )
                    X_conv.append(X_conv_cur)
                names = []
                for x in self.terminal_names:
                    if self.node_table[x].p.irfID is None:
                        names.append(sn(''.join(x.split('-')[:-1])))
                    else:
                        names.append(sn(x))
                X_conv = np.concatenate(X_conv, axis=0)
                out = pd.DataFrame(X_conv, columns=names, dtype=self.FLOAT_NP)

                self.set_predict_mode(False)

                convolution_summary = ''
                corr_conv = out.corr().to_string()
                convolution_summary += '=' * 50 + '\n'
                convolution_summary += 'Correlation matrix of convolved predictors:\n\n'
                convolution_summary += corr_conv + '\n\n'

                select = np.where(np.all(np.isclose(time_X_2d[:,-1], time_y[..., None]), axis=-1))[0]

                X_input = X_2d[:,-1,:][select]

                extra_cols = []
                for c in y.columns:
                    if c not in out:
                        extra_cols.append(c)
                out = pd.concat([y[extra_cols].reset_index(), out], axis=1)

                if X_input.shape[0] > 0:
                    out_plus = out
                    for i in range(len(self.impulse_names)):
                        c = self.impulse_names[i]
                        if c not in out_plus:
                            out_plus[c] = X_2d[:,-1,i]
                    corr_conv = out_plus.iloc[select].corr().to_string()
                    convolution_summary += '-' * 50 + '\n'
                    convolution_summary += 'Full correlation matrix of input and convolved predictors:\n'
                    convolution_summary += 'Based on %d simultaneously sampled impulse/response pairs (out of %d total data points)\n\n' %(select.shape[0], y.shape[0])
                    convolution_summary += corr_conv + '\n\n'
                    convolution_summary += '=' * 50 + '\n'

                return out, convolution_summary

    def make_plots(
            self,
            standardize_response=False,
            summed=False,
            irf_name_map=None,
            irf_ids=None,
            sort_names=True,
            plot_unscaled=True,
            plot_composite=False,
            prop_cycle_length=None,
            prop_cycle_ix=None,
            plot_dirac=False,
            plot_rangf=False,
            plot_n_time_units=2.5,
            plot_n_time_points=1000,
            plot_x_inches=6.,
            plot_y_inches=4.,
            ylim=None,
            cmap=None,
            dpi=300,
            mc=False,
            level=95,
            n_samples=None,
            prefix=None,
            legend=True,
            xlab=None,
            ylab=None,
            use_line_markers=False,
            transparent_background=False,
            keep_plot_history=False
    ):
        """
        Generate plots of current state of deconvolution.
        DTSR distinguishes plots based on two orthogonal criteria: "atomic" vs. "composite" and "scaled" vs. "unscaled".
        The "atomic"/"composite" distinction is only relevant in models containing composed IRF.
        In such models, "atomic" plots represent the shape of the IRF irrespective of any other IRF with which they are composed, while "composite" plots represent the shape of the IRF composed with any upstream IRF in the model.
        In models without composed IRF, only "atomic" plots are generated.
        The "scaled"/"unscaled" distinction concerns whether the impulse coefficients are represented in the plot ("scaled") or not ("unscaled").
        Only pre-terminal IRF (i.e. the final IRF in all IRF compositions) have coefficients, so only preterminal IRF are represented in "scaled" plots, while "unscaled" plots also contain all intermediate IRF.
        In addition, Bayesian DTSR implementations also support MC sampling of credible intervals around all curves.
        Outputs are saved to the model's output directory as PNG files with names indicating which plot type is represented.
        All plot types relevant to a given model are generated.

        :param irf_name_map: ``dict`` or ``None``; a dictionary mapping IRF tree nodes to display names.
            If ``None``, IRF tree node string ID's will be used.
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param summed: ``bool``; whether to plot individual IRFs or their sum.
        :param irf_ids: ``list`` or ``None``; list of irf ID's to plot. If ``None``, all IRF's are plotted.
        :param sort_names: ``bool``; alphabetically sort IRF names.
        :param plot_unscaled: ``bool``; plot unscaled IRFs.
        :param plot_composite: ``bool``; plot any composite IRFs. If ``False``, only plots terminal IRFs.
        :param prop_cycle_length: ``int`` or ``None``; Length of plotting properties cycle (defines step size in the color map). If ``None``, inferred from **irf_names**.
        :param prop_cycle_ix: ``list`` of ``int``, or ``None``; Integer indices to use in the properties cycle for each entry in **irf_names**. If ``None``, indices are automatically assigned.
        :param plot_dirac: ``bool``; include any linear Dirac delta IRF's (stick functions at t=0) in plot.
        :param plot_rangf: ``bool``; plot all (marginal) random effects.
        :param plot_n_time_units: ``float``; number if time units to use for plotting.
        :param plot_n_time_points: ``int``; number of points to use for plotting.
        :param plot_x_inches: ``int``; width of plot in inches.
        :param plot_y_inches: ``int``; height of plot in inches.
        :param ylim: 2-element ``tuple`` or ``list``; (lower_bound, upper_bound) to use for y axis. If ``None``, automatically inferred.
        :param cmap: ``str``; name of MatPlotLib cmap specification to use for plotting (determines the color of lines in the plot).
        :param dpi: ``int``; dots per inch.
        :param mc: ``bool``; compute and plot Monte Carlo credible intervals (only supported for DTSRBayes).
        :param level: ``float``; significance level for credible intervals, ignored unless **mc** is ``True``.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param prefix: ``str`` or ``None``; prefix appended to output filenames. If ``None``, no prefix added.
        :param legend: ``bool``; generate a plot legend.
        :param xlab: ``str`` or ``None``; x-axis label. If ``None``, no label.
        :param ylab: ``str`` or ``None``; y-axis label. If ``None``, no label.
        :param transparent_background: ``bool``; use a transparent background. If ``False``, uses a white background.
        :param keep_plot_history: ``bool``; keep the history of all plots by adding a suffix with the iteration number. Can help visualize learning but can also consume a lot of disk space. If ``False``, always overwrite with most recent plot.
        :return: ``None``
        """

        assert not mc or type(self).__name__ == 'DTSRBayes', 'Monte Carlo estimation of credible intervals (mc=True) is only supported for DTSRBayes models.'

        if mc and not hasattr(self, 'ci_curve'):
            sys.stderr.write('Credible intervals are not supported for instances of %s. Re-run ``make_plots`` with ``mc=False``.\n' % type(self))
            mc = False

        if len(self.terminal_names) == 0:
            return

        if plot_dirac:
            dirac = 'dirac'
        else:
            dirac = 'nodirac'

        if prefix is None:
            prefix = ''
        if prefix != '':
            prefix += '_'

        if summed:
            alpha = 100 - float(level)

        rangf_keys = [None]
        rangf_vals = [self.gf_defaults[0]]
        if plot_rangf:
            for i in range(len(self.rangf)):
                if self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                    for k in self.rangf_map[i].keys():
                        rangf_keys.append(str(k))
                        rangf_vals.append(np.concatenate([self.gf_defaults[0, :i], [self.rangf_map[i][k]], self.gf_defaults[0, i+1:]], axis=0))
        rangf_vals = np.stack(rangf_vals, axis=0)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if self.asymmetric_error:
                    lb = self.sess.run(self.err_dist_lb)
                    ub = self.sess.run(self.err_dist_ub)
                    n_time_units = ub - lb
                    fd = {
                        self.support_start: lb,
                        self.n_time_units: n_time_units,
                        self.n_time_points: plot_n_time_points
                    }
                    plot_x = self.sess.run(self.support, feed_dict=fd)
                    if mc:
                        plot_y, lq, uq, _ = self.ci_curve(
                            self.err_dist_plot,
                            level=level,
                            n_samples=n_samples,
                            support_start=lb,
                            n_time_units=n_time_units,
                            n_time_points=plot_n_time_points,
                        )
                        plot_y = plot_y[0, ..., None]
                        lq = lq[0, ..., None]
                        uq = uq[0, ..., None]
                        plot_name = 'mc_error_distribution_%s.png' % self.global_step.eval(sess=self.sess) \
                            if self.keep_plot_history else 'mc_error_distribution.png'
                    else:
                        plot_y = self.sess.run(self.err_dist_plot_summary, feed_dict=fd)[0]
                        lq = None
                        uq = None
                        plot_name = 'error_distribution_%s.png' % self.global_step.eval(sess=self.sess) \
                            if self.keep_plot_history else 'error_distribution.png'
                        
                    plot_irf(
                        plot_x,
                        plot_y,
                        ['Error Distribution'],
                        lq=lq,
                        uq=uq,
                        dir=self.outdir,
                        filename=prefix + plot_name,
                        legend=False,
                    )

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: plot_n_time_units,
                    self.n_time_points: plot_n_time_points,
                    self.max_tdelta_batch: plot_n_time_units,
                    self.gf_y: rangf_vals
                }

                plot_x = self.sess.run(self.support, fd)

                switches = [['atomic'], ['scaled']]
                if plot_composite:
                    switches[0].append('composite')
                if plot_unscaled:
                    switches[1].append('unscaled')

                for a in switches[0]:
                    if self.t.has_composed_irf() or a == 'atomic':
                        for b in switches[1]:
                            if summed:
                                plot_name = 'irf_%s_%s_summed_%d.png' %(a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'irf_%s_%s_summed.png' %(a, b)
                            else:
                                plot_name = 'irf_%s_%s_%d.png' %(a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'irf_%s_%s.png' %(a, b)
                            names = self.plots[a][b][dirac]['names']
                            if irf_ids is not None and len(irf_ids) > 0:
                                new_names = []
                                for i, name in enumerate(names):
                                    for ID in irf_ids:
                                        if ID==name or re.match(ID if ID.endswith('$') else ID + '$', name) is not None:
                                            new_names.append(name)
                                names = new_names
                            if len(names) > 0:
                                if mc:
                                    if summed:
                                        samples = []
                                    else:
                                        plot_y = []
                                        lq = []
                                        uq = []
                                    for name in names:
                                        mean_cur, lq_cur, uq_cur, samples_cur = self.ci_curve(
                                            self.irf_mc[name][a][b],
                                            rangf=rangf_vals,
                                            level=level,
                                            n_samples=n_samples,
                                            n_time_units=plot_n_time_units,
                                            n_time_points=plot_n_time_points,
                                        )

                                        if summed:
                                            samples.append(samples_cur)
                                        else:
                                            plot_y.append(mean_cur)
                                            lq.append(lq_cur)
                                            uq.append(uq_cur)

                                    if summed:
                                        samples = np.stack(samples, axis=3)
                                        samples = samples.sum(axis=3, keepdims=True)
                                        lq = np.percentile(samples, alpha / 2, axis=2)
                                        uq = np.percentile(samples, 100 - (alpha / 2), axis=2)
                                        plot_y = samples.mean(axis=2)
                                    else:
                                        lq = np.stack(lq, axis=2)
                                        uq = np.stack(uq, axis=2)
                                        plot_y = np.stack(plot_y, axis=2)

                                    if self.standardize_response and not standardize_response:
                                        plot_y = plot_y * self.y_train_sd
                                        lq = lq * self.y_train_sd
                                        uq = uq * self.y_train_sd

                                    plot_name = 'mc_' + plot_name

                                else:
                                    plot_y = []
                                    for i in range(len(self.plots[a][b][dirac]['plot'])):
                                        if self.plots[a][b][dirac]['names'][i] in names:
                                            plot_y_cur = self.sess.run(self.plots[a][b][dirac]['plot'][i], feed_dict=fd)
                                            if len(plot_y_cur) == 1 and len(rangf_vals) > 1:
                                                plot_y_cur = np.repeat(plot_y_cur, len(rangf_vals), axis=0)
                                            plot_y.append(plot_y_cur)
                                    lq = None
                                    uq = None
                                    plot_y = np.concatenate(plot_y, axis=2)
                                    if summed:
                                        plot_y = plot_y.sum(axis=2, keepdims=True)
                                    if self.standardize_response and not standardize_response:
                                        plot_y = plot_y * self.y_train_sd

                                if summed:
                                    names_cur = ['Sum']
                                else:
                                    names_cur = names

                                for g in range(len(rangf_keys)):
                                    if rangf_keys[g]:
                                        filename = prefix + rangf_keys[g] + '_' + plot_name
                                    else:
                                        filename = prefix + plot_name
                                    plot_irf(
                                        plot_x,
                                        plot_y[g],
                                        names_cur,
                                        lq=None if lq is None else lq[g],
                                        uq=None if uq is None else uq[g],
                                        sort_names=sort_names,
                                        prop_cycle_length=prop_cycle_length,
                                        prop_cycle_ix=prop_cycle_ix,
                                        dir=self.outdir,
                                        filename=filename,
                                        irf_name_map=irf_name_map,
                                        plot_x_inches=plot_x_inches,
                                        plot_y_inches=plot_y_inches,
                                        ylim=ylim,
                                        cmap=cmap,
                                        dpi=dpi,
                                        legend=legend,
                                        xlab=xlab,
                                        ylab=ylab,
                                        use_line_markers=use_line_markers,
                                        transparent_background=transparent_background
                                    )

                if self.pc:
                    for a in switches[0]:
                        if self.t.has_composed_irf() or a == 'atomic':
                            for b in switches[1]:
                                if b == 'scaled':
                                    if summed:
                                        plot_name = 'src_irf_%s_%s_summed_%d.png' % (a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'src_irf_%s_%s_summed.png' % (a, b)
                                    else:
                                        plot_name = 'src_irf_%s_%s_%d.png' % (a, b, self.global_step.eval(session=self.sess)) if keep_plot_history else 'src_irf_%s_%s.png' % (a, b)
                                    names = self.src_plot_tensors[a][b][dirac]['names']
                                    if irf_ids is not None and len(irf_ids) > 0:
                                        new_names = []
                                        for i, name in enumerate(names):
                                            for ID in irf_ids:
                                                if ID == name or re.match(ID if ID.endswith('$') else ID + '$',
                                                                          name) is not None:
                                                    new_names.append(name)
                                        names = new_names
                                    if len(names) > 0:
                                        if mc:
                                            if summed:
                                                samples = []
                                            else:
                                                plot_y = []
                                                lq = []
                                                uq = []
                                            for name in names:
                                                mean_cur, lq_cur, uq_cur, samples_cur = self.ci_curve(
                                                    self.src_irf_mc[name][a][b],
                                                    level=level,
                                                    n_samples=n_samples,
                                                    n_time_units=plot_n_time_units,
                                                    n_time_points=plot_n_time_points,
                                                )

                                                if summed:
                                                    samples.append(samples_cur)
                                                else:
                                                    plot_y.append(mean_cur)
                                                    lq.append(lq_cur)
                                                    uq.append(uq_cur)

                                            if summed:
                                                samples = np.stack(samples, axis=2)
                                                samples = samples.sum(axis=2, keepdims=True)
                                                lq = np.percentile(samples, alpha / 2, axis=1)
                                                uq = np.percentile(samples, 100 - (alpha / 2), axis=1)
                                                plot_y = samples.mean(axis=1)
                                            else:
                                                lq = np.stack(lq, axis=1)
                                                uq = np.stack(uq, axis=1)
                                                plot_y = np.stack(plot_y, axis=1)

                                            if self.standardize_response and not standardize_response:
                                                plot_y = plot_y * self.y_train_sd
                                                lq = lq * self.y_train_sd
                                                uq = uq * self.y_train_sd

                                            plot_name = 'mc_' + plot_name

                                        else:
                                            plot_y = [self.sess.run(self.src_plot_tensors[a][b][dirac]['plot'][i], feed_dict=fd) for i in range(len(self.src_plot_tensors[a][b][dirac]['plot'])) if self.src_plot_tensors[a][b][dirac]['names'][i] in names]
                                            lq = None
                                            uq = None
                                            plot_y = np.concatenate(plot_y, axis=1)
                                            if summed:
                                                plot_y = plot_y.sum(axis=1, keepdims=True)
                                            if self.standardize_response and not standardize_response:
                                                plot_y = plot_y * self.y_train_sd

                                        if summed:
                                            names_cur = ['Sum']
                                        else:
                                            names_cur = names

                                        plot_irf(
                                            plot_x,
                                            plot_y,
                                            names_cur,
                                            lq=lq,
                                            uq=uq,
                                            sort_names=sort_names,
                                            prop_cycle_length=prop_cycle_length,
                                            prop_cycle_ix=prop_cycle_ix,
                                            dir=self.outdir,
                                            filename=prefix + plot_name,
                                            irf_name_map=irf_name_map,
                                            plot_x_inches=plot_x_inches,
                                            plot_y_inches=plot_y_inches,
                                            ylim=ylim,
                                            cmap=cmap,
                                            dpi=dpi,
                                            legend=legend,
                                            xlab=xlab,
                                            ylab=ylab,
                                            use_line_markers=use_line_markers,
                                            transparent_background=transparent_background
                                        )

                self.set_predict_mode(False)

    def irf_rmsd(
            self,
            gold_irf_lambda,
            standardize_response=False,
            summed=False,
            n_time_units=None,
            n_time_points=1000,
            n_samples=None,
            algorithm='MAP'
    ):
        """
        Compute root mean squared deviation (RMSD) of fitted IRFs from gold.

        :param gold_irf_lambda: callable; vectorized numpy callable representing continuous IRFs. Generates response values for an array of inputs. Input has shape ``[n_time_points, 1]`` and output has shape ``[n_time_points, len(self.terminals)]``.
        :param standardize_response: ``bool``; Whether to report response using standard units. Ignored unless model was fitted using ``standardize_response==True``.
        :param summed: ``bool``; whether to compare individual IRFs or their sum.
        :param n_time_units: ``float``; number if time units to use. If ``None``, maximum temporal offset seen in training will be used.
        :param n_time_points: ``float``; number of points to use.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param algorithm: ``str``; algorithm to use for extracting IRFs from DTSRBayes, one of [``MAP``, ``sampling``]. Ignored for MLE models.
        :return: ``float``; RMSD of fitted IRFs from gold
        """

        assert algorithm.lower() in ['map', 'sampling'], 'Unrecognized algorithm for computing IRFs in RMSD comparison: "%s"' % algorithm

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not n_time_units:
                    n_time_units = self.t_delta_limit

                support = np.linspace(0., n_time_units, n_time_points+1)
                gold = gold_irf_lambda(support)
                if summed:
                    gold = gold.sum(axis=1)

                plots = self.plots['atomic']['scaled']['nodirac']

                names = plots['names']

                fd = {
                    self.support_start: 0.,
                    self.n_time_units: n_time_units,
                    self.n_time_points: n_time_points,
                    self.max_tdelta_batch: n_time_units
                }

                if type(self).__name__ == 'DTSRBayes' and algorithm.lower() == 'sampling':
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    fd[self.time_y] = np.ones((1,)) * n_time_units
                    fd[self.time_X] = np.zeros((1, self.history_length))

                    preds = []

                    for name in names:
                        posterior = self.irf_mc[name]['atomic']['scaled']

                        samples = [self.sess.run(posterior, feed_dict=fd) for _ in range(n_samples)]
                        samples = np.concatenate(samples, axis=1)
                        preds.append(samples)

                    preds = np.stack(preds, axis=2)
                    if summed:
                        preds = preds.sum(axis=2)
                    gold = np.expand_dims(gold, 1)

                else:
                    preds = [self.sess.run(plots['plot'][i], feed_dict=fd) for i in range(len(plots['plot'])) if plots['names'][i] in names]
                    preds = np.concatenate(preds, axis=1)
                    if summed:
                        preds = preds.sum(axis=1)

                if self.standardize_response and not standardize_response:
                    preds = preds * self.y_train_sd

                rmsd = np.sqrt(((gold - preds) ** 2).mean())

                return rmsd

    def plot_eigenvectors(self):
        """
        Save heatmap representation of training data eigenvector matrix to the model's output directory.
        Will throw an error unless ``self.pc == True``.

        :return: ``None``
        """
        plot_heatmap(self.eigenvec, self.src_impulse_names_norate, self.impulse_names_norate, dir=self.outdir)

    def parameter_table(self, fixed=True, level=95, n_samples=None):
        """
        Generate a pandas table of parameter names and values.

        :param fixed: ``bool``; Return a table of fixed parameters (otherwise returns a table of random parameters).
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :return: ``pandas`` ``DataFrame``; The parameter table.
        """

        assert fixed or len(self.rangf) > 0, 'Attempted to generate a random effects parameter table in a fixed-effects-only model'

        if n_samples is None and getattr(self, 'n_samples_eval', None) is not None:
            n_samples = self.n_samples_eval

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if fixed:
                    keys = self.parameter_table_fixed_keys
                    values = self._extract_parameter_values(
                        fixed=True,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({'Parameter': keys})

                else:
                    keys = self.parameter_table_random_keys
                    rangf = self.parameter_table_random_rangf
                    rangf_levels = self.parameter_table_random_rangf_levels
                    values = self._extract_parameter_values(
                        fixed=False,
                        level=level,
                        n_samples=n_samples
                    )

                    out = pd.DataFrame({'Parameter': keys, 'Group': rangf, 'Level': rangf_levels}, columns=['Parameter', 'Group', 'Level'])

                columns = self.parameter_table_columns
                out = pd.concat([out, pd.DataFrame(values, columns=columns)], axis=1)

                self.set_predict_mode(False)

                return out

    def save_parameter_table(self, random=True, level=95, n_samples=None, outfile=None):
        """
        Save space-delimited parameter table to the model's output directory.

        :param random: Include random parameters.
        :param level: ``float``; significance level for credible intervals if model is Bayesian, ignored otherwise.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param outfile: ``str``; Path to output file. If ``None``, use model defaults.
        :return: ``None``
        """

        parameter_table = self.parameter_table(
            fixed=True,
            level=level,
            n_samples=n_samples
        )
        if random and len(self.rangf) > 0:
            parameter_table = pd.concat(
                [
                    parameter_table,
                    self.parameter_table(
                        fixed=False,
                        level=level,
                        n_samples=n_samples
                    )
                ],
            axis=0
            )

        if outfile:
            outname = self.outdir + '/dtsr_parameters.csv'
        else:
            outname = outfile

        parameter_table.to_csv(outname, index=False)

    def save_integral_table(self, random=True, level=95, n_samples=None, integral_n_time_units=None, outfile=None):
        """
        Save space-delimited table of IRF integrals (effect sizes) to the model's output directory

        :param random: ``bool``; whether to compute IRF integrals for random effects estimates
        :param level: ``float``; significance level for credible intervals if Bayesian, otherwise ignored.
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw if Bayesian, ignored otherwise. If ``None``, use model defaults.
        :param integral_n_time_units: ``float``; number if time units over which to take the integral.
        :param outfile: ``str``; Path to output file. If ``None``, use model defaults.
        :return: ``str``; the IRF integrals report
        """

        if integral_n_time_units is None:
            integral_n_time_units = self.t_delta_limit

        irf_integrals = self.irf_integrals(
            random=random,
            level=level,
            n_samples=n_samples,
            n_time_units=integral_n_time_units,
            n_time_points=1000
        )

        if outfile:
            outname = self.outdir + '/dtsr_irf_integrals.csv'
        else:
            outname = outfile

        irf_integrals.to_csv(outname, index=False)

