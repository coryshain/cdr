import os
import pandas as pd
import scipy.stats
import time as pytime

from .formula import *
from .kwargs import CDR_INITIALIZATION_KWARGS
from .util import *
from .data import build_CDR_impulses, corr_cdr, get_first_last_obs_lists
from .base import Model
from .interpolate_spline import interpolate_spline
from .plot import plot_heatmap

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


def unnormalized_gamma(alpha, beta):
    return lambda x: x ** (alpha - 1) * tf.exp(-beta * x)


def unnormalized_gaussian(mu, sigma2):
    return lambda x: tf.exp(-(x - mu) ** 2 / sigma2)


def exponential_irf_lambdas(params, integral_ub=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            beta = params[:, 0:1]

            def pdf(x, beta=beta):
                return beta * tf.exp(-beta * x)

            def cdf(x, beta=beta):
                return 1 - tf.exp(-beta * x)

            if integral_ub is None:
                def irf(x, pdf=pdf):
                    return pdf(x)

                def irf_proportion_in_bounds(x, cdf=cdf):
                    return cdf(x)
            else:
                norm_const = cdf(integral_ub)

                def irf(x, pdf=pdf, norm_const=norm_const):
                    return pdf(x) / norm_const

                def irf_proportion_in_bounds(x, cdf=cdf, norm_const=norm_const):
                    return cdf(x) / norm_const

            return irf, irf_proportion_in_bounds


def gamma_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                def irf(x, pdf=pdf, epsilon=epsilon):
                    return pdf(x + epsilon)

                def irf_proportion_in_bounds(x, cdf=cdf):
                    return cdf(x)

            else:
                norm_const = cdf(integral_ub)

                def irf(x, pdf=pdf, norm_const=norm_const, epsilon=epsilon):
                    return pdf(x + epsilon) / norm_const

                def irf_proportion_in_bounds(x, cdf=cdf, norm_const=norm_const, epsilon=epsilon):
                    return cdf(x + epsilon) / norm_const

            return irf, irf_proportion_in_bounds


def shifted_gamma_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
            cdf_0 = cdf(-delta + epsilon)

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub - delta)

            norm_const = ub - cdf_0

            def irf(x, pdf=pdf, delta=delta, norm_const=norm_const, epsilon=epsilon):
                return pdf(x - delta + epsilon) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, cdf_0=cdf_0, delta=delta, norm_const=norm_const, epsilon=epsilon):
                return (cdf(x - delta + epsilon) - cdf_0) / norm_const

            return irf, irf_proportion_in_bounds


def normal_irf_lambdas(params, integral_ub=None, session=None):
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
            cdf_0 = cdf(0.)

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub)

            norm_const = ub - cdf_0

            def irf(x, pdf=pdf, norm_const=norm_const):
                return pdf(x) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, cdf_0=cdf_0, norm_const=norm_const):
                return (cdf(x) - cdf_0) / norm_const
                
            return irf, irf_proportion_in_bounds


def skew_normal_irf_lambdas(params, integral_ub, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            mu = params[:, 0:1]
            sigma = params[:, 1:2]
            alpha = params[:, 2:3]
            
            stdnorm = tf.contrib.distributions.Normal(loc=0., scale=1.)
            stdnorm_pdf = stdnorm.prob
            stdnorm_cdf = stdnorm.cdf
            
            def irf_base(x,  mu=mu, sigma=sigma, alpha=alpha, pdf=stdnorm_pdf, cdf=stdnorm_cdf):
                return (stdnorm_pdf((x - mu) / sigma) * stdnorm_cdf(alpha * (x - mu) / sigma))
            
            cdf = empirical_integral(irf_base, session=session)
            norm_const = cdf(integral_ub)
            
            def irf(x, irf_base=irf_base, norm_const=norm_const):
                return irf_base(x) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, norm_const=norm_const):
                return cdf(x) / norm_const

            return irf, irf_proportion_in_bounds


def emg_irf_lambdas(params, integral_ub=None, session=None):
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

            cdf_0 = cdf(0.)

            norm_const = ub - cdf(0)

            def irf(x, L=L, mu=mu, sigma=sigma, norm_const=norm_const):
                return (L / 2 * tf.exp(0.5 * L * (2. * mu + L * sigma ** 2. - 2. * x)) *
                       tf.erfc((mu + L * sigma ** 2 - x) / (tf.sqrt(2.) * sigma))) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, norm_const=norm_const, cdf_0=cdf_0):
                return (cdf(x) - cdf_0) / norm_const

            return irf, irf_proportion_in_bounds


def beta_prime_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps):
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

            cdf_0 = cdf(epsilon)

            norm_const = ub - cdf_0

            def irf(x, alpha=alpha, beta=beta, norm_const=norm_const, epsilon=epsilon):
                return ((x + epsilon) ** (alpha - 1.) * (1. + (x + epsilon)) ** (-alpha - beta)) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, cdf_0=cdf_0, norm_const=norm_const):
                return (cdf(x) - cdf_0) / norm_const

            return irf, irf_proportion_in_bounds


def shifted_beta_prime_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps):
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
                ub = cdf(integral_ub - delta)

            cdf_0 = cdf(-delta + epsilon)

            norm_const = ub - cdf_0

            def irf(x, alpha=alpha, beta=beta, delta=delta, norm_const=norm_const, epsilon=epsilon):
                return ((x - delta + epsilon) ** (alpha - 1) * (1 + (x - delta + epsilon)) ** (-alpha - beta)) / norm_const

            def irf_proportion_in_bounds(x, cdf=cdf, cdf_0=cdf_0, delta=delta, norm_const=norm_const, epsilon=epsilon):
                return (cdf(x - delta + epsilon) - cdf_0) / norm_const

            return irf, irf_proportion_in_bounds


def double_gamma_1_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                norm_const = 1 - c
            else:
                norm_const = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / norm_const

            def irf_proportion_in_bounds(x, cdf_main=cdf_main, cdf_undershoot=cdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (cdf_main(x) - cdf_undershoot(x)) / norm_const

            return irf, irf_proportion_in_bounds


def double_gamma_2_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                norm_const = 1 - c
            else:
                norm_const = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / norm_const

            def irf_proportion_in_bounds(x, cdf_main=cdf_main, cdf_undershoot=cdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (cdf_main(x) - cdf_undershoot(x)) / norm_const

            return irf, irf_proportion_in_bounds


def double_gamma_3_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                norm_const = 1 - c
            else:
                norm_const = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / norm_const

            def irf_proportion_in_bounds(x, cdf_main=cdf_main, cdf_undershoot=cdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (cdf_main(x) - cdf_undershoot(x)) / norm_const

            return irf, irf_proportion_in_bounds


def double_gamma_4_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                norm_const = 1 - c
            else:
                norm_const = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / norm_const

            def irf_proportion_in_bounds(x, cdf_main=cdf_main, cdf_undershoot=cdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (cdf_main(x) - cdf_undershoot(x)) / norm_const

            return irf, irf_proportion_in_bounds


def double_gamma_5_irf_lambdas(params, integral_ub=None, session=None, epsilon=4 * np.finfo('float32').eps, validate_irf_args=False):
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
                norm_const = 1 - c
            else:
                norm_const = cdf_main(integral_ub) - c * cdf_undershoot(integral_ub)

            def irf(x, pdf_main=pdf_main, pdf_undershoot=pdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (pdf_main(x + epsilon) - c * pdf_undershoot(x + epsilon)) / norm_const

            def irf_proportion_in_bounds(x, cdf_main=cdf_main, cdf_undershoot=cdf_undershoot, norm_const=norm_const, epsilon=epsilon):
                return (cdf_main(x) - cdf_undershoot(x)) / norm_const

            return irf, irf_proportion_in_bounds


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


def kernel_smooth(c, v, b, kernel='gaussian', epsilon=4 * np.finfo('float32').eps, session=None):
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

                _r = _x - _c

                if kernel.lower() == 'gaussian':
                    _r = tf.contrib.distributions.Normal(
                        loc=_c,
                        scale=_b
                    ).prob(_r)
                else:
                    raise ValueError('Unrecognized smoother kernel "%s".' % kernel)

                num = tf.reduce_sum(_r * _v, axis=-2)
                denom = tf.reduce_sum(_r, axis=-2) + epsilon

                out = num / denom

                return out

    return f


def summed_gaussians(c, v, b, integral_ub=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            _c = c
            _b = b
            _v = v

            _c = tf.expand_dims(_c, axis=-3)
            _b = tf.expand_dims(_b, axis=-3)
            _v = tf.expand_dims(_v, axis=-3)

            # _v *= _b * np.sqrt(2 * np.pi) # Rescale by Gaussian normalization constant

            dist = tf.contrib.distributions.Normal(
                loc=_c,
                scale=_b,
            )
            cdf = dist.cdf

            if integral_ub is None:
                ub = 1.
            else:
                ub = cdf(integral_ub)
                
            cdf_0 = cdf(0.)

            norm_const = tf.reduce_sum((ub - cdf_0) * _v, axis=-2)

            def irf(x, _v=_v, norm_const=norm_const):
                _x = x
                if len(x.shape) == 1:
                    _x = _x[None, :, None]
                elif len(x.shape) == 2:
                    _x = _x[None, ...]
                if len(_x.shape) != 3:
                    raise ValueError('Query to summed gaussians IRF must be exactly rank 3')
                _x = _x[..., None]

                return tf.reduce_sum(dist.prob(_x) * _v, axis=-2) / norm_const
            
            def irf_proportion_in_bounds(x, cdf=cdf, cdf_0=cdf_0, _v=_v, norm_const=norm_const):
                return (tf.reduce_sum((ub - cdf(x)) * _v, axis=-2) - tf.reduce_sum(cdf_0 * _v, axis=-2)) / norm_const

            return irf, irf_proportion_in_bounds


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
            c = params[:, 0:bases]

            # Build values at control points
            v = params[:, bases:2 * bases]

            if method.lower() == 'spline': # Pad appropriately
                # c = tf.cumsum(c, axis=1)
                c = tf.unstack(c, axis=2)
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

                def out(x, f=f, session=session, integral_ub=integral_ub):
                    return f(x) / empirical_integral(integral_ub, session=session), lambda x: empirical_integral(x, session=session)

            else:
                # Build scales at control points
                b = params[:, 2 * bases:]
                if method.lower() == 'kernel_smooth':
                    f = kernel_smooth(c, v, b, epsilon=epsilon, session=session)
                    assert support is not None, 'Argument ``support`` must be provided for kernel smooth IRFS'

                    def out(x, f=f, session=session, integral_ub=integral_ub):
                        return f(x)/ empirical_integral(integral_ub, session=session), lambda x: empirical_integral(x, session=session)

                elif method.lower() == 'summed_gaussians':
                    out = summed_gaussians(c, v, b, integral_ub=integral_ub, session=session)

                else:
                    raise ValueError('Unrecognized non-parametric IRF type: %s' % method)

            return out


class CDR(Model):

    _INITIALIZATION_KWARGS = CDR_INITIALIZATION_KWARGS

    _doc_header = """
        Abstract base class for CDR. Bayesian (:ref:`cdrbayes`) and MLE (:ref:`cdrmle`) implementations inherit from ``CDR``.
        ``CDR`` is not a complete implementation and cannot be instantiated.
        Subclasses of ``CDR`` must implement the following instance methods:
        
            * ``initialize_intercept()``
            * ``initialize_coefficient()``
            * ``initialize_irf_param_unconstrained()``
            * ``initialize_joint_distribution()``
            * ``initialize_objective()``
            * ``run_conv_op()``
            * ``run_loglik_op()``
            * ``run_predict_op()``
            * ``run_train_step()``
            
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

    def __new__(cls, *args, **kwargs):
        if cls is CDR:
            raise TypeError("CDR is an abstract class and may not be instantiated")
        return object.__new__(cls)

    def __init__(self, form_str, X, y, **kwargs):
        super(CDR, self).__init__(
            form_str,
            X,
            y,
            **kwargs
        )

        for kwarg in CDR._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        if self.pc:
            self.src_impulse_names_norate = list(filter(lambda x: x != 'rate', self.form.t.impulse_names(include_interactions=True)))
            _, self.eigenvec, self.eigenval, self.impulse_means, self.impulse_sds = pca(X[self.src_impulse_names_norate])
        else:
            self.eigenvec = self.eigenval = self.impulse_means = self.impulse_sds = None

    def _initialize_metadata(self):
        super(CDR, self)._initialize_metadata()

        assert not self.pc, 'The use of ``pc=True`` is not currently supported.'

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
        self.irf_proportion_in_bounds = {}
        self.irf_proportion_in_bounds_mc = {}
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
            self.src_nonparametric_coef_names = t_src.nonparametric_coef_names()
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

        self.parameter_table_columns = ['Estimate']

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

    def _initialize_cdr_inputs(self):
        with self.sess.as_default():
                self.is_response_aligned = tf.cast(
                    tf.logical_not(
                        tf.cast(self.t_delta[:, -1, :], dtype=tf.bool)
                    ),
                    self.FLOAT_TF
                )

                if self.pc:
                    self.e = tf.constant(self.eigenvec, dtype=self.FLOAT_TF)
                    rate_ix = names2ix('rate', self.src_impulse_names)
                    self.X_rate = tf.gather(self.X, rate_ix, axis=-1)

                # Initialize regularizers
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

                self.oob_regularizer_name = 'l1_regularizer'
                if self.oob_regularizer_scale:
                    self.oob_regularizer = getattr(tf.contrib.layers, self.oob_regularizer_name)(self.oob_regularizer_scale)
                else:
                    self.oob_regularizer = None

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

                    elif family in ['HRFDoubleGamma', 'HRFDoubleGamma5']:
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=1., default=16.)
                        self._initialize_base_irf_param('beta_main', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta_undershoot', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif family == 'HRFDoubleGammaUnconstrained':
                        self._initialize_base_irf_param('alpha_main', family, lb=1., default=6.)
                        self._initialize_base_irf_param('alpha_undershoot', family, lb=0., default=16.)
                        self._initialize_base_irf_param('beta_main', family, lb=0., default=1.)
                        self._initialize_base_irf_param('beta_undershoot', family, lb=0., default=1.)
                        self._initialize_base_irf_param('c', family, default=1./6.)

                    elif Formula.nonparametric_type(family):
                        np_type = Formula.nonparametric_type(family)
                        bases = Formula.bases(family)
                        spacing_power = Formula.spacing_power(family)
                        if spacing_power == 0:
                            x_init = np.zeros(bases)
                        else:
                            x_init = np.concatenate([[0.], np.cumsum(np.ones(bases-1)) ** spacing_power], axis=0)

                            time_limit = Formula.time_limit(family)
                            if time_limit is None:
                                time_limit = self.t_delta_limit
                            x_init *= time_limit / x_init[-1]
                            # x_init[1:] -= x_init[:-1]

                        for param_name in Formula.irf_params(family):
                            if param_name.startswith('x'):
                                n = int(param_name[1:])
                                default = x_init[n-1]
                                lb = None
                            elif param_name.startswith('y'):
                                n = int(param_name[1:])
                                if n == 1:
                                    default = 1
                                else:
                                    default = 0
                                lb = None
                            else:
                                n = int(param_name[1:])
                                if np_type == 'G':
                                    default = n
                                else:
                                    default = 1
                                lb = 0
                            self._initialize_base_irf_param(param_name, family, default=default, lb=lb)

                    else:
                        raise ValueError('Unrecognized IRF kernel family "%s".' % family)

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
                self.coefficient_fixed = self._scatter_along_axis(
                    fixef_ix,
                    self.coefficient_fixed_base,
                    [len(coef_ids)]
                )
                self.coefficient_fixed_summary = self._scatter_along_axis(
                    fixef_ix,
                    self.coefficient_fixed_base_summary,
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
                # integral_ub = self.t_delta_limit.astype(dtype=self.FLOAT_NP)
                integral_ub = None

                def exponential(params):
                    return exponential_irf_lambdas(
                        params,
                        session=self.sess
                    )

                self.irf_lambdas['Exp'] = exponential
                self.irf_lambdas['ExpRateGT1'] = exponential

                def gamma(params):
                    return gamma_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['Gamma'] = gamma
                self.irf_lambdas['SteepGamma'] = gamma
                self.irf_lambdas['GammaShapeGT1'] = gamma
                self.irf_lambdas['GammaKgt1'] = gamma
                self.irf_lambdas['HRFSingleGamma'] = gamma

                def shifted_gamma_lambdas(params):
                    return shifted_gamma_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['ShiftedGamma'] = shifted_gamma_lambdas
                self.irf_lambdas['ShiftedGammaShapeGT1'] = shifted_gamma_lambdas
                self.irf_lambdas['ShiftedGammaKgt1'] = shifted_gamma_lambdas

                def normal(params):
                    return normal_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess
                    )

                self.irf_lambdas['Normal'] = normal

                def skew_normal(params):
                    return skew_normal_irf_lambdas(
                        params,
                        self.t_delta_limit.astype(dtype=self.FLOAT_NP) if integral_ub is None else integral_ub,
                        session=self.sess
                    )

                self.irf_lambdas['SkewNormal'] = skew_normal

                def emg(params):
                    return emg_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess
                    )

                self.irf_lambdas['EMG'] = emg

                def beta_prime(params):
                    return beta_prime_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )

                self.irf_lambdas['BetaPrime'] = beta_prime

                def shifted_beta_prime(params):
                    return shifted_beta_prime_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon
                    )

                self.irf_lambdas['ShiftedBetaPrime'] = shifted_beta_prime

                def double_gamma_1(params):
                    return double_gamma_1_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma1'] = double_gamma_1

                def double_gamma_2(params):
                    return double_gamma_2_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma2'] = double_gamma_2

                def double_gamma_3(params):
                    return double_gamma_3_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma3'] = double_gamma_3

                def double_gamma_4(params):
                    return double_gamma_4_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma4'] = double_gamma_4

                def double_gamma_5(params):
                    return double_gamma_5_irf_lambdas(
                        params,
                        integral_ub=integral_ub,
                        session=self.sess,
                        epsilon=self.epsilon,
                        validate_irf_args=self.validate_irf_args
                    )

                self.irf_lambdas['HRFDoubleGamma'] = double_gamma_5
                self.irf_lambdas['HRFDoubleGamma5'] = double_gamma_5

    def _initialize_nonparametric_irf(
            self,
            order,
            bases,
            method='summed_gaussians',
            roughness_penalty=0.,
            support=None
    ):
        if support is None:
            support = self.support

        def f(
                params,
                order=order,
                bases=bases,
                method=method,
                roughness_penalty=roughness_penalty,
                support=support,
                epsilon=self.epsilon,
                int_type=self.INT_TF,
                float_type=self.FLOAT_TF,
                session=self.sess
        ):
            return nonparametric_smooth(
                method,
                params,
                bases,
                # integral_ub=self.t_delta_limit.astype(dtype=self.FLOAT_NP),
                order=order,
                roughness_penalty=roughness_penalty,
                support=support,
                epsilon=epsilon,
                int_type=int_type,
                float_type=float_type,
                session=session
            )

        return f

    def _get_irf_lambdas(self, family):
        if family in self.irf_lambdas:
            return self.irf_lambdas[family]
        elif Formula.nonparametric_type(family):
            order = Formula.order(family)
            bases = Formula.bases(family)
            roughness_penalty = Formula.roughness_penalty(family)
            return self._initialize_nonparametric_irf(
                order,
                bases,
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
                    assert t.p.name() == 'ROOT', 'DiracDelta may not be embedded under other IRF in CDR formula strings'
                    assert not t.impulse == 'rate', '"rate" is a reserved keyword in CDR formula strings and cannot be used under DiracDelta'

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

                    atomic_irf, atomic_irf_proportion_in_bounds = self._get_irf_lambdas(t.family)(params)
                    atomic_irf_plot, _ = self._get_irf_lambdas(t.family)(params_summary)

                    if t.p.name() in self.irf:
                        irf = self.irf[t.p.name()][:] + [atomic_irf]
                        irf_plot = self.irf[t.p.name()][:] + [atomic_irf_plot]
                    else:
                        irf = [atomic_irf]
                        irf_plot = [atomic_irf_plot]

                    if t.p.name() in self.irf_proportion_in_bounds:
                        irf_proportion_in_bounds = self.irf_proportion_in_bounds[t.p.name()][:] + [atomic_irf_proportion_in_bounds]
                    else:
                        irf_proportion_in_bounds = [atomic_irf_proportion_in_bounds]

                    assert t.name() not in self.irf, 'Duplicate IRF node name already in self.irf'
                    self.irf[t.name()] = irf
                    assert t.name() not in self.irf_proportion_in_bounds, 'Duplicate IRF node name already in self.irf_proportion_in_bounds'
                    self.irf_proportion_in_bounds[t.name()] = irf_proportion_in_bounds

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
                    if len(irf_proportion_in_bounds) > 1:
                        composite_irf_proportion_in_bounds = self._compose_irf(irf_proportion_in_bounds)(self.support[None, ...])
                    else:
                        composite_irf_proportion_in_bounds = atomic_irf_proportion_in_bounds

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

                    assert t.name() not in self.irf_proportion_in_bounds_mc, 'Duplicate IRF node name already in self.irf_proportion_before_mc'
                    self.irf_proportion_in_bounds_mc[t.name()] = {
                        'atomic': {
                            'unscaled': atomic_irf_proportion_in_bounds,
                            'scaled': atomic_irf_proportion_in_bounds
                        },
                        'composite': {
                            'unscaled': composite_irf_proportion_in_bounds,
                            'scaled': composite_irf_proportion_in_bounds
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
                if self.oob_regularizer_scale:
                    n = 0
                    oob_penalty = 0
                    if self.oob_regularizer_threshold:
                        t = self.oob_regularizer_threshold
                    else:
                        t = self.t_delta_limit.astype(dtype=self.FLOAT_NP)
                    for x in self.irf_proportion_in_bounds_mc:
                        prop_before_cur = self.irf_proportion_in_bounds_mc[x]['composite']['scaled'](t)
                        penalty_cur = tf.exp(1 / prop_before_cur)
                        n += 1
                        oob_penalty += penalty_cur
                    if n > 0:
                        self._regularize(oob_penalty, type='oob', var_name='out-of-bounds penalty')
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
                # Hack needed for MAP evaluation of CDRBayes
                self.out_mean = self.out





    ######################################################
    #
    #  Internal public network initialization methods.
    #  These must be implemented by all subclasses and
    #  should only be called at initialization.
    #
    ######################################################

    def initialize_coefficient(self, coef_ids=None, ran_gf=None):
        """
        Add coefficients.
        This method must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of coefficient IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random coefficient (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(coefficient, coefficient_summary)``; ``coefficient`` is the coefficient for use by the model. ``coefficient_summary`` is an identically-shaped representation of the current coefficient values for logging and plotting (can be identical to ``coefficient``). For fixed coefficients, should return a vector of ``len(coef_ids)`` trainable weights. For random coefficients, should return batch-length matrix of trainable weights with ``len(coef_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_interaction(self, interaction_ids=None, ran_gf=None):
        """
        Add (response-level) interactions.
        This method must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param coef_ids: ``list`` of ``str``: List of interaction IDs
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random interaction (if ``None``, constructs a fixed interaction)
        :return: 2-tuple of ``Tensor`` ``(interaction, interaction_summary)``; ``interaction`` is the interaction for use by the model. ``interaction_summary`` is an identically-shaped representation of the current interaction values for logging and plotting (can be identical to ``interaction``). For fixed interactions, should return a vector of ``len(interaction_ids)`` trainable weights. For random interactions, should return batch-length matrix of trainable weights with ``len(interaction_ids)`` columns for each input in the batch. Weights should be initialized around 0.
        """

        raise NotImplementedError

    def initialize_irf_param_unconstrained(self, param_name, ids, mean=0, ran_gf=None):
        """
        Add IRF parameters in the unconstrained space.
        CDR will apply appropriate constraint transformations as needed.
        This method must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
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
        This method is required for multivariate mode and must be implemented by subclasses of ``CDR`` and should only be called at model initialization.
        Correct model behavior is not guaranteed if called at any other time.

        :param means: ``Tensor``; 1-D tensor as MVN mean parameter.
        :param sds: ``Tensor``; 1-D tensor used to construct diagonal of MVN variance-covariance parameter.
        :param ran_gf: ``str`` or ``None``: Name of random grouping factor for random IRF param (if ``None``, constructs a fixed coefficient)
        :return: 2-tuple of ``Tensor`` ``(joint, join_summary)``; ``joint`` is the random variable for use by the model. ``joint_summary`` is an identically-shaped representation of the current joint for logging and plotting (can be identical to ``joint``). Returns a multivariate normal distribution of dimension len(means) in all cases.
        """

        raise NotImplementedError





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

    def _extract_parameter_values(self, fixed=True, level=95, n_samples=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if fixed:
                    out = self.parameter_table_fixed_values.eval(session=self.sess)
                else:
                    out = self.parameter_table_random_values.eval(session=self.sess)

            return out




    ######################################################
    #
    #  Public methods that must be implemented by
    #  subclasses
    #
    ######################################################

    def run_conv_op(self, feed_dict, scaled=False, standardize_response=False, n_samples=None, algorithm='MAP', verbose=True):
        """
        Convolve a batch of data in feed_dict with the model's latent IRF.
        **All CDR subclasses must implement this method.**

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
        :param level: ``float``; level of credible interval (used for ``CDRBayes`` only)
        :param n_samples: ``int`` or ``None``; number of posterior samples to draw (used for ``CDRBayes`` only). If ``None``, use model defaults.
        :param n_time_units: ``float``; number of time units over which to take the integral.
        :param n_time_points: ``float``; number of points to use in the discrete approximation of the integral.
        :return: ``float`` or 3-element ``numpy`` vector; either integral or mean, upper quantile, and lower quantile of integral (depending on whether model is instance of ``CDRBayes``).
        """
        raise NotImplementedError




    ######################################################
    #
    #  Shared public methods
    #
    ######################################################

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
        Construct the CDR network and initialize/load model parameters.
        ``build()`` is called by default at initialization and unpickling, so users generally do not need to call this method.
        ``build()`` can be used to reinitialize an existing network instance on the fly, but only if (1) no model checkpoint has been saved to the output directory or (2) ``restore`` is set to ``False``.

        :param restore: Restore saved network parameters if model checkpoint exists in the output directory.
        :param verbose: Report model details after initialization.
        :return: ``None``
        """

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './cdr_model/'
        else:
            self.outdir = outdir

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.pc:
                    n_impulse = len(self.src_impulse_names)
                else:
                    n_impulse = len(self.impulse_names)

                self._initialize_inputs(n_impulse)
                self._initialize_cdr_inputs()
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

                self.sess.graph.finalize()

    def get_plot_names(self, composite='composite', scaled='scaled', dirac='dirac', plot_type='irf_1d'):
        if plot_type.lower() == 'irf_1d':
            return self.plots[composite][scaled][dirac]['names']
        else:
            raise ValueError('Plot type "%s" not supported.' % plot_type)

    def get_plot_data(
            self,
            name,
            composite='composite',
            scaled='scaled',
            dirac='dirac',
            plot_type='irf_1d',
            support_start=0.,
            n_time_units=2.5,
            n_time_points=1000,
            t_interaction=0.,
            plot_rangf=False,
            rangf_vals=None
    ):
        if rangf_vals is None:
            rangf_keys = [None]
            rangf_vals = [self.gf_defaults[0]]
            if plot_rangf:
                for i in range(len(self.rangf)):
                    if type(self).__name__.startswith('CDRNN') or self.t.has_coefficient(self.rangf[i]) or self.t.has_irf(self.rangf[i]):
                        for k in self.rangf_map[i].keys():
                            rangf_keys.append(str(k))
                            rangf_vals.append(np.concatenate(
                                [self.gf_defaults[0, :i], [self.rangf_map[i][k]], self.gf_defaults[0, i + 1:]], axis=0))
            rangf_vals = np.stack(rangf_vals, axis=0)
        if plot_type.lower() == 'irf_1d':
            fd = {
                self.support_start: 0.,
                self.n_time_units: n_time_units,
                self.n_time_points: n_time_points,
                self.max_tdelta_batch: n_time_points,
                self.gf_y: rangf_vals,
                self.training: not self.predict_mode
            }
            ix = self.plots[composite][scaled][dirac]['names'].index(name)
            return self.sess.run([self.support, self.plots[composite][scaled][dirac]['plot'][ix]], feed_dict=fd)
        else:
            raise ValueError('Plot type "%s" not supported.' % plot_type)

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

    def report_settings(self, indent=0):
        out = super(CDR, self).report_settings(indent=indent)
        for kwarg in CDR_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * indent + '  %s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

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
        out += ' ' * indent + '#    CDR MODEL SUMMARY    #\n'
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
        Convolve input data using the fitted CDR model.

        :param X: list of ``pandas`` tables; matrices of independent variables, grouped by series and temporally sorted.
            Each element of **X** must contain the following columns (additional columns are ignored):

            * ``time``: Timestamp associated with each observation in **X**

            Across all elements of **X**, there must be a column for each independent variable in the CDR ``form_str`` provided at initialization.

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
            stderr('Using GPU: %s\n' % usingGPU)

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

        X_2d, time_X_2d, time_X_mask = build_CDR_impulses(
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
                    self.time_X_mask: time_X_mask,
                    self.training: not self.predict_mode
                }

                fd_minibatch = {
                    self.X: fd[self.X],
                    self.time_X: fd[self.time_X],
                    self.training: not self.predict_mode
                }

                X_conv = []
                n_eval_minibatch = math.ceil(len(y) / self.eval_minibatch_size)
                for i in range(0, len(y), self.eval_minibatch_size):
                    if verbose:
                        stderr('\rMinibatch %d/%d' % ((i / self.eval_minibatch_size) + 1, n_eval_minibatch))
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
        :param algorithm: ``str``; algorithm to use for extracting IRFs from CDRBayes, one of [``MAP``, ``sampling``]. Ignored for MLE models.
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

                if type(self).__name__ == 'CDRBayes' and algorithm.lower() == 'sampling':
                    if n_samples is None:
                        n_samples = self.n_samples_eval

                    # fd[self.time_y] = np.ones((1,)) * n_time_units
                    # fd[self.time_X] = np.zeros((1, self.history_length))

                    preds = []

                    for name in names:
                        posterior = self.irf_mc[name]['atomic']['scaled']

                        samples = [self.sess.run(posterior, feed_dict=fd)[0] for _ in range(n_samples)]
                        samples = np.concatenate(samples, axis=1)
                        preds.append(samples)

                    preds = np.stack(preds, axis=2)
                    if summed:
                        preds = preds.sum(axis=2)
                    gold = np.expand_dims(gold, 1)

                else:
                    preds = [self.sess.run(plots['plot'][i][0], feed_dict=fd) for i in range(len(plots['plot'])) if plots['names'][i] in names]
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
            outname = self.outdir + '/cdr_parameters.csv'
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
            outname = self.outdir + '/cdr_irf_integrals.csv'
        else:
            outname = outfile

        irf_integrals.to_csv(outname, index=False)

