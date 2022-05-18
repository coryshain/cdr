import string
import sys
import numpy as np
import scipy.stats
import pandas as pd

from .plot import plot_irf
from .util import stderr


def read_params(path):
    return pd.read_csv(path, sep=' ', index_col=0)


def irf(x, irf_name, irf_params, coefs=None):
    if irf_name.lower() == 'exp':
        out = np.exp(-x[..., None] * irf_params['beta'][None, ...])
    elif irf_name.lower() == 'normal':
        out = np.exp(-(x[..., None] - irf_params['mu'][None, ...]) ** 2 / irf_params['sigma2'][None, ...])
    elif irf_name.lower() in ['gamma', 'hrf']:
        out = scipy.stats.gamma.pdf(x[..., None], irf_params['alpha'], scale=1./irf_params['beta'])
    elif irf_name.lower() == 'shiftedgamma':
        out = scipy.stats.gamma.pdf(x[..., None], irf_params['alpha'], scale=1./irf_params['beta'], loc=irf_params['delta'])
    elif irf_name.lower() == 'periodic':
        out = np.cos(x[..., None] * irf_params['frequency'][None, ...] - irf_params['phase'][None, ...]) * np.exp(-x[..., None] * irf_params['rate'][None, ...])
    else:
        raise ValueError('Unrecognized IRF family "%s"' % irf_name)

    if coefs is not None:
        out *= coefs[None, ...]

    return out


class SyntheticModel(object):
    """
    A data structure representing a synthetic "true" model for empirical validation of CDR fits.
    Contains a randomly generated set of IRFs that can be used to convolve data, and provides methods
    for sampling data with particular structure and convolving it with the true IRFs in order to generate
    a response vector.

    :param n_pred: ``int``; Number of predictors in the synthetic model.
    :param irf_name: ``str``; Name of IRF kernel to use. One of ``['Exp', 'Normal', 'Gamma', 'ShiftedGamma']``.
    :param irf_params: ``dict`` or ``None``; Dictionary of IRF parameters to use, with parameter names as keys and numeric arrays as values. Values must each have **n_pred** cells. If ``None``, parameter values will be randomly sampled.
    :param coefs: numpy array or ``None``; Vector of coefficients to use, where ``len(coefs) == n_pred``. If ``None``, coefficients will be randomly sampled.
    :param fn: ``str`` or ``None``; Effect shape to use. One of ``['quadratic', 'exp', 'logmod', 'linear']. If ``None``, linear effects.
    :param interactions: ``bool``; Whether there are randomly sampled pairwise interactions (same bounds as those used for coefs).
    :param ranef_range: ``float`` or ``None``; Maximum magnitude of simulated random effects. If ``0`` or ``None``, no random effects.
    :param n_ranef_levels: ``int`` or ``None``; Number of random effects levels. If ``0`` or ``None``, no random effects.
    """

    IRF_PARAMS = {
        'Normal': ['mu', 'sigma2'],
        'Exp': ['beta'],
        'Gamma': ['alpha', 'beta'],
        'ShiftedGamma': ['alpha', 'beta', 'delta'],
        'HRF': ['alpha', 'beta'],
        'Periodic': ['frequency', 'phase', 'rate']
    }

    IRF_BOUNDS = {
        'Exp': {'beta': [0, 5]},
        'Normal': {'mu': [-2,2], 'sigma2': [0, 2]},
        'Gamma': {'alpha': [1, 6], 'beta': [0, 5]},
        'ShiftedGamma': {'alpha': [1, 6], 'beta': [0, 5], 'delta': [-1, 0]},
        'HRF': {'alpha': [3, 9], 'beta': [0, 2]},
        'Periodic': {'frequency': [0.1, 10], 'phase': [0, 2*np.pi], 'rate': [0, 5]}
    }

    def __init__(
            self,
            n_pred,
            irf_name,
            irf_params=None,
            coefs=None,
            fn=None,
            interactions=False,
            ranef_range=None,
            n_ranef_levels=None
    ):
        self.n_pred = n_pred
        self.irf_name = irf_name
        if irf_params is None:
            irf_params = {}
            for x in SyntheticModel.IRF_BOUNDS[irf_name]:
                l, u = SyntheticModel.IRF_BOUNDS[irf_name][x]
                irf_params[x] = np.random.uniform(l, u, (self.n_pred,))
        self.irf_params = irf_params

        if coefs is None:
            coefs = np.random.uniform(-10, 10, (self.n_pred,))
        self.coefs = coefs

        self.fn = fn
        self.interactions = interactions

        self.ranef_range = ranef_range
        self.n_ranef_levels = n_ranef_levels
        self.ranef_irf = {}
        self.ranef_coef = {}
        if ranef_range and n_ranef_levels:
            for i in range(n_ranef_levels):
                _ranef_irf = {k: np.random.uniform(-ranef_range, ranef_range, (self.n_pred,)) for k in irf_params}
                _ranef_coef = np.random.uniform(-ranef_range, ranef_range, (self.n_pred,))
                self.ranef_irf['r%d' % i] = _ranef_irf
                self.ranef_coef['r%d' % i] = _ranef_coef
        self.ranef_levels = sorted(self.ranef_irf.keys())

    def irf(self, x, coefs=False, ranef_level=None):
        """
        Computes the values of the model's IRFs elementwise over a vector of timepoints.

        :param x: numpy array; 1-D array with shape ``[N]`` containing timepoints at which to query the IRFs.
        :param coefs: ``bool``; Whether to rescale responses by coefficients
        :param ranef_level: ``str`` or ``None``; Random effects level to use (or ``None`` to use population-level effect)
        :return: numpy array; 2-D array with shape ``[N, K]`` containing values of the model's ``K`` IRFs evaluated at the timepoints in **x**.
        """

        if coefs:
            coefs = self.coefs
            if ranef_level:
                coefs += self.ranef_coef[ranef_level]
        else:
            coefs = None

        irf_params = self.irf_params.copy()
        if ranef_level:
            for k in irf_params:
                irf_params[k] += self.ranef_irfs[ranef_level][k]

        return irf(x, self.irf_name, irf_params, coefs=coefs)

    def sample_data(self, m, n=None, X_interval=None, y_interval=None, rho=None, align_X_y=True):
        """
        Samples synthetic predictors and time vectors

        :param m: ``int``; Number of predictors.
        :param n: ``int``; Number of response query points.
        :param X_interval: ``str``, ``float``, ``list``, ``tuple``, or ``None``; Predictor interval model. If ``None``, predictor offsets are randomly sampled from an exponential distribution with parameter ``1``. If ``float``, predictor offsets are evenly spaced with interval **X_interval**. If ``list`` or ``tuple``, the first element is the name of a scipy distribution to use for sampling offsets, and all remaining elements are positional arguments to that distribution.
        :param y_interval: ``str``, ``float``, ``list``, ``tuple``, or ``None``; Response interval model. If ``None``, response offsets are randomly sampled from an exponential distribution with parameter ``1``. If ``float``, response offsets are evenly spaced with interval **y_interval**. If ``list`` or ``tuple``, the first element is the name of a scipy distribution to use for sampling offsets, and all remaining elements are positional arguments to that distribution.
        :param rho: ``float``; Level of pairwise correlation between predictors.
        :param align_X_y: ``bool``; Whether predictors and responses are required to be sampled at the same points in time.
        :return: (2-D numpy array, 1-D numpy array, 1-D numpy array); Matrix of predictors, vector of predictor timestamps, vector of response timestamps
        """

        if X_interval is None:
            t_X = np.random.exponential(1., (m,)).cumsum()
        elif isinstance(X_interval, tuple) or isinstance(X_interval, list):
            t_X = getattr(np.random, X_interval[0])(*X_interval[1:], (m,)).cumsum()
        else:
            t_X = np.arange(0, m) * X_interval
        if align_X_y:
            t_y = t_X
        else:
            if not n:
                n = m
            if y_interval is None:
                t_y = np.random.exponential(1., (n,)).cumsum()
            elif isinstance(y_interval, tuple) or isinstance(y_interval, list):
                t_y = getattr(np.random, y_interval[0])(*y_interval[1:], (n,)).cumsum()
            else:
                t_y = np.arange(0, n) * y_interval
        if rho:
            if rho < 1:
                sigma = np.ones((self.n_pred, self.n_pred)) * rho
                np.fill_diagonal(sigma, 1.)
                X = np.random.multivariate_normal(np.zeros(self.n_pred), sigma, (m,))
            else:
                X = np.random.normal(0, 1, (m,))[..., None]
                X = np.tile(X, [1, 20])
        else:
            X = np.random.normal(0, 1, (m, self.n_pred))

        return X, t_X, t_y

    def convolve(
            self,
            X,
            t_X,
            t_y,
            history_length=None,
            err_sd=None,
            allow_instantaneous=True,
            ranef_level=None,
            verbose=True
    ):
        """
        Convolve data using the model's IRFs.

        :param X: numpy array; 2-D array of predictors.
        :param t_X: numpy array; 1-D vector of predictor timestamps.
        :param t_y: numpy array; 1-D vector of response timestamps.
        :param history_length: ``int`` or ``None``; Drop preceding events more than ``history_length`` steps into the past. If ``None``, no history clipping.
        :param err_sd: ``float`` or ``None``; Standard deviation of Gaussian noise to inject into responses. If ``None``, use the empirical standard deviation of the response vector.
        :param allow_instantaneous: ``bool``; Whether to compute responses when ``t==0``.
        :param ranef_level: ``str`` or ``None``; Random effects level to use (or ``None`` to use population-level effect)
        :param verbose: ``bool``; Verbosity.
        :return: (2-D numpy array, 1-D numpy array); Matrix of convolved predictors, vector of responses
        """

        y = np.zeros(len(t_y))
        X_conv = np.zeros((len(t_X), self.n_pred))

        if verbose:
            stderr('Convolving...\n')

        i = 0
        j = 0

        def cond(x, y):
            if allow_instantaneous:
                return x <= y
            else:
                return x < y

        while j < len(t_y):
            if verbose:
                if j % 1000 == 0:
                    stderr('\r%d/%d' % (j, len(t_y)))
            while i < len(t_X) and cond(t_X[i], t_y[j]):
                i += 1
            if history_length:
                s_ix = i - history_length
            else:
                s_ix = 0
            t_delta = t_y[j] - t_X[s_ix:i]
            _X_conv = self.irf(t_delta, ranef_level=ranef_level)
            X_conv[j] = np.sum(_X_conv * X[s_ix:i], axis=0, keepdims=True) * self.coefs[None, ...]
            y[j] = X_conv[j].sum(axis=-1)
            j += 1

        if verbose:
            stderr('\r%d/%d' % (len(t_y), len(t_y)))

        if verbose:
            stderr('\n')

        if err_sd != 0:
            if err_sd is None:
                err_sd = np.std(y)
            y += np.random.normal(loc=0., scale=err_sd, size=y.shape)

        return X_conv, y

    def convolve_v2(self, X, t_X, t_y, err_sd=None, allow_instantaneous=True, verbose=True):
        """
        Convolve data using the model's IRFs. Alternate memory-intensive implementation that is faster for small arrays
        but can exhaust resources for large ones.

        :param X: numpy array; 2-D array of predictors.
        :param t_X: numpy array; 1-D vector of predictor timestamps.
        :param t_y: numpy array; 1-D vector of response timestamps.
        :param err_sd: ``float``; Standard deviation of Gaussian noise to inject into responses.
        :param allow_instantaneous: ``bool``; Whether to compute responses when ``t==0``.
        :param verbose: ``bool``; Verbosity.
        :return: (2-D numpy array, 1-D numpy array); Matrix of convolved predictors, vector of responses
        """

        t_delta = t_y[..., None] - t_X[None, ...]
        if allow_instantaneous:
            mask = t_delta >= 0
        else:
            mask = t_delta > 0
        g_t = self.irf(t_delta)
        W = np.where(mask[..., None], g_t, np.zeros_like(g_t))
        X_conv = (W * X[None, ...]).sum(axis=1) * self.coefs[None, ...]
        y = X_conv.sum(axis=-1)
        if err_sd:
            y += np.random.normal(loc=0., scale=err_sd, size=y.shape)
        return X_conv, y

    def get_curves(
            self,
            n_time_units=None,
            n_time_points=None,
            ranef_level=None
    ):
        """
        Extract response curves as an array.

        :param n_time_units: ``float``; Number of units of time over which to extract curves.
        :param n_time_points: ``int``; Number of samples to extract for each curve (resolution of curve)
        :param ranef_level: ``str`` or ``None``; Random effects level to use (or ``None`` to use population-level effect)

        :return: numpy array; 2-D numpy array with shape ``[T, K]``, where ``T`` is **n_time_points** and ``K`` is the number of predictors in the model.
        """

        a = 5 if n_time_units is None else n_time_units
        b = 1000 if n_time_points is None else n_time_points
        plot_x = np.linspace(0, a, b)
        plot_y = self.irf(plot_x, coefs=True, ranef_level=ranef_level)

        return plot_x, plot_y

    def plot_irf(
            self,
            n_time_units=None,
            n_time_points=None,
            dir='.',
            filename='synth_irf.png',
            plot_x_inches=6,
            plot_y_inches=4,
            cmap='gist_rainbow',
            legend=False,
            xlab=None,
            ylab=None,
            use_line_markers=False,
            transparent_background=False
    ):
        """
        Plot impulse response functions.

        :param n_time_units: ``float``; number if time units to use for plotting.
        :param n_time_points: ``int``; number of points to use for plotting.
        :param dir: ``str``; output directory.
        :param filename: ``str``; filename.
        :param plot_x_inches: ``float``; width of plot in inches.
        :param plot_y_inches: ``float``; height of plot in inches.
        :param cmap: ``str``; name of ``matplotlib`` ``cmap`` object (determines colors of plotted IRF).
        :param legend: ``bool``; include a legend.
        :param xlab: ``str`` or ``None``; x-axis label. If ``None``, no label.
        :param ylab: ``str`` or ``None``; y-axis label. If ``None``, no label.
        :param use_line_markers: ``bool``; add markers to IRF lines.
        :param transparent_background: ``bool``; use a transparent background. If ``False``, uses a white background.
        :return: ``None``
        """


        if n_time_units is None:
            n_time_units = 5
        if n_time_points is None:
            n_time_points = 1000
        if plot_x_inches is None:
            plot_x_inches = 6
        if plot_y_inches is None:
            plot_y_inches = 4

        plot_x, plot_y = self.get_curves(n_time_units=n_time_units, n_time_points=n_time_points)

        plot_irf(
            plot_x,
            plot_y,
            string.ascii_lowercase[:self.n_pred],
            outdir=dir,
            filename=filename,
            plot_x_inches=plot_x_inches,
            plot_y_inches=plot_y_inches,
            cmap=cmap,
            legend=legend,
            xlab=xlab,
            ylab=ylab,
            use_line_markers=use_line_markers,
            transparent_background=transparent_background
        )

        for ranef_level in self.ranef_levels:
            plot_x, plot_y = self.get_curves(
                n_time_units=n_time_units,
                n_time_points=n_time_points,
                ranef_level=ranef_level
            )

            plot_irf(
                plot_x,
                plot_y,
                string.ascii_lowercase[:self.n_pred],
                outdir=dir,
                filename='%s_' % ranef_level + filename,
                plot_x_inches=plot_x_inches,
                plot_y_inches=plot_y_inches,
                cmap=cmap,
                legend=legend,
                xlab=xlab,
                ylab=ylab,
                use_line_markers=use_line_markers,
                transparent_background=transparent_background
            )
