import sys
import math
from scipy.stats import norm
import numpy as np

from .util import stderr


def permutation_test(a, b, n_iter=10000, n_tails=2, mode='loss', nested=False, verbose=True):
    """
    Perform a paired permutation test for significance.

    :param a: ``numpy`` vector; first error/loss/prediction vector.
    :param b: ``numpy`` vector; second error/loss/prediction vector.
    :param n_iter: ``int``; number of resampling iterations.
    :param n_tails: ``int``; number of tails.
    :param mode: ``str``; one of ``["mse", "loglik"]``, the type of error used (SE's are averaged while loglik's are summed).
    :param nested: ``bool``; assume that the second model is nested within the first.
    :param verbose: ``bool``; report progress logs to standard error.
    :return:
    """

    err_table = np.stack([a, b], 1)
    if mode == 'mse':
        base_diff = err_table[:,0].mean() - err_table[:,1].mean()
        if nested and base_diff <= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    elif mode == 'loglik':
        base_diff = err_table[:,0].sum() - err_table[:,1].sum()
        if nested and base_diff >= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    elif mode == 'corr':
        denom = len(err_table) - 1
        base_diff = (err_table[:,0].sum() - err_table[:,1].sum()) / denom
        if nested and base_diff >= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    else:
        raise ValueError('Unrecognized metric "%s" in permutation test' %mode)

    if base_diff == 0:
        return (1.0, base_diff, np.zeros((n_iter,)))

    hits = 0
    if verbose:
        stderr('Difference in test statistic: %s\n' % base_diff)
        stderr('Permutation testing...\n')

    diffs = np.zeros((n_iter,))

    for i in range(n_iter):
        if verbose and (i == 0 or (i + 1) % 100 == 0):
            stderr('\r%d/%d' %(i+1, n_iter))
        shuffle = (np.random.random((len(err_table))) > 0.5).astype('int')
        m1 = err_table[np.arange(len(err_table)),shuffle]
        m2 = err_table[np.arange(len(err_table)),1-shuffle]
        if mode == 'mse':
            cur_diff = m1.mean() - m2.mean()
        elif mode == 'loglik':
            cur_diff = m1.sum() - m2.sum()
        elif mode == 'corr':
            cur_diff = (m1.sum() - m2.sum()) / denom
        diffs[i] = cur_diff
        if n_tails == 1:
            if base_diff < 0 and cur_diff <= base_diff:
                hits += 1
            elif base_diff > 0 and cur_diff >= base_diff:
                hits += 1
        elif n_tails == 2:
            if math.fabs(cur_diff) > math.fabs(base_diff):
                hits += 1
        else:
            raise ValueError('Invalid bootstrap parameter n_tails: %s. Must be in {1, 2}.' % n_tails)

    p = float(hits+1)/(n_iter+1)

    if verbose:
        stderr('\n')

    return p, base_diff, diffs


def correlation_test(y, x1, x2, nested=False, verbose=True):
    """
    Perform a parametric test of difference in correlation with observations between two prediction vectors, based on Steiger (1980).

    :param y: ``numpy`` vector; observation vector.
    :param x1: ``numpy`` vector; first prediction vector.
    :param x2: ``numpy`` vector; second prediction vector.
    :param nested: ``bool``; assume that the second model is nested within the first.
    :param verbose: ``bool``; report progress logs to standard error.
    :return:
    """

    n = len(y)

    r1 = np.corrcoef(y, x1, rowvar=False)[0, 1]
    r2 = np.corrcoef(y, x2, rowvar=False)[0, 1]
    rx = np.corrcoef(x1, x2, rowvar=False)[0, 1]
    rdiff = r1 - r2
    if nested and r1 >= r2:
        return 1.0, 0.0, r1, r2, rx, rdiff

    r_2_mu = (r1**2 + r2 **2) / 2

    f = (1 - rx) / (2 * (1 - r_2_mu))

    h = (1 - f * r_2_mu) / (1 - r_2_mu)

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    Z = (z1 - z2) * np.sqrt((n-3)/(2*(1-rx)*h))

    p = 2 * norm.sf(np.abs(Z))

    return p, Z, r1, r2, rx, rdiff

