from __future__ import print_function
import re
import math
import numpy as np
from scipy import linalg


def names2ix(names, l, dtype=np.int32):
    if type(names) is not list:
        names = [names]
    ix = []
    for n in names:
        ix.append(l.index(n))
    return np.array(ix, dtype=dtype)

def mse(true, preds):
    return ((true-preds)**2).mean()

def mae(true, preds):
    return (true-preds).abs().mean()

def r_squared(true, preds):
    true_mean = true.mean()
    return ((preds-true_mean)**2).sum() / ((true-true_mean)**2).sum()

def powerset(n):
    return np.indices([2]*n).reshape(n, -1).T[1:].astype('int32')

def getRandomPermutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv

def print_tee(s, file_list):
    for f in file_list:
        print(s, file=f)

def logLik(res):
    N = len(res)
    val = - 0.5* N * (np.log(2*math.pi) + 1 - math.log(N) + np.log((res**2).sum()))
    return val

def sn(string):
    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)

def algorithm_u(ns, m):
    """
    Compute unique ``m``-way partitions from a list

    Code shamelessly stolen from Github user Adeel Zafar Soomro (https://codereview.stackexchange.com/users/2516/adeel-zafar-soomro).
    :param ns: List/set to partition
    :param m: Number of partitions
    :return: List of ``m``-way partitions
    """

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    parts_gen = f(m, n, 0, n, a)
    parts = []
    for p in parts_gen:
        parts.append(sorted(p, key=lambda x: len(x)))

    if len(parts) > 1:
        parts = sorted(sorted(parts, key=lambda x: x[0][0]), key=lambda x: len(x[0]))
    parts.append([ns,[]])

    return parts

def pca(X, n_dim=None, dtype=np.float32):
    X = np.array(X, dtype=dtype)
    assert len(X.shape) == 2, 'Wrong dimensionality for PCA (X must be rank 2).'
    means = X.mean(0, keepdims=True)
    sds = X.std(0, keepdims=True)
    X -= means
    X /= sds
    C = np.cov(X, rowvar=False)
    eigenval, eigenvec = linalg.eigh(C)
    sorted_id = np.argsort(eigenval)[::-1]
    eigenval = eigenval[sorted_id]
    eigenvec = eigenvec[:,sorted_id]
    if n_dim is not None and n_dim < eigenvec.shape[1]:
        eigenvec = eigenvec[:,:n_dim]
    Xpc = np.dot(X, eigenvec)
    return Xpc, eigenvec, eigenval, means, sds

def pca_inv(Xpc, eigenvec, means=None, sds=None):
    Xpc = np.array(Xpc)
    X = np.dot(Xpc, eigenvec.T)
    if sds is not None:
        if len(sds.shape) == 0:
            sds = np.expand_dims(sds, 0)
        X *= sds
    if means is not None:
        if len(means.shape) == 0:
            means = np.expand_dims(means, 0)
        X += means
    return X
