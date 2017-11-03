from __future__ import print_function
import math
import numpy as np

def names2ix(names, l):
    if type(names) is not list:
        names = [names]
    ix = []
    for n in names:
        ix.append(l.index(n))
    if len(ix) > 1:
        return np.array(ix)
    return ix[0]

def mse(true, preds):
    return ((true-preds)**2).mean()

def mae(true, preds):
    return (true-preds).abs().mean()

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
