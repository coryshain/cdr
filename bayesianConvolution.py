import sys
import os
os.environ['THEANO_FLAGS'] = 'optimizer=None'
import numpy as np
from numpy import inf, nan
from theano import tensor as T, function, printing
import pymc3 as pm
from pymc3.theanof import generator
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook')

def gamma_func(alpha, beta):
    def gamma(x):
        out = x**(alpha-1)
        out *= beta**alpha
        out *= pm.math.exp(-beta*x)
        out /= T.gamma(beta)
        return out
    return gamma

inp = pd.read_csv('genmodel/naturalstories.evmeasures', sep=' ', skipinitialspace=True)
inp = inp[inp['fdur'] != nan]
inp = inp.fillna(value=0)
inp = inp.head(10000)
inp['time'] = inp.groupby('subject').fdur.cumsum()

window = 100

# Define data generator
class Context(object):
    def __init__(self, df, window=inf):
        self.df = df
        self.window = window

    def __getitem__(self, i):
        return self.df[max(0, i-self.window):i]

def get_X(df, window=inf):
    i = 1
    subj_cur = df.iloc[0].subject
    doc_cur = df.iloc[0].docid
    first_row = 0
    while i <= len(df):
        if df.iloc[i-1].subject != subj_cur:
            first_row = i-1
            subj_cur = df.iloc[i-1].subject
        if df.iloc[i-1].docid != doc_cur:
            first_row = i-1
            doc_cur = df.iloc[i-1].docid
        yield np.array(df[max(first_row,i-window):i][features]).astype('float32')
        if i == len(df):
            i = 1
            subj_cur = df.iloc[0].subject
            doc_cur = df.iloc[0].docid
            first_row = 0
        else:
            i += 1

def get_t(df, window=inf):
    i = 1
    subj_cur = df.iloc[0].subject
    doc_cur = df.iloc[0].docid
    first_row = 0
    while i <= len(df):
        if df.iloc[i-1].subject != subj_cur:
            first_row = i-1
            subj_cur = df.iloc[i-1].subject
        if df.iloc[i-1].docid != doc_cur:
            first_row = i-1
            doc_cur = df.iloc[i-1].docid
        yield np.expand_dims(np.array(df.iloc[i-1]['time'] - df[max(first_row,i-window):i]['time']).astype('float32'), -1) + 1
        if i == len(df):
            i = 1
            subj_cur = df.iloc[0].subject
            doc_cur = df.iloc[0].docid
            first_row = 0
        else:
            i += 1

n_subj = len(inp['subject'].unique())
features = ['sentpos', 'nItem']
n_feat = len(features)
mu = np.zeros(n_feat)
sigma = np.eye(n_feat) * 0.05

with pm.Model() as model:
    shape = pm.Gamma('shape', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('shape', mu=mu, cov=sigma, shape=n_feat)
    rate = pm.Gamma('rate', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('rate', mu=mu, cov=sigma, shape=n_feat)
    beta = pm.Normal('beta', mu=0, sd=1, shape=n_feat)#pm.MvNormal('beta', mu=mu, cov=sigma, shape=n_feat)
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('Intercept', 0, sd=20)
    
    conv = gamma_func(shape, rate)
   
    t = generator(get_t(inp, window))
    t = T.addbroadcast(t, 1)
    X = generator(get_X(inp, window)) * conv(t)
 
    fdur = pm.Normal('fdur', mu = intercept + pm.math.sum(pm.math.dot(X,beta), 0), sd = sigma, observed=inp['fdur'])

    trace = pm.sample(100, tune=5)
    
    fig, ax = plt.subplots(5, 2)
    pm.traceplot(trace, ax=ax)
    plt.savefig('trace.jpg')

