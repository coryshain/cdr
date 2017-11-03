import sys
import os
import pickle
import string
import numpy as np
import pandas as pd
from scipy.stats import gamma
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



def z(df):
    return (df-df.mean(axis=0))/df.std(axis=0)

def c(df):
    return df-df.mean(axis=0)

def conv(x, k, theta, delta):
    return gamma.pdf(x, k, scale=theta, loc=delta)

def plot_convolutions(plot_x, plot_y, features, dir, filename='convolution_plot.jpg'):
    cm = plt.get_cmap('gist_earth')
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])
    n_feat = plot_y.shape[-1]
    for i in range(n_feat):
        plt.plot(plot_x, plot_y[:,i], label=features[i])
    plt.legend()
    plt.savefig(dir+'/'+filename)
    plt.clf()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates synthetic temporally convolved data for DTSR evaluation.
    ''')
    argparser.add_argument('-n', '--n', type=int, default=1, help='Number of covariates')
    argparser.add_argument('-x', '--x', type=int, default=10000, help='Length of obs table')
    argparser.add_argument('-y', '--y', type=int, default=10000, help='Length of dv table')
    argparser.add_argument('-s', '--s', type=float, default = 0.1, help='Temporal step length')
    argparser.add_argument('-k', '--k', type=float, default = None, help='Value for shape parameter k (randomly sampled by default)')
    argparser.add_argument('-t', '--theta', type=float, default = None, help='Value for scale parameter theta (randomly sampled by default)')
    argparser.add_argument('-d', '--delta', type=float, default = None, help='Value for location parameter delta (randomly sampled by default)')
    argparser.add_argument('-b', '--beta', type=float, default = None, help='Value for linear coefficient beta (randomly sampled by default)')
    argparser.add_argument('-e', '--error', type=float, default = None, help='SD of error distribution')
    argparser.add_argument('-o', '--outdir', type=str, default='.', help='Output directory in which to save synthetic data tables (randomly sampled by default)')
    argparser.add_argument('-N', '--name', type=str, default=None, help='Name for synthetic dataset')

    args, unknown = argparser.parse_known_args()

    if args.k is None:
        k = np.random.random(args.n)*2
    else:
        k = np.ones(args.n)*args.k
    if args.theta is None:
        theta = np.random.random(args.n)*2
    else:
        theta = np.ones(args.n)*args.theta
    if args.delta is None:
        delta = -np.random.random(args.n)*2
    else:
        delta = np.ones(args.n)*args.delta
    if args.beta is None:
        beta = np.random.random(args.n)*100-50
    else:
        beta = np.ones(args.n)*args.beta
    X = np.random.normal(0, 1, (args.x, args.n))

    time_X = np.linspace(0., args.s*args.x, args.x)
    time_y = np.linspace(0., args.s*args.y, args.y)

    y_pred = np.zeros(args.y)

    for i in range(args.y):
        delta_t = np.expand_dims(time_y[i] - time_X[0:i+1], -1)
        X_conv = np.sum(conv(delta_t, k, theta, delta) * X[0:i+1], axis=0, keepdims=True)
        y_pred[i] = np.dot(X_conv, np.expand_dims(beta, 1))
        sys.stderr.write('\r%d/%d' %(i+1, args.y))
    sys.stderr.write('\n')

    if args.error is not None:
        y = y_pred + np.random.normal(0, args.error, args.y)
    else:
        y = y_pred

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.name is None:
        exp_name = str(np.random.randint(0, 100000))
    else:
        exp_name = args.name
    os.mkdir(args.outdir + '/' + exp_name)

    names = ['time', 'subject', 'sentid', 'docid'] + list(string.ascii_lowercase[:args.n])
    df_x = pd.DataFrame(np.concatenate([time_X[:,None], np.zeros((args.x, 3)), X], axis=1), columns=names)
    df_x.to_csv(args.outdir + '/' + exp_name + '/X.evmeasures', ' ', index=False, na_rep='nan')

    names = ['time', 'subject', 'sentid', 'docid', 'y']
    df_y = pd.DataFrame(np.concatenate([time_y[:,None], np.zeros((args.y, 3)), y[:, None]], axis=1), columns=names)
    df_y.to_csv(args.outdir + '/' + exp_name + '/y.evmeasures', ' ', index=False, na_rep='nan')

    names = ['k', 'theta', 'delta', 'beta']
    row_names = list(string.ascii_lowercase[:args.n])
    df_params = pd.DataFrame(np.stack([k, theta, delta, beta], axis=1), columns=names, index=row_names)
    df_params.to_csv(args.outdir + '/' + exp_name + '/params.evmeasures', ' ', index=True, na_rep='nan')

    names = ['true', 'preds']
    df_preds = pd.Series(y_pred)
    df_preds.to_csv(args.outdir + '/' + exp_name + '/preds_full.txt', index=False, na_rep='nan')

    names = ['mse']
    df_err = pd.Series((y-y_pred)**2)
    df_err.to_csv(args.outdir + '/' + exp_name + '/MSE_losses_full.txt', index=False, na_rep='nan')

    plot_x = np.expand_dims(np.linspace(0., 2.5, 1000), -1)
    plot_y = conv(plot_x, k, theta, delta)*beta

    with open(args.outdir + '/' + exp_name + '/conv_true.obj', 'wb') as f:
        pickle.dump(plot_x, f)


    plot_convolutions(plot_x, plot_y, string.ascii_lowercase[:args.n], args.outdir + '/' + exp_name)


