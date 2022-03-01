import sys
import os
import pickle
import string
import itertools
import numpy as np
import pandas as pd
import argparse

from cdr.synth import SyntheticModel


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Create synthetic data using randomly generated IRFs.
    ''')
    argparser.add_argument('-m', '--m', nargs='+', type=int, default=[10000], help='Number of impulses')
    argparser.add_argument('-n', '--n', nargs='+', type=int, default=[10000], help='Number of responses')
    argparser.add_argument('-k', '--k', nargs='+', type=int, default=[20], help='Number of predictors')
    argparser.add_argument('-p', '--partition', nargs='*', default=[None], help='Generate len(partition) different partitions from each synthetic parameterization. If unspecified, generates 1 unnamed partition.')
    argparser.add_argument('-a', '--asynchronous', nargs='+', type=int, default=[0], help='Use asynchronous impulses/responses.')
    argparser.add_argument('-g', '--irf', nargs='+', type=str, default=['ShiftedGamma'], help='Default IRF type (one of ["Exp", "Normal", "Gamma", "ShiftedGamma"])')
    argparser.add_argument('-x', '--X_interval', nargs='+', type=str, default=[0.1], help='Interval definition for the impulse stream. Either a float (fixed-length step) or "-"-delimited tuple of <distribution>_(<param>)+')
    argparser.add_argument('-y', '--y_interval', nargs='+', type=str, default=[0.1], help='Interval definition for the response stream. Either a float (fixed-length step) or "-"-delimited tuple of <distribution>_(<param>)+')
    argparser.add_argument('-r', '--rho', nargs='+', type=float, default=[0.], help='Fixed correlation level for covariates. If ``None`` or ``0``, uncorrelated covariates.')
    argparser.add_argument('-l', '--history_length', nargs='+', type=str, default=['None'], help='Maximum history length. If ``None`` or ``0``, no history_clipping.')
    argparser.add_argument('-e', '--error', nargs='+', type=float, default=[None], help='SD of error distribution')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=None, help='Number of time units on x-axis of plots')
    argparser.add_argument('-R', '--resolution', type=float, default=None, help='Number of points on x-axis of plots')
    argparser.add_argument('-w', '--width', type=float, default=None, help='Width of plot in inches')
    argparser.add_argument('-X', '--xlab', type=str, default=None, help='x-axis label (if default -- None -- no label)')
    argparser.add_argument('-H', '--height', type=float, default=None, help='Height of plot in inches')
    argparser.add_argument('-Y', '--ylab', type=str, default=None, help='y-axis label (if default -- None -- no label)')
    argparser.add_argument('-c', '--cmap', type=str, default='gist_rainbow', help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-M', '--markers', action='store_true', help='Add markers to IRF lines')
    argparser.add_argument('-t', '--transparent_background', action='store_true', help='Use transparent background (otherwise white background)')
    argparser.add_argument('-o', '--outdir', type=str, default='.', help='Output directory in which to save synthetic data tables (randomly sampled by default)')

    args = argparser.parse_args()

    history_length = [None if x.lower() == 'none' else int(x) for x in args.history_length]

    for m, n, k, a, g, X_interval, y_interval, rho, h, partition in itertools.product(*[
        args.m,
        args.n,
        args.k,
        args.asynchronous,
        args.irf,
        args.X_interval,
        args.y_interval,
        args.rho,
        history_length,
        args.partition
    ]):
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        model_name = '_'.join([
            'k%d' % k,
            'g%s' % g,
        ])
        if not os.path.exists(args.outdir + '/' + model_name):
            os.makedirs(args.outdir + '/' + model_name)

        if os.path.exists(args.outdir + '/' + model_name + '/d.obj'):
            with open(args.outdir + '/' + model_name + '/d.obj', 'rb') as f:
                d = pickle.load(f)
        else:
            d = SyntheticModel(
                k,
                g
            )
            d.plot_irf(
                dir=args.outdir + '/' + model_name,
                n_time_units=args.ntimeunits,
                n_time_points=args.resolution,
                plot_x_inches=args.width,
                plot_y_inches=args.height,
                cmap=args.cmap,
                legend=False,
                xlab=args.xlab,
                ylab=args.ylab,
                use_line_markers=args.markers,
                transparent_background=args.transparent_background
            )
            with open(args.outdir + '/' + model_name + '/d.obj', 'wb') as f:
                pickle.dump(d, f)

        if X_interval is None:
            X_interval = 0.1
        else:
            try:
                X_interval = float(X_interval)
            except ValueError:
                X_interval_tmp = X_interval.split('-')
                name = X_interval_tmp[0]
                params = [float(x) for x in X_interval_tmp[1:]]
                X_interval = [name] + params
        if a:
            if y_interval is None:
                y_interval = 0.1
            else:
                try:
                    y_interval = float(y_interval)
                except ValueError:
                    y_interval_tmp = y_interval.split('-')
                    name = y_interval_tmp[0]
                    params = [float(x) for x in y_interval_tmp[1:]]
                    y_interval = [name] + params
        else:
            y_interval = X_interval

        X, t_X, t_y = d.sample_data(
            m,
            n,
            X_interval=X_interval,
            y_interval=y_interval,
            rho=rho,
            align_X_y=not a
        )

        X_conv, y = d.convolve(X, t_X, t_y, history_length=h, allow_instantaneous=not g == 'Gamma')

        for e in args.error:
            if e is None:
                err_sd = np.std(y)
            else:
                err_sd = e
            if err_sd:
                y_cur = y + np.random.normal(loc=0., scale=err_sd, size=y.shape)
            else:
                y_cur = y

            if isinstance(X_interval, list) or isinstance(X_interval, tuple):
                X_interval_name = '-'.join([str(x) for x in X_interval])
            else:
                X_interval_name = X_interval
            if isinstance(y_interval, list) or isinstance(y_interval, tuple):
                y_interval_name = '-'.join([str(x) for x in y_interval])
            else:
                y_interval_name = y_interval

            data_name = [
                'm%d' % m,
                'n%d' % n,
                'x%s' % X_interval_name,
                'y%s' % y_interval_name,
                'r%0.4f' % rho,
                'l%s' % h,
                'e%s' % ('None' if e is None else '%.4f' % e)
            ]
            if a:
                data_name.append('a')
            if partition is not None:
                data_name.append(partition)
            data_name = '_'.join(data_name)
            if not os.path.exists(args.outdir + '/' + model_name + '/' + data_name):
                os.makedirs(args.outdir + '/' + model_name + '/' + data_name)

            if not os.path.exists(args.outdir + '/' + model_name + '/' + data_name + '/X.evmeasures'):
                names = ['time', 'subject', 'sentid', 'docid'] + list(string.ascii_lowercase[:k])
                df_x = pd.DataFrame(np.concatenate([t_X[:,None], np.zeros((m, 3)), X], axis=1), columns=names)
                df_x.to_csv(args.outdir + '/' + model_name + '/' + data_name + '/X.evmeasures', ' ', index=False, na_rep='nan')

            if not os.path.exists(args.outdir + '/' + model_name + '/' + data_name + '/y.evmeasures'):
                names = ['time', 'subject', 'sentid', 'docid', 'y', 'true', 'se']
                df_y = pd.DataFrame(np.concatenate([t_y[:,None], np.zeros((n, 3)), y_cur[:, None], y[:, None], (y_cur[:, None] - y[:, None])**2], axis=1), columns=names)
                df_y.to_csv(args.outdir + '/' + model_name + '/' + data_name + '/y.evmeasures', ' ', index=False, na_rep='nan')
