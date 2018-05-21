import argparse
import sys
import os
import numpy as np
import pandas as pd
from dtsr.config import Config
from dtsr.util import load_dtsr
from dtsr.bin.synth import read_params, convolve
from dtsr.plot import plot_irf

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names to plot (if unspecified, plots all DTSR models)')
    argparser.add_argument('-s', '--plot_true_synthetic', action='store_true', help='If the models are fit to synthetic data, also generate plots of the true IRF')
    argparser.add_argument('-p', '--prefix', type=str, default='', help='Filename prefix to use for outputs')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=2.5, help='Number of time units on x-axis')
    argparser.add_argument('-r', '--resolution', type=float, default=500, help='Number of points per time unit')
    argparser.add_argument('-x', '--x', type=float, default=6, help='Width of plot in inches')
    argparser.add_argument('-X', '--xlab', type=str, default=None, help='x-axis label (if default -- None -- no label)')
    argparser.add_argument('-y', '--y', type=float, default=4, help='Height of plot in inches')
    argparser.add_argument('-Y', '--ylab', type=str, default=None, help='y-axis label (if default -- None -- no label)')
    argparser.add_argument('-c', '--cmap', type=str, default='gist_rainbow', help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-l', '--nolegend', action='store_true', help='Omit legend from figure')
    argparser.add_argument('-t', '--transparent_background', action='store_true', help='Use transparent background (otherwise white background)')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)
        if len(args.models) > 0:
            models = [m for m in args.models if (m in p.model_list and m.startswith('DTSR'))]
        else:
            models = p.model_list[:]
        prefix = args.prefix
        if prefix != '':
            prefix += '_'
        n_time_units = args.ntimeunits
        resolution = args.resolution
        x_inches = args.x
        y_inches = args.y
        cmap = args.cmap
        name_map = p.irf_name_map
        legend = not args.nolegend

        synth_path = os.path.dirname(p.X_train) + '/params.evmeasures'
        if args.plot_true_synthetic and os.path.exists(synth_path):
            params = read_params(synth_path)

            x = np.linspace(0, n_time_units, resolution * n_time_units)

            y = convolve(x, np.expand_dims(params.k, -1), np.expand_dims(params.theta, -1), np.expand_dims(params.delta, -1), coefficient=np.expand_dims(params.beta, -1))
            y = y.transpose([1, 0])

            names = list(params.index)

            plot_irf(
                x,
                y,
                names,
                dir=p.outdir,
                filename=prefix + 'synthetic_true.png',
                plot_x_inches=x_inches,
                plot_y_inches=y_inches,
                cmap=cmap,
                legend=legend,
                xlab=args.xlab,
                ylab=args.ylab,
                transparent_background=args.transparent_background
            )

        for m in models:

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            dtsr_model.make_plots(
                irf_name_map=name_map,
                plot_n_time_units=n_time_units,
                plot_n_points_per_time_unit=resolution,
                plot_x_inches=x_inches,
                plot_y_inches=y_inches,
                cmap=cmap,
                prefix=prefix + m,
                legend=legend,
                xlab=args.xlab,
                ylab=args.ylab,
                transparent_background=args.transparent_background
            )
            if hasattr(dtsr_model, 'inference_name'):
                dtsr_model.make_plots(
                    irf_name_map=name_map,
                    plot_n_time_units=n_time_units,
                    plot_n_points_per_time_unit=resolution,
                    plot_x_inches=x_inches,
                    plot_y_inches=y_inches,
                    cmap=cmap,
                    mc=True,
                    prefix=prefix + m,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    transparent_background=args.transparent_background
                )



