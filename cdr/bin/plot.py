import argparse
import sys
import os
import string
import pickle
import numpy as np
import pandas as pd
from cdr.config import Config
from cdr.util import load_cdr, filter_models, stderr
from cdr.plot import plot_irf, plot_qq, get_plot_config

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Plot estimates from saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names to plot. Regex permitted. If unspecified, plots all CDR models.')
    argparser.add_argument('-z', '--standardize_response', action='store_true', help='Standardize (Z-transform) response in plots. Ignored unless model was fitted using setting ``standardize_respose=True``.')
    argparser.add_argument('-S', '--summed', action='store_true', help='Plot summed rather than individual IRFs.')
    argparser.add_argument('-i', '--irf_ids', nargs='*', default = [], help='List of IDs for IRF to include in the plot. Regex supported.')
    argparser.add_argument('-U', '--unsorted_irf_ids', action='store_true', help='Leave IRF IDs unsorted (otherwise, they are sorted alphabetically).')
    argparser.add_argument('-d', '--plot_dirac', action='store_true', help='Also plot linear effects and interaction estimates (stick functions at 0).')
    argparser.add_argument('-R', '--plot_rangf', action='store_true', help='Also plot random effects estimates.')
    argparser.add_argument('-P', '--prop_cycle_length', type=int, default=None, help='Length of plotting properties cycle (defines step size in the color map). If unspecified, inferred from **irf_names**.')
    argparser.add_argument('-I', '--prop_cycle_ix', nargs='*', type=int, default=None, help='Integer indices to use in the properties cycle for each entry in **irf_names**. If unspecified, indices are automatically assigned.')
    argparser.add_argument('-s', '--plot_true_synthetic', action='store_true', help='If the models are fit to synthetic data, also generate plots of the true IRF')
    argparser.add_argument('-t', '--reference_time', type=float, default=0., help='Reference time to use for CDRNN plots.')
    argparser.add_argument('-p', '--prefix', type=str, default=None, help='Filename prefix to use for outputs. If unspecified, creates prefix based on output path.')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=None, help='Number of time units on x-axis')
    argparser.add_argument('-r', '--resolution', type=float, default=None, help='Number of points on x-axis')
    argparser.add_argument('-x', '--x', type=float, default=None, help='Width of plot in inches')
    argparser.add_argument('-X', '--xlab', type=str, default=None, help='x-axis label (if default -- None -- no label)')
    argparser.add_argument('-y', '--y', type=float, default=None, help='Height of plot in inches')
    argparser.add_argument('-Y', '--ylab', type=str, default=None, help='y-axis label (if default -- None -- no label)')
    argparser.add_argument('-b', '--ylim', type=float, nargs=2, default=None, help='Fixed ylim value to use for all IRF plots. If unspecified, automatically inferred.')
    argparser.add_argument('-c', '--cmap', type=str, default=None, help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-D', '--dpi', type=int, default=None, help='Dots per inch')
    argparser.add_argument('-l', '--nolegend', action='store_true', help='Omit legend from figure')
    argparser.add_argument('-q', '--qq', type=str, default=None, help='Generate Q-Q plot for errors over partition ``qq``. Ignored unless model directory contains saved errors for the requested partition.')
    argparser.add_argument('-Q', '--qq_axis_labels', action='store_true', help='Add axis labels to Q-Q plots.')
    argparser.add_argument('-T', '--qq_noticks', action='store_true', help='Remove ticks from Q-Q plots.')
    argparser.add_argument('-L', '--qq_nolegend', action='store_true', help='Omit legend from Q-Q plots')
    argparser.add_argument('-M', '--markers', action='store_true', help='Add markers to IRF lines')
    argparser.add_argument('-B', '--transparent_background', action='store_true', help='Use transparent background (otherwise white background)')
    argparser.add_argument('-C', '--dump_source', action='store_true', help='Dump plot source arrays to CSV')
    argparser.add_argument('-g', '--config_path', type=str, help='Path to config file specifying plot arguments')
    args = argparser.parse_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, cdr_only=True)

        prefix = args.prefix
        if prefix is None:
            prefix = '_'.join(p.outdir.split('/'))
        if prefix != '':
            prefix += '_'
        n_time_units = args.ntimeunits
        reference_time = args.reference_time
        resolution = args.resolution
        x_inches = args.x
        y_inches = args.y
        cmap = args.cmap
        name_map = p.irf_name_map
        legend = not args.nolegend

        synth_path = os.path.dirname(os.path.dirname(p.X_train)) + '/d.obj'
        if args.plot_true_synthetic and os.path.exists(synth_path):
            with open(synth_path, 'rb') as f:
                d = pickle.load(f)
            x, y = d.get_curves(
                n_time_units=p['plot_n_time_units'] if n_time_units is None else n_time_units,
                n_time_points=p['plot_n_time_points'] if resolution is None else resolution
            )
            names = string.ascii_lowercase[:d.n_pred]

            if args.summed:
                plot_irf(
                    x,
                    y.sum(axis=1, keepdims=True),
                    ['Sum'],
                    prop_cycle_length=args.prop_cycle_length,
                    prop_cycle_ix=args.prop_cycle_ix,
                    dir=p.outdir,
                    filename=prefix + 'synthetic_true_summed.png',
                    plot_x_inches=p['plot_x_inches'] if x_inches is None else x_inches,
                    plot_y_inches=p['plot_y_inches'] if y_inches is None else y_inches,
                    ylim=args.ylim,
                    cmap=p['cmap'] if cmap is None else cmap,
                    dpi=args.dpi,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    use_line_markers=args.markers,
                    transparent_background=args.transparent_background,
                    dump_source=args.dump_source
                )

            else:
                plot_irf(
                    x,
                    y,
                    names,
                    sort_names=not args.unsorted_irf_ids,
                    prop_cycle_length=args.prop_cycle_length,
                    prop_cycle_ix=args.prop_cycle_ix,
                    dir=p.outdir,
                    filename=prefix + 'synthetic_true.png',
                    plot_x_inches=p['plot_x_inches'] if x_inches is None else x_inches,
                    plot_y_inches=p['plot_y_inches'] if y_inches is None else y_inches,
                    ylim=args.ylim,
                    cmap=p['cmap'] if cmap is None else cmap,
                    dpi=args.dpi,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    use_line_markers=args.markers,
                    transparent_background=args.transparent_background,
                    dump_source=args.dump_source
                )

        for m in models:
            m_path = m.replace(':', '+')
            p.set_model(m)

            stderr('Retrieving saved model %s...\n' % m)
            cdr_model = load_cdr(p.outdir + '/' + m_path)

            kwargs = {
                'plot_n_time_units': p['plot_n_time_units'] if n_time_units is None else n_time_units,
                'plot_n_time_points': p['plot_n_time_points'] if resolution is None else resolution,
                'surface_plot_n_time_points': p['surface_plot_n_time_points'] if resolution is None else resolution,
                'reference_time': p['reference_time'] if reference_time is None else reference_time,
                'plot_x_inches': p['plot_x_inches'] if x_inches is None else x_inches,
                'plot_y_inches': p['plot_y_inches'] if y_inches is None else y_inches,
                'cmap': p['cmap'] if cmap is None else cmap,
                'dpi': p['dpi'] if args.dpi is None else args.dpi,
                'generate_irf_surface_plots': p['generate_irf_surface_plots'],
                'generate_interaction_surface_plots': p['generate_interaction_surface_plots'],
                'generate_curvature_plots': p['generate_curvature_plots'],
            }

            stderr('Plotting...\n')

            if args.qq:
                obs_path = p.outdir + '/%s/obs_%s.txt' % (m_path, args.qq)
                preds_path = p.outdir + '/%s/preds_%s.txt' % (m_path, args.qq)
                has_obs = os.path.exists(obs_path)
                has_preds = os.path.exists(preds_path)
                if has_obs and has_preds:
                    y_obs = np.array(pd.read_csv(obs_path, header=None))[:,0]
                    y_preds = np.array(pd.read_csv(preds_path, header=None))[:,0]
                    err = np.sort(y_obs - y_preds)
                    err_theoretical_q = cdr_model.error_theoretical_quantiles(len(err))
                    valid = np.isfinite(err_theoretical_q)
                    err = err[valid]
                    err_theoretical_q = err_theoretical_q[valid]

                    if args.qq_axis_labels:
                        xlab = 'Theoretical'
                        ylab = 'Empirical'
                    else:
                        xlab = None
                        ylab = None

                    if y_inches:
                        if args.qq_noticks:
                            qq_x_inches = y_inches
                        else:
                            qq_x_inches = y_inches + 0.5
                        qq_y_inches = y_inches
                    else:
                        if args.qq_noticks:
                            qq_x_inches = p['plot_y_inches']
                        else:
                            qq_x_inches = p['plot_y_inches'] + 0.5
                        qq_y_inches = p['plot_y_inches']

                    qq_kwargs = {
                        'plot_x_inches': qq_x_inches,
                        'plot_y_inches': qq_y_inches,
                        'dpi': p['dpi'] if args.dpi is None else args.dpi
                    }

                    plot_qq(
                        err_theoretical_q,
                        err,
                        dir=cdr_model.outdir,
                        filename=prefix + '%s_error_qq_plot_%s.png' % (m_path, args.qq),
                        xlab=xlab,
                        ylab=ylab,
                        legend=not args.qq_nolegend,
                        ticks=not args.qq_noticks,
                        **qq_kwargs
                    )
                else:
                    stderr('Model %s missing observation and/or prediction files, skipping Q-Q plot...\n' % m)

            cdr_model.make_plots(
                standardize_response=args.standardize_response,
                summed=args.summed,
                irf_name_map=name_map,
                irf_ids=args.irf_ids,
                sort_names=not args.unsorted_irf_ids,
                prop_cycle_length=args.prop_cycle_length,
                prop_cycle_ix=args.prop_cycle_ix,
                plot_unscaled=False,
                plot_dirac=args.plot_dirac,
                plot_rangf=args.plot_rangf,
                ylim=args.ylim,
                prefix=prefix + m_path,
                legend=legend,
                xlab=args.xlab,
                ylab=args.ylab,
                use_line_markers=args.markers,
                transparent_background=args.transparent_background,
                dump_source=args.dump_source,
                **kwargs
            )
            

            if cdr_model.is_bayesian or cdr_model.has_dropout:
                cdr_model.make_plots(
                    standardize_response=args.standardize_response,
                    summed=args.summed,
                    irf_name_map=name_map,
                    irf_ids=args.irf_ids,
                    sort_names=not args.unsorted_irf_ids,
                    prop_cycle_length=args.prop_cycle_length,
                    prop_cycle_ix=args.prop_cycle_ix,
                    plot_unscaled=False,
                    plot_dirac=args.plot_dirac,
                    plot_rangf=args.plot_rangf,
                    mc=True,
                    ylim=args.ylim,
                    prefix=prefix + m_path,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    use_line_markers=args.markers,
                    transparent_background=args.transparent_background,
                    dump_source=args.dump_source,
                    **kwargs
                )

            cdr_model.finalize()



