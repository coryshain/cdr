import argparse
import sys
import os
import string
import pickle
from dtsr.config import Config
from dtsr.util import load_dtsr, filter_models
from dtsr.plot import plot_irf

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names to plot. Regex permitted. If unspecified, plots all DTSR models.')
    argparser.add_argument('-z', '--standardize_response', action='store_true', help='Standardize (Z-transform) response in plots. Ignored unless model was fitted using setting ``standardize_respose=True``.')
    argparser.add_argument('-S', '--summed', action='store_true', help='Plot summed rather than individual IRFs.')
    argparser.add_argument('-i', '--irf_ids', nargs='*', default = [], help='List of IDs for IRF to include in the plot. Regex supported.')
    argparser.add_argument('-U', '--unsorted_irf_ids', action='store_true', help='Leave IRF IDs unsorted (otherwise, they are sorted alphabetically).')
    argparser.add_argument('-d', '--plot_dirac', action='store_true', help='Also plot linear effects and interaction estimates (stick functions at 0).')
    argparser.add_argument('-P', '--prop_cycle_length', type=int, default=None, help='Length of plotting properties cycle (defines step size in the color map). If unspecified, inferred from **irf_names**.')
    argparser.add_argument('-I', '--prop_cycle_ix', nargs='*', type=int, default=None, help='Integer indices to use in the properties cycle for each entry in **irf_names**. If unspecified, indices are automatically assigned.')
    argparser.add_argument('-s', '--plot_true_synthetic', action='store_true', help='If the models are fit to synthetic data, also generate plots of the true IRF')
    argparser.add_argument('-p', '--prefix', type=str, default='', help='Filename prefix to use for outputs')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=None, help='Number of time units on x-axis')
    argparser.add_argument('-r', '--resolution', type=float, default=None, help='Number of points on x-axis')
    argparser.add_argument('-x', '--x', type=float, default=None, help='Width of plot in inches')
    argparser.add_argument('-X', '--xlab', type=str, default=None, help='x-axis label (if default -- None -- no label)')
    argparser.add_argument('-y', '--y', type=float, default=None, help='Height of plot in inches')
    argparser.add_argument('-Y', '--ylab', type=str, default=None, help='y-axis label (if default -- None -- no label)')
    argparser.add_argument('-c', '--cmap', type=str, default=None, help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-l', '--nolegend', action='store_true', help='Omit legend from figure')
    argparser.add_argument('-M', '--markers', type=float, default=None, help='Size of line markers in pt. If ``0`` or ``None``, no markers.')
    argparser.add_argument('-t', '--transparent_background', action='store_true', help='Use transparent background (otherwise white background)')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, dtsr_only=True)

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
                    cmap=p['cmap'] if cmap is None else cmap,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    line_markers=args.markers,
                    transparent_background=args.transparent_background
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
                    cmap=p['cmap'] if cmap is None else cmap,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    line_markers=args.markers,
                    transparent_background=args.transparent_background
                )

        for m in models:
            p.set_model(m)

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            kwargs = {
                'plot_n_time_units': p['plot_n_time_units'] if n_time_units is None else n_time_units,
                'plot_n_time_points': p['plot_n_time_points'] if resolution is None else resolution,
                'plot_x_inches': p['plot_x_inches'] if x_inches is None else x_inches,
                'plot_y_inches': p['plot_y_inches'] if y_inches is None else y_inches,
                'cmap': p['cmap'] if cmap is None else cmap
            }

            dtsr_model.make_plots(
                standardize_response=args.standardize_response,
                summed=args.summed,
                irf_name_map=name_map,
                irf_ids=args.irf_ids,
                sort_names=not args.unsorted_irf_ids,
                prop_cycle_length=args.prop_cycle_length,
                prop_cycle_ix=args.prop_cycle_ix,
                plot_dirac=args.plot_dirac,
                prefix=prefix + m,
                legend=legend,
                xlab=args.xlab,
                ylab=args.ylab,
                line_markers=args.markers,
                transparent_background=args.transparent_background,
                **kwargs
            )
            if hasattr(dtsr_model, 'inference_name'):
                dtsr_model.make_plots(
                    standardize_response=args.standardize_response,
                    summed=args.summed,
                    irf_name_map=name_map,
                    irf_ids=args.irf_ids,
                    sort_names=not args.unsorted_irf_ids,
                    prop_cycle_length=args.prop_cycle_length,
                    prop_cycle_ix=args.prop_cycle_ix,
                    plot_dirac=args.plot_dirac,
                    mc=True,
                    prefix=prefix + m,
                    legend=legend,
                    xlab=args.xlab,
                    ylab=args.ylab,
                    line_markers=args.markers,
                    transparent_background=args.transparent_background,
                    **kwargs
                )

            dtsr_model.finalize()



