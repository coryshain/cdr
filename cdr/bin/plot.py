import argparse
import sys
import os
import string
import pickle
import numpy as np
import pandas as pd
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from cdr.config import Config, PlotConfig
from cdr.kwargs import plot_kwarg_docstring
from cdr.model import CDREnsemble
from cdr.util import filter_models, stderr
from cdr.plot import plot_irf, plot_qq

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Plot estimates from saved model(s).
        
        %s
    ''' % plot_kwarg_docstring())
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-c', '--plot_config_path', default=None, help='Path to config file specifying plot settings. To initialize an annotated plot config file, run ``python -m cdr.bin.create_config -t plot -a``.')
    argparser.add_argument('-k', '--kwarg_path', default=None, help='Path to optional YAML file specifying additionl keyword arguments to ``get_plot_data``.')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names to plot. Regex permitted. If unspecified, plots all CDR models.')
    argparser.add_argument('-d', '--dump_source', action='store_true', help='Dump plot source arrays to CSV')
    argparser.add_argument('-C', '--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    plot_config = PlotConfig(args.plot_config_path)
    if args.kwarg_path:
        with open(args.kwarg_path, 'r') as f:
            extra_kwargs = yaml.load(f, Loader=Loader)
    else:
        extra_kwargs = {}
    qq = plot_config.get('qq_partition', None)
    qq_use_axis_labels = plot_config.get('qq_use_axis_labels', True)
    qq_use_ticks = plot_config.get('qq_use_ticks', False)
    qq_use_legend = plot_config.get('qq_use_legend', False)

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_list = sorted(set(p.model_names) | set(p.ensemble_names) | set(p.crossval_family_names))
        models = filter_models(model_list, args.models, cdr_only=True)

        prefix = plot_config['prefix']
        if prefix is None:
            prefix = '_'.join([x for x in p.outdir.split('/') if x not in ['.', '..']])
        key = plot_config.get('key', None)

        n_time_units = plot_config['plot_n_time_units']
        reference_time = plot_config['reference_time']
        resolution = plot_config['plot_n_time_points']
        plot_x_inches = plot_config['plot_x_inches']
        plot_y_inches = plot_config['plot_y_inches']
        ylim = plot_config['ylim']
        cmap = plot_config['cmap']
        dpi = plot_config['dpi']
        name_map = p.irf_name_map
        legend = plot_config['use_legend']
        markers = plot_config['use_line_markers']
        transparent_background = plot_config['transparent_background']
        plot_true_synthetic = plot_config['plot_true_synthetic']
        sort_names =  plot_config['sort_names']
        prop_cycle_length = plot_config['prop_cycle_length']
        prop_cycle_map = plot_config['prop_cycle_map']

        plot_kwargs = {x: plot_config[x] for x in plot_config.settings_core if x not in ('prefix', 'key',)}
        plot_kwargs.update(extra_kwargs)

        synth_path = os.path.dirname(os.path.dirname(p.X_train[0])) + '/d.obj'
        if plot_true_synthetic and os.path.exists(synth_path):
            with open(synth_path, 'rb') as f:
                d = pickle.load(f)
            x, y = d.get_curves(
                n_time_units=n_time_units,
                n_time_points=resolution
            )
            names = string.ascii_lowercase[:d.n_pred]

            plot_irf(
                x,
                y,
                names,
                sort_names=sort_names,
                prop_cycle_length=prop_cycle_length,
                prop_cycle_map=prop_cycle_map,
                outdir=p.outdir,
                filename=prefix + 'synthetic_true.png',
                plot_x_inches=plot_x_inches,
                plot_y_inches=plot_y_inches,
                ylim=ylim,
                cmap=cmap,
                dpi=dpi,
                legend=legend,
                use_line_markers=markers,
                transparent_background=transparent_background,
                dump_source=args.dump_source
            )

        for m in models:
            m_path = m.replace(':', '+')
            p.set_model(m)

            stderr('Retrieving saved model %s...\n' % m)
            cdr_model = CDREnsemble(p.outdir, m_path)

            stderr('Plotting...\n')

            if prefix:
                prefix_cur = prefix + '_' + m
            else:
                prefix_cur = m

            if key:
                prefix_cur += '_' + key

            if qq:
                obs_path = p.outdir + '/%s/obs_%s.txt' % (m_path, qq)
                preds_path = p.outdir + '/%s/preds_%s.txt' % (m_path, qq)
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

                    if qq_use_axis_labels:
                        xlab = 'Theoretical'
                        ylab = 'Empirical'
                    else:
                        xlab = None
                        ylab = None

                    if qq_use_ticks:
                        qq_x_inches = plot_x_inches + 0.5
                        qq_y_inches = plot_y_inches + 0.5
                    else:
                        qq_x_inches = plot_x_inches
                        qq_y_inches = plot_y_inches

                    qq_kwargs = {
                        'plot_x_inches': qq_x_inches,
                        'plot_y_inches': qq_y_inches,
                        'dpi': dpi
                    }

                    plot_qq(
                        err_theoretical_q,
                        err,
                        outdir=cdr_model.outdir,
                        filename=prefix_cur + '%s_error_qq_plot_%s.png' % (m_path, qq),
                        xlab=xlab,
                        ylab=ylab,
                        legend=qq_use_legend,
                        ticks=qq_use_ticks,
                        **qq_kwargs
                    )
                else:
                    stderr('Model %s missing observation and/or prediction files, skipping Q-Q plot...\n' % m)

            mc = (cdr_model.is_bayesian or cdr_model.has_dropout)
            if 'n_samples' in plot_kwargs:
                mc &= bool(plot_kwargs['n_samples'])
            if mc:
                cdr_model.set_weight_type('uniform')
            else:
                cdr_model.set_weight_type('ll')

            cdr_model.make_plots(prefix=prefix_cur, dump_source=args.dump_source, **plot_kwargs)

            cdr_model.finalize()



