import os
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt

from cdr.config import Config
from cdr.util import get_irf_name

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot raw data patterns.')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    args = argparser.parse_args()

    nbin = 4

    for path in args.paths:
        dataset = os.path.basename(path)[:-4]
        p = Config(path)
        p.set_model('main.m0')
        responses = p['formula'].split(' ~ ')[0].strip().split(' + ')
        freq_name = get_irf_name('unigramsurpOWT', p.irf_name_map)
        pred_name = get_irf_name('gpt', p.irf_name_map)
       
        fig, ax = plt.subplots(figsize=(2, 2.5))
        fig.colorbar(ScalarMappable(Normalize(vmin=0, vmax=100), cmap='autumn_r'), ax=ax, label='%s percentile' % freq_name)
        ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(p.outdir, '%s_freq_cb.pdf' % os.path.basename(path)[:-4]))
        plt.close('all')

        fig, ax = plt.subplots(figsize=(2, 2.5))
        fig.colorbar(ScalarMappable(Normalize(vmin=0, vmax=100), cmap='winter_r'), ax=ax, label='%s percentile' % pred_name)
        ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(p.outdir, '%s_pred_cb.pdf' % os.path.basename(path)[:-4]))
        plt.close('all')

        df_paths = [os.path.join('lme_data', x) for x in os.listdir('lme_data') if x.startswith('%s_' % dataset)]
        df = pd.concat([pd.read_csv(df_path, sep=' ') for df_path in df_paths], axis=0)
        for response in responses:
            for main_pred, control_pred in (('unigramsurpOWT', 'gpt'), ('gpt', 'unigramsurpOWT')):
                ax = plt.gca()
                if main_pred == 'gpt':
                    cm = 'autumn_r'
                else:
                    cm = 'winter_r'
                cm = plt.get_cmap(cm)
                prop_cycle_map = np.arange(1, nbin + 1)
                n_colors = nbin + 2
                color_cycle = [cm(1. * prop_cycle_map[i] / n_colors) for i in range(len(prop_cycle_map))]
                ax.set_prop_cycle(color=color_cycle)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                _df = df[np.isfinite(df[response])]
                dv = _df[response]
                dv_name = get_irf_name(response, p.irf_name_map)
                main_name = get_irf_name(main_pred, p.irf_name_map)
                control_name = get_irf_name(control_pred, p.irf_name_map)
                main = _df[main_pred]
                control = _df[control_pred]
                control_ranges = np.nanquantile(control, np.linspace(0., 1., nbin + 1))
                control_ranges[0] = -np.inf
                control_ranges[-1] = np.inf
                control_ranges = np.stack([control_ranges[:-1], control_ranges[1:]], axis=1)
                q = np.nanquantile(main, np.linspace(0., 1., nbin + 1))
                positions = []
                for l, u in zip(q[:-1], q[1:]):
                    _main = main[(main >= l) & (main < u)]
                    positions.append(_main.median())
                positions = np.array(positions)
                # positions = np.stack([q[:-1], q[1:]], axis=1).mean(axis=1)
                _q = q.copy()
                _q[-1] = np.inf
                bins = np.digitize(main, _q)
                for _q in q[1:-1]:
                    plt.axvline(_q, color='k', alpha=0.25, linewidth=0.2)
                for i, control_range in enumerate(control_ranges):
                    ix = (control >= control_range[0]) & (control < control_range[1])
                    _dv = dv[ix]
                    _bins = bins[ix]
                    y = [_dv[_bins == j] for j in range(1, nbin + 1)]
                    y_med = np.array([np.nanmedian(_y) for _y in y])
                    y_lq = np.array([np.nanquantile(_y, 0.25) for _y in y])
                    y_uq = np.array([np.nanquantile(_y, 0.75) for _y in y])
               
                    plt.plot(positions, y_med, alpha=0.8, marker='o')
                    plt.plot(positions, y_lq, lw=1, alpha=0.6, linestyle='dotted', marker='', color=color_cycle[i]) 
                    plt.plot(positions, y_uq, lw=1, alpha=0.6, linestyle='dotted', marker='', color=color_cycle[i]) 
                plt.xlabel(main_name)
                plt.ylabel(dv_name)
                plt.gcf().set_size_inches(2.5, 2.5)
                plt.tight_layout()
                plt.savefig(os.path.join(p.outdir, '%s_%s_%s_raw.pdf' % (os.path.basename(path)[:-4], response, main_pred)))
                plt.close('all')

