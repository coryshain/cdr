import os
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from cdr.config import Config
from cdr.util import get_irf_name

N_SAMPLES = 10

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot examples of model fit.')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    args = argparser.parse_args()

    assert os.path.exists('config.yml'), 'Repository has not yet been initialized. First run `python -m initialize`.'
    with open('config.yml', 'r') as f:
        repo_cfg = yaml.load(f, Loader=Loader)
    results_dir = repo_cfg['results_dir']

    true_color = 'gray'
    cdrnn_color = 'orange'

    for path in args.paths:
        dataset = os.path.basename(path)[:-4]
        p = Config(path)
        p.set_model('main.m0')
        responses = p['formula'].split(' ~ ')[0].strip().split(' + ')
        for response in responses:
            df = pd.read_csv(
                '{results_dir}/{dataset}/main/Ysamp_{response}__test.csv'.format(
                    results_dir=results_dir,
                    dataset=dataset,
                    response=response
                ),
                sep=' '
            )
            samp_cols = [x for x in df if x.startswith('CDRsamp')]
            sample = df[samp_cols].values

            # Distribution overlay
            y = df[response]
            y = y[y < np.nanquantile(y, 0.95)]
            y_true, bins, _ = plt.hist(y, bins=100, density=True, label='True', color=true_color, alpha=0.5)
            y_true *= 0.95  # 5% of the data was dropped
            x_true = bins
            x_true = np.stack([x_true[:-1], x_true[1:]], axis=1).mean(axis=1)
            y_pred, x_pred, _ = plt.hist(sample.flatten(), bins=bins, density=True, label='CDRNN', color=cdrnn_color, alpha=0.5)
            x_pred = np.stack([x_pred[:-1], x_pred[1:]], axis=1).mean(axis=1)
            plt.close('all')
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            plt.plot(x_true, y_true, label='True', color=true_color, alpha=0.8)
            plt.plot(x_pred, y_pred, label='CDRNN', color=cdrnn_color, alpha=0.8)
            legend_kwargs = {
                'fancybox': True,
                'framealpha': 0.75,
                'frameon': False,
                'facecolor': 'white',
                'edgecolor': 'gray',
                'loc': 'lower center',
                'bbox_to_anchor': (0.5, 1),
                'ncol': 2
            }
            # ax.legend(**legend_kwargs)
            plt.xlim(y.min(), y.max())
            plt.ylabel('$p$')
            plt.xlabel(get_irf_name(response, p.irf_name_map))
            plt.gcf().set_size_inches((2.5, 2.5))
            plt.tight_layout()
            plt.savefig(os.path.join(p.outdir, 'main', '%s_%s_hist.pdf' % (dataset, response)))

            # Sample true-pred timecourses
            df['CDRpreds'] = np.nanmedian(sample, axis=1)
            obs = df[response]
            timeseries = list(df[['subject', 'docid']].drop_duplicates().itertuples(index=False))
            sample_ix = np.random.permutation(np.arange(len(timeseries)))[:N_SAMPLES]
            sample_timeseries = [timeseries[ix] for ix in sample_ix]
            for s, sample in enumerate(sample_timeseries):
                _df = df[(df.subject == sample.subject) & (df.docid == sample.docid)]
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.tick_params(top='off', bottom='off', left='on', right='off', labelleft='on', labelbottom='off')
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                for i, (_, __df) in enumerate(_df.groupby('sentid')):
                    if i == 0:
                        true_label = 'True'
                        cdrnn_label = 'CDRNN'
                    else:
                        true_label = lme_label = cdrnn_label = None
                    t = __df.index + i * 3
                    plt.plot(t, __df[response], label=true_label, color=true_color)
                    plt.plot(t, __df.CDRpreds, label=cdrnn_label, color=cdrnn_color)
                plt.ylabel(get_irf_name(response, p.irf_name_map))
                plt.xticks([])
                legend_kwargs = {
                    'fancybox': True,
                    'framealpha': 0.75,
                    'frameon': False,
                    'facecolor': 'white',
                    'edgecolor': 'gray',
                    'loc': 'lower center',
                    'bbox_to_anchor': (0.5, 1),
                    'ncol': 2
                }
                ax.legend(**legend_kwargs)
                plt.gcf().set_size_inches((6, 4))
                plt.tight_layout()
                plt.savefig(os.path.join(p.outdir, 'main', '%s_%s_fit_sample%d.pdf' % (dataset, response, s + 1)))
                plt.close('all')


