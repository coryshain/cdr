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

N_SAMPLES = 10

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot examples of model fit.')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    args = argparser.parse_args()

    true_color = 'gray'
    lme_color = 'blue'
    cdrnn_color = 'orange'

    for path in args.paths:
        dataset = os.path.basename(path)[:-4]
        p = Config(path)
        p.set_model('main.m0')
        responses = p['formula'].split(' ~ ')[0].strip().split(' + ')
        df = pd.read_csv('lme_data/%s_test.csv' % dataset, sep=' ')
        for response in responses:
            _df = df[np.isfinite(df[response])].reset_index(drop=True)[['subject', 'sentid', 'docid', 'time', response]]
            CDRpreds = pd.read_csv(
                '../results/cdrnn_freq_pred_owt/{dataset}/main/Ysamp_{response}__test.csv'.format(
                    dataset=dataset,
                    response=response
                ),
                sep=' '
            )
            cols = [x for x in CDRpreds if x.startswith('CDRsamp')]
            CDRpreds = CDRpreds[cols]
            CDRpreds = np.nanmedian(CDRpreds, axis=1)
#            CDRpreds = []
#            for i in range(10):
#                df_cdr = pd.read_csv(
#                    os.path.join(p.outdir, 'main.m%d' % i, 'output_%s_test.csv' % response),
#                    sep= ' '
#                )
#                _CDRpreds = df_cdr.CDRpreds.values
#                CDRpreds.append(_CDRpreds)
#            CDRpreds = np.nanmedian(np.stack(CDRpreds, axis=1), axis=1)
            _df['CDRpreds'] = CDRpreds
            LMEpreds = pd.read_csv(
                '../results/cdrnn_freq_pred_owt/lme/{dataset}/{dataset}_{response}_interaction_insample_output.csv'.format(
                    dataset=dataset,
                    response=response
                )
            )
            _df['LMEpreds'] = LMEpreds.LMEpred
            timeseries = list(_df[['subject', 'docid']].drop_duplicates().itertuples(index=False))
            sample_ix = np.random.permutation(np.arange(len(timeseries)))[:N_SAMPLES]
            sample_timeseries = [timeseries[ix] for ix in sample_ix]
            for s, sample in enumerate(sample_timeseries):
                __df = _df[(_df.subject == sample.subject) & (_df.docid == sample.docid)]
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.tick_params(top='off', bottom='off', left='on', right='off', labelleft='on', labelbottom='off')
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                for i, (_, ___df) in enumerate(__df.groupby('sentid')):
                    if i == 0:
                        true_label = 'True'
                        lme_label = 'LMER'
                        cdrnn_label = 'CDRNN'
                    else:
                        true_label = lme_label = cdrnn_label = None
                    t = ___df.index + i
                    plt.plot(t, ___df[response], label=true_label, color=true_color)
                    plt.plot(t, ___df.LMEpreds, label=lme_label, color=lme_color)
                    plt.plot(t, ___df.CDRpreds, label=cdrnn_label, color=cdrnn_color)
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
                    'ncol': 3
                }
                ax.legend(**legend_kwargs)
                plt.gcf().set_size_inches((6, 4))
                plt.tight_layout()
                plt.savefig(os.path.join(p.outdir, 'main', '%s_%s_fit_sample%d.pdf' % (dataset, response, s + 1)))
                plt.close('all')


