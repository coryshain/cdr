import argparse
import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None

from cdr.kwargs import MODEL_INITIALIZATION_KWARGS
from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import filter_invalid_responses, preprocess_data, compute_splitID, compute_partition
from cdr.util import mse, mae, filter_models, get_partition_list, paths_from_partition_cliarg, stderr


def plot_density(d, outdir, out_name, frac=1, color=None, size=None):
    if color is None:
        color = (0.5, 0.5, 0.5, 0.5)
    if size is None:
        size = (2, 1)
    d = d[np.isfinite(d)]
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    if isinstance(frac, tuple):
        xmin, xmax = frac
    else:
        qmin = (1 - frac) / 2
        qmax = frac + qmin
        xmin, xmax = np.quantile(d, [qmin, qmax])
    d = d[(d >= xmin) & (d <= xmax)]
    p = sns.kdeplot(d, bw_adjust=2, color=color, fill=True)
    plt.xlim((xmin, xmax))
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.gcf().set_size_inches(size)
    plt.savefig(os.path.join(outdir, out_name + '.pdf'))
    plt.close('all')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generate density plots from data.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--prefix', default='', help='Prefix to append to output files')
    argparser.add_argument('-o', '--outdir', default='surp_densities', help='Path to output directory')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Path to configuration (*.ini) file')
    args = argparser.parse_args()

    p = Config(args.config_path)
    prefix = args.prefix

    models = filter_models(p.model_list, args.models)

    cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(models, cdr_only=True)]
    cdr_formula_name_list = [m for m in filter_models(p.model_list)]
    all_rangf = [v for x in cdr_formula_list for v in x.rangf]
    partitions = get_partition_list('train')
    all_interactions = False

    X_paths, Y_paths = paths_from_partition_cliarg(partitions, p)
    X_paths_dev = Y_paths_dev = X_dev = Y_dev = None
    X, Y = read_tabular_data(
        X_paths,
        Y_paths,
        p.series_ids,
        sep=p.sep,
        categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
    )
    X, Y, select, X_in_Y_names = preprocess_data(
        X,
        Y,
        cdr_formula_list,
        p.series_ids,
        filters=p.filters,
        history_length=p.history_length,
        future_length=p.future_length,
        t_delta_cutoff=p.t_delta_cutoff,
        all_interactions=all_interactions
    )
    
    surps = ['cloze', 'ngram', 'totsurp', 'gpt', 'gptj', 'gpt3']
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for surp in surps:
        stderr('Plotting %s\n' % surp)
        for suffix in ('', 'prob', 'squared'):
            col = surp + suffix
            for frac in (1, 0.8):
                if prefix:
                    out_name = prefix + '_' + col + '_' + str(frac)
                else:
                    out_name = col + '_' + str(frac)
                if col == 'cloze':
                    _col = 'clozesurp'
                elif col.endswith('squared'):
                    _col = col[:-7]
                else:
                    _col = col
                if _col in X[0]:
                    d = X[0][_col]
                    if col.endswith('squared'):
                        d = d ** 2
                    plot_density(d, args.outdir, out_name, frac=frac)

    others = ['wlen', 'unigramsurp']
    for col in others:
        stderr('Plotting %s\n' % col)
        for frac in (1, 0.8):
            if prefix:
                out_name = prefix + '_' + col + '_' + str(frac)
            else:
                out_name = col + '_' + str(frac)
            if col in X[0]:
                d = X[0][col]
                plot_density(d, args.outdir, out_name, frac=frac)
        
