import argparse
import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score

pd.options.mode.chained_assignment = None

from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import add_responses, filter_invalid_responses, preprocess_data, compute_splitID, compute_partition, s, c, z, split_cdr_outputs
from cdr.util import mse, mae, percent_variance_explained
from cdr.util import filter_models, get_partition_list, paths_from_partition_cliarg, stderr, sn
from cdr.plot import plot_qq


spillover = re.compile('(z_)?([^ (),]+)S([0-9]+)')
zscorer = re.compile('z_([^ ()\|+,]+)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Save data from a partition to a table for fitting baselines
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names from which to extract data. Regex permitted. If unspecified, predicts from all models.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-o', '--outdir', default='./lme_data/', help='Output directory')
    args = argparser.parse_args()

    p = Config(args.config_path)

    model_list = sorted(set(p.model_list) | set(p.ensemble_list))
    models = filter_models(model_list, args.models)

    cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(models, cdr_only=True)]
    cdr_formula_name_list = [m for m in filter_models(p.model_list, cdr_only=True)]

    evaluation_sets = []
    evaluation_set_partitions = []
    evaluation_set_names = []
    evaluation_set_paths = []

    impulses = set(['docid', 'sentid', 'sentpos', 'time'])
    for m in models:
        p.set_model(m)
        form = Formula(p['formula'])
        _impulses = form.t.impulse_names()
        _responses = form.response_names()
        _rangf = form.rangf
        impulses |= set(_impulses + _responses + _rangf)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    partition = ['train-dev-test', 'train-dev', 'test']

    for i, p_name in enumerate(partition):
        partitions = get_partition_list(p_name)
        if ':' in p_name:
            partition_str = 'p%d' % (i + 1)
            X_paths = partitions[0].split(';')
            Y_paths = partitions[1].split(';')
        else:
            partition_str = '-'.join(partitions)
            X_paths, Y_paths = paths_from_partition_cliarg(partitions, p)
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
            t_delta_cutoff=p.t_delta_cutoff
        )
        evaluation_sets.append((X, Y, select, X_in_Y_names))
        evaluation_set_partitions.append(partitions)
        evaluation_set_names.append(partition_str)
        evaluation_set_paths.append((X_paths, Y_paths))

    partition_name_to_ix = {'train': 0, 'dev': 1, 'test': 2}
    for i in range(len(evaluation_sets)):
        X, Y, select = evaluation_sets[i][:3]
        assert len(X) == 1, 'Cannot run baselines on asynchronously sampled predictors'
        assert len(Y) == 1, 'Cannot run baselines on multiple responses'

        X = X[0]
        X_cols = [x for x in X.columns if is_numeric_dtype(X[x]) or x in ('sentid', 'sentpos')]
        for S in range(1, 2):
            for col in X_cols:
                X[col + '_S%s' % S] = X.groupby(p.series_ids)[col].shift(S).astype(float).fillna(0)
        Y = Y[0]
        partitions = evaluation_set_partitions[i]

        X_baseline = X

        X_baseline.time = X_baseline.time.round(3)
        Y.time = Y.time.round(3)
        if 'BOLD' in Y:
            common_cols = ['docid', 'fROI', 'hemisphere', 'network', 'sampleid', 'splitVal15', 'subject', 'time', 'tr']
        else:
            common_cols = ['subject', 'docid', 'sentid', 'sentpos', 'time']
            # common_cols = sorted(list(set(X_baseline.columns) & set(Y.columns)))
        X_cols = (set(X_baseline.columns) - set(Y.columns)) | set(common_cols)
        X_baseline = pd.merge(X_baseline[X_cols], Y, on=common_cols, how='inner', suffixes=('', '_Y'))
       
        fdurs = [c for c in X_baseline if c.startswith('fdur')]
        for fdur in fdurs:
            X_baseline['log_' + fdur] = np.log(X_baseline[fdur])               

        cols = []
        cols_set = set()
        for c in X_baseline:
            for impulse in impulses:
                if c.startswith(impulse) and c not in cols_set:
                    cols.append(c)
                    cols_set.add(c)
        X_baseline = X_baseline[cols]

        for c in X_baseline.columns:
            if X_baseline[c].dtype.name == 'category':
                X_baseline[c] = X_baseline[c].astype(str)

        dataset = os.path.basename(evaluation_set_paths[i][0][0]).split('_')[0]
        outpath = '%s/%s_%s.csv' % (args.outdir, dataset, evaluation_set_names[i])

        X_baseline = X_baseline.rename(lambda s: re.sub('[^0-9a-zA-Z]+', '_', s), axis=1)
        X_baseline.to_csv(outpath, sep=' ', index=False, na_rep='NaN')

