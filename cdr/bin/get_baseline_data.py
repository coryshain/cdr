import argparse
import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
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
    argparser.add_argument('-o', '--outdir', default='./', help='Output directory')
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

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for i, p_name in enumerate(args.partition):
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

    if True:
        partition_name_to_ix = {'train': 0, 'dev': 1, 'test': 2}
        for i in range(len(evaluation_sets)):
            X, Y, select = evaluation_sets[i][:3]
            assert len(X) == 1, 'Cannot run baselines on asynchronously sampled predictors'
            assert len(Y) == 1, 'Cannot run baselines on multiple responses'

            X = X[0]
            Y = Y[0]
            partitions = evaluation_set_partitions[i]

            X_baseline = X

            X_baseline.time = X_baseline.time.round(3)
            Y.time = Y.time.round(3)
            if 'BOLD' in Y:
                common_cols = ['docid', 'fROI', 'hemisphere', 'network', 'sampleid', 'splitVal15', 'subject', 'time', 'tr']
            else:
                common_cols = sorted(list(set(X_baseline.columns) & set(Y.columns)))
            X_baseline = pd.merge(X_baseline, Y, on=common_cols, how='inner', suffixes=('', '_Y'))
            
            for m in models:
                if not m in cdr_formula_name_list:
                    p.set_model(m)
                    form = p['formula']
                    form_pieces = form.split('~')
                    lhs = form_pieces[0]
                    rhs = '~'.join(form_pieces[1:])
                    preds = rhs.split('+')
                    for pred in preds:
                        sp = spillover.search(pred)
                        if sp and sp.group(2) in X_baseline:
                            x_id = sp.group(2)
                            n = int(sp.group(3))
                            x_id_sp = x_id + 'S' + str(n)
                            if x_id_sp not in X_baseline:
                                X_baseline[x_id_sp] = X_baseline.groupby(p.series_ids)[x_id].shift(n, fill_value=0.)
                        z = zscorer.search(pred)
                        if z:
                            col = z.group(1)
                            if 'z_' + col not in X_baseline:
                                X_baseline['z_' + col] = (X_baseline[col] - X_baseline[col].mean()) / X_baseline[col].std()
            
            fdurs = [c for c in X_baseline if c.startswith('fdur')]
            for fdur in fdurs:
                X_baseline['log_' + fdur] = np.log(X_baseline[fdur])               

            cols = []
            for c in X_baseline:
                for m in models:
                    if not m in cdr_formula_name_list:
                        p.set_model(m)
                        form = p['formula']
                        if c in form or c.startswith('fdur') or c.startswith('log_fdur'):
                            cols.append(c)
                            break
            X_baseline = X_baseline[cols] 

            for c in X_baseline.columns:
                if X_baseline[c].dtype.name == 'category':
                    X_baseline[c] = X_baseline[c].astype(str)

            dataset = os.path.basename(evaluation_set_paths[i][0][0])[:-4]
            outpath = '%s/%s_%s.csv' % (args.outdir, dataset, evaluation_set_names[i])
            j = 2
            while os.path.exists(outpath):
                outpath = '%s/%s_%s_%s.csv' % (args.outdir, dataset, j, evaluation_set_names[i])
                j += 1

            X_baseline.to_csv(outpath, sep=' ', index=False, na_rep='NaN')

