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
from cdr.model import CDREnsemble
from cdr.util import filter_models, get_partition_list, paths_from_partition_cliarg, stderr, sn
from cdr.plot import plot_irf

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names from which to predict. Regex permitted. If unspecified, predicts from all models.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-n', '--nsamples', type=int, default=1024, help='Number of posterior samples to average (only used for CDRBayes)')
    argparser.add_argument('-M', '--mode', default='eval', help='Evaluation mode(s), either "predict" (to just generate predictions) or "eval" (to evaluate predictions, compute likelihoods, etc).')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions from CDRBayes. Ignored for CDRMLE.')
    argparser.add_argument('-O', '--optimize_memory', action='store_true', help="Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).")
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    p = Config(args.config_path)

    model_list = sorted(set(p.model_list) | set(p.ensemble_list))
    models = filter_models(model_list, args.models, cdr_only=True)

    model_cache = {}
    model_cache_twostep = {}

    cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(models, cdr_only=True)]
    cdr_formula_name_list = [m for m in filter_models(p.model_list, cdr_only=True)]

    evaluation_sets = []
    evaluation_set_partitions = []
    evaluation_set_names = []
    evaluation_set_paths = []

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

    for d in range(len(evaluation_sets)):
        X, Y, select, X_in_Y_names = evaluation_sets[d]
        partition_str = evaluation_set_names[d]

        for m in models:
            formula = p.models[m]['formula']
            p.set_model(m)
            m_path = m.replace(':', '+')
            if not os.path.exists(p.outdir + '/' + m_path):
                os.makedirs(p.outdir + '/' + m_path)
            with open(p.outdir + '/' + m_path + '/pred_inputs_%s.txt' % partition_str, 'w') as f:
                f.write('%s\n' % (' '.join(evaluation_set_paths[d][0])))
                f.write('%s\n' % (' '.join(evaluation_set_paths[d][1])))

            if m in model_cache:
                _model = model_cache[m]
            else:
                stderr('Retrieving saved model %s...\n' % m)
                _model = CDREnsemble(p.outdir, m_path)
                model_cache[m] = _model

            if not p.use_gpu_if_available or args.cpu_only:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            dv = [x.strip() for x in formula.strip().split('~')[0].strip().split('+')]
            if _model.use_crossval:
                crossval_factor = _model.crossval_factor
                crossval_fold = _model.crossval_fold
            else:
                crossval_factor = None
                crossval_fold = None

            Y_valid, select_Y_valid = filter_invalid_responses(Y, dv)

            cdr_mse = {}
            cdr_corr = {}
            cdr_f1 = {}
            cdr_loglik = {}
            cdr_loss = {}
            cdr_percent_variance_explained = {}
            cdr_true_variance = {}

            metrics, _ = _model.evaluate(
                X,
                Y_valid,
                X_in_Y_names=X_in_Y_names,
                n_samples=args.nsamples,
                algorithm=args.algorithm,
                sum_outputs_along_T=False,
                sum_outputs_along_K=True,
                partition=partition_str,
                optimize_memory=args.optimize_memory
            )

            x = np.arange(-_model.future_length, _model.history_length)
            y = []
            names = []
            for metric in [x for x in metrics if x in ['mse', 'f1', 'acc']]:
                name = metric
                for response in [x for x in metrics[metric] if x in _model.response_names]:
                    name += ', %s' % response
                    for ix in metrics[metric][response]:
                        name += ', f%d' % ix
                        if metrics[metric][response][ix] is not None:
                            y.append(metrics[metric][response][ix][::-1, 0]) # Reverse time
                            names.append(name)
            if len(y):
                y = np.stack(y, axis=-1)
                plot_irf(x, y, names, filename=_model.outdir + '/performance_%s.png' % partition_str)
