import argparse
import sys
import os
import pandas as pd
from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import preprocess_data, filter_invalid_responses
from cdr.model import CDREnsemble
from cdr.util import filter_models, get_partition_list, paths_from_partition_cliarg, stderr

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Resample data from a fitted CDR posterior
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-r', '--response', nargs='+', default=None, help='Names of response variables to convolve toward. If ``None``, convolves toward all variables.')
    argparser.add_argument('-n', '--nsamples', type=int, default=None, help='Number of posterior samples to average (only used for CDRBayes)')
    argparser.add_argument('-e', '--extra_cols', action='store_true', help='Whether to include columns from the response dataframe in the outputs.')
    argparser.add_argument('-O', '--optimize_memory', action='store_true', help="Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).")
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    for path in args.config_paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_list = sorted(set(p.model_list) | set(p.ensemble_list))
        models = filter_models(model_list, args.models, cdr_only=True)

        cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(models, cdr_only=True)]
        cdr_models = [m for m in filter_models(models, cdr_only=True)]

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

            for m in cdr_models:
                formula = p.models[m]['formula']
                m_path = m.replace(':', '+')

                dv = [x.strip() for x in formula.strip().split('~')[0].strip().split('+')]
                Y_valid, select_Y_valid = filter_invalid_responses(Y, dv)

                stderr('Retrieving saved model %s...\n' % m)
                cdr_model = CDREnsemble(p.outdir, m_path)

                cdr_model.set_weight_type('uniform')

                stderr('Sampling %s...\n' % m)
                cdr_model.sample(
                    X,
                    Y_valid,
                    X_in_Y_names=X_in_Y_names,
                    responses=args.response,
                    n_samples=args.nsamples,
                    extra_cols=args.extra_cols,
                    partition=partition_str,
                    optimize_memory=args.optimize_memory,
                    dump=True
                )

                cdr_model.finalize()
