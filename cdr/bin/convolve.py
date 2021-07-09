import argparse
import sys
import os
import pandas as pd
from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import preprocess_data, filter_invalid_responses
from cdr.util import load_cdr, filter_models, get_partition_list, paths_from_partition_cliarg, stderr

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Adds convolved columns to dataframe using pre-trained CDR model
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-n', '--nsamples', type=int, default=None, help='Number of posterior samples to average (only used for CDRBayes)')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Perform convolution using ablated models. Otherwise only convolves using the full model in each ablation set.')
    argparser.add_argument('-e', '--extra_cols', action='store_true', help='Whether to include columns from the response dataframe in the outputs.')
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args, unknown = argparser.parse_known_args()

    for path in args.config_paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models)

        cdr_formula_list = [Formula(p.models[m]['formula']) for m in models if (m.startswith('CDR') or m.startswith('DTSR'))]
        cdr_models = [m for m in models if (m.startswith('CDR') or m.startswith('DTSR'))]

        if not args.ablated_models:
            cdr_models_new = []
            for model_name in cdr_models:
                if len(model_name.split('!')) == 1: #No ablated variables, which are flagged with "!"
                    cdr_models_new.append(model_name)
            cdr_models = cdr_models_new

        evaluation_sets = []
        evaluation_set_partitions = []
        evaluation_set_names = []
        evaluation_set_paths = []

        for i, p_name in enumerate(args.partition):
            partitions = get_partition_list(p_name)
            if ':' in p_name:
                partition_str = '%d' % (i + 1)
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
                categorical_columns=list(
                    set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
            )
            X, Y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, \
            X_2d_predictors = preprocess_data(
                X,
                Y,
                cdr_formula_list,
                p.series_ids,
                filters=p.filters,
                compute_history=True,
                history_length=p.history_length
            )
            evaluation_sets.append((X, Y, select, X_response_aligned_predictor_names, X_response_aligned_predictors,
                                    X_2d_predictor_names, X_2d_predictors))
            evaluation_set_partitions.append(partitions)
            evaluation_set_names.append(partition_str)
            evaluation_set_paths.append((X_paths, Y_paths))

        for d in range(len(evaluation_sets)):
            X, Y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, \
            X_2d_predictors = evaluation_sets[d]
            partition_str = evaluation_set_names[d]

            for m in cdr_models:
                formula = p.models[m]['formula']
                m_path = m.replace(':', '+')

                dv = formula.strip().split('~')[0].strip()
                Y_valid, select_Y_valid = filter_invalid_responses(Y, dv)
                X_response_aligned_predictors_valid = X_response_aligned_predictors
                if X_response_aligned_predictors_valid is not None:
                    X_response_aligned_predictors_valid = X_response_aligned_predictors_valid[select_Y_valid]

                stderr('Retrieving saved model %s...\n' % m)
                cdr_model = load_cdr(p.outdir + '/' + m_path)

                stderr('Convolving %s...\n' % m)
                cdr_model.convolve_inputs(
                    X,
                    Y_valid,
                    X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                    X_response_aligned_predictors=X_response_aligned_predictors_valid,
                    X_2d_predictor_names=X_2d_predictor_names,
                    X_2d_predictors=X_2d_predictors,
                    n_samples=args.nsamples,
                    algorithm=args.algorithm,
                    extra_cols=args.extra_cols,
                    partition=partition_str,
                    dump=True
                )

                cdr_model.finalize()

