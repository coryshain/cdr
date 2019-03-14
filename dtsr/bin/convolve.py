import argparse
import sys
import os
import pandas as pd
from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import preprocess_data, filter_invalid_responses
from dtsr.util import load_dtsr, filter_models, get_partition_list, paths_from_partition_cliarg

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Adds convolved columns to dataframe using pre-trained DTSR model
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-d', '--data', nargs='+', default=[], help='Pairs of paths or buffers <predictors, responses>, allowing convolution of arbitrary data. Predictors may consist of ``;``-delimited paths designating files containing predictors with different timestamps. Within each file, timestamps are treated as identical across predictors.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='Name of partition to use ("train", "dev", "test", or hyphen-delimited subset of these). Ignored if **data** is provided.')
    argparser.add_argument('-n', '--nsamples', type=int, default=None, help='Number of posterior samples to average (only used for DTSRBayes)')
    argparser.add_argument('-u', '--unscaled', action='store_true', help='Do not multiply outputs by DTSR-fitted coefficients')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Perform convolution using ablated models. Otherwise only convolves using the full model in each ablation set.')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    if not p.use_gpu_if_available:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    models = filter_models(p.model_list, args.models)

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in models if m.startswith('DTSR')]
    dtsr_models = [m for m in models if m.startswith('DTSR')]

    if not args.ablated_models:
        dtsr_models_new = []
        for model_name in dtsr_models:
            if len(model_name.split('!')) == 1: #No ablated variables, which are flagged with "!"
                dtsr_models_new.append(model_name)
        dtsr_models = dtsr_models_new

    evaluation_sets = []
    evaluation_set_partitions = []
    evaluation_set_names = []
    evaluation_set_paths = []

    for p_name in args.partition:
        partitions = get_partition_list(p_name)
        partition_str = '-'.join(partitions)
        X_paths, y_paths = paths_from_partition_cliarg(partitions, p)
        X, y = read_data(X_paths, y_paths, p.series_ids, categorical_columns=list(
            set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
        X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = preprocess_data(
            X,
            y,
            dtsr_formula_list,
            p.series_ids,
            filter_map=p.filter_map,
            compute_history=True,
            history_length=p.history_length
        )
        evaluation_sets.append((X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors,
                                X_2d_predictor_names, X_2d_predictors))
        evaluation_set_partitions.append(partitions)
        evaluation_set_names.append(partition_str)
        evaluation_set_paths.append((X_paths, y_paths))

    assert len(args.data) % 2 == 0, 'Argument ``data`` must be a list with an even number of elements.'
    for i in range(0, len(args.data), 2):
        partition_str = '%d' % (int(i / 2) + 1)
        X_paths, y_paths = args.data[i:i + 2]
        X, y = read_data(X_paths, y_paths, p.series_ids, categorical_columns=list(
            set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
        X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = preprocess_data(
            X,
            y,
            dtsr_formula_list,
            p.series_ids,
            filter_map=p.filter_map,
            compute_history=True,
            history_length=p.history_length
        )
        evaluation_sets.append((X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors,
                                X_2d_predictor_names, X_2d_predictors))
        evaluation_set_partitions.append(None)
        evaluation_set_names.append(partition_str)
        evaluation_set_paths.append((X_paths, y_paths))

    for d in range(len(evaluation_sets)):
        X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = evaluation_sets[d]
        partition_str = evaluation_set_names[d]

        for m in dtsr_models:
            formula = p.models[m]['formula']

            dv = formula.strip().split('~')[0].strip()
            y_valid, select_y_valid = filter_invalid_responses(y, dv)
            X_response_aligned_predictors_valid = X_response_aligned_predictors
            if X_response_aligned_predictors_valid is not None:
                X_response_aligned_predictors_valid = X_response_aligned_predictors_valid[select_y_valid]

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            X_conv, X_conv_summary = dtsr_model.convolve_inputs(
                X,
                y_valid,
                X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                X_response_aligned_predictors=X_response_aligned_predictors_valid,
                X_2d_predictor_names=X_2d_predictor_names,
                X_2d_predictors=X_2d_predictors,
                scaled=not args.unscaled,
                n_samples=args.nsamples,
                algorithm=args.algorithm
            )

            X_conv.to_csv(p.outdir + '/' + m + '/X_conv_%s.csv' %partition_str, sep=' ', index=False, na_rep='nan')

            sys.stderr.write(X_conv_summary)
            with open(p.outdir + '/' + m + '/X_conv_%s_summary.txt' %partition_str, 'w') as f:
                f.write(X_conv_summary)

            dtsr_model.finalize()

