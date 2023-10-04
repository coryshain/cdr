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
        Adds convolved columns to dataframe using pre-trained CDR model
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file(s)')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names from which to convolve. Regex permitted. If unspecified, predicts from all models.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-r', '--response', nargs='+', default=None, help='Names of response variables to convolve toward. If ``None``, convolves toward all variables.')
    argparser.add_argument('-P', '--response_param', nargs='+', default=None, help='Names of any parameters of predictive distribution(s) to convolve toward. If ``None``, convolves toward the first parameter of the predictive distribution for each resposne.')
    argparser.add_argument('-n', '--nsamples', type=int, default=None, help='Number of posterior samples to average (only used for CDRBayes)')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Perform convolution using ablated models. Otherwise only convolves using the full model in each ablation set.')
    argparser.add_argument('-e', '--extra_cols', action='store_true', help='Whether to include columns from the response dataframe in the outputs.')
    argparser.add_argument('-O', '--optimize_memory', action='store_true', help="Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).")
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    for path in args.config_paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_names = filter_models(p.model_names, args.models, cdr_only=True)

        cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(model_names, cdr_only=True)]
        cdr_models = [m for m in filter_models(model_names, cdr_only=True)]

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
        training_set = None
        training_set_paths = None

        for i, p_name in enumerate(args.partition):
            if p_name in ('CVdev', 'CVtest'):
                if training_set is None:
                    X_paths, Y_paths = paths_from_partition_cliarg('train', p)
                    X, Y = read_tabular_data(
                        X_paths,
                        Y_paths,
                        p.series_ids,
                        sep=p.sep,
                        categorical_columns=list(
                            set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
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
                    training_set = (X, Y, select, X_in_Y_names)
                    training_set_paths = (X_paths, Y_paths)
                evaluation_sets.append(training_set)
                evaluation_set_partitions.append(p_name)
                evaluation_set_names.append(p_name)
                evaluation_set_paths.append(training_set_paths)
            else:
                partitions = get_partition_list(p_name)
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

                stderr('Retrieving saved model %s...\n' % m)
                cdr_model = CDREnsemble(p.outdir, m_path)

                dv = formula.strip().split('~')[0].strip()
                if partition_str in ('CVdev', 'CVtest'):
                    assert cdr_model.use_crossval, 'Model %s was fitted without cross-validation and cannot predict over partition %s.' % (
                    m, partition_str)
                    if partition_str == 'CVdev':
                        assert cdr_model.crossval_dev_fold is not None, 'Model %s did not use a cross-validation dev fold and cannot predict over partition %s.' % (
                        m, partition_str)
                        _Y = [_Y[_Y[cdr_model.crossval_factor] == cdr_model.crossval_dev_fold] for _Y in Y]
                    else:  # partition_str == 'CVtest'
                        _Y = [_Y[_Y[cdr_model.crossval_factor] == cdr_model.crossval_fold] for _Y in Y]
                else:
                    _Y = Y
                Y_valid, select_Y_valid = filter_invalid_responses(_Y, dv)

                if args.algorithm.lower() == 'map':
                    cdr_model.set_weight_type('ll')
                else:
                    cdr_model.set_weight_type('uniform')

                stderr('Convolving %s...\n' % m)
                cdr_model.convolve_inputs(
                    X,
                    Y_valid,
                    X_in_Y_names=X_in_Y_names,
                    responses=args.response,
                    response_params=args.response_param,
                    n_samples=args.nsamples,
                    algorithm=args.algorithm,
                    extra_cols=args.extra_cols,
                    partition=partition_str,
                    optimize_memory=args.optimize_memory,
                    dump=True
                )

                cdr_model.finalize()
