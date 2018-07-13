import argparse
import sys
import os
import pandas as pd
from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import preprocess_data, filter_invalid_responses
from dtsr.util import load_dtsr

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Adds convolved columns to dataframe using pre-trained DTSR model
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-n', '--nsamples', type=int, default=None, help='Number of posterior samples to average (only used for DTSRBayes)')
    argparser.add_argument('-u', '--unscaled', action='store_true', help='Do not multiply outputs by DTSR-fitted coefficients')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Perform convolution using ablated models. Otherwise only convolves using the full model in each ablation set.')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    if not p.use_gpu_if_available:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in models if m.startswith('DTSR')]
    dtsr_models = [m for m in models if m.startswith('DTSR')]

    if not args.ablated_models:
        dtsr_models_new = []
        for model_name in dtsr_models:
            if len(model_name.split('!')) == 1: #No ablated variables, which are flagged with "!"
                dtsr_models_new.append(model_name)
        dtsr_models = dtsr_models_new

    if args.partition == 'train':
        X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'dev':
        X, y = read_data(p.X_dev, p.y_dev, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'test':
        X, y = read_data(p.X_test, p.y_test, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    else:
        raise ValueError('Unrecognized value for "partition" argument: %s' %args.partition)
    X, y, select, X_2d_predictor_names, X_2d_predictors = preprocess_data(
        X,
        y,
        p,
        dtsr_formula_list,
        compute_history=True
    )

    for m in dtsr_models:
        formula = p.models[m]['formula']

        dv = formula.strip().split('~')[0].strip()
        y_valid = filter_invalid_responses(y, dv)

        sys.stderr.write('Retrieving saved model %s...\n' % m)
        dtsr_model = load_dtsr(p.outdir + '/' + m)

        X_conv, X_conv_summary = dtsr_model.convolve_inputs(
            X,
            y_valid,
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            scaled=not args.unscaled,
            n_samples=args.nsamples,
            algorithm=args.algorithm
        )

        X_conv.to_csv(p.outdir + '/' + m + '/X_conv_%s.csv' %args.partition, sep=' ', index=False, na_rep='nan')

        sys.stderr.write(X_conv_summary)
        with open(p.outdir + '/' + m + '/X_conv_%s_summary.txt' %args.partition, 'w') as f:
            f.write(X_conv_summary)

        dtsr_model.finalize()

