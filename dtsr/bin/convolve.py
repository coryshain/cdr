import argparse
import sys
import pickle
import pandas as pd
from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import preprocess_data
from dtsr.dtsr import load_dtsr

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Adds convolved columns to dataframe using pre-trained DTSR model
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-n', '--nsamples', type=int, default=1024, help='Number of posterior samples to average (only used for BDTSR)')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in p.model_list if m.startswith('DTSR')]
    dtsr_formula_name_list = [m for m in p.model_list if m.startswith('DTSR')]

    if args.partition == 'train':
        X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'dev':
        X, y = read_data(p.X_dev, p.y_dev, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'test':
        X, y = read_data(p.X_test, p.y_test, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    else:
        raise ValueError('Unrecognized value for "partition" argument: %s' %args.partition)
    X, y, select = preprocess_data(X, y, p, dtsr_formula_list, compute_history=True)

    for m in dtsr_formula_name_list:
        formula = p.models[m]['formula']

        dv = formula.strip().split('~')[0].strip()

        sys.stderr.write('Retrieving saved model %s...\n' % m)
        dtsr_model = load_dtsr(p.logdir + '/' + m)

        X_conv = dtsr_model.convolve_inputs(X, y, scaled=False, n_samples=args.nsamples)
        X_conv.to_csv(p.logdir + '/' + m + '/X_conv.csv', sep=' ', index=False, na_rep='nan')


