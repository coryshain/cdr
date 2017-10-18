import argparse
import os
import sys
import pickle
import pandas as pd

pd.options.mode.chained_assignment = None

from dtsr import Config, read_data, DTSR, Formula, preprocess_data, print_tee, mse, mae, c

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type='str', default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    run_baseline = False
    run_dtsr = False
    for m in models:
        if not run_baseline and m.startswith('LM') or m.startswith('GAM'):
            run_baseline = True
        elif not run_dtsr and m.startswith('DTSR'):
            run_dtsr = True
            break

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in p.model_list if m.startswith('DTSR')]
    dtsr_formula_name_list = [m for m in p.model_list if m.startswith('DTSR')]
    if args.partition == 'train':
        X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'dev':
        X, y = read_data(p.X_dev, p.y_dev, p.series_ids, categorical_columns=list(set(p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'test':
        X, y = read_data(p.X_test, p.y_test, p.series_ids, categorical_columns=list(set(p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    else:
        raise ValueError('Unrecognized value for "partition" argument: %s' %args.parition)
    X, y, select = preprocess_data(X, y, p, dtsr_formula_list, compute_history=run_dtsr)

    if run_baseline:
        X_baseline = X[select]

    if run_baseline:
        from dtsr.baselines import py2ri
        X_baseline = py2ri(X_baseline)

    for m in models:
        formula = p.models[m]['formula']
        if not os.path.exists(p.logdir + '/' + m):
            os.makedirs(p.logdir + '/' + m)
        if m.startswith('LME'):
            from dtsr.baselines import LME

            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                lme = pickle.load(m_file)

            lme_preds = lme.predict(X_baseline)
            with open(p.logdir + '/' + m + '/preds.txt', 'w') as p_file:
                for i in range(len(lme_preds)):
                    p_file.write(str(lme_preds[i]))
        elif m.startswith('LM'):
            from dtsr.baselines import LM

            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                lm = pickle.load(m_file)

            lm_preds = lm.predict(X_baseline)
            with open(p.logdir + '/' + m + '/preds.txt', 'w') as p_file:
                for i in range(len(lm_preds)):
                    p_file.write(str(lm_preds[i]))
        elif m.startswith('GAM'):
            import re
            from dtsr.baselines import GAM

            dv = formula.strip().split('~')[0].strip()

            ## For some reason, GAM can't predict using custom functions, so we have to translate them
            z_term = re.compile('z.\((.*)\)')
            c_term = re.compile('c.\((.*)\)')
            formula = [t.strip() for t in formula.strip().split() if t.strip() != '']
            for i in range(len(formula)):
                formula[i] = z_term.sub(r'scale(\1)', formula[i])
                formula[i] = c_term.sub(r'scale(\1, scale=FALSE)', formula[i])
            formula = ' '.join(formula)

            sys.stderr.write('Retrieving saved model %s...\n\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                gam = pickle.load(m_file)
            gam_preds = gam.predict(X_baseline)
            with open(p.logdir + '/' + m + '/preds.txt', 'w') as p_file:
                for i in range(len(gam_preds)):
                    p_file.write(str(gam_preds[i]))
        elif m.startswith('DTSR'):
            model = DTSR(formula,
                         X,
                         y,
                         outdir=p.logdir + '/' + m,
                         fixef_name_map=p.fixef_name_map)


