import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from dtsr import Config, read_data, Formula, preprocess_data, print_tee, mse, mae, compute_splitID, compute_partition

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
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
        from dtsr.baselines import py2ri
        X['splitID'] = compute_splitID(X, p.split_ids)
        part = compute_partition(X, p.modulus, 3)
        if args.partition == 'train':
            part_select = part[0]
        elif args.partition == 'dev':
            part_select = part[1]
        elif args.partition == 'test':
            part_select = part[2]
        X_baseline = X[part_select]
        X_baseline = X_baseline.reset_index(drop=True)[select]
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
            with open(p.logdir + '/' + m + '/preds_%s.txt'%args.partition, 'w') as p_file:
                for i in range(len(lme_preds)):
                    p_file.write(str(lme_preds[i]) + '\n')
            if p.loss.lower() == 'mae':
                losses = np.array(y[dv] - lme_preds).abs()
            else:
                losses = np.array(y[dv] - lme_preds) ** 2
            with open(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), 'w') as p_file:
                for i in range(len(losses)):
                    p_file.write(str(losses[i]) + '\n')
            lme_mse = mse(y[dv], lme_preds)
            lme_mae = mae(y[dv], lme_preds)
            summary = '=' * 50 + '\n'
            summary += 'LME regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(lme.summary()) + '\n'
            summary += 'Loss (%s set):\n' % args.partition
            summary += '  MSE: %.4f\n' % lme_mse
            summary += '  MAE: %.4f\n' % lme_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/eval_%s.txt'%args.partition, 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')
        elif m.startswith('LM'):
            from dtsr.baselines import LM

            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                lm = pickle.load(m_file)

            lm_preds = lm.predict(X_baseline)
            with open(p.logdir + '/' + m + '/preds_%s.txt'%args.partition, 'w') as p_file:
                for i in range(len(lm_preds)):
                    p_file.write(str(lm_preds[i]) + '\n')
            if p.loss.lower() == 'mae':
                losses = np.array(y[dv] - lm_preds).abs()
            else:
                losses = np.array(y[dv] - lm_preds) ** 2
            with open(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), 'w') as p_file:
                for i in range(len(losses)):
                    p_file.write(str(losses[i]) + '\n')
            lm_mse = mse(y[dv], lm_preds)
            lm_mae = mae(y[dv], lm_preds)
            summary = '=' * 50 + '\n'
            summary += 'Linear regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(lm.summary()) + '\n'
            summary += 'Loss (%s set):\n' % args.partition
            summary += '  MSE: %.4f\n' % lm_mse
            summary += '  MAE: %.4f\n' % lm_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/eval_%s.txt' % args.partition, 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')
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
                    p_file.write(str(gam_preds[i]) + '\n')
            if p.loss.lower() == 'mae':
                losses = np.array(y[dv] - gam_preds).abs()
            else:
                losses = np.array(y[dv] - gam_preds) ** 2
            with open(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), 'w') as p_file:
                for i in range(len(losses)):
                    p_file.write(str(losses[i]) + '\n')
            gam_mse = mse(y[dv], gam_preds)
            gam_mae = mae(y[dv], gam_preds)
            summary = '=' * 50 + '\n'
            summary += 'GAM regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(lm.summary()) + '\n'
            summary += 'Loss (%s set):\n' % args.partition
            summary += '  MSE: %.4f\n' % gam_mse
            summary += '  MAE: %.4f\n' % gam_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/eval_%s.txt' % args.partition, 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')
        elif m.startswith('DTSR'):
            from dtsr import DTSR

            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                dtsr_model = pickle.load(m_file)

            dtsr_preds = dtsr_model.predict(X, y.time, y[dtsr_model.form.rangf], y.first_obs, y.last_obs)
            with open(p.logdir + '/' + m + '/preds_%s.txt'%args.partition, 'w') as p_file:
                for i in range(len(dtsr_preds)):
                    p_file.write(str(dtsr_preds[i]) + '\n')
            if p.loss.lower() == 'mae':
                losses = np.array(y[dv] - dtsr_preds).abs()
            else:
                losses = np.array(y[dv] - dtsr_preds) ** 2
            with open(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), 'w') as l_file:
                for i in range(len(losses)):
                    l_file.write(str(losses[i]) + '\n')
            dtsr_mse = mse(y[dv], dtsr_preds)
            dtsr_mae = mae(y[dv], dtsr_preds)
            summary = '=' * 50 + '\n'
            summary += 'DTSR regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += 'Loss (%s set):\n' % args.partition
            summary += '  MSE: %.4f\n' % dtsr_mse
            summary += '  MAE: %.4f\n' % dtsr_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/eval_%s.txt'%args.partition, 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')


