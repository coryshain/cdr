import argparse
import os
import sys
import pickle
import pandas as pd

pd.options.mode.chained_assignment = None

from dtsr import Config, read_data, Formula, preprocess_data, print_tee, mse, mae, compute_splitID, compute_partition

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Trains model(s) from formula string(s) given data.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Path to configuration (*.ini) file')
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
    X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    X, y, select = preprocess_data(X, y, p, dtsr_formula_list, compute_history=run_dtsr)

    if run_baseline:
        from dtsr.baselines import py2ri
        X['splitID'] = compute_splitID(X, p.split_ids)
        part = compute_partition(X, p.modulus, 3)
        part_select = part[0]
        X_baseline = X[part_select]
        X_baseline = X_baseline.reset_index(drop=True)[select]

    n_train_sample = len(y)

    sys.stderr.write('\nNumber of training samples: %d\n' %n_train_sample)

    for i in range(len(dtsr_formula_list)):
        x = dtsr_formula_list[i]
        name = dtsr_formula_name_list[i]
        if run_baseline and x.dv not in X_baseline.columns:
            X_baseline[x.dv] = y[x.dv]
        sys.stderr.write('Correlation matrix for DTSR model %s:\n' %name)
        rho = X[x.allsl].corr()
        sys.stderr.write(str(rho) + '\n\n')

    if run_baseline:
        for c in X_baseline.columns:
            if X_baseline[c].dtype.name == 'category':
                X_baseline[c] = X_baseline[c].astype(str)
        X_baseline = py2ri(X_baseline)

    for m in models:
        formula = p.models[m]['formula']
        if not os.path.exists(p.logdir + '/' + m):
            os.makedirs(p.logdir + '/' + m)
        if m.startswith('LME'):
            from dtsr.baselines import LME

            dv = formula.strip().split('~')[0].strip()

            if os.path.exists(p.logdir + '/' + m + '/m.obj'):
                sys.stderr.write('Retrieving saved model %s...\n' % m)
                with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                    lme = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting model %s...\n' % m)
                lme = LME(formula, X_baseline)
                with open(p.logdir + '/' + m + '/m.obj', 'wb') as m_file:
                    pickle.dump(lme, m_file)

            lme_preds = lme.predict(X_baseline)
            lme_mse = mse(y[dv], lme_preds)
            lme_mae = mae(y[dv], lme_preds)
            summary = '=' * 50 + '\n'
            summary += 'LME regression\n\n'
            summary += 'Model name: %s\n\n' %m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(lme.summary()) + '\n'
            summary += 'Training set loss:\n'
            summary += '  MSE: %.4f\n' % lme_mse
            summary += '  MAE: %.4f\n' % lme_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/summary.txt', 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')
        elif m.startswith('LM'):
            from dtsr.baselines import LM

            dv = formula.strip().split('~')[0].strip()

            if os.path.exists(p.logdir + '/' + m + '/m.obj'):
                sys.stderr.write('Retrieving saved model %s...\n' % m)
                with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                    lm = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting model %s...\n' % m)
                lm = LM(formula, X_baseline)
                with open(p.logdir + '/' + m + '/m.obj', 'wb') as m_file:
                    pickle.dump(lm, m_file)

            lm_preds = lm.predict(X_baseline)
            lm_mse = mse(y[dv], lm_preds)
            lm_mae = mae(y[dv], lm_preds)
            summary = '=' * 50 + '\n'
            summary += 'Linear regression\n\n'
            summary += 'Model name: %s\n\n' %m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(lm.summary()) + '\n'
            summary += 'Training set loss:\n'
            summary += '  MSE: %.4f\n' % lm_mse
            summary += '  MAE: %.4f\n' % lm_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/summary.txt', 'w') as f_out:
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

            if os.path.exists(p.logdir + '/' + m + '/m.obj'):
                sys.stderr.write('Retrieving saved model %s...\n' % m)
                with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                    gam = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting model %s...\n' % m)
                gam = GAM(formula, X_baseline)
                with open(p.logdir + '/' + m + '/m.obj', 'wb') as m_file:
                    pickle.dump(gam, m_file)

            gam_preds = gam.predict(X_baseline)
            gam_mse = mse(y[dv], gam_preds)
            gam_mae = mae(y[dv], gam_preds)
            summary = '=' * 50 + '\n'
            summary += 'GAM regression\n\n'
            summary += 'Model name: %s\n\n' %m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(gam.summary()) + '\n'
            summary += 'Training set loss:\n'
            summary += '  MSE: %.4f\n' % gam_mse
            summary += '  MAE: %.4f\n' % gam_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/summary.txt', 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')
        elif m.startswith('DTSR'):
            from dtsr.dtsr import DTSR

            if not p.use_gpu_if_available:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''

            dv = formula.strip().split('~')[0].strip()

            print(p.irf)
            if os.path.exists(p.logdir + '/' + m + '/m.obj'):
                sys.stderr.write('Retrieving saved model %s...\n\n' % m)
                with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                    dtsr_model = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting model %s...\n\n' % m)
                dtsr_model = DTSR(formula,
                                  y,
                                  outdir=p.logdir + '/' + m,
                                  irf=p.irf,
                                  learning_rate=p.learning_rate
                                  )
                with open(p.logdir + '/' + m + '/m.obj', 'wb') as m_file:
                    pickle.dump(dtsr_model, m_file)
            dtsr_model.train(X,
                             y,
                             n_epoch_train=p.n_epoch_train,
                             n_epoch_tune=p.n_epoch_tune,
                             minibatch_size=p.minibatch_size,
                             fixef_name_map=p.fixef_name_map,
                             plot_x_inches=p.plot_x_inches,
                             plot_y_inches=p.plot_y_inches,
                             cmap=p.cmap
                             )

            dtsr_preds = dtsr_model.predict(X, y.time, y[dtsr_model.form.rangf], y.first_obs, y.last_obs)
            dtsr_mse = mse(y[dv], dtsr_preds)
            dtsr_mae = mae(y[dv], dtsr_preds)
            summary = '=' * 50 + '\n'
            summary += 'DTSR regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += 'Training set loss:\n'
            summary += '  MSE: %.4f\n' % dtsr_mse
            summary += '  MAE: %.4f\n' % dtsr_mae
            summary += '=' * 50 + '\n'
            with open(p.logdir + '/' + m + '/summary.txt', 'w') as f_out:
                print_tee(summary, [sys.stdout, f_out])
            sys.stderr.write('\n\n')


