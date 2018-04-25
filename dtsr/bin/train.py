import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import preprocess_data, compute_splitID, compute_partition
from dtsr.util import mse, mae, r_squared, load_dtsr

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Trains model(s) from formula string(s) given data.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Path to configuration (*.ini) file')
    argparser.add_argument('-e', '--eval', default=1, type=int, help='Whether to evaluate the trained DTSR models before exiting.')
    args, unknown = argparser.parse_known_args()

    eval = bool(args.eval)

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
    X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    X, y, select = preprocess_data(X, y, p, dtsr_formula_list, compute_history=run_dtsr)
    #
    # from matplotlib import pyplot as plt
    # plt.scatter(X.totsurp, X.fwprob5surp)
    # plt.show()
    # print(X.totsurp.std())
    # print(X.fwprob5surp.std())
    # exit()

    if run_baseline:
        X['splitID'] = compute_splitID(X, p.split_ids)
        part = compute_partition(X, p.modulus, 3)
        part_select = part[0]
        X_baseline = X[part_select]
        X_baseline = X_baseline.reset_index(drop=True)[select]

    n_train_sample = len(y)

    sys.stderr.write('\nNumber of training samples: %d\n' %n_train_sample)

    for i in range(len(dtsr_formula_list)):
        x = dtsr_formula_list[i]
        if run_baseline and x.dv not in X_baseline.columns:
            X_baseline[x.dv] = y[x.dv]

    if run_baseline:
        from dtsr.baselines import py2ri
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

            dv = formula.strip().split('~')[0].strip().replace('.','')

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
                f_out.write(summary)
            sys.stderr.write(summary)
            sys.stderr.write('\n\n')

        elif m.startswith('LM'):
            from dtsr.baselines import LM

            dv = formula.strip().split('~')[0].strip().replace('.','')

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
                f_out.write(summary)
            sys.stderr.write(summary)
            sys.stderr.write('\n\n')

        elif m.startswith('GAM'):
            import re
            from dtsr.baselines import GAM

            dv = formula.strip().split('~')[0].strip().replace('.','')

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
                f_out.write(summary)
            sys.stderr.write(summary)
            sys.stderr.write('\n\n')

        elif m.startswith('DTSR'):
            if not p.use_gpu_if_available:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Fitting model %s...\n\n' % m)

            if p.network_type == 'nn':
                bayes = False
            else:
                bayes = True

            if os.path.exists(p.logdir + '/m.obj'):
                dtsr_model = load_dtsr(p.logdir)
            elif p.network_type == 'nn':
                from dtsr.nndtsr import NNDTSR
                dtsr_model = NNDTSR(
                    formula,
                    X,
                    y,
                    outdir=p.logdir + '/' + m,
                    history_length=p.history_length,
                    low_memory=p.low_memory,
                    pc=p.pc,
                    float_type=p.float_type,
                    int_type=p.int_type,
                    minibatch_size=p.minibatch_size,
                    eval_minibatch_size=p.eval_minibatch_size,
                    n_interp=p.n_interp,
                    log_random=p.log_random,
                    log_freq=p.log_freq,
                    save_freq=p.save_freq,
                    optim=p.optim,
                    learning_rate=p.learning_rate,
                    learning_rate_min=p.learning_rate_min,
                    lr_decay_family=p.lr_decay_family,
                    lr_decay_steps=p.lr_decay_steps,
                    lr_decay_rate=p.lr_decay_rate,
                    lr_decay_staircase=p.lr_decay_staircase,
                    init_sd=p.init_sd,
                    ema_decay=p.ema_decay,
                    loss=p.loss,
                    regularizer=p.regularizer,
                    regularizer_scale=p.regularizer_scale
                )
            elif p.network_type.startswith('bayes'):
                from dtsr.bdtsr import BDTSR
                dtsr_model = BDTSR(
                    formula,
                    X,
                    y,
                    outdir=p.logdir + '/' + m,
                    history_length=p.history_length,
                    low_memory=p.low_memory,
                    pc=p.pc,
                    float_type=p.float_type,
                    int_type=p.int_type,
                    minibatch_size=p.minibatch_size,
                    eval_minibatch_size=p.eval_minibatch_size,
                    n_interp=p.n_interp,
                    inference_name=p.inference_name,
                    n_samples=p.n_samples,
                    n_samples_eval=p.n_samples_eval,
                    n_iter=p.n_iter,
                    log_random=p.log_random,
                    log_freq=p.log_freq,
                    save_freq=p.save_freq,
                    optim=p.optim,
                    learning_rate=p.learning_rate,
                    learning_rate_min=p.learning_rate_min,
                    lr_decay_family=p.lr_decay_family,
                    lr_decay_steps=p.lr_decay_steps,
                    lr_decay_rate=p.lr_decay_rate,
                    lr_decay_staircase=p.lr_decay_staircase,
                    intercept_prior_sd=p.intercept_prior_sd,
                    coef_prior_sd=p.coef_prior_sd,
                    conv_prior_sd=p.conv_prior_sd,
                    mv=p.mv,
                    mv_ran=p.mv_ran,
                    y_scale_fixed=p.y_scale,
                    y_scale_prior_sd=p.y_scale_prior_sd,
                    init_sd=p.init_sd,
                    ema_decay=p.ema_decay,
                    mh_proposal_sd=p.mh_proposal_sd,
                    asymmetric_error=p.asymmetric_error
                )
            else:
                raise ValueError('Network type "%s" not supported' %p.network_type)

            dtsr_model.fit(
                X,
                y,
                n_iter=p.n_iter,
                irf_name_map=p.irf_name_map,
                plot_n_time_units=p.plot_n_time_units,
                plot_n_points_per_time_unit=p.plot_n_points_per_time_unit,
                plot_x_inches=p.plot_x_inches,
                plot_y_inches=p.plot_y_inches,
                cmap=p.cmap
            )

            if eval:
                dtsr_preds = dtsr_model.predict(
                    X,
                    y.time,
                    y[dtsr_model.form.rangf],
                    y.first_obs,
                    y.last_obs
                )

                dtsr_mse = mse(y[dv], dtsr_preds)
                dtsr_mae = mae(y[dv], dtsr_preds)
                y_dv_mean = y[dv].mean()

                summary = '=' * 50 + '\n'
                summary += 'DTSR regression\n\n'
                summary += 'Model name: %s\n\n' % m
                summary += 'Formula:\n'
                summary += '  ' + formula + '\n\n'

                dtsr_loglik_vector = dtsr_model.log_lik(X, y)
                dtsr_loglik = dtsr_loglik_vector.sum()
                summary += 'Log likelihood: %s\n' %dtsr_loglik

                if bayes:
                    if dtsr_model.pc:
                        terminal_names = dtsr_model.src_terminal_names
                    else:
                        terminal_names = dtsr_model.terminal_names
                    posterior_summaries = np.zeros((len(terminal_names), 3))
                    for i in range(len(terminal_names)):
                        terminal = terminal_names[i]
                        row = np.array(dtsr_model.ci_integral(terminal, n_time_units=10))
                        posterior_summaries[i] += row
                    posterior_summaries = pd.DataFrame(posterior_summaries, index=terminal_names, columns=['Mean', '2.5%', '97.5%'])

                    summary += '\nPosterior integral summaries by predictor:\n'
                    summary += posterior_summaries.to_string() + '\n\n'

                summary += 'Training set loss:\n'
                summary += '  MSE: %.4f\n' % dtsr_mse
                summary += '  MAE: %.4f\n' % dtsr_mae
                summary += '=' * 50 + '\n'

                with open(p.logdir + '/' + m + '/summary.txt', 'w') as f_out:
                    f_out.write(summary)
                sys.stderr.write(summary)
                sys.stderr.write('\n\n')

