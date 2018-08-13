import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import filter_invalid_responses, preprocess_data, compute_splitID, compute_partition
from dtsr.util import mse, mae, percent_variance_explained
from dtsr.util import load_dtsr

def predict_LME(model_path, outdir, X, y, dv, model_name=''):
    sys.stderr.write('Retrieving saved model %s...\n' % m)
    with open(model_path, 'rb') as m_file:
        lme = pickle.load(m_file)

    summary = '=' * 50 + '\n'
    summary += 'LME regression\n\n'
    summary += 'Model name: %s\n\n' % m
    summary += 'Formula:\n'
    summary += '  ' + formula + '\n'
    summary += str(lme.summary()) + '\n'

    if args.mode in [None, 'response']:
        lme_preds = lme.predict(X)

        with open(outdir + '/%spreds_%s.txt' % ('' if model_name=='' else model_name + '_', args.partition), 'w') as p_file:
            for i in range(len(lme_preds)):
                p_file.write(str(lme_preds[i]) + '\n')
        if p['loss_name'].lower() == 'mae':
            losses = np.array(y[dv] - lme_preds).abs()
        else:
            losses = np.array(y[dv] - lme_preds) ** 2
        with open(outdir + '/%s%s_losses_%s.txt' % ('' if model_name=='' else model_name + '_', p['loss_name'], args.partition), 'w') as p_file:
            for i in range(len(losses)):
                p_file.write(str(losses[i]) + '\n')
        lme_mse = mse(y[dv], lme_preds)
        lme_mae = mae(y[dv], lme_preds)

        summary += 'Loss (%s set):\n' % args.partition
        summary += '  MSE: %.4f\n' % lme_mse
        summary += '  MAE: %.4f\n' % lme_mae

    if args.mode in [None, 'loglik']:
        lme_loglik = lme.log_lik(newdata=X, summed=False)
        with open(outdir + '/%sloglik_%s.txt' % ('' if model_name=='' else model_name + '_', args.partition), 'w') as p_file:
            for i in range(len(lme_loglik)):
                p_file.write(str(lme_loglik[i]) + '\n')
        lme_loglik = np.sum(lme_loglik)

        summary += '  Log Lik: %.4f\n' % lme_loglik

    summary += '=' * 50 + '\n'
    with open(outdir + '/%seval_%s.txt' % ('' if model_name=='' else model_name + '_', args.partition), 'w') as f_out:
        f_out.write(summary)
    sys.stderr.write(summary)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-n', '--nsamples', type=int, default=1024, help='Number of posterior samples to average (only used for DTSRBayes)')
    argparser.add_argument('-M', '--mode', type=str, default=None, help='Predict mode ("response" or "loglik") or default None, which does both')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For DTSR models, predict from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='For two-step prediction from DTSR models, predict from data convolved using the ablated model. Otherwise predict from data convolved using the full model.')
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
        X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'dev':
        X, y = read_data(p.X_dev, p.y_dev, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    elif args.partition == 'test':
        X, y = read_data(p.X_test, p.y_test, p.series_ids, categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    else:
        raise ValueError('Unrecognized value for "partition" argument: %s' %args.partition)
    X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = preprocess_data(
        X,
        y,
        p,
        dtsr_formula_list,
        compute_history=run_dtsr
    )

    if run_baseline:
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
        p.set_model(m)
        if not os.path.exists(p.outdir + '/' + m):
            os.makedirs(p.outdir + '/' + m)
        if m.startswith('LME'):
            dv = formula.strip().split('~')[0].strip()

            predict_LME(
                p.outdir + '/' + m + '/m.obj',
                p.outdir + '/' + m,
                X,
                y,
                dv
            )

        elif m.startswith('LM'):
            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            with open(p.outdir + '/' + m + '/m.obj', 'rb') as m_file:
                lm = pickle.load(m_file)

            lm_preds = lm.predict(X_baseline)
            with open(p.outdir + '/' + m + '/preds_%s.txt' % args.partition, 'w') as p_file:
                for i in range(len(lm_preds)):
                    p_file.write(str(lm_preds[i]) + '\n')
            if p['loss_name'].lower() == 'mae':
                losses = np.array(y[dv] - lm_preds).abs()
            else:
                losses = np.array(y[dv] - lm_preds) ** 2
            with open(p.outdir + '/' + m + '/%s_losses_%s.txt' % (p['loss_name'], args.partition), 'w') as p_file:
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
            with open(p.outdir + '/' + m + '/eval_%s.txt' % args.partition, 'w') as f_out:
                f_out.write(summary)
            sys.stderr.write(summary)

        elif m.startswith('GAM'):
            import re
            dv = formula.strip().split('~')[0].strip()

            ## For some reason, GAM can't predict using custom functions, so we have to translate them
            z_term = re.compile('z.\((.*)\)')
            c_term = re.compile('c.\((.*)\)')
            formula = [t.strip() for t in formula.strip().split() if t.strip() != '']
            for i in range(len(formula)):
                formula[i] = z_term.sub(r'scale(\1)', formula[i])
                formula[i] = c_term.sub(r'scale(\1, scale=FALSE)', formula[i])
            formula = ' '.join(formula)

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            with open(p.outdir + '/' + m + '/m.obj', 'rb') as m_file:
                gam = pickle.load(m_file)
            gam_preds = gam.predict(X_baseline)
            with open(p.outdir + '/' + m + '/preds_%s.txt' % args.partition, 'w') as p_file:
                for i in range(len(gam_preds)):
                    p_file.write(str(gam_preds[i]) + '\n')
            if p['loss_name'].lower() == 'mae':
                losses = np.array(y[dv] - gam_preds).abs()
            else:
                losses = np.array(y[dv] - gam_preds) ** 2
            with open(p.outdir + '/' + m + '/%s_losses_%s.txt' % (p['loss_name'], args.partition), 'w') as p_file:
                for i in range(len(losses)):
                    p_file.write(str(losses[i]) + '\n')
            gam_mse = mse(y[dv], gam_preds)
            gam_mae = mae(y[dv], gam_preds)
            summary = '=' * 50 + '\n'
            summary += 'GAM regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n'
            summary += str(gam.summary()) + '\n'
            summary += 'Loss (%s set):\n' % args.partition
            summary += '  MSE: %.4f\n' % gam_mse
            summary += '  MAE: %.4f\n' % gam_mae
            summary += '=' * 50 + '\n'
            with open(p.outdir + '/' + m + '/eval_%s.txt' % args.partition, 'w') as f_out:
                f_out.write(summary)
            sys.stderr.write(summary)

        elif m.startswith('DTSR'):
            if not p.use_gpu_if_available:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            dv = formula.strip().split('~')[0].strip()
            y_valid, select_y_valid = filter_invalid_responses(y, dv)
            X_response_aligned_predictors_valid = X_response_aligned_predictors
            if X_response_aligned_predictors_valid is not None:
                X_response_aligned_predictors_valid = X_response_aligned_predictors_valid[select_y_valid]

            if args.twostep:
                from dtsr.baselines import py2ri

                if args.ablated_models:
                    data_path = p.outdir + '/' + m + '/X_conv_' + args.partition + '.csv'
                else:
                    data_path = p.outdir + '/' + m.split('!')[0] + '/X_conv_' + args.partition + '.csv'

                df = pd.read_csv(data_path, sep=' ', skipinitialspace=True)
                for c in df.columns:
                    if df[c].dtype.name == 'object':
                        df[c] = df[c].astype(str)

                new_cols = []
                for c in df.columns:
                    new_cols.append(c.replace('-', '_'))
                df.columns = new_cols

                df_r = py2ri(df)

                predict_LME(
                    p.outdir + '/' + m + '/lmer_train.obj',
                    p.outdir + '/' + m,
                    df_r,
                    df,
                    dv,
                    model_name='LMER_2STEP'
                )

            else:
                sys.stderr.write('Retrieving saved model %s...\n' % m)
                dtsr_model = load_dtsr(p.outdir + '/' + m)

                bayes = p['network_type'] == 'bayes'

                summary = '=' * 50 + '\n'
                summary += 'DTSR regression\n\n'
                summary += 'Model name: %s\n\n' % m
                summary += 'Formula:\n'
                summary += '  ' + formula + '\n\n'
                summary += 'Partition: %s\n\n' % args.partition

                dtsr_mse = dtsr_mae = dtsr_loglik = dtsr_percent_variance_explained = None

                if args.mode in [None, 'response']:
                    dtsr_preds = dtsr_model.predict(
                        X,
                        y_valid.time,
                        y_valid[dtsr_model.form.rangf],
                        y_valid.first_obs,
                        y_valid.last_obs,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors_valid,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors,
                        n_samples=args.nsamples,
                        algorithm=args.algorithm
                    )
                    with open(p.outdir + '/' + m + '/preds_%s.txt' % args.partition, 'w') as p_file:
                        for i in range(len(dtsr_preds)):
                            p_file.write(str(dtsr_preds[i]) + '\n')
                    if p['loss_name'].lower() == 'mae':
                        losses = np.array(y_valid[dv] - dtsr_preds).abs()
                    else:
                        losses = np.array(y_valid[dv] - dtsr_preds) ** 2
                    with open(p.outdir + '/' + m + '/%s_losses_%s.txt' % (p['loss_name'], args.partition), 'w') as l_file:
                        for i in range(len(losses)):
                            l_file.write(str(losses[i]) + '\n')
                    with open(p.outdir + '/' + m + '/obs_%s.txt' % args.partition, 'w') as p_file:
                        for i in range(len(y_valid[dv])):
                            p_file.write(str(y_valid[dv].iloc[i]) + '\n')
                    dtsr_mse = mse(y_valid[dv], dtsr_preds)
                    dtsr_mae = mae(y_valid[dv], dtsr_preds)
                    dtsr_percent_variance_explained = percent_variance_explained(y_valid[dv], dtsr_preds)
                    y_dv_mean = y_valid[dv].mean()

                if args.mode in [None, 'loglik']:
                    dtsr_loglik_vector = dtsr_model.log_lik(
                        X,
                        y_valid,
                        X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                        X_response_aligned_predictors=X_response_aligned_predictors_valid,
                        X_2d_predictor_names=X_2d_predictor_names,
                        X_2d_predictors=X_2d_predictors,
                        n_samples=args.nsamples,
                        algorithm=args.algorithm
                    )
                    with open(p.outdir + '/' + m + '/loglik_%s.txt' % args.partition, 'w') as l_file:
                        for i in range(len(dtsr_loglik_vector)):
                            l_file.write(str(dtsr_loglik_vector[i]) + '\n')
                    dtsr_loglik = dtsr_loglik_vector.sum()

                if bayes:
                    if dtsr_model.pc:
                        terminal_names = dtsr_model.src_terminal_names
                    else:
                        terminal_names = dtsr_model.terminal_names
                    posterior_summaries = np.zeros((len(terminal_names), 3))
                    for i in range(len(terminal_names)):
                        terminal = terminal_names[i]
                        row = np.array(dtsr_model.irf_integral(terminal, n_time_units=10))
                        posterior_summaries[i] += row
                    posterior_summaries = pd.DataFrame(posterior_summaries, index=terminal_names,
                                                       columns=['Mean', '2.5%', '97.5%'])


                summary += dtsr_model.report_evaluation(
                    mse=dtsr_mse,
                    mae=dtsr_mae,
                    loglik=dtsr_loglik,
                    percent_variance_explained=dtsr_percent_variance_explained
                )

                summary += '=' * 50 + '\n'

                with open(p.outdir + '/' + m + '/eval_%s.txt' % args.partition, 'w') as f_out:
                    f_out.write(summary)
                sys.stderr.write(summary)
                sys.stderr.write('\n\n')

                dtsr_model.finalize()


