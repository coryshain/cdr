import argparse
import os
import sys
import re
import pickle
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from cdr.config import Config
from cdr.io import read_data
from cdr.formula import Formula
from cdr.data import add_dv, filter_invalid_responses, preprocess_data, compute_splitID, compute_partition, get_first_last_obs_lists, s, c, z
from cdr.util import mse, mae, percent_variance_explained
from cdr.util import load_cdr, filter_models, get_partition_list, paths_from_partition_cliarg, stderr
from cdr.plot import plot_qq


spillover = re.compile('(z_)?([^ (),]+)S([0-9]+)')


# These code blocks are factored out because they are used by both LM/E objects and CDR objects under 2-step analysis
def predict_LM(lm, outdir, X, y, dv, partition_name, model_name=''):
    lm_preds = lm.predict(X)
    with open(outdir + '/' + model_name + '/%spreds_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as p_file:
        for i in range(len(lm_preds)):
            p_file.write(str(lm_preds[i]) + '\n')
    squared_error = np.array(y[dv] - lm_preds) ** 2
    with open(outdir + '/' + model_name + '/%s_squared_error_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as p_file:
        for i in range(len(squared_error)):
            p_file.write(str(squared_error[i]) + '\n')
    lm_mse = mse(y[dv], lm_preds)
    lm_mae = mae(y[dv], lm_preds)
    summary = '=' * 50 + '\n'
    summary += 'Linear regression\n\n'
    summary += 'Model name: %s\n\n' % model_name
    summary += 'Formula:\n'
    summary += '  ' + formula + '\n'
    summary += str(lm.summary()) + '\n'
    summary += 'Error (%s set):\n' % partition_name
    summary += '  MSE: %.4f\n' % lm_mse
    summary += '  MAE: %.4f\n' % lm_mae
    summary += '=' * 50 + '\n'
    with open(outdir + '/%seval_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as f_out:
        f_out.write(summary)
    stderr(summary)

def predict_LME(model_path, outdir, X, y, dv, partition_name, model_name=''):
    stderr('Retrieving saved model %s...\n' % m)
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

        with open(outdir + '/%spreds_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as p_file:
            for i in range(len(lme_preds)):
                p_file.write(str(lme_preds[i]) + '\n')
        squared_error = np.array(y[dv] - lme_preds) ** 2
        with open(outdir + '/%ssquared_error_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as p_file:
            for i in range(len(squared_error)):
                p_file.write(str(squared_error[i]) + '\n')
        lme_mse = mse(y[dv], lme_preds)
        lme_mae = mae(y[dv], lme_preds)

        summary += 'Error (%s set):\n' % partition_name
        summary += '  MSE: %.4f\n' % lme_mse
        summary += '  MAE: %.4f\n' % lme_mae

    summary += '=' * 50 + '\n'
    with open(outdir + '/%seval_%s.txt' % ('' if model_name=='' else model_name + '_', partition_name), 'w') as f_out:
        f_out.write(summary)
    stderr(summary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names from which to predict. Regex permitted. If unspecified, predicts from all models.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", "PREDICTOR_PATH;RESPONSE_PATH", or hyphen-delimited subset of these).')
    argparser.add_argument('-z', '--standardize_response', action='store_true', help='Standardize (Z-transform) response. Ignored for non-CDR models, and ignored for CDR models unless fitting used setting ``standardize_respose=True``.')
    argparser.add_argument('-n', '--nsamples', type=int, default=1024, help='Number of posterior samples to average (only used for CDRBayes)')
    argparser.add_argument('-M', '--mode', nargs='+', default=None, help='Predict mode(s) (set of "response", "loglik", "loss", and/or "err") or default ``None``, which does loglik, response, and err. Modes "loglik" and "loss" are only valid for CDR.')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions from CDRBayes. Ignored for CDRMLE.')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For CDR models, predict from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-T', '--training_mode', action='store_true', help='Use training mode for prediction.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='For two-step prediction from CDR models, predict from data convolved using the ablated model. Otherwise predict from data convolved using the full model.')
    argparser.add_argument('-e', '--extra_cols', action='store_true', help='For prediction from CDR models, dump prediction outputs and response metadata to a single csv.')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    models = filter_models(p.model_list, args.models)

    model_cache = {}
    model_cache_twostep = {}

    run_baseline = False
    run_cdr = False
    for m in models:
        if not run_baseline and m.startswith('LM') or m.startswith('GAM'):
            run_baseline = True
        elif not run_cdr and (m.startswith('CDR') or m.startswith('DTSR')):
            run_cdr = True

    cdr_formula_list = [Formula(p.models[m]['formula']) for m in models if (m.startswith('CDR') or m.startswith('DTSR'))]
    cdr_formula_name_list = [m for m in p.model_list if (m.startswith('CDR') or m.startswith('DTSR'))]

    evaluation_sets = []
    evaluation_set_partitions = []
    evaluation_set_names = []
    evaluation_set_paths = []

    for i, p_name in enumerate(args.partition):
        partitions = get_partition_list(p_name)
        if ';' in p_name:
            partition_str = '%d' % (i + 1)
            X_paths = [partitions[0]]
            y_paths = [partitions[1]]
        else:
            partition_str = '-'.join(partitions)
            X_paths, y_paths = paths_from_partition_cliarg(partitions, p)
        X, y = read_data(
            X_paths,
            y_paths,
            p.series_ids,
            sep=p.sep,
            categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
        )
        X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = preprocess_data(
            X,
            y,
            cdr_formula_list,
            p.series_ids,
            filters=p.filters,
            compute_history=run_cdr,
            history_length=p.history_length
        )
        evaluation_sets.append((X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors))
        evaluation_set_partitions.append(partitions)
        evaluation_set_names.append(partition_str)
        evaluation_set_paths.append((X_paths, y_paths))

    if run_baseline:
        from cdr.baselines import py2ri
        evaluation_set_baselines = []
        partition_name_to_ix = {'train': 0, 'dev': 1, 'test': 2}
        for i in range(len(evaluation_sets)):
            X, y, select = evaluation_sets[i][:3]
            assert len(X) == 1, 'Cannot run baselines on asynchronously sampled predictors'
            X_cur = X[0]
            partitions = evaluation_set_partitions[i]
            X_cur['splitID'] = compute_splitID(X_cur, p.split_ids)

            X_baseline = X_cur

            for m in models:
                if not m in cdr_formula_name_list:
                    p.set_model(m)
                    form = p['formula']
                    lhs, rhs = form.split('~')
                    preds = rhs.split('+')
                    for pred in preds:
                        sp = spillover.search(pred)
                        if sp and sp.group(2) in X_baseline.columns:
                            x_id = sp.group(2)
                            n = int(sp.group(3))
                            x_id_sp = x_id + 'S' + str(n)
                            if x_id_sp not in X_baseline.columns:
                                X_baseline[x_id_sp] = X_baseline.groupby(p.series_ids)[x_id].shift_activations(n, fill_value=0.)

            if partitions is not None:
                part = compute_partition(X_cur, p.modulus, 3)
                part_select = None
                for partition in partitions:
                    if part_select is None:
                        part_select = part[partition_name_to_ix[partition]]
                    else:
                        part_select &= part[partition_name_to_ix[partition]]

                X_baseline = X_baseline[part_select]
            common_cols = sorted(list(set(X_baseline.columns) & set(y.columns)))
            X_baseline = pd.merge(X_baseline, y, on=common_cols, how='inner')

            for m in models:
                if not m in cdr_formula_name_list:
                    p.set_model(m)
                    form = p['formula']
                    dv = form.split('~')[0].strip()
                    y = add_dv(dv, y)
                    if not dv in X_baseline.columns:
                        X_baseline[dv] = y[dv]

            for c in X_baseline.columns:
                if X_baseline[c].dtype.name == 'category':
                    X_baseline[c] = X_baseline[c].astype(str)

            X_baseline = py2ri(X_baseline)
            evaluation_set_baselines.append(X_baseline)

    for d in range(len(evaluation_sets)):
        X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = evaluation_sets[d]
        partition_str = evaluation_set_names[d]
        if run_baseline:
            X_baseline = evaluation_set_baselines[d]

        for m in models:
            formula = p.models[m]['formula']
            p.set_model(m)
            m_path = m.replace(':', '+')
            if not os.path.exists(p.outdir + '/' + m_path):
                os.makedirs(p.outdir + '/' + m_path)
            with open(p.outdir + '/' + m_path + '/pred_inputs_%s.txt' % partition_str, 'w') as f:
                f.write('%s\n' % (' '.join(evaluation_set_paths[d][0])))
                f.write('%s\n' % (' '.join(evaluation_set_paths[d][1])))

            if m in model_cache:
                model_cur = model_cache[m]
            else:
                stderr('Retrieving saved model %s...\n' % m)
                if (m.startswith('CDR') or m.startswith('DTSR')):
                    model_cur = load_cdr(p.outdir + '/' + m_path)
                else:
                    with open(p.outdir + '/' + m_path + '/m.obj', 'rb') as m_file:
                        model_cur = pickle.load(m_file)
                model_cache[m] = model_cur

            if m.startswith('LME'):
                dv = formula.strip().split('~')[0].strip()

                predict_LME(
                    model_cur,
                    p.outdir + '/' + m_path,
                    X_baseline,
                    y,
                    dv,
                    partition_str
                )

            elif m.startswith('LM'):
                dv = formula.strip().split('~')[0].strip()

                predict_LM(
                    model_cur,
                    p.outdir + '/' + m_path,
                    X_baseline,
                    y,
                    dv,
                    partition_str
                )

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

                gam_preds = model_cur.predict(X_baseline)
                with open(p.outdir + '/' + m_path + '/preds_%s.txt' % partition_str, 'w') as p_file:
                    for i in range(len(gam_preds)):
                        p_file.write(str(gam_preds[i]) + '\n')
                squared_error = np.array(y[dv] - gam_preds) ** 2
                with open(p.outdir + '/' + m_path + '/squared_error_%s.txt' % partition_str, 'w') as p_file:
                    for i in range(len(squared_error)):
                        p_file.write(str(squared_error[i]) + '\n')
                gam_mse = mse(y[dv], gam_preds)
                gam_mae = mae(y[dv], gam_preds)
                summary = '=' * 50 + '\n'
                summary += 'GAM regression\n\n'
                summary += 'Model name: %s\n\n' % m
                summary += 'Formula:\n'
                summary += '  ' + formula + '\n'
                summary += str(model_cur.summary()) + '\n'
                summary += 'Loss (%s set):\n' % partition_str
                summary += '  MSE: %.4f\n' % gam_mse
                summary += '  MAE: %.4f\n' % gam_mae
                summary += '=' * 50 + '\n'
                with open(p.outdir + '/' + m_path + '/eval_%s.txt' % partition_str, 'w') as f_out:
                    f_out.write(summary)
                stderr(summary)

            elif (m.startswith('CDR') or m.startswith('DTSR')):
                if not p.use_gpu_if_available:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

                dv = formula.strip().split('~')[0].strip()
                if model_cur.use_crossval:
                    crossval_factor = model_cur.crossval_factor
                    crossval_fold = model_cur.crossval_fold
                else:
                    crossval_factor = None
                    crossval_fold = None
                y_valid, select_y_valid = filter_invalid_responses(
                    y,
                    dv,
                    crossval_factor=crossval_factor,
                    crossval_fold=crossval_fold
                )
                X_response_aligned_predictors_valid = X_response_aligned_predictors
                if X_response_aligned_predictors_valid is not None:
                    X_response_aligned_predictors_valid = X_response_aligned_predictors_valid[select_y_valid]

                if args.twostep:
                    from cdr.baselines import py2ri

                    if args.ablated_models:
                        data_path = p.outdir + '/' + m_path + '/X_conv_' + partition_str + '.csv'
                    else:
                        data_path = p.outdir + '/' + m_path.split('!')[0] + '/X_conv_' + partition_str + '.csv'

                    df = pd.read_csv(data_path, sep=' ', skipinitialspace=True)
                    for c in df.columns:
                        if df[c].dtype.name == 'object':
                            df[c] = df[c].astype(str)

                    new_cols = []
                    for c in df.columns:
                        new_cols.append(c.replace('-', '_'))
                    df.columns = new_cols

                    df_r = py2ri(df)

                    is_lme = '|' in Formula(p['formula']).to_lmer_formula_string()

                    if m in model_cache_twostep:
                        model_cur = model_cache_twostep[m]
                    else:
                        stderr('Retrieving saved model %s...\n' % m)
                        with open(p.outdir + '/' + m_path + '/lm_train.obj', 'rb') as m_file:
                            model_cur = pickle.load(m_file)
                        model_cache_twostep[m] = model_cur

                    if is_lme:
                        predict_LME(
                            model_cur,
                            p.outdir + '/' + m_path,
                            df_r,
                            df,
                            dv,
                            partition_str,
                            model_name='LM_2STEP'
                        )
                    else:
                        predict_LM(
                            model_cur,
                            p.outdir + '/' + m_path,
                            df_r,
                            df,
                            dv,
                            partition_str,
                            model_name='LM_2STEP'
                        )

                else:
                    bayes = p['network_type'] == 'bayes'

                    summary = '=' * 50 + '\n'
                    summary += 'CDR regression\n\n'
                    summary += 'Model name: %s\n\n' % m
                    summary += 'Formula:\n'
                    summary += '  ' + formula + '\n\n'
                    summary += 'Partition: %s\n\n' % partition_str

                    cdr_mse = cdr_mae = cdr_corr = cdr_loglik = cdr_loss = cdr_percent_variance_explained = cdr_true_variance = None

                    if model_cur.standardize_response and args.standardize_response:
                        y_cur = (y_valid[dv] - model_cur.y_train_mean) / model_cur.y_train_sd
                    else:
                        y_cur = y_valid[dv]
                    if args.mode is None or 'response' in args.mode:
                        first_obs, last_obs = get_first_last_obs_lists(y_valid)
                        cdr_preds = model_cur.predict(
                            X,
                            y_valid.time,
                            y_valid[model_cur.form.rangf],
                            first_obs,
                            last_obs,
                            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                            X_response_aligned_predictors=X_response_aligned_predictors_valid,
                            X_2d_predictor_names=X_2d_predictor_names,
                            X_2d_predictors=X_2d_predictors,
                            n_samples=args.nsamples,
                            algorithm=args.algorithm,
                            standardize_response=args.standardize_response
                        )

                        squared_error = np.array(y_cur - cdr_preds) ** 2

                        if args.extra_cols:
                            df_out = pd.DataFrame(
                                {
                                    'CDRsquarederror': squared_error,
                                    'CDRpreds': cdr_preds,
                                    'CDRobs': y_cur
                                }
                            )
                            df_out = pd.concat([y_valid.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1)
                        else:
                            preds_outfile = p.outdir + '/' + m_path + '/preds_%s.txt' % partition_str
                            err_outfile = p.outdir + '/' + m_path + '/squared_error_%s.txt' % partition_str
                            obs_outfile = p.outdir + '/' + m_path + '/obs_%s.txt' % partition_str

                            with open(preds_outfile, 'w') as p_file:
                                for i in range(len(cdr_preds)):
                                    p_file.write(str(cdr_preds[i]) + '\n')
                            with open(err_outfile, 'w') as l_file:
                                for i in range(len(squared_error)):
                                    l_file.write(str(squared_error[i]) + '\n')
                            with open(obs_outfile, 'w') as p_file:
                                for i in range(len(y_cur)):
                                    p_file.write(str(y_cur.iloc[i]) + '\n')

                        cdr_mse = mse(y_cur, cdr_preds)
                        cdr_mae = mae(y_cur, cdr_preds)
                        cdr_corr = np.corrcoef(y_cur, cdr_preds, rowvar=False)[0,1]
                        cdr_percent_variance_explained = percent_variance_explained(y_cur, cdr_preds)
                        cdr_true_variance = np.std(y_cur) ** 2
                        y_dv_mean = y_cur.mean()

                        err = np.sort(y_cur - cdr_preds)
                        err_theoretical_q = model_cur.error_theoretical_quantiles(len(err))
                        valid = np.isfinite(err_theoretical_q)
                        err = err[valid]
                        err_theoretical_q = err_theoretical_q[valid]

                        plot_qq(
                            err_theoretical_q,
                            err,
                            dir=model_cur.outdir,
                            filename='error_qq_plot_%s.png' % partition_str,
                            xlab='Theoretical',
                            ylab='Empirical'
                        )

                        D, p_value = model_cur.error_ks_test(err)

                    if args.mode is None or 'loglik' in args.mode:
                        cdr_loglik_vector = model_cur.log_lik(
                            X,
                            y_valid,
                            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                            X_response_aligned_predictors=X_response_aligned_predictors_valid,
                            X_2d_predictor_names=X_2d_predictor_names,
                            X_2d_predictors=X_2d_predictors,
                            n_samples=args.nsamples,
                            algorithm=args.algorithm,
                            standardize_response=args.standardize_response,
                            training=args.training_mode
                        )

                        if args.extra_cols:
                            df_ll = pd.DataFrame({'CDRloglik': cdr_loglik_vector})
                            df_out= pd.concat([df_out, df_ll], axis=1)
                        else:
                            ll_outfile = p.outdir + '/' + m_path + '/loglik_%s.txt' % partition_str
                            with open(ll_outfile, 'w') as l_file:
                                for i in range(len(cdr_loglik_vector)):
                                    l_file.write(str(cdr_loglik_vector[i]) + '\n')
                        cdr_loglik = cdr_loglik_vector.sum()
                    if args.mode is not None and 'loss' in args.mode:
                        cdr_loss = model_cur.loss(
                            X,
                            y_valid,
                            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                            X_response_aligned_predictors=X_response_aligned_predictors_valid,
                            X_2d_predictor_names=X_2d_predictor_names,
                            X_2d_predictors=X_2d_predictors,
                            n_samples=args.nsamples,
                            algorithm=args.algorithm,
                            training=args.training_mode
                        )

                    if bayes:
                        if model_cur.pc:
                            terminal_names = model_cur.src_terminal_names
                        else:
                            terminal_names = model_cur.terminal_names

                    if args.extra_cols:
                        preds_outfile = p.outdir + '/' + m_path + '/pred_table_%s.csv' % partition_str
                        df_out.to_csv(preds_outfile, sep=' ', na_rep='NaN', index=False)

                    summary += 'Training iterations completed: %d\n\n' % model_cur.global_step.eval(session=model_cur.sess)

                    summary += model_cur.report_evaluation(
                        mse=cdr_mse,
                        mae=cdr_mae,
                        corr=cdr_corr,
                        loglik=cdr_loglik,
                        loss=cdr_loss,
                        percent_variance_explained=cdr_percent_variance_explained,
                        true_variance=cdr_true_variance,
                        ks_results=(D, p_value) if args.mode in [None, 'response'] else None
                    )

                    summary += '=' * 50 + '\n'

                    with open(p.outdir + '/' + m_path + '/eval_%s.txt' % partition_str, 'w') as f_out:
                        f_out.write(summary)
                    stderr(summary)
                    stderr('\n\n')
