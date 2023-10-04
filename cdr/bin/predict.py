import argparse
import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

pd.options.mode.chained_assignment = None

from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import add_responses, filter_invalid_responses, preprocess_data, compute_splitID, compute_partition, s, c, z, split_cdr_outputs
from cdr.model import CDREnsemble
from cdr.util import mse, mae, percent_variance_explained
from cdr.util import filter_models, get_partition_list, paths_from_partition_cliarg, stderr, sn
from cdr.plot import plot_qq


spillover = re.compile('(z_)?([^ (),]+)S([0-9]+)')


# These code blocks are factored out because they are used by both LM/E objects and CDR objects under 2-step analysis
def predict_LM(model, outdir, X, dv, partition_name, model_name='', file_ix=None):
    preds = model.predict(X)

    model_str = '' if model_name=='' else model_name + '_'
    dv_str = sn(dv)
    file_str = '' if file_ix is None else 'file%s' % file_ix
    preds_out_name = '%s%s_preds%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    err_out_name = '%s%s_squared_error%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    eval_out_name = '%s%s_eval%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    with open(outdir + '/' + model_name + '/' + preds_out_name, 'w') as p_file:
        for i in range(len(preds)):
            p_file.write(str(preds[i]) + '\n')
    squared_error = np.array(X[dv] - preds) ** 2
    with open(outdir + '/' + model_name + '/' + err_out_name, 'w') as p_file:
        for i in range(len(squared_error)):
            p_file.write(str(squared_error[i]) + '\n')

    lm_mse = mse(X[dv], preds)
    lm_mae = mae(X[dv], preds)

    summary = '=' * 50 + '\n'
    summary += 'Linear regression\n\n'
    summary += 'Model name: %s\n\n' % model_name
    summary += 'Formula:\n'
    summary += '  ' + formula + '\n'
    summary += str(model.summary()) + '\n'
    summary += 'Error (%s set):\n' % partition_name
    summary += '  MSE: %.4f\n' % lm_mse
    summary += '  MAE: %.4f\n' % lm_mae
    summary += '=' * 50 + '\n'

    with open(outdir + '/' + eval_out_name, 'w') as f_out:
        f_out.write(summary)
    stderr(summary)


def predict_LME(model, outdir, X, dv, partition_name, model_name='', file_ix=None):
    preds = model.predict(X)

    model_str = '' if model_name=='' else model_name + '_'
    dv_str = sn(dv)
    file_str = '' if file_ix is None else 'file%s' % file_ix
    preds_out_name = '%s%s_preds%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    err_out_name = '%s%s_squared_error%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    eval_out_name = '%s%s_eval%s_%s.txt' % (model_str, dv_str, file_str, partition_name)
    with open(outdir + '/' + model_name + '/' + preds_out_name, 'w') as p_file:
        for i in range(len(preds)):
            p_file.write(str(preds[i]) + '\n')
    squared_error = np.array(X[dv] - preds) ** 2
    with open(outdir + '/' + model_name + '/' + err_out_name, 'w') as p_file:
        for i in range(len(squared_error)):
            p_file.write(str(squared_error[i]) + '\n')

    lme_mse = mse(X[dv], preds)
    lme_mae = mae(X[dv], preds)

    summary = '=' * 50 + '\n'
    summary += 'LME regression\n\n'
    summary += 'Model name: %s\n\n' % m
    summary += 'Formula:\n'
    summary += '  ' + formula + '\n'
    summary += str(model.summary()) + '\n'
    summary += 'Error (%s set):\n' % partition_name
    summary += '  MSE: %.4f\n' % lme_mse
    summary += '  MAE: %.4f\n' % lme_mae
    summary += '=' * 50 + '\n'

    with open(outdir + '/' + eval_out_name, 'w') as f_out:
        f_out.write(summary)
    stderr(summary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file(s)')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names from which to predict. Regex permitted. If unspecified, predicts from all models.')
    argparser.add_argument('-p', '--partition', nargs='+', default=['dev'], help='List of names of partitions to use ("train", "dev", "test", or a hyphen-delimited subset of these, or "PREDICTOR_PATH(;PREDICTOR_PATH):(RESPONSE_PATH;RESPONSE_PATH)").')
    argparser.add_argument('-n', '--nsamples', type=int, default=1024, help='Number of posterior samples to average')
    argparser.add_argument('-M', '--mode', default='eval', help='Evaluation mode(s), either "predict" (to just generate predictions) or "eval" (to evaluate predictions, compute likelihoods, etc).')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions.')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For CDR models, predict from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='For two-step prediction from CDR models, predict from data convolved using the ablated model. Otherwise predict from data convolved using the full model.')
    argparser.add_argument('-e', '--extra_cols', action='store_true', help='For prediction from CDR models, dump prediction outputs and response metadata to a single csv.')
    argparser.add_argument('-O', '--optimize_memory', action='store_true', help="Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).")
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    for path in args.config_paths:
        p = Config(path)

        model_names = filter_models(p.model_names, args.models)

        model_cache = {}
        model_cache_twostep = {}

        run_baseline = False
        run_cdr = False
        for m in model_names:
            if m.startswith('LM') or m.startswith('GAM'):
                run_baseline = True
            else:
                run_cdr = True

        cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(model_names, cdr_only=True)]
        cdr_formula_name_list = [m for m in filter_models(p.model_names, cdr_only=True)]

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
                    categorical_columns=list(set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
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

        if run_baseline:
            from cdr.baselines import py2ri
            evaluation_set_baselines = []
            partition_name_to_ix = {'train': 0, 'dev': 1, 'test': 2}
            for i in range(len(evaluation_sets)):
                X, Y, select = evaluation_sets[i][:3]
                assert len(X) == 1, 'Cannot run baselines on asynchronously sampled predictors'
                assert len(Y) == 1, 'Cannot run baselines on multiple responses'

                X = X[0]
                Y = Y[0]
                partitions = evaluation_set_partitions[i]
                X['splitID'] = compute_splitID(X, p.split_ids)

                X_baseline = X

                for m in model_names:
                    if not m in cdr_formula_name_list:
                        p.set_model(m)
                        form = p['formula']
                        lhs, rhs = form.split('~')
                        preds = rhs.split('+')
                        for pred in preds:
                            sp = spillover.search(pred)
                            if sp and sp.group(2) in X_baseline:
                                x_id = sp.group(2)
                                n = int(sp.group(3))
                                x_id_sp = x_id + 'S' + str(n)
                                if x_id_sp not in X_baseline:
                                    X_baseline[x_id_sp] = X_baseline.groupby(p.series_ids)[x_id].shift_activations(n, fill_value=0.)

                if partitions is not None:
                    part = compute_partition(X, p.modulus, 3)
                    part_select = None
                    for partition in partitions:
                        if part_select is None:
                            part_select = part[partition_name_to_ix[partition]]
                        else:
                            part_select &= part[partition_name_to_ix[partition]]

                    X_baseline = X_baseline[part_select]
                common_cols = sorted(list(set(X_baseline.columns) & set(Y.columns)))
                X_baseline = pd.merge(X_baseline, Y, on=common_cols, how='inner')

                for m in model_names:
                    if not m in cdr_formula_name_list:
                        p.set_model(m)
                        form = p['formula']
                        dv = form.split('~')[0].strip()
                        Y = add_responses(dv, Y)
                        if not dv in X_baseline:
                            X_baseline[dv] = Y[dv]

                for c in X_baseline.columns:
                    if X_baseline[c].dtype.name == 'category':
                        X_baseline[c] = X_baseline[c].astype(str)

                X_baseline = py2ri(X_baseline)
                evaluation_set_baselines.append(X_baseline)

        for d in range(len(evaluation_sets)):
            X, Y, select, X_in_Y_names = evaluation_sets[d]
            partition_str = evaluation_set_names[d]
            if run_baseline:
                X_baseline = evaluation_set_baselines[d]

            for m in model_names:
                formula = p.models[m]['formula']
                p.set_model(m)
                m_path = m.replace(':', '+')
                if not os.path.exists(p.outdir + '/' + m_path):
                    os.makedirs(p.outdir + '/' + m_path)
                with open(p.outdir + '/' + m_path + '/pred_inputs_%s.txt' % partition_str, 'w') as f:
                    f.write('%s\n' % (' '.join(evaluation_set_paths[d][0])))
                    f.write('%s\n' % (' '.join(evaluation_set_paths[d][1])))

                if m in model_cache:
                    _model = model_cache[m]
                else:
                    stderr('Retrieving saved model %s...\n' % m)
                    is_cdr = not (m.startswith('LM') or m.startswith('GAM'))
                    if is_cdr:
                        _model = CDREnsemble(p.outdir, m_path)
                    else:
                        with open(p.outdir + '/' + m_path + '/m.obj', 'rb') as m_file:
                            _model = pickle.load(m_file)
                    model_cache[m] = _model

                if m.startswith('LME'):
                    dv = formula.strip().split('~')[0].strip()

                    predict_LME(
                        _model,
                        p.outdir + '/' + m_path,
                        X_baseline,
                        dv,
                        partition_str
                    )

                elif m.startswith('LM'):
                    dv = formula.strip().split('~')[0].strip()

                    predict_LM(
                        _model,
                        p.outdir + '/' + m_path,
                        X_baseline,
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

                    gam_preds = _model.predict(X_baseline)
                    with open(p.outdir + '/' + m_path + '/preds_%s.txt' % partition_str, 'w') as p_file:
                        for i in range(len(gam_preds)):
                            p_file.write(str(gam_preds[i]) + '\n')
                    squared_error = np.array(Y[dv] - gam_preds) ** 2
                    with open(p.outdir + '/' + m_path + '/squared_error_%s.txt' % partition_str, 'w') as p_file:
                        for i in range(len(squared_error)):
                            p_file.write(str(squared_error[i]) + '\n')
                    gam_mse = mse(Y[dv], gam_preds)
                    gam_mae = mae(Y[dv], gam_preds)
                    summary = '=' * 50 + '\n'
                    summary += 'GAM regression\n\n'
                    summary += 'Model name: %s\n\n' % m
                    summary += 'Formula:\n'
                    summary += '  ' + formula + '\n'
                    summary += str(_model.summary()) + '\n'
                    summary += 'Loss (%s set):\n' % partition_str
                    summary += '  MSE: %.4f\n' % gam_mse
                    summary += '  MAE: %.4f\n' % gam_mae
                    summary += '=' * 50 + '\n'
                    with open(p.outdir + '/' + m_path + '/eval_%s.txt' % partition_str, 'w') as f_out:
                        f_out.write(summary)
                    stderr(summary)

                elif not (m.startswith('LM') or m.startswith('GAM')):
                    if not p.use_gpu_if_available or args.cpu_only:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

                    dv = [x.strip() for x in formula.strip().split('~')[0].strip().split('+')]
                    if partition_str in ('CVdev', 'CVtest'):
                        assert _model.use_crossval, 'Model %s was fitted without cross-validation and cannot predict over partition %s.' % (m, partition_str)
                        if partition_str == 'CVdev':
                            assert _model.crossval_dev_fold is not None, 'Model %s did not use a cross-validation dev fold and cannot predict over partition %s.' % (m, partition_str)
                            _Y = [_Y[_Y[_model.crossval_factor] == _model.crossval_dev_fold] for _Y in Y]
                        else:  # partition_str == 'CVtest'
                            _Y = [_Y[_Y[_model.crossval_factor] == _model.crossval_fold] for _Y in Y]
                    else:
                        _Y = Y
                    Y_valid, select_Y_valid = filter_invalid_responses(_Y, dv)

                    if args.twostep:
                        from cdr.baselines import py2ri

                        for _response in _model.response_names:
                            file_ix = _model.response_to_df_ix[_response]
                            multiple_files = len(file_ix) > 1
                            for ix in file_ix:
                                if multiple_files:
                                    data_name = 'X_conv_%s_mu_file%s_%s.csv' % (sn(_response), ix, partition_str)
                                else:
                                    data_name = 'X_conv_%s_mu_%s.csv' % (sn(_response), partition_str)
                                if args.ablated_models:
                                    data_path = p.outdir + '/' + m_path + '/' + data_name
                                else:
                                    data_path = p.outdir + '/' + m_path.split('!')[0] + '/' + data_name

                                df = pd.read_csv(data_path, sep=' ', skipinitialspace=True)
                                for c in df.columns:
                                    if df[c].dtype.name == 'object':
                                        df[c] = df[c].astype(str)

                                new_cols = []
                                for c in df.columns:
                                    new_cols.append(c.replace('-', '_'))
                                df.columns = new_cols
                                df[_response] = Y[ix][_response]

                                df_r = py2ri(df)

                                is_lme = '|' in Formula(p['formula']).to_lmer_formula_string()

                                if m in model_cache_twostep:
                                    _model = model_cache_twostep[m]
                                else:
                                    stderr('Retrieving saved model %s...\n' % m)
                                    if multiple_files:
                                        out_name = 'lm_%s_file%s_train.obj' % (sn(_response), ix)
                                    else:
                                        out_name = 'lm_%s_train.obj' % sn(_response)
                                    with open(p.outdir + '/%s/%s' % (m_path, out_name), 'rb') as m_file:
                                        _model = pickle.load(m_file)
                                    model_cache_twostep[m] = _model

                                if is_lme:
                                    predict_LME(
                                        _model,
                                        p.outdir + '/' + m_path,
                                        df_r,
                                        dv,
                                        partition_str,
                                        model_name='LM_2STEP',
                                        file_ix=file_ix if multiple_files else None
                                    )
                                else:
                                    predict_LM(
                                        _model,
                                        p.outdir + '/' + m_path,
                                        df_r,
                                        dv,
                                        partition_str,
                                        model_name='LM_2STEP',
                                        file_ix=file_ix if multiple_files else None
                                    )

                    else:
                        cdr_mse = {}
                        cdr_corr = {}
                        cdr_f1 = {}
                        cdr_loglik = {}
                        cdr_loss = {}
                        cdr_percent_variance_explained = {}
                        cdr_true_variance = {}

                        if args.algorithm.lower() == 'map':
                            _model.set_weight_type('ll')
                        else:
                            _model.set_weight_type('uniform')

                        if args.mode == 'predict':
                            _model.predict(
                                X,
                                Y_valid,
                                X_in_Y_names=X_in_Y_names,
                                n_samples=args.nsamples,
                                algorithm=args.algorithm,
                                extra_cols=args.extra_cols,
                                dump=True,
                                partition=partition_str,
                                optimize_memory=args.optimize_memory
                            )
                        elif args.mode.startswith('eval'):
                            _cdr_out = _model.evaluate(
                                X,
                                Y_valid,
                                X_in_Y_names=X_in_Y_names,
                                n_samples=args.nsamples,
                                algorithm=args.algorithm,
                                extra_cols=args.extra_cols,
                                dump=True,
                                partition=partition_str,
                                optimize_memory=args.optimize_memory
                            )
                        else:
                            raise ValueError('Unrecognized evaluation mode %s.' % args.mode)
