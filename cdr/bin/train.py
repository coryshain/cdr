import argparse
import os
import re
import pickle
import pandas as pd

pd.options.mode.chained_assignment = None

from cdr.kwargs import MODEL_INITIALIZATION_KWARGS
from cdr.config import Config
from cdr.io import read_tabular_data
from cdr.formula import Formula
from cdr.data import filter_invalid_responses, preprocess_data, compute_splitID, compute_partition
from cdr.model import CDRModel
from cdr.util import mse, mae, filter_models, get_partition_list, paths_from_partition_cliarg, stderr


spillover = re.compile('(z_)?([^ (),]+)S([0-9]+)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Trains model(s) from formula string(s) given data.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Path to configuration (*.ini) file')
    argparser.add_argument('-e', '--force_training_evaluation', action='store_true', help='Recompute training evaluation even for models that are already finished.')
    argparser.add_argument('-s', '--save_and_exit', action='store_true', help='Initialize, save, and exit (CDR only). Useful for bringing non-backward compatible trained models up to spec for plotting and evaluation.')
    argparser.add_argument('-S', '--skip_confirmation', action='store_true', help='If running with **-s**, skip interactive confirmation. Useful for batch re-saving many models. Use with caution, since old models will be overwritten without the option to confirm.')
    argparser.add_argument('-O', '--optimize_memory', action='store_true', help="Compute expanded impulse arrays on the fly rather than pre-computing. Can reduce memory consumption by orders of magnitude but adds computational overhead at each minibatch, slowing training (typically around 1.5-2x the unoptimized training time).")
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    p = Config(args.config_path)

    if not p.use_gpu_if_available or args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_names = filter_models(p.model_names, args.models)

    run_R = False
    run_cdr = False
    for m in model_names:
        if m.startswith('LM') or m.startswith('GAM'):
            run_R = True
        else:
            run_cdr = True

    if not (run_R or run_cdr):
        stderr('No models to run. Exiting...\n')
        exit()

    cdr_formula_list = [Formula(p.models[m]['formula']) for m in filter_models(model_names, cdr_only=True)]
    cdr_formula_name_list = [m for m in filter_models(p.model_names)]
    all_rangf = [v for x in cdr_formula_list for v in x.rangf]
    partitions = get_partition_list('train')
    all_interactions = False

    X_paths, Y_paths = paths_from_partition_cliarg(partitions, p)
    X_paths_dev = Y_paths_dev = X_dev = Y_dev = None
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
        t_delta_cutoff=p.t_delta_cutoff,
        all_interactions=all_interactions
    )

    if run_R:
        # from cdr.baselines import py2ri
        assert len(X) == 1, 'Cannot run baselines on asynchronously sampled predictors'
        assert len(Y) == 1, 'Cannot run baselines on asynchronously sampled responses'
        X_cur = X[0]
        X_cur['splitID'] = compute_splitID(X_cur, p.split_ids)
        part = compute_partition(X_cur, p.modulus, 3)
        part_select = None
        partition_name_to_ix = {'train': 0, 'dev': 1, 'test': 2}
        for partition in partitions:
            if part_select is None:
                part_select = part[partition_name_to_ix[partition]]
            else:
                part_select &= part[partition_name_to_ix[partition]]

        X_baseline = X_cur

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

        X_baseline = X_baseline[part_select]
        if p.merge_cols is None:
            merge_cols = sorted(list(set(X_baseline.columns) & set(Y[0].columns)))
        else:
            merge_cols = p.merge_cols
        X_baseline = pd.merge(X_baseline, Y[0], on=merge_cols, how='inner')

    n_train_sample = sum(len(_Y) for _Y in Y)

    for m in model_names:
        p.set_model(m)
        formula = p['formula']
        m_path = m.replace(':', '+')
        if not os.path.exists(p.outdir + '/' + m_path):
            os.makedirs(p.outdir + '/' + m_path)
        if m.startswith('LME'):
            from cdr.baselines import LME

            dv = formula.strip().split('~')[0].strip().replace('.','')

            if os.path.exists(p.outdir + '/' + m_path + '/m.obj'):
                stderr('Retrieving saved model %s...\n' % m)
                with open(p.outdir + '/' + m_path + '/m.obj', 'rb') as m_file:
                    lme = pickle.load(m_file)
            else:
                stderr('Fitting model %s...\n' % m)
                lme = LME(formula, X_baseline)
                with open(p.outdir + '/' + m_path + '/m.obj', 'wb') as m_file:
                    pickle.dump(lme, m_file)

            lme_preds = lme.predict(X_baseline)
            lme_mse = mse(Y[dv], lme_preds)
            lme_mae = mae(Y[dv], lme_preds)
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
            with open(p.outdir + '/' + m_path + '/summary.txt', 'w') as f_out:
                f_out.write(summary)
            stderr(summary)
            stderr('\n\n')

        elif m.startswith('LM'):
            from cdr.baselines import LM

            dv = formula.strip().split('~')[0].strip().replace('.','')

            if os.path.exists(p.outdir + '/' + m_path + '/m.obj'):
                stderr('Retrieving saved model %s...\n' % m)
                with open(p.outdir + '/' + m_path + '/m.obj', 'rb') as m_file:
                    lm = pickle.load(m_file)
            else:
                stderr('Fitting model %s...\n' % m)
                lm = LM(formula, X_baseline)
                with open(p.outdir + '/' + m_path + '/m.obj', 'wb') as m_file:
                    pickle.dump(lm, m_file)

            lm_preds = lm.predict(X_baseline)
            lm_mse = mse(Y[dv], lm_preds)
            lm_mae = mae(Y[dv], lm_preds)
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
            with open(p.outdir + '/' + m_path + '/summary.txt', 'w') as f_out:
                f_out.write(summary)
            stderr(summary)
            stderr('\n\n')

        elif m.startswith('GAM'):
            import re
            from cdr.baselines import GAM

            dv = formula.strip().split('~')[0].strip().replace('.','')
            ran_gf = ['subject', 'word', 'sentid']

            ## For some reason, GAM can't predict using custom functions, so we have to translate them
            z_term = re.compile('z.\((.*)\)')
            c_term = re.compile('c.\((.*)\)')
            formula = [t.strip() for t in formula.strip().split() if t.strip() != '']
            for i in range(len(formula)):
                formula[i] = z_term.sub(r'scale(\1)', formula[i])
                formula[i] = c_term.sub(r'scale(\1, scale=FALSE)', formula[i])
            formula = ' '.join(formula)

            if os.path.exists(p.outdir + '/' + m_path + '/m.obj'):
                stderr('Retrieving saved model %s...\n' % m)
                with open(p.outdir + '/' + m_path + '/m.obj', 'rb') as m_file:
                    gam = pickle.load(m_file)
            else:
                stderr('Fitting model %s...\n' % m)
                gam = GAM(formula, X_baseline, ran_gf=ran_gf)
                with open(p.outdir + '/' + m_path + '/m.obj', 'wb') as m_file:
                    pickle.dump(gam, m_file)

            gam_preds = gam.predict(X_baseline)
            gam_mse = mse(Y[dv], gam_preds)
            gam_mae = mae(Y[dv], gam_preds)
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
            with open(p.outdir + '/' + m_path + '/summary.txt', 'w') as f_out:
                f_out.write(summary)
            stderr(summary)
            stderr('\n\n')

        else: # is CDR
            dv = [x.strip() for x in formula.strip().split('~')[0].strip().split('+')]
            Y_valid, select_Y_valid = filter_invalid_responses(Y, dv)

            stderr('\nInitializing model %s...\n\n' % m)

            kwargs = {}
            for kwarg in MODEL_INITIALIZATION_KWARGS:
                if kwarg.key not in ['outdir', 'history_length', 'future_length', 't_delta_cutoff']:
                    kwargs[kwarg.key] = p[kwarg.key]
            kwargs['crossval_factor'] = p['crossval_factor']
            kwargs['crossval_folds'] = p['crossval_folds']
            kwargs['crossval_fold'] = p['crossval_fold']
            kwargs['crossval_dev_fold'] = p['crossval_dev_fold']
            kwargs['irf_name_map'] = p.irf_name_map

            cdr_model = CDRModel(
                formula,
                X,
                Y_valid,
                ablated=p['ablated'],
                outdir=p.outdir + '/' + m_path,
                history_length=p.history_length,
                future_length=p.future_length,
                t_delta_cutoff=p.t_delta_cutoff,
                **kwargs
            )

            if args.save_and_exit:
                save = True
                if not args.skip_confirmation:
                    ans = input('Model initialized. Continue saving? [y]/n >>> ')
                    if ans.strip().lower() == 'n':
                        save = False
                if save:
                    stderr('Saving...\n')
                    cdr_model.save()
                    with open(cdr_model.outdir + '/initialization_summary.txt', 'w') as i_file:
                        i_file.write(cdr_model.initialization_summary())
                continue

            if cdr_model.eval_freq > 0:
                if (X_paths_dev is None or Y_paths_dev is None) and not (p['crossval_fold'] and p['crossval_dev_fold']):
                    X_paths_dev, Y_paths_dev = paths_from_partition_cliarg('dev', p)
                    for X_path in X_paths_dev:
                        if X_path is None:
                            raise ValueError('X_dev must be specified in order to use eval_freq > 0')
                    for Y_path in Y_paths_dev:
                        if Y_path is None:
                            raise ValueError('Y_dev must be specified in order to use eval_freq > 0')

                    assert X_paths_dev and Y_paths_dev, 'X_dev and Y_dev must be specified in order to use eval_freq > 0'
                    X_dev, Y_dev = read_tabular_data(
                        X_paths_dev,
                        Y_paths_dev,
                        p.series_ids,
                        sep=p.sep,
                        categorical_columns=list(
                            set(p.split_ids + p.series_ids + [v for x in cdr_formula_list for v in x.rangf]))
                    )
                    X_dev, Y_dev, select_dev, _ = preprocess_data(
                        X_dev,
                        Y_dev,
                        cdr_formula_list,
                        p.series_ids,
                        filters=p.filters,
                        history_length=p.history_length,
                        future_length=p.future_length,
                        t_delta_cutoff=p.t_delta_cutoff,
                        all_interactions=all_interactions
                    )

            stderr('\nFitting model %s...\n\n' % m)

            cdr_model.fit(
                X,
                Y_valid,
                X_dev=X_dev,
                Y_dev=Y_dev,
                n_iter=p['n_iter'],
                X_in_Y_names=X_in_Y_names,
                force_training_evaluation=args.force_training_evaluation,
                optimize_memory=args.optimize_memory
            )

            summary = cdr_model.summary()

            with open(p.outdir + '/' + m_path + '/summary.txt', 'w') as f_out:
                f_out.write(summary)
            stderr(summary)
            stderr('\n\n')

            cdr_model.finalize()

