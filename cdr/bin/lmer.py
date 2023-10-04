import argparse
import sys
import os
import pickle
import pandas as pd
from cdr.config import Config
from cdr.baselines import py2ri, LM, LME
from cdr.formula import Formula
from cdr.util import mse, mae, filter_models, get_partition_list, stderr

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Run LME follow-up regression on CDR models in config for which convolved input data has been generated. 
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names to use for LMER fitting. Regex permitted. If unspecified, fits LMER to all CDR models.')
    argparser.add_argument('-p', '--partition', type=str, default='train', help='Name of partition to use ("train", "dev", "test", or space- or hyphen-delimited subset of these)')
    argparser.add_argument('-z', '--zscore', action='store_true', help='Z-transform (center and scale) the convolved predictors prior to fitting')
    argparser.add_argument('-u', '--uncorrelated', action='store_true', help='Use uncorrelated random intercepts and slopes. Simplifies the model and can help avoid convergence problems.')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Fit ablated models to data convolved using the ablated model. Otherwise fits ablated models to data convolved using the full model.')
    argparser.add_argument('-f', '--force', action='store_true', help='Refit and overwrite any previously trained models. Otherwise, previously trained models are skipped.')
    args = argparser.parse_args()

    for path in args.config_paths:

        p = Config(path)

        models = filter_models(p.model_names, args.models, cdr_only=True)

        models = [x for x in filter_models(models, cdr_only=True)]

        partitions = get_partition_list(args.partition)
        partition_str = '-'.join(partitions)

        for m in models:
            m_path = m.replace(':', '+')
            dir_path = p.outdir + '/' + m_path
            if args.ablated_models:
                data_path = dir_path + '/X_conv_' + partition_str + '.csv'
            else:
                data_path = p.outdir + '/' + m_path.split('!')[0] + '/X_conv_' + partition_str + '.csv'

            stderr('Two-step analysis using data file %s\n' %data_path)

            if os.path.exists(data_path):
                p.set_model(m)
                f = Formula(p['formula'])
                model_form = f.to_lmer_formula_string(z=args.zscore, correlated=not args.uncorrelated)
                model_form = model_form.replace('-', '_')

                is_lme = '|' in model_form

                df = pd.read_csv(data_path, sep=' ', skipinitialspace=True)
                for c in df.columns:
                    if df[c].dtype.name == 'object':
                        df[c] = df[c].astype(str)

                new_cols = []
                for c in df.columns:
                    new_cols.append(c.replace('-', '_'))
                df.columns = new_cols

                df_r = py2ri(df)

                dv = f.dv

                model_path = dir_path + '/lm_%s.obj' % ('z_%s' % partition_str if args.zscore else partition_str)
                model_summary_path = dir_path + '/lm_%s_summary.txt' % ('z_%s' % partition_str if args.zscore else partition_str)

                if not args.force and os.path.exists(model_path):
                    stderr('Retrieving saved L(ME) regression of CDR model %s...\n' % m)
                    with open(model_path, 'rb') as m_file:
                        lm = pickle.load(m_file)
                else:
                    stderr('Fitting L(ME) regression of CDR model %s...\n' % m)
                    if is_lme:
                        lm = LME(model_form, df_r)
                    else:
                        lm = LM(model_form, df_r)
                    with open(model_path, 'wb') as m_file:
                        pickle.dump(lm, m_file)

                lm_preds = lm.predict(df_r)
                lm_mse = mse(df[dv], lm_preds)
                lm_mae = mae(df[dv], lm_preds)
                summary = '=' * 50 + '\n'
                summary += '%s regression\n\n' % ('LME' if is_lme else 'Linear')
                summary += 'Model name: %s\n\n' % m
                summary += 'Formula:\n'
                summary += '  ' + model_form + '\n'
                summary += str(lm.summary()) + '\n'
                summary += 'Training set loss:\n'
                summary += '  MSE: %.4f\n' % lm_mse
                summary += '  MAE: %.4f\n' % lm_mae
                summary += '=' * 50 + '\n'
                with open(model_summary_path, 'w') as f_out:
                    f_out.write(summary)
                stderr(summary)
                stderr('\n\n')

