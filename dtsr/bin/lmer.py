import argparse
import sys
import os
import pickle
import pandas as pd
from dtsr.config import Config
from dtsr.baselines import py2ri, LME
from dtsr.formula import Formula
from dtsr.util import mse, mae, filter_models

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Run LME follow-up regression on DTSR models in config for which convolved input data has been generated. 
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names to use for LMER fitting. Regex permitted. If unspecified, fits LMER to all DTSR models.')
    argparser.add_argument('-p', '--partition', type=str, default='train', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-z', '--zscore', action='store_true', help='Z-transform (center and scale) the convolved predictors prior to fitting')
    argparser.add_argument('-A', '--ablated_models', action='store_true', help='Fit ablated models to data convolved using the ablated model. Otherwise fits ablated models to data convolved using the full model.')
    argparser.add_argument('-f', '--force', action='store_true', help='Refit and overwrite any previously trained models. Otherwise, previously trained models are skipped.')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    models = filter_models(p.model_list, args.models, dtsr_only=True)

    models = [x for x in models if x.startswith('DTSR_')]

    for m in models:
        dir_path = p.outdir + '/' + m
        if args.ablated_models:
            data_path = dir_path + '/X_conv_' + args.partition + '.csv'
        else:
            data_path = p.outdir + '/' + m.split('!')[0] + '/X_conv_' + args.partition + '.csv'

        sys.stderr.write('Two-step analysis using data file %s\n' %data_path)

        if os.path.exists(data_path):
            p.set_model(m)
            f = Formula(p['formula'])
            lmeform = f.to_lmer_formula_string(z=args.zscore)
            lmeform = lmeform.replace('-', '_')

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

            model_path = dir_path + '/lmer_%s.obj' %args.partition
            model_summary_path = dir_path + '/lmer_%s_summary.txt' %args.partition

            if not args.force and os.path.exists(model_path):
                sys.stderr.write('Retrieving saved LMER regression of DTSR model %s...\n' % m)
                with open(model_path, 'rb') as m_file:
                    lme = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting LMER regression of DTSR model %s...\n' % m)
                lme = LME(lmeform, df_r)
                with open(model_path, 'wb') as m_file:
                    pickle.dump(lme, m_file)

            lme_preds = lme.predict(df_r)
            lme_mse = mse(df[dv], lme_preds)
            lme_mae = mae(df[dv], lme_preds)
            summary = '=' * 50 + '\n'
            summary += 'LME regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + lmeform + '\n'
            summary += str(lme.summary()) + '\n'
            summary += 'Training set loss:\n'
            summary += '  MSE: %.4f\n' % lme_mse
            summary += '  MAE: %.4f\n' % lme_mae
            summary += '=' * 50 + '\n'
            with open(model_summary_path, 'w') as f_out:
                f_out.write(summary)
            sys.stderr.write(summary)
            sys.stderr.write('\n\n')

