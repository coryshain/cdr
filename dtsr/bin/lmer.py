import argparse
import sys
import os
import pickle
import pandas as pd
from dtsr.config import Config
from dtsr.baselines import py2ri, LME, inspect_df
from dtsr.formula import Formula
from dtsr.util import mse, mae

pd.options.mode.chained_assignment = None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Run LME follow-up regression on DTSR models in config for which convolved input data has been generated. 
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-z', '--zscore', action='store_true', help='Z-transform (center and scale) the convolved predictors prior to fitting')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    models = [x for x in models if x.startswith('DTSR_')]

    for m in models:
        dir_path = p.outdir + '/' + m
        data_path = dir_path + '/X_conv_' + args.partition + '.csv'

        if os.path.exists(data_path):
            p.set_model(m)
            f = Formula(p['formula'])
            lmeform = f.to_lmer_formula_string(z=args.zscore)

            df = pd.read_csv(data_path, sep=' ', skipinitialspace=True)
            for c in df.columns:
                if df[c].dtype.name == 'object':
                    df[c] = df[c].astype(str)
            df_r = py2ri(df)

            dv = f.dv

            if os.path.exists(dir_path + '/lmer_%s.obj' %args.partition):
                sys.stderr.write('Retrieving saved LMER regression of DTSR model %s...\n' % m)
                with open(dir_path + '/lmer_%s.obj' %args.partition, 'rb') as m_file:
                    lme = pickle.load(m_file)
            else:
                sys.stderr.write('Fitting LMER regression of DTSR model %s...\n' % m)
                lme = LME(lmeform, df_r)
                with open(dir_path + '/lmer_%s.obj' %args.partition, 'wb') as m_file:
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
            with open(p.outdir + '/' + m + '/lmer_%s_summary.txt' %args.partition, 'w') as f_out:
                f_out.write(summary)
            sys.stderr.write(summary)
            sys.stderr.write('\n\n')



