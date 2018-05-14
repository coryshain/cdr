import argparse
import os
import sys
import itertools
import re
import pandas as pd
from dtsr.baselines import LM
import gc

from dtsr.config import Config
from dtsr.io import read_data
from dtsr.formula import Formula
from dtsr.data import preprocess_data, compute_splitID, compute_partition
from dtsr.util import mse, mae

pd.options.mode.chained_assignment = None

splitter = re.compile(' *[-+|:] *')
pred_name = re.compile('[^(]+\((.+)\)')

def clean_parens(l):
    for i, x in enumerate(l):
        x_new = x
        if x.startswith('('):
            x_new = x[1:]
        if x.endswith(')'):
            x_new = x[:-1]
        l[i] = x_new
    return l

def get_preds(bform):
    preds = set()
    l = bform.split('~')[1].strip()
    if l.startswith('('):
        l = splitter.split(l.strip())
        l_new = clean_parens(l)
        p_list = l
    else:
        p_list = splitter.split(l.strip())
    for p in p_list:
        name = p
        while pred_name.match(name):
            name = pred_name.match(name).group(1)
        if name not in preds:
            preds.add(name)
    return preds

def update_preds(preds, perm):
    preds_new = preds[:]
    for i,x in enumerate(preds):
        if perm[i] > 0:
            preds_new[i] = x + 'S%d' %perm[i]
    return preds_new

def permute_spillover(bform, preds, perms):
    forms = []
    for perm in perms:
        form_name = []
        for i in range(len(preds)):
            form_name.append(preds[i][:2] + str(perm[i]))
        preds_new = update_preds(preds, perm)
        l = bform
        for i in range(len(preds)):
            l = re.sub(r'([+^ (])'+preds[i]+'([+$ )])', r'\1'+preds_new[i]+r'\2', l)
        forms.append(''.join(l))
    return(forms)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Trains model(s) from formula string(s) given data.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    models = p.model_list[:]
    lm_formula = p.models['LMnoS_noRE']['formula']

    preds = get_preds(lm_formula)
    preds = list(preds)
    preds.sort(reverse=True, key=lambda x: len(x))
    n_pred = len(preds)

    perms = list(itertools.product(range(0, 4), repeat=n_pred))

    forms = permute_spillover(lm_formula, preds, perms)

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in p.model_list if m.startswith('DTSR')]
    dtsr_formula_name_list = [m for m in p.model_list if m.startswith('DTSR')]
    X, y = read_data(p.X_train, p.y_train, p.series_ids, categorical_columns=list(set(p.series_ids + [v for x in dtsr_formula_list for v in x.rangf])))
    X, y, select, X_2d_predictor_names, X_2d_predictors = preprocess_data(X, y, p, dtsr_formula_list, compute_history=False)

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
        if x.dv not in X_baseline.columns:
            X_baseline[x.dv] = y[x.dv]

    for c in X_baseline.columns:
        if X_baseline[c].dtype.name == 'category':
            X_baseline[c] = X_baseline[c].astype(str)
    X_baseline = py2ri(X_baseline)

    if not os.path.exists(p.logdir + '/spillover/'):
        os.makedirs(p.logdir + '/spillover/')

    for formula in forms:
        m = '_'.join(sorted(list(get_preds(formula))))
        dv = formula.strip().split('~')[0].strip()

        if os.path.exists(p.logdir + '/spillover/' + m + '.txt'):
            sys.stderr.write('Model %s already exists. Skipping...\n' % m)
            continue
        else:
            sys.stderr.write('Fitting model %s...\n' % m)
            lm = LM(formula, X_baseline)
            gc.collect()

        lm_preds = lm.predict(X_baseline)
        lm_mse = mse(y[dv], lm_preds)
        lm_mae = mae(y[dv], lm_preds)
        summary = '=' * 50 + '\n'
        summary += 'Linear regression\n\n'
        summary += 'Model name: %s\n\n' % m
        summary += 'Formula:\n'
        summary += '  ' + formula + '\n'
        summary += str(lm.summary()) + '\n'
        summary += 'Training set loss:\n'
        summary += '  MSE: %.4f\n' % lm_mse
        summary += '  MAE: %.4f\n' % lm_mae
        summary += '=' * 50 + '\n'
        with open(p.logdir + '/spillover/' + m + '.txt', 'w') as f_out:
            f_out.write(summary)
        sys.stderr.write(summary)
        sys.stderr.write('\n\n')