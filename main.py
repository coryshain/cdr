from __future__ import print_function
import sys
import os
import shutil
import yaml
import math
import time
import pickle
import numpy as np
from numpy import inf, nan
import pandas as pd
pd.options.mode.chained_assignment = None
import configparser
import statsmodels.formula.api as smf
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

import tensorflow as tf
from tensorflow.python.platform.test import is_gpu_available

def bootstrap(true, preds_1, preds_2, err_type='mse', n_iter=10000, n_tails=2):
    pb = tf.contrib.keras.utils.Progbar(n_iter)
    err_table = np.stack([preds_1, preds_2], 1)
    err_table -= np.expand_dims(true, -1)
    if err_type.lower == 'mae':
        err_table = err_table.abs()
    else:
        err_table *= err_table
    base_diff = err_table[:,0].mean() - err_table[:,1].mean()
    hits = 0
    for i in range(n_iter):
        shuffle = (np.random.random((len(true))) > 0.5).astype('int')
        m1 = err_table[np.arange(len(err_table)),shuffle]
        m2 = err_table[np.arange(len(err_table)),1-shuffle]
        cur_diff = m1.mean() - m2.mean()
        if n_tails == 1:
            if base_diff < 0 and cur_diff <= base_diff:
                hits += 1
            elif base_diff > 0 and cur_diff >= base_diff:
                hits += 1
            elif base_diff == 0:
                hits += 1
        elif n_tails == 2:
            if math.fabs(cur_diff) > math.fabs(base_diff):
                hits += 1
        else:
            raise ValueError('Invalid bootstrap parameter n_tails: %s. Must be in {0, 1}.' %n_tails)
        pb.update(i+1, force=True)

    p = float(hits+1)/(n_iter+1)
    print()
    return p

def parse_var_inner(var):
    if var.strip().endswith(')'):
        op = ''
        inp = ''
        inside_var = False
        for i in var[:-1]:
            if inside_var:
                inp += i
            else:
                if i == '(':
                    inside_var = True
                else:
                    op += i
    else:
        op = ''
        inp = var
    return op, inp

def parse_var(var):
    n_lp = var.count('(')
    n_rp = var.count(')')
    assert n_lp == n_rp, 'Unmatched parens in formula variable "%s".' %var
    ops = [[], var.strip()]
    while ops[1].endswith(')'):
        op, inp = parse_var_inner(ops[1])
        ops[0].insert(0, op)
        ops[1] = inp.strip()
    return ops

def extract_cross(var):
    vars = var.split(':')
    if len(vars) > 1:
        vars = [v.strip() for v in vars]
    return vars

def extract_interaction(var):
    vars = var.split('*')
    if len(vars) > 1:
        vars = [v.strip() for v in vars]
    return vars

def process_interactions(vars):
    new_vars = []
    interactions = []
    for v1 in vars:
        new_v = extract_cross(v1)
        if len(new_v) > 0:
            interactions.append(new_v)
        new_v = extract_interaction(v1)
        if len(new_v) > 0:
            interactions.append(new_v)
            for v2 in new_v:
                if v2 not in new_vars:
                    new_vars.append(v2)
    return new_vars, interactions

def apply_op(op, arr):
    if op == 'c':
        out = c(arr)
    elif op == 'z':
        out = z(arr)
    elif op == 'log':
        out = np.log(arr)
    elif op == 'log1p':
        out = np.log(arr + 1)
    elif op.startswith('pow'):
        exponent = float(op[3:])
        out = arr ** exponent
    elif op.startswith('bc'):
        L = float(op[2:])
        if L == 0:
            out = np.log(arr)
        else:
            out = (arr ** L - 1) / L
    else:
        sys.stderr.write('Ignoring unrecognized op "%s" for column "%s".\n' %(op, col))
        out = arr
    return out

def apply_ops(ops, col, df):
    if len(ops[0]) > 0:
        new_col = ops[-1]
        for op in ops[0]:
            new_col = op + '(' + new_col + ')'
        df[new_col] = df[col]
        for op in ops[0]:
            df[new_col] = apply_op(op, df[new_col])

def apply_ops_from_str(s, df):
    ops = parse_var(s)
    col = ops[1]
    apply_ops(ops, col, df)

def parse_formula(s):
    n_lp = s.count('(')
    n_rp = s.count(')')
    assert n_lp == n_rp, 'Unmatched parens in formula "%s".' %s

    m = {'dv': '', 'fixed': [''], 'random': []}
    dv, s = s.strip().split('~')
    m['dv'] = dv.strip()

    in_random_term = False
    in_grouping_factor = False
    for c in s.strip():
        if c == '(' and m['fixed'][-1] == '':
            in_random_term = True
            if m['fixed'][-1] == '':
                m['fixed'].pop(-1)
            m['random'].append({'vars': [''], 'grouping_factor': ''})
        elif c == ')' and in_random_term:
            n_lb = m['random'][-1]['vars'][-1].count('(')
            n_rb = m['random'][-1]['vars'][-1].count(')')
            if n_lb == n_rb:
                in_random_term = False
                in_grouping_factor = False
            else:
                m['random'][-1]['vars'][-1] += c
        elif c in [' ', '+', '|']:
            if c == '+':
                if in_random_term:
                    m['random'][-1]['vars'].append('')
                else:
                    m['fixed'].append('')
            elif c == '|':
                in_grouping_factor = True
        else:
            if in_random_term:
                if in_grouping_factor:
                    m['random'][-1]['grouping_factor'] += c
                else:
                    m['random'][-1]['vars'][-1] += c
            else:
                m['fixed'][-1] += c
    for i in range(len(m['random'])):
        if '0' in m['random'][i]['vars']:
            m['random'][i]['Intercept'] = False
        else:
            m['random'][i]['Intercept'] = True
        m['random'][i]['vars'] = [v for v in m['random'][i]['vars'] if v not in ['0', '1']]



    ## Random effects sanity check
    weird_random = []
    for r in m['random']:
        if r['grouping_factor'] != 'subject' and len(r['vars']) > 0:
            weird_random.append(r['grouping_factor'])
    if len(weird_random) > 0:
        sys.stderr.write('\nWARNING: Model contains random slopes for one or more grouping factors other than "subject".\n' +
            'Since independent variables in DTSR are temporally convolved, this means that the model will\n' +
            'learn group-specific coefficients on the history of the independent variable(s) with random\n' +
            'slope(s). This is almost never what you want. Make sure your model is specified as intended.\n\n'
        )
    return m

def apply_formula(bform_parsed, df):
    apply_ops_from_str(bform_parsed['dv'], df)
    for f in bform_parsed['fixed']:
        apply_ops_from_str(f, df)
    for r in bform_parsed['random']:
        for v in r['vars']:
            apply_ops_from_str(v, df)

def names2ix(names, ref):
    if type(names) is not list:
        names = [names]
    ix = []
    for n in names:
        ix.append(ref.index(n))
    if len(ix) > 1:
        return np.array(ix)
    return ix[0]

def names2mask(names, ref):
    mask = np.zeros((1,len(ref)))
    if type(names) is not list:
        names = [names]
    for n in names:
        mask[0,ref.index(n)] = 1.
    return mask.astype('int32')

def getRandomPermutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv

def z(df):
    return (df-df.mean(axis=0))/df.std(axis=0)

def c(df):
    return df-df.mean(axis=0)


def plot_convolutions(plot_x, plot_y, features, dir, filename='convolution_plot.jpg'):
    n_feat = plot_y.shape[-1]
    for i in range(n_feat):
        plt.plot(plot_x, plot_y[:,i], label=features[i])
    plt.legend()
    plt.savefig(dir+'/'+filename)
    plt.clf()

def mse(true, preds):
    return ((true-preds)**2).mean()

def mae(true, preds):
    return (true-preds).abs().mean()

def print_tee(s, file_list):
    for f in file_list:
        print(s, file=f)

def powerset(n):
    return np.indices([2]*n).reshape(n, -1).T[1:].astype('int32')

def compute_history_intervals(y, X, check_output=False):
    pb = tf.contrib.keras.utils.Progbar(len(y))

    subject_y = np.array(y.subject.cat.codes)
    docid_y = np.array(y.docid.cat.codes)
    time_y = np.array(y.time)

    subject_X = np.array(X.subject.cat.codes)
    docid_X = np.array(X.docid.cat.codes)
    time_X = np.array(X.time)

    subject = subject_y[0]
    docid = docid_y[0]

    first_obs = np.zeros(len(y)).astype('int32')
    last_obs = np.zeros(len(y)).astype('int32')

    i = j = 0
    start = 0
    end = 0
    while i < len(y) and j < len(X):
        if subject_y[i] != subject or docid_y[i] != docid:
            start = end = i
            subject = subject_y[i]
            docid = docid_y[i]
        while j < len(X) and time_X[j] <= time_y[i] and subject_X[j] == subject and docid_X[j] == docid:
            end += 1
            j += 1
        if check_output:
            assert subject_X[start] == subject_y[i], 'Subject mismatch at start: y = %s, X = %s at y index %d' %(subject_X[start], subject_y[i], i)
            assert docid_X[start] == docid_y[i], 'Docid mismatch at start: y = %s, X = %s at y index %d' %(docid_X[start], docid_y[i], i)
            assert subject_X[end-1] == subject_y[i], 'Subject mismatch at end: y = %s, X = %s at y index %d' % (subject_X[end-1], subject_y[i], i)
            assert docid_X[end-1] == docid_y[i], 'Docid mismatch at end: y = %s, X = %s at y index %d' % (docid_X[end-1], docid_y[i], i)
            assert start < end, 'Start == end disallowed. Both had value %d at y index %d' %(start, i)
        first_obs[i] = start
        last_obs[i] = end
        pb.update(i, force=True)

        i += 1
    return first_obs, last_obs

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    ## Required
    path_X = config.get('settings', 'path_X')
    path_y = config.get('settings', 'path_y')
    bform = config.get('settings', 'bform')

    ## Optional
    logdir = config.get('settings', 'logdir', fallback='log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy2(sys.argv[1], logdir + '/config.ini')
    modality = config.get('settings', 'modality', fallback='ET')
    network_type = config.get('settings', 'network_type', fallback='mle')
    conv_func = config.get('settings', 'conv_func', fallback='gamma')
    partition = config.get('settings', 'partition', fallback='train')
    loss = config.get('settings', 'loss', fallback='MSE')
    window_length = int(config.get('settings', 'window_length', fallback=100))
    modulus = config.getint('settings', 'modulus', fallback=3)
    cv_modulus = config.getint('settings', 'cv_modulus', fallback=5)
    bform_LM = config.get('settings', 'bform_LM', fallback='')
    baseline_LM = len(bform_LM) > 0
    bform_LME = config.get('settings', 'bform_LME', fallback='')
    baseline_LME = len(bform_LME) > 0
    bform_GAM = config.get('settings', 'bform_GAM', fallback='')
    baseline_GAM = len(bform_GAM) > 0
    filter_sents = config.getboolean('settings', 'filter_sents', fallback=False)
    filter_lines = config.getboolean('settings', 'filter_lines', fallback=False)
    filter_screens = config.getboolean('settings', 'filter_screens', fallback=False)
    filter_files = config.getboolean('settings', 'filter_files', fallback=False)
    attribution_model = config.getboolean('settings', 'attribution_model', fallback=False)
    random_subjects_model = config.getboolean('settings', 'random_subjects_model', fallback=True)
    random_subjects_conv_params = config.getboolean('settings', 'random_subjects_conv_params', fallback=False)
    log_random = config.getboolean('settings', 'log_random', fallback=False)
    log_convolution_plots = config.getboolean('settings', 'log_convolution_plots', fallback=False)
    n_epoch_train = config.getint('settings', 'n_epoch_train', fallback=50)
    n_epoch_finetune = config.getint('settings', 'n_epoch_finetune', fallback=250)

    sys.stderr.write('Loading data...\n')
    X = pd.read_csv(path_X, sep=' ', skipinitialspace=True)
    y = pd.read_csv(path_y, sep=' ', skipinitialspace=True)

    sys.stderr.write('Pre-processing data...\n')
    if modality == 'ET':
        y.rename(columns={'fdurGP': 'fdur'}, inplace=True)

    y = y[y.fdur.notnull() & (y.fdur > 0)]

    X.subject = X.subject.astype('category')
    X.docid = X.docid.astype('category')
    X.sentid = X.sentid.astype('category')

    y.subject = y.subject.astype('category')
    y.docid = y.docid.astype('category')
    y.sentid = y.sentid.astype('category')

    X._get_numeric_data().fillna(value=0, inplace=True)

    bform_parsed = parse_formula(bform)
    dv = bform_parsed['dv']
    fixef = bform_parsed['fixed']
    ransl = []
    for f in bform_parsed['random']:
        for v in f['vars']:
            if f not in ransl:
                ransl.append(v)
    rangf = []
    for f in bform_parsed['random']:
        if f['grouping_factor'] not in rangf:
            rangf.append(f['grouping_factor'])
    allsl = fixef[:]
    for f in ransl:
        if f not in allsl:
            allsl.append(f)
    allvar = allsl[:]
    for gf in rangf:
        if gf not in allvar:
            allvar.append(gf)

    apply_formula(bform_parsed, X)

    X_features = ['time'] + allvar
    y_features = [dv] + ['time', 'first_obs', 'last_obs', 'subject', 'docid', 'sentid']
    if 'correct' in X.columns:
        y_features.append('correct')
    if filter_sents and 'startofsentence' in X.columns and 'endofsentence' in X.columns:
        y_features += ['startofsentence', 'endofsentence']
    if filter_lines and 'startofline' in X.columns and 'endofline' in X.columns:
        y_features += ['startofline', 'endofline']
    if filter_screens and 'startofscreen' in X.columns and 'endofscreen' in X.columns:
        y_features += ['startofscreen', 'endofscreen']
    if filter_files and 'startoffile' in X.columns and 'endoffile' in X.columns:
        y_features += ['startoffile', 'endoffile']

    for f in ransl:
        if f not in y_features:
            y_features.append(f)
    for gf in rangf:
        if gf not in y_features:
            y_features.append(gf)

    for gf in rangf:
        X[gf] = X[gf].astype('category')
        y[gf] = y[gf].astype('category')

    gf_y = y[rangf]
    for r in rangf:
        gf_y[r] = gf_y[r].cat.codes
    X[allsl] = z(X[allsl])

    print('Computing history intervals for each regression target...')
    first_obs, last_obs = compute_history_intervals(y, X)
    y['first_obs'] = first_obs
    y['last_obs'] = last_obs
    sample = np.random.randint(0, len(y), 10)
    if False:
        for i in sample:
            print(i)
            row = y.iloc[i]
            print(row)
            print(X[row.first_obs:row.last_obs])

    if len(bform_parsed['random']) > 0:
        n_level = []
        for t in bform_parsed['random']:
            gf = t['grouping_factor']
            n_level.append(len(y[gf].cat.categories))

    select = y[dv] > 100
    select &= y[dv] < 3000
    if 'correct' in y.columns:
        select &= y.correct > 4

    if filter_sents:
        if 'startofsentence' in y.columns:
            select &= y.startofsentence != 1
        if 'endofsentence' in y.columns:
            select &= y.endofsentence != 1
    if filter_sents:
        if 'startofline' in y.columns:
            select &= y.startofline != 1
        if 'endofline' in y.columns:
            select &= y.endofline != 1
    if filter_screens:
        if 'startofscreen' in y.columns:
            select &= y.startofscreen != 1
        if 'endofscreen' in y.columns:
            select &= y.endofscreen != 1
    if filter_files:
        if 'startoffile' in y.columns:
            select &= y.startoffile != 1
        if 'endoffile' in y.columns:
            select &= y.endoffile != 1

    if partition == 'test':
        select &= ((y.subject.cat.codes+1)+y.sentid.cat.codes) % modulus == modulus-1
    elif partition == 'dev':
        select &= ((y.subject.cat.codes+1)+y.sentid.cat.codes) % modulus == modulus-2
    elif partition == 'train':
        select &= ((y.subject.cat.codes+1)+y.sentid.cat.codes) % modulus < modulus-2
        select_train = select & (((y.subject.cat.codes+1)+y.sentid.cat.codes) % cv_modulus < cv_modulus-1)
        select_cv = select & (((y.subject.cat.codes+1)+y.sentid.cat.codes) % cv_modulus == cv_modulus-1)
    else:
        raise ValueError('Partition type %s not supported (must be one of ["train", "dev", "test"]')

    if partition == 'train':
        y_train = y[select_train]
        y_train[dv] = c(y_train[dv])
        gf_y_train = gf_y[select_train]
        y_cv = y[select_cv]
        y_cv[dv] = c(y_cv[dv])
        gf_y_cv = gf_y[select_cv]


        n_train_sample = len(y_train)
        n_cv_sample = len(y_cv)
    else:
        y[dv] = c(y[dv])
        n_train_sample = len(y)

    print()
    print('Evaluating on %s partition' %partition)
    print('Number of training samples: %d' %n_train_sample)
    if partition == 'train':
        #pass
        print('Number of cross-validation samples: %d' %n_cv_sample)
    print()

    print('Correlation matrix (training data only):')
    rho = X[allsl].corr()
    print(rho)
    print()

    if partition == 'train':
        if baseline_LM or baseline_LME or baseline_GAM:
            X_train = X[select_train]
            X_train[dv] = y_train[dv]
            X_cv = X[select_cv]
            X_cv[dv] = y_cv[dv]
        if baseline_LM:
            if not os.path.exists('baselines'):
                os.makedirs('baselines')
            if not os.path.exists('baselines/LM'):
                os.makedirs('baselines/LM')

            print('Getting linear regression baseline...')
            print()

            with open('baselines/LM/summary.txt', 'w') as f:
                print()

                _, X_LM_train = patsy.dmatrices(bform_LM, X_train, return_type='dataframe')
                _, X_LM_cv = patsy.dmatrices(bform_LM, X_cv, return_type='dataframe')

                lm = smf.ols(formula=bform_LM, data=X_train)
                lm_results = lm.fit()

                print_tee('='*50, [sys.stdout, f])
                print_tee('Linear regression baseline results summary', [sys.stdout, f])
                print_tee('Results summary:', [sys.stdout, f])
                print_tee('', [sys.stdout, f])
                print_tee('Betas:', [sys.stdout, f])
                print_tee(lm_results.params, [sys.stdout, f])
                print_tee('', [sys.stdout, f])

                lm_preds_train = lm.predict(lm_results.params, X_LM_train)
                lm_mse_train = mse(y_train[dv], lm_preds_train)
                lm_mae_train = mae(y_train[dv], lm_preds_train)
                print_tee('Training set loss:', [sys.stdout, f])
                print_tee('  MSE: %.4f' % lm_mse_train, [sys.stdout, f])
                print_tee('  MAE: %.4f' % lm_mae_train, [sys.stdout, f])
                print_tee('', [sys.stdout, f])

                lm_preds_cv = lm.predict(lm_results.params, X_LM_cv)
                lm_mse_cv = mse(y_cv[dv], lm_preds_cv)
                lm_mae_cv = mae(y_cv[dv], lm_preds_cv)
                print_tee('Cross-validation set loss:', [sys.stdout, f])
                print_tee('  MSE: %.4f' % lm_mse_cv, [sys.stdout, f])
                print_tee('  MAE: %.4f' % lm_mae_cv, [sys.stdout, f])
                print_tee('='*50, [sys.stdout, f])

                print()


        if baseline_LME or baseline_GAM:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri
            import rpy2
            pandas2ri.activate()

            if baseline_LME:
                if not os.path.exists('baselines'):
                    os.makedirs('baselines')
                if not os.path.exists('baselines/LME'):
                    os.makedirs('baselines/LME')

                print('Getting linear mixed-effects regression baseline...')
                print()

                with open('baselines/LME/summary.txt', 'w') as f:
                    lmer = importr('lme4')

                    rstring = '''
                        function(bform, df) {
                            data_frame = df
                            m = lmer(bform, data=data_frame, REML=FALSE)
                            return(m)
                        }
                    '''

                    regress_lme = robjects.r(rstring)

                    rstring = '''
                        function(model) {
                            s = summary(model)
                            convWarn <- model@optinfo$conv$lme4$messages
                            if (is.null(convWarn)) {
                                convWarn <- 'No convergence warnings.'
                            }
                            s$convWarn = convWarn
                            return(s)
                        }
                    '''

                    get_model_summary = robjects.r(rstring)

                    rstring = '''
                        function(model, df) {
                            preds <- predict(model, df, allow.new.levels=TRUE)
                            return(preds)
                        }
                    '''

                    predict = robjects.r(rstring)

                    if os.path.exists('baselines/LME/lme_fit.obj'):
                        with open('baselines/LME/lme_fit.obj', 'rb') as f2:
                            lme = pickle.load(f2)
                    else:
                        lme = regress_lme(bform_LME, X_train)
                        with open('baselines/LME/lme_fit.obj', 'wb') as f2:
                            pickle.dump(lme, f2)

                    print_tee('='*50, [sys.stdout, f])
                    print_tee('Linear mixed-effects regression baseline results summary', [sys.stdout, f])
                    print_tee('', [sys.stdout, f])

                    s = get_model_summary(lme)

                    print_tee(s, [sys.stdout, f])
                    print_tee('Convergence warnings:', [sys.stdout, f])
                    print_tee(s.rx2('convWarn'), [sys.stdout, f])
                    print_tee('', [sys.stdout, f])

                    lme_preds_train = predict(lme, X_train)
                    lme_mse_train = mse(y_train[dv], lme_preds_train)
                    lme_mae_train = mae(y_train[dv], lme_preds_train)
                    print_tee('Training set loss:', [sys.stdout, f])
                    print_tee('  MSE: %.4f' % lme_mse_train, [sys.stdout, f])
                    print_tee('  MAE: %.4f' % lme_mae_train, [sys.stdout, f])

                    lme_preds_cv = predict(lme, X_cv)
                    lme_mse_cv = mse(y_cv[dv], lme_preds_cv)
                    lme_mae_cv = mae(y_cv[dv], lme_preds_cv)
                    print_tee('Cross-validation set loss:', [sys.stdout, f])
                    print_tee('  MSE: %.4f' % lme_mse_cv, [sys.stdout, f])
                    print_tee('  MAE: %.4f' % lme_mae_cv, [sys.stdout, f])
                    print_tee('='*50, [sys.stdout, f])

            if baseline_GAM:
                if not os.path.exists('baselines'):
                    os.makedirs('baselines')
                if not os.path.exists('baselines/GAM'):
                    os.makedirs('baselines/GAM')

                print('Getting GAM regression baseline...')
                print()

                with open('baselines/GAM/summary.txt', 'w') as f:
                    mgcv = importr('mgcv')

                    rstring = '''
                        function(bform, df) {
                            data_frame = df
                            m = gam(as.formula(bform), data=data_frame, drop.unused.levels=FALSE)
                            return(m)
                        }
                    '''

                    regress_gam = robjects.r(rstring)

                    rstring = '''
                        function(model) {10000
                            s = summary(model)
                            return(s)
                        }
                    '''

                    get_model_summary = robjects.r(rstring)

                    rstring = '''
                        function(model, df) {
                            preds <- predict(model, df)
                            return(preds)
                        }
                    '''

                    predict = robjects.r(rstring)

                    if os.path.exists('baselines/GAM/gam_fit.obj'):
                        with open('baselines/GAM/gam_fit.obj', 'rb') as f2:
                            gam = pickle.load(f2)
                    else:
                        gam = regress_gam(bform_GAM, X_train)
                        with open('baselines/GAM/gam_fit.obj', 'wb') as f2:
                            pickle.dump(gam, f2)

                    print_tee('='*50, [sys.stdout, f])
                    print_tee('GAM regression baseline results summary', [sys.stdout, f])
                    print_tee('', [sys.stdout, f])

                    s = get_model_summary(gam)
                    print_tee(s, [sys.stdout, f])

                    gam_preds_train = predict(gam, X_train)
                    gam_mse_train = mse(y_train[dv], gam_preds_train)
                    gam_mae_train = mae(y_train[dv], gam_preds_train)
                    print_tee('Training set loss:', [sys.stdout, f])
                    print_tee('  MSE: %.4f' % gam_mse_train, [sys.stdout, f])
                    print_tee('  MAE: %.4f' % gam_mae_train, [sys.stdout, f])

                    ## MGCV can't handle subjects that were unseen in training, so we filter them out from the CV eval
                    if 'subject' in bform_GAM:
                        select_gam_cv = y_cv.subject.isin(y_train.subject.unique())
                    if 'word' in bform_GAM:
                        select_gam_cv &= y_cv.word.isin(y_train.word.unique())
                    X_gam_cv = X_cv[select_gam_cv]
                    y_gam_cv = y_cv[select_gam_cv]
                    gam_preds_cv = predict(gam, X_gam_cv)
                    gam_mse_cv = mse(y_gam_cv[dv], gam_preds_cv)
                    gam_mae_cv = mae(y_gam_cv[dv], gam_preds_cv)
                    print_tee('Cross-validation set loss:', [sys.stdout, f])
                    print_tee('  MSE: %.4f' % gam_mse_cv, [sys.stdout, f])
                    print_tee('  MAE: %.4f' % gam_mae_cv, [sys.stdout, f])
                    print_tee('='*50, [sys.stdout, f])




    if network_type == 'bayesian':
        os.environ['THEANO_FLAGS'] = 'optimizer=None'
        from theano import tensor as T, function, printing
        import pymc3 as pm

        mu = np.zeros(n_feat)
        sigma = np.eye(n_feat) * 0.05

        def gamma(x, alpha, beta):
            x = np.expand_dims(np.array(x).astype('float'), -1) + 1
            out = x**(alpha-1)
            out *= beta**alpha
            out *= pm.math.exp(-beta*x)
            out /= T.gamma(beta)
            return out

        with pm.Model() as model:
            K = pm.Gamma('shape', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('shape', mu=mu, cov=sigma, shape=n_feat)
            rate = pm.Gamma('rate', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('rate', mu=mu, cov=sigma, shape=n_feat)
            beta = pm.Normal('beta', mu=0, sd=0.24, shape=n_feat)#pm.MvNormal('beta', mu=mu, cov=sigma, shape=n_feat)
            sigma = pm.HalfNormal('sigma', sd=1)
            intercept = pm.Normal('Intercept', y.mean(), sd=y.std())

            X_conv = conv(X[0].time - X[0].time, K, rate) * X[0][fixef]
            for i in range(1, len(X)):
                t_delta = X[0].time-X[i].time
                X_conv += conv(t_delta, K, rate) * X[i][fixef]
            E_fdur = intercept + pm.math.dot(X_conv, beta)

            fdur = pm.Normal('fdur', mu = E_fdur, sd = sigma, observed=y)

            trace = pm.sample(1000, tune=500)

            fig, ax = plt.subplots(5, 2)
            pm.traceplot(trace, ax=ax)
            plt.savefig('trace.jpg')

    elif network_type.lower() == 'mle':
        print()
        print('Fitting parameters using maximum likelihood...')

        learning_rate = 0.01

        sess = tf.Session()

        usingGPU = is_gpu_available()
        print('Using GPU: %s' %usingGPU)
        with sess.graph.as_default():
            if attribution_model:
                feature_powset = powerset(len(fixef))

            X_iv = tf.placeholder(shape=[None, len(allsl)], dtype=tf.float32, name='X_raw')
            time_X = tf.placeholder(shape=[None], dtype=tf.float32, name='time_X')

            y_ = tf.placeholder(shape=[None], dtype=tf.float32, name='y_')
            time_y = tf.placeholder(shape=[None], dtype=tf.float32, name='time_y')
            gf_y_ = tf.placeholder(shape=[None, len(rangf)], dtype=tf.int32, name='y_gf')
            first_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='first_obs')
            last_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='last_obs')

            if conv_func == 'exp':
                L_global = tf.Variable(tf.truncated_normal(shape=[1, len(allsl)], mean=0., stddev=.1, dtype=tf.float32), name='lambda')
            elif conv_func == 'gamma':
                log_k_global = tf.Variable(tf.truncated_normal(shape=[1, len(allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_k')
                log_theta_global = tf.Variable(tf.truncated_normal(shape=[1, len(allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_theta')
                log_neg_delta_global = tf.Variable(tf.truncated_normal(shape=[1, len(allsl)], mean=0., stddev=.1, dtype=tf.float32), name='log_neg_delta')

            intercept_global = tf.Variable(tf.constant(float(y_train[dv].mean()), shape=[1]), name='intercept')
            out = intercept_global

            ## Exponential convolution function
            if conv_func == 'exp':
                conv_global = lambda x: tf.exp(L_global) * tf.exp(-tf.exp(L_global) * x)
                conv = conv_global
            ## Gamma convolution function
            elif conv_func == 'gamma':
                conv_global_delta_zero = tf.contrib.distributions.Gamma(concentration=tf.exp(log_k_global[0]), rate=tf.exp(log_theta_global[0]), validate_args=True).prob
                conv_global = lambda x: conv_global_delta_zero(x + tf.exp(log_neg_delta_global)[0])
                conv_delta_zero = conv_global_delta_zero
                conv = conv_global

            def convolve_events(time_target, first_obs, last_obs):
                col_ix = names2ix(allsl, allsl)
                input_rows = tf.gather(X_iv[first_obs:last_obs], col_ix, axis=1)
                input_times = time_X[first_obs:last_obs]
                t_delta = time_target - input_times
                conv_coef = conv(tf.expand_dims(t_delta, -1))

                return tf.reduce_sum(conv_coef * input_rows, 0)

            X_conv = tf.map_fn(lambda x: convolve_events(*x), [time_y, first_obs, last_obs], dtype=tf.float32)

            if attribution_model:
                phi_global = tf.Variable(tf.truncated_normal(shape=[1, len(feature_powset)], stddev=0.1, dtype=tf.float32), name='phi')
                if random_subjects_model:
                    phi_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, len(feature_powset)], mean=0., stddev=.1, dtype=tf.float32), name='subject_phi_coef')
                    phi_correction_by_subject -= tf.reduce_mean(phi_correction_by_subject, axis=0)
                    phi = phi_global + phi_correction_by_subject
                else:
                    phi = phi_global
                out = tf.constant(0.)
                beta_global = tf.constant(0., shape=[1, len(allsl)])
                beta_global_marginal = tf.constant(0., shape=[1, len(allsl)])

                sess.run(tf.global_variables_initializer())

                for i in range(len(feature_powset)):
                    indices = np.where(feature_powset[i])[0].astype('int32')
                    vals = tf.gather(X_conv, indices, axis=1)
                    vals *= tf.expand_dims(tf.gather(phi, subject_ix)[:, i], -1) / feature_powset[i].sum()
                    vals = tf.reduce_sum(vals, axis=1)
                    out += vals

                    ## Record marginal betas
                    mask = tf.constant(feature_powset[i].astype('float32'))
                    beta_global += phi_global[0, i] * mask / feature_powset[i].sum()
                    beta_global_marginal += phi_global[0, i] * mask
            else:
                beta_global = tf.Variable(tf.truncated_normal(shape=[1, len(fixef)], mean=0., stddev=0.1, dtype=tf.float32), name='beta')
                fixef_ix = names2ix(fixef, allsl)
                out += tf.squeeze(tf.matmul(tf.gather(X_conv, fixef_ix, axis=1), tf.transpose(beta_global)), axis=-1)
                random_intercepts = []
                Z = []
                for i in range(len(rangf)):
                    r = bform_parsed['random'][i]
                    vars = r['vars']
                    n_var = len(vars)

                    if r['Intercept']:
                        random_intercept = tf.Variable(tf.truncated_normal(shape=[n_level[i]], mean=0., stddev=.1, dtype=tf.float32), name='intercept_by_%s' % r['grouping_factor'])
                        random_intercept -= tf.reduce_mean(random_intercept, axis=0)
                        random_intercepts.append(random_intercept)
                        out += tf.gather(random_intercept, gf_y_[:, i])

                    if len(vars) > 0:
                        random_effects_matrix = tf.Variable(tf.truncated_normal(shape=[n_level[i], n_var], mean=0., stddev=.1, dtype=tf.float32), name='random_slopes_by_%s' % (r['grouping_factor']))
                        random_effects_matrix -= tf.reduce_mean(random_effects_matrix, axis=0)
                        Z.append(random_effects_matrix)

                        ransl_ix = names2ix(vars, allsl)

                        out += tf.reduce_sum(tf.gather(X_conv, ransl_ix, axis=1) * tf.gather(random_effects_matrix, gf_y_[:, i]), axis=1)

            mae_loss = tf.losses.absolute_difference(y_, out)
            mse_loss = tf.losses.mean_squared_error(y_, out)
            if loss.lower() == 'mae':
                loss_func = mae_loss
            else:
                loss_func = mse_loss

            global_step = tf.Variable(0, name='global_step', trainable=False)
            incr_global_step = tf.assign(global_step, global_step+1)
            global_batch_step = tf.Variable(0, name='global_batch_step', trainable=False)
            #optim = tf.train.FtrlOptimizer(learning_rate)
            optim = tf.train.AdamOptimizer(learning_rate)
            #optim = tf.train.RMSPropOptimizer(learning_rate)
            #optim = tf.train.AdagradOptimizer(learning_rate)
            #optim = tf.train.AdadeltaOptimizer(learning_rate)
            #optim = tf.contrib.opt.NadamOptimizer()
            train_op = optim.minimize(loss_func, global_step=global_batch_step, name='optim')

            train_writer = tf.summary.FileWriter(logdir + '/train')
            cv_writer = tf.summary.FileWriter(logdir + '/cv')
            tf.summary.scalar('Intercept', intercept_global[0], collections=['params'])
            support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)
            for i in range(len(fixef)):
                tf.summary.scalar('beta/%s' % fixef[i], beta_global[0, i], collections=['params'])
                if attribution_model:
                    tf.summary.scalar('beta_marginal/%s' % fixef[i], beta_global_marginal[0, i], collections=['params'])
                if conv_func == 'exp':
                    tf.summary.scalar('log_lambda/%s' % fixef[i], L_global[0, i], collections=['params'])
                elif conv_func == 'gamma':
                    tf.summary.scalar('log_k/%s' % fixef[i], log_k_global[0, i], collections=['params'])
                    tf.summary.scalar('log_theta/%s' % fixef[i], log_theta_global[0, i], collections=['params'])
                    tf.summary.scalar('log_neg_delta/%s' % fixef[i], log_neg_delta_global[0, i], collections=['params'])
                if log_convolution_plots:
                    tf.summary.histogram('conv/%s' % fixef[i], conv_global(support)[:, i], collections=['params'])
            if attribution_model:
                for i in range(len(feature_powset)):
                    name = 'phi_' + '_'.join([fixef[j] for j in range(len(fixef)) if feature_powset[i, j] == 1])
                    tf.summary.scalar('phi/' + name, phi_global[0,i], collections=['params'])


            tf.summary.scalar('loss/%s'%loss, loss_func, collections=['loss'])
            summary_params = tf.summary.merge_all(key='params')
            summary_losses = tf.summary.merge_all(key='loss')
            if random_subjects_model and log_random:
                summary_by_subject = tf.summary.merge_all(key='by_subject')

            def parse_summary(s):
                s_str = tf.Summary()
                s_str.ParseFromString(s)
                s_dict = {}
                for val in s_str.value:
                    s_dict[val.tag] = val.simple_value
                s_dict_grouped = {'Intercept': s_dict['Intercept']}
                for k in s_dict:
                    if '/' in k and not k.startswith('conv'):
                        parent, key = k.strip().split('/')
                        val = s_dict[k]
                        if parent not in s_dict_grouped:
                            s_dict_grouped[parent] = {}
                        s_dict_grouped[parent][key] = val
                        if parent.startswith('log_neg_'):
                            unlog_name = parent[8:]
                            if unlog_name not in s_dict_grouped:
                                s_dict_grouped[unlog_name] = {}
                            unlog_val = -math.exp(val)
                            s_dict_grouped[unlog_name][key] = unlog_val
                        elif parent.startswith('log_'):
                            unlog_name = parent[4:]
                            if unlog_name not in s_dict_grouped:
                                s_dict_grouped[unlog_name] = {}
                            unlog_val = math.exp(val)
                            s_dict_grouped[unlog_name][key] = unlog_val
                return s_dict_grouped

            def print_summary(s_dict, f=None):
                s = yaml.dump(s_dict, default_flow_style=False)
                if f is None:
                    yaml.dump(s_dict, default_flow_style=False, stream=sys.stdout)
                else:
                    with sys.stdout if f is None else open(f, 'w') as F:
                        yaml.dump(s_dict, default_flow_style=False, stream=F)

            saver = tf.train.Saver()
            if os.path.exists(logdir + '/checkpoint'):
                saver.restore(sess, logdir + '/model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())

            print()
            print('='*50)
            print('Starting training')
            n_params = 0
            for v in sess.run(tf.trainable_variables()):
                n_params += np.prod(np.array(v).shape)
            print('Network contains %d trainable parameters' %n_params)
            print()

            fd_train = {}
            fd_train[X_iv] = X[allsl]
            fd_train[time_X] = X.time
            fd_train[y_] = y_train[dv]
            fd_train[time_y] = y_train.time
            fd_train[gf_y_] = gf_y_train
            fd_train[first_obs] = y_train.first_obs
            fd_train[last_obs] = y_train.last_obs

            fd_cv = {}
            fd_cv[X_iv] = X[allsl]
            fd_cv[time_X] = X.time
            fd_cv[y_] = y_cv[dv]
            fd_cv[time_y] = y_cv.time
            fd_cv[gf_y_] = gf_y_cv
            fd_cv[first_obs] = y_cv.first_obs
            fd_cv[last_obs] = y_cv.last_obs


            summary_cv_losses_batch, loss_cv = sess.run([summary_losses, loss_func], feed_dict=fd_cv)
            cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
            print('Initial CV loss: %.4f'%loss_cv)
            print()

            y_range = np.arange(len(y_train))

            while global_step.eval(session=sess) < n_epoch_train + n_epoch_finetune:
                if global_step.eval(session=sess) < n_epoch_train:
                    minibatch_size = 128
                    p, p_inv = getRandomPermutation(len(y_train))
                else:
                    minibatch_size = len(y_train)
                    p = y_range
                n_minibatch = math.ceil(float(len(y_train)) / minibatch_size)

                t0_iter = time.time()
                print('-'*50)
                print('Iteration %d' %int(global_step.eval(session=sess)+1))
                print()

                pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                for j in range(0, len(y_train), minibatch_size):
                    fd_minibatch = {}
                    fd_minibatch[X_iv] = X[allsl]
                    fd_minibatch[time_X] = X.time
                    fd_minibatch[y_] = y_train[dv].iloc[p[j:j + minibatch_size]]
                    fd_minibatch[time_y] = y_train.time.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[gf_y_] = gf_y_train.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[first_obs] = y_train.first_obs.iloc[p[j:j + minibatch_size]]
                    fd_minibatch[last_obs] = y_train.last_obs.iloc[p[j:j + minibatch_size]]

                    summary_params_batch, summary_train_losses_batch, _, loss_minibatch = sess.run([summary_params, summary_losses, train_op, loss_func], feed_dict=fd_minibatch)
                    train_writer.add_summary(summary_params_batch, global_batch_step.eval(session=sess))
                    train_writer.add_summary(summary_train_losses_batch, global_batch_step.eval(session=sess))

                    pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

                sess.run(incr_global_step)

                s = parse_summary(summary_params_batch)

                print('Parameter summary: \n')
                print_summary(s)
                print()

                print_summary(s, f=logdir + '/parameter_summary.txt')

                # loss_minibatch, preds_train = sess.run([loss_func, out], feed_dict=fd_train)
                summary_cv_losses_batch, loss_cv, preds_cv = sess.run([summary_losses, loss_func, out], feed_dict=fd_cv)
                mae_cv = mae(y_cv[dv], preds_cv)
                cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
                with open(logdir + '/results.txt', 'w') as f:
                    f.write('='*50 + '\n')
                    # print_tee('Train loss: %.4f' % loss_minibatch, [sys.stdout, f])
                    # print_tee('Train MAE loss: %.4f' % mae_train, [sys.stdout, f])
                    print_tee('CV loss: %.4f'%loss_cv, [sys.stdout, f])
                    print_tee('CV MAE loss: %.4f'%mae_cv, [sys.stdout, f])
                    print()
                    f.write('='*50 + '\n')

                plot_x = support.eval(session=sess)
                plot_y = (beta_global * conv_global(support)).eval(session=sess)

                plot_convolutions(plot_x, plot_y, fixef, dir=logdir, filename='convolution_plot.jpg')
                if attribution_model:
                    plot_y = (beta_global_marginal * conv_global(support)).eval(session=sess)
                    plot_convolutions(plot_x, plot_y, fixef, dir=logdir, filename='marginal_convolution_plot.jpg')



                saver.save(sess, logdir + '/model.ckpt')
                t1_iter = time.time()
                print('Iteration time: %.2fs' %(t1_iter-t0_iter))

            loss_cv, preds_cv = sess.run([loss_func, out], feed_dict=fd_cv)

            print('Final CV loss: %.4f'%loss_cv)

            if baseline_LM:
                print('.' * 50)
                print('Bootstrap significance testing')
                print('Model vs. LM baseline on CV data')
                print()
                print('MSE loss improvement: %.4f' % (lm_mse_cv - mse(y_cv[dv], preds_cv)))
                p_cv = bootstrap(y_cv[dv], lm_preds_cv, preds_cv, err_type=loss)
                print('p = %.4e' % p_cv)
                print('.' * 50)
                print()
            if baseline_LME:
                print('.' * 50)
                print('Bootstrap significance testing')
                print('Model vs. LME baseline on CV data')
                print()
                print('MSE loss improvement: %.4f' % (lme_mse_cv - mse(y_cv[dv], preds_cv)))
                p_cv = bootstrap(y_cv[dv], lme_preds_cv, preds_cv, err_type=loss)
                print('p = %.4e' % p_cv)
                print('.' * 50)
                print()
            if baseline_GAM:
                print('.' * 50)
                print('Bootstrap significance testing')
                print('Model vs. GAM baseline on CV data')
                print()
                print('MSE loss improvement: %.4f' % (gam_mse_cv - mse(y_cv[select_gam_cv][dv], preds_cv[select_gam_cv])))
                p_cv = bootstrap(y_cv[select_gam_cv][dv], gam_preds_cv, preds_cv[select_gam_cv], err_type=loss)
                print('p = %.4e' % p_cv)
                print('.' * 50)
                print()
            print()




