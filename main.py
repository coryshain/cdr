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
import scipy
import pandas as pd
import configparser
import statsmodels.formula.api as smf
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

import tensorflow as tf
from tensorflow.python.platform.test import is_gpu_available

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

def accumulate_row(row, col, fdur):
    global val
    if row['sentpos'] < 2:
        val = row[col]
    else:
        val += row[col]
    out = val
    if row[fdur] > 0:
        val = 0
    return out

def accumulate_column(X, col, fdur):
    X['cum' + col] = X.apply(accumulate_row, axis=1, args=(col,fdur))
    return X

def get_input_range_inner(y, X):
    condition = X.subject == y.subject
    condition &= X.docid == y.docid
    condition &= X.time <= y.time
    matching_indices = np.where(condition)[0]
    lower_bound = matching_indices[0]
    upper_bound = matching_indices[-1]+1
    return lower_bound, upper_bound

def get_history_intervals(y, X):
    pb = tf.contrib.keras.utils.Progbar(len(y))

    lower_bounds = np.zeros(len(y)).astype('int32')
    upper_bounds = np.zeros(len(y)).astype('int32')

    for i in range(len(y)):
        lower_bounds[i], upper_bounds[i] = get_input_range_inner(y.iloc[i], X)
        pb.update(i, force=True)
    return lower_bounds, upper_bounds

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

config = configparser.ConfigParser()
config.read(sys.argv[1])

## Required
data_path = config.get('settings', 'data_path')
features = config.get('settings', 'features')

## Optional
logdir = config.get('settings', 'logdir', fallback='log')
if not os.path.exists(logdir):
    os.makedirs(logdir)
shutil.copy2(sys.argv[1], logdir + '/config.ini')
modality = config.get('settings', 'modality', fallback='ET')
network_type = config.get('settings', 'network_type', fallback='mle')
data_rep = config.get('settings', 'data_rep', fallback='conv')
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
lme_baseline_spillover = config.get('settings', 'lme_baseline_spillover', fallback=features)
lme_baseline_spillover = lme_baseline_spillover.strip().split()
lme_baseline_cum = config.get('settings', 'lme_baseline_cum', fallback=features)
lme_baseline_cum = lme_baseline_cum.strip().split()
features = features.strip().split()
attribution_model = config.getboolean('settings', 'attribution_model', fallback=False)
random_subjects_model = config.getboolean('settings', 'random_subjects_model', fallback=True)
random_subjects_conv_params = config.getboolean('settings', 'random_subjects_conv_params', fallback=False)
log_random = config.getboolean('settings', 'log_random', fallback=False)
log_convolution_plots = config.getboolean('settings', 'log_convolution_plots', fallback=False)
compute_signif = config.getboolean('settings', 'compute_signif', fallback=False)
n_epoch_train = config.getint('settings', 'n_epoch_train', fallback=50)
n_epoch_finetune = config.getint('settings', 'n_epoch_finetune', fallback=250)

other_full = [
        'word',
        'time',
        'fdur',
        'subject',
        'docid',
        'sentid',
        'correct',
        'startoffile',
        'endoffile',
        'startofscreen',
        'endofscreen',
        'startofsentence',
        'endofsentence',
        'startofline',
        'endofline'
    ]

sys.stderr.write('Loading data...\n')
X = pd.read_csv(data_path, sep=' ', skipinitialspace=True)

sys.stderr.write('Pre-processing data...\n')
if modality == 'ET':
    X.rename(columns={'fdurGP': 'fdur'}, inplace=True)
    for col in lme_baseline_cum:
        X = accumulate_column(X, col, 'fdur')

X = X[X.fdur.notnull() & (X.fdur > 0)]
X.subject = X.subject.astype('category')
X.docid = X.docid.astype('category')
X.sentid = X.sentid.astype('category')

X['time'] = X.groupby(['subject', 'docid']).fdur.shift(1).fillna(value=0)
X.time = X.groupby(['subject', 'docid']).time.cumsum() / 1000 # Convert ms to s
other_relevant = set(other_full).intersection(set(X.columns))
other_relevant = list(other_relevant)
X._get_numeric_data().fillna(value=0, inplace=True)
X = X[other_relevant + features]
X[features] = z(X[features])

for col in lme_baseline_spillover:
    s_name = col + 'S1'
    X[s_name] = X.groupby(['subject', 'docid'])[col].shift(1)
    if X[s_name].dtype == object:
        X[s_name].fillna('null', inplace=True)
    else:
        X[s_name].fillna(0, inplace=True)

y = X[other_relevant]

print('Computing history intervals for each regression target sample...')
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

n_subj = len(X.subject.unique())
n_feat = len(features)

select = y.fdur > 100
select &= y.fdur < 3000
if 'correct' in y.columns:
    select &= y.correct > 4

if 'startofsentence' in y.columns:
    select &= y.startofsentence != 1
if 'endofsentence' in y.columns:
    select &= y.endofsentence != 1
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
    y_train.fdur = c(y_train.fdur)
    y_cv = y[select_cv]
    y_cv.fdur = c(y_cv.fdur)

    n_train_sample = len(y_train)
    n_cv_sample = len(y_cv)
else:
    y.fdur = c(y.fdur)
    n_train_sample = len(y)

print()
print('Evaluating on %s partition' %partition)
print('Number of training samples: %d' %n_train_sample)
if partition == 'train':
    #pass
    print('Number of cross-validation samples: %d' %n_cv_sample)
print()

print('Correlation matrix (training data only):')
rho = X[features].corr()
print(rho)
print()

if partition == 'train':
    if baseline_LM or baseline_LME:
        X_train = X[select_train]
        X_train.fdur = y_train.fdur
        X_cv = X[select_cv]
        X_cv.fdur = y_cv.fdur
    if baseline_LM:
        if not os.path.exists('baselines'):
            os.makedirs('baselines')
        if not os.path.exists('baselines/LM'):
            os.makedirs('baselines/LM')

        print('Getting linear regression baseline...')

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
            lm_mse_train = mse(y_train.fdur, lm_preds_train)
            lm_mae_train = mae(y_train.fdur, lm_preds_train)
            print_tee('Training set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lm_mse_train, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lm_mae_train, [sys.stdout, f])
            print_tee('', [sys.stdout, f])

            lm_preds_cv = lm.predict(lm_results.params, X_LM_cv)
            lm_mse_cv = mse(y_cv.fdur, lm_preds_cv)
            lm_mae_cv = mae(y_cv.fdur, lm_preds_cv)
            print_tee('Cross-validation set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lm_mse_cv, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lm_mae_cv, [sys.stdout, f])
            print_tee('='*50, [sys.stdout, f])

            print()


    if baseline_LME:
        if not os.path.exists('baselines'):
            os.makedirs('baselines')
        if not os.path.exists('baselines/LME'):
            os.makedirs('baselines/LME')

        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        import rpy2
        pandas2ri.activate()

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

            if os.path.exists('baselines/LME/lme_fit.obj'):
                with open('baselines/LME/lme_fit.obj', 'rb') as f2:
                    lme = pickle.load(f2)
            else:
                lme = regress_lme(bform_LME, X_train)
                with open('baselines/LME/lme_fit.obj', 'wb') as f2:
                    pickle.dump(lme, f2)

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

            print_tee('='*50, [sys.stdout, f])
            print_tee('Linear mixed-effects regression baseline results summary', [sys.stdout, f])
            print_tee('', [sys.stdout, f])

            s = get_model_summary(lme)

            print_tee(s, [sys.stdout, f])
            print_tee('Convergence warnings:', [sys.stdout, f])
            print_tee(s.rx2('convWarn'), [sys.stdout, f])
            print_tee('', [sys.stdout, f])

            lme_preds_train = predict(lme, X_train)
            lme_mse_train = mse(y_train.fdur, lme_preds_train)
            lme_mae_train = mae(y_train.fdur, lme_preds_train)
            print_tee('Training set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lme_mse_train, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lme_mae_train, [sys.stdout, f])

            lme_preds_cv = predict(lme, X_cv)
            lme_mse_cv = mse(y_cv.fdur, lme_preds_cv)
            lme_mae_cv = mae(y_cv.fdur, lme_preds_cv)
            print_tee('Cross-validation set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lme_mse_cv, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lme_mae_cv, [sys.stdout, f])
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
      
        X_conv = conv(X[0].time - X[0].time, K, rate) * X[0][features]
        for i in range(1, len(X)):
            t_delta = X[0].time-X[i].time
            X_conv += conv(t_delta, K, rate) * X[i][features]
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
            feature_powset = powerset(n_feat)

        X_raw = tf.placeholder(shape=[None, n_feat], dtype=tf.float32, name='X_raw')
        subject_id_X = tf.placeholder(shape=[None], dtype=tf.int32, name='subject_id_X')
        doc_id_X = tf.placeholder(shape=[None], dtype=tf.float32, name='doc_id_X')
        time_X = tf.placeholder(shape=[None], dtype=tf.float32, name='time_X')

        y_ = tf.placeholder(shape=[None], dtype=tf.float32, name='y_')
        subject_id_y = tf.placeholder(shape=[None], dtype=tf.int32, name='subject_id_y')
        doc_id_y = tf.placeholder(shape=[None], dtype=tf.float32, name='doc_id_y')
        time_y = tf.placeholder(shape=[None], dtype=tf.float32, name='time_y')
        first_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='first_obs')
        last_obs = tf.placeholder(shape=[None], dtype=tf.int32, name='last_obs')

        if data_rep == 'conv':
            if conv_func == 'exp':
                L_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='lambda')
            elif conv_func == 'gamma':
                K_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='K')
                theta_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='theta')
                delta_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='loc')

        intercept_global = tf.Variable(tf.constant(float(y_train.fdur.mean()), shape=[1]), name='intercept')

        if random_subjects_model:
            subject_y_ix = subject_id_y
            intercept_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj], mean=1., stddev=.1, dtype=tf.float32), name='subject_intercept_coef')
            intercept_correction_by_subject -= tf.reduce_mean(intercept_correction_by_subject, axis=0)
            intercept = intercept_global + intercept_correction_by_subject
        else:
            subject_y_ix = tf.constant(0, shape=[1])
            intercept = intercept_global

        if data_rep == 'conv':
            ## Exponential distribution convolution function
            if conv_func == 'exp':
                conv_global = lambda x: tf.exp(L_global) * tf.exp(-tf.exp(L_global) * x)
                if random_subjects_model and random_subjects_conv_params:
                    ## By-subject correction terms to rate parameter lambda
                    L_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='L_correction_by_subject')
                    L_correction_by_subject -= tf.reduce_mean(L_correction_by_subject, axis=0)
                    L = L_global + L_correction_by_subject
                    L = tf.gather(L, subject_y_ix)
                    conv = lambda x: tf.exp(L) * tf.exp(-tf.exp(L) * x)
                else:
                    conv = conv_global
            ## Gamma distribution convolution function
            elif conv_func == 'gamma':
                conv_global_delta_zero = tf.contrib.distributions.Gamma(concentration=tf.exp(K_global[0]), rate=tf.exp(theta_global[0]), validate_args=True).prob
                conv_global = lambda x: conv_global_delta_zero(x + tf.exp(delta_global)[0])
                if random_subjects_model and random_subjects_conv_params:
                    ## By-subject correction terms to shape parameter K
                    K_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='K_correction_by_subject')
                    K_correction_by_subject -= tf.reduce_mean(K_correction_by_subject, axis=0)
                    K = K_global + K_correction_by_subject

                    ## By-subject correction terms to scale parameter theta
                    theta_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='theta_correction_by_subject')
                    theta_correction_by_subject -= tf.reduce_mean(theta_correction_by_subject, axis=0)
                    theta = theta_global + theta_correction_by_subject

                    ## By-subject correction terms to location parameter delta
                    delta_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='loc_correction_by_subject')
                    delta_correction_by_subject -= tf.reduce_mean(delta_correction_by_subject, axis=0)
                    delta = delta_global + delta_correction_by_subject

                    conv_delta_zero = tf.contrib.distributions.Gamma(concentration=tf.gather(tf.exp(K), subject_y_ix), rate=tf.gather(tf.exp(theta), subject_y_ix), validate_args=True).prob
                    conv = lambda x: conv_delta_zero(x + tf.gather(tf.exp(delta), subject_y_ix))
                else:
                    conv_delta_zero = conv_global_delta_zero
                    conv = conv_global

            def convolve_events(time_target, first_obs, last_obs):
                input_rows = X_raw[first_obs:last_obs]
                input_times = time_X[first_obs:last_obs]
                t_delta = time_target - input_times
                conv_coef = conv(tf.expand_dims(t_delta, -1))

                return tf.reduce_sum(conv_coef * input_rows, 0)

            X_conv = tf.map_fn(lambda x: convolve_events(*x), [time_y, first_obs, last_obs], dtype=tf.float32, parallel_iterations=1)


        if attribution_model:
            phi_global = tf.Variable(tf.truncated_normal(shape=[1, len(feature_powset)], stddev=0.1, dtype=tf.float32), name='phi')
            if random_subjects_model:
                phi_correction_by_subject = tf.Variable(tf.truncated_normal(shape=[n_subj, len(feature_powset)], mean=0., stddev=.1, dtype=tf.float32), name='subject_phi_coef')
                phi_correction_by_subject -= tf.reduce_mean(phi_correction_by_subject, axis=0)
                phi = phi_global + phi_correction_by_subject
            else:
                phi = phi_global
            out = tf.constant(0.)
            beta_global = tf.constant(0., shape=[1, n_feat])
            beta_global_marginal = tf.constant(0., shape=[1, n_feat])

            sess.run(tf.global_variables_initializer())

            for i in range(len(feature_powset)):
                indices = np.where(feature_powset[i])[0].astype('int32')
                vals = tf.gather(X_conv, indices, axis=1)
                vals *= tf.expand_dims(tf.gather(phi, subject_y_ix)[:, i], -1) / feature_powset[i].sum()
                vals = tf.reduce_sum(vals, axis=1)
                out += vals

                ## Record marginal betas
                mask = tf.constant(feature_powset[i].astype('float32'))
                beta_global += phi_global[0, i] * mask / feature_powset[i].sum()
                beta_global_marginal += phi_global[0, i] * mask
        else:
            beta_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=0., stddev=0.1, dtype=tf.float32), name='beta')
            if random_subjects_model:
                subject_beta = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=0., stddev=.1, dtype=tf.float32), name='subject_beta_coef')
                subject_beta -= tf.reduce_mean(subject_beta, axis=0)
                beta = beta_global + subject_beta
                out = X_conv * tf.gather(beta, subject_y_ix)
                out = tf.reduce_sum(out, 1)
            else:
                out = tf.squeeze(tf.matmul(X_conv, tf.transpose(tf.gather(beta_global, subject_y_ix))), -1)

        out += tf.gather(intercept, subject_y_ix)

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
        if random_subjects_model and log_random:
            subject_writers = [tf.summary.FileWriter(logdir + '/subjects/%d'%i) for i in range(n_subj)]
            subject_indexer = tf.placeholder(dtype=tf.int32)
            tf.summary.scalar('by_subject/Intercept', intercept_correction_by_subject[subject_indexer], collections=['by_subject'])
        tf.summary.scalar('Intercept', intercept_global[0], collections=['params'])
        support = tf.expand_dims(tf.lin_space(0., 2.5, 1000), -1)
        for i in range(len(features)):
            tf.summary.scalar('beta/%s'%features[i], beta_global[0,i], collections=['params'])
            if attribution_model:
                tf.summary.scalar('beta_marginal/%s' % features[i], beta_global_marginal[0, i], collections=['params'])
            if random_subjects_model and log_random:
                if not attribution_model:
                    tf.summary.scalar('by_subject/beta/%s'%features[i], subject_beta[subject_indexer,i], collections=['by_subject'])
            if data_rep == 'conv':
                if conv_func == 'exp':
                    tf.summary.scalar(logdir + '_lambda/%s' % features[i], L_global[0, i], collections=['params'])
                    if random_subjects_model and random_subjects_conv_params and log_random:
                        tf.summary.scalar('by_subject/log_lambda/%s' % features[i], L_correction_by_subject[subject_indexer, i], collections=['by_subject'])
                elif conv_func == 'gamma':
                    tf.summary.scalar(logdir + '_k/%s' % features[i], K_global[0, i], collections=['params'])
                    tf.summary.scalar(logdir + '_theta/%s' % features[i], theta_global[0, i], collections=['params'])
                    tf.summary.scalar(logdir + '_neg_delta/%s' % features[i], delta_global[0, i], collections=['params'])
                    if random_subjects_model and random_subjects_conv_params and log_random:
                        tf.summary.scalar('by_subject/log_k/%s' % features[i], K_correction_by_subject[subject_indexer, i], collections=['by_subject'])
                        tf.summary.scalar('by_subject/log_theta/%s' % features[i], theta_correction_by_subject[subject_indexer, i], collections=['by_subject'])
                        tf.summary.scalar('by_subject/log_delta/%s' % features[i], delta_correction_by_subject[subject_indexer, i], collections=['by_subject'])
                if log_convolution_plots:
                    tf.summary.histogram('conv/%s' % features[i], conv_global(support)[:,i], collections=['params'])
        if attribution_model:
            for i in range(len(feature_powset)):
                name = 'phi_' + '_'.join([features[j] for j in range(len(features)) if feature_powset[i,j] == 1])
                tf.summary.scalar('phi/' + name, phi_global[0,i], collections=['params'])
                if random_subjects_model and random_subjects_conv_params and log_random:
                    tf.summary.scalar('by_subject/phi/' + name, phi_correction_by_subject[subject_indexer, i], collections=['by_subject'])


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
                    if parent.startswith(logdir + '_neg_'):
                        unlog_name = parent[8:]
                        if unlog_name not in s_dict_grouped:
                            s_dict_grouped[unlog_name] = {}
                        unlog_val = -math.exp(val)
                        s_dict_grouped[unlog_name][key] = unlog_val
                    elif parent.startswith(logdir + '_'):
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
        fd_train[subject_id_X] = X.subject.cat.codes
        fd_train[subject_id_y] = y_train.subject.cat.codes
        fd_train[doc_id_X] = X.docid.cat.codes
        fd_train[doc_id_y] = y_train.docid.cat.codes
        fd_train[time_X] = X.time
        fd_train[time_y] = y_train.time
        fd_train[first_obs] = y_train.first_obs
        fd_train[last_obs] = y_train.last_obs
        fd_train[X_raw] = X[features]
        fd_train[y_] = y_train.fdur

        fd_cv = {}
        fd_cv[subject_id_X] = X.subject.cat.codes
        fd_cv[subject_id_y] = y_cv.subject.cat.codes
        fd_cv[doc_id_X] = X.docid.cat.codes
        fd_cv[doc_id_y] = y_cv.docid.cat.codes
        fd_cv[time_X] = X.time
        fd_cv[time_y] = y_cv.time
        fd_cv[first_obs] = y_cv.first_obs
        fd_cv[last_obs] = y_cv.last_obs
        fd_cv[X_raw] = X[features]
        fd_cv[y_] = y_cv.fdur

            
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
                fd_minibatch[subject_id_X] = X.subject.cat.codes
                fd_minibatch[subject_id_y] = y_train.subject.cat.codes.iloc[p[j:j + minibatch_size]]
                fd_minibatch[doc_id_X] = X.docid.cat.codes
                fd_minibatch[doc_id_y] = y_train.docid.cat.codes.iloc[p[j:j + minibatch_size]]
                fd_minibatch[time_X] = X.time
                fd_minibatch[time_y] = y_train.time.iloc[p[j:j + minibatch_size]]
                fd_minibatch[first_obs] = y_train.first_obs.iloc[p[j:j + minibatch_size]]
                fd_minibatch[last_obs] = y_train.last_obs.iloc[p[j:j + minibatch_size]]
                fd_minibatch[X_raw] = X[features]
                fd_minibatch[y_] = y_train.fdur.iloc[p[j:j + minibatch_size]]

                summary_params_batch, summary_train_losses_batch, _, loss_minibatch = sess.run([summary_params, summary_losses, train_op, loss_func], feed_dict=fd_minibatch)
                train_writer.add_summary(summary_params_batch, global_batch_step.eval(session=sess))
                train_writer.add_summary(summary_train_losses_batch, global_batch_step.eval(session=sess))
                if random_subjects_model and log_random:
                    for i in range(n_subj):
                        summary_by_subject_batch = sess.run(summary_by_subject, feed_dict={subject_indexer: i})
                        subject_writers[i].add_summary(summary_by_subject_batch, global_batch_step.eval(session=sess))

                pb.update((j / minibatch_size) + 1, values=[('loss', loss_minibatch)], force=True)

            sess.run(incr_global_step)

            s = parse_summary(summary_params_batch)

            print('Parameter summary: \n')
            print_summary(s)
            print()

            print_summary(s, f=logdir + '/parameter_summary.txt')

            # loss_minibatch, preds_train = sess.run([loss_func, out], feed_dict=fd_train)
            summary_cv_losses_batch, loss_cv, preds_cv = sess.run([summary_losses, loss_func, out], feed_dict=fd_cv)
            # mae_train = mae(y_train.fdur, preds_train)
            mae_cv = mae(y_cv.fdur, preds_cv)
            cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
            with open(logdir + '/results.txt', 'w') as f:
                f.write('='*50 + '\n')
                # print_tee('Train loss: %.4f' % loss_minibatch, [sys.stdout, f])
                # print_tee('Train MAE loss: %.4f' % mae_train, [sys.stdout, f])
                print_tee('CV loss: %.4f'%loss_cv, [sys.stdout, f])
                print_tee('CV MAE loss: %.4f'%mae_cv, [sys.stdout, f])
                print()
                f.write('='*50 + '\n')

            if compute_signif:
                if baseline_LM:
                    print('.'*50)
                    print('Bootstrap significance testing')
                    print('Model vs. LM baseline on CV data')
                    print()
                    print('MSE loss improvement: %.4f' %(lm_mse_cv-mse(y_cv.fdur, preds_cv)))
                    p_cv = bootstrap(y_cv.fdur, lm_preds_cv, preds_cv, err_type=loss)
                    print('p = %.4e' %p_cv)
                    print('.' * 50)
                    print()
                if baseline_LME:
                    print('.' * 50)
                    print('Bootstrap significance testing')
                    print('Model vs. LME baseline on CV data')
                    print()
                    print('MSE loss improvement: %.4f' %(lme_mse_cv-mse(y_cv.fdur, preds_cv)))
                    p_cv = bootstrap(y_cv.fdur, lme_preds_cv, preds_cv, err_type=loss)
                    print('p = %.4e' % p_cv)
                    print('.' * 50)
                    print()
            print()

            if data_rep == 'conv':
                plot_x = support.eval(session=sess)
                plot_y = (beta_global * conv_global(support)).eval(session=sess)

                plot_convolutions(plot_x, plot_y, features, dir=logdir, filename='convolution_plot.jpg')
                if attribution_model:
                    plot_y = (beta_global_marginal * conv_global(support)).eval(session=sess)
                    plot_convolutions(plot_x, plot_y, features, dir=logdir, filename='marginal_convolution_plot.jpg')



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
            print('MSE loss improvement: %.4f' % (lm_mse_cv - mse(y_cv.fdur, preds_cv)))
            p_cv = bootstrap(y_cv.fdur, lm_preds_cv, preds_cv, err_type=loss)
            print('p = %.4e' % p_cv)
            print('.' * 50)
            print()
        if baseline_LME:
            print('.' * 50)
            print('Bootstrap significance testing')
            print('Model vs. LME baseline on CV data')
            print()
            print('MSE loss improvement: %.4f' % (lme_mse_cv - mse(y_cv.fdur, preds_cv)))
            p_cv = bootstrap(y_cv.fdur, lme_preds_cv, preds_cv, err_type=loss)
            print('p = %.4e' % p_cv)
            print('.' * 50)
            print()
        print()




