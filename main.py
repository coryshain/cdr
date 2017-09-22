import sys
import os
import yaml
import math
import pickle
import numpy as np
from numpy import inf, nan
import scipy
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def bootstrap(true, preds_1, preds_2, err_type='mse', n_iter=10000):
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
        if base_diff < 0 and cur_diff <= base_diff:
            hits += 1
        elif base_diff > 0 and cur_diff >= base_diff:
            hits += 1
        elif base_diff == 0:
            hits += 1
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


def plot_convolutions(shape, scale, beta, features):
    support = np.expand_dims(np.linspace(1e-5, 2.5, num=250), -1)
    convolutions = beta*scipy.stats.gamma.pdf(support, shape, scale=scale)
    for i in range(len(shape)):
        plt.plot(support, convolutions[:,i], label=features[i])
        #plt.fill_between(np.squeeze(support), convolutions[:,i], 0, alpha=0.5)
    plt.legend()
    plt.savefig('log/convolution_plot.jpg')
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

features   = [
               'sentpos',
               'wlen',
               'fwprob5surp',
               'totsurp'
             ]
other_full = [
               'word',
               'onset_time',
               'offset_time',
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

network_type = 'mle'
data_rep = 'conv'
window = 50
partition = 'train'
loss = 'MSE'
modulus = 3
cv_modulus = 5
offset = 1e-5
baseline_LM = True
baseline_LME = True
attribution_model = True
random_subjects_model = True
compute_signif = False

sys.stderr.write('Loading data...\n')
X = pd.read_csv(sys.argv[1], sep=' ', skipinitialspace=True)

sys.stderr.write('Pre-processing data...\n')
#X = X.head(10000)
X = X.loc[X['fdur'].notnull()]
X.subject = X.subject.astype('category')
X.docid = X.docid.astype('category')
X.sentid = X.sentid.astype('category')
X['onset_time'] = X.groupby(['subject', 'docid']).fdur.shift(1).fillna(value=0)
X['onset_time'] = X.groupby(['subject', 'docid']).onset_time.cumsum() / 1000 # Convert ms to s
X['offset_time'] = X.groupby(['subject', 'docid']).fdur.cumsum() / 1000 # Convert ms to s
other_relevant = list(set(other_full).intersection(set(X.columns)))
X._get_numeric_data().fillna(value=0, inplace=True)
X = X[other_relevant + features]
X[features] = z(X[features])
X = [X]
for i in range(1,window+1):
    X.append(X[0].groupby(['subject', 'docid'], as_index=False).shift(i)[['onset_time', 'offset_time']+features])
    X[i].fillna(value=0, inplace=True)

n_subj = len(X[0]['subject'].unique())
n_feat = len(features)

select = X[0].fdur > 100
select &= X[0].fdur < 3000
select &= X[0].correct > 4

if 'startofsentence' in X[0].columns:
    select &= X[0].startofsentence != 1
if 'endofsentence' in X[0].columns:
    select &= X[0].endofsentence != 1
if partition == 'test':
    select &= ((X[0].subject.cat.codes+1)+X[0].sentid.cat.codes) % modulus == modulus-1
elif partition == 'dev':
    select &= ((X[0].subject.cat.codes+1)+X[0].sentid.cat.codes) % modulus == modulus-2
else:
    select &= ((X[0].subject.cat.codes+1)+X[0].sentid.cat.codes) % modulus < modulus-2
    select_cv = select & (((X[0].subject.cat.codes+1)+X[0].sentid.cat.codes) % cv_modulus == cv_modulus-1)
    select_train = select & (((X[0].subject.cat.codes+1)+X[0].sentid.cat.codes) % cv_modulus < cv_modulus-1)
#    mean_train = X[0][features][select_train].mean(axis=0)
#    std_train = X[0][features][select_train].mean(axis=0)
#    mean_cv = X[0][features][select_cv].mean(axis=0)
#    std_cv = X[0][features][select_cv].mean(axis=0)

for i in range(len(X)):
    if partition == 'train':
        X[i] = [X[i][select_train], X[i][select_cv]]
        #X[i] = [X[i][select]]
    else:
        X[i] = X[i][select]
if partition == 'train':
    y = [c(X[0][0].fdur), c(X[0][1].fdur)]
    #y = [X[0][0].fdur]
    n_train_sample = len(X[0][0])
    n_cv_sample = len(X[0][1])
else:
    y = c(X[0].fdur)
    n_train_sample = len(X[0])

print()
print('Evaluating on %s partition' %partition)
print('Number of training samples: %d' %n_train_sample)
if partition == 'train':
    #pass
    print('Number of cross-validation samples: %d' %n_cv_sample)
print()

if partition == 'train':
    if baseline_LM:
        if not os.path.exists('baselines'):
            os.makedirs('baselines')
        if not os.path.exists('baselines/LM'):
            os.makedirs('baselines/LM')

        print('Getting linear regression baseline...')

        with open('baselines/LM/summary.txt', 'w') as f:
            print()
            feats_train = sm.add_constant(X[0][0][features])
            feats_cv = sm.add_constant(X[0][1][features])
            if os.path.exists('baselines/LM/lm_fit.obj'):
                with open('baselines/LM/lm_fit.obj', 'rb') as f2:
                    lm = pickle.load(f2)
            else:
                lm = sm.OLS(y[0], feats_train)
                with open('baselines/LM/lm_fit.obj', 'wb') as f2:
                    pickle.dump(lm, f2)
            lm_results = lm.fit()
            print_tee('='*50, [sys.stdout, f])
            print_tee('Linear regression baseline results summary', [sys.stdout, f])
            print_tee('Results summary:', [sys.stdout, f])
            print_tee('', [sys.stdout, f])
            print_tee('Betas:', [sys.stdout, f])
            print_tee(lm_results.params, [sys.stdout, f])
            print_tee('', [sys.stdout, f])

            lm_preds_train = lm.predict(lm_results.params, feats_train)
            lm_mse_train = mse(y[0], lm_preds_train)
            lm_mae_train = mae(y[0], lm_preds_train)
            print_tee('Training set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lm_mse_train, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lm_mae_train, [sys.stdout, f])
            print_tee('', [sys.stdout, f])

            lm_preds_cv = lm.predict(lm_results.params, feats_cv)
            lm_mse_cv = mse(y[1], lm_preds_cv)
            lm_mae_cv = mae(y[1], lm_preds_cv)
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

            X_lme = X[0][0]
            bform = 'scale(fdur, center=TRUE, scale=FALSE) ~ sentpos + wlen + fwprob5surp + totsurp + (0 + sentpos + wlen + fwprob5surp + totsurp | subject)' # + (1 | word) + (1 | subject:sentid)'
            if os.path.exists('baselines/LME/lme_fit.obj'):
                with open('baselines/LME/lme_fit.obj', 'rb') as f2:
                    lme = pickle.load(f2)
            else:
                lme = regress_lme(bform, X_lme)
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

            lme_preds_train = predict(lme, X[0][0])
            lme_mse_train = mse(y[0], lme_preds_train)
            lme_mae_train = mae(y[0], lme_preds_train)
            print_tee('Training set loss:', [sys.stdout, f])
            print_tee('  MSE: %.4f' % lme_mse_train, [sys.stdout, f])
            print_tee('  MAE: %.4f' % lme_mae_train, [sys.stdout, f])

            lme_preds_cv = predict(lme, X[0][1])
            lme_mse_cv = mse(y[1], lme_preds_cv)
            lme_mae_cv = mae(y[1], lme_preds_cv)
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
        shape = pm.Gamma('shape', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('shape', mu=mu, cov=sigma, shape=n_feat)
        rate = pm.Gamma('rate', alpha=2, beta=2, shape=n_feat) #pm.MvNormal('rate', mu=mu, cov=sigma, shape=n_feat)
        beta = pm.Normal('beta', mu=0, sd=0.24, shape=n_feat)#pm.MvNormal('beta', mu=mu, cov=sigma, shape=n_feat)
        sigma = pm.HalfNormal('sigma', sd=1)
        intercept = pm.Normal('Intercept', y.mean(), sd=y.std())
      
        X_conv = gamma(X[0].offset_time-X[0].onset_time, shape, rate) * X[0][features] 
        for i in range(1, len(X)):
            t_delta = X[0].offset_time-X[i].onset_time
            X_conv += gamma(t_delta, shape, rate) * X[i][features]
        E_fdur = intercept + pm.math.dot(X_conv, beta)
     
        fdur = pm.Normal('fdur', mu = E_fdur, sd = sigma, observed=y)

        trace = pm.sample(1000, tune=500)
        
        fig, ax = plt.subplots(5, 2)
        pm.traceplot(trace, ax=ax)
        plt.savefig('trace.jpg')

elif network_type == 'mle':
    print()
    print('Fitting parameters using maximum likelihood...')
    import tensorflow as tf
    from tensorflow.python.platform.test import is_gpu_available
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    n_epoch = 5000
    batch_size = len(X[0][0])
    n_batch = math.ceil(float(len(X[0][0]))/batch_size)

    sess = tf.Session()

    usingGPU = is_gpu_available()
    print('Using GPU: %s' %usingGPU)
    with sess.graph.as_default():
        if attribution_model:
            feature_powset = powerset(n_feat)


        X_raw = [tf.placeholder(shape=[None, 2+n_feat], dtype=tf.float32, name='input_%d'%i) for i in range(len(X))]
        subjects_raw = tf.placeholder(shape=[None], dtype=tf.int32, name='subject_ids')
        if not data_rep == 'linear':
            shape_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=1., stddev=.1, dtype=tf.float32), name='shape')
            shape_global = tf.clip_by_value(shape_global, 1e-10, inf)
            scale_global = tf.Variable(tf.truncated_normal(shape=[1, n_feat], mean=1., stddev=.1, dtype=tf.float32), name='scale')
            scale_global = tf.clip_by_value(scale_global, 1e-10, inf)

        intercept_global = tf.Variable(tf.constant(float(y[0].mean()), shape=[1]), name='intercept')

        if random_subjects_model:
            subject_ids = subjects_raw
            if not data_rep == 'linear':
                subject_shape = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=1., stddev=.1, dtype=tf.float32), name='subject_shape_coef')
                subject_shape = tf.clip_by_value(subject_shape, 1e-10, inf)
                subject_shape -= tf.reduce_mean(subject_shape, axis=0)
                shape = shape_global + subject_shape

                subject_scale = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=1., stddev=.1, dtype=tf.float32), name='subject_scalee_coef')
                subject_scale = tf.clip_by_value(subject_scale, 1e-10, inf)
                subject_scale -= tf.reduce_mean(subject_scale, axis=0)
                scale = scale_global + subject_scale

            subject_intercept = tf.Variable(tf.truncated_normal(shape=[n_subj], mean=1., stddev=.1, dtype=tf.float32), name='subject_intercept_coef')
            subject_intercept -= tf.reduce_mean(subject_intercept, axis=0)
            intercept = intercept_global + subject_intercept
        else:
            subject_ids = tf.constant(0, shape=[1])
            intercept = intercept_global


        
        if data_rep == 'linear':
            X_conv = X_raw[0][:,2:]
        else:
            gamma = tf.contrib.distributions.Gamma(concentration=tf.gather(shape, subject_ids), rate=tf.gather(scale, subject_ids), validate_args=True)
            
            t_delta = offset
            conv_coef = gamma.prob(tf.expand_dims(t_delta, -1))
            X_conv = conv_coef*X_raw[0][:,2:] 

            for i in range(1, window+1):
                t_delta = X_raw[0][:,0] - X_raw[i][:,0] + offset
                conv_coef = gamma.prob(tf.expand_dims(t_delta, -1))
                weighted = conv_coef*X_raw[i][:,2:]
                X_conv += weighted


        if attribution_model:
            phi_global = tf.Variable(tf.truncated_normal(shape=[1, len(feature_powset)], stddev=0.1, dtype=tf.float32), name='phi')
            if random_subjects_model:
                subject_phi = tf.Variable(tf.truncated_normal(shape=[n_subj, len(feature_powset)], mean=1., stddev=.1, dtype=tf.float32), name='subject_phi_coef')
                subject_phi -= tf.reduce_mean(subject_phi, axis=0)
                phi = phi_global + subject_phi
            else:
                phi = phi_global
            out = tf.constant(0.)
            beta_global = tf.constant(0., shape=[1, n_feat])
            beta_global_attribution = tf.constant(0., shape=[1, n_feat])

            sess.run(tf.global_variables_initializer())

            for i in range(len(feature_powset)):
                indices = np.where(feature_powset[i])[0].astype('int32')
                vals = tf.gather(X_conv, indices, axis=1)
                vals *= tf.expand_dims(tf.gather(phi, subject_ids)[:,i], -1) / feature_powset[i].sum()
                vals = tf.reduce_sum(vals, axis=1)
                out += vals

                ## Record marginal betas
                for j in indices:
                    mask = tf.constant(feature_powset[i].astype('float32'))
                    beta_global += phi_global[0, i] * mask / feature_powset[i].sum()
                    beta_global_attribution += phi_global[0, i] * mask


        else:
            beta_global = tf.Variable(tf.truncated_normal(shape=[1,n_feat], stddev=0.1, dtype=tf.float32), name='beta')
            if random_subjects_model:
                subject_beta = tf.Variable(tf.truncated_normal(shape=[n_subj, n_feat], mean=1., stddev=.1, dtype=tf.float32), name='subject_beta_coef')
                subject_beta -= tf.reduce_mean(subject_beta, axis=0)
                beta = beta_global + subject_beta
                out = X_conv * tf.gather(beta, subject_ids)
                out = tf.reduce_sum(out, 1)
            else:
                out = tf.squeeze(tf.matmul(X_conv, tf.expand_dims(tf.gather(beta_global, subject_ids), -1)), -1)

        out += tf.gather(intercept, subject_ids)

        y_ = tf.placeholder(tf.float32, shape=[None])

        mae_loss = tf.losses.absolute_difference(y_, out)
        mse_loss = tf.losses.mean_squared_error(y_, out)
        if loss.lower() == 'mae':
            loss_func = mae_loss
        else:
            loss_func = mse_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step+1)
        global_batch_step = tf.Variable(0, name='global_batch_step', trainable=False)
        #optim = tf.train.FtrlOptimizer(0.01)
        #optim = tf.train.AdamOptimizer()
        #optim = tf.contrib.keras.optimizers.Nadam()
        optim = tf.train.RMSPropOptimizer(0.1)
        #optim = tf.train.AdagradOptimizer(0.01)
        #optim = tf.train.AdadeltaOptimizer(0.01)
        train_op = optim.minimize(loss_func, global_step=global_batch_step, name='optim')
    
        train_writer = tf.summary.FileWriter('log/train')
        cv_writer = tf.summary.FileWriter('log/cv')
        if random_subjects_model:
            subject_writers = [tf.summary.FileWriter('log/subjects/%d'%i) for i in range(n_subj)]
            subject_indexer = tf.placeholder(dtype=tf.int32)
        tf.summary.scalar('Intercept', intercept_global[0], collections=['params'])
        for i in range(len(features)):
            tf.summary.scalar('beta/%s'%features[i], beta_global[0,i], collections=['params'])
            if not data_rep == 'linear':
                tf.summary.scalar('shape/%s'%features[i], shape_global[0,i], collections=['params'])
                tf.summary.scalar('scale/%s'%features[i], scale_global[0,i], collections=['params'])
            if random_subjects_model:
                if not attribution_model:
                    tf.summary.scalar('by_subject/beta/%s'%features[i], subject_beta[subject_indexer,i], collections=['by_subject'])
                if not data_rep == 'linear':
                    tf.summary.scalar('by_subject/shape/%s'%features[i], subject_shape[subject_indexer,i], collections=['by_subject'])
                    tf.summary.scalar('by_subject/scale/%s'%features[i], subject_scale[subject_indexer,i], collections=['by_subject'])
            if attribution_model:
                tf.summary.scalar('beta_global_attribution/%s'%features[i], beta_global_attribution[0,i], collections=['params'])
        tf.summary.scalar('loss/%s'%loss, loss_func, collections=['loss'])
        summary_params = tf.summary.merge_all(key='params')
        summary_losses = tf.summary.merge_all(key='loss')
        summary_by_subject = tf.summary.merge_all(key='by_subject')
    
        def parse_summary(s):
            s_str = tf.Summary()
            s_str.ParseFromString(s)
            s_dict = {}
            for val in s_str.value:
                s_dict[val.tag] = val.simple_value
            s_dict['beta'] = {}
            if attribution_model:
                s_dict['beta_attribution'] = {}
            if not data_rep == 'linear':
                s_dict['shape'] = {}
                s_dict['scale'] = {}
            key_list = list(s_dict.keys())
            for k in key_list:
                if '/' in k:
                   parent, key = k.strip().split('/')
                   val = s_dict[k]
                   s_dict[parent][key] = val
                   del s_dict[k]
            return s_dict

        def print_summary(s_dict, f=None):
            if f is None:
                yaml.dump(s_dict, default_flow_style=False, stream=sys.stdout)
            else:
                with sys.stdout if f is None else open(f, 'w') as F:
                    yaml.dump(s_dict, default_flow_style=False, stream=F)

        saver = tf.train.Saver()
        if os.path.exists('log/checkpoint'):
            saver.restore(sess, 'log/model.ckpt')
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

        fd_cv = {}
        fd_cv[subjects_raw] = X[0][1].subject.cat.codes
        for k in range(len(X)):
            fd_cv[X_raw[k]] = X[k][1][['onset_time', 'offset_time']+features]
        fd_cv[y_] = y[1]
            
        summary_cv_losses_batch, loss_cv = sess.run([summary_losses, loss_func], feed_dict=fd_cv)
        cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
        print('CV loss: %.4f'%loss_cv)
        print()

        while global_step.eval(session=sess) < n_epoch:
            print('-'*50)
            print('Iteration %d' %int(global_step.eval(session=sess)+1))
            print()

            p, p_inv = getRandomPermutation(len(X[0][0]))
            pb = tf.contrib.keras.utils.Progbar(n_batch)

            for j in range(0, len(X[0][0]), batch_size):
                fd_train = {}
                fd_train[subjects_raw] = X[0][0].subject.cat.codes[j:j+batch_size]
                for k in range(len(X)):
                    fd_train[X_raw[k]] = X[k][0][['onset_time', 'offset_time']+features].iloc[p[j:j+batch_size]]
                fd_train[y_] = y[0].iloc[p[j:j+batch_size]]

                summary_params_batch, summary_train_losses_batch, _, loss_train = sess.run([summary_params, summary_losses, train_op, loss_func], feed_dict=fd_train)
                train_writer.add_summary(summary_params_batch, global_batch_step.eval(session=sess))
                train_writer.add_summary(summary_train_losses_batch, global_batch_step.eval(session=sess))
                if random_subjects_model:
                    for i in range(n_subj):
                        summary_by_subject_batch = sess.run(summary_by_subject, feed_dict={subject_indexer: i})
                        subject_writers[i].add_summary(summary_by_subject_batch, global_batch_step.eval(session=sess))
                
                if False:
                    summary_cv_losses_batch, loss_cv = sess.run([summary_losses, loss_func], feed_dict=fd_cv)
                    cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))

                #beta_batch_list = [('beta_%s'%features[i], beta_batch[i]) for i in range(len(beta_batch))]
                pb.update((j/batch_size)+1, values=[('loss',loss_train)], force=True)

#                ## Sanity check
#                summary_cv_losses_batch, loss_cv = sess.run([summary_losses, loss_func], feed_dict=fd_cv)
#                cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
#                
#                print()
#                print('Sanity check')
#                print('CV loss (tf): %.4f'%loss_cv)
#                
#                ref = X[0][1].onset_time
#                X_sc = np.zeros((len(X[0][1]),len(features)))
#                for i in range(len(X[0])):
#                    X_sc += X[i][1][features] * scipy.stats.gamma.pdf(np.expand_dims(ref - X[i][1].onset_time + offset, -1), a = shape.eval(session=sess), scale = scale.eval(session=sess))
#                preds = intercept.eval(session=sess) + np.dot(X_sc, beta.eval(session=sess))
#                print('MSE (manual): %.4f' %((y[1]-preds)**2).mean())
#                print('MAE (manual): %.4f' %(y[1]-preds).abs().mean())
#                print()
            
            sess.run(incr_global_step)

            s = parse_summary(summary_params_batch)

            print('Parameter summary: \n')
            print_summary(s)
            print()

            print_summary(s, f='log/parameter_summary.txt') 

            summary_cv_losses_batch, loss_cv, preds_cv = sess.run([summary_losses, loss_func, out], feed_dict=fd_cv)
            mae_cv = mae(y[1], preds_cv)
            cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
            print('CV loss: %.4f'%loss_cv)
            print('CV MAE loss: %.4f'%mae_cv) 
            print()
            if compute_signif:
                if baseline_LM:
                    print('.'*50)
                    print('Bootstrap significance testing')
                    print('Model vs. LM baseline on CV data')
                    print()
                    print('MSE loss improvement: %.4f' %(lm_mse_cv-mse(y[1], preds_cv)))
                    p_cv = bootstrap(y[1], lm_preds_cv, preds_cv, err_type=loss)
                    print('p = %.4e' %p_cv)
                    print('.' * 50)
                    print()
                if baseline_LME:
                    print('.' * 50)
                    print('Bootstrap significance testing')
                    print('Model vs. LME baseline on CV data')
                    print()
                    print('MSE loss improvement: %.4f' %(lme_mse_cv-mse(y[1], preds_cv)))
                    p_cv = bootstrap(y[1], lme_preds_cv, preds_cv, err_type=loss)
                    print('p = %.4e' % p_cv)
                    print('.' * 50)
                    print()
            print()

            if not data_rep == 'linear':
                plot_convolutions(shape_global.eval(session=sess)[0],
                                  scale_global.eval(session=sess)[0],
                                  beta_global.eval(session=sess)[0],
                                  features)

            saver.save(sess, 'log/model.ckpt')

        summary_cv_losses_batch, loss_cv = sess.run([summary_losses, loss_func], feed_dict=fd_cv)
        cv_writer.add_summary(summary_cv_losses_batch, global_batch_step.eval(session=sess))
        print('CV loss: %.4f'%loss_cv)




