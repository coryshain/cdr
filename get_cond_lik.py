import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def get_mse(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('MSE'):
                mse = float(line.split()[1])
                return mse

def get_ll(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Log likelihood'):
                ll = float(line.split()[2])
                return ll
            elif line.startswith('Loglik'):
                ll = float(line.split()[1])
                return ll

df = []

for dir_path in [
    '../results/cognition/reading/natstor_log/LMEnoS',
    '../results/cognition/reading/natstor_log/LMEfullS',
    '../results/cognition/reading/natstor_log/GAMnoS',
    '../results/cognition/reading/natstor_log/GAMfullS',
    '../results/cdrnn_journal/natstor_log/GAMLSSnoS',
    '../results/cdrnn_journal/natstor_log/GAMLSSfullS',
    '../results/cognition/reading/natstor_log/CDR_G_bbvi',
    '../results/cdrnn_journal/natstor_log/CDR_main',
    '../results/cognition/reading/natstor_raw/LMEnoS',
    '../results/cognition/reading/natstor_raw/LMEfullS',
    '../results/cognition/reading/natstor_raw/GAMnoS',
    '../results/cognition/reading/natstor_raw/GAMfullS',
    '../results/cdrnn_journal/natstor_raw/GAMLSSnoS',
    '../results/cdrnn_journal/natstor_raw/GAMLSSfullS',
    '../results/cognition/reading/natstor_raw/CDR_G_bbvi',
    '../results/cdrnn_journal/natstor_raw/CDR_main',
    '../results/cognition/reading/dundee_raw_fp/LMEnoS',
    '../results/cognition/reading/dundee_raw_fp/LMEfullS',
    '../results/cognition/reading/dundee_raw_fp/GAMnoS',
    '../results/cognition/reading/dundee_raw_fp/GAMfullS',
    '../results/cdrnn_journal/dundee_raw_fp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_raw_fp/GAMLSSfullS',
    '../results/cognition/reading/dundee_raw_fp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_raw_fp/CDR_main',
    '../results/cognition/reading/dundee_raw_gp/LMEnoS',
    '../results/cognition/reading/dundee_raw_gp/LMEfullS',
    '../results/cognition/reading/dundee_raw_gp/GAMnoS',
    '../results/cognition/reading/dundee_raw_gp/GAMfullS',
    '../results/cdrnn_journal/dundee_raw_gp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_raw_gp/GAMLSSfullS',
    '../results/cognition/reading/dundee_raw_gp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_raw_gp/CDR_main',
    '../results/cognition/reading/dundee_raw_sp/LMEnoS',
    '../results/cognition/reading/dundee_raw_sp/GAMnoS',
    '../results/cognition/reading/dundee_raw_sp/GAMfullS',
    '../results/cdrnn_journal/dundee_raw_sp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_raw_sp/GAMLSSfullS',
    '../results/cognition/reading/dundee_raw_sp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_raw_sp/CDR_main',
    '../results/cognition/reading/dundee_log_fp/LMEnoS',
    '../results/cognition/reading/dundee_log_fp/LMEfullS',
    '../results/cognition/reading/dundee_log_fp/GAMnoS',
    '../results/cognition/reading/dundee_log_fp/GAMfullS',
    '../results/cdrnn_journal/dundee_log_fp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_log_fp/GAMLSSfullS',
    '../results/cognition/reading/dundee_log_fp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_log_fp/CDR_main',
    '../results/cognition/reading/dundee_log_gp/LMEnoS',
    '../results/cognition/reading/dundee_log_gp/LMEfullS',
    '../results/cognition/reading/dundee_log_gp/GAMnoS',
    '../results/cognition/reading/dundee_log_gp/GAMfullS',
    '../results/cdrnn_journal/dundee_log_gp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_log_gp/GAMLSSfullS',
    '../results/cognition/reading/dundee_log_gp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_log_gp/CDR_main',
    '../results/cognition/reading/dundee_log_sp/LMEnoS',
    '../results/cognition/reading/dundee_log_sp/GAMnoS',
    '../results/cognition/reading/dundee_log_sp/GAMfullS',
    '../results/cdrnn_journal/dundee_log_sp/GAMLSSnoS',
    '../results/cdrnn_journal/dundee_log_sp/GAMLSSfullS',
    '../results/cognition/reading/dundee_log_sp/CDR_G_bbvi',
    '../results/cdrnn_journal/dundee_log_sp/CDR_main',
    '../results/cognition/fMRI/fMRI_convolved/LME',
    '../results/cognition/fMRI/fMRI_interpolated/LME',
    '../results/cognition/fMRI/fMRI_averaged/LME',
    '../results/cognition/fMRI/fMRI_lanczos/LME',
    '../results/cdrnn_journal/fmriraw/GAMnoS',
    '../results/cdrnn_journal/fmriraw/GAMLSSnoS',
    '../results/cognition/fMRI/fMRI_cdr/CDR_DG5_bbvi',
    '../results/cdrnn_journal/fmri/CDR_main'
]:
    if 'LMEnoS' in dir_path or 'convolved' in dir_path:
        model = 'base'
    elif 'interpolated' in dir_path:
        model = 'LinInterp'
    elif 'averaged' in dir_path:
        model = 'Binned'
    elif 'lanczos' in dir_path:
        model = 'Interpolated'
    elif 'LMEnoS' in dir_path:
        model = 'LMEnoS'
    elif 'LME' in dir_path:
        model = 'LME'
    elif 'GAMLSSexGnoS' in dir_path:
        model = 'GAMLSSexGnoS'
    elif 'GAMLSSexG' in dir_path:
        model = 'GAMLSSexG'
    elif 'GAMLSSnoS' in dir_path:
        model = 'GAMLSSnoS'
    elif 'GAMLSS' in dir_path:
        model = 'GAMLSS'
    elif 'GAMnoS' in dir_path:
        model = 'GAMnoS'
    elif 'GAM' in dir_path:
        model = 'GAM'
    elif 'cdrnn' in dir_path:
        model = 'CDRNN'
    elif 'CDR' in dir_path:
        model = 'CDR'

    if 'natstor_log' in dir_path:
        dataset = 'natstor_log'
    elif 'natstor' in dir_path:
        dataset = 'natstor'
    elif 'dundee_log_fp' in dir_path:
        dataset = 'dundee_log_fp'
    elif 'dundee_log_gp' in dir_path:
        dataset = 'dundee_log_gp'
    elif 'dundee_log_sp' in dir_path:
        dataset = 'dundee_log_sp'
    elif 'dundee_raw_gp' in dir_path:
        dataset = 'dundee_raw_gp'
    elif 'dundee_raw_sp' in dir_path:
        dataset = 'dundee_raw_sp'
    elif 'dundee' in dir_path:
        dataset = 'dundee'
    elif 'fmri' in dir_path.lower():
        dataset = 'fMRI'
    else:
        raise ValueError('Unknown dataset for path: %s' % dir_path)

    for partition in ['train', 'dev', 'test']:
        eval_paths = [x for x in os.listdir(dir_path) if x.startswith('eval') and x.endswith('_%s.txt' % partition)]
        assert len(eval_paths) == 1, 'Got wrong number of eval paths: %s, %s, %s, %s.' % (dataset, model, partition, eval_paths)
        eval_path = dir_path + '/' + eval_paths[0]
        if model in ('CDR', 'CDRNN'):
            ll = get_ll(eval_path)
        elif model.startswith('GAMLSSexG'):
            obs = np.sqrt(pd.read_csv(dir_path + '/obs_%s.txt' % partition, header=None)[0].values)
            mu = np.sqrt(pd.read_csv(dir_path + '/preds_%s.txt' % partition, header=None)[0].values)
            sigma = np.sqrt(pd.read_csv(dir_path + '/preds_sigma_%s.txt' % partition, header=None)[0].values)
            nu = np.sqrt(pd.read_csv(dir_path + '/preds_nu_%s.txt' % partition, header=None)[0].values)
        else:
            err = np.sqrt(pd.read_csv(dir_path + '/losses_mse_%s.txt' % partition, header=None)[0].values)
            if model in ('GAMLSSnoS', 'GAMLSS'):
                sigma = pd.read_csv(dir_path + '/preds_sigma_%s.txt' % partition, header=None)[0].values 
            else:
                sigma = np.sqrt(get_mse(eval_path))
            ll = norm.logpdf(err, loc=0, scale=sigma)
            ll = pd.DataFrame({'LL': ll})
            ll.to_csv(dir_path + '/conditional_ll_%s.txt' % partition, header=False, index=False)
            ll = ll.LL.sum()
        print('%s | %s | %s: %s' % (model, dataset, partition, ll))
        df.append((model, dataset, partition, ll))

df = pd.DataFrame(df, columns=['Model', 'Dataset', 'Partition', 'LL'])
df.to_csv('conditional_ll.csv', index=False)

